"""Selftest sintético da avaliação só em `val` e da decisão GO/NO-GO do piloto GMDCSA24.

Reaproveita o mundo sintético do selftest do treino (adapter, pose, Le2i-CS
com um vídeo de `val` e um de `test`, e um controle treinado com a receita
congelada), prepara e treina T e S na seed 42 pelo `train` do CLI e grava o
`event_metrics.json` canônico do controle com a orquestração canônica. A
leitura de features do tratamento recusa qualquer vídeo de `test`, e a
predição é espionada, então qualquer acesso a `test` reprova. As regras de
decisão são conferidas à parte, nos limites exatos, com cinco seeds
sintéticas. Requer FFmpeg; não toca no dataset real nem em `runs/`.
"""

import contextlib
import json
import math
import random
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from gatefall.config import EVAL_STRIDE
from gatefall.data.gmdcsa24.selftests.adapter import _check
from gatefall.data.gmdcsa24.selftests.pose_extract import _quiet
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.eval.analysis.multiseed_summary import validate_run_event_artifacts
from gatefall.eval.baseline_a.cli import _load_model, _predict_with_identity
from gatefall.eval.gmdcsa24_weak_label import validation
from gatefall.eval.gmdcsa24_weak_label.cli import (
    decision_lock,
    publish_once,
    record_decision,
    run_decide,
    run_val_evaluation,
)
from gatefall.eval.gmdcsa24_weak_label.decision import (
    G1_MIN_MEAN_DELTA,
    G1_MIN_POSITIVE_SEEDS,
    G4_MIN_MEAN_DELTA,
    GO,
    GO_FRAGILE,
    INVALID,
    NO_GO,
    S_FRAGILE_MAX_MEAN_DELTA,
    VALIDITY_CONDITIONS,
    PairedSeed,
    SeedMetrics,
    decide,
)
from gatefall.eval.gmdcsa24_weak_label.validation import (
    ALARM_PROTOCOL_FILE,
    CANONICAL_EVENT_METRICS_FILE,
    RUN_CONDITION,
    VAL_EVENT_METRICS_FILE,
    VAL_SPLITS,
    CheckedRun,
    check_run,
    val_event_evaluation,
    val_report,
    val_source,
)
from gatefall.eval.shared.event_artifacts import (
    EVENT_TRANSACTION_FILE,
    EventEvaluationLock,
    validate_event_metrics,
)
from gatefall.eval.shared import orchestration
from gatefall.eval.shared.orchestration import (
    EventEvaluation,
    Predictions,
    SplitEvaluator,
    run_event_evaluation,
)
from gatefall.features.standardization import load_stats
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.config import load_config
from gatefall.train.gmdcsa24_weak_label.assembly import arm_a_features
from gatefall.train.gmdcsa24_weak_label.cli import run_prepare
from gatefall.train.gmdcsa24_weak_label.gates import (
    DECISION_FILE,
    DEFAULT_RUN_ROOT,
    PREPARED_STATS_FILE,
    TREATMENT_RECORD_FILE,
    TreatmentInputs,
    canonical_run_dir,
)
from gatefall.train.gmdcsa24_weak_label.selftests.treatment import (
    _TEST_VIDEOS,
    _SEED,
    _build,
    _canonical_train,
    _from_frozen_checkout,
    _prepare,
    _Spy,
    _tree,
    _World,
)

_VAL_VIDEOS = {"home_02/video_3"}
_SEEDS = (42, 43, 44, 45, 46)


class _Config:
    run_name = "canonical"


class _NoTest:
    """Leitura de features que reprova qualquer vídeo de `test` do Le2i-CS."""

    def __init__(self, spy: _Spy) -> None:
        self.spy = spy

    def __call__(self, video_id: str, pose_root: Path) -> np.ndarray:
        if video_id in _TEST_VIDEOS:
            raise AssertionError(f"feature de test lida: {video_id}")
        return self.spy(video_id, pose_root)


def _inputs(world: _World, treatment: str, **changes: Any) -> TreatmentInputs:
    return world.inputs(treatment, load_features=_NoTest(world.spy), **changes)


def _run_dir(world: _World, treatment: str) -> Path:
    return world.tmp / canonical_run_dir(treatment, _SEED)


def _evaluate(world: _World, treatment: str, force: bool = False, **changes: Any) -> bool:
    with contextlib.chdir(world.tmp), _quiet():
        return run_val_evaluation(_inputs(world, treatment, **changes), _SEED, world.control, force)


def _decide(world: _World, **changes: Any) -> bool:
    with contextlib.chdir(world.tmp), _quiet():
        return run_decide({treatment: _inputs(world, treatment, **changes) for treatment in "TS"})


def _checked(world: _World, treatment: str) -> CheckedRun | None:
    with contextlib.chdir(world.tmp), _quiet():
        return check_run(_inputs(world, treatment), _SEED, world.control).run


def _val_report(world: _World, treatment: str) -> list[str]:
    run = _checked(world, treatment)
    if run is None:
        return ["run reprovado nos portões"]
    with contextlib.chdir(world.tmp):
        return val_report(run)[1]


@contextlib.contextmanager
def _tampered(path: Path, mutate: Callable[[Path], object]) -> Iterator[None]:
    """Altera `path` (arquivo ou diretório) e restaura o original na saída."""
    backup = path.with_name(f"{path.name}.selftest-backup")
    if path.is_dir():
        shutil.copytree(path, backup)
    else:
        shutil.copy2(path, backup)
    try:
        mutate(path)
        yield
    finally:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        backup.rename(path)


def _edit_json(update: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def mutate(path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        update(data)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return mutate


def _edit_yaml(field: str, value: object) -> Callable[[Path], None]:
    def mutate(path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data[field] = value
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    return mutate


def _control_events(world: _World) -> None:
    """`event_metrics.json` canônico do controle (`val` e `test`), pela orquestração
    canônica, com a padronização e a pose do controle."""
    run_dir = world.control.run_dir
    config = load_config(run_dir / "config.yaml")
    frames = world.le2i.frames

    def prepare() -> tuple[pd.DataFrame, SplitEvaluator]:
        model = _load_model(config, run_dir / "checkpoint.pt", "cpu")
        stats = load_stats(world.le2i.stats_path)

        def evaluate_split(split: str) -> tuple[int, Predictions]:
            source = PoseWindowDataset(
                frames,
                split,
                EVAL_STRIDE,
                lambda video_id: arm_a_features(video_id, world.le2i.pose_root),
                drop_ignored=False,
            )
            return len(source), _predict_with_identity(
                model, source, stats, "cpu", batch_size=config.batch_size
            )

        return frames, evaluate_split

    with _quiet(), EventEvaluationLock(run_dir) as lock:
        run_event_evaluation(True, "A", run_dir, lock, lambda: EventEvaluation(config, prepare))


def _setup(world: _World) -> bool:
    prepared = all(_prepare(world, treatment)[0] for treatment in "TS")
    trained = all(
        _canonical_train(world, world.tmp, inputs=world.inputs(treatment)) for treatment in "TS"
    )
    _control_events(world)
    return _check(
        "setup: T e S preparados e treinados na seed 42 e controle com event_metrics.json canônico",
        prepared and trained and (world.control.run_dir / CANONICAL_EVENT_METRICS_FILE).is_file(),
    )


def check_canonical_contract(world: _World) -> list[bool]:
    """O caminho canônico continua val e test em event_metrics.json, sem campo novo."""
    evaluation = EventEvaluation(_Config(), lambda: (pd.DataFrame(), lambda _: (0, ([], [], [], []))))
    report = json.loads(
        (world.control.run_dir / CANONICAL_EVENT_METRICS_FILE).read_text(encoding="utf-8")
    )
    config = load_config(world.control.run_dir / "config.yaml")
    try:
        validate_run_event_artifacts(world.control.run_dir, config)
        control_valid = True
    except RuntimeError:
        control_valid = False
    return [
        _check(
            "EventEvaluation canônica mantém val e test, event_metrics.json e nenhuma proveniência",
            evaluation.splits == ("val", "test")
            and evaluation.metrics_file == CANONICAL_EVENT_METRICS_FILE
            and not evaluation.provenance
            and evaluation.check_splits is None,
        ),
        _check(
            "relatório canônico do controle mantém só os campos de antes e passa no validador",
            control_valid
            and set(report)
            == {
                "run_name",
                "checkpoint_path",
                "alarm_protocol_path",
                "splits",
                "checkpoint_sha256",
                "training_metrics_sha256",
                "alarm_protocol_sha256",
            }
            and set(report["splits"]) == {"val", "test"},
        ),
    ]


def check_evaluation(world: _World) -> list[bool]:
    world.spy.loaded.clear()
    predicted: list[str] = []

    def spy_predict(*args: Any, **kwargs: Any) -> Predictions:
        predictions = _predict_with_identity(*args, **kwargs)
        predicted.extend(predictions[0])
        return predictions

    try:
        with patch.object(validation, "_predict_with_identity", spy_predict):
            evaluated = {treatment: _evaluate(world, treatment) for treatment in "TS"}
        test_untouched = True
    except AssertionError:
        evaluated, test_untouched = {"T": False, "S": False}, False
    if not all(evaluated.values()):
        return [_check("evaluate de T e S grava val_event_metrics.json sem tocar em test", False)]
    run_dir = _run_dir(world, "T")
    data = json.loads((run_dir / VAL_EVENT_METRICS_FILE).read_text(encoding="utf-8"))
    control_report = json.loads(
        (world.control.run_dir / CANONICAL_EVENT_METRICS_FILE).read_text(encoding="utf-8")
    )
    run = cast(CheckedRun, _checked(world, "T"))
    with contextlib.chdir(world.tmp):
        frames, evaluate_split = val_event_evaluation(_inputs(world, "T"), run).prepare()
        source = val_source(_inputs(world, "T"), frames)
    try:
        evaluate_split("test")
        test_split_refused = False
    except RuntimeError:
        test_split_refused = True
    try:
        validate_run_event_artifacts(run_dir, run.config)
        canonical_consumes = True
    except RuntimeError:
        canonical_consumes = False
    try:
        with contextlib.chdir(world.tmp):
            validate_event_metrics(
                data,
                run.config,
                run.run_dir / "checkpoint.pt",
                run.run_dir / ALARM_PROTOCOL_FILE,
                training_metrics_path=run.run_dir / "metrics.json",
                require_hashes=True,
            )
        canonical_accepts = True
    except ValueError:
        canonical_accepts = False
    before = (run_dir / VAL_EVENT_METRICS_FILE).read_bytes()
    rerun = _evaluate(world, "T")
    record_sha = sha256_file(run_dir / TREATMENT_RECORD_FILE)
    return [
        _check("evaluate de T e S grava val_event_metrics.json sem tocar em test", test_untouched),
        _check(
            "nenhuma feature nem predição de test: só o vídeo de val é lido e predito",
            not (world.spy.videos() & _TEST_VIDEOS)
            and _VAL_VIDEOS <= world.spy.videos()
            and set(predicted) == _VAL_VIDEOS,
        ),
        _check(
            "a grade entregue à orquestração só tem val, a fonte só janelas de val e "
            "o avaliador recusa o split test",
            set(frames["split"].astype(str)) == set(VAL_SPLITS)
            and set(source.windows["video_id"].astype(str)) == _VAL_VIDEOS
            and test_split_refused,
        ),
        _check(
            "o relatório tem só val, registra evaluated_splits e liga run, tratamento, seed, "
            "manifesto preparado e controle",
            set(data["splits"]) == {"val"}
            and data["evaluated_splits"] == ["val"]
            and data["treatment"]
            == {
                "treatment": "T",
                "seed": _SEED,
                "treatment_record_sha256": record_sha,
                "prepared_manifest_sha256": json.loads(
                    (run_dir / TREATMENT_RECORD_FILE).read_text(encoding="utf-8")
                )["prepared_manifest_sha256"],
                "control_checkpoint_sha256": world.gate.control_checkpoint_sha256[_SEED],
            }
            and data["checkpoint_sha256"] == sha256_file(run_dir / "checkpoint.pt"),
        ),
        _check(
            "nenhum event_metrics.json no run, e os validadores canônicos recusam o relatório só de val",
            not (run_dir / CANONICAL_EVENT_METRICS_FILE).exists()
            and not canonical_consumes
            and not canonical_accepts,
        ),
        _check(
            "a estrutura de val (janelas, horas, 1 queda) é a do controle",
            all(
                data["splits"]["val"][field] == control_report["splits"]["val"][field]
                for field in validation.EVENT_STRUCTURE_FIELDS
            )
            and data["splits"]["val"]["n_fall_events"] == 1,
        ),
        _check(
            "relatório válido não é refeito",
            rerun and (run_dir / VAL_EVENT_METRICS_FILE).read_bytes() == before,
        ),
    ]


def check_report_tampering(world: _World) -> list[bool]:
    report = _run_dir(world, "T") / VAL_EVENT_METRICS_FILE
    protocol = _run_dir(world, "T") / ALARM_PROTOCOL_FILE
    record = _run_dir(world, "T") / TREATMENT_RECORD_FILE
    control_report = world.control.run_dir / CANONICAL_EVENT_METRICS_FILE

    def refused_report(update: Callable[[dict[str, Any]], None]) -> bool:
        with _tampered(report, _edit_json(update)):
            return bool(_val_report(world, "T"))

    def add_test(data: dict[str, Any]) -> None:
        data["splits"]["test"] = data["splits"]["val"]

    def rename_split(data: dict[str, Any]) -> None:
        data["splits"] = {"test": data["splits"]["val"]}

    def extra_split(data: dict[str, Any]) -> None:
        data["evaluated_splits"] = ["val", "test"]

    def other_seed(data: dict[str, Any]) -> None:
        data["treatment"]["seed"] = 43

    def other_treatment(data: dict[str, Any]) -> None:
        data["treatment"]["treatment"] = "S"

    def other_manifest(data: dict[str, Any]) -> None:
        data["treatment"]["prepared_manifest_sha256"] = "0" * 64

    def other_control(data: dict[str, Any]) -> None:
        data["treatment"]["control_checkpoint_sha256"] = "0" * 64

    def other_checkpoint(data: dict[str, Any]) -> None:
        data["checkpoint_sha256"] = "0" * 64

    def missed_count(data: dict[str, Any]) -> None:
        val = data["splits"]["val"]
        val["n_missed_events"] += 1

    def shifted_falls(data: dict[str, Any]) -> None:
        val = data["splits"]["val"]
        val["n_fall_events"] += 1
        val["n_missed_events"] += 1

    def record_seed(data: dict[str, Any]) -> None:
        data["seed"] = 43

    with _tampered(protocol, _edit_yaml("trigger_consecutive", 4)):
        protocol_refused = bool(_val_report(world, "T")) and not _evaluate(world, "T")
    with _tampered(record, _edit_json(record_seed)):
        record_refused = not _evaluate(world, "T") and not _evaluate(world, "T", force=True)
    with _tampered(control_report, _edit_json(shifted_falls)):
        control_shift_refused = bool(_val_report(world, "T")) and not _evaluate(
            world, "T", force=True
        )
    planted = _run_dir(world, "T") / CANONICAL_EVENT_METRICS_FILE
    shutil.copy2(control_report, planted)
    with contextlib.chdir(world.tmp), _quiet():
        planted_results = check_run(_inputs(world, "T"), _SEED, world.control).results
    planted_refused = bool(planted_results[RUN_CONDITION]) and not _evaluate(world, "T")
    planted.unlink()
    return [
        _check(
            "relatório com test, só com test ou com evaluated_splits a mais é recusado",
            refused_report(add_test) and refused_report(rename_split) and refused_report(extra_split),
        ),
        _check(
            "relatório com seed, tratamento, manifesto, controle ou checkpoint trocados é recusado",
            all(
                refused_report(update)
                for update in (
                    other_seed,
                    other_treatment,
                    other_manifest,
                    other_control,
                    other_checkpoint,
                )
            ),
        ),
        _check(
            "contagem de eventos inconsistente no relatório é recusada",
            refused_report(missed_count),
        ),
        _check(
            "alarm_protocol.yaml do run diferente do protocolo congelado reprova relatório e evaluate",
            protocol_refused,
        ),
        _check(
            "treatment.json adulterado bloqueia evaluate, com e sem --force",
            record_refused,
        ),
        _check(
            "val do controle com outra contagem de quedas reprova a estrutura do tratamento",
            control_shift_refused,
        ),
        _check(
            "event_metrics.json (val e test) plantado no run do tratamento é recusado",
            planted_refused,
        ),
        _check("relatório íntegro depois das restaurações", not _val_report(world, "T")),
    ]


def _pair(run_dir: Path) -> dict[str, str | None]:
    return {
        name: sha256_file(run_dir / name) if (run_dir / name).is_file() else None
        for name in (ALARM_PROTOCOL_FILE, VAL_EVENT_METRICS_FILE)
    }


def _other_seed(data: dict[str, Any]) -> None:
    data["treatment"]["seed"] = 43


def check_evaluation_lifecycle(world: _World) -> list[bool]:
    """Antes da decisão: par completo imutável, par parcial ou inconsistente só
    recuperado com --force."""
    run_dir = _run_dir(world, "T")
    report = run_dir / VAL_EVENT_METRICS_FILE
    before = _pair(run_dir)
    forced = _evaluate(world, "T", force=True)
    immutable = not forced and _pair(run_dir) == before
    skipped = _evaluate(world, "T") and _pair(run_dir) == before

    report.unlink()
    partial_refused = not _evaluate(world, "T") and not report.exists()
    partial_recovered = _evaluate(world, "T", force=True) and not _val_report(world, "T")

    _edit_json(_other_seed)(report)
    tampered = report.read_bytes()
    inconsistent_refused = not _evaluate(world, "T") and report.read_bytes() == tampered
    inconsistent_recovered = _evaluate(world, "T", force=True) and not _val_report(world, "T")

    journal = run_dir / EVENT_TRANSACTION_FILE
    journal.write_text("{}", encoding="utf-8")
    pending_refused = (
        bool(_val_report(world, "T"))
        and not _evaluate(world, "T")
        and not _evaluate(world, "T", force=True)
        and journal.read_text(encoding="utf-8") == "{}"
    )
    journal.unlink()
    return [
        _check(
            "avaliação completa e íntegra é imutável: --force recusa sem alterar o par, "
            "e sem --force ela é só conferida",
            immutable and skipped,
        ),
        _check(
            "par parcial é recusado sem --force e recuperado com --force antes da decisão",
            partial_refused and partial_recovered,
        ),
        _check(
            "par inconsistente é recusado sem --force e recuperado com --force antes da decisão",
            inconsistent_refused and inconsistent_recovered,
        ),
        _check(
            "journal malformado invalida o par e bloqueia evaluate, com e sem --force, sem mudar",
            pending_refused,
        ),
    ]


class _Crash(BaseException):
    """Queda do processo no meio da promoção: escapa do `except Exception` que
    recupera a publicação, deixando o journal como um processo morto deixaria."""


_INTERRUPTED = "interrupted_publication"


def _interrupt(world: _World, step: str) -> bool:
    """Publicação real de `evaluate --force` interrompida depois de `step`."""
    original = orchestration._promote_event_outputs

    def crash(name: str) -> None:
        if name == step:
            raise _Crash(name)

    def promote(*args: Any, **kwargs: Any) -> None:
        original(*args, **{**kwargs, "after_step": crash})

    with patch.object(orchestration, "_promote_event_outputs", promote):
        try:
            _evaluate(world, "T", force=True)
        except _Crash:
            return True
    return False


def _journal_valid(run_dir: Path) -> bool:
    """Journal versão 1 de uma promoção parada entre o protocolo e as métricas."""
    journal = json.loads((run_dir / EVENT_TRANSACTION_FILE).read_text(encoding="utf-8"))
    return (
        journal.get("version") == 1
        and journal["metrics_path"] == VAL_EVENT_METRICS_FILE
        and journal["staged_metrics"].startswith(".val_event_metrics.pending-")
        and (run_dir / journal["staged_metrics"]).is_file()
        and (run_dir / journal["metrics_backup"]).is_file()
        and sha256_file(run_dir / ALARM_PROTOCOL_FILE) == journal["new_protocol_sha256"]
        and not (run_dir / VAL_EVENT_METRICS_FILE).exists()
    )


def _leftovers(run_dir: Path) -> list[Path]:
    return [
        path
        for path in run_dir.iterdir()
        if path.name == EVENT_TRANSACTION_FILE or ".backup-" in path.name or ".pending-" in path.name
    ]


def check_interrupted_publication(world: _World) -> list[bool]:
    """Antes da decisão, uma publicação interrompida com journal válido só é
    recuperada com --force; sem --force, nada muda."""
    run_dir = _run_dir(world, "T")
    _edit_json(_other_seed)(run_dir / VAL_EVENT_METRICS_FILE)
    interrupted = _interrupt(world, "publish_protocol")
    if not interrupted or not _journal_valid(run_dir):
        return [_check("publicação real interrompida deixa um journal válido", False)]
    shutil.copytree(run_dir, world.tmp / _INTERRUPTED)
    snapshot = _tree(run_dir)
    decision_path = world.tmp / DEFAULT_RUN_ROOT / DECISION_FILE
    untouched = not _evaluate(world, "T") and _tree(run_dir) == snapshot
    undecided = not _decide(world) and not decision_path.exists() and _tree(run_dir) == snapshot
    recovered = (
        _evaluate(world, "T", force=True) and not _val_report(world, "T") and not _leftovers(run_dir)
    )
    return [
        _check("publicação real interrompida deixa um journal válido", True),
        _check(
            "evaluate sem --force não finaliza, restaura nem limpa a publicação interrompida: "
            "artefatos e journal intactos",
            untouched,
        ),
        _check("decide recusa com a publicação pendente, sem registrar nem alterar nada", undecided),
        _check(
            "evaluate --force recupera antes da decisão: par íntegro sem journal, backup nem staging",
            recovered,
        ),
    ]


def check_decide(world: _World) -> list[bool]:
    decision_path = world.tmp / DEFAULT_RUN_ROOT / DECISION_FILE
    decided = _decide(world)
    if not decided or not decision_path.is_file():
        return [_check("decide registra a decisão sobre T e S completos", False)]
    recorded = decision_path.read_bytes()
    record = json.loads(recorded)
    rerun = _decide(world)
    unchanged = decision_path.read_bytes() == recorded
    with _tampered(decision_path, _edit_json(lambda data: data.update({"decision": INVALID}))):
        changed_refused = not _decide(world)
    decision_path.unlink()

    def leaves_no_decision(outcome: bool) -> bool:
        return not outcome and not decision_path.exists()

    missing_seed = leaves_no_decision(
        _decide(
            world,
            gate=replace(
                world.gate,
                control_checkpoint_sha256={**world.gate.control_checkpoint_sha256, 43: "0" * 64},
            ),
        )
    )
    extra = world.tmp / DEFAULT_RUN_ROOT / "t" / "seed_042"
    extra.mkdir()
    extra_seed = leaves_no_decision(_decide(world))
    extra.rmdir()
    s_report = _run_dir(world, "S") / VAL_EVENT_METRICS_FILE
    with _tampered(s_report, lambda path: path.unlink()):
        missing_report = leaves_no_decision(_decide(world))
    with _tampered(s_report, _edit_json(lambda data: data["splits"].update(test={}))):
        test_report = leaves_no_decision(_decide(world))

    copy = world.tmp / "control_copy" / "seed_42"
    shutil.copytree(world.control.run_dir, copy)

    def other_control(data: dict[str, Any]) -> None:
        data["control"]["run_dir"] = copy.as_posix()

    s_record = _run_dir(world, "S") / TREATMENT_RECORD_FILE
    with _tampered(s_record, _edit_json(other_control)):
        control_mismatch = leaves_no_decision(_decide(world))
    with _tampered(s_record, _edit_json(lambda data: data.update({"seed": 43}))):
        record_tampered = leaves_no_decision(_decide(world))
    dirty = leaves_no_decision(_decide(world, checkout=("e" * 40, ["?? arquivo"])))

    invalid_gate = replace(world.gate, control_config_fingerprint="0" * 64)
    invalid = _decide(world, gate=invalid_gate)
    invalid_record = json.loads(decision_path.read_text(encoding="utf-8")) if invalid else {}
    decision_path.unlink(missing_ok=True)
    with _tampered(
        world.control.run_dir / ALARM_PROTOCOL_FILE, _edit_yaml("refractory_period_s", 4.0)
    ):
        alarm_invalid = _decide(world)
        alarm_record = (
            json.loads(decision_path.read_text(encoding="utf-8")) if alarm_invalid else {}
        )
    decision_path.unlink(missing_ok=True)
    restored = _decide(world) and json.loads(decision_path.read_text(encoding="utf-8"))[
        "decision"
    ] == record["decision"]
    return [
        _check(
            "decide registra a decisão só de val com seeds, condições e hashes das entradas",
            record["decision"] in (GO, GO_FRAGILE, NO_GO)
            and record["evaluated_splits"] == ["val"]
            and record["seeds"] == [_SEED]
            and set(record["conditions"]) == {"T", "S"}
            and set(record["inputs"]["T"][str(_SEED)])
            >= {"val_event_metrics_sha256", "training_metrics_sha256", "treatment_record_sha256"},
        ),
        _check("decisão igual recalculada é pulada, sem regravar", rerun and unchanged),
        _check("decisão registrada diferente da recalculada nunca é substituída", changed_refused),
        _check(
            "seed do controle sem run, diretório de seed a mais, relatório de val ausente ou "
            "com test recusam sem registrar decisão",
            missing_seed and extra_seed and missing_report and test_report,
        ),
        _check(
            "controle diferente entre T e S ou treatment.json adulterado recusam sem decisão",
            control_mismatch and record_tampered,
        ),
        _check("checkout sujo recusa sem decisão", dirty),
        _check(
            "V4 reprovada (configuração do controle) registra INVÁLIDO sem métricas nem condições",
            invalid
            and invalid_record.get("decision") == INVALID
            and set(invalid_record["validity_failures"]) == {"T", "S"}
            and all("V4" in failures for failures in invalid_record["validity_failures"].values())
            and invalid_record["inputs"] is None
            and "conditions" not in invalid_record,
        ),
        _check(
            "alarme do controle diferente do protocolo congelado (V2) registra INVÁLIDO",
            alarm_invalid
            and alarm_record.get("decision") == INVALID
            and all("V2" in failures for failures in alarm_record["validity_failures"].values()),
        ),
        _check("restaurado, decide volta a registrar a decisão de antes", restored),
    ]


def check_post_decision(world: _World) -> list[bool]:
    """Com a decisão registrada, nada que a sustenta é gravado ou recuperado,
    e as leituras de resultados inalterados continuam aceitas."""
    decision_path = world.tmp / DEFAULT_RUN_ROOT / DECISION_FILE
    if not decision_path.is_file():
        return [_check("decisão registrada antes das checagens pós-decisão", False)]
    decision = decision_path.read_bytes()
    run_dir = _run_dir(world, "T")
    report = run_dir / VAL_EVENT_METRICS_FILE
    before = _pair(run_dir)
    read = _evaluate(world, "T") and _decide(world)
    forced = _evaluate(world, "T", force=True)
    with _tampered(report, lambda path: path.unlink()):
        partial = (
            not _evaluate(world, "T")
            and not _evaluate(world, "T", force=True)
            and not report.exists()
        )
    with _tampered(report, _edit_json(_other_seed)):
        tampered = report.read_bytes()
        inconsistent = not _evaluate(world, "T", force=True) and report.read_bytes() == tampered
    checkpoint = sha256_file(run_dir / "checkpoint.pt")
    with _tampered(run_dir / TREATMENT_RECORD_FILE, lambda path: path.unlink()):
        train_refused = (
            not _canonical_train(world, world.tmp, True, "falha técnica")
            and not (run_dir / TREATMENT_RECORD_FILE).exists()
            and sha256_file(run_dir / "checkpoint.pt") == checkpoint
        )
    stats = _inputs(world, "T").prepared_dir / PREPARED_STATS_FILE

    def shift_stats(path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["mean"][0] += 1.0
        path.write_text(json.dumps(data), encoding="utf-8")

    with _tampered(stats, shift_stats):
        shifted = stats.read_bytes()
        with contextlib.chdir(world.tmp), _quiet():
            prepare_refused = not run_prepare(_inputs(world, "T"), force=True)
        prepare_refused = prepare_refused and stats.read_bytes() == shifted

    def interrupted_state(path: Path) -> None:
        shutil.rmtree(path)
        shutil.copytree(world.tmp / _INTERRUPTED, path)

    with _tampered(run_dir, interrupted_state):
        snapshot = _tree(run_dir)
        no_recovery = (
            not _evaluate(world, "T")
            and not _evaluate(world, "T", force=True)
            and _tree(run_dir) == snapshot
            and _journal_valid(run_dir)
        )

    record = json.loads(decision)
    record["inputs"]["T"][str(_SEED)]["val_event_metrics_sha256"] = "0" * 64
    other = world.tmp / "other_decision" / DECISION_FILE
    with contextlib.chdir(world.tmp), _quiet():
        stale_refused = not record_decision(other, record) and not other.exists()
    return [
        _check(
            "depois da decisão, evaluate e decide sobre resultados inalterados só conferem",
            read and _pair(run_dir) == before and decision_path.read_bytes() == decision,
        ),
        _check(
            "depois da decisão, --force sobre o par completo é recusado",
            not forced and _pair(run_dir) == before,
        ),
        _check(
            "depois da decisão, par parcial ou inconsistente não é recuperado, nem com --force",
            partial and inconsistent,
        ),
        _check(
            "depois da decisão, publicação interrompida com journal válido não é recuperada, "
            "nem com --force: artefatos e journal intactos",
            no_recovery,
        ),
        _check(
            "depois da decisão, train --force e prepare --force recusam sem gravar",
            train_refused and prepare_refused,
        ),
        _check(
            "decisão cujos arquivos lidos mudaram antes da publicação é recusada sem registro",
            stale_refused,
        ),
        _check(
            "a decisão registrada e o par seguem intactos",
            decision_path.read_bytes() == decision and _pair(run_dir) == before,
        ),
    ]


def _decision_record(decision: str, index: int) -> dict[str, object]:
    return {
        "decision": decision,
        "inputs": None,
        "code": {"commit": f"{index:040d}", "dirty": False},
        "recorded_at": f"2026-10-20T00:00:{index:02d}+00:00",
    }


def check_concurrent_publication() -> list[bool]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "runs"
        root.mkdir()
        path = root / DECISION_FILE
        records = [_decision_record(GO if index % 2 else NO_GO, index) for index in range(8)]
        results: list[bool | None] = [None] * len(records)
        barrier = threading.Barrier(len(records))

        def worker(index: int) -> None:
            barrier.wait()
            results[index] = record_decision(path, records[index])

        with _quiet():
            threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        published = (
            json.loads(path.read_text(encoding="utf-8")).get("decision") if path.is_file() else None
        )
        expected = [record["decision"] == published for record in records]
        leftovers = list(root.glob(f".{DECISION_FILE}.*.tmp"))

        blocked_path = root / "blocked" / DECISION_FILE
        blocked_path.parent.mkdir()
        blocked_results: list[bool] = []
        with _quiet():
            with decision_lock(blocked_path.parent, exclusive=True):
                thread = threading.Thread(
                    target=lambda: blocked_results.append(
                        record_decision(blocked_path, _decision_record(GO, 0))
                    )
                )
                thread.start()
                thread.join(timeout=0.5)
                waited = thread.is_alive() and not blocked_path.exists()
            thread.join()
        existing = root / "existing" / DECISION_FILE
        existing.parent.mkdir()
        existing.write_text("{}", encoding="utf-8")
        never_replaced = not publish_once(existing, _decision_record(GO, 0)) and (
            existing.read_text(encoding="utf-8") == "{}"
        )
        return [
            _check(
                "oito decide concorrentes com resultados diferentes: um publica, os iguais "
                "confirmam, os diferentes são recusados, sem temporário restante",
                results == expected and not leftovers,
            ),
            _check(
                "record_decision espera o lock exclusivo antes de conferir e publicar",
                waited and blocked_results == [True] and blocked_path.is_file(),
            ),
            _check("a publicação nunca substitui uma decisão existente", never_replaced),
        ]


def _metrics(macro: float = 0.0, fall: float = 0.0, missed: int = 2, false: int = 3) -> SeedMetrics:
    return SeedMetrics(macro, fall, missed, false)


def _arm(
    macro: list[float] | float = G1_MIN_MEAN_DELTA,
    fall: list[float] | float = 0.0,
    missed: list[int] | int = 2,
    false: list[int] | int = 3,
    seeds: tuple[int, ...] = _SEEDS,
) -> list[PairedSeed]:
    def per_seed(value: Any) -> list[Any]:
        return value if isinstance(value, list) else [value] * len(seeds)

    return [
        PairedSeed(seed, _metrics(m, f, n, a), _metrics())
        for seed, m, f, n, a in zip(seeds, per_seed(macro), per_seed(fall), per_seed(missed), per_seed(false))
    ]


def _valid(**problems: list[str]) -> dict[str, dict[str, list[str]]]:
    validity = {treatment: {condition: [] for condition in VALIDITY_CONDITIONS} for treatment in "TS"}
    for key, value in problems.items():
        treatment, condition = key.split("_")
        validity[treatment][condition] = value
    return validity


def _decision(t: list[PairedSeed], s: list[PairedSeed] | None = None, **validity: list[str]) -> object:
    return decide(_valid(**validity), {"T": t, "S": s if s is not None else _arm(0.01)}, _SEEDS)[
        "decision"
    ]


def _raises(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def check_decision_rules() -> list[bool]:
    below = math.nextafter(G1_MIN_MEAN_DELTA, 0.0)
    boundary = decide(_valid(), {"T": _arm(), "S": _arm(0.01)}, _SEEDS)
    conditions = cast(dict[str, dict[str, dict[str, Any]]], boundary["conditions"])
    shuffled = _arm()
    random.Random(0).shuffle(shuffled)
    reordered = decide(_valid(), {"T": shuffled, "S": _arm(0.01)}, _SEEDS)
    duplicated = [*_arm()[:4], _arm()[0]]
    other_control = _arm(0.01)
    other_control[2] = replace(other_control[2], control=_metrics(missed=1))
    return [
        _check(
            "limites congelados: G1 0,015 e 4 de 5 seeds, G4 0, S frágil com média <= 0",
            (G1_MIN_MEAN_DELTA, G1_MIN_POSITIVE_SEEDS, G4_MIN_MEAN_DELTA, S_FRAGILE_MAX_MEAN_DELTA)
            == (0.015, 4, 0.0, 0.0),
        ),
        _check(
            "no limite exato (Δ médio 0,015, perdas e alarmes iguais, Δ F1 de fall 0) T dá GO",
            boundary["decision"] == GO
            and all(result["passed"] for result in conditions["T"].values()),
        ),
        _check(
            "G1: Δ médio logo abaixo de 0,015 dá NO-GO",
            _decision(_arm(below)) == NO_GO,
        ),
        _check(
            "G1: 4 de 5 seeds com Δ > 0 bastam; 3, com Δ = 0 nas outras, dão NO-GO",
            _decision(_arm([0.01875] * 4 + [0.0])) == GO
            and _decision(_arm([0.025] * 3 + [0.0, 0.0])) == NO_GO,
        ),
        _check(
            "G2 e G3: uma perda ou um alarme falso a mais na soma de T dão NO-GO",
            _decision(_arm(missed=[3, 2, 2, 2, 2])) == NO_GO
            and _decision(_arm(false=[3, 3, 3, 3, 4])) == NO_GO
            and _decision(_arm(missed=[1, 3, 2, 2, 2])) == GO,
        ),
        _check(
            "G4: Δ médio de F1 de fall logo abaixo de 0 dá NO-GO",
            _decision(_arm(fall=[0.0, 0.0, 0.0, 0.0, -1e-12])) == NO_GO,
        ),
        _check(
            "GO frágil: T com GO e média de Δ de S igual a 0 ou negativa; positiva mantém GO",
            _decision(_arm(), _arm(0.0)) == GO_FRAGILE
            and _decision(_arm(), _arm(-0.02)) == GO_FRAGILE
            and _decision(_arm(), _arm(1e-12)) == GO,
        ),
        _check(
            "S nunca transforma NO-GO em GO, e G1–G4 de S são relatadas",
            _decision(_arm(below), _arm(0.2, fall=0.1, missed=0, false=0)) == NO_GO
            and set(conditions["S"]) == {"G1", "G2", "G3", "G4"},
        ),
        _check(
            "qualquer falha de W1, W2 ou V2–V5 em T ou S dá INVÁLIDO, mesmo com G1–G4 aprovadas",
            _decision(_arm(), S_V3=["controle"]) == INVALID
            and _decision(_arm(), T_W2=["agregados"]) == INVALID
            and decide(_valid(T_V5=["quadros"]), None, _SEEDS)["decision"] == INVALID,
        ),
        _check(
            "validade incompleta ou decisão válida sem pares levanta erro",
            _raises(lambda: decide({"T": _valid()["T"]}, None, _SEEDS))
            and _raises(lambda: decide(_valid(), None, _SEEDS)),
        ),
        _check(
            "seed ausente, repetida ou fora das do controle levanta erro",
            _raises(lambda: decide(_valid(), {"T": _arm()[:4], "S": _arm()}, _SEEDS))
            and _raises(lambda: decide(_valid(), {"T": duplicated, "S": _arm()}, _SEEDS))
            and _raises(
                lambda: decide(_valid(), {"T": _arm(seeds=(42, 43, 44, 45, 47)), "S": _arm()}, _SEEDS)
            ),
        ),
        _check(
            "controle diferente entre T e S na mesma seed levanta erro",
            _raises(lambda: decide(_valid(), {"T": _arm(), "S": other_control}, _SEEDS)),
        ),
        _check(
            "decisão determinística e independente da ordem das seeds",
            json.dumps(reordered, sort_keys=True) == json.dumps(boundary, sort_keys=True),
        ),
    ]


def run_val_decision_selftest() -> None:
    checks: list[bool] = []
    with _from_frozen_checkout(checks), tempfile.TemporaryDirectory() as tmp_dir:
        checks.extend(check_decision_rules())
        checks.extend(check_concurrent_publication())
        repository = Path(tmp_dir) / "gatefall"
        repository.mkdir()
        world = _build(repository)
        checks.append(_setup(world))
        if checks[-1]:
            checks.extend(check_canonical_contract(world))
            checks.extend(check_evaluation(world))
            checks.extend(check_report_tampering(world))
            checks.extend(check_evaluation_lifecycle(world))
            checks.extend(check_interrupted_publication(world))
            checks.extend(check_decide(world))
            checks.extend(check_post_decision(world))
    if not all(checks):
        print("\nselftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: todos os casos passaram")
