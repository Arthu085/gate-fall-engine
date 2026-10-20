"""Selftest sintético do `test` descritivo do piloto GMDCSA24 com rótulos fracos.

Reaproveita o mundo sintético do selftest da decisão: T e S treinados na seed
42, o controle com `event_metrics.json` canônico, `val` avaliada pelo
`evaluate` e a decisão registrada pelo `decide`. A leitura de features passa
por uma trava que reprova qualquer vídeo de `test` enquanto ele não é
liberado, então toda recusa precisa acontecer antes de qualquer feature ou
predição de `test`. Requer FFmpeg; não toca no dataset real nem em `runs/`.
"""

import contextlib
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from gatefall.data.gmdcsa24.selftests.adapter import _check
from gatefall.data.gmdcsa24.selftests.pose_extract import _quiet
from gatefall.eval.gmdcsa24_weak_label import descriptive
from gatefall.eval.gmdcsa24_weak_label.decision import INVALID, NO_GO
from gatefall.eval.gmdcsa24_weak_label.selftests.val_decision import (
    _decide,
    _edit_json,
    _evaluate,
    _setup,
    _tampered,
)
from gatefall.eval.gmdcsa24_weak_label.descriptive import (
    TEST_CLAIM_FILE,
    TEST_METRICS_FILE,
    result_dir,
    run_test_description,
    split_metrics,
)
from gatefall.eval.gmdcsa24_weak_label.validation import (
    CANONICAL_EVENT_METRICS_FILE,
    VAL_EVENT_METRICS_FILE,
)
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.features.standardization import load_stats
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.config import load_config
from gatefall.train.gmdcsa24_weak_label.assembly import arm_a_features
from gatefall.train.gmdcsa24_weak_label.gates import (
    DECISION_FILE,
    DEFAULT_RUN_ROOT,
    TEST_DESCRIPTIVE_DIR,
    TREATMENT_RECORD_FILE,
    canonical_run_dir,
    test_access_problems,
)
from gatefall.train.gmdcsa24_weak_label.selftests.treatment import (
    _SEED,
    _TEST_VIDEOS,
    _build,
    _canonical_train,
    _from_frozen_checkout,
    _tree,
    _World,
)


class _TestLock:
    """Leitura de features que reprova `test` até `allow` e registra o que leu e
    o que tentou ler travado."""

    def __init__(self) -> None:
        self.allow = False
        self.test_reads: list[str] = []
        self.blocked: list[str] = []

    def __call__(self, video_id: str, pose_root: Path) -> np.ndarray:
        if video_id in _TEST_VIDEOS:
            if not self.allow:
                self.blocked.append(video_id)
                raise AssertionError(f"feature de test lida antes da liberação: {video_id}")
            self.test_reads.append(video_id)
        return arm_a_features(video_id, pose_root)


class _Crash(BaseException):
    """Queda do processo no meio da avaliação de `test`, antes da publicação."""


class _Context:
    def __init__(self, world: _World) -> None:
        self.world = world
        self.reader = _TestLock()
        self.decision_sha256 = ""

    @property
    def decision_path(self) -> Path:
        return self.world.tmp / DEFAULT_RUN_ROOT / DECISION_FILE

    @property
    def test_root(self) -> Path:
        return self.world.tmp / DEFAULT_RUN_ROOT / TEST_DESCRIPTIVE_DIR

    def artifact(self, treatment: str) -> Path:
        return self.world.tmp / result_dir(treatment, _SEED) / TEST_METRICS_FILE

    def describe(
        self,
        treatment: str = "T",
        reason: str | None = None,
        decision_sha256: str | None = None,
        seed: int = _SEED,
        **changes: Any,
    ) -> bool:
        """`describe-test`; uma leitura de `test` travada conta como falha."""
        inputs = self.world.inputs(treatment, load_features=self.reader, **changes)
        try:
            with contextlib.chdir(self.world.tmp), _quiet():
                return run_test_description(
                    inputs, seed, reason, decision_sha256 or self.decision_sha256
                )
        except AssertionError:
            return False

    def refused_untouched(self, outcome: bool) -> bool:
        """Recusa sem ler `test` e sem criar resultado de `test`."""
        return self.refused_unread(outcome) and not self.test_root.exists()

    def refused_unread(self, outcome: bool, reads: int = 0) -> bool:
        """Recusa sem nenhuma leitura de `test` além das `reads` anteriores, nem tentada."""
        return not outcome and len(self.reader.test_reads) == reads and not self.reader.blocked


def _protected(world: _World) -> dict[str, dict[str, str]]:
    """Runs do tratamento, controle, preparações e a decisão: nada disso muda."""
    root = world.tmp / DEFAULT_RUN_ROOT
    return {
        "t": _tree(root / "t"),
        "s": _tree(root / "s"),
        "control": _tree(world.control.run_dir),
        "training": _tree(world.tmp / "training"),
        "decision": {DECISION_FILE: sha256_file(root / DECISION_FILE)},
    }


def _setup_decision(context: _Context) -> bool:
    world = context.world
    ready = _setup(world) and all(_evaluate(world, treatment) for treatment in "TS")
    premature = ready and context.refused_untouched(context.describe())
    decided = premature and _decide(world) and context.decision_path.is_file()
    if decided:
        context.decision_sha256 = sha256_file(context.decision_path)
    return all(
        [
            _check("setup: val de T e S avaliada antes da decisão", ready),
            _check(
                "sem decision.json, describe-test recusa sem ler test nem criar resultado",
                premature,
            ),
            _check("decide registra a decisão sobre T e S", decided),
        ]
    )


def check_decision_safeguards(context: _Context) -> list[bool]:
    world = context.world
    decision = context.decision_path
    before = decision.read_bytes()

    def refused_with(update: Callable[[dict[str, Any]], None]) -> bool:
        with _tampered(decision, _edit_json(update)):
            return context.refused_untouched(
                context.describe(decision_sha256=sha256_file(decision))
            )

    def invalid(data: dict[str, Any]) -> None:
        for field in ("conditions", "per_seed"):
            data.pop(field)
        data.update(decision=INVALID, validity_failures={"T": {"V2": ["controle"]}}, inputs=None)

    def other_decision(data: dict[str, Any]) -> None:
        data["decision"] = "GO exploratório" if data["decision"] == NO_GO else NO_GO

    def missing_seed(data: dict[str, Any]) -> None:
        data["inputs"]["S"].pop(str(_SEED))

    def no_seeds(data: dict[str, Any]) -> None:
        data["seeds"] = []

    def test_split(data: dict[str, Any]) -> None:
        data["evaluated_splits"] = ["val", "test"]

    def other_run(data: dict[str, Any]) -> None:
        data["inputs"]["T"][str(_SEED)]["run_dir"] = "runs/local/outro/seed_42"

    def other_metric(data: dict[str, Any]) -> None:
        data["per_seed"]["T"][str(_SEED)]["treatment"]["n_missed_events"] += 1

    def dirty(data: dict[str, Any]) -> None:
        data["code"]["dirty"] = True

    s_report = world.tmp / canonical_run_dir("S", _SEED) / VAL_EVENT_METRICS_FILE
    with _tampered(s_report, _edit_json(lambda data: data["splits"]["val"].update(n_alarms_total=99))):
        tampered_input = context.refused_untouched(context.describe())
    control_report = world.control.run_dir / CANONICAL_EVENT_METRICS_FILE
    with _tampered(control_report, _edit_json(lambda data: data.update(run_name="outro"))):
        tampered_control = context.refused_untouched(context.describe())
    copy = world.tmp / "control_copy" / "seed_42"
    shutil.copytree(world.control.run_dir, copy)
    record = world.tmp / canonical_run_dir("T", _SEED) / TREATMENT_RECORD_FILE
    with _tampered(record, _edit_json(lambda data: data["control"].update(run_dir=copy.as_posix()))):
        other_control = context.refused_untouched(context.describe())
    shutil.rmtree(copy.parent)
    outside = world.tmp / canonical_run_dir("T", 47)
    outside.mkdir(parents=True)
    shutil.copy2(record, outside / TREATMENT_RECORD_FILE)
    outside_seed = context.refused_untouched(context.describe(seed=47))
    shutil.rmtree(outside)
    link = decision.with_name("decision.target.json")
    with _tampered(decision, lambda path: (path.rename(link), path.symlink_to(link.name))):
        symlinked = context.refused_untouched(context.describe())
    link.unlink(missing_ok=True)
    return [
        _check(
            "decision.json com SHA-256 diferente do congelado recusa sem ler test",
            context.refused_untouched(context.describe(decision_sha256="0" * 64)),
        ),
        _check(
            "decisão INVÁLIDO não libera test, mesmo com o SHA-256 dela",
            refused_with(invalid),
        ),
        _check(
            "decisão diferente da recalculada dos arquivos lidos ou medida de seed trocada "
            "recusam sem ler test",
            refused_with(other_decision) and refused_with(other_metric),
        ),
        _check(
            "seed ausente nas entradas ou seeds diferentes das do controle recusam sem ler test",
            refused_with(missing_seed) and refused_with(no_seeds),
        ),
        _check(
            "decisão com test avaliado, run fora do canônico ou código sujo recusa sem ler test",
            refused_with(test_split) and refused_with(other_run) and refused_with(dirty),
        ),
        _check(
            "relatório de val, event_metrics.json do controle ou treatment.json alterados "
            "depois da decisão recusam sem ler test",
            tampered_input and tampered_control and other_control,
        ),
        _check("decision.json como link simbólico recusa sem ler test", symlinked),
        _check(
            "seed sem run, seed com run mas fora da decisão e checkout sujo recusam sem ler test",
            context.refused_untouched(context.describe(seed=43))
            and outside_seed
            and context.refused_untouched(context.describe(checkout=("e" * 40, ["?? arquivo"]))),
        ),
        _check("decision.json intacto depois das recusas", decision.read_bytes() == before),
    ]


def _remove_marker(context: _Context) -> None:
    """Remove o marcador plantado sem seguir links: alvos ficam intactos."""
    if context.test_root.is_symlink():
        context.test_root.unlink()
    elif context.test_root.exists():
        shutil.rmtree(context.test_root)


def _stray_outputs(world: _World) -> list[Path]:
    """Locks e reservas de `test` em qualquer lugar do mundo sintético."""
    return [
        path
        for name in (descriptive.TEST_LOCK_FILE, TEST_CLAIM_FILE, TEST_METRICS_FILE)
        for path in world.tmp.rglob(name)
        if not path.is_symlink()
    ]


def check_output_isolation(context: _Context) -> list[bool]:
    """Saída de `test` com link simbólico ou sobreposta a algo protegido: recusa
    antes de criar diretório, lock ou reserva, sem ler `test` e sem mudar nada."""
    world = context.world
    root = world.tmp / DEFAULT_RUN_ROOT
    run_dir = world.tmp / canonical_run_dir("T", _SEED)
    elsewhere = world.tmp / "elsewhere"
    elsewhere.mkdir()

    def seed_link(target: Path) -> Callable[[], None]:
        def plant() -> None:
            (context.test_root / "t").mkdir(parents=True)
            (context.test_root / "t" / f"seed_{_SEED}").symlink_to(target)

        return plant

    def treatment_link(target: Path) -> Callable[[], None]:
        def plant() -> None:
            context.test_root.mkdir()
            (context.test_root / "t").symlink_to(target)

        return plant

    def lock_link() -> None:
        directory = context.test_root / "t" / f"seed_{_SEED}"
        directory.mkdir(parents=True)
        (directory / descriptive.TEST_LOCK_FILE).symlink_to(run_dir / "metrics.json")

    cases: dict[str, tuple[Callable[[], None], str]] = {
        "seed → run de T": (seed_link(run_dir), "link simbólico"),
        "marcador → runs de T": (lambda: context.test_root.symlink_to(root / "t"), "se sobrepõe"),
        "tratamento → preparações": (treatment_link(world.tmp / "training"), "se sobrepõe"),
        "seed → controle": (seed_link(world.control.run_dir), "se sobrepõe"),
        "seed → diretório qualquer": (seed_link(elsewhere), "link simbólico"),
        "lock → metrics.json do run": (lock_link, "link simbólico"),
        "marcador pendente": (
            lambda: context.test_root.symlink_to(world.tmp / "missing"),
            "link simbólico",
        ),
    }
    protected = _protected(world)
    outcomes: dict[str, bool] = {}
    for name, (plant, reason) in cases.items():
        plant()
        refusals: list[str] = []

        def spy(command: str, problems: list[str]) -> bool:
            refusals.extend(problems)
            return False

        with patch.object(descriptive, "refuse", spy):
            refused = context.refused_unread(context.describe("T"))
        outcomes[name] = (
            refused
            and any(reason in problem for problem in refusals)
            and _protected(world) == protected
            and not _stray_outputs(world)
            and not any(elsewhere.iterdir())
            and not (root / "t" / "t").exists()
            and not (world.tmp / "missing").exists()
        )
        _remove_marker(context)

    context.test_root.symlink_to(world.tmp / "missing")
    frozen = bool(test_access_problems(root))
    with _tampered(context.decision_path, lambda path: path.unlink()):
        decide_refused = not _decide(world) and not context.decision_path.exists()
        report = run_dir / VAL_EVENT_METRICS_FILE
        with _tampered(report, lambda path: path.unlink()):
            evaluate_refused = not _evaluate(world, "T", force=True) and not report.exists()
    dangling_intact = context.test_root.is_symlink() and not (world.tmp / "missing").exists()
    _remove_marker(context)
    shutil.rmtree(elsewhere)
    return [
        _check(
            f"saída de test recusada antes de diretório, lock ou reserva ({name}), sem ler test "
            "e com runs, preparações, controle e decisão intactos",
            ok,
        )
        for name, ok in outcomes.items()
    ] + [
        _check(
            "marcador test_descriptive/ como link pendente congela a decisão: decide e "
            "evaluate --force recusam sem gravar",
            frozen and decide_refused and evaluate_refused and dangling_intact,
        ),
        _check(
            "nenhum marcador restante antes do caminho canônico",
            not context.test_root.exists() and not context.test_root.is_symlink(),
        ),
    ]


def _canonical_test(world: _World) -> dict[str, dict]:
    """O `test` canônico do controle: `final.test` e `splits.test` publicados."""
    run_dir = world.control.run_dir
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    events = json.loads((run_dir / CANONICAL_EVENT_METRICS_FILE).read_text(encoding="utf-8"))
    return {"classification": metrics["final"]["test"], "events": events["splits"]["test"]}


def check_description(context: _Context) -> list[bool]:
    world = context.world
    protected = _protected(world)
    context.reader.allow = True
    described = context.describe("T")
    context.reader.allow = False
    artifact = context.artifact("T")
    if not described or not artifact.is_file():
        return [_check("describe-test de T publica o test descritivo depois da decisão", False)]
    record = json.loads(artifact.read_text(encoding="utf-8"))
    claim = json.loads((artifact.parent / TEST_CLAIM_FILE).read_text(encoding="utf-8"))
    run_dir = world.tmp / canonical_run_dir("T", _SEED)
    canonical = _canonical_test(world)
    config = load_config(world.control.run_dir / "config.yaml")
    with contextlib.chdir(world.tmp):
        recomputed_control = split_metrics(
            config,
            world.control.run_dir / "checkpoint.pt",
            load_stats(world.le2i.stats_path),
            world.le2i.frames,
            lambda video_id: arm_a_features(video_id, world.le2i.pose_root),
            world.gate.eval_index_sha256["test"],
        )
    treatment, control = record["metrics"]["treatment"], record["metrics"]["control"]
    delta = record["delta"]
    decision = json.loads(context.decision_path.read_text(encoding="utf-8"))
    return [
        _check(
            "describe-test de T publica o test descritivo depois da decisão",
            described and claim["published_sha256"] == sha256_file(artifact),
        ),
        _check(
            "só depois de todas as conferências test é lido, e só o vídeo de test",
            set(context.reader.test_reads) == _TEST_VIDEOS,
        ),
        _check(
            "o resultado registra só test, descritivo, com checkpoint, split, controle e decisão",
            record["descriptive_only"] is True
            and record["evaluated_splits"] == ["test"]
            and (record["treatment"], record["seed"]) == ("T", _SEED)
            and record["checkpoint"]["sha256"] == sha256_file(run_dir / "checkpoint.pt")
            and record["run"]["files_sha256"][VAL_EVENT_METRICS_FILE]
            == sha256_file(run_dir / VAL_EVENT_METRICS_FILE)
            and record["split_provenance"]["test_index_sha256"] == world.gate.eval_index_sha256["test"]
            and record["control"]["files_sha256"]["checkpoint.pt"]
            == world.gate.control_checkpoint_sha256[_SEED]
            and record["decision"]["sha256"] == context.decision_sha256
            and record["decision"]["decision"] == decision["decision"]
            and record["alarm_protocol"] == BASELINE_A_ALARM_PROTOCOL.to_dict(),
        ),
        _check(
            "o test do controle é o canônico publicado, e o caminho do tratamento aplicado "
            "ao controle reproduz final.test e splits.test canônicos",
            control == canonical and recomputed_control == canonical,
        ),
        _check(
            "Δ = tratamento − controle, sem limiar nem aprovação",
            delta["macro_f1_restricted"]
            == treatment["classification"]["macro_f1_restricted"]
            - control["classification"]["macro_f1_restricted"]
            and delta["events"]["n_missed_events"]
            == treatment["events"]["n_missed_events"] - control["events"]["n_missed_events"]
            and delta["events"]["n_false_alarms"]
            == treatment["events"]["n_false_alarms"] - control["events"]["n_false_alarms"]
            and "passed" not in json.dumps(record),
        ),
        _check(
            "runs, relatórios de val, preparações, controle e decision.json intactos",
            _protected(world) == protected,
        ),
    ]


def check_once(context: _Context) -> list[bool]:
    world = context.world
    artifact = context.artifact("T")
    before = artifact.read_bytes()
    reads = len(context.reader.test_reads)
    duplicate = context.refused_unread(context.describe("T"), reads) and context.refused_unread(
        context.describe("T", reason="falha técnica"), reads
    )
    with _tampered(artifact, lambda path: path.unlink()):
        deleted = (
            context.refused_unread(context.describe("T", reason="falha técnica"), reads)
            and not artifact.exists()
        )

    def drop_claim_and_result(directory: Path) -> None:
        (directory / TEST_CLAIM_FILE).unlink()
        (directory / TEST_METRICS_FILE).unlink()

    with _tampered(artifact.parent, drop_claim_and_result):
        erased = context.refused_unread(context.describe("T"), reads) and not artifact.exists()
    with _tampered(context.decision_path, lambda path: path.unlink()):
        decide_refused = not _decide(world) and not context.decision_path.exists()
        report = world.tmp / canonical_run_dir("T", _SEED) / VAL_EVENT_METRICS_FILE
        with _tampered(report, lambda path: path.unlink()):
            evaluate_refused = not _evaluate(world, "T", force=True) and not report.exists()
        record = world.tmp / canonical_run_dir("T", _SEED) / TREATMENT_RECORD_FILE
        with _tampered(record, lambda path: path.unlink()):
            train_refused = (
                not _canonical_train(world, world.tmp, True, "falha técnica") and not record.exists()
            )
    return [
        _check(
            "segundo describe-test da mesma seed recusa, com e sem motivo, sem tentar ler test "
            "nem regravar",
            duplicate and artifact.read_bytes() == before,
        ),
        _check(
            "resultado publicado e apagado nunca é refeito, nem com --recovery-reason",
            deleted,
        ),
        _check(
            "resultado e reserva apagados com o diretório da seed presente recusam sem ler test",
            erased,
        ),
        _check(
            "com test acessado e sem decision.json, decide não registra nova decisão e "
            "evaluate --force e train --force recusam sem gravar",
            decide_refused and evaluate_refused and train_refused,
        ),
    ]


def _crashed(context: _Context, treatment: str, reason: str | None = None) -> bool:
    """`describe-test` com queda do processo no meio da avaliação de `test`."""

    def crash(*args: Any, **kwargs: Any) -> object:
        raise _Crash()

    context.reader.allow = True
    try:
        with patch.object(descriptive, "_split_reports", crash):
            context.describe(treatment, reason=reason)
        return False
    except _Crash:
        return True
    finally:
        context.reader.allow = False


def _claim(context: _Context, treatment: str) -> dict[str, Any]:
    path = context.artifact(treatment).parent / TEST_CLAIM_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def check_interrupted(context: _Context) -> list[bool]:
    """Tentativa de S interrompida antes da publicação: refeita uma vez, com motivo."""
    artifact = context.artifact("S")
    directory = artifact.parent
    interrupted = _crashed(context, "S") and not artifact.exists() and (directory / TEST_CLAIM_FILE).is_file()
    reads = len(context.reader.test_reads)
    without_reason = context.refused_unread(context.describe("S"), reads)

    def second_crash(_: Path) -> None:
        _crashed(context, "S", reason="falha técnica")

    with _tampered(directory, second_crash):
        reads = len(context.reader.test_reads)
        capped = len(_claim(context, "S")["attempts"]) == 2 and context.refused_unread(
            context.describe("S", reason="outra falha"), reads
        )

    def integrity_failure(_: Path) -> None:
        context.reader.allow = True
        with patch.object(descriptive, "structure_problems", lambda *_: ["estrutura divergente"]):
            context.describe("S", reason="falha técnica")
        context.reader.allow = False

    with _tampered(directory, integrity_failure):
        reads = len(context.reader.test_reads)
        terminal = (
            _claim(context, "S").get("integrity_failure") == "estrutura divergente"
            and not artifact.exists()
            and context.refused_unread(context.describe("S", reason="falha técnica"), reads)
        )
    context.reader.allow = True
    recovered = context.describe("S", reason="falha técnica: processo interrompido")
    context.reader.allow = False
    record = json.loads(artifact.read_text(encoding="utf-8")) if recovered else {}
    attempts = record.get("attempts", [])
    return [
        _check(
            "tentativa interrompida antes da publicação deixa a reserva e nenhum resultado",
            interrupted,
        ),
        _check(
            "tentativa interrompida sem --recovery-reason recusa sem tentar ler test",
            without_reason,
        ),
        _check(
            "depois de uma repetição, uma terceira tentativa recusa sem tentar ler test",
            capped,
        ),
        _check(
            "reprovação de integridade depois de calcular test fica registrada e nunca é refeita",
            terminal,
        ),
        _check(
            "com --recovery-reason, publica uma vez e registra as duas tentativas e o motivo",
            recovered
            and len(attempts) == 2
            and attempts[0]["recovery_reason"] is None
            and attempts[1]["recovery_reason"] == "falha técnica: processo interrompido"
            and _claim(context, "S")["published_sha256"] == sha256_file(artifact),
        ),
        _check(
            "T e S têm resultados distintos, cada um com o próprio tratamento",
            context.artifact("T") != artifact
            and json.loads(context.artifact("T").read_text(encoding="utf-8"))["treatment"] == "T"
            and record.get("treatment") == "S",
        ),
    ]


def run_descriptive_selftest() -> None:
    checks: list[bool] = []
    with _from_frozen_checkout(checks), tempfile.TemporaryDirectory() as tmp_dir:
        repository = Path(tmp_dir) / "gatefall"
        repository.mkdir()
        context = _Context(_build(repository))
        checks.append(_setup_decision(context))
        if checks[-1]:
            checks.extend(check_decision_safeguards(context))
            checks.extend(check_output_isolation(context))
            checks.extend(check_description(context))
            checks.extend(check_once(context))
            checks.extend(check_interrupted(context))
    if not all(checks):
        print("\nselftest de test descritivo FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: test descritivo, todos os casos passaram")
