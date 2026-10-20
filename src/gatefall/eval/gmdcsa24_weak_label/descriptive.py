"""Métricas descritivas de `test` do piloto GMDCSA24 com rótulos fracos (braço A).

`describe-test` roda estritamente depois da decisão GO/NO-GO registrada, uma
única vez por tratamento e seed, e não decide nada. Antes de montar qualquer
feature ou predição de `test`, confere o `decision.json` (SHA-256 congelado,
campos, seeds, proveniência, hashes de tudo o que ele leu e a decisão
recalculada desses arquivos), o run canônico pelos portões de `evaluate` e o
controle pareado registrado. Reaproveita do braço A o carregamento do
checkpoint, a padronização (com o JSON do tratamento), as métricas de
classificação de `final.test` do treino canônico, a associação de eventos e
`BASELINE_A_ALARM_PROTOCOL`. O controle não é reavaliado: o `test` dele vem do
`metrics.json` e do `event_metrics.json` canônicos já publicados. O resultado
fica num diretório próprio, fora dos runs, e nunca é substituído.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gatefall.config import EVAL_STRIDE
from gatefall.data.gmdcsa24.pose_extract import read_json, section
from gatefall.data.gmdcsa24.weak_label import TREATMENTS, compare_expected
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import build_window_index
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.multiseed_summary import validate_run_event_artifacts
from gatefall.eval.baseline_a.cli import _load_model, _predict_with_identity
from gatefall.eval.gmdcsa24_weak_label.cli import (
    DECISION_CRITERION,
    DECISION_LOCK_FILE,
    RUN_INPUT_FILES,
    changed_inputs,
    decision_lock,
    publish_once,
    seed_metrics,
)
from gatefall.eval.gmdcsa24_weak_label.decision import (
    GO,
    GO_FRAGILE,
    NO_GO,
    VALIDITY_CONDITIONS,
    PairedSeed,
    decide,
)
from gatefall.eval.gmdcsa24_weak_label.validation import (
    CANONICAL_EVENT_METRICS_FILE,
    EVENT_STRUCTURE_FIELDS,
    VAL_EVENT_METRICS_FILE,
    CheckedRun,
    check_run,
    control_from_record,
    val_report,
)
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.shared.orchestration import Predictions, _split_reports
from gatefall.features.standardization import StandardizationStats
from gatefall.hashing import sha256_dataframe, sha256_file
from gatefall.train.baseline_a.config import TrainConfig, load_config
from gatefall.train.baseline_a.engine import _evaluate_split, _StandardizedTorchDataset
from gatefall.train.gmdcsa24_weak_label.assembly import aligned_loader
from gatefall.train.gmdcsa24_weak_label.cli import refuse
from gatefall.train.gmdcsa24_weak_label.gates import (
    DECISION_FILE,
    DEFAULT_RUN_ROOT,
    DEFAULT_TRAINING_DIR,
    TEST_DESCRIPTIVE_DIR,
    TREATMENT_RECORD_FILE,
    ControlRun,
    TreatmentInputs,
    canonical_run_dir,
    code_problems,
    execution_root_problems,
    location_problems,
    overlaps,
    report,
)
from gatefall.train.gmdcsa24_weak_label.lifecycle import _recorded_at_problems, run_artifacts

# SHA-256 do `decision.json` NO-GO registrado pelo responsável em 2026-10-20,
# no commit 8c99670 de `experiment/external-fall-data`.
PINNED_DECISION_SHA256 = "9f57ab36d628194950a7edb490a30175a5aba90ae7bcf72c41a7445904ceae5d"

TEST_SPLITS: tuple[str, ...] = ("test",)
TEST_METRICS_FILE = "test_metrics.json"
TEST_CLAIM_FILE = "claim.json"
TEST_LOCK_FILE = ".test.lock"
# A primeira tentativa e no máximo uma repetição depois de falha técnica.
MAX_ATTEMPTS = 2
RECORDED_DECISIONS = (GO, GO_FRAGILE, NO_GO)
DECISION_FIELDS = frozenset(
    {
        "criterion",
        "split",
        "seeds",
        "decision",
        "conditions",
        "per_seed",
        "evaluated_splits",
        "inputs",
        "code",
        "recorded_at",
    }
)
RECOMPUTED_FIELDS = ("split", "seeds", "decision", "conditions", "per_seed")
INPUT_FIELDS = frozenset({"run_dir", "control_run_dir", *(field for field, _, _ in RUN_INPUT_FILES)})
LATENCY_STATISTICS = ("mean", "median")

FeatureReader = Callable[[str], np.ndarray]


def result_dir(treatment: str, seed: int) -> Path:
    """O único local do resultado de `test` de um tratamento e uma seed."""
    return DEFAULT_RUN_ROOT / TEST_DESCRIPTIVE_DIR / treatment.lower() / f"seed_{seed}"


def protected_paths(inputs: TreatmentInputs, controls: Iterable[Path]) -> dict[str, Path]:
    """O que a saída de `test` nunca pode alcançar: runs de T e S, preparações,
    controles pareados, artefatos da decisão e as entradas do piloto."""
    return {
        "runs de T": DEFAULT_RUN_ROOT / "t",
        "runs de S": DEFAULT_RUN_ROOT / "s",
        "decision.json": DEFAULT_RUN_ROOT / DECISION_FILE,
        "lock da decisão": DEFAULT_RUN_ROOT / DECISION_LOCK_FILE,
        "--training-dir": inputs.training_dir,
        "preparações padrão": DEFAULT_TRAINING_DIR,
        "adapter do GMDCSA24": inputs.adapter_dir,
        "pose do GMDCSA24": inputs.pose_dir,
        "pose do Le2i-CS": inputs.le2i.pose_root,
        "grade do Le2i-CS": inputs.le2i.frames_path,
        "JSON de padronização do controle": inputs.le2i.stats_path,
        **{f"controle {control}": control for control in controls},
    }


def output_isolation_problems(
    inputs: TreatmentInputs, seed: int, controls: Iterable[Path]
) -> list[str]:
    """A saída de `test` da seed fica no layout canônico sob a raiz dos runs, sem
    link simbólico em nenhum componente, e não alcança, pelos caminhos
    resolvidos, nada protegido; conferido antes de criar diretório, lock ou reserva."""
    directory = result_dir(inputs.treatment, seed)
    marker = DEFAULT_RUN_ROOT / TEST_DESCRIPTIVE_DIR
    components = [
        marker,
        marker / inputs.treatment.lower(),
        directory,
        *(directory / name for name in (TEST_CLAIM_FILE, TEST_METRICS_FILE, TEST_LOCK_FILE)),
    ]
    problems = [
        f"{path} é link simbólico; a saída de test não segue links"
        for path in components
        if path.is_symlink()
    ]
    expected = DEFAULT_RUN_ROOT.resolve() / directory.relative_to(DEFAULT_RUN_ROOT)
    if directory.resolve() != expected:
        problems.append(f"{directory} resolve para {directory.resolve()}, não para {expected}")
    for name, target in protected_paths(inputs, controls).items():
        problems.extend(
            f"{path} (resolve para {path.resolve()}) se sobrepõe a {name} ({target})"
            for path in (marker, directory)
            if overlaps(path, target)
        )
    problems.extend(location_problems(directory))
    return problems


def record_problems(record: Mapping[str, object], inputs: TreatmentInputs) -> list[str]:
    """Critério, split, seeds do controle, commit e um par de
    entradas por tratamento e seed, no run canônico e com o mesmo controle em T e S."""
    seeds = sorted(inputs.gate.control_checkpoint_sha256)
    expected = {
        "criterion": DECISION_CRITERION,
        "split": "val",
        "evaluated_splits": ["val"],
        "seeds": seeds,
    }
    label = "decisão registrada"
    problems = compare_expected(expected, {key: record.get(key) for key in expected}, f"{label}: ")
    problems.extend(_recorded_at_problems(record.get("recorded_at"), label))
    problems.extend(code_problems(inputs, record.get("code"), label))
    by_treatment = record.get("inputs")
    if not isinstance(by_treatment, dict) or set(by_treatment) != set(TREATMENTS):
        return [*problems, f"{label}: inputs deve trazer exatamente T e S"]
    controls: dict[tuple[str, int], object] = {}
    for treatment in TREATMENTS:
        by_seed = by_treatment[treatment]
        if not isinstance(by_seed, dict):
            problems.append(f"{label}: inputs.{treatment} não é um objeto")
            continue
        keys = {str(seed) for seed in seeds}
        problems.extend(
            f"{label}: inputs.{treatment} sem a seed {seed}" for seed in sorted(keys - set(by_seed))
        )
        problems.extend(
            f"{label}: inputs.{treatment} com a seed {seed} fora das do controle {seeds}"
            for seed in sorted(set(by_seed) - keys)
        )
        for seed in seeds:
            if str(seed) not in by_seed:
                continue
            entry = by_seed[str(seed)]
            if (
                not isinstance(entry, dict)
                or set(entry) != INPUT_FIELDS
                or not all(isinstance(value, str) for value in entry.values())
            ):
                problems.append(
                    f"{label}: inputs.{treatment}.{seed} deve ter só textos em {sorted(INPUT_FIELDS)}"
                )
                continue
            if entry["run_dir"] != canonical_run_dir(treatment, seed).as_posix():
                problems.append(
                    f"{label}: inputs.{treatment}.{seed} lê {entry['run_dir']!r}, não o run "
                    f"canônico {canonical_run_dir(treatment, seed).as_posix()}"
                )
            controls[(treatment, seed)] = entry["control_run_dir"]
    problems.extend(
        f"{label}: seed {seed} com controles diferentes em T e S"
        for seed in seeds
        if ("T", seed) in controls
        and ("S", seed) in controls
        and controls[("T", seed)] != controls[("S", seed)]
    )
    return problems


def recomputation_problems(record: Mapping[str, object]) -> list[str]:
    """A decisão recalculada dos arquivos que ela leu, com a validade cumprida
    (só GO exploratório, GO frágil ou NO-GO chegam aqui), é a registrada."""
    by_treatment = cast(dict[str, dict[str, dict[str, str]]], record["inputs"])
    pairs: dict[str, list[PairedSeed]] = {treatment: [] for treatment in TREATMENTS}
    try:
        for treatment, by_seed in by_treatment.items():
            for seed, entry in by_seed.items():
                run_dir, control_dir = Path(entry["run_dir"]), Path(entry["control_run_dir"])
                events = section(section(read_json(run_dir / VAL_EVENT_METRICS_FILE), "splits"), "val")
                control = read_json(control_dir / CANONICAL_EVENT_METRICS_FILE)
                control_events = section(section(control, "splits"), "val")
                pairs[treatment].append(
                    PairedSeed(
                        int(seed),
                        seed_metrics(run_dir, events),
                        seed_metrics(control_dir, control_events),
                    )
                )
        validity = {
            treatment: {condition: [] for condition in VALIDITY_CONDITIONS} for treatment in TREATMENTS
        }
        recomputed = decide(validity, pairs, cast(list[int], record["seeds"]))
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return [f"decisão registrada não recalculável dos arquivos que ela leu ({exc})"]
    observed = json.loads(json.dumps({key: record[key] for key in RECOMPUTED_FIELDS}))
    expected = json.loads(json.dumps({key: recomputed[key] for key in RECOMPUTED_FIELDS}))
    return compare_expected(expected, observed, "decisão registrada × recalculada: ")


def decision_record(
    inputs: TreatmentInputs, expected_sha256: str
) -> tuple[dict[str, object] | None, list[str]]:
    """O `decision.json` íntegro e coerente com os arquivos que ele leu; antes
    dele, `test` não é acessado."""
    path = DEFAULT_RUN_ROOT / DECISION_FILE
    if not path.exists() and not path.is_symlink():
        return None, [
            f"{path} ausente: test só é avaliado depois da decisão GO/NO-GO registrada"
        ]
    if path.is_symlink() or not path.is_file():
        return None, [f"{path} não é um arquivo regular"]
    digest = sha256_file(path)
    if digest != expected_sha256:
        return None, [
            f"{path}: SHA-256 {digest} difere do da decisão registrada {expected_sha256}"
        ]
    try:
        record = read_json(path)
    except (OSError, ValueError) as exc:
        return None, [f"{path} ilegível ({exc})"]
    if isinstance(record, dict) and record.get("decision") not in RECORDED_DECISIONS:
        return None, [
            f"{path}: decisão {record.get('decision')!r} não libera test; só "
            f"{', '.join(RECORDED_DECISIONS)} encerram a trilha"
        ]
    if not isinstance(record, dict) or set(record) != DECISION_FIELDS:
        fields = sorted(record) if isinstance(record, dict) else record
        return None, [f"{path}: campos {fields!r}, esperados {sorted(DECISION_FIELDS)}"]
    problems = record_problems(record, inputs)
    if not problems:
        problems = changed_inputs(record)
    if not problems:
        problems = recomputation_problems(record)
    return (None if problems else record), problems


def paired_control(
    treatment: str, seed: int, record: Mapping[str, object] | None = None
) -> tuple[ControlRun | None, list[str]]:
    """O controle do `treatment.json` do run, o mesmo que a decisão leu."""
    path = canonical_run_dir(treatment, seed) / TREATMENT_RECORD_FILE
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        return None, [f"{path} ilegível ({exc})"]
    control = control_from_record(data) if isinstance(data, dict) else None
    if control is None:
        return None, [f"{path}: bloco control ilegível"]
    if record is None:
        return control, []
    inputs = cast(dict[str, dict[str, dict[str, str]]], record["inputs"])
    if seed not in cast(list[int], record["seeds"]) or str(seed) not in inputs[treatment]:
        return None, [f"seed {seed} fora das seeds da decisão registrada {record['seeds']}"]
    recorded = Path(inputs[treatment][str(seed)]["control_run_dir"])
    if control.run_dir.resolve() != recorded.resolve():
        return None, [f"{path}: controle {control.run_dir}, a decisão leu {recorded}"]
    return control, []


def split_frames(frames: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, frames[frames["split"].isin(TEST_SPLITS)])


def split_metrics(
    config: TrainConfig,
    checkpoint: Path,
    stats: StandardizationStats,
    frames: pd.DataFrame,
    read_features: FeatureReader,
    expected_index_sha256: str,
) -> dict[str, dict]:
    """Classificação de `test` como em `final.test` do treino canônico (janelas
    rotuladas em `EVAL_STRIDE`) e eventos como no avaliador canônico (sem
    descartar `IGNORE_LABEL`), sob `BASELINE_A_ALARM_PROTOCOL`."""
    frames = split_frames(frames)
    classification_source = PoseWindowDataset(frames, "test", EVAL_STRIDE, read_features)
    index_sha256 = sha256_dataframe(classification_source.windows)
    if index_sha256 != expected_index_sha256:
        raise RuntimeError(
            f"índice de test com hash {index_sha256}, o publicado é {expected_index_sha256}"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(config, checkpoint, device)
    loader = DataLoader(
        _StandardizedTorchDataset(classification_source, stats),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    classification = _evaluate_split(model, loader, device, config.num_classes, LE2I_LABEL_NAMES)
    event_source = PoseWindowDataset(
        frames, "test", EVAL_STRIDE, read_features, drop_ignored=False
    )

    def evaluate_split(split: str) -> tuple[int, Predictions]:
        if split not in TEST_SPLITS:
            raise RuntimeError(f"split {split!r} recusado: só test é descrito aqui")
        predictions = _predict_with_identity(
            model, event_source, stats, device, batch_size=config.batch_size
        )
        return len(event_source), predictions

    events = _split_reports(frames, evaluate_split, TEST_SPLITS)["test"]
    return json.loads(json.dumps({"classification": classification, "events": events}))


def treatment_reader(inputs: TreatmentInputs) -> FeatureReader:
    windows = build_window_index(split_frames(inputs.le2i.frames), stride=EVAL_STRIDE, drop_ignored=False)
    return aligned_loader(windows, inputs.roots, inputs.load_features)


def control_test(control_dir: Path) -> dict[str, dict]:
    """O `test` canônico já publicado do controle: `final.test` do `metrics.json`
    e `splits.test` do `event_metrics.json`, validados pelo sumário multi-seed."""
    config = load_config(control_dir / "config.yaml")
    events = validate_run_event_artifacts(control_dir, config)["splits"]["test"]
    metrics = json.loads((control_dir / "metrics.json").read_text(encoding="utf-8"))
    return {"classification": metrics["final"]["test"], "events": events}


def structure_problems(treatment: Mapping[str, dict], control: Mapping[str, dict]) -> list[str]:
    """Mesmas janelas, horas, quedas e suporte por classe em `test` nos dois lados."""
    problems = compare_expected(
        {field: control["events"].get(field) for field in EVENT_STRUCTURE_FIELDS},
        {field: treatment["events"].get(field) for field in EVENT_STRUCTURE_FIELDS},
        "test do tratamento × test do controle: ",
    )
    problems.extend(
        compare_expected(
            control["classification"].get("support"),
            treatment["classification"].get("support"),
            "suporte de test do tratamento × do controle: ",
        )
    )
    return problems


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def descriptive_delta(treatment: Mapping[str, dict], control: Mapping[str, dict]) -> dict[str, object]:
    """Δ = tratamento − controle na mesma seed, sem limiar nem aprovação."""
    t_cls, c_cls = treatment["classification"], control["classification"]
    t_events, c_events = treatment["events"], control["events"]
    latency = {
        statistic: (
            t_events["latency_seconds"][statistic] - c_events["latency_seconds"][statistic]
            if is_number(t_events["latency_seconds"][statistic])
            and is_number(c_events["latency_seconds"][statistic])
            else None
        )
        for statistic in LATENCY_STATISTICS
    }
    return {
        "macro_f1_restricted": t_cls["macro_f1_restricted"] - c_cls["macro_f1_restricted"],
        "f1_by_class": {
            label: t_cls["f1_by_class"][label] - c_cls["f1_by_class"][label]
            for label in t_cls["f1_by_class"]
        },
        "events": {
            field: t_events[field] - c_events[field]
            for field in t_events
            if field not in EVENT_STRUCTURE_FIELDS
            and is_number(t_events[field])
            and is_number(c_events.get(field))
        },
        "latency_seconds": latency,
    }


@contextlib.contextmanager
def exclusive_lock(directory: Path) -> Iterator[None]:
    """Lock entre processos do resultado de uma seed; espera em vez de falhar.
    O arquivo de lock nunca é aberto através de um link simbólico."""
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        directory / TEST_LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def write_claim(path: Path, claim: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(claim, indent=2, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def claim_attempts(
    directory: Path,
    checkout: tuple[str, list[str]],
    recovery_reason: str | None,
    existed: bool,
) -> tuple[list[dict[str, object]] | None, list[str]]:
    """Reserva a única avaliação de `test` da seed antes de qualquer acesso a ele.
    Só uma tentativa interrompida por falha técnica antes da publicação é
    refeita, uma vez e com motivo registrado; um resultado publicado, uma
    reprovação de integridade depois de calcular `test` ou uma reserva apagada
    nunca são refeitos."""
    artifact = directory / TEST_METRICS_FILE
    claim_path = directory / TEST_CLAIM_FILE
    if artifact.exists() or artifact.is_symlink():
        return None, [
            f"{artifact}: test já avaliado e publicado; um resultado de test nunca é "
            "refeito nem substituído"
        ]
    if existed and not claim_path.exists():
        return None, [
            f"{directory} existe sem {TEST_CLAIM_FILE}: reserva ou resultado apagados; "
            "nada é refeito; inspecione manualmente"
        ]
    attempt: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {"commit": checkout[0], "dirty": bool(checkout[1])},
        "recovery_reason": recovery_reason,
    }
    if not claim_path.exists():
        if recovery_reason is not None:
            return None, [f"{claim_path} ausente: não há tentativa interrompida a recuperar"]
        write_claim(claim_path, {"attempts": [attempt], "published_sha256": None})
        return [attempt], []
    try:
        claim = read_json(claim_path)
    except (OSError, ValueError) as exc:
        return None, [f"{claim_path} ilegível ({exc}); inspecione manualmente"]
    attempts = claim.get("attempts") if isinstance(claim, dict) else None
    if not isinstance(claim, dict) or not isinstance(attempts, list) or not attempts:
        return None, [f"{claim_path}: reserva malformada; inspecione manualmente"]
    if claim.get("published_sha256") is not None:
        return None, [
            f"{claim_path}: test já publicado nesta seed e {artifact} ausente; nada é refeito; "
            "inspecione manualmente"
        ]
    if claim.get("integrity_failure") is not None:
        return None, [
            f"{claim_path}: test calculado e recusado por integridade "
            f"({claim['integrity_failure']}); não é falha técnica e nada é refeito"
        ]
    if len(attempts) >= MAX_ATTEMPTS:
        return None, [
            f"{claim_path}: {len(attempts)} tentativas registradas; a repetição única por "
            "falha técnica já foi usada; inspecione manualmente"
        ]
    if recovery_reason is None or not recovery_reason.strip():
        return None, [
            f"{claim_path}: tentativa de test interrompida antes da publicação; só uma "
            "falha técnica permite refazê-la, com --recovery-reason"
        ]
    attempts = [*cast(list[dict[str, object]], attempts), attempt]
    write_claim(claim_path, {"attempts": attempts, "published_sha256": None})
    return attempts, []


def published_bytes(record: Mapping[str, object]) -> bytes:
    """O conteúdo exato que `publish_once` grava."""
    return (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def directory_files(directory: Path) -> dict[str, str]:
    """SHA-256 de cada arquivo de `directory`, fora os de lock entre processos."""
    if not directory.is_dir():
        return {}
    return {
        name: digest for name, digest in run_artifacts(directory).items() if not name.endswith(".lock")
    }


def snapshot(run_dir: Path, control: ControlRun, inputs: TreatmentInputs) -> dict[str, dict[str, str]]:
    """SHA-256 de cada arquivo do run, do controle e da preparação."""
    return {
        "run": directory_files(run_dir),
        "control": directory_files(control.run_dir),
        "prepared": directory_files(inputs.prepared_dir),
    }


def result_record(
    inputs: TreatmentInputs,
    seed: int,
    run: CheckedRun,
    decision: Mapping[str, object],
    decision_sha256: str,
    files: Mapping[str, Mapping[str, str]],
    attempts: list[dict[str, object]],
    treatment: Mapping[str, dict],
    control: Mapping[str, dict],
) -> dict[str, object]:
    head, status = inputs.checkout
    return {
        "descriptive_only": True,
        "split": "test",
        "evaluated_splits": list(TEST_SPLITS),
        "treatment": inputs.treatment,
        "seed": seed,
        "decision": {
            "path": (DEFAULT_RUN_ROOT / DECISION_FILE).as_posix(),
            "sha256": decision_sha256,
            "decision": decision["decision"],
            "recorded_at": decision["recorded_at"],
            "code": decision["code"],
        },
        "checkpoint": {
            "path": (run.run_dir / "checkpoint.pt").as_posix(),
            "sha256": files["run"]["checkpoint.pt"],
        },
        "run": {"run_dir": run.run_dir.as_posix(), "files_sha256": dict(files["run"])},
        "prepared": {
            "dir": inputs.prepared_dir.as_posix(),
            "files_sha256": dict(files["prepared"]),
        },
        "control": {
            "run_dir": run.control_run_dir.as_posix(),
            "files_sha256": dict(files["control"]),
        },
        "split_provenance": {
            "frames_path": inputs.le2i.frames_path.as_posix(),
            "frames_sha256": sha256_file(inputs.le2i.frames_path),
            "test_index_sha256": inputs.gate.eval_index_sha256["test"],
        },
        "alarm_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
        "metrics": {"treatment": dict(treatment), "control": dict(control)},
        "delta": descriptive_delta(treatment, control),
        "attempts": attempts,
        "code": {"commit": head, "dirty": bool(status)},
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def print_summary(record: Mapping[str, object]) -> None:
    metrics = cast(dict[str, dict[str, dict[str, object]]], record["metrics"])
    delta = cast(dict[str, object], record["delta"])
    events_delta = cast(dict[str, object], delta["events"])
    fall = BASELINE_A_ALARM_PROTOCOL.fall_label
    print(f"\n=== test descritivo de {record['treatment']}, seed {record['seed']} ===")
    for side in ("treatment", "control"):
        classification = metrics[side]["classification"]
        events = metrics[side]["events"]
        f1_by_class = cast(dict[str, float], classification["f1_by_class"])
        print(
            f"{side}: macro_f1_restricted={classification['macro_f1_restricted']:.4f}, "
            f"F1 fall={f1_by_class[str(fall)]:.4f}, n_missed_events={events['n_missed_events']}, "
            f"n_false_alarms={events['n_false_alarms']}"
        )
    print(
        f"Δ: macro_f1_restricted={delta['macro_f1_restricted']:+.4f}, "
        f"n_missed_events={events_delta['n_missed_events']:+}, "
        f"n_false_alarms={events_delta['n_false_alarms']:+}"
    )
    decision = cast(dict[str, object], record["decision"])
    print(f"descritivo: a decisão registrada ({decision['decision']}) não muda")


def run_test_description(
    inputs: TreatmentInputs,
    seed: int,
    recovery_reason: str | None = None,
    decision_sha256: str = PINNED_DECISION_SHA256,
) -> bool:
    """O `describe-test` do CLI: decisão íntegra, portões do run e controle
    pareado conferidos antes de qualquer acesso a `test`; uma única avaliação
    por tratamento e seed, publicada sem nunca substituir nada."""
    command = "describe-test"
    control, problems = paired_control(inputs.treatment, seed)
    if control is None:
        return refuse(command, problems)
    problems = execution_root_problems(inputs, seed, control)
    problems.extend(output_isolation_problems(inputs, seed, [control.run_dir]))
    if problems:
        return refuse(command, problems)
    directory = result_dir(inputs.treatment, seed)
    claim_path = directory / TEST_CLAIM_FILE
    if recovery_reason is not None and not claim_path.exists():
        return refuse(command, [f"{claim_path} ausente: não há tentativa interrompida a recuperar"])
    with decision_lock(DEFAULT_RUN_ROOT, exclusive=False):
        decision, problems = decision_record(inputs, decision_sha256)
        if decision is None:
            return refuse(command, problems)
        control, problems = paired_control(inputs.treatment, seed, decision)
        if control is None:
            return refuse(command, problems)
        files = snapshot(canonical_run_dir(inputs.treatment, seed), control, inputs)
        check = check_run(inputs, seed, control)
        if not report(f"test descritivo de {inputs.treatment}, seed {seed}", check.results):
            return False
        run = cast(CheckedRun, check.run)
        _, problems = val_report(run)
        recorded = cast(dict[str, dict[str, dict[str, str]]], decision["inputs"])
        controls = [
            Path(entry["control_run_dir"])
            for by_seed in recorded.values()
            for entry in by_seed.values()
        ]
        problems.extend(output_isolation_problems(inputs, seed, [control.run_dir, *controls]))
        if snapshot(run.run_dir, control, inputs) != files:
            problems.append("run, controle ou preparação mudaram durante a conferência")
        if problems:
            return refuse(command, problems)
        existed = directory.exists()
        with exclusive_lock(directory):
            attempts, problems = claim_attempts(
                directory, inputs.checkout, recovery_reason, existed
            )
            if attempts is None:
                return refuse(command, problems)
            try:
                control_metrics = control_test(run.control_run_dir)
                treatment_metrics = split_metrics(
                    run.config,
                    run.run_dir / "checkpoint.pt",
                    run.stats,
                    inputs.le2i.frames,
                    treatment_reader(inputs),
                    inputs.gate.eval_index_sha256["test"],
                )
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                return refuse(command, [str(exc)])
            problems = structure_problems(treatment_metrics, control_metrics)
            if sha256_file(DEFAULT_RUN_ROOT / DECISION_FILE) != decision_sha256:
                problems.append(f"{DEFAULT_RUN_ROOT / DECISION_FILE} mudou durante a avaliação")
            problems.extend(changed_inputs(decision))
            if snapshot(run.run_dir, control, inputs) != files:
                problems.append("run, controle ou preparação mudaram durante a avaliação")
            if problems:
                write_claim(
                    claim_path,
                    {
                        "attempts": attempts,
                        "published_sha256": None,
                        "integrity_failure": "; ".join(problems),
                    },
                )
                return refuse(command, problems)
            record = result_record(
                inputs,
                seed,
                run,
                decision,
                decision_sha256,
                files,
                attempts,
                treatment_metrics,
                control_metrics,
            )
            artifact = directory / TEST_METRICS_FILE
            expected_sha256 = hashlib.sha256(published_bytes(record)).hexdigest()
            write_claim(claim_path, {"attempts": attempts, "published_sha256": expected_sha256})
            if not publish_once(artifact, record) or sha256_file(artifact) != expected_sha256:
                return refuse(command, [f"{artifact} já existe ou difere do reservado; inspecione"])
    print_summary(record)
    print(f"\n{artifact}: test descritivo publicado uma vez; nada da decisão foi alterado")
    return True
