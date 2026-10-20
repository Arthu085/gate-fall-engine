"""Ciclo de vida dos runs de T e S: um único run por seed.

Um run completo é o run do braço A válido sem `test` mais o `treatment.json`
conferido contra a preparação, a seed e o controle pareado; ele só é pulado,
nunca refeito, nem com `--force`. Um run incompleto só é refeito depois de
falha técnica, com motivo registrado e a mesma seed, tratamento e
configuração. Um run treinado com registro divergente é recusado sempre.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from gatefall.data.gmdcsa24.pose_extract import read_json, section
from gatefall.data.gmdcsa24.weak_label import compare_expected
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.artifacts import validate_training_run
from gatefall.train.baseline_a.config import TrainConfig, load_config
from gatefall.train.gmdcsa24_weak_label.gates import (
    PREPARED_MANIFEST_FILE,
    TREATMENT_RECORD_FILE,
    VALIDATION_SPLITS,
    ControlRun,
    TreatmentInputs,
    code_problems,
)

RUN_ABSENT = "ausente"
RUN_COMPLETE = "completo"
RUN_INCOMPLETE = "incompleto"
RUN_DIVERGENT = "registro divergente"

RECOVERY_FIELDS = frozenset({"reason", "recorded_at", "replaced"})


@dataclass(frozen=True)
class RunState:
    status: str
    problems: list[str]


def treatment_record(
    inputs: TreatmentInputs, seed: int, control: ControlRun, control_checkpoint: str | None
) -> dict[str, object]:
    """Campos do `treatment.json` fixados pela preparação, pela seed e pelo controle."""
    manifest_path = inputs.prepared_dir / PREPARED_MANIFEST_FILE
    manifest = read_json(manifest_path)
    return {
        "treatment": inputs.treatment,
        "seed": seed,
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "standardization_stats_sha256": section(manifest, "standardization").get("sha256"),
        "train_index_sha256": section(manifest, "windows").get("train_index_sha256"),
        "class_weights": manifest.get("class_weights"),
        "eval_index_sha256": {"val": section(manifest, "eval_index_sha256").get("val")},
        "evaluated_splits": list(VALIDATION_SPLITS),
        "control": {
            "run_dir": control.run_dir.as_posix(),
            "checkpoint_sha256": control_checkpoint,
            "acceptance": (
                "retreinado nesta máquina" if control.retrained else "checkpoint_sha256 publicado"
            ),
        },
    }


def _read_record(path: Path) -> dict[str, object] | str:
    try:
        record = read_json(path)
    except (OSError, ValueError) as exc:
        return f"{path} ilegível ({exc})"
    return record if isinstance(record, dict) else f"{path} não é um objeto JSON"


def _recorded_at_problems(value: object, label: str) -> list[str]:
    try:
        datetime.fromisoformat(cast(str, value))
    except (TypeError, ValueError):
        return [f"{label}: recorded_at inválido {value!r}"]
    return []


def recovery_problems(recovery: object) -> list[str]:
    label = "registro do tratamento: recovery"
    if not isinstance(recovery, dict) or set(recovery) != RECOVERY_FIELDS:
        return [f"{label} deve ter exatamente {sorted(RECOVERY_FIELDS)}"]
    problems = _recorded_at_problems(recovery["recorded_at"], label)
    reason = recovery["reason"]
    if not isinstance(reason, str) or not reason.strip():
        problems.append(f"{label}: motivo da recuperação ausente")
    if not isinstance(recovery["replaced"], dict):
        problems.append(f"{label}: replaced deve registrar o run substituído")
    return problems


def record_problems(
    inputs: TreatmentInputs, run_dir: Path, expected: dict[str, object]
) -> list[str]:
    """`treatment.json` igual ao recalculado, ligado ao checkpoint do run e a um
    commit limpo entre a preparação e o HEAD atual."""
    record = _read_record(run_dir / TREATMENT_RECORD_FILE)
    if isinstance(record, str):
        return [record]
    observed = {key: value for key, value in record.items() if key not in ("code", "recovery")}
    recorded_control = observed.get("control")
    expected_control = cast(dict[str, object], expected["control"])
    if isinstance(recorded_control, dict) and isinstance(recorded_control.get("run_dir"), str):
        recorded_dir = Path(cast(str, recorded_control["run_dir"]))
        if recorded_dir.resolve() == Path(cast(str, expected_control["run_dir"])).resolve():
            observed["control"] = {**recorded_control, "run_dir": expected_control["run_dir"]}
    problems = compare_expected(
        {**expected, "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt")},
        observed,
        "registro do tratamento: ",
    )
    prepared = section(read_json(inputs.prepared_dir / PREPARED_MANIFEST_FILE), "code")
    problems.extend(
        code_problems(
            inputs,
            record.get("code"),
            "registro do tratamento",
            after=cast(str, prepared.get("commit")),
        )
    )
    if "recovery" in record:
        problems.extend(recovery_problems(record["recovery"]))
    return problems


def run_state(
    inputs: TreatmentInputs, run_dir: Path, config: TrainConfig, expected: dict[str, object]
) -> RunState:
    if not run_dir.exists():
        return RunState(RUN_ABSENT, [])
    try:
        validate_training_run(run_dir, expected_config=config, splits=VALIDATION_SPLITS)
    except RuntimeError as exc:
        return RunState(RUN_INCOMPLETE, [str(exc)])
    if not (run_dir / TREATMENT_RECORD_FILE).is_file():
        return RunState(RUN_INCOMPLETE, [f"{run_dir / TREATMENT_RECORD_FILE} ausente"])
    problems = record_problems(inputs, run_dir, expected)
    return RunState(RUN_DIVERGENT if problems else RUN_COMPLETE, problems)


def identity_problems(
    inputs: TreatmentInputs, seed: int, run_dir: Path, config: TrainConfig
) -> list[str]:
    """A recuperação só refaz o run da mesma seed, tratamento e configuração."""
    problems: list[str] = []
    try:
        existing = load_config(run_dir / "config.yaml")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        problems.append(
            f"{run_dir / 'config.yaml'} ausente ou ilegível ({exc}); "
            "a identidade do run não pode ser conferida"
        )
    else:
        problems.extend(
            compare_expected(config.to_dict(), existing.to_dict(), "config.yaml do run: ")
        )
    record_path = run_dir / TREATMENT_RECORD_FILE
    if record_path.exists():
        record = _read_record(record_path)
        if isinstance(record, str):
            problems.append(record)
        else:
            problems.extend(
                compare_expected(
                    {"treatment": inputs.treatment, "seed": seed},
                    {key: record.get(key) for key in ("treatment", "seed")},
                    "registro do tratamento: ",
                )
            )
    return problems


def run_artifacts(run_dir: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(run_dir.iterdir()) if path.is_file()}
