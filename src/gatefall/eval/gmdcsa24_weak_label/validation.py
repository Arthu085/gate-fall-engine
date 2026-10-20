"""Avaliação de eventos só em `val` dos runs de T e S do piloto GMDCSA24 (braço A).

Reaproveita do avaliador canônico do braço A o carregamento do checkpoint, a
predição por janela, a associação de eventos, a orquestração com lock e
journal e `BASELINE_A_ALARM_PROTOCOL`. Muda só o que o piloto exige: a `val`
do Le2i-CS é padronizada pelo JSON do tratamento, a grade entregue à
orquestração tem só as linhas de `val` e o avaliador recusa qualquer outro
split. Nenhuma feature, predição, métrica ou artefato de `test` é montado,
lido ou gravado. O relatório fica em `val_event_metrics.json` no run canônico,
ao lado de `alarm_protocol.yaml`; os validadores canônicos, que exigem `val` e
`test` em `event_metrics.json`, não o consomem.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import torch
import yaml

from gatefall.config import EVAL_STRIDE
from gatefall.data.gmdcsa24.pose_extract import read_json, section
from gatefall.data.gmdcsa24.weak_label import compare_expected
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis.multiseed_summary import validate_run_event_artifacts
from gatefall.eval.baseline_a.cli import _load_model, _predict_with_identity
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, load_alarm_protocol
from gatefall.eval.shared.event_artifacts import EVENT_TRANSACTION_FILE, validate_event_metrics
from gatefall.eval.shared.orchestration import EventEvaluation, Predictions, SplitEvaluator
from gatefall.features.standardization import StandardizationStats
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.config import TrainConfig, load_config
from gatefall.train.gmdcsa24_weak_label.assembly import aligned_loader
from gatefall.train.gmdcsa24_weak_label.gates import (
    CONDITIONS,
    TREATMENT_RECORD_FILE,
    ControlRun,
    TreatmentInputs,
    assess_training,
    canonical_run_dir,
)
from gatefall.train.gmdcsa24_weak_label.lifecycle import RUN_COMPLETE, run_state, treatment_record

VAL_SPLITS: tuple[str, ...] = ("val",)
VAL_EVENT_METRICS_FILE = "val_event_metrics.json"
ALARM_PROTOCOL_FILE = "alarm_protocol.yaml"
CANONICAL_EVENT_METRICS_FILE = "event_metrics.json"
CONTROL_RETRAINED = "retreinado nesta máquina"

# Campos de um split de eventos que dependem só da grade de `val`, não do
# modelo: iguais no tratamento e no controle da mesma `val` do Le2i-CS.
EVENT_STRUCTURE_FIELDS = (
    "usable_windows",
    "total_windows",
    "labeled_windows",
    "total_video_time_hours",
    "labeled_time_hours",
    "n_fall_events",
)

RUN_CONDITION = "run"
CONTROL_EVENTS_CONDITION = "eventos do controle"
RUN_CONDITIONS = (*CONDITIONS, RUN_CONDITION, CONTROL_EVENTS_CONDITION)


@dataclass(frozen=True)
class CheckedRun:
    """Run de T ou S completo, com portões, registro e controle pareado conferidos."""

    run_dir: Path
    config: TrainConfig
    stats: StandardizationStats
    provenance: dict[str, object]
    control_run_dir: Path
    control_val: dict[str, object]


@dataclass(frozen=True)
class RunCheck:
    results: dict[str, list[str]]
    run: CheckedRun | None

    @property
    def passed(self) -> bool:
        return self.run is not None and all(not problems for problems in self.results.values())


def control_from_record(record: Mapping[str, object]) -> ControlRun | None:
    """O controle pareado registrado no `treatment.json`; None se o bloco é ilegível."""
    control = record.get("control")
    if not isinstance(control, dict) or not isinstance(control.get("run_dir"), str):
        return None
    return ControlRun(Path(control["run_dir"]), control.get("acceptance") == CONTROL_RETRAINED)


def control_val_events(control: ControlRun) -> dict[str, object]:
    """`val` do `event_metrics.json` canônico do controle, validado por inteiro
    pelo validador do sumário multi-seed (protocolo e hashes do run)."""
    config = load_config(control.run_dir / "config.yaml")
    report = validate_run_event_artifacts(control.run_dir, config)
    return cast(dict[str, object], report["splits"]["val"])


def structure_problems(treatment_val: Mapping[str, object], control_val: Mapping[str, object]) -> list[str]:
    return compare_expected(
        {field: control_val.get(field) for field in EVENT_STRUCTURE_FIELDS},
        {field: treatment_val.get(field) for field in EVENT_STRUCTURE_FIELDS},
        "val do tratamento × val do controle: ",
    )


def report_provenance(
    inputs: TreatmentInputs, seed: int, run_dir: Path, expected: dict[str, object]
) -> dict[str, object]:
    """Campos que ligam o relatório de `val` ao run, ao tratamento e ao controle."""
    control = cast(dict[str, object], expected["control"])
    return {
        "evaluated_splits": list(VAL_SPLITS),
        "treatment": {
            "treatment": inputs.treatment,
            "seed": seed,
            "treatment_record_sha256": sha256_file(run_dir / TREATMENT_RECORD_FILE),
            "prepared_manifest_sha256": expected["prepared_manifest_sha256"],
            "control_checkpoint_sha256": control["checkpoint_sha256"],
        },
    }


def check_run(inputs: TreatmentInputs, seed: int, control: ControlRun) -> RunCheck:
    """Portões de pré-treino, run completo com `treatment.json` conferido, nenhum
    relatório canônico de `val` e `test` no run e `val` canônica do controle."""
    assessment = assess_training(inputs, seed, control)
    results = {condition: list(assessment.results.get(condition, [])) for condition in CONDITIONS}
    results[RUN_CONDITION] = []
    results[CONTROL_EVENTS_CONDITION] = []
    if not assessment.passed or assessment.config is None or assessment.stats is None:
        results[RUN_CONDITION].append("não verificado: portões de pré-treino reprovados")
        return RunCheck(results, None)
    run_dir = canonical_run_dir(inputs.treatment, seed)
    expected = treatment_record(inputs, seed, control, assessment.control_checkpoint_sha256)
    state = run_state(inputs, run_dir, assessment.config, expected)
    if state.status != RUN_COMPLETE:
        results[RUN_CONDITION].extend([f"{run_dir}: run {state.status}", *state.problems])
    if (run_dir / CANONICAL_EVENT_METRICS_FILE).exists():
        results[RUN_CONDITION].append(
            f"{run_dir / CANONICAL_EVENT_METRICS_FILE} existe: o run do piloto só admite "
            f"{VAL_EVENT_METRICS_FILE}, sem test"
        )
    try:
        control_val = control_val_events(control)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        results[CONTROL_EVENTS_CONDITION].append(f"{control.run_dir}: {exc}")
        control_val = {}
    if any(results.values()):
        return RunCheck(results, None)
    provenance = report_provenance(inputs, seed, run_dir, expected)
    run = CheckedRun(
        run_dir, assessment.config, assessment.stats, provenance, control.run_dir, control_val
    )
    return RunCheck(results, run)


def val_frames(frames: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, frames[frames["split"].isin(VAL_SPLITS)])


def val_source(inputs: TreatmentInputs, frames: pd.DataFrame) -> PoseWindowDataset:
    """`val` do Le2i-CS como no avaliador canônico (`EVAL_STRIDE`, sem descartar
    `IGNORE_LABEL`), com as features lidas pelo leitor do tratamento."""
    windows = build_window_index(frames, stride=EVAL_STRIDE, drop_ignored=False)
    return PoseWindowDataset(
        frames,
        "val",
        EVAL_STRIDE,
        aligned_loader(windows, inputs.roots, inputs.load_features),
        drop_ignored=False,
    )


def val_event_evaluation(inputs: TreatmentInputs, run: CheckedRun) -> EventEvaluation:
    def prepare() -> tuple[pd.DataFrame, SplitEvaluator]:
        frames = val_frames(inputs.le2i.frames)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _load_model(run.config, run.run_dir / "checkpoint.pt", device)
        source = val_source(inputs, frames)

        def evaluate_split(split: str) -> tuple[int, Predictions]:
            if split not in VAL_SPLITS:
                raise RuntimeError(
                    f"split {split!r} recusado: o piloto avalia só val antes da decisão"
                )
            predictions = _predict_with_identity(
                model, source, run.stats, device, batch_size=run.config.batch_size
            )
            return len(source), predictions

        return frames, evaluate_split

    def check_splits(splits: dict[str, dict]) -> None:
        problems = structure_problems(splits["val"], run.control_val)
        if problems:
            raise RuntimeError("; ".join(problems))

    return EventEvaluation(
        run.config,
        prepare,
        splits=VAL_SPLITS,
        metrics_file=VAL_EVENT_METRICS_FILE,
        provenance=run.provenance,
        check_splits=check_splits,
    )


def val_report(run: CheckedRun) -> tuple[dict[str, object] | None, list[str]]:
    """O `val` publicado do run: protocolo de alarme congelado, só `val`, hashes
    do run, proveniência do tratamento e estrutura igual à `val` do controle, sem
    publicação interrompida pendente."""
    protocol_path = run.run_dir / ALARM_PROTOCOL_FILE
    metrics_path = run.run_dir / VAL_EVENT_METRICS_FILE
    journal = run.run_dir / EVENT_TRANSACTION_FILE
    if journal.exists():
        return None, [f"{journal}: publicação interrompida pendente de recuperação"]
    try:
        if load_alarm_protocol(protocol_path) != BASELINE_A_ALARM_PROTOCOL:
            return None, [f"{protocol_path} diverge de BASELINE_A_ALARM_PROTOCOL"]
        data = read_json(metrics_path)
        validate_event_metrics(
            data,
            run.config,
            run.run_dir / "checkpoint.pt",
            protocol_path,
            training_metrics_path=run.run_dir / "metrics.json",
            require_hashes=True,
            splits=VAL_SPLITS,
            provenance=run.provenance,
        )
    except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
        return None, [f"{metrics_path}: {exc}"]
    split = section(section(data, "splits"), "val")
    problems = structure_problems(split, run.control_val)
    return (None if problems else split), problems


def training_val(run_dir: Path) -> dict[str, object]:
    """`final.val` do `metrics.json` de um run já validado."""
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return cast(dict[str, object], metrics["final"]["val"])
