"""Orquestração compartilhada da avaliação de eventos por arma."""

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.config import EVAL_STRIDE
from gatefall.data.windowing import build_window_index
from gatefall.eval.shared.alarm_protocol import (
    BASELINE_A_ALARM_PROTOCOL,
    load_alarm_protocol,
    save_alarm_protocol,
)
from gatefall.eval.shared.event_artifacts import (
    EVENT_SPLITS,
    EventEvaluationLock,
    EventRunConfig,
    _promote_event_outputs,
    _recover_event_publication,
    _require_event_lock,
    validate_event_metrics,
)
from gatefall.eval.shared.events import extract_label_segments, split_event_report
from gatefall.hashing import sha256_file

Predictions = tuple[list[str], list[int], list[int], list[int]]
SplitEvaluator = Callable[[str], tuple[int, Predictions]]


@dataclass(frozen=True)
class EventEvaluation:
    """Avaliação de eventos de um run. Os padrões são os das armas canônicas
    (`val` e `test` em `event_metrics.json`); um caminho experimental restringe
    os splits, troca o arquivo, acrescenta campos de proveniência ao relatório e
    confere os splits calculados antes da publicação."""

    config: EventRunConfig
    prepare: Callable[[], tuple[pd.DataFrame, SplitEvaluator]]
    splits: tuple[str, ...] = EVENT_SPLITS
    metrics_file: str = "event_metrics.json"
    provenance: Mapping[str, object] = field(default_factory=dict)
    check_splits: Callable[[dict[str, dict]], None] | None = None


def _existing_pair_valid(
    force: bool,
    arm: str,
    run_dir: Path,
    evaluation: EventEvaluation,
    checkpoint_path: Path,
    alarm_protocol_path: Path,
    event_metrics_path: Path,
) -> tuple[bool, bool]:
    event_outputs = (alarm_protocol_path, event_metrics_path)
    present_outputs = [path for path in event_outputs if path.is_file()]
    previous_pair_valid = False
    if len(present_outputs) == len(event_outputs):
        try:
            protocol = load_alarm_protocol(alarm_protocol_path)
            if protocol != BASELINE_A_ALARM_PROTOCOL:
                if arm == "A":
                    raise ValueError("alarm_protocol.yaml incompatível com o braço A")
                raise ValueError(f"alarm_protocol.yaml incompatível com a arma {arm}")
            with event_metrics_path.open(encoding="utf-8") as stream:
                existing_report = json.load(stream)
            validate_event_metrics(
                existing_report,
                evaluation.config,
                checkpoint_path,
                alarm_protocol_path,
                training_metrics_path=run_dir / "metrics.json",
                require_hashes=True,
                splits=evaluation.splits,
                provenance=evaluation.provenance,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            if not force:
                raise RuntimeError(
                    f"avaliação de eventos inconsistente ({exc}); use --force para reconstruir"
                ) from exc
        else:
            previous_pair_valid = True
            if not force:
                print(f"skip {event_metrics_path} (avaliação completa e íntegra)")
                return previous_pair_valid, True
    elif present_outputs and not force:
        missing_outputs = [str(path) for path in event_outputs if not path.is_file()]
        raise RuntimeError(
            "avaliação de eventos parcial; artefatos ausentes: "
            + ", ".join(missing_outputs)
            + "; use --force para reconstruir"
        )
    return previous_pair_valid, False


def _n_fall_segments_in_annotation(frames: pd.DataFrame, split: str) -> int:
    split_frames = cast(
        pd.DataFrame, frames[frames["split"] == split]
    ).sort_values(["video_id", "frame_index"])
    total_segments = 0
    for _video_id, group in split_frames.groupby("video_id", sort=False):
        total_segments += len(
            extract_label_segments(
                group["frame_index"].to_numpy(),
                group["label"].to_numpy(),
                BASELINE_A_ALARM_PROTOCOL.fall_label,
            )
        )
    return total_segments


def _split_reports(
    frames: pd.DataFrame,
    evaluate_split: SplitEvaluator,
    split_names: tuple[str, ...] = EVENT_SPLITS,
) -> dict[str, dict]:
    splits: dict[str, dict] = {}
    for split in split_names:
        usable_windows, predictions = evaluate_split(split)
        split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
        total_windows = len(
            build_window_index(split_frames, stride=EVAL_STRIDE, drop_ignored=False)
        )
        if usable_windows != total_windows:
            raise RuntimeError(
                f"split={split!r}: usable_windows ({usable_windows}) != "
                f"total_windows ({total_windows}) apesar de drop_ignored=False"
            )
        labeled_windows = len(
            build_window_index(split_frames, stride=EVAL_STRIDE, drop_ignored=True)
        )
        split_report = split_event_report(
            *predictions,
            BASELINE_A_ALARM_PROTOCOL,
            usable_windows,
            total_windows,
            labeled_windows,
        )
        annotated_fall_segments = _n_fall_segments_in_annotation(frames, split)
        if split_report["n_fall_events"] != annotated_fall_segments:
            raise ValueError(
                f"split={split!r}: n_fall_events extraído das janelas usáveis "
                f"({split_report['n_fall_events']}) diverge da contagem de "
                "segmentos fall na anotação bruta "
                f"({annotated_fall_segments}) — possível janela IGNORE_LABEL "
                "descartada dentro de um run fall, dividindo um evento real em dois"
            )
        splits[split] = split_report
    return splits


def _publish_report(
    run_dir: Path,
    lock: EventEvaluationLock,
    evaluation: EventEvaluation,
    splits: dict[str, dict],
    previous_pair_valid: bool,
    checkpoint_path: Path,
    alarm_protocol_path: Path,
    event_metrics_path: Path,
) -> None:
    config = evaluation.config
    report = {
        "run_name": config.run_name,
        "checkpoint_path": str(checkpoint_path),
        "alarm_protocol_path": str(alarm_protocol_path),
        "splits": splits,
    }
    report.update(evaluation.provenance)
    run_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    protocol_tmp = run_dir / f".alarm_protocol.pending-{token}.yaml"
    metrics_tmp = run_dir / f".{Path(event_metrics_path.name).stem}.pending-{token}.json"
    save_alarm_protocol(BASELINE_A_ALARM_PROTOCOL, protocol_tmp, force=True)
    report["checkpoint_sha256"] = sha256_file(checkpoint_path)
    report["training_metrics_sha256"] = sha256_file(run_dir / "metrics.json")
    report["alarm_protocol_sha256"] = sha256_file(protocol_tmp)
    with metrics_tmp.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    with metrics_tmp.open(encoding="utf-8") as stream:
        staged_report = json.load(stream)
    if load_alarm_protocol(protocol_tmp) != BASELINE_A_ALARM_PROTOCOL:
        raise RuntimeError("staging de alarm_protocol.yaml divergiu do protocolo")
    validate_event_metrics(
        staged_report,
        config,
        checkpoint_path,
        alarm_protocol_path,
        training_metrics_path=run_dir / "metrics.json",
        protocol_file_path=protocol_tmp,
        require_hashes=True,
        splits=evaluation.splits,
        provenance=evaluation.provenance,
    )
    _promote_event_outputs(
        protocol_tmp,
        metrics_tmp,
        alarm_protocol_path,
        event_metrics_path,
        lock,
        preserve_previous=previous_pair_valid,
    )
    print(f"{event_metrics_path}: métricas de eventos gravadas (run_name={config.run_name})")


def run_event_evaluation(
    force: bool,
    arm: str,
    run_dir: Path,
    lock: EventEvaluationLock,
    load_run: Callable[[], EventEvaluation],
) -> None:
    _require_event_lock(run_dir, lock)
    recovery = _recover_event_publication(run_dir, lock)
    if recovery is not None:
        print(f"recovery de avaliação concluído: {recovery}")
    evaluation = load_run()
    checkpoint_path = run_dir / "checkpoint.pt"
    alarm_protocol_path = run_dir / "alarm_protocol.yaml"
    event_metrics_path = run_dir / evaluation.metrics_file
    previous_pair_valid, skip = _existing_pair_valid(
        force,
        arm,
        run_dir,
        evaluation,
        checkpoint_path,
        alarm_protocol_path,
        event_metrics_path,
    )
    if skip:
        return
    frames, evaluate_split = evaluation.prepare()
    splits = _split_reports(frames, evaluate_split, evaluation.splits)
    if evaluation.check_splits is not None:
        evaluation.check_splits(splits)
    _publish_report(
        run_dir,
        lock,
        evaluation,
        splits,
        previous_pair_valid,
        checkpoint_path,
        alarm_protocol_path,
        event_metrics_path,
    )
