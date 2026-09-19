"""Avaliação de eventos da arma A: protocolo de alarme sobre o checkpoint treinado."""

import argparse
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import build_window_index
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.eval.alarm_protocol import (
    BASELINE_A_ALARM_PROTOCOL,
    load_alarm_protocol,
    save_alarm_protocol,
)
from gatefall.eval.events import extract_label_segments, split_event_report
from gatefall.eval.events_selftest import run_events_selftest
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization,
    load_stats,
    validate_stats_layout,
)
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import default_run_dir, validate_local_run_dir
from gatefall.hashing import sha256_file
from gatefall.train.artifacts import load_compatible_checkpoint, validate_training_run
from gatefall.train.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.tcn import TCNClassifier

from gatefall.eval.event_artifacts import (
    EVENT_COUNT_FIELDS,
    EVENT_RATE_FIELDS,
    EVENT_SPLIT_FIELDS,
    EventEvaluationLock,
    _promote_event_outputs,
    _recover_event_publication,
    _require_event_lock,
    validate_event_metrics,
)


def _load_model(
    config: TrainConfig, checkpoint_path: Path, device: str
) -> TCNClassifier:
    model = load_compatible_checkpoint(checkpoint_path, config).to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict_with_identity(
    model: TCNClassifier,
    source: PoseWindowDataset,
    stats: StandardizationStats,
    device: str,
    batch_size: int,
) -> tuple[list[str], list[int], list[int], list[int]]:
    video_ids: list[str] = []
    k_ends: list[int] = []
    true_labels: list[int] = []
    pred_labels: list[int] = []

    batch_windows: list[np.ndarray] = []
    batch_labels: list[int] = []
    batch_identity: list[tuple[str, int]] = []

    def flush() -> None:
        if not batch_windows:
            return
        stacked = np.stack(batch_windows, axis=0)
        standardized = apply_standardization(stacked, stats)
        x = torch.from_numpy(standardized).to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        for (video_id, k_end), label, pred in zip(batch_identity, batch_labels, preds):
            video_ids.append(video_id)
            k_ends.append(k_end)
            true_labels.append(label)
            pred_labels.append(int(pred))

        batch_windows.clear()
        batch_labels.clear()
        batch_identity.clear()

    for i in range(len(source)):
        window, label, (video_id, k_end) = source[i]
        batch_windows.append(window)
        batch_labels.append(label)
        batch_identity.append((video_id, k_end))
        if len(batch_windows) == batch_size:
            flush()
    flush()

    return video_ids, k_ends, true_labels, pred_labels


def _n_fall_segments_in_annotation(frames: pd.DataFrame, split: str) -> int:
    split_frames = cast(
        pd.DataFrame, frames[frames["split"] == split]
    ).sort_values(["video_id", "frame_index"])
    total_segments = 0
    for _video_id, group in split_frames.groupby("video_id", sort=False):
        frame_indices = group["frame_index"].to_numpy()
        labels = group["label"].to_numpy()
        total_segments += len(
            extract_label_segments(frame_indices, labels, BASELINE_A_ALARM_PROTOCOL.fall_label)
        )
    return total_segments


def _run_evaluate_locked(
    force: bool,
    dataset_name: str,
    run_dir: Path,
    lock: EventEvaluationLock,
) -> None:
    _require_event_lock(run_dir, lock)
    adapter = get_dataset(dataset_name)
    checkpoint_path = run_dir / "checkpoint.pt"
    alarm_protocol_path = run_dir / "alarm_protocol.yaml"
    event_metrics_path = run_dir / "event_metrics.json"
    recovery = _recover_event_publication(run_dir, lock)
    if recovery is not None:
        print(f"recovery de avaliação concluído: {recovery}")
    expected_config = replace(
        BASELINE_A_CONFIG,
        standardization_stats_path=str(adapter.pose_stats_path),
        standardization_stats_sha256=sha256_file(adapter.pose_stats_path),
    )
    try:
        config = validate_training_run(
            run_dir,
            expected_config=expected_config,
            fields_allowed_to_differ=frozenset({"seed"}),
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"run de treino inválido em {run_dir}: {exc}"
        ) from exc
    if config.eval_stride != EVAL_STRIDE:
        raise ValueError(
            f"config.eval_stride ({config.eval_stride}) diverge de "
            f"EVAL_STRIDE ({EVAL_STRIDE})"
        )
    event_outputs = (alarm_protocol_path, event_metrics_path)
    present_outputs = [path for path in event_outputs if path.is_file()]
    previous_pair_valid = False
    if len(present_outputs) == len(event_outputs):
        try:
            protocol = load_alarm_protocol(alarm_protocol_path)
            if protocol != BASELINE_A_ALARM_PROTOCOL:
                raise ValueError("alarm_protocol.yaml incompatível com o braço A")
            with event_metrics_path.open(encoding="utf-8") as stream:
                existing_report = json.load(stream)
            validate_event_metrics(
                existing_report,
                config,
                checkpoint_path,
                alarm_protocol_path,
                training_metrics_path=run_dir / "metrics.json",
                require_hashes=True,
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
                return
    elif present_outputs and not force:
        missing_outputs = [str(path) for path in event_outputs if not path.is_file()]
        raise RuntimeError(
            "avaliação de eventos parcial; artefatos ausentes: "
            + ", ".join(missing_outputs)
            + "; use --force para reconstruir"
        )
    stats = load_stats(adapter.pose_stats_path)
    validate_stats_layout(stats)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(config, checkpoint_path, device)

    frames = adapter.load_frames()

    splits: dict[str, dict] = {}
    for split in ("val", "test"):
        source = PoseWindowDataset(
            frames,
            split,
            EVAL_STRIDE,
            lambda video_id: build_pose_features(
                video_id, pose_root=adapter.pose_root
            )[0],
            drop_ignored=False,
        )
        video_ids, k_ends, true_labels, pred_labels = _predict_with_identity(
            model, source, stats, device, batch_size=config.batch_size
        )
        usable_windows = len(source)
        total_windows = len(
            build_window_index(
                cast(pd.DataFrame, frames[frames["split"] == split]),
                stride=EVAL_STRIDE,
                drop_ignored=False,
            )
        )
        if usable_windows != total_windows:
            raise RuntimeError(
                f"split={split!r}: usable_windows ({usable_windows}) != "
                f"total_windows ({total_windows}) apesar de drop_ignored=False"
            )
        labeled_windows = len(
            build_window_index(
                cast(pd.DataFrame, frames[frames["split"] == split]),
                stride=EVAL_STRIDE,
                drop_ignored=True,
            )
        )
        split_report = split_event_report(
            video_ids,
            k_ends,
            true_labels,
            pred_labels,
            BASELINE_A_ALARM_PROTOCOL,
            usable_windows,
            total_windows,
            labeled_windows,
        )

        n_fall_segments_annotation = _n_fall_segments_in_annotation(frames, split)
        if split_report["n_fall_events"] != n_fall_segments_annotation:
            raise ValueError(
                f"split={split!r}: n_fall_events extraído das janelas usáveis "
                f"({split_report['n_fall_events']}) diverge da contagem de "
                "segmentos fall na anotação bruta "
                f"({n_fall_segments_annotation}) — possível janela IGNORE_LABEL "
                "descartada dentro de um run fall, dividindo um evento real em "
                "dois"
            )

        splits[split] = split_report

    report = {
        "run_name": config.run_name,
        "checkpoint_path": str(checkpoint_path),
        "alarm_protocol_path": str(alarm_protocol_path),
        "splits": splits,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    protocol_tmp = run_dir / f".alarm_protocol.pending-{token}.yaml"
    metrics_tmp = run_dir / f".event_metrics.pending-{token}.json"
    save_alarm_protocol(BASELINE_A_ALARM_PROTOCOL, protocol_tmp, force=True)
    report["checkpoint_sha256"] = sha256_file(checkpoint_path)
    report["training_metrics_sha256"] = sha256_file(run_dir / "metrics.json")
    report["alarm_protocol_sha256"] = sha256_file(protocol_tmp)
    with metrics_tmp.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with metrics_tmp.open(encoding="utf-8") as f:
        staged_report = json.load(f)
    staged_protocol = load_alarm_protocol(protocol_tmp)
    if staged_protocol != BASELINE_A_ALARM_PROTOCOL:
        raise RuntimeError("staging de alarm_protocol.yaml divergiu do protocolo")
    validate_event_metrics(
        staged_report,
        config,
        checkpoint_path,
        alarm_protocol_path,
        training_metrics_path=run_dir / "metrics.json",
        protocol_file_path=protocol_tmp,
        require_hashes=True,
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


def run_evaluate(
    force: bool, dataset_name: str = "le2i", run_dir: Path | None = None
) -> None:
    if run_dir is None:
        run_dir = default_run_dir(dataset_name)
    validate_local_run_dir(run_dir, dataset_name)
    with EventEvaluationLock(run_dir) as lock:
        _run_evaluate_locked(force, dataset_name, run_dir, lock)


def run_selftest() -> None:
    if not run_events_selftest():
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Roda o protocolo de alarme sobre o checkpoint treinado e grava event_metrics.json",
    )
    evaluate_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o event_metrics.json já existente"
    )
    evaluate_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)
    evaluate_parser.add_argument("--run-dir", type=Path, default=None)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do protocolo de eventos")

    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(
            force=args.force, dataset_name=args.dataset, run_dir=args.run_dir
        )
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
