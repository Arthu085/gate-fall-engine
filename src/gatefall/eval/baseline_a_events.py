"""Avaliação de eventos da arma A: protocolo de alarme sobre o checkpoint treinado."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE
from gatefall.data.frames import read_frames
from gatefall.data.le2i.frames import FRAMES_PATH
from gatefall.data.le2i.pose_dataset import PoseWindowDataset, load_le2i_pose_window_dataset
from gatefall.data.windowing import build_window_index
from gatefall.eval.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, save_alarm_protocol
from gatefall.eval.events import extract_label_segments, split_event_report
from gatefall.eval.events_selftest import run_events_selftest
from gatefall.features.standardization import (
    StandardizationStats,
    STATS_PATH,
    apply_standardization,
    load_stats,
)
from gatefall.train.config import TrainConfig, load_config
from gatefall.train.tcn import TCNClassifier

RUN_DIR = Path("runs/baseline_a")
CHECKPOINT_PATH = RUN_DIR / "checkpoint.pt"
CONFIG_PATH = RUN_DIR / "config.yaml"
ALARM_PROTOCOL_PATH = RUN_DIR / "alarm_protocol.yaml"
EVENT_METRICS_PATH = RUN_DIR / "event_metrics.json"


def _load_model(config: TrainConfig, device: str) -> TCNClassifier:
    model = TCNClassifier(
        input_dim=config.input_dim,
        channels=config.channels,
        kernel_size=config.kernel_size,
        dilations=config.dilations,
        dropout=config.dropout,
        num_classes=config.num_classes,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
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


def run_evaluate(force: bool) -> None:
    if EVENT_METRICS_PATH.exists() and not force:
        print(f"skip {EVENT_METRICS_PATH} (já existe, use --force para sobrescrever)")
        return

    config = load_config(CONFIG_PATH)
    assert config.eval_stride == EVAL_STRIDE, (
        f"config.eval_stride ({config.eval_stride}) diverge de EVAL_STRIDE ({EVAL_STRIDE})"
    )
    stats = load_stats(STATS_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(config, device)

    save_alarm_protocol(BASELINE_A_ALARM_PROTOCOL, ALARM_PROTOCOL_PATH, force=True)

    frames = read_frames(FRAMES_PATH)

    splits: dict[str, dict] = {}
    for split in ("val", "test"):
        source = load_le2i_pose_window_dataset(split, EVAL_STRIDE)
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
        split_report = split_event_report(
            video_ids,
            k_ends,
            true_labels,
            pred_labels,
            BASELINE_A_ALARM_PROTOCOL,
            usable_windows,
            total_windows,
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
        "checkpoint_path": str(CHECKPOINT_PATH),
        "alarm_protocol_path": str(ALARM_PROTOCOL_PATH),
        "splits": splits,
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = EVENT_METRICS_PATH.with_suffix(EVENT_METRICS_PATH.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, EVENT_METRICS_PATH)

    print(f"{EVENT_METRICS_PATH}: métricas de eventos gravadas (run_name={config.run_name})")


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
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do protocolo de eventos")

    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(force=args.force)
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
