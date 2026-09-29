"""Avaliação de eventos da arma A: protocolo de alarme sobre o checkpoint treinado."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.eval.shared.orchestration import (
    EventEvaluation,
    Predictions,
    SplitEvaluator,
    run_event_evaluation,
)
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization,
    load_stats,
    validate_stats_layout,
)
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import default_run_dir, validate_local_run_dir
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.artifacts import load_compatible_checkpoint, validate_training_run
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.shared.tcn import TCNClassifier

from gatefall.eval.shared.event_artifacts import EventEvaluationLock


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


def _run_evaluate_locked(
    force: bool,
    dataset_name: str,
    run_dir: Path,
    lock: EventEvaluationLock,
) -> None:
    def load_run() -> EventEvaluation:
        adapter = get_dataset(dataset_name)
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
            raise RuntimeError(f"run de treino inválido em {run_dir}: {exc}") from exc
        if config.eval_stride != EVAL_STRIDE:
            raise ValueError(
                f"config.eval_stride ({config.eval_stride}) diverge de "
                f"EVAL_STRIDE ({EVAL_STRIDE})"
            )

        def prepare() -> tuple[pd.DataFrame, SplitEvaluator]:
            stats = load_stats(adapter.pose_stats_path)
            validate_stats_layout(stats)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = _load_model(config, run_dir / "checkpoint.pt", device)
            frames = adapter.load_frames()

            def evaluate_split(split: str) -> tuple[int, Predictions]:
                source = PoseWindowDataset(
                    frames,
                    split,
                    EVAL_STRIDE,
                    lambda video_id: build_pose_features(
                        video_id, pose_root=adapter.pose_root
                    )[0],
                    drop_ignored=False,
                )
                predictions = _predict_with_identity(
                    model, source, stats, device, batch_size=config.batch_size
                )
                return len(source), predictions

            return frames, evaluate_split

        return EventEvaluation(config, prepare)

    run_event_evaluation(force, "A", run_dir, lock, load_run)


def run_evaluate(
    force: bool, dataset_name: str = "le2i", run_dir: Path | None = None
) -> None:
    if run_dir is None:
        run_dir = default_run_dir(dataset_name)
    validate_local_run_dir(run_dir, dataset_name)
    with EventEvaluationLock(run_dir) as lock:
        _run_evaluate_locked(force, dataset_name, run_dir, lock)


def run_selftest() -> None:
    from gatefall.eval.shared.selftests.events import run_events_selftest
    from gatefall.eval.shared.selftests.orchestration import run_orchestration_selftest

    if not all((run_events_selftest(), run_orchestration_selftest())):
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
