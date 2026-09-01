"""Arma A: TCN dilatada rasa treinada sobre o vetor de pose de 134 dimensões."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets import get_dataset
from gatefall.features.standardization import load_stats, validate_stats_layout
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import validate_local_run_dir
from gatefall.train.config import BASELINE_A_CONFIG
from gatefall.train.engine import run_training
from gatefall.train.metrics_selftest import run_metrics_selftest
from gatefall.train.tcn_selftest import run_tcn_selftest

RUN_DIR = Path("runs/local/le2i/baseline_a")


def run_train(force: bool, dataset_name: str = "le2i", run_dir: Path = RUN_DIR) -> None:
    validate_local_run_dir(run_dir)
    adapter = get_dataset(dataset_name)
    stats = load_stats(adapter.stats_path)
    validate_stats_layout(stats)
    config = replace(
        BASELINE_A_CONFIG,
        standardization_stats_path=str(adapter.stats_path),
        standardization_stats_sha256=sha256_file(adapter.stats_path),
    )
    frames = adapter.load_frames()
    loader = lambda video_id: build_pose_features(video_id)[0]

    train_source = PoseWindowDataset(frames, "train", TRAIN_STRIDE, loader)
    val_source = PoseWindowDataset(frames, "val", EVAL_STRIDE, loader)
    test_source = PoseWindowDataset(frames, "test", EVAL_STRIDE, loader)

    run_training(
        input_dim=adapter.feature_dim,
        train_source=train_source,
        val_source=val_source,
        test_source=test_source,
        stats=stats,
        config=config,
        run_dir=run_dir,
        force=force,
        label_names=adapter.label_names,
    )


def run_selftest() -> None:
    tcn_ok = run_tcn_selftest()
    metrics_ok = run_metrics_selftest()
    if not (tcn_ok and metrics_ok):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Treina a arma A (TCN) e grava config.yaml/metrics.json/checkpoint.pt"
    )
    train_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o run_dir já existente"
    )
    train_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    train_parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da TCN e das métricas")

    args = parser.parse_args()
    if args.command == "train":
        run_train(force=args.force, dataset_name=args.dataset, run_dir=args.run_dir)
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
