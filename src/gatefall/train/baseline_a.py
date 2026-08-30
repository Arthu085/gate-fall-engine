"""Arma A: TCN dilatada rasa treinada sobre o vetor de pose de 134 dimensões."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.le2i.pose_dataset import (
    EXPECTED_FEATURE_DIM,
    load_le2i_pose_window_dataset,
)
from gatefall.features.standardization import STATS_PATH, load_stats
from gatefall.hashing import sha256_file
from gatefall.train.config import BASELINE_A_CONFIG
from gatefall.train.engine import run_training
from gatefall.train.metrics_selftest import run_metrics_selftest
from gatefall.train.tcn_selftest import run_tcn_selftest

RUN_DIR = Path("runs/baseline_a")


def run_train(force: bool) -> None:
    stats = load_stats(STATS_PATH)
    config = replace(BASELINE_A_CONFIG, standardization_stats_sha256=sha256_file(STATS_PATH))

    train_source = load_le2i_pose_window_dataset("train", TRAIN_STRIDE)
    val_source = load_le2i_pose_window_dataset("val", EVAL_STRIDE)
    test_source = load_le2i_pose_window_dataset("test", EVAL_STRIDE)

    run_training(
        input_dim=EXPECTED_FEATURE_DIM,
        train_source=train_source,
        val_source=val_source,
        test_source=test_source,
        stats=stats,
        config=config,
        run_dir=RUN_DIR,
        force=force,
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
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da TCN e das métricas")

    args = parser.parse_args()
    if args.command == "train":
        run_train(force=args.force)
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
