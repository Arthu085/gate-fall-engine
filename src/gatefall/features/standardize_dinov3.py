"""Padronização (z-score) das features DINOv3 do Le2i, treinada só no split de treino."""

import argparse
import sys
from pathlib import Path

import numpy as np

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.dinov3.dataset_guard import (
    DINOV3_SUPPORTED_DATASET_IDENTIFIERS,
    ensure_dinov3_dataset_supported,
)
from gatefall.dinov3.storage import read_features
from gatefall.features.dinov3_standardization import (
    FEATURE_DIM,
    TRAIN_SPLIT,
    apply_standardization,
    compute_train_stats,
    load_stats,
    save_stats,
    stale_stats_mismatches,
    validate_stats_layout,
)
from gatefall.features.dinov3_standardization_selftest import (
    run_dinov3_standardization_selftest,
)
from gatefall.features.standardize import EXPECTED_USABLE_WINDOWS_STRIDE4
from gatefall.hashing import sha256_file

DINOV3_STATS_PATH = Path("src/gatefall/features/stats/dinov3_le2i_cs.json")

EVAL_SPLITS = ["val", "test"]


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _dinov3_feature_loader(adapter: DatasetAdapter, video_id: str) -> np.ndarray:
    from gatefall.dinov3.storage import dinov3_path

    path = dinov3_path(video_id, dinov3_root=adapter.dinov3_root)
    return read_features(path).astype(np.float32)


def run_build(force: bool, dataset_name: str = "le2i") -> None:
    adapter = get_dataset(dataset_name)
    ensure_dinov3_dataset_supported(adapter)
    source = PoseWindowDataset(
        adapter.load_frames(),
        TRAIN_SPLIT,
        TRAIN_STRIDE,
        lambda video_id: _dinov3_feature_loader(adapter, video_id),
    )
    stats = compute_train_stats(
        source, adapter.frames_path, adapter.identifier, stride=TRAIN_STRIDE
    )
    save_stats(stats, DINOV3_STATS_PATH, force=force)


def run_report(dataset_name: str = "le2i") -> None:
    adapter = get_dataset(dataset_name)
    ensure_dinov3_dataset_supported(adapter)
    if not DINOV3_STATS_PATH.exists():
        print(
            f"{DINOV3_STATS_PATH} não existe; rode `build` antes de `report`",
            file=sys.stderr,
        )
        sys.exit(1)

    stats = load_stats(DINOV3_STATS_PATH)
    validate_stats_layout(stats, dataset_name=adapter.identifier)
    frames = adapter.load_frames()
    expected_usable_windows = EXPECTED_USABLE_WINDOWS_STRIDE4[adapter.identifier]

    checks: list[bool] = []

    mismatches = stale_stats_mismatches(stats, dataset_name=adapter.identifier)
    checks.append(
        _check(
            "estatísticas DINOv3 persistidas batem com o layout atual (campos "
            f"divergentes: {mismatches if mismatches else 'nenhum'})",
            not mismatches,
        )
    )

    checks.append(
        _check(
            f"hash persistido de {adapter.frames_path} bate com o arquivo atual",
            stats.frames_hash == sha256_file(adapter.frames_path),
        )
    )

    mean = np.asarray(stats.mean, dtype=np.float64)
    std = np.asarray(stats.std, dtype=np.float64)
    checks.append(
        _check(
            f"mean e std têm shape [{FEATURE_DIM}] e são finitos",
            mean.shape == (FEATURE_DIM,)
            and std.shape == (FEATURE_DIM,)
            and bool(np.isfinite(mean).all())
            and bool(np.isfinite(std).all()),
        )
    )

    train_dataset = PoseWindowDataset(
        frames,
        TRAIN_SPLIT,
        TRAIN_STRIDE,
        lambda video_id: _dinov3_feature_loader(adapter, video_id),
    )

    checks.append(
        _check(
            f"contagem de janelas de treino em stride={TRAIN_STRIDE} == "
            f"{expected_usable_windows[TRAIN_SPLIT]}",
            len(train_dataset) == expected_usable_windows[TRAIN_SPLIT]
            and stats.window_count == expected_usable_windows[TRAIN_SPLIT],
        )
    )

    train_raw_windows: list[np.ndarray] = []
    for i in range(len(train_dataset)):
        window, _, _ = train_dataset[i]
        train_raw_windows.append(window)
    train_raw = np.concatenate(train_raw_windows, axis=0)
    train_standardized = apply_standardization(train_raw, stats)

    standardized_dim_mask = ~np.asarray(stats.guarded_mask, dtype=bool)
    per_dim_mean = train_standardized[:, standardized_dim_mask].astype(np.float64).mean(axis=0)
    per_dim_std = train_standardized[:, standardized_dim_mask].astype(np.float64).std(
        axis=0, ddof=0
    )
    checks.append(
        _check(
            "após padronizar: mean por dimensão dentro de 1e-3 de 0 e std dentro de "
            "1e-3 de 1 nas dimensões padronizadas",
            bool(np.allclose(per_dim_mean, 0.0, atol=1e-3))
            and bool(np.allclose(per_dim_std, 1.0, atol=1e-3)),
        )
    )

    eval_non_finite: dict[str, int] = {}
    for split in EVAL_SPLITS:
        dataset = PoseWindowDataset(
            frames,
            split,
            EVAL_STRIDE,
            lambda video_id: _dinov3_feature_loader(adapter, video_id),
        )
        non_finite = 0
        for i in range(len(dataset)):
            window, _, _ = dataset[i]
            standardized = apply_standardization(window, stats)
            non_finite += int(np.sum(~np.isfinite(standardized)))
        eval_non_finite[split] = non_finite
        checks.append(
            _check(f"split={split}: nenhum valor não finito após padronizar", non_finite == 0)
        )

    print(f"\ndimensões guardadas: {stats.guarded_count}")

    if not all(checks):
        print("\nstandardize_dinov3 report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nstandardize_dinov3 report OK: todas as checagens passaram")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Calcula e persiste as estatísticas de padronização DINOv3 do treino"
    )
    build_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o arquivo já existente"
    )
    build_parser.add_argument(
        "--dataset", default="le2i", choices=DINOV3_SUPPORTED_DATASET_IDENTIFIERS
    )
    selftest_parser = subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas da padronização DINOv3"
    )
    selftest_parser.add_argument(
        "--dataset", default="le2i", choices=DINOV3_SUPPORTED_DATASET_IDENTIFIERS
    )
    report_parser = subparsers.add_parser(
        "report",
        help="Roda a padronização DINOv3 sobre o dataset real do Le2i e reporta estatísticas",
    )
    report_parser.add_argument(
        "--dataset", default="le2i", choices=DINOV3_SUPPORTED_DATASET_IDENTIFIERS
    )
    return parser


def main() -> None:
    args = build_cli_parser().parse_args()
    if args.command == "build":
        run_build(force=args.force, dataset_name=args.dataset)
    elif args.command == "selftest":
        ok = run_dinov3_standardization_selftest()
        if not ok:
            sys.exit(1)
    elif args.command == "report":
        run_report(dataset_name=args.dataset)


if __name__ == "__main__":
    main()
