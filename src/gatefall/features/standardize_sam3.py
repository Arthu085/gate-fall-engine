"""Padronização (z-score) do descritor SAM 3 V_t do Le2i, treinada só no split de treino."""

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.features.sam3_standardization import (
    FEATURE_DIM,
    TRAIN_SPLIT,
    apply_standardization,
    compute_train_stats,
    load_stats,
    save_stats,
    stale_stats_mismatches,
    validate_stats_layout,
)
from gatefall.features.standardize import EXPECTED_USABLE_WINDOWS_STRIDE4
from gatefall.hashing import sha256_file
from gatefall.sam3.dataset_guard import (
    SAM3_SUPPORTED_DATASET_IDENTIFIERS,
    ensure_sam3_dataset_supported,
)
from gatefall.sam3.features import collect_sam3_provenance, load_v_t, sam3_set_sha256

SAM3_STATS_PATH = Path("src/gatefall/features/stats/sam3_le2i_cs.json")

EVAL_SPLITS = ["val", "test"]


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def split_by_video(frames: pd.DataFrame) -> dict[str, str]:
    splits = cast(pd.Series, frames.groupby("video_id")["split"].first())
    return {str(video_id): str(split) for video_id, split in splits.items()}


def validated_sam3_features_sha256(adapter: DatasetAdapter, frames: pd.DataFrame) -> str:
    """Valida identidade/proveniência de todos os `.h5` e devolve o digest do conjunto."""
    splits = split_by_video(frames)
    collect_sam3_provenance(splits, sam3_root=adapter.sam3_root)
    return sam3_set_sha256(list(splits), sam3_root=adapter.sam3_root)


def _window_source(adapter: DatasetAdapter, frames: pd.DataFrame, split: str, stride: int):
    return PoseWindowDataset(
        frames,
        split,
        stride,
        lambda video_id: load_v_t(video_id, sam3_root=adapter.sam3_root),
    )


def run_build(force: bool, dataset_name: str = "le2i") -> None:
    adapter = get_dataset(dataset_name)
    ensure_sam3_dataset_supported(adapter)
    frames = adapter.load_frames()
    features_sha256 = validated_sam3_features_sha256(adapter, frames)
    stats = compute_train_stats(
        _window_source(adapter, frames, TRAIN_SPLIT, TRAIN_STRIDE),
        adapter.frames_path,
        adapter.identifier,
        features_sha256,
        stride=TRAIN_STRIDE,
    )
    save_stats(stats, SAM3_STATS_PATH, force=force)


def run_report(dataset_name: str = "le2i") -> None:
    adapter = get_dataset(dataset_name)
    ensure_sam3_dataset_supported(adapter)
    if not SAM3_STATS_PATH.exists():
        print(f"{SAM3_STATS_PATH} não existe; rode `build` antes de `report`", file=sys.stderr)
        sys.exit(1)

    stats = load_stats(SAM3_STATS_PATH)
    validate_stats_layout(stats, dataset_name=adapter.identifier)
    frames = adapter.load_frames()
    features_sha256 = validated_sam3_features_sha256(adapter, frames)
    expected_usable_windows = EXPECTED_USABLE_WINDOWS_STRIDE4[adapter.identifier]

    checks: list[bool] = []
    mismatches = stale_stats_mismatches(stats, dataset_name=adapter.identifier)
    checks.append(
        _check(
            "estatísticas SAM 3 persistidas batem com o layout atual (campos "
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
    checks.append(
        _check(
            "sam3_features_sha256 persistido bate com o conjunto de .h5 atual",
            stats.sam3_features_sha256 == features_sha256,
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

    train_dataset = _window_source(adapter, frames, TRAIN_SPLIT, TRAIN_STRIDE)
    checks.append(
        _check(
            f"contagem de janelas de treino em stride={TRAIN_STRIDE} == "
            f"{expected_usable_windows[TRAIN_SPLIT]}",
            len(train_dataset) == expected_usable_windows[TRAIN_SPLIT]
            and stats.window_count == expected_usable_windows[TRAIN_SPLIT],
        )
    )
    train_raw = np.concatenate(
        [train_dataset[i][0] for i in range(len(train_dataset))], axis=0
    )
    train_standardized = apply_standardization(train_raw, stats).astype(np.float64)
    standardized_mask = ~np.asarray(stats.guarded_mask, dtype=bool)
    checks.append(
        _check(
            "após padronizar: mean por canal dentro de 1e-3 de 0 e std dentro de "
            "1e-3 de 1 nos canais padronizados",
            bool(np.allclose(train_standardized[:, standardized_mask].mean(axis=0), 0.0, atol=1e-3))
            and bool(np.allclose(train_standardized[:, standardized_mask].std(axis=0), 1.0, atol=1e-3)),
        )
    )

    for split in EVAL_SPLITS:
        dataset = _window_source(adapter, frames, split, EVAL_STRIDE)
        non_finite = sum(
            int(np.sum(~np.isfinite(apply_standardization(dataset[i][0], stats))))
            for i in range(len(dataset))
        )
        checks.append(
            _check(f"split={split}: nenhum valor não finito após padronizar", non_finite == 0)
        )

    print(f"\ncanais guardados: {stats.guarded_count}")
    if not all(checks):
        print("\nstandardize_sam3 report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nstandardize_sam3 report OK: todas as checagens passaram")


def main() -> None:
    from gatefall.features.sam3_standardization_selftest import (
        run_sam3_standardization_selftest,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser(
        "build", help="Calcula e persiste as estatísticas de padronização SAM 3 do treino"
    )
    build_parser.add_argument("--force", action="store_true", help="Sobrescreve o arquivo já existente")
    build_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da padronização SAM 3")
    report_parser = subparsers.add_parser(
        "report", help="Valida as estatísticas SAM 3 persistidas contra o Le2i real"
    )
    report_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )

    args = parser.parse_args()
    if args.command == "build":
        run_build(force=args.force, dataset_name=args.dataset)
    elif args.command == "selftest":
        if not run_sam3_standardization_selftest():
            sys.exit(1)
    elif args.command == "report":
        run_report(dataset_name=args.dataset)


if __name__ == "__main__":
    main()
