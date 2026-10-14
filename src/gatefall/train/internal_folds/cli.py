"""Treino exploratório e não canônico de A, B0 e B1 nas dobras internas por ambiente do Le2i-CV (Etapa 2C)."""

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd

from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.dinov3 import storage as dinov3_storage
from gatefall.dinov3.report import find_provenance_divergences
from gatefall.dinov3.storage import dinov3_path, read_features
from gatefall.eval.analysis.classification_diagnosis import (
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
)
from gatefall.eval.analysis.internal_fold_feasibility import (
    read_train_partition,
    validate_official_train_partition,
)
from gatefall.features.dinov3_standardization import FEATURE_DIM as DINOV3_FEATURE_DIM
from gatefall.features.quality_extract import current_provenance, pose_source_sha256
from gatefall.features.quality_storage import quality_path, quality_set_sha256, read_quality
from gatefall.features.quality_storage import validate_existing_file as validate_quality_file
from gatefall.pose.kinematics import build_pose_features
from gatefall.train.internal_folds.folds import (
    ARMS,
    FOLD_RUN_ROOT,
    FOLDS_FILE,
    SEED_SETS,
    FoldPartition,
    build_fold_partitions,
    fold_names,
    fold_run_dir,
    folds_document,
    read_folds_document,
    run_plan,
    stats_dir,
    validate_folds_document,
    validate_output_root,
)
from gatefall.train.internal_folds.stats import fit_fold_stats, load_fold_stats
from gatefall.train.internal_folds.summary import write_summary
from gatefall.train.internal_folds.training import FeatureLoaders, QualityFeatures, train_fold_run

ADAPTER = Le2iDatasetAdapter(protocol="cv")


def _official_train(
    frames_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    snapshot = snapshot_inputs([frames_path, manifest_path])
    frames = read_train_partition(frames_path)
    manifest = read_train_partition(manifest_path)
    validate_official_train_partition(frames, manifest)
    input_sha256 = {display_path(Path(path)): digest for path, digest in snapshot.items()}
    return frames, manifest, snapshot, input_sha256


def validate_feature_files(frames: pd.DataFrame, manifest: pd.DataFrame) -> QualityFeatures:
    """Confere os HDF5 DINOv3 e de qualidade só dos vídeos do `train` oficial.

    Os validadores canônicos leem a tabela inteira do Le2i (todos os splits);
    aqui os atributos são conferidos contra o manifesto já filtrado.
    """
    sizes = cast(pd.Series, frames.groupby("video_id").size())
    frame_counts = {str(video_id): int(cast(int, count)) for video_id, count in sizes.items()}
    metadata = {
        str(video_id): (str(env), int(subject), float(fps))
        for video_id, env, subject, fps in zip(
            manifest["video_id"], manifest["env"], manifest["subject"], manifest["fps"]
        )
    }
    provenance_by_video: dict[str, dict[str, object]] = {}
    for video_id in sorted(frame_counts):
        env, subject, fps = metadata[video_id]
        expected_k = frame_counts[video_id]
        identity = {"video_id": video_id, "K": expected_k, "env": env, "subject": subject}
        visual = dinov3_path(video_id, dinov3_root=ADAPTER.dinov3_root)
        if not visual.exists():
            raise FileNotFoundError(f"feature DINOv3 ausente: {visual}")
        reasons = dinov3_storage.validate_existing_file(
            visual, expected_k=expected_k, feature_dim=DINOV3_FEATURE_DIM, expected_attrs=identity
        )
        quality = quality_path(video_id, quality_root=ADAPTER.quality_root)
        if not quality.exists():
            raise FileNotFoundError(f"sidecar de qualidade ausente: {quality}")
        reasons += validate_quality_file(
            quality,
            expected_k=expected_k,
            expected_attrs={
                **identity,
                "fps": fps,
                **current_provenance(pose_source_digest=pose_source_sha256(video_id, pose_root=ADAPTER.pose_root)),
            },
        )
        if reasons:
            raise ValueError(f"{video_id}: features inválidas: {'; '.join(reasons)}")
        with h5py.File(visual, "r") as h5_file:
            provenance_by_video[video_id] = {
                name: h5_file.attrs[name] for name in dinov3_storage.PROVENANCE_ATTR_NAMES if name in h5_file.attrs
            }
    divergences = find_provenance_divergences(provenance_by_video)
    if divergences:
        raise ValueError("proveniência DINOv3 inválida: " + "; ".join(divergences))
    return QualityFeatures(
        root=ADAPTER.quality_root, sha256=quality_set_sha256(list(frame_counts), quality_root=ADAPTER.quality_root)
    )


def feature_loaders() -> FeatureLoaders:
    def pose(video_id: str) -> np.ndarray:
        return build_pose_features(video_id, pose_root=ADAPTER.pose_root)[0]

    def visual(video_id: str) -> np.ndarray:
        return read_features(dinov3_path(video_id, dinov3_root=ADAPTER.dinov3_root)).astype(np.float32)

    def quality(video_id: str) -> np.ndarray:
        return read_quality(quality_path(video_id, quality_root=ADAPTER.quality_root)).astype(np.float32)

    return FeatureLoaders(pose=pose, visual=visual, quality=quality)


def _write_folds_document(root: Path, document: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / FOLDS_FILE
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"{path}: gravado")


def run_prepare(
    root: Path,
    frames_path: Path,
    manifest_path: Path,
    force: bool,
    loaders: FeatureLoaders | None = None,
    quality: QualityFeatures | None = None,
) -> list[FoldPartition]:
    validate_output_root(root, frames_path, manifest_path)
    frames, manifest, snapshot, input_sha256 = _official_train(frames_path, manifest_path)
    quality = quality or validate_feature_files(frames, manifest)
    loaders = loaders or feature_loaders()
    partitions = build_fold_partitions(frames)
    document = folds_document(partitions, input_sha256, str(quality.root), quality.sha256)
    if (root / FOLDS_FILE).exists() and not force:
        existing = read_folds_document(root)
        if existing != document:
            raise RuntimeError(f"{root / FOLDS_FILE} diverge das dobras atuais; use --force para refazer")
    for partition in partitions:
        fit_fold_stats(
            partition,
            stats_dir(root, partition.fold),
            frames_path,
            input_sha256,
            loaders.pose,
            loaders.visual,
            force,
        )
    _write_folds_document(root, document)
    verify_inputs_unchanged(snapshot)
    return partitions


def run_train(
    root: Path,
    frames_path: Path,
    manifest_path: Path,
    seed_set: str,
    folds: list[str],
    arms: list[str],
    force: bool,
    loaders: FeatureLoaders | None = None,
    quality: QualityFeatures | None = None,
) -> None:
    validate_output_root(root, frames_path, manifest_path)
    frames, manifest, snapshot, input_sha256 = _official_train(frames_path, manifest_path)
    quality = quality or validate_feature_files(frames, manifest)
    loaders = loaders or feature_loaders()
    partitions = build_fold_partitions(frames)
    document = read_folds_document(root)
    validate_folds_document(document, partitions, input_sha256)
    if (document.get("quality_features_path"), document.get("quality_features_sha256")) != (
        str(quality.root),
        quality.sha256,
    ):
        raise ValueError(f"{FOLDS_FILE}: os sidecars de qualidade mudaram desde o `prepare`")
    for partition in partitions:
        if partition.fold not in folds:
            continue
        stats = load_fold_stats(partition, stats_dir(root, partition.fold), frames_path, input_sha256)
        for arm in arms:
            for seed in SEED_SETS[seed_set]:
                run_dir = fold_run_dir(root, partition.fold, arm, seed)
                print(f"== {partition.fold} / {arm} / seed {seed} -> {run_dir}")
                train_fold_run(partition, arm, seed, stats, quality, loaders, run_dir, force)
    verify_inputs_unchanged(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_root(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--output-root", type=Path, default=FOLD_RUN_ROOT)

    def add_inputs(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--frames", type=Path, default=ADAPTER.frames_path)
        subparser.add_argument("--manifest", type=Path, default=ADAPTER.manifest_path)

    plan_parser = subparsers.add_parser("plan", help="Lista, sem ler dados, os runs de um conjunto de seeds")
    add_root(plan_parser)
    plan_parser.add_argument("--seed-set", choices=tuple(SEED_SETS), required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Define as dobras e ajusta as estatísticas de pose e DINOv3 de cada dobra"
    )
    add_root(prepare_parser)
    add_inputs(prepare_parser)
    prepare_parser.add_argument("--force", action="store_true")

    train_parser = subparsers.add_parser("train", help="Treina A, B0 e B1 nas dobras com a receita congelada")
    add_root(train_parser)
    add_inputs(train_parser)
    train_parser.add_argument("--seed-set", choices=tuple(SEED_SETS), required=True)
    train_parser.add_argument("--fold", nargs="+", choices=fold_names(), default=fold_names())
    train_parser.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    train_parser.add_argument("--force", action="store_true")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Resume os runs das dobras e aplica o critério congelado da Etapa 2A"
    )
    add_root(summarize_parser)
    summarize_parser.add_argument("--seed-set", choices=tuple(SEED_SETS), required=True)
    summarize_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("selftest", help="Roda checagens sintéticas das dobras internas")

    args = parser.parse_args()
    if args.command == "plan":
        validate_output_root(args.output_root, ADAPTER.frames_path, ADAPTER.manifest_path)
        print(json.dumps(run_plan(args.output_root, args.seed_set), indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        run_prepare(args.output_root, args.frames, args.manifest, args.force)
    elif args.command == "train":
        run_train(args.output_root, args.frames, args.manifest, args.seed_set, args.fold, args.arm, args.force)
    elif args.command == "summarize":
        validate_output_root(args.output_root, ADAPTER.frames_path, ADAPTER.manifest_path)
        write_summary(args.output_root, args.seed_set, args.force)
    elif args.command == "selftest":
        from gatefall.train.internal_folds.selftests import run_selftest

        if not run_selftest():
            sys.exit(1)


if __name__ == "__main__":
    main()
