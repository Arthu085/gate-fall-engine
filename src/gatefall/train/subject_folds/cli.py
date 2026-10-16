"""Braço A exploratório e não canônico nas dobras por sujeito do Le2i-CS: linha de base (Etapa 4B) e perda
simulada de detecção (Etapa 4C)."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.eval.analysis.classification_diagnosis import display_path, snapshot_inputs, verify_inputs_unchanged
from gatefall.eval.analysis.internal_fold_feasibility import read_train_partition, validate_official_train_partition
from gatefall.eval.analysis.subject_fold_feasibility import EXPECTED_CS_TRAIN, validate_subject_identifiers
from gatefall.pose.kinematics import build_pose_features
from gatefall.pose.loading import PoseArrays, load_pose
from gatefall.train.internal_folds.folds import FeatureLoader
from gatefall.train.subject_folds.detection_dropout import RawPoseLoader
from gatefall.train.subject_folds.dropout_summary import write_dropout_summary
from gatefall.train.subject_folds.folds import (
    ARMS,
    BASELINE_VARIANT,
    DETECTION_DROPOUT_VARIANT,
    FOLDS_FILE,
    SEEDS,
    VARIANTS,
    SUBJECT_FOLD_ROOT,
    SubjectFoldPartition,
    build_subject_fold_partitions,
    fold_names,
    fold_run_dir,
    folds_document,
    pose_features_sha256,
    read_folds_document,
    run_plan,
    stats_dir,
    validate_folds_document,
    validate_output_root,
)
from gatefall.train.subject_folds.stats import fit_fold_stats, load_fold_stats
from gatefall.train.subject_folds.summary import control_fold_run_sha256, write_summary
from gatefall.train.subject_folds.training import detection_dropout_run_document, fold_run_document, train_fold_run

ADAPTER = Le2iDatasetAdapter(protocol="cs")


@dataclass(frozen=True)
class PoseInputs:
    loader: FeatureLoader
    sha256: str
    raw_loader: RawPoseLoader | None = None


def missing_prerequisites(
    frames_path: Path, manifest_path: Path, pose_root: Path, require_gpu: bool
) -> list[str]:
    missing = [display_path(path) for path in (frames_path, manifest_path) if not path.is_file()]
    if not pose_root.is_dir():
        missing.append(display_path(pose_root))
    if require_gpu and not torch.cuda.is_available():
        missing.append("GPU CUDA (torch.cuda.is_available() é falso)")
    return missing


def _require_prerequisites(frames_path: Path, manifest_path: Path, pose_root: Path, require_gpu: bool) -> None:
    missing = missing_prerequisites(frames_path, manifest_path, pose_root, require_gpu)
    if missing:
        print(f"pré-requisitos ausentes: {missing}; nada foi gravado", file=sys.stderr)
        sys.exit(2)


def _official_train(
    frames_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    snapshot = snapshot_inputs([frames_path, manifest_path])
    frames = read_train_partition(frames_path)
    manifest = read_train_partition(manifest_path)
    validate_subject_identifiers(frames, manifest)
    validate_official_train_partition(frames, manifest, EXPECTED_CS_TRAIN)
    input_sha256 = {"frames": snapshot[str(frames_path)], "manifest": snapshot[str(manifest_path)]}
    return frames, snapshot, input_sha256


def real_pose_inputs(frames: pd.DataFrame, pose_root: Path) -> PoseInputs:
    video_ids = sorted(str(video_id) for video_id in frames["video_id"].unique())

    def loader(video_id: str) -> np.ndarray:
        return build_pose_features(video_id, pose_root=pose_root)[0]

    def raw_loader(video_id: str) -> PoseArrays:
        return load_pose(video_id, pose_root=pose_root)

    return PoseInputs(loader=loader, sha256=pose_features_sha256(video_ids, pose_root), raw_loader=raw_loader)


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
    pose_root: Path,
    force: bool,
    pose: PoseInputs | None = None,
) -> list[SubjectFoldPartition]:
    validate_output_root(root, frames_path, manifest_path, pose_root)
    frames, snapshot, input_sha256 = _official_train(frames_path, manifest_path)
    pose = pose or real_pose_inputs(frames, pose_root)
    partitions = build_subject_fold_partitions(frames)
    document = folds_document(partitions, input_sha256, pose.sha256)
    if (root / FOLDS_FILE).exists() and not force:
        existing = read_folds_document(root)
        if existing != document:
            raise RuntimeError(f"{root / FOLDS_FILE} diverge das dobras atuais; use --force para refazer")
    for partition in partitions:
        fit_fold_stats(
            partition, stats_dir(root, partition.fold), frames_path, input_sha256, pose.sha256, pose.loader, force
        )
    _write_folds_document(root, document)
    verify_inputs_unchanged(snapshot)
    return partitions


def run_train(
    root: Path,
    frames_path: Path,
    manifest_path: Path,
    pose_root: Path,
    folds: list[str],
    arms: list[str],
    seeds: list[int],
    force: bool,
    pose: PoseInputs | None = None,
    variant: str = BASELINE_VARIANT,
) -> None:
    """Treina a variante pedida; `detection_dropout` exige e valida antes os controles `baseline` das dobras."""
    validate_output_root(root, frames_path, manifest_path, pose_root)
    frames, snapshot, input_sha256 = _official_train(frames_path, manifest_path)
    pose = pose or real_pose_inputs(frames, pose_root)
    all_partitions = build_subject_fold_partitions(frames)
    document = read_folds_document(root)
    validate_folds_document(document, all_partitions, input_sha256, pose.sha256)
    partitions = [partition for partition in all_partitions if partition.fold in folds]
    membership = str(document["membership_sha256"])
    controls: dict[tuple[str, str, int], str] = {}
    if variant == DETECTION_DROPOUT_VARIANT:
        controls = {
            (partition.fold, arm, seed): control_fold_run_sha256(root, document, partition.membership_record, arm, seed)
            for partition in partitions
            for arm in arms
            for seed in seeds
        }
    for partition in partitions:
        record = partition.membership_record
        stats = load_fold_stats(record, stats_dir(root, partition.fold), input_sha256, pose.sha256)
        for arm in arms:
            for seed in seeds:
                run_dir = fold_run_dir(root, partition.fold, arm, seed, variant)
                print(f"== {partition.fold} / {arm} / seed {seed} / {variant} -> {run_dir}")
                run_document = fold_run_document(
                    record, arm, seed, stats.sha256, input_sha256, pose.sha256, membership
                )
                if variant == DETECTION_DROPOUT_VARIANT:
                    run_document = detection_dropout_run_document(run_document, controls[(partition.fold, arm, seed)])
                train_fold_run(
                    partition, seed, stats, pose.loader, run_document, run_dir, force, variant, pose.raw_loader
                )
    verify_inputs_unchanged(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_root(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--output-root", type=Path, default=SUBJECT_FOLD_ROOT)

    def add_variant(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--variant", choices=VARIANTS, default=BASELINE_VARIANT)

    def add_inputs(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--frames", type=Path, default=ADAPTER.frames_path)
        subparser.add_argument("--manifest", type=Path, default=ADAPTER.manifest_path)
        subparser.add_argument("--pose-root", type=Path, default=ADAPTER.pose_root)

    plan_parser = subparsers.add_parser("plan", help="Lista, sem ler dados, os runs das dobras por sujeito")
    add_root(plan_parser)
    add_variant(plan_parser)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Fixa as dobras por sujeito e ajusta a estatística de pose de cada dobra"
    )
    add_root(prepare_parser)
    add_inputs(prepare_parser)
    prepare_parser.add_argument("--force", action="store_true")

    train_parser = subparsers.add_parser("train", help="Treina o braço A nas dobras com a receita congelada")
    add_root(train_parser)
    add_inputs(train_parser)
    train_parser.add_argument("--fold", nargs="+", choices=fold_names(), default=fold_names())
    train_parser.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    train_parser.add_argument("--seed", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    train_parser.add_argument("--force", action="store_true")
    add_variant(train_parser)

    summarize_parser = subparsers.add_parser(
        "summarize", help="Valida os runs e relata macro-F1 restrita, F1 e suporte por classe de cada dobra"
    )
    add_root(summarize_parser)
    summarize_parser.add_argument("--force", action="store_true")
    add_variant(summarize_parser)

    subparsers.add_parser("selftest", help="Roda checagens sintéticas das dobras por sujeito")

    args = parser.parse_args()
    if args.command == "plan":
        validate_output_root(args.output_root, ADAPTER.frames_path, ADAPTER.manifest_path, ADAPTER.pose_root)
        print(json.dumps(run_plan(args.output_root, args.variant), indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        _require_prerequisites(args.frames, args.manifest, args.pose_root, require_gpu=False)
        run_prepare(args.output_root, args.frames, args.manifest, args.pose_root, args.force)
    elif args.command == "train":
        _require_prerequisites(args.frames, args.manifest, args.pose_root, require_gpu=True)
        run_train(
            args.output_root,
            args.frames,
            args.manifest,
            args.pose_root,
            args.fold,
            args.arm,
            args.seed,
            args.force,
            variant=args.variant,
        )
    elif args.command == "summarize":
        validate_output_root(args.output_root, ADAPTER.frames_path, ADAPTER.manifest_path, ADAPTER.pose_root)
        if args.variant == DETECTION_DROPOUT_VARIANT:
            write_dropout_summary(args.output_root, args.force)
        else:
            write_summary(args.output_root, args.force)
    elif args.command == "selftest":
        from gatefall.train.subject_folds.selftests import run_selftest

        if not run_selftest():
            sys.exit(1)


if __name__ == "__main__":
    main()
