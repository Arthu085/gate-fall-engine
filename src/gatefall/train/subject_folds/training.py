"""Treino do braço A numa dobra por sujeito com a receita e o engine congelados.

O treino usa só o treino de desenvolvimento da dobra, em `TRAIN_STRIDE`; o
grupo de sujeitos retido é avaliado em `EVAL_STRIDE` e ocupa as posições
`val` e `test` do engine, que só registra a curva de `val` por época e guarda
o checkpoint da última época. Nenhum split oficial além de `train` é lido.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.pose.kinematics import POSE_FEATURE_DIM
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.baseline_a.engine import run_training
from gatefall.train.internal_folds.folds import DEVELOPMENT_SPLIT, RETAINED_SPLIT, FeatureLoader, restricted_loader
from gatefall.train.internal_folds.training import guard_existing_fold_run, publish_fold_run_document
from gatefall.train.subject_folds.folds import (
    ARMS,
    CHECKPOINT_POLICY,
    DATASET,
    EVALUATION_SPLIT_MAPPING,
    PROTOCOL,
    SEEDS,
    STAGE,
    SubjectFoldPartition,
)
from gatefall.train.subject_folds.stats import OFFICIAL_POSE_STATS, FoldPoseStats


def resolve_fold_config(seed: int, stats_path: Path, stats_sha256: str) -> TrainConfig:
    if seed not in SEEDS:
        raise ValueError(f"seed não suportada nas dobras por sujeito: {seed}; opções: {list(SEEDS)}")
    if stats_path.resolve() == OFFICIAL_POSE_STATS.resolve():
        raise ValueError("as dobras por sujeito não podem usar a estatística oficial de pose do Le2i-CS")
    return replace(
        BASELINE_A_CONFIG,
        seed=seed,
        standardization_stats_path=str(stats_path),
        standardization_stats_sha256=stats_sha256,
    )


def fold_run_document(
    fold_record: dict[str, object],
    arm: str,
    seed: int,
    stats_sha256: str,
    input_sha256: dict[str, str],
    pose_features_sha256: str,
    membership_sha256: str,
) -> dict[str, object]:
    if arm not in ARMS:
        raise ValueError(f"arma não suportada nas dobras por sujeito: {arm!r}; opções: {', '.join(ARMS)}")
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "canonical": False,
        "dataset": DATASET,
        "fold": fold_record["fold"],
        "arm": arm,
        "seed": seed,
        "retained_subjects": fold_record["retained_subjects"],
        "development_subjects": fold_record["development_subjects"],
        "evaluation_split_mapping": EVALUATION_SPLIT_MAPPING,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "input_sha256": input_sha256,
        "pose_features_sha256": pose_features_sha256,
        "membership_sha256": membership_sha256,
        "standardization_stats_sha256": {"pose": stats_sha256},
    }


def train_fold_run(
    partition: SubjectFoldPartition,
    seed: int,
    stats: FoldPoseStats,
    loader: FeatureLoader,
    document: dict[str, object],
    run_dir: Path,
    force: bool,
) -> None:
    config = resolve_fold_config(seed, stats.path, stats.sha256)
    guard_existing_fold_run(run_dir, document, force)
    context = f"{partition.fold}/{document['arm']}"
    train_source = PoseWindowDataset(
        partition.frames,
        DEVELOPMENT_SPLIT,
        TRAIN_STRIDE,
        restricted_loader(loader, partition.development_video_ids, f"{context}/{DEVELOPMENT_SPLIT}"),
    )
    if len(train_source) != stats.window_count:
        raise RuntimeError(
            f"{partition.fold}: {len(train_source)} janelas de treino, mas as estatísticas foram ajustadas em "
            f"{stats.window_count}"
        )
    retained_source = PoseWindowDataset(
        partition.frames,
        RETAINED_SPLIT,
        EVAL_STRIDE,
        restricted_loader(loader, partition.retained_video_ids, f"{context}/{RETAINED_SPLIT}"),
    )
    if len(retained_source) != cast(int, partition.membership_record["retained_windows_eval_stride"]):
        raise RuntimeError(f"{partition.fold}: janelas do retido divergem da composição da dobra")
    metrics = run_training(
        input_dim=POSE_FEATURE_DIM,
        train_source=train_source,
        val_source=retained_source,
        test_source=retained_source,
        stats=stats.stats,
        config=config,
        run_dir=run_dir,
        force=force,
        label_names=LE2I_LABEL_NAMES,
    )
    if metrics is not None:
        publish_fold_run_document(run_dir, document)
