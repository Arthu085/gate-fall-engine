"""Treino de A, B0 e B1 numa dobra interna com a receita e o engine congelados.

O treino usa só o treino de desenvolvimento da dobra, em `TRAIN_STRIDE`; o
ambiente retido é avaliado em `EVAL_STRIDE` e ocupa as posições `val` e
`test` do engine, que só registra a curva de `val` por época e guarda o
checkpoint da última época. Nenhum split oficial além de `train` é lido.
"""

import json
import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.fusion_dataset import FusionWindowDataset
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM
from gatefall.train.baseline_a.artifacts import validate_training_run
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.baseline_a.engine import run_training
from gatefall.train.baseline_b0.artifacts import validate_b0_training_run
from gatefall.train.baseline_b0.config import B0TrainConfig
from gatefall.train.baseline_b0.engine import run_b0_training
from gatefall.train.baseline_b0.run import resolve_b0_config
from gatefall.train.baseline_b1.artifacts import validate_b1_training_run
from gatefall.train.baseline_b1.config import B1TrainConfig
from gatefall.train.baseline_b1.engine import run_b1_training
from gatefall.train.baseline_b1.run import resolve_b1_config
from gatefall.train.internal_folds.folds import (
    ARMS,
    CHECKPOINT_POLICY,
    DATASET,
    DEVELOPMENT_SPLIT,
    EVALUATION_SPLIT_MAPPING,
    PROTOCOL,
    RETAINED_SPLIT,
    FeatureLoader,
    FoldPartition,
    restricted_loader,
)
from gatefall.train.internal_folds.stats import FoldStats, StatsFiles

FOLD_RUN_FILE = "fold_run.json"
RUN_ARTIFACTS = ("config.yaml", "metrics.json", "checkpoint.pt")

FoldConfig = TrainConfig | B0TrainConfig | B1TrainConfig


@dataclass(frozen=True)
class FeatureLoaders:
    pose: FeatureLoader
    visual: FeatureLoader
    quality: FeatureLoader


@dataclass(frozen=True)
class QualityFeatures:
    root: Path
    sha256: str


def resolve_fold_config(arm: str, seed: int, files: StatsFiles, quality: QualityFeatures) -> FoldConfig:
    if arm == "A":
        return replace(
            BASELINE_A_CONFIG,
            seed=seed,
            standardization_stats_path=str(files.pose_path),
            standardization_stats_sha256=files.pose_sha256,
        )
    if arm == "B0":
        return resolve_b0_config(seed, files.pose_path, files.pose_sha256, files.dinov3_path, files.dinov3_sha256)
    if arm == "B1":
        return resolve_b1_config(
            seed,
            files.pose_path,
            files.pose_sha256,
            files.dinov3_path,
            files.dinov3_sha256,
            quality.root,
            quality.sha256,
        )
    raise ValueError(f"arma não suportada nas dobras internas: {arm!r}; opções: {', '.join(ARMS)}")


def validate_fold_run(run_dir: Path, config: FoldConfig) -> FoldConfig:
    if isinstance(config, TrainConfig):
        return validate_training_run(run_dir, expected_config=config)
    allowed = frozenset({"trainable_param_count"})
    if isinstance(config, B0TrainConfig):
        return validate_b0_training_run(run_dir, expected_config=config, fields_allowed_to_differ=allowed)
    return validate_b1_training_run(run_dir, expected_config=config, fields_allowed_to_differ=allowed)


def fold_run_document(
    fold: str, retained_environment: str, development_environments: list[str], arm: str, seed: int, files: StatsFiles
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "canonical": False,
        "dataset": DATASET,
        "fold": fold,
        "arm": arm,
        "seed": seed,
        "retained_environment": retained_environment,
        "development_environments": development_environments,
        "evaluation_split_mapping": EVALUATION_SPLIT_MAPPING,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "standardization_stats_sha256": {"pose": files.pose_sha256, "dinov3": files.dinov3_sha256},
    }


@dataclass(frozen=True)
class _Side:
    split: str
    stride: int
    pose: FeatureLoader
    visual: FeatureLoader
    quality: FeatureLoader


def _sides(partition: FoldPartition, arm: str, loaders: FeatureLoaders) -> tuple[_Side, _Side]:
    def side(split: str, stride: int, video_ids: tuple[str, ...]) -> _Side:
        context = f"{partition.fold}/{arm}/{split}"
        return _Side(
            split=split,
            stride=stride,
            pose=restricted_loader(loaders.pose, video_ids, f"{context}/pose"),
            visual=restricted_loader(loaders.visual, video_ids, f"{context}/dinov3"),
            quality=restricted_loader(loaders.quality, video_ids, f"{context}/qualidade"),
        )

    return (
        side(DEVELOPMENT_SPLIT, TRAIN_STRIDE, partition.development_video_ids),
        side(RETAINED_SPLIT, EVAL_STRIDE, partition.retained_video_ids),
    )


def _check_window_count(partition: FoldPartition, train_windows: int, stats: FoldStats) -> None:
    if train_windows != stats.window_count:
        raise RuntimeError(
            f"{partition.fold}: {train_windows} janelas de treino, mas as estatísticas foram ajustadas em "
            f"{stats.window_count}"
        )


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    return {name: sha256_file(run_dir / name) for name in RUN_ARTIFACTS}


def validate_fold_run_document(run_dir: Path, document: dict[str, object]) -> None:
    """Exige `fold_run.json` íntegro, da dobra pedida e ligado aos artefatos atuais do run."""
    path = run_dir / FOLD_RUN_FILE
    if not path.is_file():
        raise RuntimeError(
            f"{path} ausente: o run não tem proveniência de dobra; use --force para retreiná-lo"
        )
    try:
        with path.open(encoding="utf-8") as stream:
            recorded = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path} ilegível ({exc}); use --force para retreinar o run") from exc
    expected = {**document, "artifacts_sha256": _artifact_hashes(run_dir)}
    if recorded != expected:
        divergent = sorted(
            field for field in set(expected) | set(recorded if isinstance(recorded, dict) else {})
            if not isinstance(recorded, dict) or recorded.get(field) != expected.get(field)
        )
        raise RuntimeError(
            f"{path} diverge da dobra solicitada ou dos artefatos do run ({', '.join(divergent)}); "
            "use --force para retreinar o run"
        )


def publish_fold_run_document(run_dir: Path, document: dict[str, object]) -> None:
    path = run_dir / FOLD_RUN_FILE
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    data = {**document, "artifacts_sha256": _artifact_hashes(run_dir)}
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def guard_existing_fold_run(run_dir: Path, document: dict[str, object], force: bool) -> None:
    # Sem esta guarda, o engine pularia um run canônico completo e o
    # fold_run.json seria criado depois, sem que o run tenha sido treinado
    # nesta dobra. Runs parciais seguem para o engine, que os recusa.
    if force or not all((run_dir / name).is_file() for name in RUN_ARTIFACTS):
        return
    validate_fold_run_document(run_dir, document)


def train_fold_run(
    partition: FoldPartition,
    arm: str,
    seed: int,
    stats: FoldStats,
    quality: QualityFeatures,
    loaders: FeatureLoaders,
    run_dir: Path,
    force: bool,
) -> None:
    config = resolve_fold_config(arm, seed, stats.files, quality)
    document = fold_run_document(
        partition.fold,
        partition.retained_environment,
        list(partition.development_environments),
        arm,
        seed,
        stats.files,
    )
    guard_existing_fold_run(run_dir, document, force)
    development, retained = _sides(partition, arm, loaders)
    frames = partition.frames
    if isinstance(config, TrainConfig):
        train_source = PoseWindowDataset(frames, development.split, development.stride, development.pose)
        _check_window_count(partition, len(train_source), stats)
        retained_source = PoseWindowDataset(frames, retained.split, retained.stride, retained.pose)
        metrics = run_training(
            input_dim=POSE_FEATURE_DIM,
            train_source=train_source,
            val_source=retained_source,
            test_source=retained_source,
            stats=stats.pose,
            config=config,
            run_dir=run_dir,
            force=force,
            label_names=LE2I_LABEL_NAMES,
        )
    elif isinstance(config, B0TrainConfig):
        train_fusion = FusionWindowDataset(
            frames, development.split, development.stride, development.pose, development.visual
        )
        _check_window_count(partition, len(train_fusion), stats)
        retained_fusion = FusionWindowDataset(frames, retained.split, retained.stride, retained.pose, retained.visual)
        metrics = run_b0_training(
            train_source=train_fusion,
            val_source=retained_fusion,
            test_source=retained_fusion,
            pose_stats=stats.pose,
            visual_stats=stats.dinov3,
            config=config,
            run_dir=run_dir,
            force=force,
            label_names=LE2I_LABEL_NAMES,
        )
    else:
        train_gated = GatedFusionWindowDataset(
            frames, development.split, development.stride, development.pose, development.visual, development.quality
        )
        _check_window_count(partition, len(train_gated), stats)
        retained_gated = GatedFusionWindowDataset(
            frames, retained.split, retained.stride, retained.pose, retained.visual, retained.quality
        )
        metrics = run_b1_training(
            train_source=train_gated,
            val_source=retained_gated,
            test_source=retained_gated,
            pose_stats=stats.pose,
            visual_stats=stats.dinov3,
            config=config,
            run_dir=run_dir,
            force=force,
            label_names=LE2I_LABEL_NAMES,
        )
    if metrics is not None:
        publish_fold_run_document(run_dir, document)
