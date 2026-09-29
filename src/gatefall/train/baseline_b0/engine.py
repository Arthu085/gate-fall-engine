"""Treino da fusão por concatenação da arma B0."""

from pathlib import Path

from gatefall.features.dinov3_standardization import Dinov3StandardizationStats
from gatefall.features.dinov3_standardization import apply_standardization as apply_visual_standardization
from gatefall.features.standardization import StandardizationStats
from gatefall.train.baseline_b0.artifacts import validate_b0_training_run
from gatefall.train.baseline_b0.config import B0TrainConfig, save_config
from gatefall.train.baseline_b0.model import B0FusionClassifier
from gatefall.train.shared.concat_engine import (
    FusionWindowSource,
    _class_weights,
    _collect_labels,
    _evaluate_split,
    _predict,
    run_concat_training,
)
from gatefall.train.shared.concat_engine import (
    _StandardizedFusionTorchDataset as SharedFusionTorchDataset,
)


_FusionWindowSource = FusionWindowSource


class _StandardizedFusionTorchDataset(SharedFusionTorchDataset):
    def __init__(
        self,
        source: FusionWindowSource,
        pose_stats: StandardizationStats,
        visual_stats: Dinov3StandardizationStats,
    ) -> None:
        super().__init__(source, pose_stats, visual_stats, apply_visual_standardization)


def run_b0_training(
    train_source: FusionWindowSource,
    val_source: FusionWindowSource,
    test_source: FusionWindowSource,
    pose_stats: StandardizationStats,
    visual_stats: Dinov3StandardizationStats,
    config: B0TrainConfig,
    run_dir: Path,
    force: bool,
    label_names: tuple[str, ...],
) -> dict | None:
    return run_concat_training(
        train_source, val_source, test_source, pose_stats, visual_stats,
        config, run_dir, force, label_names,
        dataset_factory=_StandardizedFusionTorchDataset,
        model_factory=B0FusionClassifier,
        save_config=save_config,
        validate_run=validate_b0_training_run,
    )
