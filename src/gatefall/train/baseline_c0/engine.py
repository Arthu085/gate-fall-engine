"""Treino da fusão por concatenação da arma C0."""

from pathlib import Path

from gatefall.features.sam3_standardization import Sam3StandardizationStats
from gatefall.features.sam3_standardization import apply_standardization as apply_visual_standardization
from gatefall.features.standardization import StandardizationStats
from gatefall.train.baseline_c0.artifacts import validate_c0_training_run
from gatefall.train.baseline_c0.config import C0TrainConfig, save_config
from gatefall.train.baseline_c0.model import C0FusionClassifier
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
        visual_stats: Sam3StandardizationStats,
    ) -> None:
        super().__init__(source, pose_stats, visual_stats, apply_visual_standardization)


def run_c0_training(
    train_source: FusionWindowSource,
    val_source: FusionWindowSource,
    test_source: FusionWindowSource,
    pose_stats: StandardizationStats,
    visual_stats: Sam3StandardizationStats,
    config: C0TrainConfig,
    run_dir: Path,
    force: bool,
    label_names: tuple[str, ...],
) -> dict | None:
    return run_concat_training(
        train_source, val_source, test_source, pose_stats, visual_stats,
        config, run_dir, force, label_names,
        dataset_factory=_StandardizedFusionTorchDataset,
        model_factory=C0FusionClassifier,
        save_config=save_config,
        validate_run=validate_c0_training_run,
    )
