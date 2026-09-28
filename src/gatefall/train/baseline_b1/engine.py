"""Treino da arma B1 sobre o loop compartilhado de fusão adaptativa."""

from collections.abc import Callable
from pathlib import Path

from gatefall.features.dinov3_standardization import Dinov3StandardizationStats
from gatefall.features.standardization import StandardizationStats
from gatefall.train.baseline_b1.artifacts import validate_b1_training_run
from gatefall.train.baseline_b1.config import B1TrainConfig
from gatefall.train.baseline_b1.model import B1AdaptiveGateClassifier
from gatefall.train.shared.gated_engine import _GatedFusionWindowSource, run_gated_training


def run_b1_training(
    train_source: _GatedFusionWindowSource,
    val_source: _GatedFusionWindowSource,
    test_source: _GatedFusionWindowSource,
    pose_stats: StandardizationStats,
    visual_stats: Dinov3StandardizationStats,
    config: B1TrainConfig,
    run_dir: Path,
    force: bool,
    label_names: tuple[str, ...],
    validate_run: Callable[..., B1TrainConfig] = validate_b1_training_run,
) -> dict | None:
    return run_gated_training(
        train_source,
        val_source,
        test_source,
        pose_stats,
        visual_stats,
        config,
        run_dir,
        force,
        label_names,
        model_type=B1AdaptiveGateClassifier,
        validate_run=validate_run,
    )
