"""Treino C1 com o mesmo loop determinístico e gate do B1."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

from gatefall.features.sam3_standardization import Sam3StandardizationStats
from gatefall.features.standardization import StandardizationStats
from gatefall.train.baseline_b1.config import B1TrainConfig
from gatefall.train.baseline_b1.engine import _GatedFusionWindowSource, run_b1_training
from gatefall.train.baseline_c1.artifacts import validate_c1_training_run
from gatefall.train.baseline_c1.config import C1TrainConfig


def run_c1_training(
    train_source: _GatedFusionWindowSource,
    val_source: _GatedFusionWindowSource,
    test_source: _GatedFusionWindowSource,
    pose_stats: StandardizationStats,
    visual_stats: Sam3StandardizationStats,
    config: C1TrainConfig,
    run_dir: Path,
    force: bool,
    label_names: tuple[str, ...],
) -> dict | None:
    return run_b1_training(
        train_source,
        val_source,
        test_source,
        pose_stats,
        visual_stats,
        config,
        run_dir,
        force,
        label_names,
        validate_run=cast(Callable[..., B1TrainConfig], validate_c1_training_run),
    )
