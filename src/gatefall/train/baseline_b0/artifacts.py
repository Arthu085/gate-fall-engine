"""Validação dos artefatos de treino da arma B0."""

from pathlib import Path

from gatefall.train.baseline_b0.config import B0TrainConfig, load_config
from gatefall.train.baseline_b0.model import B0FusionClassifier
from gatefall.train.shared.concat_artifacts import (
    REQUIRED_CONCAT_TRAINING_ARTIFACTS,
    load_compatible_concat_checkpoint,
    validate_concat_training_metrics,
    validate_concat_training_run,
)

REQUIRED_B0_TRAINING_ARTIFACTS = REQUIRED_CONCAT_TRAINING_ARTIFACTS


def validate_b0_training_metrics(
    data: object,
    config: B0TrainConfig,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> None:
    validate_concat_training_metrics(data, config, config_path, checkpoint_path)


def load_compatible_b0_checkpoint(path: Path, config: B0TrainConfig) -> B0FusionClassifier:
    return load_compatible_concat_checkpoint(path, config, B0FusionClassifier)


def validate_b0_training_run(
    run_dir: Path,
    expected_config: B0TrainConfig | None = None,
    fields_allowed_to_differ: frozenset[str] = frozenset(),
) -> B0TrainConfig:
    return validate_concat_training_run(
        run_dir,
        expected_config,
        fields_allowed_to_differ,
        load_config=load_config,
        validate_metrics=validate_b0_training_metrics,
        load_checkpoint=load_compatible_b0_checkpoint,
    )
