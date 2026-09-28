"""Validação semântica dos artefatos persistidos de treino da arma B1."""

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from gatefall.train.baseline_b1.config import B1TrainConfig, load_config
from gatefall.train.baseline_b1.model import B1AdaptiveGateClassifier
from gatefall.train.shared.gated_artifacts import (
    REQUIRED_GATED_TRAINING_ARTIFACTS,
    validate_gated_training_metrics as validate_b1_training_metrics,
)

REQUIRED_B1_TRAINING_ARTIFACTS = REQUIRED_GATED_TRAINING_ARTIFACTS


def load_compatible_b1_checkpoint(path: Path, config: B1TrainConfig) -> B1AdaptiveGateClassifier:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint não contém um state_dict")
        model = B1AdaptiveGateClassifier(
            channels=config.channels,
            kernel_size=config.kernel_size,
            dilations=config.dilations,
            dropout=config.dropout,
            num_classes=config.num_classes,
        )
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(
            f"checkpoint.pt não é carregável/compatível com config.yaml: {exc}"
        ) from exc
    return model


def validate_b1_training_run(
    run_dir: Path,
    expected_config: B1TrainConfig | None = None,
    fields_allowed_to_differ: frozenset[str] = frozenset(),
) -> B1TrainConfig:
    present = [
        name for name in REQUIRED_B1_TRAINING_ARTIFACTS if (run_dir / name).is_file()
    ]
    if len(present) != len(REQUIRED_B1_TRAINING_ARTIFACTS):
        missing = [
            name for name in REQUIRED_B1_TRAINING_ARTIFACTS if name not in present
        ]
        raise RuntimeError(
            f"run parcial em {run_dir}: artefatos ausentes: {', '.join(missing)}"
        )

    try:
        config = load_config(run_dir / "config.yaml")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"config.yaml inválido em {run_dir}: {exc}") from exc
    if expected_config is not None:
        expected_dict = expected_config.to_dict()
        actual_dict = config.to_dict()
        disallowed_diffs = sorted(
            field
            for field in expected_dict
            if field not in fields_allowed_to_differ
            and actual_dict.get(field) != expected_dict[field]
        )
        if disallowed_diffs:
            raise RuntimeError(
                f"config.yaml em {run_dir} não corresponde à configuração "
                f"solicitada: campo(s) divergente(s): {', '.join(disallowed_diffs)}"
            )

    try:
        with (run_dir / "metrics.json").open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        validate_b1_training_metrics(
            metrics,
            config,
            config_path=run_dir / "config.yaml",
            checkpoint_path=run_dir / "checkpoint.pt",
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"metrics.json inválido em {run_dir}: {exc}") from exc

    try:
        load_compatible_b1_checkpoint(run_dir / "checkpoint.pt", config)
    except ValueError as exc:
        raise RuntimeError(f"checkpoint.pt inválido em {run_dir}: {exc}") from exc
    return config
