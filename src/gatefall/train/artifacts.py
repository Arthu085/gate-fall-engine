"""Validação semântica dos artefatos persistidos de treino."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from gatefall.hashing import sha256_file
from gatefall.train.config import TrainConfig, load_config
from gatefall.train.metrics import RESTRICTED_CLASSES
from gatefall.train.tcn import TCNClassifier

REQUIRED_TRAINING_ARTIFACTS = ("config.yaml", "metrics.json", "checkpoint.pt")


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} deve ser um objeto")
    return value


def validate_training_metrics(
    data: object,
    config: TrainConfig,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> None:
    metrics = _require_mapping(data, "metrics.json")
    if metrics.get("run_name") != config.run_name:
        raise ValueError("metrics.json: run_name diverge de config.yaml")
    if metrics.get("epochs_trained") != config.epochs:
        raise ValueError("metrics.json: epochs_trained diverge de config.yaml")
    history = metrics.get("history")
    if not isinstance(history, list) or len(history) != config.epochs:
        raise ValueError(
            f"metrics.json: history deve conter {config.epochs} épocas"
        )
    for expected_epoch, entry in enumerate(history, start=1):
        epoch = _require_mapping(entry, f"metrics.json.history[{expected_epoch - 1}]")
        if epoch.get("epoch") != expected_epoch:
            raise ValueError("metrics.json: sequência de épocas incompatível")
        for field in ("train_loss", "val_macro_f1_restricted"):
            value = epoch.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"metrics.json: {field} deve ser finito")
    if metrics.get("restricted_classes") != RESTRICTED_CLASSES:
        raise ValueError("metrics.json: restricted_classes incompatível")
    expected_excluded = [
        index for index in range(config.num_classes) if index not in RESTRICTED_CLASSES
    ]
    if metrics.get("excluded_classes") != expected_excluded:
        raise ValueError("metrics.json: excluded_classes incompatível")
    if config_path is not None and metrics.get("config_sha256") != sha256_file(
        config_path
    ):
        raise ValueError("metrics.json: config_sha256 incompatível")
    if checkpoint_path is not None and metrics.get(
        "checkpoint_sha256"
    ) != sha256_file(checkpoint_path):
        raise ValueError("metrics.json: checkpoint_sha256 incompatível")

    final = _require_mapping(metrics.get("final"), "metrics.json.final")
    for split in ("train", "val", "test"):
        split_metrics = _require_mapping(
            final.get(split), f"metrics.json.final.{split}"
        )
        macro_f1 = split_metrics.get("macro_f1_restricted")
        if not isinstance(macro_f1, (int, float)) or not math.isfinite(macro_f1):
            raise ValueError(
                f"metrics.json.final.{split}.macro_f1_restricted deve ser numérico"
            )
        f1_by_class = _require_mapping(
            split_metrics.get("f1_by_class"),
            f"metrics.json.final.{split}.f1_by_class",
        )
        if set(f1_by_class) != {str(index) for index in RESTRICTED_CLASSES}:
            raise ValueError(
                f"metrics.json.final.{split}.f1_by_class tem classes incompatíveis"
            )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in f1_by_class.values()
        ):
            raise ValueError(
                f"metrics.json.final.{split}.f1_by_class deve ser numérico e finito"
            )
        support = _require_mapping(
            split_metrics.get("support"), f"metrics.json.final.{split}.support"
        )
        if len(support) != config.num_classes or any(
            not isinstance(value, int) or value < 0 for value in support.values()
        ):
            raise ValueError(
                f"metrics.json.final.{split}.support deve conter "
                f"{config.num_classes} contagens não negativas"
            )


def load_compatible_checkpoint(path: Path, config: TrainConfig) -> TCNClassifier:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint não contém um state_dict")
        model = TCNClassifier(
            input_dim=config.input_dim,
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


def validate_training_run(
    run_dir: Path, expected_config: TrainConfig | None = None
) -> TrainConfig:
    present = [
        name for name in REQUIRED_TRAINING_ARTIFACTS if (run_dir / name).is_file()
    ]
    if len(present) != len(REQUIRED_TRAINING_ARTIFACTS):
        missing = [
            name for name in REQUIRED_TRAINING_ARTIFACTS if name not in present
        ]
        raise RuntimeError(
            f"run parcial em {run_dir}: artefatos ausentes: {', '.join(missing)}"
        )

    try:
        config = load_config(run_dir / "config.yaml")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"config.yaml inválido em {run_dir}: {exc}") from exc
    if expected_config is not None and config != expected_config:
        raise RuntimeError(
            f"config.yaml em {run_dir} não corresponde à configuração solicitada"
        )

    try:
        with (run_dir / "metrics.json").open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        validate_training_metrics(
            metrics,
            config,
            config_path=run_dir / "config.yaml",
            checkpoint_path=run_dir / "checkpoint.pt",
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"metrics.json inválido em {run_dir}: {exc}") from exc

    try:
        load_compatible_checkpoint(run_dir / "checkpoint.pt", config)
    except ValueError as exc:
        raise RuntimeError(f"checkpoint.pt inválido em {run_dir}: {exc}") from exc
    return config
