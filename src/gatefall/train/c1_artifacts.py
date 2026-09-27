"""Validação de identidade, métricas e checkpoint dos runs C1."""

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from gatefall.train.b1_artifacts import validate_b1_training_metrics
from gatefall.train.c1_config import C1_ADAPTIVE_GATE_CONFIG, C1TrainConfig, load_config
from gatefall.train.c1_model import C1AdaptiveGateClassifier

REQUIRED_C1_TRAINING_ARTIFACTS = ("config.yaml", "metrics.json", "checkpoint.pt")


def load_compatible_c1_checkpoint(path: Path, config: C1TrainConfig) -> C1AdaptiveGateClassifier:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint não contém um state_dict")
        model = C1AdaptiveGateClassifier(
            channels=config.channels,
            kernel_size=config.kernel_size,
            dilations=config.dilations,
            dropout=config.dropout,
            num_classes=config.num_classes,
        )
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"checkpoint.pt incompatível com config.yaml: {exc}") from exc
    return model


def validate_c1_training_run(
    run_dir: Path,
    expected_config: C1TrainConfig | None = None,
    fields_allowed_to_differ: frozenset[str] = frozenset(),
) -> C1TrainConfig:
    missing = [name for name in REQUIRED_C1_TRAINING_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"run parcial em {run_dir}: artefatos ausentes: {', '.join(missing)}")
    try:
        config = load_config(run_dir / "config.yaml")
        frozen = C1_ADAPTIVE_GATE_CONFIG
        if (config.arm != frozen.arm or config.run_name != frozen.run_name
                or config.visual_dim != frozen.visual_dim
                or config.pose_dim != frozen.pose_dim
                or config.projection_dim != frozen.projection_dim
                or config.fused_dim != frozen.fused_dim
                or config.gate_input_dim != frozen.gate_input_dim
                or config.gate_output_dim != frozen.gate_output_dim
                or config.gate_activation != frozen.gate_activation
                or config.gate_weighted_branch != frozen.gate_weighted_branch
                or config.pose_quality_source != frozen.pose_quality_source
                or config.visual_quality_source != frozen.visual_quality_source):
            raise ValueError("identidade ou arquitetura da arma C1 incompatível")
        required_sources = (
            config.pose_standardization_stats_sha256,
            config.visual_standardization_stats_sha256,
            config.pose_features_sha256,
            config.sam3_features_sha256,
            config.quality_features_sha256,
        )
        if any(not value for value in required_sources) or not config.sam3_provenance:
            raise ValueError("hashes ou proveniência das fontes C1 ausentes")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"config.yaml inválido em {run_dir}: {exc}") from exc
    if expected_config is not None:
        expected = expected_config.to_dict()
        actual = config.to_dict()
        differences = sorted(key for key in expected
                             if key not in fields_allowed_to_differ and actual.get(key) != expected[key])
        if differences:
            raise RuntimeError(f"config.yaml em {run_dir} não corresponde à configuração solicitada: {', '.join(differences)}")
    try:
        with (run_dir / "metrics.json").open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        validate_b1_training_metrics(metrics, config, run_dir / "config.yaml", run_dir / "checkpoint.pt")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"metrics.json inválido em {run_dir}: {exc}") from exc
    try:
        load_compatible_c1_checkpoint(run_dir / "checkpoint.pt", config)
    except ValueError as exc:
        raise RuntimeError(f"checkpoint.pt inválido em {run_dir}: {exc}") from exc
    return config
