"""Configuração reproduzível da arma C1 com fontes e qualidade auditáveis."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from gatefall.sam3.descriptors import V_T_DIM
from gatefall.train.shared.gated_config import GatedTrainConfig, gated_config_fields
from gatefall.train.shared.gated_config import save_config as save_gated_config


@dataclass
class C1TrainConfig(GatedTrainConfig):
    sam3_features_path: str = ""
    sam3_features_sha256: str = ""
    sam3_provenance: dict[str, str] = field(default_factory=dict)
    pose_features_sha256: str = ""
    pose_quality_source: str = "gatefall.pose.quality.compute_pose_quality"
    visual_quality_source: str = "gatefall.sam3.quality.compute_sam3_quality"

    @staticmethod
    def from_dict(data: dict) -> "C1TrainConfig":
        return C1TrainConfig(**data)


C1_ADAPTIVE_GATE_CONFIG = C1TrainConfig(
    **gated_config_fields("c1_adaptive_gate", "C1", V_T_DIM)
)


def save_config(config: C1TrainConfig, path: Path, force: bool) -> bool:
    return save_gated_config(config, path, force)


def load_config(path: Path) -> C1TrainConfig:
    with path.open(encoding="utf-8") as stream:
        return C1TrainConfig.from_dict(yaml.safe_load(stream))
