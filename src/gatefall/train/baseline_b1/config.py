"""Configuração de treino da arma B1 (fusão adaptativa por gate escalar),
persistida como receita reproduzível por run.

Invariante experimental #1: A, B0, B1 e as demais armas só podem diferir no
vetor de feature por timestep — todo campo da receita compartilhada abaixo é
copiado campo a campo de `BASELINE_A_CONFIG` (train/baseline_a/config.py). Só os campos
exclusivos de auditoria da fusão adaptativa (dimensões, parametrização do
gate, caminhos e hashes das fontes de estatística e dos sidecars de
qualidade) são específicos do B1.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from gatefall.train.shared.gated_config import GatedTrainConfig, gated_config_fields, save_config


@dataclass
class B1TrainConfig(GatedTrainConfig):
    @staticmethod
    def from_dict(data: dict) -> "B1TrainConfig":
        return B1TrainConfig(**data)


B1_ADAPTIVE_GATE_CONFIG = B1TrainConfig(
    **gated_config_fields("b1_adaptive_gate", "B1", 1536)
)


def load_config(path: Path) -> B1TrainConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return B1TrainConfig.from_dict(data)
