"""Configuração de treino da arma B1 (fusão adaptativa por gate escalar),
persistida como receita reproduzível por run.

Invariante experimental #1: A, B0, B1 e as demais armas só podem diferir no
vetor de feature por timestep — todo campo da receita compartilhada abaixo é
copiado campo a campo de `BASELINE_A_CONFIG` (train/config.py). Só os campos
exclusivos de auditoria da fusão adaptativa (dimensões, parametrização do
gate, caminhos e hashes das fontes de estatística e dos sidecars de
qualidade) são específicos do B1.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from gatefall.train.config import BASELINE_A_CONFIG


@dataclass
class B1TrainConfig:
    run_name: str
    arm: str
    seed: int
    window_frames: int
    train_stride: int
    eval_stride: int
    num_classes: int
    kernel_size: int
    dilations: list[int]
    channels: list[int]
    dropout: float
    receptive_field: int
    optimizer_name: str
    lr: float
    weight_decay: float
    grad_clip_norm: float
    lr_schedule_name: str
    batch_size: int
    epochs: int
    loss_name: str
    class_weighted: bool
    pose_dim: int
    visual_dim: int
    projection_dim: int
    fused_dim: int
    gate_input_dim: int
    gate_output_dim: int
    gate_activation: str
    gate_weighted_branch: str
    pose_standardization_stats_path: str
    pose_standardization_stats_sha256: str
    visual_standardization_stats_path: str
    visual_standardization_stats_sha256: str
    quality_features_path: str
    quality_features_sha256: str
    trainable_param_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "B1TrainConfig":
        return B1TrainConfig(**data)


B1_ADAPTIVE_GATE_CONFIG = B1TrainConfig(
    run_name="b1_adaptive_gate",
    arm="B1",
    seed=BASELINE_A_CONFIG.seed,
    window_frames=BASELINE_A_CONFIG.window_frames,
    train_stride=BASELINE_A_CONFIG.train_stride,
    eval_stride=BASELINE_A_CONFIG.eval_stride,
    num_classes=BASELINE_A_CONFIG.num_classes,
    kernel_size=BASELINE_A_CONFIG.kernel_size,
    dilations=BASELINE_A_CONFIG.dilations,
    channels=BASELINE_A_CONFIG.channels,
    dropout=BASELINE_A_CONFIG.dropout,
    receptive_field=BASELINE_A_CONFIG.receptive_field,
    optimizer_name=BASELINE_A_CONFIG.optimizer_name,
    lr=BASELINE_A_CONFIG.lr,
    weight_decay=BASELINE_A_CONFIG.weight_decay,
    grad_clip_norm=BASELINE_A_CONFIG.grad_clip_norm,
    lr_schedule_name=BASELINE_A_CONFIG.lr_schedule_name,
    batch_size=BASELINE_A_CONFIG.batch_size,
    epochs=BASELINE_A_CONFIG.epochs,
    loss_name=BASELINE_A_CONFIG.loss_name,
    class_weighted=BASELINE_A_CONFIG.class_weighted,
    pose_dim=134,
    visual_dim=1536,
    projection_dim=128,
    fused_dim=256,
    gate_input_dim=2,
    gate_output_dim=1,
    gate_activation="sigmoid",
    gate_weighted_branch="pose",
    pose_standardization_stats_path="",
    pose_standardization_stats_sha256="",
    visual_standardization_stats_path="",
    visual_standardization_stats_sha256="",
    quality_features_path="",
    quality_features_sha256="",
    trainable_param_count=0,
)


def save_config(config: B1TrainConfig, path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"skip {path} (já existe, use --force para sobrescrever)")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    os.replace(tmp_path, path)

    with path.open("r", encoding="utf-8") as f:
        read_back = yaml.safe_load(f)
    if read_back != data:
        raise RuntimeError(
            f"verificação de leitura pós-gravação falhou para {path}: conteúdo "
            "lido não bate byte a byte com o conteúdo gravado"
        )

    print(f"{path}: configuração de treino B1 gravada (run_name={config.run_name})")
    return True


def load_config(path: Path) -> B1TrainConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return B1TrainConfig.from_dict(data)
