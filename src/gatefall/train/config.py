"""Configuração de treino da TCN, persistida como receita reproduzível por run."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from gatefall.config import EVAL_STRIDE, NUM_CLASSES, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.pose.kinematics import EXPECTED_D
from gatefall.train.tcn import receptive_field


@dataclass
class TrainConfig:
    run_name: str
    arm: str
    feature_source: str
    seed: int
    input_dim: int
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
    standardization_stats_path: str
    standardization_stats_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "TrainConfig":
        return TrainConfig(**data)


_BASELINE_A_KERNEL_SIZE = 3
_BASELINE_A_DILATIONS = [1, 2, 4]

BASELINE_A_CONFIG = TrainConfig(
    run_name="baseline_a",
    arm="A",
    feature_source="pose",
    seed=42,
    input_dim=EXPECTED_D,
    window_frames=WINDOW_FRAMES,
    train_stride=TRAIN_STRIDE,
    eval_stride=EVAL_STRIDE,
    num_classes=NUM_CLASSES,
    kernel_size=_BASELINE_A_KERNEL_SIZE,
    dilations=_BASELINE_A_DILATIONS,
    channels=[32, 32, 32],
    dropout=0.3,
    receptive_field=receptive_field(_BASELINE_A_KERNEL_SIZE, _BASELINE_A_DILATIONS),
    optimizer_name="adamw",
    lr=1e-3,
    weight_decay=1e-2,
    grad_clip_norm=1.0,
    lr_schedule_name="cosine",
    batch_size=64,
    epochs=30,
    loss_name="cross_entropy",
    class_weighted=True,
    standardization_stats_path="",
    standardization_stats_sha256="",
)


def save_config(config: TrainConfig, path: Path, force: bool) -> bool:
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

    print(f"{path}: configuração de treino gravada (run_name={config.run_name})")
    return True


def load_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TrainConfig.from_dict(data)
