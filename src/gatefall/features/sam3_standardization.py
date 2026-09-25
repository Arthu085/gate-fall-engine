"""Padronização (z-score) por canal do descritor SAM 3 V_t, treinada apenas no split de treino.

Espelha `dinov3_standardization.py`: a estatística é congelada por *fonte* de
feature (sam3), não por arma. Além do hash de `frames.parquet`, grava o
digest do conjunto de `.h5` do SAM 3 (`sam3_set_sha256`), porque uma
reextração muda V_t sem mudar a tabela de frames.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gatefall.config import TARGET_FPS, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.features.standardization import (
    GUARD_STD_THRESHOLD,
    TRAIN_SPLIT,
    WindowSource,
    mean_std_from_accumulators,
)
from gatefall.hashing import sha256_file
from gatefall.sam3.descriptors import CHANNEL_NAMES, V_T_DIM

SOURCE_NAME = "sam3"
FEATURE_DIM = V_T_DIM


@dataclass
class Sam3StandardizationStats:
    source: str
    dataset: str
    split: str
    target_fps: float
    window_frames: int
    stride: int
    window_count: int
    feature_dim: int
    channel_names: list[str]
    mean: list[float]
    std: list[float]
    guarded_count: int
    guarded_mask: list[bool]
    frames_hash: str
    sam3_features_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Sam3StandardizationStats":
        return Sam3StandardizationStats(**data)


def stale_stats_mismatches(
    stats: Sam3StandardizationStats, dataset_name: str | None = None
) -> list[str]:
    mismatches: list[str] = []
    if stats.feature_dim != FEATURE_DIM:
        mismatches.append("feature_dim")
    if stats.channel_names != list(CHANNEL_NAMES):
        mismatches.append("channel_names")
    if stats.stride != TRAIN_STRIDE:
        mismatches.append("stride")
    if stats.source != SOURCE_NAME:
        mismatches.append("source")
    if stats.split != TRAIN_SPLIT:
        mismatches.append("split")
    if dataset_name is not None and stats.dataset != dataset_name:
        mismatches.append("dataset")
    return mismatches


def validate_stats_layout(
    stats: Sam3StandardizationStats, dataset_name: str | None = None
) -> None:
    mismatches = stale_stats_mismatches(stats, dataset_name)
    vector_fields = {
        "mean": len(stats.mean),
        "std": len(stats.std),
        "guarded_mask": len(stats.guarded_mask),
    }
    mismatches.extend(
        name for name, length in vector_fields.items() if length != FEATURE_DIM
    )
    if mismatches:
        unique = list(dict.fromkeys(mismatches))
        raise ValueError(
            "layout das estatísticas SAM 3 incompatível com o descritor V_t "
            "atual: " + ", ".join(unique)
        )


def validate_stats_freshness(
    stats: Sam3StandardizationStats, frames_path: Path, sam3_features_sha256: str
) -> None:
    current_frames_hash = sha256_file(frames_path)
    if stats.frames_hash != current_frames_hash:
        raise ValueError(
            "estatísticas SAM 3 obsoletas: frames_hash persistido "
            f"({stats.frames_hash}) diverge do frames_hash atual de "
            f"{frames_path} ({current_frames_hash}); rode "
            "`standardize_sam3 build --force` antes de treinar"
        )
    if stats.sam3_features_sha256 != sam3_features_sha256:
        raise ValueError(
            "estatísticas SAM 3 obsoletas: sam3_features_sha256 persistido "
            f"({stats.sam3_features_sha256}) diverge do conjunto de .h5 atual "
            f"({sam3_features_sha256}); rode `standardize_sam3 build --force` "
            "antes de treinar"
        )


def compute_train_stats(
    dataset: WindowSource,
    frames_path: Path,
    dataset_name: str,
    sam3_features_sha256: str,
    stride: int = TRAIN_STRIDE,
) -> Sam3StandardizationStats:
    count = 0
    sum_ = np.zeros(FEATURE_DIM, dtype=np.float64)
    sumsq = np.zeros(FEATURE_DIM, dtype=np.float64)
    for i in range(len(dataset)):
        window, _, _ = dataset[i]
        window64 = window.astype(np.float64)
        sum_ += window64.sum(axis=0)
        sumsq += np.square(window64).sum(axis=0)
        count += window64.shape[0]
    mean, std = mean_std_from_accumulators(count, sum_, sumsq)

    guarded_mask = std < GUARD_STD_THRESHOLD
    mean = mean.copy()
    std = std.copy()
    mean[guarded_mask] = 0.0
    std[guarded_mask] = 1.0

    return Sam3StandardizationStats(
        source=SOURCE_NAME,
        dataset=dataset_name,
        split=TRAIN_SPLIT,
        target_fps=TARGET_FPS,
        window_frames=WINDOW_FRAMES,
        stride=stride,
        window_count=len(dataset),
        feature_dim=FEATURE_DIM,
        channel_names=list(CHANNEL_NAMES),
        mean=mean.tolist(),
        std=std.tolist(),
        guarded_count=int(guarded_mask.sum()),
        guarded_mask=guarded_mask.tolist(),
        frames_hash=sha256_file(frames_path),
        sam3_features_sha256=sam3_features_sha256,
    )


def apply_standardization(x: np.ndarray, stats: Sam3StandardizationStats) -> np.ndarray:
    validate_stats_layout(stats)
    mean = np.asarray(stats.mean, dtype=np.float64)
    std = np.asarray(stats.std, dtype=np.float64)
    if x.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"última dimensão de x ({x.shape[-1]}) não bate com feature_dim das "
            f"estatísticas ({mean.shape[0]})"
        )
    result = (x.astype(np.float64) - mean) / std
    return result.astype(np.float32)


def save_stats(stats: Sam3StandardizationStats, path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"skip {path} (já existe, use --force para sobrescrever)")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    data = stats.to_dict()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

    with path.open("r", encoding="utf-8") as f:
        read_back = json.load(f)
    if read_back != data:
        raise RuntimeError(
            f"verificação de leitura pós-gravação falhou para {path}: conteúdo "
            "lido não bate byte a byte com o conteúdo gravado"
        )

    print(f"{path}: estatísticas SAM 3 gravadas, {stats.feature_dim} canais")
    return True


def load_stats(path: Path) -> Sam3StandardizationStats:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Sam3StandardizationStats.from_dict(data)
