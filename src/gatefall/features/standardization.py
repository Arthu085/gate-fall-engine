"""Padronização (z-score) por dimensão das features de pose, treinada apenas no split de treino.

A estatística é congelada por *fonte* de feature (pose), não por arma A/B/C:
qualquer arma que consuma este mesmo vetor de 134 dimensões reusa o mesmo
arquivo de estatísticas, já que a padronização depende só de como a feature é
construída, não de qual cabeça de fusão a consome depois.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from gatefall.config import TARGET_FPS, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_blocks, feature_names

SOURCE_NAME = "pose"
TRAIN_SPLIT = "train"

KP_CONF_BLOCK_NAME = "kp_conf"

GUARD_STD_THRESHOLD = 1e-6


@dataclass
class StandardizationStats:
    source: str
    split: str
    target_fps: float
    window_frames: int
    stride: int
    window_count: int
    feature_dim: int
    feature_names: list[str]
    excluded_mask: list[bool]
    mean: list[float]
    std: list[float]
    guarded_count: int
    guarded_mask: list[bool]
    frames_hash: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "StandardizationStats":
        return StandardizationStats(**data)


class WindowSource(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, object]: ...


def stale_stats_mismatches(stats: StandardizationStats) -> list[str]:
    """Compara `stats` persistidas com a fonte da verdade viva em `kinematics.py`.

    Retorna os nomes dos campos divergentes; lista vazia significa que `stats`
    ainda descreve o layout de feature atual.
    """
    mismatches: list[str] = []
    if stats.feature_names != feature_names():
        mismatches.append("feature_names")
    if stats.feature_dim != POSE_FEATURE_DIM:
        mismatches.append("feature_dim")
    if len(stats.feature_names) != POSE_FEATURE_DIM:
        mismatches.append("len(feature_names)")
    if len(stats.excluded_mask) != POSE_FEATURE_DIM:
        mismatches.append("len(excluded_mask)")
    if stats.stride != TRAIN_STRIDE:
        mismatches.append("stride")
    if stats.source != SOURCE_NAME:
        mismatches.append("source")
    if stats.split != TRAIN_SPLIT:
        mismatches.append("split")
    return mismatches


def validate_stats_layout(stats: StandardizationStats) -> None:
    mismatches = stale_stats_mismatches(stats)
    vector_fields = {
        "mean": len(stats.mean),
        "std": len(stats.std),
        "guarded_mask": len(stats.guarded_mask),
    }
    mismatches.extend(
        name for name, length in vector_fields.items() if length != POSE_FEATURE_DIM
    )
    if mismatches:
        unique = list(dict.fromkeys(mismatches))
        raise ValueError(
            "layout das estatísticas incompatível com as features atuais: "
            + ", ".join(unique)
        )


def excluded_dimension_mask(names: list[str]) -> np.ndarray:
    mask = np.zeros(len(names), dtype=bool)
    found = False
    for name, start, end in feature_blocks():
        if name == KP_CONF_BLOCK_NAME:
            mask[start:end] = True
            found = True
            break
    if not found:
        raise ValueError(f"bloco '{KP_CONF_BLOCK_NAME}' não encontrado em feature_blocks()")
    return mask


def _accumulate_train_statistics(
    dataset: WindowSource,
) -> tuple[int, np.ndarray, np.ndarray]:
    feature_dim = POSE_FEATURE_DIM
    count = 0
    sum_ = np.zeros(feature_dim, dtype=np.float64)
    sumsq = np.zeros(feature_dim, dtype=np.float64)
    for i in range(len(dataset)):
        window, _, _ = dataset[i]
        window64 = window.astype(np.float64)
        sum_ += window64.sum(axis=0)
        sumsq += np.square(window64).sum(axis=0)
        count += window64.shape[0]
    return count, sum_, sumsq


def mean_std_from_accumulators(
    count: int, sum_: np.ndarray, sumsq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = sum_ / count
    variance = sumsq / count - np.square(mean)
    # Clipa ruído numérico negativo (variância populacional nunca é negativa
    # na aritmética exata) antes da raiz.
    std = np.sqrt(np.maximum(variance, 0.0))
    return mean, std


def compute_train_stats(
    dataset: WindowSource, frames_path: Path, stride: int = TRAIN_STRIDE
) -> StandardizationStats:
    names = feature_names()
    if len(names) != POSE_FEATURE_DIM:
        raise RuntimeError(
            f"layout de features inválido: {len(names)} nomes para {POSE_FEATURE_DIM} dimensões"
        )
    window_count = len(dataset)

    count, sum_, sumsq = _accumulate_train_statistics(dataset)
    mean, std = mean_std_from_accumulators(count, sum_, sumsq)

    excluded_mask = excluded_dimension_mask(names)

    guarded_mask = (std < GUARD_STD_THRESHOLD) & ~excluded_mask
    guarded_count = int(guarded_mask.sum())

    mean = mean.copy()
    std = std.copy()
    mean[guarded_mask] = 0.0
    std[guarded_mask] = 1.0
    mean[excluded_mask] = 0.0
    std[excluded_mask] = 1.0

    frames_hash = sha256_file(frames_path)

    return StandardizationStats(
        source=SOURCE_NAME,
        split=TRAIN_SPLIT,
        target_fps=TARGET_FPS,
        window_frames=WINDOW_FRAMES,
        stride=stride,
        window_count=window_count,
        feature_dim=POSE_FEATURE_DIM,
        feature_names=names,
        excluded_mask=excluded_mask.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
        guarded_count=guarded_count,
        guarded_mask=guarded_mask.tolist(),
        frames_hash=frames_hash,
    )


def apply_standardization(x: np.ndarray, stats: StandardizationStats) -> np.ndarray:
    validate_stats_layout(stats)
    mean = np.asarray(stats.mean, dtype=np.float64)
    std = np.asarray(stats.std, dtype=np.float64)
    if x.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"última dimensão de x ({x.shape[-1]}) não bate com feature_dim das "
            f"estatísticas ({mean.shape[0]})"
        )
    x64 = x.astype(np.float64)
    result = (x64 - mean) / std
    return result.astype(np.float32)


def save_stats(stats: StandardizationStats, path: Path, force: bool) -> bool:
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

    print(f"{path}: estatísticas gravadas, {len(data['feature_names'])} dimensões")
    return True


def load_stats(path: Path) -> StandardizationStats:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return StandardizationStats.from_dict(data)
