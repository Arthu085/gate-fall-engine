"""Padronização (z-score) por dimensão das features DINOv3, treinada apenas no split de treino.

A estatística é congelada por *fonte* de feature (dinov3), no mesmo espírito de
`features/standardization.py` para pose: qualquer arma que consuma este vetor
de 1536 dimensões reusa o mesmo arquivo de estatísticas. Diferente da pose,
não há bloco de confiança a excluir aqui — DINOv3 não expõe um canal de
confiança por dimensão, então não existe `excluded_mask`.
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

SOURCE_NAME = "dinov3"
FEATURE_DIM = 1536


@dataclass
class Dinov3StandardizationStats:
    source: str
    dataset: str
    split: str
    target_fps: float
    window_frames: int
    stride: int
    window_count: int
    feature_dim: int
    mean: list[float]
    std: list[float]
    guarded_count: int
    guarded_mask: list[bool]
    frames_hash: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Dinov3StandardizationStats":
        return Dinov3StandardizationStats(**data)


def stale_stats_mismatches(
    stats: Dinov3StandardizationStats, dataset_name: str | None = None
) -> list[str]:
    """Compara `stats` persistidas com o layout DINOv3 atual.

    Retorna os nomes dos campos divergentes; lista vazia significa que `stats`
    ainda descreve o layout de feature atual. Quando `dataset_name` é
    informado, também sinaliza estatísticas ajustadas para outro dataset.
    """
    mismatches: list[str] = []
    if stats.feature_dim != FEATURE_DIM:
        mismatches.append("feature_dim")
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
    stats: Dinov3StandardizationStats, dataset_name: str | None = None
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
            "layout das estatísticas DINOv3 incompatível com as features "
            "atuais: " + ", ".join(unique)
        )


def validate_stats_freshness(stats: Dinov3StandardizationStats, frames_path: Path) -> None:
    """Garante que `stats` foram ajustadas sobre o `frames.csv`/parquet atual.

    `report_stats_hash` é o hash persistido junto das estatísticas;
    `current_frames_hash` é o hash do arquivo de frames vivo agora.
    """
    current_frames_hash = sha256_file(frames_path)
    if stats.frames_hash != current_frames_hash:
        raise ValueError(
            "estatísticas DINOv3 obsoletas: frames_hash persistido "
            f"({stats.frames_hash}) diverge do frames_hash atual de "
            f"{frames_path} ({current_frames_hash}); rode "
            "`standardize_dinov3 build --force` antes de treinar"
        )


def _accumulate_train_statistics(
    dataset: WindowSource,
) -> tuple[int, np.ndarray, np.ndarray]:
    count = 0
    sum_ = np.zeros(FEATURE_DIM, dtype=np.float64)
    sumsq = np.zeros(FEATURE_DIM, dtype=np.float64)
    for i in range(len(dataset)):
        window, _, _ = dataset[i]
        window64 = window.astype(np.float64)
        sum_ += window64.sum(axis=0)
        sumsq += np.square(window64).sum(axis=0)
        count += window64.shape[0]
    return count, sum_, sumsq


def compute_train_stats(
    dataset: WindowSource, frames_path: Path, dataset_name: str, stride: int = TRAIN_STRIDE
) -> Dinov3StandardizationStats:
    window_count = len(dataset)

    count, sum_, sumsq = _accumulate_train_statistics(dataset)
    mean, std = mean_std_from_accumulators(count, sum_, sumsq)

    guarded_mask = std < GUARD_STD_THRESHOLD
    guarded_count = int(guarded_mask.sum())

    mean = mean.copy()
    std = std.copy()
    mean[guarded_mask] = 0.0
    std[guarded_mask] = 1.0

    frames_hash = sha256_file(frames_path)

    return Dinov3StandardizationStats(
        source=SOURCE_NAME,
        dataset=dataset_name,
        split=TRAIN_SPLIT,
        target_fps=TARGET_FPS,
        window_frames=WINDOW_FRAMES,
        stride=stride,
        window_count=window_count,
        feature_dim=FEATURE_DIM,
        mean=mean.tolist(),
        std=std.tolist(),
        guarded_count=guarded_count,
        guarded_mask=guarded_mask.tolist(),
        frames_hash=frames_hash,
    )


def apply_standardization(x: np.ndarray, stats: Dinov3StandardizationStats) -> np.ndarray:
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


def save_stats(stats: Dinov3StandardizationStats, path: Path, force: bool) -> bool:
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

    print(f"{path}: estatísticas DINOv3 gravadas, {stats.feature_dim} dimensões")
    return True


def load_stats(path: Path) -> Dinov3StandardizationStats:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Dinov3StandardizationStats.from_dict(data)
