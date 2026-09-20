"""Gravação atômica e leitura dos proxies de qualidade por vídeo em HDF5.

Espelha `dinov3/storage.py`: um arquivo por vídeo, nunca um por quadro
(`CLAUDE.md`, invariante 5). O módulo não importa nenhum backbone — treino e
avaliação leem apenas o que a extração offline já persistiu.
"""

import hashlib
import os
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from gatefall.hashing import sha256_file

# Ordem de canal do dataset `quality [K, 2]`: coluna 0 = q_pose, coluna 1 =
# q_visual. Essa ordem é contrato persistido e é consumida diretamente pelo
# gate adaptativo do B1 (`AdaptiveGate`), que espera exatamente esse par.
QUALITY_CHANNEL_NAMES: tuple[str, ...] = ("q_pose", "q_visual")
QUALITY_CHANNELS = len(QUALITY_CHANNEL_NAMES)

PROVENANCE_ATTR_NAMES: tuple[str, ...] = (
    "channel_names",
    "target_fps",
    "pose_quality_source",
    "visual_quality_source",
    "pose_source_sha256",
    "resize_height",
    "resize_width",
)


class QualityStorageError(Exception):
    """Verificação pós-escrita divergente do arquivo de qualidade."""


def quality_path(video_id: str, *, quality_root: Path) -> Path:
    env, _, video_name = video_id.partition("/")
    return quality_root / env / f"{video_name}.h5"


def write_quality_atomic(
    path: Path, quality: np.ndarray, attrs: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with h5py.File(tmp_path, "w") as h5_file:
        h5_file.create_dataset("quality", data=quality)
        for key, value in attrs.items():
            h5_file.attrs[key] = value
    os.replace(tmp_path, path)


def _is_array_like(value: object) -> bool:
    return isinstance(value, (np.ndarray, tuple, list))


def attrs_equal(actual: object, expected: object) -> bool:
    if _is_array_like(expected):
        return bool(np.array_equal(cast(np.ndarray, actual), cast(np.ndarray, expected)))
    return bool(actual == expected)


def read_quality(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5_file:
        return cast(h5py.Dataset, h5_file["quality"])[()]


def read_quality_shape_dtype(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with h5py.File(path, "r") as h5_file:
        dataset = cast(h5py.Dataset, h5_file["quality"])
        return dataset.shape, dataset.dtype


def validate_existing_file(
    path: Path, *, expected_k: int, expected_attrs: dict[str, object]
) -> list[str]:
    try:
        shape, dtype = read_quality_shape_dtype(path)
    except (OSError, KeyError):
        return ["arquivo corrompido ou ilegível"]

    reasons: list[str] = []
    if shape != (expected_k, QUALITY_CHANNELS):
        reasons.append(
            f"shape {shape} diverge do esperado ({expected_k}, {QUALITY_CHANNELS})"
        )
    if dtype != np.float32:
        reasons.append(f"dtype {dtype} diverge do esperado float32")

    with h5py.File(path, "r") as h5_file:
        for key, expected_value in expected_attrs.items():
            if key not in h5_file.attrs:
                reasons.append(f"atributo '{key}' ausente")
                continue
            if not attrs_equal(h5_file.attrs[key], expected_value):
                reasons.append(f"atributo '{key}' divergente")

    return reasons


def verify_written_file(
    path: Path, *, quality: np.ndarray, attrs: dict[str, object]
) -> None:
    with h5py.File(path, "r") as h5_file:
        dataset = cast(h5py.Dataset, h5_file["quality"])
        if dataset.shape != quality.shape or dataset.dtype != quality.dtype:
            raise QualityStorageError(
                f"\nquality extract FALHOU: dataset 'quality' relido de {path} "
                "diverge do esperado (shape/dtype)"
            )
        if not np.array_equal(dataset[()], quality):
            raise QualityStorageError(
                f"\nquality extract FALHOU: dataset 'quality' relido de {path} "
                "diverge dos valores gravados"
            )
        for key, expected_value in attrs.items():
            if key not in h5_file.attrs:
                raise QualityStorageError(
                    f"\nquality extract FALHOU: atributo '{key}' ausente em {path}"
                )
            if not attrs_equal(h5_file.attrs[key], expected_value):
                raise QualityStorageError(
                    f"\nquality extract FALHOU: atributo '{key}' relido de "
                    f"{path} diverge do gravado"
                )


def quality_set_sha256(video_ids: list[str], *, quality_root: Path) -> str:
    """Digest determinístico do conjunto de sidecars de qualidade.

    A qualidade é persistida por vídeo, então não existe um único arquivo a
    hashear: o digest é o sha256 das linhas `<video_id> <sha256 do arquivo>`
    ordenadas por `video_id`, o que torna a receita do B1 sensível a qualquer
    reextração de qualquer vídeo.
    """
    digest = hashlib.sha256()
    for video_id in sorted(video_ids):
        path = quality_path(video_id, quality_root=quality_root)
        if not path.exists():
            raise FileNotFoundError(
                f"sidecar de qualidade não encontrado: {path}; rode "
                "`uv run python -m gatefall.features.quality_extract extract-all` "
                "primeiro"
            )
        digest.update(f"{video_id} {sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()
