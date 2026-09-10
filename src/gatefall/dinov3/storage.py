"""Gravação atômica e verificação de features DINOv3 em HDF5, agrupadas por vídeo."""

import os
from pathlib import Path
from typing import cast

import h5py
import numpy as np


class Dinov3StorageError(Exception):
    """Verificação pós-escrita divergente do arquivo de features DINOv3."""


def dinov3_path(video_id: str, *, dinov3_root: Path) -> Path:
    env, _, video_name = video_id.partition("/")
    return dinov3_root / env / f"{video_name}.h5"


def write_dinov3_features_atomic(
    path: Path, features: np.ndarray, attrs: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with h5py.File(tmp_path, "w") as h5_file:
        h5_file.create_dataset("features", data=features)
        for key, value in attrs.items():
            h5_file.attrs[key] = value
    os.replace(tmp_path, path)


def _is_array_like(value: object) -> bool:
    return isinstance(value, (np.ndarray, tuple, list))


def verify_written_file(
    path: Path, *, features: np.ndarray, attrs: dict[str, object]
) -> None:
    with h5py.File(path, "r") as h5_file:
        actual_dataset = cast(h5py.Dataset, h5_file["features"])
        if actual_dataset.shape != features.shape or actual_dataset.dtype != features.dtype:
            raise Dinov3StorageError(
                f"\ndinov3 extract FALHOU: dataset 'features' relido de {path} "
                "diverge do esperado (shape/dtype)"
            )
        if not np.array_equal(actual_dataset[()], features):
            raise Dinov3StorageError(
                f"\ndinov3 extract FALHOU: dataset 'features' relido de {path} "
                "diverge dos valores gravados"
            )

        for key, expected_value in attrs.items():
            if key not in h5_file.attrs:
                raise Dinov3StorageError(
                    f"\ndinov3 extract FALHOU: atributo '{key}' ausente em {path}"
                )
            actual_value = h5_file.attrs[key]
            if _is_array_like(expected_value):
                matches = np.array_equal(
                    cast(np.ndarray, actual_value), cast(np.ndarray, expected_value)
                )
            else:
                matches = actual_value == expected_value
            if not matches:
                raise Dinov3StorageError(
                    f"\ndinov3 extract FALHOU: atributo '{key}' relido de {path} "
                    f"diverge (esperado {expected_value!r}, encontrado "
                    f"{actual_value!r})"
                )
