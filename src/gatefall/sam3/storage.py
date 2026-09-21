"""Gravação atômica e verificação de features SAM 3 em HDF5, agrupadas por vídeo.

Espelha `dinov3/storage.py`. O score do SAM fica em um dataset separado de
`v_t` (`sam_score`), nunca dentro do vetor V_t: o contrato de V_t é
congelado e um canal de confiança do detector não pode vazar para dentro
dele.
"""

import os
from pathlib import Path
from typing import cast

import h5py
import numpy as np

PROVENANCE_ATTR_NAMES: tuple[str, ...] = (
    "model_name",
    "text_prompt",
    "sam3_checkpoint_sha256",
    "sam3_runtime_lock_sha256",
    "target_fps",
)


class Sam3StorageError(Exception):
    """Verificação pós-escrita divergente do arquivo de features SAM 3."""


def sam3_path(video_id: str, *, sam3_root: Path) -> Path:
    env, _, video_name = video_id.partition("/")
    return sam3_root / env / f"{video_name}.h5"


def write_sam3_atomic(
    path: Path,
    v_t: np.ndarray,
    sam_score: np.ndarray,
    n_instances: np.ndarray,
    attrs: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with h5py.File(tmp_path, "w") as h5_file:
        h5_file.create_dataset("v_t", data=v_t)
        h5_file.create_dataset("sam_score", data=sam_score)
        h5_file.create_dataset("n_instances", data=n_instances)
        for key, value in attrs.items():
            h5_file.attrs[key] = value
    os.replace(tmp_path, path)


def _is_array_like(value: object) -> bool:
    return isinstance(value, (np.ndarray, tuple, list))


def attrs_equal(actual: object, expected: object) -> bool:
    if _is_array_like(expected):
        return bool(np.array_equal(cast(np.ndarray, actual), cast(np.ndarray, expected)))
    return bool(actual == expected)


def read_v_t(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5_file:
        return cast(h5py.Dataset, h5_file["v_t"])[()]


def read_sam_score(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5_file:
        return cast(h5py.Dataset, h5_file["sam_score"])[()]


def read_n_instances(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5_file:
        return cast(h5py.Dataset, h5_file["n_instances"])[()]


def _read_v_t_shape_dtype(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with h5py.File(path, "r") as h5_file:
        dataset = cast(h5py.Dataset, h5_file["v_t"])
        return dataset.shape, dataset.dtype


def validate_existing_file(
    path: Path,
    *,
    expected_k: int,
    v_t_dim: int,
    expected_attrs: dict[str, object],
) -> list[str]:
    try:
        shape, dtype = _read_v_t_shape_dtype(path)
    except (OSError, KeyError):
        return ["arquivo corrompido ou ilegível"]

    reasons: list[str] = []
    if shape != (expected_k, v_t_dim):
        reasons.append(f"shape {shape} diverge do esperado ({expected_k}, {v_t_dim})")
    if dtype != np.float32:
        reasons.append(f"dtype {dtype} diverge do esperado float32")

    expected_dtypes: dict[str, np.dtype] = {
        "sam_score": np.dtype(np.float32),
        "n_instances": np.dtype(np.int16),
    }

    with h5py.File(path, "r") as h5_file:
        for name, expected_dtype in expected_dtypes.items():
            if name not in h5_file:
                reasons.append(f"dataset '{name}' ausente")
                continue
            dataset = cast(h5py.Dataset, h5_file[name])
            if dataset.shape != (expected_k,):
                reasons.append(
                    f"dataset '{name}' com shape {dataset.shape} diverge do "
                    f"esperado ({expected_k},)"
                )
            if dataset.dtype != expected_dtype:
                reasons.append(
                    f"dataset '{name}' com dtype {dataset.dtype} diverge do "
                    f"esperado {expected_dtype}"
                )

        for key, expected_value in expected_attrs.items():
            if key not in h5_file.attrs:
                reasons.append(f"atributo '{key}' ausente")
                continue
            actual_value = h5_file.attrs[key]
            if not attrs_equal(actual_value, expected_value):
                reasons.append(f"atributo '{key}' divergente")

    return reasons


def verify_written_file(
    path: Path,
    *,
    v_t: np.ndarray,
    sam_score: np.ndarray,
    n_instances: np.ndarray,
    attrs: dict[str, object],
) -> None:
    with h5py.File(path, "r") as h5_file:
        for name, expected_array in (
            ("v_t", v_t),
            ("sam_score", sam_score),
            ("n_instances", n_instances),
        ):
            actual_dataset = cast(h5py.Dataset, h5_file[name])
            if (
                actual_dataset.shape != expected_array.shape
                or actual_dataset.dtype != expected_array.dtype
            ):
                raise Sam3StorageError(
                    f"\nsam3 extract FALHOU: dataset '{name}' relido de {path} "
                    "diverge do esperado (shape/dtype)"
                )
            if not np.array_equal(actual_dataset[()], expected_array):
                raise Sam3StorageError(
                    f"\nsam3 extract FALHOU: dataset '{name}' relido de {path} "
                    "diverge dos valores gravados"
                )

        for key, expected_value in attrs.items():
            if key not in h5_file.attrs:
                raise Sam3StorageError(
                    f"\nsam3 extract FALHOU: atributo '{key}' ausente em {path}"
                )
            actual_value = h5_file.attrs[key]
            if not attrs_equal(actual_value, expected_value):
                raise Sam3StorageError(
                    f"\nsam3 extract FALHOU: atributo '{key}' relido de {path} "
                    f"diverge (esperado {expected_value!r}, encontrado "
                    f"{actual_value!r})"
                )
