"""Cache por vídeo, íntegro e publicado somente após conclusão atômica."""

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from gatefall.hashing import sha256_array
from gatefall.train.temporal_rgb.contract import FEATURE_DIM, MODES, digest


def cache_path(root: Path, video_id: str) -> Path:
    return root / f"{digest(video_id)}.h5"


def array_hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: digest({"dtype": str(value.dtype), "shape": list(value.shape), "bytes": sha256_array(value)})
            for name, value in arrays.items()}


def read_cache(path: Path, expected: dict[str, Any], alignment: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as stream:
        if json.loads(str(stream.attrs["provenance"])) != expected:
            raise ValueError(f"proveniência do cache diverge: {path}")
        arrays = {str(name): np.asarray(stream[name]) for name in stream}
        if set(arrays) != set(alignment) | set(MODES):
            raise ValueError(f"cache incompleto: {path}")
        if json.loads(str(stream.attrs["array_sha256"])) != array_hashes(arrays):
            raise ValueError(f"hash das features ou alinhamento diverge: {path}")
    if any(not np.array_equal(arrays[name], value) for name, value in alignment.items()):
        raise ValueError(f"alinhamento do cache diverge: {path}")
    for mode in MODES:
        feature = arrays[mode]
        if feature.shape != (len(alignment["k_end"]), FEATURE_DIM) or feature.dtype != np.float32 or not np.isfinite(feature).all():
            raise ValueError(f"features inválidas: {path}/{mode}")
    return arrays


def publish_cache(
    path: Path, expected: dict[str, Any], alignment: dict[str, np.ndarray],
    produce: Callable[[], dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if path.exists():
        return read_cache(path, expected, alignment)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {**alignment, **produce()}
    descriptor, temporary_name = tempfile.mkstemp(prefix=".partial-", suffix=".h5", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as stream:
            stream.attrs["provenance"] = json.dumps(expected, sort_keys=True, allow_nan=False)
            stream.attrs["array_sha256"] = json.dumps(array_hashes(arrays), sort_keys=True)
            for name, value in arrays.items():
                stream.create_dataset(name, data=value, compression="gzip")
        read_cache(temporary, expected, alignment)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return read_cache(path, expected, alignment)


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
