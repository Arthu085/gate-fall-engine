from pathlib import Path

import h5py
import numpy as np


def _write_synthetic_pose(
    pose_root: Path,
    video_id: str,
    k: int = 3,
    *,
    keypoints: np.ndarray | None = None,
    bbox: np.ndarray | None = None,
    person_found: np.ndarray | None = None,
) -> Path:
    """Escreve um artefato de pose sintético.

    Sem os parâmetros opcionais, reproduz exatamente o artefato usado pelas
    checagens já existentes; com eles, serve de fixture para cenários de
    detecção intermitente.
    """
    env, _, video_name = video_id.partition("/")
    path = pose_root / env / f"{video_name}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)

    if keypoints is None:
        keypoints = np.zeros((k, 17, 3), dtype=np.float32)
        keypoints[:, :, 0] = np.linspace(12.0, 28.0, 17, dtype=np.float32)
        keypoints[:, :, 1] = np.linspace(24.0, 56.0, 17, dtype=np.float32)
        keypoints[:, :, 2] = 0.9
    if bbox is None:
        bbox = np.tile(
            np.array([[10.0, 20.0, 30.0, 60.0]], dtype=np.float32), (k, 1)
        )
    if person_found is None:
        person_found = np.ones((k,), dtype=np.bool_)

    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("keypoints", data=keypoints)
        h5_file.create_dataset("bbox", data=bbox)
        h5_file.create_dataset("person_found", data=person_found)
        h5_file.attrs["K"] = k
        h5_file.attrs["width"] = 100
        h5_file.attrs["height"] = 200
    return path
