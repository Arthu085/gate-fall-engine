"""Carregamento e imputação de pose (YOLO-Pose) a partir de arquivos HDF5.

Política de imputação: forward-fill a partir do último quadro válido e
zero-fill do trecho anterior à primeira detecção — caso que inclui, na sua
forma degenerada, o vídeo inteiro sem nenhuma detecção.

Zero-fill em todo quadro ausente foi descartado porque um quadro zerado
entre dois quadros válidos produz um salto de posição do tamanho do corpo
em 0,1 s — ou seja, um pico espúrio de velocidade/aceleração no dado de
entrada. A perda de pose se concentra nas janelas `fall` (9,6% das janelas
`fall` de treino não têm pose no quadro do rótulo) e em Home_01, onde a
perda é sobretudo flicker quadro a quadro, não blocos longos — cenário em
que o forward-fill preserva a pose sem introduzir esse salto.

Antes da primeira observação, porém, não há pose passada para segurar, e o
back-fill fazia um quadro depender de uma detecção que ainda não havia
acontecido; um detector causal não pode fazer isso. Nesse trecho a ausência
é representada por zeros em coordenadas, descritores de bbox e confiança.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np


@dataclass(frozen=True)
class PoseArrays:
    keypoints: np.ndarray
    bbox: np.ndarray
    person_found: np.ndarray
    k: int
    width: int
    height: int


def first_observed_index(person_found: np.ndarray) -> int:
    """Índice da primeira observação; K quando o vídeo não tem nenhuma."""
    if not np.any(person_found):
        return int(person_found.shape[0])
    return int(np.argmax(person_found))


def pose_path(video_id: str, *, pose_root: Path) -> Path:
    env, _, video_name = video_id.partition("/")
    return pose_root / env / f"{video_name}.h5"


def load_pose(video_id: str, *, pose_root: Path) -> PoseArrays:
    path = pose_path(video_id, pose_root=pose_root)
    if not path.exists():
        raise FileNotFoundError(f"arquivo de pose não encontrado: {path}")
    with h5py.File(path, "r") as h5_file:
        keypoints = cast(h5py.Dataset, h5_file["keypoints"])[()]
        bbox = cast(h5py.Dataset, h5_file["bbox"])[()]
        person_found = cast(h5py.Dataset, h5_file["person_found"])[()]
        k = int(cast(int, h5_file.attrs["K"]))
        width = int(cast(int, h5_file.attrs["width"]))
        height = int(cast(int, h5_file.attrs["height"]))
    if keypoints.shape[0] != k:
        raise ValueError(
            f"artefato de pose incompatível em {path}: keypoints tem "
            f"{keypoints.shape[0]} linhas, atributo K={k}"
        )
    return PoseArrays(
        keypoints=keypoints,
        bbox=bbox,
        person_found=person_found,
        k=k,
        width=width,
        height=height,
    )


def load_person_found(video_id: str, *, pose_root: Path) -> np.ndarray:
    """Lê apenas `person_found`, sem materializar keypoints e bbox."""
    path = pose_path(video_id, pose_root=pose_root)
    if not path.exists():
        raise FileNotFoundError(f"arquivo de pose não encontrado: {path}")
    with h5py.File(path, "r") as h5_file:
        person_found = cast(h5py.Dataset, h5_file["person_found"])[()]
    return person_found


def normalize_keypoints(
    keypoints: np.ndarray, bbox: np.ndarray, person_found: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k = keypoints.shape[0]
    xy = np.full((k, 17, 2), np.nan, dtype=np.float32)
    conf = keypoints[:, :, 2].astype(np.float32)

    box_width = bbox[:, 2] - bbox[:, 0]
    box_height = bbox[:, 3] - bbox[:, 1]
    cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
    cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
    # Escala isotrópica pela diagonal da bbox, não pela altura: a altura
    # colapsa quando a pessoa cai (queda), o que faria as coordenadas
    # normalizadas explodirem justamente na classe de interesse (fall).
    # Dividir x e y pelo mesmo escalar também preserva ângulos exatamente,
    # ao contrário da escala por eixo (largura/altura), que os distorce.
    scale = np.sqrt(box_width**2 + box_height**2)

    valid = person_found
    xy[valid, :, 0] = (keypoints[valid, :, 0] - cx[valid, None]) / scale[valid, None]
    xy[valid, :, 1] = (keypoints[valid, :, 1] - cy[valid, None]) / scale[valid, None]

    return xy, conf


def bbox_descriptors(
    bbox: np.ndarray, person_found: np.ndarray, width: int, height: int
) -> np.ndarray:
    k = bbox.shape[0]
    descriptors = np.full((k, 4), np.nan, dtype=np.float32)

    box_width = bbox[:, 2] - bbox[:, 0]
    box_height = bbox[:, 3] - bbox[:, 1]
    cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
    cy = (bbox[:, 1] + bbox[:, 3]) / 2.0

    valid = person_found
    descriptors[valid, 0] = cx[valid] / width
    descriptors[valid, 1] = cy[valid] / height
    descriptors[valid, 2] = box_width[valid] / width
    descriptors[valid, 3] = box_height[valid] / height

    return descriptors


def impute_missing(
    xy: np.ndarray,
    conf: np.ndarray,
    bbox_desc: np.ndarray,
    person_found: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = xy.shape[0]
    xy_out = xy.copy()
    conf_out = conf.copy()
    bbox_out = bbox_desc.copy()

    prefix = first_observed_index(person_found)
    xy_out[:prefix] = 0.0
    conf_out[:prefix] = 0.0
    bbox_out[:prefix] = 0.0

    # confiança não é preenchida em nenhum dos ramos abaixo: permanece 0.0 em
    # todo quadro sem detecção, para que a pose imputada continue
    # distinguível da pose observada só pelo canal de confiança — sinal
    # reaproveitado depois pelo gating adaptativo.
    last_valid: np.ndarray | None = None
    last_valid_bbox: np.ndarray | None = None
    for i in range(k):
        if person_found[i]:
            last_valid = xy_out[i].copy()
            last_valid_bbox = bbox_out[i].copy()
        elif last_valid is not None:
            xy_out[i] = last_valid
            conf_out[i] = 0.0
            bbox_out[i] = cast(np.ndarray, last_valid_bbox)

    if not (
        np.isfinite(xy_out).all()
        and np.isfinite(conf_out).all()
        and np.isfinite(bbox_out).all()
    ):
        raise ValueError("imputação de pose produziu valor não finito")
    return xy_out, conf_out, bbox_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "selftest",
        help="Roda checagens sintéticas de normalização e imputação de pose",
    )

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.pose.selftests.loading import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
