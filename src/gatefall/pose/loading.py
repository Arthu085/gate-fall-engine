"""Carregamento e imputação de pose (YOLO-Pose) a partir dos .h5 do Le2i.

Política de imputação: forward-fill a partir do último quadro válido,
back-fill do trecho inicial quando o vídeo começa sem detecção, e
zero-fill apenas quando o vídeo inteiro não tem nenhuma detecção.

Zero-fill em todo quadro ausente foi descartado porque um quadro zerado
entre dois quadros válidos produz um salto de posição do tamanho do corpo
em 0,1 s — ou seja, um pico espúrio de velocidade/aceleração no dado de
entrada. A perda de pose se concentra nas janelas `fall` (9,6% das janelas
`fall` de treino não têm pose no quadro do rótulo) e em Home_01, onde a
perda é sobretudo flicker quadro a quadro, não blocos longos — cenário em
que forward/back-fill preserva a pose sem introduzir esse salto.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from gatefall.pose.extract import POSE_ROOT, _output_path


@dataclass(frozen=True)
class PoseArrays:
    keypoints: np.ndarray
    bbox: np.ndarray
    person_found: np.ndarray
    k: int
    width: int
    height: int


def load_pose(video_id: str) -> PoseArrays:
    path = _output_path(video_id)
    if not path.exists():
        raise FileNotFoundError(f"arquivo de pose não encontrado: {path}")
    with h5py.File(path, "r") as h5_file:
        keypoints = cast(h5py.Dataset, h5_file["keypoints"])[()]
        bbox = cast(h5py.Dataset, h5_file["bbox"])[()]
        person_found = cast(h5py.Dataset, h5_file["person_found"])[()]
        k = int(cast(int, h5_file.attrs["K"]))
        width = int(cast(int, h5_file.attrs["width"]))
        height = int(cast(int, h5_file.attrs["height"]))
    assert keypoints.shape[0] == k
    return PoseArrays(
        keypoints=keypoints,
        bbox=bbox,
        person_found=person_found,
        k=k,
        width=width,
        height=height,
    )


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

    if not np.any(person_found):
        xy_out[:] = 0.0
        # confiança não é preenchida: permanece 0.0 em todo quadro sem
        # detecção, para que a pose imputada continue distinguível da pose
        # observada só pelo canal de confiança — sinal reaproveitado depois
        # pelo gating adaptativo.
        conf_out[:] = 0.0
        bbox_out[:] = 0.0
        return xy_out, conf_out, bbox_out

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

    first_valid_index = int(np.argmax(person_found))
    if first_valid_index > 0:
        xy_out[:first_valid_index] = xy_out[first_valid_index]
        conf_out[:first_valid_index] = 0.0
        bbox_out[:first_valid_index] = bbox_out[first_valid_index]

    assert np.isfinite(xy_out).all()
    assert np.isfinite(conf_out).all()
    assert np.isfinite(bbox_out).all()
    return xy_out, conf_out, bbox_out


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _selftest_normalization() -> bool:
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    keypoints[0, 0] = [15.0, 20.0, 0.9]
    bbox = np.array([[10.0, 10.0, 30.0, 60.0]], dtype=np.float32)
    person_found = np.array([True])

    xy, conf = normalize_keypoints(keypoints, bbox, person_found)
    scale = np.sqrt(20.0**2 + 50.0**2)
    expected_x = (15.0 - 20.0) / scale
    expected_y = (20.0 - 35.0) / scale

    ok = (
        np.isclose(xy[0, 0, 0], expected_x)
        and np.isclose(xy[0, 0, 1], expected_y)
        and np.isclose(conf[0, 0], 0.9)
    )
    return _check("normalize_keypoints: bbox/keypoint conhecido dá valores esperados", ok)


def _selftest_normalization_preserves_angle() -> bool:
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    keypoints[0, 0] = [0.0, 0.0, 0.9]
    keypoints[0, 1] = [10.0, 10.0, 0.9]
    bbox = np.array([[-50.0, -25.0, 50.0, 25.0]], dtype=np.float32)
    person_found = np.array([True])

    xy, _ = normalize_keypoints(keypoints, bbox, person_found)
    vector = xy[0, 1] - xy[0, 0]
    angle_deg = np.degrees(np.arctan2(vector[1], vector[0]))

    ok = np.isclose(angle_deg, 45.0, atol=1e-5)
    return _check(
        "normalize_keypoints: escala isotrópica preserva ângulo (bbox larga não distorce)",
        ok,
    )


def _selftest_forward_fill_single_gap() -> bool:
    xy = np.zeros((3, 17, 2), dtype=np.float32)
    conf = np.full((3, 17), 0.9, dtype=np.float32)
    bbox_desc = np.zeros((3, 4), dtype=np.float32)
    xy[0] = 0.1
    xy[1] = np.nan
    xy[2] = 0.3
    bbox_desc[0] = 0.4
    bbox_desc[1] = np.nan
    bbox_desc[2] = 0.6
    person_found = np.array([True, False, True])

    xy_out, conf_out, bbox_out = impute_missing(xy, conf, bbox_desc, person_found)

    ok = (
        np.allclose(xy_out[1], xy_out[0])
        and np.allclose(conf_out[1], 0.0)
        and np.allclose(xy_out[0], 0.1)
        and np.allclose(xy_out[2], 0.3)
        and np.allclose(bbox_out[1], bbox_out[0])
        and not np.isnan(xy_out).any()
    )
    return _check(
        "impute_missing: quadro ausente único é forward-filled com conf 0.0", ok
    )


def _selftest_back_fill_leading_run() -> bool:
    xy = np.zeros((3, 17, 2), dtype=np.float32)
    conf = np.full((3, 17), 0.9, dtype=np.float32)
    bbox_desc = np.zeros((3, 4), dtype=np.float32)
    xy[0] = np.nan
    xy[1] = np.nan
    xy[2] = 0.5
    bbox_desc[0] = np.nan
    bbox_desc[1] = np.nan
    bbox_desc[2] = 0.7
    person_found = np.array([False, False, True])

    xy_out, conf_out, bbox_out = impute_missing(xy, conf, bbox_desc, person_found)

    ok = (
        np.allclose(xy_out[0], 0.5)
        and np.allclose(xy_out[1], 0.5)
        and np.allclose(conf_out[0], 0.0)
        and np.allclose(conf_out[1], 0.0)
        and np.allclose(bbox_out[0], 0.7)
        and np.allclose(bbox_out[1], 0.7)
        and not np.isnan(xy_out).any()
    )
    return _check(
        "impute_missing: trecho inicial ausente é back-filled do primeiro quadro válido",
        ok,
    )


def _selftest_all_missing() -> bool:
    xy = np.full((4, 17, 2), np.nan, dtype=np.float32)
    conf = np.zeros((4, 17), dtype=np.float32)
    bbox_desc = np.full((4, 4), np.nan, dtype=np.float32)
    person_found = np.zeros((4,), dtype=bool)

    xy_out, conf_out, bbox_out = impute_missing(xy, conf, bbox_desc, person_found)

    ok = (
        np.allclose(xy_out, 0.0)
        and np.allclose(conf_out, 0.0)
        and np.allclose(bbox_out, 0.0)
        and not np.isnan(xy_out).any()
    )
    return _check(
        "impute_missing: vídeo inteiro sem detecção dá xy e conf zerados, sem NaN",
        ok,
    )


def _selftest_shapes_and_dtypes() -> bool:
    k = 5
    xy = np.zeros((k, 17, 2), dtype=np.float32)
    conf = np.zeros((k, 17), dtype=np.float32)
    bbox_desc = np.zeros((k, 4), dtype=np.float32)
    person_found = np.array([True, False, True, False, True])

    xy_out, conf_out, bbox_out = impute_missing(xy, conf, bbox_desc, person_found)

    ok = (
        xy_out.shape == (k, 17, 2)
        and xy_out.dtype == np.float32
        and conf_out.shape == (k, 17)
        and conf_out.dtype == np.float32
        and bbox_out.shape == (k, 4)
        and bbox_out.dtype == np.float32
    )
    return _check(
        "impute_missing: shapes e dtypes de saída são [K,17,2]/[K,17]/[K,4] float32", ok
    )


def _selftest_bbox_descriptors() -> bool:
    bbox = np.array([[10.0, 20.0, 30.0, 60.0]], dtype=np.float32)
    person_found = np.array([True])
    width, height = 100, 200

    descriptors = bbox_descriptors(bbox, person_found, width, height)
    expected = np.array(
        [
            (10.0 + 30.0) / 2.0 / width,
            (20.0 + 60.0) / 2.0 / height,
            (30.0 - 10.0) / width,
            (60.0 - 20.0) / height,
        ],
        dtype=np.float32,
    )

    ok = np.allclose(descriptors[0], expected)
    return _check(
        "bbox_descriptors: bbox e frame size conhecidos dão os quatro valores esperados",
        ok,
    )


def _selftest_bbox_descriptors_imputation() -> bool:
    bbox_desc = np.zeros((3, 4), dtype=np.float32)
    bbox_desc[0] = [0.1, 0.2, 0.3, 0.4]
    bbox_desc[1] = np.nan
    bbox_desc[2] = [0.5, 0.6, 0.7, 0.8]
    person_found = np.array([True, False, True])
    xy = np.zeros((3, 17, 2), dtype=np.float32)
    conf = np.zeros((3, 17), dtype=np.float32)

    _, _, bbox_out = impute_missing(xy, conf, bbox_desc, person_found)

    ok = bool(np.allclose(bbox_out[1], bbox_out[0])) and bool(
        np.isfinite(bbox_out).all()
    )

    all_missing_desc = np.full((3, 4), np.nan, dtype=np.float32)
    all_missing_found = np.zeros((3,), dtype=bool)
    _, _, all_missing_out = impute_missing(
        xy, conf, all_missing_desc, all_missing_found
    )
    ok = (
        ok
        and bool(np.allclose(all_missing_out, 0.0))
        and bool(np.isfinite(all_missing_out).all())
    )

    return _check(
        "bbox_descriptors + impute_missing: quadro isolado ausente é forward-filled, "
        "vídeo inteiro ausente dá zeros sem NaN/inf",
        ok,
    )


def run_selftest() -> None:
    checks = [
        _selftest_normalization(),
        _selftest_normalization_preserves_angle(),
        _selftest_forward_fill_single_gap(),
        _selftest_back_fill_leading_run(),
        _selftest_all_missing(),
        _selftest_shapes_and_dtypes(),
        _selftest_bbox_descriptors(),
        _selftest_bbox_descriptors_imputation(),
    ]
    if not all(checks):
        print("\npose loading selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose loading selftest OK: todas as checagens passaram")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "selftest",
        help="Roda checagens sintéticas de normalização e imputação de pose",
    )

    args = parser.parse_args()
    if args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
