import sys
import tempfile
from pathlib import Path
from typing import Callable, cast

import numpy as np

from gatefall.pose.loading import (
    PoseArrays,
    bbox_descriptors,
    impute_missing,
    load_pose,
    normalize_keypoints,
)
from gatefall.pose.selftests.fixtures import _write_synthetic_pose


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


def _selftest_leading_run_is_zero_filled() -> bool:
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

    # Igualdade exata, não allclose: o trecho anterior à primeira detecção não
    # pode carregar nenhum resíduo do quadro futuro.
    leading = slice(0, 2)
    ok = (
        bool(np.array_equal(xy_out[leading], np.zeros((2, 17, 2), dtype=np.float32)))
        and bool(np.array_equal(conf_out[leading], np.zeros((2, 17), dtype=np.float32)))
        and bool(np.array_equal(bbox_out[leading], np.zeros((2, 4), dtype=np.float32)))
        and bool(np.allclose(xy_out[2], 0.5))
        and bool(np.allclose(bbox_out[2], 0.7))
        and not bool(np.isnan(xy_out).any())
    )
    return _check(
        "impute_missing: trecho inicial sem detecção é zerado, não back-filled do futuro",
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
        "impute_missing: vídeo inteiro sem detecção é o caso degenerado do trecho "
        "inicial ausente — matriz inteira zerada, sem NaN",
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


def _selftest_load_pose_uses_injected_root() -> bool:
    video_id = "synthetic_env/injected_video"
    with tempfile.TemporaryDirectory() as temporary_dir:
        pose_root = Path(temporary_dir) / "alternate_pose_root"
        _write_synthetic_pose(pose_root, video_id)
        load_with_root = cast(Callable[..., PoseArrays], load_pose)
        try:
            pose = load_with_root(video_id, pose_root=pose_root)
        except TypeError:
            return _check("load_pose: usa pose_root injetado", False)

    ok = (
        pose.k == 3
        and pose.keypoints.shape == (3, 17, 3)
        and bool(np.all(pose.person_found))
    )
    return _check("load_pose: usa pose_root injetado", ok)


def _selftest_load_pose_error_names_injected_path() -> bool:
    video_id = "synthetic_env/missing_video"
    with tempfile.TemporaryDirectory() as temporary_dir:
        pose_root = Path(temporary_dir) / "alternate_pose_root"
        expected_path = pose_root / "synthetic_env" / "missing_video.h5"
        load_with_root = cast(Callable[..., PoseArrays], load_pose)
        try:
            load_with_root(video_id, pose_root=pose_root)
        except FileNotFoundError as error:
            ok = str(expected_path) in str(error)
        except TypeError:
            ok = False
        else:
            ok = False
    return _check("load_pose: erro aponta o caminho da raiz injetada", ok)


def _selftest_build_pose_features_uses_injected_root() -> bool:
    from gatefall.pose.kinematics import EXPECTED_D, build_pose_features

    video_id = "synthetic_env/injected_features"
    with tempfile.TemporaryDirectory() as temporary_dir:
        pose_root = Path(temporary_dir) / "alternate_pose_root"
        _write_synthetic_pose(pose_root, video_id)
        build_with_root = cast(
            Callable[..., tuple[np.ndarray, list[str]]], build_pose_features
        )
        try:
            matrix, names = build_with_root(video_id, pose_root=pose_root)
        except TypeError:
            return _check("build_pose_features: propaga pose_root injetado", False)

    ok = (
        matrix.shape == (3, EXPECTED_D)
        and len(names) == EXPECTED_D
        and bool(np.isfinite(matrix).all())
    )
    return _check("build_pose_features: propaga pose_root injetado", ok)


def run_selftest() -> None:
    checks = [
        _selftest_normalization(),
        _selftest_normalization_preserves_angle(),
        _selftest_forward_fill_single_gap(),
        _selftest_leading_run_is_zero_filled(),
        _selftest_all_missing(),
        _selftest_shapes_and_dtypes(),
        _selftest_bbox_descriptors(),
        _selftest_bbox_descriptors_imputation(),
        _selftest_load_pose_uses_injected_root(),
        _selftest_load_pose_error_names_injected_path(),
        _selftest_build_pose_features_uses_injected_root(),
    ]
    if not all(checks):
        print("\npose loading selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose loading selftest OK: todas as checagens passaram")
