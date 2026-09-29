import sys
import tempfile
from pathlib import Path

import numpy as np

from gatefall.config import TARGET_FPS
from gatefall.pose.kinematics import (
    EXPECTED_D,
    HIP_LEFT,
    HIP_RIGHT,
    SHOULDER_LEFT,
    SHOULDER_RIGHT,
    _BLOCKS,
    _assemble_matrix,
    _block_range,
    _column_index,
    _effective_dt,
    _feature_names,
    _first_difference,
    _second_difference,
    _trunk_orientation,
    build_pose_features,
    feature_names,
)
from gatefall.pose.loading import impute_missing
from gatefall.pose.selftests.fixtures import _write_synthetic_pose


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _selftest_bbox_constant_velocity() -> bool:
    dt = 1.0 / TARGET_FPS
    k = 5
    bbox_desc = np.zeros((k, 4), dtype=np.float32)
    bbox_desc[:, 1] = np.arange(k, dtype=np.float32) * 2.0
    dt_eff = _effective_dt(np.ones((k,), dtype=bool), dt)

    velocity = _first_difference(bbox_desc, dt_eff)
    acceleration = _second_difference(velocity, dt_eff, first_observed=0)

    expected_v = 2.0 / dt
    ok = (
        bool(np.allclose(velocity[1:, 1], expected_v))
        and bool(np.isclose(velocity[0, 1], 0.0))
        and bool(np.allclose(acceleration[2:, 1], 0.0, atol=1e-4))
    )
    return _check(
        "bbox velocidade constante: bbox_vcy constante e bbox_acy zero após a borda",
        ok,
    )


def _selftest_trunk_upright_and_horizontal() -> bool:
    dt = 1.0 / TARGET_FPS
    xy = np.zeros((2, 17, 2), dtype=np.float32)
    xy[0, SHOULDER_LEFT] = [0.0, -0.2]
    xy[0, SHOULDER_RIGHT] = [0.0, -0.2]
    xy[0, HIP_LEFT] = [0.0, 0.2]
    xy[0, HIP_RIGHT] = [0.0, 0.2]

    xy[1, SHOULDER_LEFT] = [-0.2, 0.0]
    xy[1, SHOULDER_RIGHT] = [-0.2, 0.0]
    xy[1, HIP_LEFT] = [0.2, 0.0]
    xy[1, HIP_RIGHT] = [0.2, 0.0]

    dt_eff = _effective_dt(np.ones((2,), dtype=bool), dt)
    trunk = _trunk_orientation(xy, dt_eff, first_observed=0)
    upright_ok = np.isclose(trunk[0, 0], 1.0, atol=1e-5) and np.isclose(
        trunk[0, 1], 0.0, atol=1e-5
    )
    horizontal_ok = np.isclose(trunk[1, 0], 0.0, atol=1e-5) and np.isclose(
        trunk[1, 1], 1.0, atol=1e-5
    )
    return _check(
        "trunk orientation: tronco vertical e horizontal dão sin/cos esperados",
        bool(upright_ok and horizontal_ok),
    )


def _selftest_trunk_wrap() -> bool:
    dt = 1.0 / TARGET_FPS
    k = 3
    xy = np.zeros((k, 17, 2), dtype=np.float32)
    angles = [np.pi - 0.01, np.pi - 0.001, -np.pi + 0.01]
    for i, angle in enumerate(angles):
        vector = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        xy[i, SHOULDER_LEFT] = -vector / 2.0
        xy[i, SHOULDER_RIGHT] = -vector / 2.0
        xy[i, HIP_LEFT] = vector / 2.0
        xy[i, HIP_RIGHT] = vector / 2.0

    dt_eff = _effective_dt(np.ones((k,), dtype=bool), dt)
    trunk = _trunk_orientation(xy, dt_eff, first_observed=0)
    dtheta_wrap = trunk[2, 2]
    huge_wrap = (2.0 * np.pi) / dt

    ok = bool(abs(dtheta_wrap) < huge_wrap / 10.0)
    return _check(
        "trunk_dtheta: cruzar o wrap +pi/-pi dá dtheta pequeno, não ~2*pi/dt",
        ok,
    )


def _selftest_trunk_gap_uses_gap_length() -> bool:
    dt = 1.0 / TARGET_FPS
    k = 4
    xy = np.zeros((k, 17, 2), dtype=np.float32)
    angles = [0.0, 0.0, 0.0, 0.3]
    for i, angle in enumerate(angles):
        vector = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        xy[i, SHOULDER_LEFT] = -vector / 2.0
        xy[i, SHOULDER_RIGHT] = -vector / 2.0
        xy[i, HIP_LEFT] = vector / 2.0
        xy[i, HIP_RIGHT] = vector / 2.0
    person_found = np.array([True, False, False, True])

    dt_eff = _effective_dt(person_found, dt)
    trunk = _trunk_orientation(xy, dt_eff, first_observed=0)

    expected_dtheta = 0.3 / (3.0 * dt)
    ok = bool(np.isclose(trunk[3, 2], expected_dtheta)) and not bool(
        np.isclose(trunk[3, 2], 0.3 / dt)
    )
    return _check(
        "trunk_dtheta: gap de 3 quadros usa 0.3/(3*dt) na reaparição, não 0.3/dt",
        ok,
    )


def _selftest_imputed_frame_zero_velocity() -> bool:
    dt = 1.0 / TARGET_FPS
    k = 4
    xy = np.zeros((k, 17, 2), dtype=np.float32)
    conf = np.full((k, 17), 0.9, dtype=np.float32)
    bbox_desc = np.zeros((k, 4), dtype=np.float32)
    xy[0] = 0.1
    xy[1] = np.nan
    xy[2] = 0.1
    xy[3] = 0.3
    bbox_desc[0] = 0.4
    bbox_desc[1] = np.nan
    bbox_desc[2] = 0.4
    bbox_desc[3] = 0.6
    person_found = np.array([True, False, True, True])

    xy_out, conf_out, bbox_out = impute_missing(xy, conf, bbox_desc, person_found)
    xy_flat = xy_out.reshape(k, 34)
    dt_eff = _effective_dt(person_found, dt)

    kp_velocity = _first_difference(xy_flat, dt_eff)
    bbox_velocity = _first_difference(bbox_out, dt_eff)

    ok = (
        bool(np.allclose(kp_velocity[1], 0.0))
        and bool(np.allclose(bbox_velocity[1], 0.0))
        and bool(np.isclose(conf_out[1, 0], 0.0))
    )
    return _check(
        "quadro imputado por forward-fill: velocidade exatamente zero", ok
    )


def _selftest_gap_velocity_uses_gap_length() -> bool:
    dt = 1.0 / TARGET_FPS
    k = 4
    bbox_desc = np.zeros((k, 4), dtype=np.float32)
    conf = np.full((k, 17), 0.9, dtype=np.float32)
    xy = np.zeros((k, 17, 2), dtype=np.float32)
    bbox_desc[0, 1] = 0.0
    bbox_desc[1] = np.nan
    bbox_desc[2] = np.nan
    bbox_desc[3, 1] = 0.3
    person_found = np.array([True, False, False, True])

    _, _, bbox_out = impute_missing(xy, conf, bbox_desc, person_found)
    dt_eff = _effective_dt(person_found, dt)
    velocity = _first_difference(bbox_out, dt_eff)

    expected_v = 0.3 / (3.0 * dt)
    ok = bool(np.isclose(velocity[3, 1], expected_v)) and not bool(
        np.isclose(velocity[3, 1], 0.3 / dt)
    )
    return _check(
        "gap de 3 quadros: velocidade na reaparição usa 0.3/(3*dt), não 0.3/dt",
        ok,
    )


def _selftest_output_shape_and_finiteness() -> bool:
    k = 6
    xy = np.random.default_rng(0).normal(size=(k, 17, 2)).astype(np.float32)
    conf = np.full((k, 17), 0.9, dtype=np.float32)
    bbox_desc = np.random.default_rng(1).normal(size=(k, 4)).astype(np.float32)
    person_found = np.ones((k,), dtype=bool)

    xy_out, conf_out, bbox_out = impute_missing(xy, conf, bbox_desc, person_found)
    dt = 1.0 / TARGET_FPS
    xy_flat = xy_out.reshape(k, 34)
    dt_eff = _effective_dt(person_found, dt)
    kp_velocity = _first_difference(xy_flat, dt_eff)
    kp_acceleration = _second_difference(kp_velocity, dt_eff, first_observed=0)
    bbox_velocity = _first_difference(bbox_out, dt_eff)
    bbox_acceleration = _second_difference(
        bbox_velocity, dt_eff, first_observed=0
    )
    trunk = _trunk_orientation(xy_out, dt_eff, first_observed=0)

    matrix = _assemble_matrix(
        xy_flat,
        conf_out,
        kp_velocity,
        kp_acceleration,
        bbox_out,
        bbox_velocity,
        bbox_acceleration,
        trunk,
    )
    feature_names = _feature_names()

    ok = (
        matrix.shape == (k, EXPECTED_D)
        and matrix.dtype == np.float32
        and len(feature_names) == EXPECTED_D
        and bool(np.isfinite(matrix).all())
    )
    return _check(
        "saída: shape [K,134] float32, 134 nomes de feature, matriz finita", ok
    )


def _selftest_blocks_cover_range_without_gaps_or_overlaps() -> bool:
    blocks_by_start = sorted(_BLOCKS, key=lambda block: block[1])
    ok = bool(blocks_by_start) and blocks_by_start[0][1] == 0
    ok = ok and blocks_by_start[-1][2] == EXPECTED_D
    for (_, _, end), (_, next_start, _) in zip(blocks_by_start, blocks_by_start[1:]):
        ok = ok and end == next_start
    return _check(
        "_BLOCKS: fronteiras cobrem 0..134 sem lacunas nem sobreposições", ok
    )


def _synthetic_keypoints_and_bbox(k: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keypoints = np.zeros((k, 17, 3), dtype=np.float32)
    keypoints[:, :, 0] = rng.uniform(10.0, 90.0, size=(k, 17))
    keypoints[:, :, 1] = rng.uniform(10.0, 190.0, size=(k, 17))
    keypoints[:, :, 2] = rng.uniform(0.4, 1.0, size=(k, 17))
    # bbox estritamente não degenerada em todo quadro: normalize_keypoints
    # divide pela diagonal da bbox, que precisa ser > 0 onde há detecção.
    x_min = rng.uniform(0.0, 40.0, size=k)
    y_min = rng.uniform(0.0, 80.0, size=k)
    bbox = np.stack(
        [
            x_min,
            y_min,
            x_min + rng.uniform(10.0, 50.0, size=k),
            y_min + rng.uniform(20.0, 100.0, size=k),
        ],
        axis=1,
    ).astype(np.float32)
    return keypoints, bbox


def _prefix_causality_fixtures() -> list[tuple[str, np.ndarray]]:
    return [
        # Cobre todos os regimes: trecho inicial ausente, primeira detecção,
        # gaps curtos, reaparições e trecho ausente no meio.
        (
            "gaps",
            np.array(
                [
                    False, False, False, True, True, False,
                    True, True, True, False, False, True,
                ],
                dtype=np.bool_,
            ),
        ),
        # f == 0: não há trecho inicial ausente, então first_observed não
        # protege nenhuma linha e as bordas de derivada caem em 0, 1 e 2.
        (
            "first_observed_zero",
            np.array(
                [True, True, False, True, False, False, True, True],
                dtype=np.bool_,
            ),
        ),
        # K == 1 e K == 2: vídeos mais curtos que a janela de duas linhas usada
        # por _second_difference, onde os slices de borda ficam vazios.
        ("single_frame", np.array([True], dtype=np.bool_)),
        ("two_frames", np.array([False, True], dtype=np.bool_)),
    ]


def _selftest_prefix_causality() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as temporary_dir:
        pose_root = Path(temporary_dir)
        for fixture_name, base_found in _prefix_causality_fixtures():
            k = int(base_found.shape[0])
            base_keypoints, base_bbox = _synthetic_keypoints_and_bbox(k, seed=20240917)
            alt_keypoints, alt_bbox = _synthetic_keypoints_and_bbox(k, seed=20240918)

            base_id = f"prefix_env/{fixture_name}_base"
            _write_synthetic_pose(
                pose_root,
                base_id,
                k,
                keypoints=base_keypoints,
                bbox=base_bbox,
                person_found=base_found,
            )
            base_matrix, names = build_pose_features(base_id, pose_root=pose_root)

            ok = ok and base_matrix.shape == (k, EXPECTED_D)
            ok = ok and len(names) == EXPECTED_D

            for t in range(k - 1):
                variant_found = base_found.copy()
                variant_found[t + 1 :] = ~variant_found[t + 1 :]
                variant_keypoints = base_keypoints.copy()
                variant_bbox = base_bbox.copy()
                variant_keypoints[t + 1 :] = alt_keypoints[t + 1 :]
                variant_bbox[t + 1 :] = alt_bbox[t + 1 :]

                video_id = f"prefix_env/{fixture_name}_variant_{t}"
                _write_synthetic_pose(
                    pose_root,
                    video_id,
                    k,
                    keypoints=variant_keypoints,
                    bbox=variant_bbox,
                    person_found=variant_found,
                )
                variant_matrix, _ = build_pose_features(video_id, pose_root=pose_root)

                # Igualdade exata: qualquer vazamento do futuro aparece como
                # diferença de bit, não como diferença dentro de tolerância.
                ok = ok and bool(
                    np.array_equal(base_matrix[: t + 1], variant_matrix[: t + 1])
                )
                # Guarda de não trivialidade: o sufixo realmente difere na
                # primeira linha alterada. Aqui ela vale em todo t porque o flip
                # inverte person_found[t+1], e o lado com detecção tem kp_conf
                # não nulo enquanto o outro tem kp_conf exatamente zero.
                ok = ok and not bool(
                    np.array_equal(base_matrix[t + 1], variant_matrix[t + 1])
                )

    return _check(
        "build_pose_features: prefixo 0..t é idêntico quando só o sufixo muda "
        "(nenhuma feature depende do futuro)",
        ok,
    )


def _selftest_leading_absent_rows_are_fully_zero() -> bool:
    k = 5
    person_found = np.array([False, False, True, True, True], dtype=np.bool_)
    keypoints, bbox = _synthetic_keypoints_and_bbox(k, seed=7)

    with tempfile.TemporaryDirectory() as temporary_dir:
        pose_root = Path(temporary_dir)
        _write_synthetic_pose(
            pose_root,
            "leading_env/video",
            k,
            keypoints=keypoints,
            bbox=bbox,
            person_found=person_found,
        )
        matrix, _ = build_pose_features("leading_env/video", pose_root=pose_root)

    leading = matrix[:2]
    # O bloco trunk é o caso não óbvio dentro dessas linhas: sem o zeramento
    # explícito, arctan2(0, 0) daria trunk_cos = 1.0 em vez de 0.0.
    ok = (
        bool(np.array_equal(leading, np.zeros((2, EXPECTED_D), dtype=np.float32)))
        and bool(np.any(matrix[2] != 0.0))
    )
    return _check(
        "build_pose_features: linhas antes da primeira detecção são exatamente "
        "zero em todas as 134 colunas, inclusive o bloco trunk",
        ok,
    )


def _selftest_first_observation_has_zero_derivatives() -> bool:
    k = 5
    person_found = np.array([False, False, True, True, True], dtype=np.bool_)
    keypoints, bbox = _synthetic_keypoints_and_bbox(k, seed=11)
    first_observed = 2

    with tempfile.TemporaryDirectory() as temporary_dir:
        pose_root = Path(temporary_dir)
        _write_synthetic_pose(
            pose_root,
            "first_env/video",
            k,
            keypoints=keypoints,
            bbox=bbox,
            person_found=person_found,
        )
        matrix, _ = build_pose_features("first_env/video", pose_root=pose_root)

    derivative_blocks = [
        "kp_velocity",
        "kp_acceleration",
        "bbox_velocity",
        "bbox_acceleration",
    ]
    acceleration_blocks = ["kp_acceleration", "bbox_acceleration"]
    dtheta_col = _column_index("trunk_dtheta")
    sin_col = _column_index("trunk_sin")
    cos_col = _column_index("trunk_cos")

    ok = True
    for block_name in derivative_blocks:
        start, end = _block_range(block_name)
        ok = ok and bool(
            np.array_equal(
                matrix[first_observed, start:end],
                np.zeros(end - start, dtype=np.float32),
            )
        )
    ok = ok and bool(matrix[first_observed, dtheta_col] == np.float32(0.0))

    for block_name in ("kp_xy", "kp_conf", "bbox_pos"):
        start, end = _block_range(block_name)
        ok = ok and bool(np.any(matrix[first_observed, start:end] != 0.0))
    # Orientação real no quadro observado: sin e cos não podem ser ambos zero.
    ok = ok and not bool(
        matrix[first_observed, sin_col] == np.float32(0.0)
        and matrix[first_observed, cos_col] == np.float32(0.0)
    )

    for block_name in acceleration_blocks:
        start, end = _block_range(block_name)
        ok = ok and bool(
            np.array_equal(
                matrix[first_observed + 1, start:end],
                np.zeros(end - start, dtype=np.float32),
            )
        )
    # Não trivialidade: a partir de f+2 a aceleração volta a ser real.
    start, end = _block_range("kp_acceleration")
    ok = ok and bool(np.any(matrix[first_observed + 2, start:end] != 0.0))

    return _check(
        "build_pose_features: primeira detecção tem derivadas zero e f+1 tem "
        "aceleração zero, sem zerar posição/confiança/orientação",
        ok,
    )


def _selftest_all_missing_features_are_zero() -> bool:
    k = 5
    person_found = np.zeros((k,), dtype=np.bool_)
    keypoints, bbox = _synthetic_keypoints_and_bbox(k, seed=13)

    with tempfile.TemporaryDirectory() as temporary_dir:
        pose_root = Path(temporary_dir)
        _write_synthetic_pose(
            pose_root,
            "empty_env/video",
            k,
            keypoints=keypoints,
            bbox=bbox,
            person_found=person_found,
        )
        matrix, _ = build_pose_features("empty_env/video", pose_root=pose_root)

    ok = bool(np.array_equal(matrix, np.zeros((k, EXPECTED_D), dtype=np.float32)))
    return _check(
        "build_pose_features: vídeo inteiro sem detecção dá matriz [K,134] "
        "exatamente zerada, inclusive trunk_cos",
        ok,
    )


def _selftest_gap_after_acquisition_uses_gap_length() -> bool:
    k = 5
    height = 200
    person_found = np.array([False, True, False, False, True], dtype=np.bool_)
    keypoints, bbox = _synthetic_keypoints_and_bbox(k, seed=17)
    # bbox_cy = cy / height: 40/200 = 0.2 no quadro 1 e 100/200 = 0.5 no
    # quadro 4, ou seja, um delta conhecido de 0.3 ao longo de um gap de 3.
    bbox[1] = np.array([10.0, 20.0, 30.0, 60.0], dtype=np.float32)
    bbox[4] = np.array([10.0, 80.0, 30.0, 120.0], dtype=np.float32)

    with tempfile.TemporaryDirectory() as temporary_dir:
        pose_root = Path(temporary_dir)
        _write_synthetic_pose(
            pose_root,
            "gap_env/video",
            k,
            keypoints=keypoints,
            bbox=bbox,
            person_found=person_found,
        )
        matrix, _ = build_pose_features("gap_env/video", pose_root=pose_root)

    dt = 1.0 / TARGET_FPS
    velocity_col = _column_index("bbox_vcy")
    delta = 100.0 / height - 40.0 / height
    expected_v = delta / (3.0 * dt)

    ok = (
        bool(np.isclose(matrix[4, velocity_col], expected_v))
        and not bool(np.isclose(matrix[4, velocity_col], delta / dt))
        and bool(matrix[1, velocity_col] == np.float32(0.0))
    )
    return _check(
        "build_pose_features: reaparição após gap de 3 usa delta/(3*dt) e a "
        "primeira detecção continua com velocidade zero",
        ok,
    )


def run_selftest() -> None:
    checks = [
        _selftest_bbox_constant_velocity(),
        _selftest_trunk_upright_and_horizontal(),
        _selftest_trunk_wrap(),
        _selftest_trunk_gap_uses_gap_length(),
        _selftest_imputed_frame_zero_velocity(),
        _selftest_gap_velocity_uses_gap_length(),
        _selftest_output_shape_and_finiteness(),
        _selftest_blocks_cover_range_without_gaps_or_overlaps(),
        _selftest_prefix_causality(),
        _selftest_leading_absent_rows_are_fully_zero(),
        _selftest_first_observation_has_zero_derivatives(),
        _selftest_all_missing_features_are_zero(),
        _selftest_gap_after_acquisition_uses_gap_length(),
    ]
    if not all(checks):
        print("\npose kinematics selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose kinematics selftest OK: todas as checagens passaram")
