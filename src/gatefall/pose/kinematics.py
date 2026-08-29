"""Descritores cinemáticos derivados da pose (YOLO-Pose) do Le2i.

Nada aqui é gravado em disco: são features derivadas, computadas em tempo de
carregamento por design, para que uma escolha de suavização ou janelamento
nunca fique congelada em um arquivo.
"""

import argparse
import sys
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import TARGET_FPS
from gatefall.data.frames import read_frames
from gatefall.data.le2i.frames import FRAMES_PATH
from gatefall.pose.loading import (
    bbox_descriptors,
    impute_missing,
    load_pose,
    normalize_keypoints,
)

EXPECTED_K_SUM = 30494
EXPECTED_D = 134

SHOULDER_LEFT = 5
SHOULDER_RIGHT = 6
HIP_LEFT = 11
HIP_RIGHT = 12


def _block_definitions() -> list[tuple[str, list[str]]]:
    kp_xy: list[str] = []
    for i in range(17):
        kp_xy.append(f"kp_x_{i}")
        kp_xy.append(f"kp_y_{i}")
    kp_conf = [f"kp_conf_{i}" for i in range(17)]
    kp_velocity: list[str] = []
    for i in range(17):
        kp_velocity.append(f"kp_vx_{i}")
        kp_velocity.append(f"kp_vy_{i}")
    kp_acceleration: list[str] = []
    for i in range(17):
        kp_acceleration.append(f"kp_ax_{i}")
        kp_acceleration.append(f"kp_ay_{i}")
    bbox_pos = ["bbox_cx", "bbox_cy", "bbox_w", "bbox_h"]
    bbox_velocity = ["bbox_vcx", "bbox_vcy", "bbox_vw", "bbox_vh"]
    bbox_acceleration = ["bbox_acx", "bbox_acy", "bbox_aw", "bbox_ah"]
    trunk = ["trunk_sin", "trunk_cos", "trunk_dtheta"]
    return [
        ("kp_xy", kp_xy),
        ("kp_conf", kp_conf),
        ("kp_velocity", kp_velocity),
        ("kp_acceleration", kp_acceleration),
        ("bbox_pos", bbox_pos),
        ("bbox_velocity", bbox_velocity),
        ("bbox_acceleration", bbox_acceleration),
        ("trunk", trunk),
    ]


def _feature_names() -> list[str]:
    names: list[str] = []
    for _, block_names in _block_definitions():
        names.extend(block_names)
    return names


def _blocks_from_definitions() -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    offset = 0
    for name, block_names in _block_definitions():
        blocks.append((name, offset, offset + len(block_names)))
        offset += len(block_names)
    return blocks


_BLOCKS: list[tuple[str, int, int]] = _blocks_from_definitions()


def _backfill_source_indices(person_found: np.ndarray) -> np.ndarray:
    k = person_found.shape[0]
    src = np.zeros(k, dtype=np.int64)
    if not np.any(person_found):
        return src

    # Espelha exatamente o forward-fill de impute_missing: cada quadro ausente
    # aponta para o último quadro observado antes dele.
    last_valid_idx = 0
    for i in range(k):
        if person_found[i]:
            last_valid_idx = i
        src[i] = last_valid_idx

    # E o back-fill do trecho inicial, quando o vídeo começa sem detecção.
    first_valid_index = int(np.argmax(person_found))
    if first_valid_index > 0:
        src[:first_valid_index] = first_valid_index

    return src


def _effective_dt(person_found: np.ndarray, dt: float) -> np.ndarray:
    src = _backfill_source_indices(person_found)
    dt_eff = np.zeros(src.shape[0], dtype=np.float32)
    # No quadro em que a pessoa reaparece após um gap de N quadros, o
    # deslocamento observado se acumulou ao longo do gap inteiro, não de um
    # único intervalo de quadro; dividir por dt fixo infla a velocidade em N
    # vezes (e a aceleração em ~N^2). dt_eff carrega esse N implícito.
    dt_eff[1:] = (src[1:] - src[:-1]).astype(np.float32) * dt
    return dt_eff


def _safe_divide(numerator: np.ndarray, dt_eff: np.ndarray) -> np.ndarray:
    dt_col = dt_eff.reshape(-1, 1).astype(np.float32)
    denom = np.where(dt_col != 0, dt_col, np.float32(1.0))
    result = np.where(dt_col != 0, numerator / denom, np.float32(0.0))
    return result.astype(np.float32)


def _first_difference(values: np.ndarray, dt_eff: np.ndarray) -> np.ndarray:
    numerator = np.zeros_like(values, dtype=np.float32)
    # Sem diferença de primeira ordem no primeiro quadro; a posição inicial
    # preenche a posição líder com 0.0, replicando a borda já usada na grade
    # temporal. dt_eff[0] é sempre 0.0, então _safe_divide já emite 0.0 aqui.
    numerator[1:] = values[1:] - values[:-1]
    return _safe_divide(numerator, dt_eff)


def _second_difference(first_diff: np.ndarray, dt_eff: np.ndarray) -> np.ndarray:
    numerator = np.zeros_like(first_diff, dtype=np.float32)
    # Sem diferença de segunda ordem nos dois primeiros quadros pelo mesmo
    # motivo: não há vizinho anterior suficiente para formar a diferença.
    numerator[2:] = first_diff[2:] - first_diff[1:-1]
    return _safe_divide(numerator, dt_eff)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _trunk_orientation(xy: np.ndarray, dt_eff: np.ndarray) -> np.ndarray:
    shoulder_mid = (xy[:, SHOULDER_LEFT] + xy[:, SHOULDER_RIGHT]) / 2.0
    hip_mid = (xy[:, HIP_LEFT] + xy[:, HIP_RIGHT]) / 2.0
    trunk_vector = hip_mid - shoulder_mid
    theta = np.arctan2(trunk_vector[:, 1], trunk_vector[:, 0]).astype(np.float32)

    trunk_sin = np.sin(theta).astype(np.float32)
    trunk_cos = np.cos(theta).astype(np.float32)

    # sin/cos em vez do ângulo bruto: o ângulo bruto salta em 2*pi entre
    # quadros adjacentes puramente por causa do wrap +pi/-pi, e o codificador
    # temporal leria isso como uma movimentação enorme.
    dtheta_numerator = np.zeros((theta.shape[0], 1), dtype=np.float32)
    raw_delta = theta[1:] - theta[:-1]
    dtheta_numerator[1:, 0] = _wrap_angle(raw_delta)
    # Mesmo raciocínio de _effective_dt: no quadro em que a pessoa reaparece
    # após um gap, o delta de ângulo wrapped se acumulou ao longo do gap
    # inteiro, então usamos dt_eff (não dt fixo) via _safe_divide.
    dtheta = _safe_divide(dtheta_numerator, dt_eff)[:, 0]

    return np.stack([trunk_sin, trunk_cos, dtheta], axis=1).astype(np.float32)


def _assemble_matrix(
    xy_flat: np.ndarray,
    conf: np.ndarray,
    kp_velocity: np.ndarray,
    kp_acceleration: np.ndarray,
    bbox_desc: np.ndarray,
    bbox_velocity: np.ndarray,
    bbox_acceleration: np.ndarray,
    trunk: np.ndarray,
) -> np.ndarray:
    # Não emitimos deslocamento bruto como bloco separado: deslocamento é
    # velocidade vezes uma constante (dt), logo é exatamente redundante com o
    # bloco de velocidade acima e só acrescentaria 34 colunas colineares.
    #
    # Os blocos de bbox (posição/velocidade/aceleração) não são decoração
    # opcional: normalize_keypoints centra os keypoints no centro da bbox, o
    # que remove deliberadamente a translação global do corpo de `xy`. O
    # movimento descendente de uma queda vive inteiramente em
    # d(bbox_cy)/dt. Descartar os blocos de bbox deixaria a baseline
    # pose-only cega para o sinal mais forte de queda.
    return np.concatenate(
        [
            xy_flat,
            conf,
            kp_velocity,
            kp_acceleration,
            bbox_desc,
            bbox_velocity,
            bbox_acceleration,
            trunk,
        ],
        axis=1,
    ).astype(np.float32)


def build_pose_features(video_id: str) -> tuple[np.ndarray, list[str]]:
    dt = 1.0 / TARGET_FPS

    pose = load_pose(video_id)
    xy, conf = normalize_keypoints(pose.keypoints, pose.bbox, pose.person_found)
    bbox_desc = bbox_descriptors(pose.bbox, pose.person_found, pose.width, pose.height)
    xy, conf, bbox_desc = impute_missing(xy, conf, bbox_desc, pose.person_found)

    k = xy.shape[0]
    xy_flat = xy.reshape(k, 34)

    dt_eff = _effective_dt(pose.person_found, dt)

    kp_velocity = _first_difference(xy_flat, dt_eff)
    kp_acceleration = _second_difference(kp_velocity, dt_eff)

    bbox_velocity = _first_difference(bbox_desc, dt_eff)
    bbox_acceleration = _second_difference(bbox_velocity, dt_eff)

    trunk = _trunk_orientation(xy, dt_eff)

    matrix = _assemble_matrix(
        xy_flat,
        conf,
        kp_velocity,
        kp_acceleration,
        bbox_desc,
        bbox_velocity,
        bbox_acceleration,
        trunk,
    )

    feature_names = _feature_names()
    assert matrix.shape[1] == len(feature_names)
    assert matrix.shape[1] == EXPECTED_D
    return matrix, feature_names


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
    acceleration = _second_difference(velocity, dt_eff)

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
    trunk = _trunk_orientation(xy, dt_eff)
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
    trunk = _trunk_orientation(xy, dt_eff)
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
    trunk = _trunk_orientation(xy, dt_eff)

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
    kp_acceleration = _second_difference(kp_velocity, dt_eff)
    bbox_velocity = _first_difference(bbox_out, dt_eff)
    bbox_acceleration = _second_difference(bbox_velocity, dt_eff)
    trunk = _trunk_orientation(xy_out, dt_eff)

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
    ]
    if not all(checks):
        print("\npose kinematics selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose kinematics selftest OK: todas as checagens passaram")


def run_report() -> None:
    frames = read_frames(FRAMES_PATH)
    video_ids = [str(video_id) for video_id in frames["video_id"].unique()]
    group_sizes = cast(pd.Series, frames.groupby("video_id").size())

    matrices: list[np.ndarray] = []
    k_mismatches: list[str] = []
    for video_id in video_ids:
        matrix = build_pose_features(video_id)[0]
        matrices.append(matrix)
        expected_rows = int(cast(int, group_sizes[video_id]))
        if matrix.shape[0] != expected_rows:
            k_mismatches.append(
                f"{video_id} (build_pose_features={matrix.shape[0]}, "
                f"frames.parquet={expected_rows})"
            )

    all_features = np.concatenate(matrices, axis=0)
    total_rows = all_features.shape[0]
    d = all_features.shape[1]

    print(f"\nvídeos processados: {len(video_ids)}")
    print(f"total de linhas: {total_rows} (esperado {EXPECTED_K_SUM})")
    print(f"D: {d} (esperado {EXPECTED_D})")

    print("\n=== estatísticas por bloco de features ===")
    for name, start, end in _BLOCKS:
        block = all_features[:, start:end]
        abs_block = np.abs(block)
        percentiles = np.percentile(abs_block, [0.1, 1, 50, 99, 99.9])
        p99 = percentiles[3]
        frac_exceeds_10x_p99 = float(np.mean(abs_block > 10 * p99))
        print(
            f"  {name}: min={block.min():.6f}, max={block.max():.6f}, "
            f"mean={block.mean():.6f}, p0.1_abs={percentiles[0]:.6f}, "
            f"p1_abs={percentiles[1]:.6f}, p50_abs={percentiles[2]:.6f}, "
            f"p99_abs={percentiles[3]:.6f}, p99.9_abs={percentiles[4]:.6f}, "
            f"frac_exceeds_10x_p99={frac_exceeds_10x_p99:.6f}"
        )

    non_finite = int(np.sum(~np.isfinite(all_features)))
    print(f"\nvalores não finitos: {non_finite} (esperado 0)")

    print("\n=== checagem: K por vídeo (build_pose_features vs frames.parquet) ===")
    if k_mismatches:
        print("video_ids com divergência:")
        for mismatch in k_mismatches:
            print(f"  {mismatch}")
    ok_k_per_video = _check(
        "K de build_pose_features == contagem de quadros em frames.parquet, "
        "para todo video_id",
        len(k_mismatches) == 0,
    )

    ok_rows = _check(f"total de linhas == {EXPECTED_K_SUM}", total_rows == EXPECTED_K_SUM)
    ok_finite = _check("nenhum valor não finito", non_finite == 0)

    if not (ok_rows and ok_finite and ok_k_per_video):
        print("\npose kinematics report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose kinematics report OK: todas as checagens passaram")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "selftest",
        help="Roda checagens sintéticas dos descritores cinemáticos",
    )
    subparsers.add_parser(
        "report",
        help="Roda build_pose_features sobre todos os vídeos e reporta estatísticas",
    )

    args = parser.parse_args()
    if args.command == "selftest":
        run_selftest()
    elif args.command == "report":
        run_report()


if __name__ == "__main__":
    main()
