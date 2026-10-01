"""Descritores cinemáticos derivados de pose (YOLO-Pose).

Nada aqui é gravado em disco: são features derivadas, computadas em tempo de
carregamento por design, para que uma escolha de suavização ou janelamento
nunca fique congelada em um arquivo.
"""

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import TARGET_FPS
from gatefall.datasets import DatasetAdapter, SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.pose.loading import (
    bbox_descriptors,
    first_observed_index,
    impute_missing,
    PoseArrays,
    load_person_found,
    load_pose,
    normalize_keypoints,
)

EXPECTED_K_SUM = 30494
EXPECTED_VIDEOS_WITH_PREFIX = 63
EXPECTED_PREFIX_ROWS = 1384
POSE_FEATURE_DIM = 134
EXPECTED_D = POSE_FEATURE_DIM

SHOULDER_LEFT = 5
SHOULDER_RIGHT = 6
HIP_LEFT = 11
HIP_RIGHT = 12

COCO17_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO17_SKELETON_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
)


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


def feature_names() -> list[str]:
    return _feature_names()


def feature_blocks() -> list[tuple[str, int, int]]:
    return list(_BLOCKS)


def _last_observed_indices(person_found: np.ndarray) -> np.ndarray:
    k = person_found.shape[0]
    src = np.full(k, -1, dtype=np.int64)   # -1 = nenhuma observação ainda
    # Espelha o forward-fill de impute_missing: cada quadro aponta para a
    # última observação em ou antes dele.
    last_valid_idx = -1
    for i in range(k):
        if person_found[i]:
            last_valid_idx = i
        src[i] = last_valid_idx
    return src


def _effective_dt(person_found: np.ndarray, dt: float) -> np.ndarray:
    src = _last_observed_indices(person_found)
    dt_eff = np.zeros(src.shape[0], dtype=np.float32)
    # No quadro em que a pessoa reaparece após um gap de N quadros, o
    # deslocamento observado se acumulou ao longo do gap inteiro, não de um
    # único intervalo de quadro; dividir por dt fixo infla a velocidade em N
    # vezes (e a aceleração em ~N^2). dt_eff carrega esse N implícito.
    # Antes da primeira observação o sentinela -1 marca "sem passado": ali
    # dt_eff é 0.0, e não a diferença espúria contra o sentinela.
    observed_before = src[:-1] >= 0
    dt_eff[1:] = np.where(observed_before, src[1:] - src[:-1], 0).astype(np.float32) * dt
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


def _second_difference(
    first_diff: np.ndarray, dt_eff: np.ndarray, *, first_observed: int
) -> np.ndarray:
    numerator = np.zeros_like(first_diff, dtype=np.float32)
    # Sem diferença de segunda ordem nos dois primeiros quadros após a
    # aquisição, pelo mesmo motivo: não há vizinho anterior suficiente para
    # formar a diferença. first_diff[f] é 0.0 pela convenção de ausência, não
    # uma velocidade medida, então first_diff[f+1] - first_diff[f] injetaria
    # um pico de aceleração v[f+1]/dt cuja magnitude só depende de onde a
    # aquisição começou. Com f == 0 isto reduz ao clássico numerator[2:].
    start = first_observed + 2
    numerator[start:] = first_diff[start:] - first_diff[start - 1 : -1]
    return _safe_divide(numerator, dt_eff)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _trunk_orientation(
    xy: np.ndarray, dt_eff: np.ndarray, *, first_observed: int
) -> np.ndarray:
    shoulder_mid = (xy[:, SHOULDER_LEFT] + xy[:, SHOULDER_RIGHT]) / 2.0
    hip_mid = (xy[:, HIP_LEFT] + xy[:, HIP_RIGHT]) / 2.0
    trunk_vector = hip_mid - shoulder_mid
    theta = np.arctan2(trunk_vector[:, 1], trunk_vector[:, 0]).astype(np.float32)

    trunk_sin = np.sin(theta).astype(np.float32)
    trunk_cos = np.cos(theta).astype(np.float32)

    # Nas linhas anteriores à primeira observação o tronco é o vetor nulo, e
    # arctan2(0, 0) == 0.0 daria trunk_cos = 1.0 — um tronco horizontal
    # sintético onde não há pose nenhuma.
    trunk_sin[:first_observed] = 0.0
    trunk_cos[:first_observed] = 0.0

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


def build_pose_features(
    video_id: str, *, pose_root: Path
) -> tuple[np.ndarray, list[str]]:
    return build_pose_features_from_arrays(load_pose(video_id, pose_root=pose_root))


def build_pose_features_from_arrays(
    pose: PoseArrays,
) -> tuple[np.ndarray, list[str]]:
    dt = 1.0 / TARGET_FPS
    xy, conf = normalize_keypoints(pose.keypoints, pose.bbox, pose.person_found)
    bbox_desc = bbox_descriptors(pose.bbox, pose.person_found, pose.width, pose.height)
    xy, conf, bbox_desc = impute_missing(xy, conf, bbox_desc, pose.person_found)

    k = xy.shape[0]
    xy_flat = xy.reshape(k, 34)

    dt_eff = _effective_dt(pose.person_found, dt)
    first_observed = first_observed_index(pose.person_found)

    kp_velocity = _first_difference(xy_flat, dt_eff)
    kp_acceleration = _second_difference(
        kp_velocity, dt_eff, first_observed=first_observed
    )

    bbox_velocity = _first_difference(bbox_desc, dt_eff)
    bbox_acceleration = _second_difference(
        bbox_velocity, dt_eff, first_observed=first_observed
    )

    trunk = _trunk_orientation(xy, dt_eff, first_observed=first_observed)

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
    if (
        matrix.shape[1] != len(feature_names)
        or matrix.shape[1] != POSE_FEATURE_DIM
    ):
        raise RuntimeError(
            f"layout de features inválido: matrix={matrix.shape[1]}, "
            f"names={len(feature_names)}, esperado={POSE_FEATURE_DIM}"
        )
    return matrix, feature_names


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _block_range(block_name: str) -> tuple[int, int]:
    for name, start, end in feature_blocks():
        if name == block_name:
            return start, end
    raise KeyError(f"bloco de features desconhecido: {block_name}")


def _column_index(column_name: str) -> int:
    return _feature_names().index(column_name)


def run_report(*, adapter: DatasetAdapter) -> None:
    frames = adapter.load_frames()
    video_ids = [str(video_id) for video_id in frames["video_id"].unique()]
    group_sizes = cast(pd.Series, frames.groupby("video_id").size())

    derivative_blocks = [
        _block_range(name)
        for name in (
            "kp_velocity",
            "kp_acceleration",
            "bbox_velocity",
            "bbox_acceleration",
        )
    ]
    acceleration_blocks = [
        _block_range(name) for name in ("kp_acceleration", "bbox_acceleration")
    ]
    dtheta_col = _column_index("trunk_dtheta")

    matrices: list[np.ndarray] = []
    k_mismatches: list[str] = []
    prefix_violations: list[str] = []
    acquisition_violations: list[str] = []
    videos_with_prefix = 0
    prefix_rows_total = 0
    for video_id in video_ids:
        matrix = build_pose_features(video_id, pose_root=adapter.pose_root)[0]
        matrices.append(matrix)
        expected_rows = int(cast(int, group_sizes[video_id]))
        if matrix.shape[0] != expected_rows:
            k_mismatches.append(
                f"{video_id} (build_pose_features={matrix.shape[0]}, "
                f"frames.parquet={expected_rows})"
            )

        person_found = load_person_found(video_id, pose_root=adapter.pose_root)
        prefix = first_observed_index(person_found)
        if prefix > 0:
            videos_with_prefix += 1
            prefix_rows_total += prefix
            leading = matrix[:prefix]
            if not np.array_equal(
                leading, np.zeros((prefix, matrix.shape[1]), dtype=np.float32)
            ):
                nonzero_rows = int(np.sum(np.any(leading != 0.0, axis=1)))
                prefix_violations.append(
                    f"{video_id} (prefixo={prefix}, linhas não zeradas={nonzero_rows})"
                )

        k = matrix.shape[0]
        offenders: list[str] = []
        if prefix < k:
            for start, end in derivative_blocks:
                if np.any(matrix[prefix, start:end] != 0.0):
                    offenders.append(f"f[{start}:{end}]")
            if matrix[prefix, dtheta_col] != np.float32(0.0):
                offenders.append("f[trunk_dtheta]")
        if prefix + 1 < k:
            for start, end in acceleration_blocks:
                if np.any(matrix[prefix + 1, start:end] != 0.0):
                    offenders.append(f"f+1[{start}:{end}]")
        if offenders:
            acquisition_violations.append(
                f"{video_id} (f={prefix}, colunas não zeradas: {', '.join(offenders)})"
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

    print(
        "\n=== checagem: prefixo anterior à primeira detecção é exatamente zero ==="
    )
    print(
        f"vídeos com prefixo sem detecção: {videos_with_prefix} "
        f"(esperado {EXPECTED_VIDEOS_WITH_PREFIX}), "
        f"linhas de prefixo: {prefix_rows_total} "
        f"(esperado {EXPECTED_PREFIX_ROWS})"
    )
    if prefix_violations:
        print("video_ids com prefixo não zerado:")
        for violation in prefix_violations:
            print(f"  {violation}")
    ok_prefix_zero = _check(
        "linhas anteriores à primeira detecção são zero nas 134 colunas, "
        "para todo video_id",
        len(prefix_violations) == 0,
    )
    # Sem estas duas contagens congeladas a checagem acima passaria vazia: um
    # first_observed_index quebrado que devolvesse 0 pularia todo vídeo.
    ok_prefix_videos = _check(
        f"vídeos com prefixo sem detecção == {EXPECTED_VIDEOS_WITH_PREFIX}",
        videos_with_prefix == EXPECTED_VIDEOS_WITH_PREFIX,
    )
    ok_prefix_rows = _check(
        f"linhas de prefixo == {EXPECTED_PREFIX_ROWS}",
        prefix_rows_total == EXPECTED_PREFIX_ROWS,
    )

    print("\n=== checagem: derivadas na aquisição (linhas f e f+1) ===")
    if acquisition_violations:
        print("video_ids com derivada não zerada na aquisição:")
        for violation in acquisition_violations:
            print(f"  {violation}")
    ok_acquisition = _check(
        "linha f tem velocidade, aceleração e trunk_dtheta zero e linha f+1 tem "
        "aceleração zero, para todo video_id",
        len(acquisition_violations) == 0,
    )

    ok_rows = _check(f"total de linhas == {EXPECTED_K_SUM}", total_rows == EXPECTED_K_SUM)
    ok_finite = _check("nenhum valor não finito", non_finite == 0)

    if not (
        ok_rows
        and ok_finite
        and ok_k_per_video
        and ok_prefix_zero
        and ok_prefix_videos
        and ok_prefix_rows
        and ok_acquisition
    ):
        print("\npose kinematics report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose kinematics report OK: todas as checagens passaram")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    selftest_parser = subparsers.add_parser(
        "selftest",
        help="Roda checagens sintéticas dos descritores cinemáticos",
    )
    selftest_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)
    report_parser = subparsers.add_parser(
        "report",
        help="Roda build_pose_features sobre todos os vídeos e reporta estatísticas",
    )
    report_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.pose.selftests.kinematics import run_selftest

        run_selftest()
    elif args.command == "report":
        run_report(adapter=get_dataset(args.dataset))


if __name__ == "__main__":
    main()
