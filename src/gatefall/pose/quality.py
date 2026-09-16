"""Índice de qualidade da pose (`q_pose`) por quadro.

`q_pose` é um índice normalizado de confiabilidade em [0, 1], NÃO uma
probabilidade calibrada: mede o quanto a detecção de pose de um quadro
parece confiável (confiança do detector, cobertura de keypoints e
consistência estrutural com o quadro observado anterior), sem qualquer
garantia de calibração estatística.

É calculado a partir de `gatefall.pose.loading.load_pose` (dado bruto
persistido), sem passar por `impute_missing`: um quadro forward-filled
nunca deve ser tratado como observado, ou o componente temporal ficaria
espuriamente perfeito (comparando um quadro consigo mesmo via
preenchimento, em vez de com uma observação real).
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.pose.kinematics import COCO17_SKELETON_EDGES, EXPECTED_K_SUM
from gatefall.pose.loading import load_pose, normalize_keypoints

N_KEYPOINTS = 17
CONF_CLIP_MIN = 0.0
CONF_CLIP_MAX = 1.0
VALID_CONF_THRESHOLD = 0.5
MIN_VALID_EDGES = 3
TEMPORAL_LENGTH_EPS = 1e-6


@dataclass(frozen=True)
class QualityComponents:
    q_conf: np.ndarray
    q_valid: np.ndarray
    q_temporal: np.ndarray
    q_pose: np.ndarray


def clip_confidence(conf: np.ndarray) -> np.ndarray:
    return np.clip(conf, CONF_CLIP_MIN, CONF_CLIP_MAX).astype(np.float32)


def compute_q_conf(conf_clipped: np.ndarray) -> np.ndarray:
    return conf_clipped.mean(axis=-1).astype(np.float32)


def compute_q_valid(conf_clipped: np.ndarray) -> np.ndarray:
    valid = conf_clipped >= VALID_CONF_THRESHOLD
    return valid.mean(axis=-1).astype(np.float32)


def _edge_lengths(xy_frame: np.ndarray, conf_frame: np.ndarray) -> dict[tuple[int, int], float]:
    lengths: dict[tuple[int, int], float] = {}
    for a, b in COCO17_SKELETON_EDGES:
        if conf_frame[a] < VALID_CONF_THRESHOLD or conf_frame[b] < VALID_CONF_THRESHOLD:
            continue
        diff = xy_frame[a] - xy_frame[b]
        length = float(np.hypot(diff[0], diff[1]))
        if not np.isfinite(length):
            continue
        lengths[(a, b)] = length
    return lengths


def compute_q_temporal(
    xy: np.ndarray, conf_clipped: np.ndarray, person_found: np.ndarray
) -> np.ndarray:
    k = xy.shape[0]
    q_temporal = np.ones(k, dtype=np.float32)

    last_observed_index: int | None = None
    for t in range(k):
        if not person_found[t]:
            continue
        if last_observed_index is None:
            q_temporal[t] = 1.0
            last_observed_index = t
            continue

        prev = last_observed_index
        prev_lengths = _edge_lengths(xy[prev], conf_clipped[prev])
        curr_lengths = _edge_lengths(xy[t], conf_clipped[t])
        common_edges = set(prev_lengths) & set(curr_lengths)

        if len(common_edges) < MIN_VALID_EDGES:
            q_temporal[t] = 1.0
        else:
            relative_changes = []
            for edge in common_edges:
                l_prev = prev_lengths[edge]
                l_curr = curr_lengths[edge]
                denom = max(l_curr, l_prev, TEMPORAL_LENGTH_EPS)
                relative_changes.append(abs(l_curr - l_prev) / denom)
            q_temporal[t] = float(np.clip(1.0 - np.median(relative_changes), 0.0, 1.0))

        last_observed_index = t

    return q_temporal.astype(np.float32)


def pose_quality_from_arrays(
    keypoints: np.ndarray, bbox: np.ndarray, person_found: np.ndarray
) -> QualityComponents:
    conf_clipped = clip_confidence(keypoints[:, :, 2])
    xy, _ = normalize_keypoints(keypoints, bbox, person_found)

    q_conf = compute_q_conf(conf_clipped)
    q_valid = compute_q_valid(conf_clipped)
    q_temporal = compute_q_temporal(xy, conf_clipped, person_found)
    q_pose = (q_conf * q_valid * q_temporal).astype(np.float32)

    absent = ~person_found
    q_conf = q_conf.copy()
    q_valid = q_valid.copy()
    q_temporal = q_temporal.copy()
    q_pose = q_pose.copy()
    q_conf[absent] = 0.0
    q_valid[absent] = 0.0
    q_temporal[absent] = 0.0
    q_pose[absent] = 0.0

    return QualityComponents(
        q_conf=q_conf, q_valid=q_valid, q_temporal=q_temporal, q_pose=q_pose
    )


def compute_pose_quality(video_id: str, *, pose_root: Path) -> QualityComponents:
    pose = load_pose(video_id, pose_root=pose_root)
    return pose_quality_from_arrays(pose.keypoints, pose.bbox, pose.person_found)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _percentiles(values: np.ndarray) -> dict[str, float]:
    labels = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]
    qs = [1, 5, 25, 50, 75, 95, 99]
    result = np.percentile(values, qs)
    return {label: float(value) for label, value in zip(labels, result)}


def run_report(*, adapter: DatasetAdapter) -> None:
    frames = adapter.load_frames()
    video_ids = [str(video_id) for video_id in frames["video_id"].unique()]

    rows: list[pd.DataFrame] = []
    for video_id in video_ids:
        pose = load_pose(video_id, pose_root=adapter.pose_root)
        components = pose_quality_from_arrays(
            pose.keypoints, pose.bbox, pose.person_found
        )
        video_frames = cast(
            pd.DataFrame, frames[frames["video_id"] == video_id]
        ).sort_values("frame_index")

        video_df = pd.DataFrame(
            {
                "video_id": video_id,
                "frame_index": np.arange(pose.k, dtype=np.int32),
                "person_found": pose.person_found,
                "q_conf": components.q_conf,
                "q_valid": components.q_valid,
                "q_temporal": components.q_temporal,
                "q_pose": components.q_pose,
            }
        )
        merged = video_df.merge(
            video_frames[["frame_index", "split"]], on="frame_index", how="left"
        )
        rows.append(merged)

    all_frames = pd.concat(rows, ignore_index=True)
    total_rows = len(all_frames)

    print(f"\nvídeos processados: {len(video_ids)}")
    print(f"total de linhas: {total_rows} (esperado {EXPECTED_K_SUM})")

    component_columns = ["q_conf", "q_valid", "q_temporal", "q_pose"]
    ok_bounds = True
    for column in component_columns:
        values = all_frames[column].to_numpy()
        column_ok = bool(np.isfinite(values).all()) and bool(
            np.all(values >= 0.0) and np.all(values <= 1.0)
        )
        ok_bounds = ok_bounds and column_ok
    ok_bounds = _check(
        "todos os componentes são finitos e limitados a [0,1]", ok_bounds
    )

    absent_mask = ~all_frames["person_found"].to_numpy()
    ok_absent_zero = True
    for column in component_columns:
        absent_values = all_frames.loc[absent_mask, column].to_numpy()
        ok_absent_zero = ok_absent_zero and bool(np.all(absent_values == 0.0))
    ok_absent_zero = _check(
        "q_conf, q_valid, q_temporal e q_pose == 0.0 exatamente em todo "
        "quadro person_found == False",
        ok_absent_zero,
    )

    ok_row_count = _check(
        f"total de linhas == EXPECTED_K_SUM ({EXPECTED_K_SUM})",
        total_rows == EXPECTED_K_SUM,
    )

    ok_split_not_null = _check(
        "nenhuma linha com split nulo após o merge",
        bool(all_frames["split"].notna().all()),
    )

    print(
        "\naviso: as seções val/test abaixo são diagnósticas e descritivas — "
        "nenhuma alegação de validação deve se apoiar nelas. Evidência de "
        "degradação é exclusiva do split train; nesta mudança a validação de "
        "degradação é inteiramente sintética (sem varredura de corrupção em "
        "dado real)."
    )

    for split, group in all_frames.groupby("split"):
        group = cast(pd.DataFrame, group)
        print(f"\n=== split={split} (n={len(group)}) ===")
        for column in component_columns:
            values = group[column].to_numpy()
            percentiles = _percentiles(values)
            percentiles_str = ", ".join(
                f"{label}={value:.4f}" for label, value in percentiles.items()
            )
            print(f"  {column}: n={len(values)}, {percentiles_str}")

    if not (ok_bounds and ok_absent_zero and ok_row_count and ok_split_not_null):
        print("\npose quality report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose quality report OK: todas as checagens passaram")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "selftest",
        help="Roda os casos sintéticos do índice de qualidade de pose",
    )
    subparsers.add_parser(
        "degradation",
        help="Roda a varredura determinística de severidade por canal de degradação",
    )
    report_parser = subparsers.add_parser(
        "report",
        help="Roda o índice de qualidade de pose sobre todos os vídeos e reporta estatísticas",
    )
    report_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.pose.quality_selftest import run_selftest

        run_selftest()
    elif args.command == "degradation":
        from gatefall.pose.quality_selftest import run_degradation

        run_degradation()
    elif args.command == "report":
        run_report(adapter=get_dataset(args.dataset))


if __name__ == "__main__":
    main()
