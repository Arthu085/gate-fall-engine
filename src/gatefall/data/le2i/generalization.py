"""Relatório auditável de fatores de domínio por split (Le2i-CS e Le2i-CV).

Le2i-CV depende inteiramente da disjunção de ambiente/câmera entre splits
para ser evidência de generalização cross-environment; Le2i-CS sobrepõe
ambientes entre splits por desenho (é cross-subject, não cross-environment).
Resolução, fps e qualidade de pose co-variam com o ambiente em ambos os
protocolos — nenhum desses fatores pode ser isolado causalmente a partir
deste relatório.
"""

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, NUM_CLASSES, TRAIN_STRIDE
from gatefall.data.windowing import build_window_index
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.pose.loading import load_pose
from gatefall.pose.quality import pose_quality_from_arrays

NARRATIVE = (
    "Le2i-CV é evidência de generalização entre ambientes (cross-environment), "
    "mas constitui um domain shift confundido: ambiente/câmera, resolução, fps "
    "e qualidade de pose co-variam entre os splits de treino, validação e "
    "teste. Nenhuma diferença de desempenho observada pode ser atribuída "
    "causalmente a um único desses fatores isolado. Resultados cross-subject "
    "do Le2i-CS não devem ser descritos como evidência cross-domain: o treino "
    "Le2i-CS já contém os seis ambientes do Le2i, então ele não testa "
    "generalização a um ambiente não visto."
)

EXPECTED_CV_SPLIT_SHAPE: dict[str, dict[str, int]] = {
    "train": {"n_videos": 97, "n_segments": 490},
    "val": {"n_videos": 33, "n_segments": 195},
    "test": {"n_videos": 60, "n_segments": 282},
}

PERCENTILE_LABELS = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]
PERCENTILE_QS = [1, 5, 25, 50, 75, 95, 99]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    result = np.percentile(values, PERCENTILE_QS)
    return {label: float(value) for label, value in zip(PERCENTILE_LABELS, result)}


def report_environment_camera_disjointness(
    per_video: pd.DataFrame, *, critical: bool
) -> bool:
    print("\n=== disjunção de ambiente/câmera entre splits ===")
    any_overlap = False
    split_names = sorted(cast(pd.Series, per_video["split"]).unique())
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            first_rows = cast(pd.DataFrame, per_video[per_video["split"] == first])
            second_rows = cast(pd.DataFrame, per_video[per_video["split"] == second])
            env_overlap = set(first_rows["env"]) & set(second_rows["env"])
            cam_overlap = set(first_rows["cam"]) & set(second_rows["cam"])
            if env_overlap or cam_overlap:
                any_overlap = True
                print(
                    f"{first} x {second}: ambientes sobrepostos="
                    f"{sorted(env_overlap)}, câmeras sobrepostas="
                    f"{sorted(cam_overlap)}"
                )
    is_disjoint = not any_overlap
    if is_disjoint:
        print("OK: ambientes e câmeras disjuntos entre todos os splits")
    elif critical:
        print(
            "FALHA CRÍTICA: a validade do Le2i-CV como evidência "
            "cross-environment depende de ambientes/câmeras disjuntos entre "
            "splits"
        )
    else:
        print(
            "informativo: sobreposição de ambientes/câmeras entre splits é "
            "esperada no Le2i-CS (protocolo cross-subject, não "
            "cross-environment)"
        )
    return is_disjoint


def report_subject_overlap(per_video: pd.DataFrame) -> tuple[bool, list[int]]:
    print("\n=== sobreposição de subjects entre splits ===")
    overlapping: set[int] = set()
    split_names = sorted(cast(pd.Series, per_video["split"]).unique())
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            first_subjects = set(per_video[per_video["split"] == first]["subject"])
            second_subjects = set(per_video[per_video["split"] == second]["subject"])
            overlap = first_subjects & second_subjects
            if overlap:
                overlapping |= overlap
                print(f"{first} x {second}: subjects sobrepostos={sorted(overlap)}")
    is_disjoint = not overlapping
    if is_disjoint:
        print("OK: subjects disjuntos entre todos os splits")
    else:
        print(
            "informativo: subjects se sobrepõem entre splits — este protocolo "
            "é environment/camera-disjoint, não subject-disjoint (ids de "
            "subject do Le2i não são globalmente únicos entre ambientes)"
        )
    return is_disjoint, sorted(int(subject) for subject in overlapping)


def labels_absent_from_train(train_frames: pd.DataFrame) -> list[str]:
    train_windows = build_window_index(train_frames, stride=TRAIN_STRIDE, drop_ignored=True)
    present = set(int(label) for label in train_windows["label"].unique())
    return [
        LE2I_LABEL_NAMES[label_id]
        for label_id in range(NUM_CLASSES)
        if label_id not in present
    ]


def resolution_distribution(manifest_split: pd.DataFrame) -> dict[str, int]:
    counts = cast(pd.Series, manifest_split.groupby(["width", "height"]).size())
    result: dict[str, int] = {}
    for key, count in counts.items():
        width, height = cast(tuple[int, int], key)
        result[f"{int(width)}x{int(height)}"] = int(cast(int, count))
    return result


def fps_distribution(manifest_split: pd.DataFrame) -> dict[str, int]:
    counts = cast(pd.Series, manifest_split["fps"].value_counts())
    result: dict[str, int] = {}
    for fps_value, count in counts.items():
        result[str(float(cast(float, fps_value)))] = int(cast(int, count))
    return result


def pose_coverage_and_quality(
    video_ids: list[str], pose_root: Path
) -> tuple[float, dict[str, float]]:
    found_total = 0
    frame_total = 0
    q_pose_parts: list[np.ndarray] = []
    for video_id in video_ids:
        pose = load_pose(video_id, pose_root=pose_root)
        found_total += int(np.sum(pose.person_found))
        frame_total += pose.k
        components = pose_quality_from_arrays(
            pose.keypoints, pose.bbox, pose.person_found
        )
        q_pose_parts.append(components.q_pose)
    coverage = (found_total / frame_total) if frame_total else 0.0
    all_q_pose = (
        np.concatenate(q_pose_parts) if q_pose_parts else np.array([], dtype=np.float32)
    )
    return coverage, _percentiles(all_q_pose)


def build_split_block(
    manifest_split: pd.DataFrame,
    n_segments: int,
    frames_split: pd.DataFrame,
    pose_root: Path,
    labels_absent_from_train_names: list[str],
) -> dict[str, object]:
    video_ids = [str(video_id) for video_id in manifest_split["video_id"].unique()]
    train_windows = build_window_index(frames_split, stride=TRAIN_STRIDE, drop_ignored=True)
    eval_windows = build_window_index(frames_split, stride=EVAL_STRIDE, drop_ignored=True)
    coverage, q_pose_percentiles = pose_coverage_and_quality(video_ids, pose_root)
    return {
        "environments": sorted(str(env) for env in manifest_split["env"].unique()),
        "cameras": sorted(int(cam) for cam in manifest_split["cam"].unique()),
        "n_videos": len(video_ids),
        "n_segments": n_segments,
        "n_windows_train_stride": len(train_windows),
        "n_windows_eval_stride": len(eval_windows),
        "resolution_distribution": resolution_distribution(manifest_split),
        "fps_distribution": fps_distribution(manifest_split),
        "pose_detection_coverage": coverage,
        "q_pose_percentiles": q_pose_percentiles,
        "labels_absent_from_train": labels_absent_from_train_names,
    }


def build_generalization_report(
    manifest: pd.DataFrame,
    frames: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    pose_root: Path,
    *,
    identifier: str,
) -> dict[str, object]:
    critical = identifier != "le2i"
    per_video = cast(
        pd.DataFrame,
        manifest[["video_id", "env", "cam", "split", "subject"]].drop_duplicates(),
    )
    is_disjoint = report_environment_camera_disjointness(per_video, critical=critical)
    if critical and not is_disjoint:
        print(
            "\ngeneralization report FALHOU: ambientes/câmeras sobrepostos "
            "entre splits de um protocolo cuja validade depende de disjunção",
            file=sys.stderr,
        )
        sys.exit(1)

    subject_disjoint, subjects_overlapping = report_subject_overlap(per_video)

    train_frames = cast(pd.DataFrame, frames[frames["split"] == "train"])
    labels_absent = labels_absent_from_train(train_frames)
    if labels_absent:
        print(f"\nclasses ausentes do treino: {labels_absent}")

    split_names = sorted(cast(pd.Series, manifest["split"]).unique())
    split_report: dict[str, object] = {}
    for split in split_names:
        manifest_split = cast(pd.DataFrame, manifest[manifest["split"] == split])
        frames_split = cast(pd.DataFrame, frames[frames["split"] == split])
        n_segments = len(splits[split]) if split in splits else 0
        split_report[split] = build_split_block(
            manifest_split, n_segments, frames_split, pose_root, labels_absent
        )

    if identifier == "le2i-cv":
        for split, expected in EXPECTED_CV_SPLIT_SHAPE.items():
            block = split_report.get(split)
            if block is None:
                print(
                    f"\ngeneralization report FALHOU: split {split!r} ausente "
                    "no relatório do Le2i-CV",
                    file=sys.stderr,
                )
                sys.exit(1)
            block = cast(dict[str, object], block)
            actual_videos = block["n_videos"]
            actual_segments = block["n_segments"]
            if (
                actual_videos != expected["n_videos"]
                or actual_segments != expected["n_segments"]
            ):
                print(
                    f"\ngeneralization report FALHOU: split={split} tem "
                    f"n_videos={actual_videos}/n_segments={actual_segments}, "
                    f"esperado n_videos={expected['n_videos']}/"
                    f"n_segments={expected['n_segments']} — revisão pinada ou "
                    "configuração pode ter mudado",
                    file=sys.stderr,
                )
                sys.exit(1)

    return {
        "splits": split_report,
        "environment_camera_disjoint": is_disjoint,
        "subject_disjoint": subject_disjoint,
        "subjects_overlapping_across_splits": subjects_overlapping,
        "narrative": NARRATIVE,
    }
