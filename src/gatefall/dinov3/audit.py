"""Auditoria de qualidade das features DINOv3 já extraídas para o Le2i."""

import sys
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from gatefall.datasets import DatasetAdapter
from gatefall.dinov3 import storage
from gatefall.dinov3.storage import dinov3_path
from gatefall.pose.loading import load_pose

FLOAT16_MAX = 65504.0


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def count_non_finite(features: np.ndarray) -> int:
    return int(np.sum(~np.isfinite(features)))


def max_abs_and_headroom(
    features: np.ndarray, *, ceiling: float = FLOAT16_MAX
) -> tuple[float, float]:
    max_abs = float(np.max(np.abs(features)))
    headroom = ceiling - max_abs
    return max_abs, headroom


def count_duplicate_consecutive_rows(features: np.ndarray) -> int:
    k = features.shape[0]
    return sum(
        1 for i in range(1, k) if np.array_equal(features[i], features[i - 1])
    )


def frame_index_is_contiguous(frame_indices: list[int]) -> bool:
    return sorted(frame_indices) == list(range(len(frame_indices)))


@dataclass
class DimensionStatsAccumulator:
    count: int = 0
    sum_: np.ndarray | None = field(default=None)
    sumsq: np.ndarray | None = field(default=None)

    def update(self, features: np.ndarray) -> None:
        values = features.astype(np.float64)
        if self.sum_ is None or self.sumsq is None:
            self.sum_ = np.zeros(values.shape[1], dtype=np.float64)
            self.sumsq = np.zeros(values.shape[1], dtype=np.float64)
        self.sum_ += values.sum(axis=0)
        self.sumsq += (values**2).sum(axis=0)
        self.count += values.shape[0]

    def variance(self) -> np.ndarray:
        assert self.sum_ is not None and self.sumsq is not None
        mean = self.sum_ / self.count
        mean_of_squares = self.sumsq / self.count
        return mean_of_squares - mean**2

    def dead_dimensions(self, *, eps: float = 0.0) -> list[int]:
        variance = self.variance()
        return [int(i) for i in np.where(variance <= eps)[0]]


def run_dinov3_audit(*, adapter: DatasetAdapter) -> None:
    if not adapter.frames_path.exists():
        print(
            f"\ndinov3 audit FALHOU: {adapter.frames_path} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = adapter.load_frames()
    per_video_ids = cast(list[str], sorted(frames["video_id"].unique().tolist()))

    accumulator = DimensionStatsAccumulator()
    non_finite_failures: list[str] = []
    duplicate_failures: list[str] = []
    frame_index_failures: list[str] = []
    pose_k_failures: list[str] = []
    max_abs_dataset_wide = 0.0

    for video_id in per_video_ids:
        video_id = str(video_id)
        path = dinov3_path(video_id, dinov3_root=adapter.dinov3_root)
        features = storage.read_features(path)

        non_finite_count = count_non_finite(features)
        if non_finite_count != 0:
            non_finite_failures.append(f"{video_id} ({non_finite_count} valores)")

        accumulator.update(features)

        duplicate_count = count_duplicate_consecutive_rows(features)
        if duplicate_count != 0:
            duplicate_failures.append(f"{video_id} ({duplicate_count} linhas)")

        max_abs, headroom = max_abs_and_headroom(features)
        max_abs_dataset_wide = max(max_abs_dataset_wide, max_abs)
        print(f"{video_id}: max_abs={max_abs:.2f}, headroom={headroom:.2f}")

        video_frames = cast(
            pd.DataFrame, frames[frames["video_id"] == video_id]
        ).sort_values("frame_index")
        frame_indices = [int(x) for x in video_frames["frame_index"]]
        if not frame_index_is_contiguous(frame_indices):
            frame_index_failures.append(video_id)

        try:
            pose = load_pose(video_id, pose_root=adapter.pose_root)
            if pose.k != features.shape[0]:
                pose_k_failures.append(
                    f"{video_id} (pose K={pose.k}, dinov3 K={features.shape[0]})"
                )
        except FileNotFoundError:
            pose_k_failures.append(f"{video_id} (pose .h5 ausente)")

    dead = accumulator.dead_dimensions()

    print(f"\nmax_abs no dataset inteiro: {max_abs_dataset_wide:.2f} (informativo)")
    if non_finite_failures:
        print(f"\nvídeos com valores não finitos ({len(non_finite_failures)}): {non_finite_failures}")
    if duplicate_failures:
        print(f"\nvídeos com linhas consecutivas duplicadas ({len(duplicate_failures)}): {duplicate_failures}")
    if frame_index_failures:
        print(f"\nvídeos com frame_index não contíguo ({len(frame_index_failures)}): {frame_index_failures}")
    if pose_k_failures:
        print(f"\nvídeos com K de pose divergente ({len(pose_k_failures)}): {pose_k_failures}")
    if dead:
        print(f"\ndimensões mortas (variância <= 0): {dead}")

    checks = [
        _check("nenhum valor não finito em nenhum vídeo", not non_finite_failures),
        _check(
            "nenhuma linha consecutiva duplicada em nenhum vídeo",
            not duplicate_failures,
        ),
        _check(
            "frame_index contíguo (0..K-1) em todos os vídeos",
            not frame_index_failures,
        ),
        _check("K de pose == K de dinov3 em todos os vídeos", not pose_k_failures),
        _check(
            "nenhuma dimensão morta (variância > 0 em todas as 1536 dimensões)",
            not dead,
        ),
    ]

    if not all(checks):
        print("\ndinov3 audit FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\ndinov3 audit OK: todas as checagens passaram")
