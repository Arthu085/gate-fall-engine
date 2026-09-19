"""Dataset de janelas fundidas pose+DINOv3 (arma B0) e CLI de selftest."""

import argparse
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from gatefall.data.windowing import build_window_index, window_frame_indices

_POSE_DIM = 134
_VISUAL_DIM = 1536


class FusionWindowDataset:
    def __init__(
        self,
        frames: pd.DataFrame,
        split: str,
        stride: int,
        pose_loader: Callable[[str], np.ndarray],
        visual_loader: Callable[[str], np.ndarray],
        drop_ignored: bool = True,
    ) -> None:
        self._pose_loader = pose_loader
        self._visual_loader = visual_loader
        split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
        self._windows = build_window_index(
            split_frames, stride=stride, drop_ignored=drop_ignored
        )
        self._expected_k_by_video: dict[str, int] = {
            str(video_id): int(group["n_frames"].iloc[0])
            for video_id, group in self._windows.groupby("video_id", sort=False)
        }
        self._pose_cache: dict[str, np.ndarray] = {}
        self._visual_cache: dict[str, np.ndarray] = {}
        for video_id in pd.unique(cast(pd.Series, self._windows["video_id"])):
            self._features_for_video(str(video_id))

    def _features_for_video(self, video_id: str) -> tuple[np.ndarray, np.ndarray]:
        if video_id not in self._pose_cache:
            pose_array = self._pose_loader(video_id)
            visual_array = self._visual_loader(video_id)
            expected_k = self._expected_k_by_video[video_id]
            if pose_array.shape[0] != expected_k:
                raise ValueError(
                    f"video_id={video_id!r}: fonte pose tem K={pose_array.shape[0]}, "
                    f"esperado K={expected_k} (n_frames da tabela de frames)"
                )
            if visual_array.shape[0] != expected_k:
                raise ValueError(
                    f"video_id={video_id!r}: fonte visual tem K={visual_array.shape[0]}, "
                    f"esperado K={expected_k} (n_frames da tabela de frames)"
                )
            if pose_array.shape[0] != visual_array.shape[0]:
                raise ValueError(
                    f"video_id={video_id!r}: pose array shape {pose_array.shape} "
                    f"e visual array shape {visual_array.shape} têm K diferentes"
                )
            if pose_array.shape[1] != _POSE_DIM:
                raise ValueError(
                    f"video_id={video_id!r}: pose array shape {pose_array.shape} "
                    f"tem feature_dim != {_POSE_DIM}"
                )
            if visual_array.shape[1] != _VISUAL_DIM:
                raise ValueError(
                    f"video_id={video_id!r}: visual array shape "
                    f"{visual_array.shape} tem feature_dim != {_VISUAL_DIM}"
                )
            self._pose_cache[video_id] = pose_array
            self._visual_cache[video_id] = visual_array
        return self._pose_cache[video_id], self._visual_cache[video_id]

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, int, tuple[str, int]]:
        row = self._windows.iloc[index]
        video_id = str(row["video_id"])
        k_end = int(row["k_end"])
        n_frames = int(row["n_frames"])
        label = int(row["label"])
        frame_indices = window_frame_indices(k_end, n_frames)
        pose_array, visual_array = self._features_for_video(video_id)
        pose_window = pose_array[frame_indices].astype(np.float32)
        visual_window = visual_array[frame_indices].astype(np.float32)
        return pose_window, visual_window, label, (video_id, k_end)


def main() -> None:
    from gatefall.data.fusion_dataset_selftest import run_fusion_dataset_selftest

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "selftest", help="Verifica o FusionWindowDataset contra entradas sintéticas"
    )

    args = parser.parse_args()
    if args.command == "selftest":
        ok = run_fusion_dataset_selftest()
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
