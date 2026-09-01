"""Dataset genérico de janelas e CLI dos diagnósticos de pose."""

import argparse
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from gatefall.data.windowing import build_window_index, window_frame_indices


class PoseWindowDataset:
    def __init__(
        self,
        frames: pd.DataFrame,
        split: str,
        stride: int,
        feature_loader: Callable[[str], np.ndarray],
        drop_ignored: bool = True,
    ) -> None:
        self._feature_loader = feature_loader
        split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
        self._windows = build_window_index(
            split_frames, stride=stride, drop_ignored=drop_ignored
        )
        self._feature_cache: dict[str, np.ndarray] = {}

    def _features_for_video(self, video_id: str) -> np.ndarray:
        if video_id not in self._feature_cache:
            self._feature_cache[video_id] = self._feature_loader(video_id)
        return self._feature_cache[video_id]

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, tuple[str, int]]:
        row = self._windows.iloc[index]
        video_id = str(row["video_id"])
        k_end = int(row["k_end"])
        n_frames = int(row["n_frames"])
        label = int(row["label"])
        frame_indices = window_frame_indices(k_end, n_frames)
        window = self._features_for_video(video_id)[frame_indices].astype(np.float32)
        return window, label, (video_id, k_end)


def main() -> None:
    from gatefall.data.le2i.pose_dataset import report_pose_dataset
    from gatefall.data.le2i.pose_dataset_selftest import run_pose_dataset_selftest

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report", help="Relata as contagens reais de janelas do PoseWindowDataset"
    )
    report_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    selftest_parser = subparsers.add_parser(
        "selftest", help="Verifica o PoseWindowDataset contra entradas sintéticas"
    )
    selftest_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    args = parser.parse_args()
    if args.command == "report":
        report_pose_dataset()
    elif args.command == "selftest":
        run_pose_dataset_selftest()


if __name__ == "__main__":
    main()
