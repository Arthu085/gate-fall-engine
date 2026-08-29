"""Dataset de janelas de pose do Le2i, com features carregadas de forma preguiçosa por vídeo."""

import sys
from typing import Callable, cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, NUM_CLASSES, TRAIN_STRIDE
from gatefall.data.frames import read_frames
from gatefall.data.le2i.frames import FRAMES_PATH
from gatefall.data.le2i.windows import EXPECTED_USABLE_WINDOWS_STRIDE1
from gatefall.data.windowing import build_window_index, window_frame_indices
from gatefall.pose.kinematics import build_pose_features
from gatefall.pose.loading import load_pose

# Convenção de nomes de label inferida rodando `windows report` em stride 4 e
# comparando as contagens por (split, label) contra os números nomeados desta
# tarefa — não existe outra fonte no repositório para essa correspondência.
# O índice 6 (lying) nunca ocorre no Le2i; é um placeholder para manter o
# comprimento da lista igual a NUM_CLASSES.
LABEL_NAMES: list[str] = [
    "walk",
    "fall",
    "fallen",
    "sit_down",
    "sitting",
    "lie_down",
    "lying",
    "stand_up",
    "standing",
    "other",
]
assert len(LABEL_NAMES) == NUM_CLASSES

EXPECTED_USABLE_WINDOWS_STRIDE4: dict[str, int] = {
    "train": 5219,
    "val": 527,
    "test": 1412,
}

EXPECTED_USABLE_WINDOWS_BY_LABEL_STRIDE4: dict[str, dict[str, int]] = {
    "walk": {"train": 1547, "val": 179, "test": 608},
    "fall": {"train": 499, "val": 63, "test": 115},
    "fallen": {"train": 624, "val": 52, "test": 108},
    "sit_down": {"train": 159, "val": 31, "test": 69},
    "sitting": {"train": 696, "val": 73, "test": 181},
    "lie_down": {"train": 0, "val": 1, "test": 0},
    "stand_up": {"train": 351, "val": 97, "test": 103},
    "standing": {"train": 173, "val": 23, "test": 12},
    "other": {"train": 1170, "val": 8, "test": 216},
}


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


class PoseWindowDataset:
    def __init__(
        self,
        frames: pd.DataFrame,
        split: str,
        stride: int,
        feature_loader: Callable[[str], np.ndarray],
    ) -> None:
        self._feature_loader = feature_loader
        split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
        self._windows = build_window_index(split_frames, stride=stride, drop_ignored=True)
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

        matrix = self._features_for_video(video_id)
        frame_indices = window_frame_indices(k_end, n_frames)
        window = matrix[frame_indices].astype(np.float32)

        return window, label, (video_id, k_end)


def load_le2i_pose_window_dataset(split: str, stride: int) -> PoseWindowDataset:
    frames = read_frames(FRAMES_PATH)
    return PoseWindowDataset(
        frames,
        split,
        stride,
        lambda video_id: build_pose_features(video_id)[0],
    )


def report_pose_dataset() -> None:
    frames = read_frames(FRAMES_PATH)
    splits = sorted(cast(list[str], frames["split"].unique().tolist()))

    def feature_loader(video_id: str) -> np.ndarray:
        return build_pose_features(video_id)[0]

    checks: list[bool] = []

    print(f"\n=== janelas úteis por split, stride={TRAIN_STRIDE} ===")
    datasets_stride4: dict[str, PoseWindowDataset] = {}
    for split in splits:
        dataset = PoseWindowDataset(frames, split, TRAIN_STRIDE, feature_loader)
        datasets_stride4[split] = dataset
        print(f"  {split}: {len(dataset)}")

    checks.append(
        _check(
            f"stride={TRAIN_STRIDE}: contagem de janelas úteis por split == "
            f"{EXPECTED_USABLE_WINDOWS_STRIDE4}",
            all(
                len(datasets_stride4[split]) == expected
                for split, expected in EXPECTED_USABLE_WINDOWS_STRIDE4.items()
            ),
        )
    )

    print(f"\n=== janelas úteis por split, stride={EVAL_STRIDE} ===")
    datasets_stride1: dict[str, PoseWindowDataset] = {}
    for split in splits:
        dataset = PoseWindowDataset(frames, split, EVAL_STRIDE, feature_loader)
        datasets_stride1[split] = dataset
        print(f"  {split}: {len(dataset)}")

    checks.append(
        _check(
            f"stride={EVAL_STRIDE}: contagem de janelas úteis por split == "
            f"{EXPECTED_USABLE_WINDOWS_STRIDE1}",
            all(
                len(datasets_stride1[split]) == expected
                for split, expected in EXPECTED_USABLE_WINDOWS_STRIDE1.items()
            ),
        )
    )

    print(f"\n=== janelas úteis por (split, label), stride={TRAIN_STRIDE} ===")
    counts_by_split_label: dict[str, dict[int, int]] = {}
    for split in splits:
        windows = build_window_index(
            cast(pd.DataFrame, frames[frames["split"] == split]),
            stride=TRAIN_STRIDE,
            drop_ignored=True,
        )
        counts = cast(pd.Series, windows.groupby("label").size())
        counts_by_split_label[split] = {
            int(cast(int, k)): int(v) for k, v in counts.items()
        }
        print(f"  {split}: {counts_by_split_label[split]}")

    for label_index, label_name in enumerate(LABEL_NAMES):
        if label_name not in EXPECTED_USABLE_WINDOWS_BY_LABEL_STRIDE4:
            continue
        expected_by_split = EXPECTED_USABLE_WINDOWS_BY_LABEL_STRIDE4[label_name]
        ok = all(
            counts_by_split_label.get(split, {}).get(label_index, 0) == expected
            for split, expected in expected_by_split.items()
        )
        checks.append(
            _check(
                f"stride={TRAIN_STRIDE}, label={label_name} ({label_index}): "
                f"contagem por split == {expected_by_split}",
                ok,
            )
        )

    print(f"\n=== diagnósticos de person_found, stride={TRAIN_STRIDE} ===")
    person_found_cache: dict[str, np.ndarray] = {}

    def person_found_for_video(video_id: str) -> np.ndarray:
        if video_id not in person_found_cache:
            person_found_cache[video_id] = load_pose(video_id).person_found
        return person_found_cache[video_id]

    for split in splits:
        windows = build_window_index(
            cast(pd.DataFrame, frames[frames["split"] == split]),
            stride=TRAIN_STRIDE,
            drop_ignored=True,
        )
        total = len(windows)
        if total == 0:
            print(f"  {split}: sem janelas")
            continue

        last_frame_missing = 0
        missing_frame_counts: list[int] = []
        for video_id, k_end, n_frames in zip(
            windows["video_id"].to_numpy(),
            windows["k_end"].to_numpy(),
            windows["n_frames"].to_numpy(),
        ):
            video_id = str(video_id)
            person_found = person_found_for_video(video_id)
            k_end = int(k_end)
            n_frames = int(n_frames)
            if not bool(person_found[k_end]):
                last_frame_missing += 1
            frame_indices = window_frame_indices(k_end, n_frames)
            missing_frame_counts.append(int(np.sum(~person_found[frame_indices])))

        last_frame_missing_pct = last_frame_missing / total * 100
        mean_missing = float(np.mean(missing_frame_counts))
        print(
            f"  {split}: k_end sem pessoa detectada = {last_frame_missing_pct:.2f}%, "
            f"média de quadros ausentes por janela = {mean_missing:.2f}"
        )

    if not all(checks):
        print("\npose dataset report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose dataset report OK: todas as checagens passaram")
