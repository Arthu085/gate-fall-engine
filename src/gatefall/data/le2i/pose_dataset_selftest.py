"""Selftest sintético do `PoseWindowDataset` (`pose_dataset.py`).

Não toca no dataset real — features vêm de um `feature_loader` sintético.
"""

import sys
from typing import Callable, cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, WINDOW_FRAMES
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import build_window_index

_D = 3
_N_FRAMES_A = 30
_N_FRAMES_B = 32


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _make_frames() -> pd.DataFrame:
    video_a = pd.DataFrame(
        {
            "video_id": "video_a",
            "split": "train",
            "env": "coffee_room",
            "subject": 1,
            "frame_index": np.arange(_N_FRAMES_A, dtype=np.int32),
            "label": np.zeros(_N_FRAMES_A, dtype=np.int8),
        }
    )
    video_b = pd.DataFrame(
        {
            "video_id": "video_b",
            "split": "train",
            "env": "office",
            "subject": 2,
            "frame_index": np.arange(_N_FRAMES_B, dtype=np.int32),
            "label": np.zeros(_N_FRAMES_B, dtype=np.int8),
        }
    )
    return pd.concat([video_a, video_b], ignore_index=True)


def _make_frames_with_ignore() -> pd.DataFrame:
    n_frames = 26
    labels = np.zeros(n_frames, dtype=np.int8)
    labels[10] = IGNORE_LABEL
    labels[25] = IGNORE_LABEL
    return pd.DataFrame(
        {
            "video_id": "video_c",
            "split": "train",
            "env": "home",
            "subject": 3,
            "frame_index": np.arange(n_frames, dtype=np.int32),
            "label": labels,
        }
    )


def _synthetic_feature_loader(
    n_frames_by_video: dict[str, int],
) -> Callable[[str], np.ndarray]:
    def loader(video_id: str) -> np.ndarray:
        n_frames = n_frames_by_video[video_id]
        base = (np.arange(n_frames, dtype=np.float32) + 1).reshape(n_frames, 1)
        return np.tile(base, (1, _D))

    return loader


def check_unpadded_window_matches_source_rows() -> bool:
    frames = _make_frames()
    loader = _synthetic_feature_loader(
        {"video_a": _N_FRAMES_A, "video_b": _N_FRAMES_B}
    )
    dataset = PoseWindowDataset(frames, split="train", stride=1, feature_loader=loader)

    k_end = _N_FRAMES_A - 1
    windows = build_window_index(
        cast(pd.DataFrame, frames[frames["split"] == "train"]), stride=1, drop_ignored=True
    )
    match = windows[
        (windows["video_id"] == "video_a") & (windows["k_end"] == k_end)
    ]
    index = int(cast(int, match.index[0]))

    window, _label, (video_id, returned_k_end) = dataset[index]
    expected_rows = (
        np.arange(k_end - WINDOW_FRAMES + 1, k_end + 1, dtype=np.float32) + 1
    )
    ok = (
        window.shape == (WINDOW_FRAMES, _D)
        and bool(np.array_equal(window[:, 0], expected_rows))
        and video_id == "video_a"
        and returned_k_end == k_end
    )
    return _check(
        "k_end >= WINDOW_FRAMES-1: janela cobre exatamente as linhas "
        "k_end-23..k_end, em ordem",
        ok,
    )


def check_padded_window_repeats_leading_row() -> bool:
    frames = _make_frames()
    loader = _synthetic_feature_loader(
        {"video_a": _N_FRAMES_A, "video_b": _N_FRAMES_B}
    )
    dataset = PoseWindowDataset(frames, split="train", stride=1, feature_loader=loader)

    k_end = 5
    windows = build_window_index(
        cast(pd.DataFrame, frames[frames["split"] == "train"]), stride=1, drop_ignored=True
    )
    match = windows[
        (windows["video_id"] == "video_a") & (windows["k_end"] == k_end)
    ]
    index = int(cast(int, match.index[0]))

    window, _label, _diag = dataset[index]
    n_padding = WINDOW_FRAMES - 1 - k_end
    row_0 = loader("video_a")[0, 0]
    padded_ok = bool(np.all(window[:n_padding, 0] == row_0))
    remainder_ok = bool(
        np.array_equal(
            window[n_padding:, 0],
            np.arange(k_end + 1, dtype=np.float32) + 1,
        )
    )
    return _check(
        "k_end < WINDOW_FRAMES-1: linha 0 repetida em (WINDOW_FRAMES-1-k_end) "
        "posições líderes",
        padded_ok and remainder_ok,
    )


def check_ignore_label_dropped_only_at_window_end() -> bool:
    frames = _make_frames_with_ignore()
    loader = _synthetic_feature_loader({"video_c": 26})
    dataset = PoseWindowDataset(frames, split="train", stride=1, feature_loader=loader)

    k_ends = set()
    for i in range(len(dataset)):
        _window, _label, (_video_id, k_end) = dataset[i]
        k_ends.add(k_end)

    ok = 10 not in k_ends and 25 not in k_ends
    ok = ok and 15 in k_ends
    return _check(
        "janela com k_end em IGNORE_LABEL está ausente; janela contendo "
        "IGNORE_LABEL só no contexto permanece presente",
        ok,
    )


def check_len_matches_usable_window_count() -> bool:
    frames = _make_frames()
    loader = _synthetic_feature_loader(
        {"video_a": _N_FRAMES_A, "video_b": _N_FRAMES_B}
    )
    dataset = PoseWindowDataset(frames, split="train", stride=1, feature_loader=loader)
    windows = build_window_index(
        cast(pd.DataFrame, frames[frames["split"] == "train"]), stride=1, drop_ignored=True
    )
    return _check("len(dataset) == contagem de janelas úteis", len(dataset) == len(windows))


def run_pose_dataset_selftest() -> None:
    checks = [
        check_unpadded_window_matches_source_rows(),
        check_padded_window_repeats_leading_row(),
        check_ignore_label_dropped_only_at_window_end(),
        check_len_matches_usable_window_count(),
    ]
    if not all(checks):
        print("\npose dataset selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose dataset selftest OK: todos os casos passaram")
