"""Relatório do índice de janelas do Le2i, lido a partir de `frames.parquet`."""

import sys
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.frames import read_frames
from gatefall.data.windowing import build_window_index, window_frame_indices
from gatefall.datasets.le2i import LE2I_DATASET, Le2iDatasetAdapter

EXPECTED_TOTAL_WINDOWS_STRIDE1: dict[str, dict[str, int]] = {
    "le2i": {
        "train": 22246,
        "val": 2080,
        "test": 6168,
    },
    "le2i-cv": {
        "train": 17905,
        "val": 6607,
        "test": 5982,
    },
}

EXPECTED_USABLE_WINDOWS_STRIDE1: dict[str, dict[str, int]] = {
    "le2i": {
        "train": 20740,
        "val": 2079,
        "test": 5616,
    },
    "le2i-cv": {
        "train": 16595,
        "val": 5922,
        "test": 5918,
    },
}


def expected_counts_for_identifier(
    table: dict[str, dict[str, int]], identifier: str
) -> dict[str, int]:
    if identifier not in table:
        raise ValueError(
            f"contagens esperadas ainda não medidas para {identifier}"
        )
    return table[identifier]


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def load_le2i_frames_for_windows(
    adapter: Le2iDatasetAdapter = LE2I_DATASET,
) -> pd.DataFrame:
    frames_path = adapter.frames_path
    if not frames_path.exists():
        print(
            f"\nwindows report FALHOU: {frames_path} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)
    return read_frames(frames_path)


def report_window_counts_by_stride(frames: pd.DataFrame, stride: int) -> pd.DataFrame:
    print(f"\n=== janelas por stride={stride} ===")
    total_windows = build_window_index(frames, stride=stride, drop_ignored=False)
    usable_windows = build_window_index(frames, stride=stride, drop_ignored=True)

    total_by_split = cast(pd.Series, total_windows.groupby("split").size())
    usable_by_split = cast(pd.Series, usable_windows.groupby("split").size())

    print("total de janelas por split (antes de descartar IGNORE_LABEL):")
    for split, count in total_by_split.items():
        print(f"  {split}: {count}")

    print("janelas úteis por split (depois de descartar IGNORE_LABEL):")
    for split, total in total_by_split.items():
        usable = int(cast(int, usable_by_split.get(split, 0)))
        dropped = int(total) - usable
        dropped_pct = (dropped / total * 100) if total else 0.0
        print(f"  {split}: {usable} úteis, {dropped} descartadas ({dropped_pct:.2f}%)")

    print("janelas úteis por (split, label):")
    print(usable_windows.groupby(["split", "label"]).size())

    print("janelas úteis por (split, env):")
    print(usable_windows.groupby(["split", "env"]).size())

    return usable_windows


def report_context_diagnostics(frames: pd.DataFrame, usable_windows: pd.DataFrame) -> None:
    print(f"\n=== diagnósticos de contexto, stride={TRAIN_STRIDE} (apenas TRAIN_STRIDE) ===")

    labels_by_video: dict[str, np.ndarray] = {
        str(video_id): cast(
            pd.DataFrame, group.sort_values("frame_index")
        )["label"].to_numpy()
        for video_id, group in frames.groupby("video_id", sort=False)
    }

    print("fração de janelas úteis cujo contexto contém ao menos um quadro IGNORE_LABEL:")
    for split, group in usable_windows.groupby("split"):
        group = cast(pd.DataFrame, group)
        total = len(group)
        with_ignore_context = 0
        video_ids = group["video_id"].to_numpy()
        k_ends = group["k_end"].to_numpy()
        n_frames_column = group["n_frames"].to_numpy()
        for video_id, k_end, n_frames in zip(video_ids, k_ends, n_frames_column):
            video_labels = labels_by_video[str(video_id)]
            context_indices = window_frame_indices(int(k_end), int(n_frames))
            if bool(np.any(video_labels[context_indices] == IGNORE_LABEL)):
                with_ignore_context += 1
        pct = (with_ignore_context / total * 100) if total else 0.0
        print(f"  {split}: {with_ignore_context}/{total} ({pct:.2f}%)")

    print("fração de janelas úteis com ao menos um quadro de edge padding:")
    for split, group in usable_windows.groupby("split"):
        group = cast(pd.DataFrame, group)
        total = len(group)
        padded = int((group["k_end"] < WINDOW_FRAMES - 1).sum())
        pct = (padded / total * 100) if total else 0.0
        print(f"  {split}: {padded}/{total} ({pct:.2f}%)")


def check_total_window_counts_stride1(frames: pd.DataFrame, identifier: str) -> bool:
    expected = expected_counts_for_identifier(EXPECTED_TOTAL_WINDOWS_STRIDE1, identifier)
    total_windows = build_window_index(frames, stride=EVAL_STRIDE, drop_ignored=False)
    counts = cast(pd.Series, total_windows.groupby("split").size())
    ok = all(
        int(cast(int, counts.get(split, -1))) == expected_count
        for split, expected_count in expected.items()
    )
    return _check(
        "stride=1, drop_ignored=False: contagem de janelas por split == "
        f"{expected}",
        ok,
    )


def check_usable_window_counts_stride1(frames: pd.DataFrame, identifier: str) -> bool:
    expected = expected_counts_for_identifier(EXPECTED_USABLE_WINDOWS_STRIDE1, identifier)
    usable_windows = build_window_index(frames, stride=EVAL_STRIDE, drop_ignored=True)
    counts = cast(pd.Series, usable_windows.groupby("split").size())
    ok = all(
        int(cast(int, counts.get(split, -1))) == expected_count
        for split, expected_count in expected.items()
    )
    return _check(
        "stride=1, drop_ignored=True: contagem de janelas por split == "
        f"{expected}",
        ok,
    )


def check_window_count_matches_ceil_per_video(frames: pd.DataFrame) -> bool:
    n_frames_by_video = cast(pd.Series, frames.groupby("video_id").size())
    ok = True
    for stride in (TRAIN_STRIDE, EVAL_STRIDE):
        total_windows = build_window_index(frames, stride=stride, drop_ignored=False)
        counts = cast(pd.Series, total_windows.groupby("video_id").size())
        for video_id, n_frames in n_frames_by_video.items():
            expected = -(-int(n_frames) // stride)
            ok = ok and int(cast(int, counts.get(video_id, -1))) == expected
    return _check(
        "para todo vídeo e ambos os strides, contagem de janelas antes do "
        "descarte == ceil(K / stride)",
        ok,
    )


def check_no_window_end_beyond_n_frames(frames: pd.DataFrame) -> bool:
    ok = True
    for stride in (TRAIN_STRIDE, EVAL_STRIDE):
        total_windows = build_window_index(frames, stride=stride, drop_ignored=False)
        ok = ok and bool((total_windows["k_end"] < total_windows["n_frames"]).all())
    return _check("nenhuma janela tem k_end >= n_frames, para ambos os strides", ok)


def report_le2i_windows(adapter: Le2iDatasetAdapter = LE2I_DATASET) -> None:
    frames = load_le2i_frames_for_windows(adapter)

    usable_by_stride: dict[int, pd.DataFrame] = {}
    for stride in (TRAIN_STRIDE, EVAL_STRIDE):
        usable_by_stride[stride] = report_window_counts_by_stride(frames, stride)

    report_context_diagnostics(frames, usable_by_stride[TRAIN_STRIDE])

    print("\n=== checagens críticas ===")
    checks = [
        check_total_window_counts_stride1(frames, adapter.identifier),
        check_usable_window_counts_stride1(frames, adapter.identifier),
        check_window_count_matches_ceil_per_video(frames),
        check_no_window_end_beyond_n_frames(frames),
    ]
    if not all(checks):
        print("\nwindows report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nwindows report OK: todas as checagens críticas passaram")
