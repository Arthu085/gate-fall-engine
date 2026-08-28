"""Relatório de cobertura de pose e sua interação com o contrato de janelamento."""

import sys
from typing import cast

import h5py
import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.frames import read_frames
from gatefall.data.le2i.frames import FRAMES_PATH
from gatefall.data.windowing import build_window_index, window_frame_indices
from gatefall.pose.extract import POSE_ROOT, _output_path

EXPECTED_PERSON_FOUND_SUM = 27561
EXPECTED_K_SUM = 30494

MISSING_COUNT_BIN_EDGES: list[tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-6", 3, 6),
    ("7-12", 7, 12),
    ("13-23", 13, 23),
    (str(WINDOW_FRAMES), WINDOW_FRAMES, WINDOW_FRAMES),
]


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def load_le2i_frames_for_report() -> pd.DataFrame:
    if not FRAMES_PATH.exists():
        print(
            f"\npose report FALHOU: {FRAMES_PATH} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)
    return read_frames(FRAMES_PATH)


def load_pose_coverage(
    frames: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, int], list[str]]:
    person_found_by_video: dict[str, np.ndarray] = {}
    k_by_video: dict[str, int] = {}
    missing_h5: list[str] = []
    for video_id in frames["video_id"].unique():
        video_id = str(video_id)
        path = _output_path(video_id)
        if not path.exists():
            missing_h5.append(video_id)
            continue
        with h5py.File(path, "r") as h5_file:
            person_found_by_video[video_id] = cast(
                h5py.Dataset, h5_file["person_found"]
            )[()]
            k_by_video[video_id] = int(cast(int, h5_file.attrs["K"]))
    return person_found_by_video, k_by_video, missing_h5


def check_k_matches_frame_counts(
    frames: pd.DataFrame, k_by_video: dict[str, int], missing_h5: list[str]
) -> bool:
    print(f"\n=== checagem: K do .h5 (em {POSE_ROOT}) vs contagem de quadros em frames.parquet ===")
    group_sizes = cast(pd.Series, frames.groupby("video_id").size())
    mismatches: list[str] = []
    for video_id, n_frames in group_sizes.items():
        video_id = str(video_id)
        if video_id in missing_h5:
            mismatches.append(f"{video_id} (.h5 ausente)")
            continue
        k = k_by_video[video_id]
        if k != int(n_frames):
            mismatches.append(f"{video_id} (K={k}, frames.parquet={int(n_frames)})")
    if mismatches:
        print("video_ids com divergência:")
        for mismatch in mismatches:
            print(f"  {mismatch}")
    ok = len(mismatches) == 0
    return _check(
        "K do .h5 == contagem de quadros em frames.parquet, para todo video_id", ok
    )


def check_global_person_found_sums(
    person_found_by_video: dict[str, np.ndarray], k_by_video: dict[str, int]
) -> bool:
    print("\n=== checagem: somas globais de cobertura de pose ===")
    total_found = sum(int(arr.sum()) for arr in person_found_by_video.values())
    total_k = sum(k_by_video.values())
    print(f"soma de person_found: {total_found} (esperado {EXPECTED_PERSON_FOUND_SUM})")
    print(f"soma de K: {total_k} (esperado {EXPECTED_K_SUM})")
    ok_found = _check(
        f"soma de person_found == {EXPECTED_PERSON_FOUND_SUM}",
        total_found == EXPECTED_PERSON_FOUND_SUM,
    )
    ok_k = _check(f"soma de K == {EXPECTED_K_SUM}", total_k == EXPECTED_K_SUM)
    return ok_found and ok_k


def build_frames_with_person_found(
    frames: pd.DataFrame,
    person_found_by_video: dict[str, np.ndarray],
    k_by_video: dict[str, int],
) -> tuple[pd.DataFrame, list[str]]:
    group_sizes = cast(pd.Series, frames.groupby("video_id").size())
    usable_video_ids = {
        str(video_id)
        for video_id, n_frames in group_sizes.items()
        if str(video_id) in person_found_by_video
        and k_by_video[str(video_id)] == int(n_frames)
    }
    all_video_ids = {str(video_id) for video_id in frames["video_id"].unique()}
    excluded = sorted(all_video_ids - usable_video_ids)

    parts: list[pd.DataFrame] = []
    for video_id, group in frames.groupby("video_id", sort=False):
        video_id = str(video_id)
        if video_id not in usable_video_ids:
            continue
        ordered = cast(pd.DataFrame, group.sort_values("frame_index"))
        person_found = person_found_by_video[video_id][
            ordered["frame_index"].to_numpy()
        ]
        parts.append(ordered.assign(person_found=person_found))

    if not parts:
        usable_frames = frames.iloc[0:0].assign(
            person_found=pd.Series(dtype=bool)
        )
    else:
        usable_frames = pd.concat(parts, ignore_index=True)

    return usable_frames, excluded


def compute_window_pose_metrics(
    windows: pd.DataFrame, person_found_by_video: dict[str, np.ndarray]
) -> pd.DataFrame:
    video_ids = windows["video_id"].to_numpy()
    k_ends = windows["k_end"].to_numpy()
    n_frames_column = windows["n_frames"].to_numpy()

    missing_counts = np.empty(len(windows), dtype=np.int64)
    last_missing = np.empty(len(windows), dtype=bool)
    for i, (video_id, k_end, n_frames) in enumerate(
        zip(video_ids, k_ends, n_frames_column)
    ):
        person_found = person_found_by_video[str(video_id)]
        context_indices = window_frame_indices(int(k_end), int(n_frames))
        context_found = person_found[context_indices]
        missing_counts[i] = int(np.sum(~context_found))
        last_missing[i] = not bool(person_found[int(k_end)])

    return windows.assign(missing_count=missing_counts, last_missing=last_missing)


def _print_missing_count_distribution(group: pd.DataFrame) -> None:
    total = len(group)
    counts = group["missing_count"].to_numpy()
    for label, low, high in MISSING_COUNT_BIN_EDGES:
        n = int(np.sum((counts >= low) & (counts <= high)))
        pct = (n / total * 100) if total else 0.0
        print(f"    missing_count={label}: {n} ({pct:.2f}%)")


def _print_last_missing_pct(group: pd.DataFrame) -> None:
    total = len(group)
    n_last_missing = int(cast(int, group["last_missing"].sum()))
    pct = (n_last_missing / total * 100) if total else 0.0
    print(f"    last_missing: {n_last_missing}/{total} ({pct:.2f}%)")


def report_windows_by_key(
    windows_with_metrics: pd.DataFrame, key: str, title: str
) -> None:
    print(f"\n=== {title} ===")
    for value, group in windows_with_metrics.groupby(key):
        group = cast(pd.DataFrame, group)
        print(f"  {key}={value} (n={len(group)}):")
        _print_missing_count_distribution(group)
        _print_last_missing_pct(group)


def report_fall_windows(windows_with_metrics: pd.DataFrame) -> None:
    print("\n=== janelas com label == 1 (fall), por split ===")
    fall_windows = cast(
        pd.DataFrame, windows_with_metrics[windows_with_metrics["label"] == 1]
    )
    for split, group in fall_windows.groupby("split"):
        group = cast(pd.DataFrame, group)
        total = len(group)
        n_last_missing = int(cast(int, group["last_missing"].sum()))
        fraction_missing = group["missing_count"] / WINDOW_FRAMES
        mean_fraction_missing = (
            float(cast(float, fraction_missing.mean())) if total else 0.0
        )
        print(
            f"  {split}: n={total}, last_missing={n_last_missing}/{total}, "
            f"fração média ausente do contexto={mean_fraction_missing:.4f}"
        )


def report_frame_level_by_env(usable_frames: pd.DataFrame) -> None:
    print(
        "\n=== nível de quadro: fração sem pessoa detectada, por ambiente (env) ==="
    )
    for env, group in usable_frames.groupby("env"):
        group = cast(pd.DataFrame, group)
        is_leading = cast(
            pd.Series, (group["gap_position"] == "leading").fillna(False)
        )
        leading = cast(pd.DataFrame, group[is_leading])
        labeled = cast(pd.DataFrame, group[group["label"] != IGNORE_LABEL])
        leading_pct = (
            float(cast(float, (~leading["person_found"]).mean())) * 100
            if len(leading)
            else 0.0
        )
        labeled_pct = (
            float(cast(float, (~labeled["person_found"]).mean())) * 100
            if len(labeled)
            else 0.0
        )
        print(
            f"  {env}: leading sem pessoa={leading_pct:.2f}% (n={len(leading)}), "
            f"rotulado (label != IGNORE_LABEL) sem pessoa={labeled_pct:.2f}% "
            f"(n={len(labeled)})"
        )


def run_pose_report() -> None:
    frames = load_le2i_frames_for_report()

    person_found_by_video, k_by_video, missing_h5 = load_pose_coverage(frames)

    usable_frames, excluded_video_ids = build_frames_with_person_found(
        frames, person_found_by_video, k_by_video
    )
    if excluded_video_ids:
        print(
            "\naviso: vídeos excluídos das seções de janelamento por .h5 "
            f"ausente ou K divergente ({len(excluded_video_ids)}): "
            f"{excluded_video_ids}"
        )

    windows = build_window_index(usable_frames, stride=TRAIN_STRIDE, drop_ignored=True)
    windows_with_metrics = compute_window_pose_metrics(windows, person_found_by_video)

    report_windows_by_key(
        windows_with_metrics,
        "split",
        f"distribuição de missing_count e last_missing por split, stride={TRAIN_STRIDE}",
    )
    report_windows_by_key(
        windows_with_metrics,
        "env",
        f"distribuição de missing_count e last_missing por env, stride={TRAIN_STRIDE}",
    )
    report_fall_windows(windows_with_metrics)
    report_frame_level_by_env(usable_frames)

    print("\n=== checagens críticas ===")
    checks = [
        check_k_matches_frame_counts(frames, k_by_video, missing_h5),
        check_global_person_found_sums(person_found_by_video, k_by_video),
    ]
    if not all(checks):
        print("\npose report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose report OK: todas as checagens críticas passaram")
