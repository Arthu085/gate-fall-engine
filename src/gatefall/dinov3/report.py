"""Relatório de cobertura das features DINOv3 extraídas para o Le2i."""

import sys
from typing import cast

import h5py
import pandas as pd

from gatefall.datasets import DatasetAdapter
from gatefall.dinov3.storage import dinov3_path

EXPECTED_VIDEO_COUNT = 190
EXPECTED_TOTAL_FRAMES = 30494
EXPECTED_SPLIT_FRAME_COUNTS = {"train": 22246, "val": 2080, "test": 6168}


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def run_dinov3_report(adapter: DatasetAdapter) -> None:
    if not adapter.frames_path.exists():
        print(
            f"\ndinov3 report FALHOU: {adapter.frames_path} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = adapter.load_frames()
    group_sizes = cast(pd.Series, frames.groupby("video_id").size())
    split_by_video = cast(
        pd.Series, frames.groupby("video_id")["split"].first()
    )

    missing: list[str] = []
    mismatched: list[str] = []
    frames_by_split: dict[str, int] = {}
    total_bytes = 0

    for video_id, n_frames in group_sizes.items():
        video_id = str(video_id)
        split = str(split_by_video[video_id])
        path = dinov3_path(video_id, dinov3_root=adapter.dinov3_root)
        if not path.exists():
            missing.append(video_id)
            continue
        with h5py.File(path, "r") as h5_file:
            k = int(cast(int, h5_file.attrs["K"]))
        if k != int(n_frames):
            mismatched.append(f"{video_id} (K={k}, frames.parquet={int(n_frames)})")
            continue
        frames_by_split[split] = frames_by_split.get(split, 0) + k
        total_bytes += path.stat().st_size

    n_videos = len(group_sizes)
    total_frames = sum(frames_by_split.values())

    print(f"\nvídeos no manifesto de frames: {n_videos} (esperado {EXPECTED_VIDEO_COUNT})")
    if missing:
        print(f"\nvídeos sem .h5 de DINOv3 ({len(missing)}): {missing}")
    if mismatched:
        print(f"\nvídeos com K divergente ({len(mismatched)}): {mismatched}")

    print("\nquadros por split:")
    for split, expected in EXPECTED_SPLIT_FRAME_COUNTS.items():
        actual = frames_by_split.get(split, 0)
        print(f"  {split}: {actual} (esperado {expected})")
    print(f"  total: {total_frames} (esperado {EXPECTED_TOTAL_FRAMES})")

    print(f"\ntamanho total em disco: {total_bytes / (1024 ** 2):.2f} MiB")

    checks = [
        _check(f"vídeos == {EXPECTED_VIDEO_COUNT}", n_videos == EXPECTED_VIDEO_COUNT),
        _check("nenhum .h5 ausente", not missing),
        _check("nenhum K divergente", not mismatched),
        _check(f"total de quadros == {EXPECTED_TOTAL_FRAMES}", total_frames == EXPECTED_TOTAL_FRAMES),
    ]
    for split, expected in EXPECTED_SPLIT_FRAME_COUNTS.items():
        checks.append(
            _check(
                f"quadros do split {split} == {expected}",
                frames_by_split.get(split, 0) == expected,
            )
        )

    if not all(checks):
        print("\ndinov3 report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\ndinov3 report OK: todas as checagens passaram")
