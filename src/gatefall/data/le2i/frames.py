"""Construção e persistência da tabela de quadros da grade do Le2i."""

import sys
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.data.frames import apply_frame_schema, read_frames, write_frames
from gatefall.data.le2i.annotations import load_annotation_splits
from gatefall.data.le2i.timeline import build_grid_frames
from gatefall.data.le2i.verification import load_le2i_manifest
from gatefall.hashing import sha256_dataframe

FRAMES_PATH = Path("data/labels/le2i/frames.parquet")


def build_le2i_frames_table(
    grid_frames: pd.DataFrame, per_video: pd.DataFrame
) -> pd.DataFrame:
    subject_by_video = cast(pd.Series, per_video.set_index("video_id")["subject"])
    video_id_column = cast(pd.Series, grid_frames["video_id"])
    with_subject = grid_frames.assign(subject=video_id_column.map(subject_by_video))
    ordered = cast(
        pd.DataFrame, with_subject.sort_values(["video_id", "frame_index"])
    ).reset_index(drop=True)
    return apply_frame_schema(ordered)


def build_le2i_timegrid() -> None:
    manifest = load_le2i_manifest()
    splits = load_annotation_splits()
    grid_frames, per_video, _skipped_segments = build_grid_frames(manifest, splits)

    frames = build_le2i_frames_table(grid_frames, per_video)
    write_frames(frames, FRAMES_PATH)

    reloaded = read_frames(FRAMES_PATH)
    if not frames.equals(reloaded):
        print(
            "\ntimegrid build FALHOU: o DataFrame relido de "
            f"{FRAMES_PATH} não é idêntico ao gravado (valores e/ou dtypes)",
            file=sys.stderr,
        )
        sys.exit(1)

    size_kb = FRAMES_PATH.stat().st_size / 1024
    content_hash = sha256_dataframe(frames)
    print(f"\n{FRAMES_PATH}: {size_kb:.2f} KB, {len(frames)} linhas")
    print("linhas por split:")
    for split, count in cast(pd.Series, frames.groupby("split").size()).items():
        print(f"  {split}: {count}")
    print(f"sha256 (conteúdo, hash_pandas_object): {content_hash}")
    print("\ntimegrid build OK: releitura idêntica ao gravado")
