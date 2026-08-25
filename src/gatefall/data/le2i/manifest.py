"""Construção do manifesto de vídeos Le2i."""

import shutil
import sys
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.data.le2i.annotations import (
    LE2I_DATASET_NAME,
    build_video_annotation_index,
    load_annotation_splits,
)
from gatefall.data.le2i.path_matching import (
    discover_extracted_videos,
    index_annotation_paths,
    match_annotation_paths_to_videos,
    normalize_annotation_video_path,
)
from gatefall.data.manifest import apply_manifest_schema, write_manifest
from gatefall.data.video_metadata import probe_video, resolve_frame_rate
from gatefall.hashing import sha256_file

RAW_DIR = Path("data/raw/le2i")
MANIFEST_PATH = Path("data/manifest.parquet")


def build_le2i_manifest(
    raw_directory: Path, annotation_index: pd.DataFrame
) -> pd.DataFrame:
    if shutil.which("ffprobe") is None:
        print("erro: ffprobe não encontrado no PATH", file=sys.stderr)
        sys.exit(1)

    local_videos = discover_extracted_videos(raw_directory)
    annotation_paths = index_annotation_paths(
        cast(pd.Series, annotation_index["path"])
    )
    matched_paths = match_annotation_paths_to_videos(
        local_videos, annotation_paths
    )
    annotation_records = annotation_index.set_index("path").to_dict("index")

    rows: list[dict[str, object]] = []
    for annotation_path, relative_path in matched_paths.items():
        absolute_path = (raw_directory / relative_path).resolve()
        metadata = probe_video(absolute_path)
        fps, fps_source = resolve_frame_rate(
            metadata["r_frame_rate"], metadata["avg_frame_rate"]
        )
        annotation = annotation_records[annotation_path]
        environment = annotation_path.split("/", 1)[0]

        rows.append(
            {
                "video_id": normalize_annotation_video_path(annotation_path),
                "dataset": LE2I_DATASET_NAME,
                "relative_path": str(relative_path),
                "absolute_path": str(absolute_path),
                "env": environment,
                "subject": int(annotation["subject"]),
                "cam": int(annotation["cam"]),
                "split": str(annotation["split"]),
                "fps": fps,
                "fps_source": fps_source,
                "n_frames_header": metadata["n_frames_header"],
                "n_frames_counted": metadata["n_frames_counted"],
                "duration_s": metadata["duration_s"],
                "width": metadata["width"],
                "height": metadata["height"],
                "codec": metadata["codec"],
                "sha256": sha256_file(absolute_path),
                "pose_status": "pending",
                "dino_status": "pending",
                "sam_status": "pending",
            }
        )

    manifest = apply_manifest_schema(pd.DataFrame(rows))
    return cast(
        pd.DataFrame, manifest.sort_values("video_id", ignore_index=True)
    )


def ingest_le2i_dataset(force: bool) -> None:
    if MANIFEST_PATH.exists() and not force:
        print(
            f"skip {MANIFEST_PATH} (já existe, use --force para sobrescrever)"
        )
        return

    annotation_splits = load_annotation_splits()
    annotation_index = build_video_annotation_index(annotation_splits)
    manifest = build_le2i_manifest(RAW_DIR, annotation_index)
    write_manifest(manifest, MANIFEST_PATH, force)
