"""Manifesto versionado, identidades, grade temporal e proveniência dos datasets encenados."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import TARGET_FPS, WINDOW_FRAMES
from gatefall.data.frames import read_frames, write_frames
from gatefall.data.le2i.annotations import OMNIFALL_REVISION
from gatefall.data.manifest import apply_manifest_schema, read_manifest
from gatefall.data.staged.labels import (
    labels_path,
    load_staged_segments,
    map_to_gatefall,
    verify_staged_labels,
)
from gatefall.data.staged.sequences import (
    ImageSequence,
    image_size,
    read_image_sequence,
    sequence_duration_s,
    sequence_sha256,
    source_fps,
)
from gatefall.data.staged.specs import StagedDatasetSpec
from gatefall.data.staged.timeline import (
    VideoGrid,
    build_video_frames,
    grid_from_timestamps,
    grid_from_video,
)
from gatefall.data.video_metadata import probe_video, resolve_frame_rate
from gatefall.hashing import sha256_dataframe, sha256_file
from gatefall.provenance import git_state, utc_now

MANIFEST_SCHEMA_VERSION = "staged-manifest-v1"
IDENTITY_COLUMNS = (
    "video_id", "recording_id", "subject", "activity", "trial", "camera", "category", "split",
)


@dataclass(frozen=True)
class StagedPaths:
    manifest: Path
    identities: Path
    sequence_index: Path
    frames: Path
    alignment: Path
    coverage: Path
    provenance: Path


def staged_paths(spec: StagedDatasetSpec) -> StagedPaths:
    root = spec.processed_dir
    return StagedPaths(
        manifest=root / "manifest.parquet",
        identities=root / "identities.parquet",
        sequence_index=root / "sequence_index.parquet",
        frames=root / "frames.parquet",
        alignment=root / "alignment.json",
        coverage=root / "coverage.json",
        provenance=root / "dataset_provenance.json",
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    dataframe.to_parquet(tmp)
    os.replace(tmp, path)


def labeled_video_ids(spec: StagedDatasetSpec, segments: pd.DataFrame) -> list[str]:
    video_ids = sorted(str(path) for path in segments["path"].unique())
    invalid = []
    for video_id in video_ids:
        try:
            spec.parse_video_id(video_id)
        except ValueError:
            invalid.append(video_id)
    if invalid:
        raise ValueError(f"{spec.identifier}: caminhos de anotação fora do padrão: {invalid[:10]}")
    return video_ids


def identity_rows(spec: StagedDatasetSpec, video_ids: list[str], segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_path = segments.groupby("path")
    for video_id in video_ids:
        parts = spec.parse_video_id(video_id)
        group = by_path.get_group(video_id)
        subjects = set(group["subject"].astype(int))
        cams = set(group["cam"].astype(int))
        if subjects != {int(parts["subject"])}:
            raise ValueError(f"{video_id}: sujeito das anotações {sorted(subjects)} diverge do caminho")
        camera = int(cast(int, parts.get("camera", next(iter(cams)))))
        if cams != {camera}:
            raise ValueError(f"{video_id}: câmera das anotações {sorted(cams)} diverge do caminho")
        rows.append(
            {
                "video_id": video_id,
                "recording_id": spec.recording_id(video_id),
                "subject": int(parts["subject"]),
                "activity": int(parts.get("activity", -1)),
                "trial": int(parts.get("trial", -1)),
                "camera": camera,
                "category": str(parts.get("category", "")),
                "split": spec.split.split_of(int(parts["subject"])),
            }
        )
    return pd.DataFrame(rows, columns=list(IDENTITY_COLUMNS))


def unlabeled_sources(spec: StagedDatasetSpec, labeled: set[str]) -> list[str]:
    if not spec.raw_dir.exists():
        return []
    if spec.source_kind == "image_sequence":
        candidates = [path for path in spec.raw_dir.glob("Subject*/Activity*/Trial*/*") if path.is_dir()]
    else:
        candidates = list(spec.raw_dir.glob("Subject */*/*.mp4"))
    found = []
    for path in candidates:
        relative = path.relative_to(spec.raw_dir)
        if spec.source_kind == "video":
            subject = relative.parts[0].removeprefix("Subject ")
            video_id = f"Subject_{subject}/{relative.parts[1]}/{relative.stem}"
        else:
            video_id = relative.as_posix()
        if video_id not in labeled:
            found.append(video_id)
    return sorted(found)


def _describe_sequence(sequence: ImageSequence) -> dict[str, object]:
    width, height = image_size(sequence)
    return {
        "fps": source_fps(sequence),
        "fps_source": "image_timestamps",
        "n_frames_header": None,
        "n_frames_counted": sequence.n_frames,
        "duration_s": sequence_duration_s(sequence),
        "width": width,
        "height": height,
        "codec": "image_sequence",
        "sha256": sequence_sha256(sequence),
    }


def _describe_video(path: Path) -> dict[str, object]:
    metadata = probe_video(path)
    fps, fps_source = resolve_frame_rate(metadata["r_frame_rate"], metadata["avg_frame_rate"])
    return {
        "fps": fps,
        "fps_source": fps_source,
        "n_frames_header": metadata["n_frames_header"],
        "n_frames_counted": metadata["n_frames_counted"],
        "duration_s": metadata["n_frames_counted"] / fps,
        "width": metadata["width"],
        "height": metadata["height"],
        "codec": metadata["codec"],
        "sha256": sha256_file(path),
    }


def build_manifest(spec: StagedDatasetSpec, force: bool = False) -> None:
    paths = staged_paths(spec)
    if paths.manifest.exists() and not force:
        print(f"skip {paths.manifest} (já existe, use --force para sobrescrever)")
        return
    problems = verify_staged_labels(spec)
    if problems:
        raise ValueError("anotações OmniFall não verificadas:\n  " + "\n  ".join(problems))
    segments = load_staged_segments(spec)
    video_ids = labeled_video_ids(spec, segments)
    identities = identity_rows(spec, video_ids, segments)

    missing = [
        video_id for video_id in video_ids
        if not (spec.raw_dir / spec.source_relative_path(video_id)).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{spec.identifier}: {len(missing)} fontes anotadas ausentes em {spec.raw_dir}; "
            f"primeiras: {missing[:10]}"
        )

    manifest_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for row in identities.to_dict("records"):
        video_id = str(row["video_id"])
        relative = spec.source_relative_path(video_id)
        absolute = (spec.raw_dir / relative).resolve()
        if spec.source_kind == "image_sequence":
            sequence = read_image_sequence(absolute, video_id)
            description = _describe_sequence(sequence)
            index_rows.extend(
                {
                    "video_id": video_id,
                    "src_index": position,
                    "filename": name,
                    "timestamp_s": float(sequence.timestamps_s[position]),
                    "origin_iso": sequence.origin_iso,
                }
                for position, name in enumerate(sequence.filenames)
            )
        else:
            description = _describe_video(absolute)
        manifest_rows.append(
            {
                "video_id": video_id,
                "dataset": spec.identifier,
                "relative_path": relative.as_posix(),
                "absolute_path": str(absolute),
                "env": spec.env,
                "subject": int(cast(int, row["subject"])),
                "cam": int(cast(int, row["camera"])),
                "split": str(row["split"]),
                **description,
            }
        )
        print(f"{video_id}: {description['n_frames_counted']} quadros, {float(cast(float, description['fps'])):.3f} fps")

    manifest = apply_manifest_schema(pd.DataFrame(manifest_rows))
    write_parquet(manifest, paths.manifest)
    write_parquet(identities, paths.identities)
    if index_rows:
        write_parquet(pd.DataFrame(index_rows), paths.sequence_index)
    write_json(
        paths.provenance,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset": spec.identifier,
            "role": spec.role,
            "source_kind": spec.source_kind,
            "omnifall_revision": OMNIFALL_REVISION,
            "labels_sha256": sha256_file(labels_path(spec)),
            "split": spec.split.to_dict(),
            "split_sha256": split_sha256(manifest),
            "manifest_content_sha256": sha256_dataframe(
                manifest.drop(columns=["absolute_path"]).sort_values("video_id").reset_index(drop=True)
            ),
            "n_videos": len(manifest),
            "unlabeled_sources": unlabeled_sources(spec, set(video_ids)),
            "code": git_state(),
            "created_at": utc_now(),
        },
    )
    print(f"{paths.manifest}: {len(manifest)} vídeos; partição {spec.split.version}")


def split_sha256(manifest: pd.DataFrame) -> str:
    ordered = cast(pd.DataFrame, manifest[["video_id", "split"]]).astype("string")
    return sha256_dataframe(ordered.sort_values("video_id").reset_index(drop=True))


def load_staged_manifest(spec: StagedDatasetSpec) -> pd.DataFrame:
    path = staged_paths(spec).manifest
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: ausente; rode `uv run python -m gatefall.data.staged manifest --dataset {spec.identifier}`"
        )
    return read_manifest(path)


def load_sequence_index(spec: StagedDatasetSpec) -> pd.DataFrame:
    path = staged_paths(spec).sequence_index
    if not path.exists():
        raise FileNotFoundError(f"{path}: ausente; reconstrua o manifesto de {spec.identifier}")
    return pd.read_parquet(path)


def _video_grid(spec: StagedDatasetSpec, row: pd.Series, index: dict[str, np.ndarray]) -> VideoGrid:
    if spec.source_kind == "image_sequence":
        return grid_from_timestamps(index[str(row["video_id"])], float(cast(float, row["duration_s"])))
    return grid_from_video(int(cast(int, row["n_frames_counted"])), float(cast(float, row["fps"])))


def build_timegrid(spec: StagedDatasetSpec, force: bool = False) -> None:
    paths = staged_paths(spec)
    if paths.frames.exists() and not force:
        print(f"skip {paths.frames} (já existe, use --force para sobrescrever)")
        return
    manifest = load_staged_manifest(spec)
    segments, n_unsupported = map_to_gatefall(load_staged_segments(spec))
    by_path = {str(path): group for path, group in segments.groupby("path")}
    timestamps: dict[str, np.ndarray] = {}
    if spec.source_kind == "image_sequence":
        index = load_sequence_index(spec).sort_values(["video_id", "src_index"])
        timestamps = {
            str(video_id): group["timestamp_s"].to_numpy(dtype=np.float64)
            for video_id, group in index.groupby("video_id")
        }
    parts: list[pd.DataFrame] = []
    alignment: list[dict[str, object]] = []
    for _, row in manifest.sort_values("video_id").iterrows():
        video_id = str(row["video_id"])
        frames, report = build_video_frames(
            video_id=video_id,
            split=str(row["split"]),
            env=str(row["env"]),
            subject=int(cast(int, row["subject"])),
            grid=_video_grid(spec, row, timestamps),
            duration_s=float(cast(float, row["duration_s"])),
            segments=by_path[video_id],
        )
        if len(frames) < WINDOW_FRAMES:
            report["shorter_than_window"] = True
        parts.append(frames)
        alignment.append(report)
    table = cast(pd.DataFrame, pd.concat(parts, ignore_index=True))
    write_frames(table, paths.frames)
    if not table.equals(read_frames(paths.frames)):
        raise RuntimeError(f"{paths.frames}: releitura diverge do gravado")
    write_json(
        paths.alignment,
        {
            "target_fps": TARGET_FPS,
            "frames_content_sha256": sha256_dataframe(table),
            "n_unsupported_segments": n_unsupported,
            "videos": alignment,
        },
    )
    print(f"{paths.frames}: {len(table)} timesteps em {len(parts)} vídeos")
