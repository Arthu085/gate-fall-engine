"""Cobertura, vazamento entre partições, duplicatas e sincronização de vistas."""

from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, NUM_CLASSES, TARGET_FPS
from gatefall.data.frames import read_frames
from gatefall.data.staged.build import (
    load_staged_manifest,
    staged_paths,
    split_sha256,
    write_json,
)
from gatefall.data.staged.labels import load_reference_splits
from gatefall.data.staged.specs import SPLIT_NAMES, StagedDatasetSpec
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.shared.events import fall_events_for_video

# Diferença máxima aceita entre o início das vistas de uma mesma gravação.
MAX_VIEW_START_OFFSET_S = 1.0


def leakage_problems(manifest: pd.DataFrame, identities: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    duplicated_ids = manifest["video_id"][manifest["video_id"].duplicated()].tolist()
    if duplicated_ids:
        problems.append(f"video_id repetido: {duplicated_ids[:10]}")
    for column in ("subject",):
        spread = cast(pd.Series, manifest.groupby(column)["split"].nunique())
        crossing = [key for key, count in spread.items() if int(count) > 1]
        if crossing:
            problems.append(f"{column} presente em mais de uma partição: {crossing}")
    recordings = cast(pd.Series, identities.groupby("recording_id")["split"].nunique())
    crossing_recordings = [key for key, count in recordings.items() if int(count) > 1]
    if crossing_recordings:
        problems.append(f"vistas sincronizadas em partições diferentes: {crossing_recordings[:10]}")
    hashes = manifest.groupby("sha256")["video_id"].apply(list)
    duplicates = [ids for ids in hashes if len(ids) > 1]
    if duplicates:
        problems.append(f"conteúdo idêntico em vídeos distintos: {duplicates[:5]}")
    return problems


def view_synchronization(identities: pd.DataFrame, sequence_index: pd.DataFrame | None) -> dict:
    if sequence_index is None:
        return {"applicable": False}
    origins = cast(pd.Series, sequence_index.groupby("video_id")["origin_iso"].first())
    origin_by_video = {str(key): value for key, value in origins.items()}
    joined = identities.assign(origin=[origin_by_video.get(str(v)) for v in identities["video_id"]])
    offsets: list[dict[str, object]] = []
    for recording_id, group in joined.groupby("recording_id"):
        if len(group) < 2:
            continue
        starts = pd.to_datetime(group["origin"])
        offset = float((starts.max() - starts.min()).total_seconds())
        offsets.append({"recording_id": recording_id, "views": len(group), "start_offset_s": offset})
    values = np.array([float(cast(float, item["start_offset_s"])) for item in offsets])
    return {
        "applicable": True,
        "recordings_with_multiple_views": len(offsets),
        "max_start_offset_s": float(values.max()) if len(values) else None,
        "recordings_above_tolerance": [
            item for item in offsets if float(cast(float, item["start_offset_s"])) > MAX_VIEW_START_OFFSET_S
        ],
        "tolerance_s": MAX_VIEW_START_OFFSET_S,
    }


def reference_split_audit(spec: StagedDatasetSpec) -> dict | None:
    reference = load_reference_splits(spec)
    if reference is None:
        return None
    by_subject = cast(pd.Series, reference.groupby("subject")["reference_split"].unique())
    shared = {
        int(cast(int, subject)): sorted(map(str, splits))
        for subject, splits in by_subject.items()
        if len(splits) > 1
    }
    by_path = reference.groupby("path")["reference_split"].nunique()
    return {
        "config": spec.omnifall_reference_config,
        "subjects_in_multiple_splits": shared,
        "paths_in_multiple_splits": int((by_path > 1).sum()),
        "used_by_study": False,
    }


def split_coverage(frames: pd.DataFrame) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for split in SPLIT_NAMES:
        part = cast(pd.DataFrame, frames[frames["split"] == split])
        if part.empty:
            continue
        labels = part["label"].to_numpy()
        counts = np.bincount(labels[labels != IGNORE_LABEL].astype(np.int64), minlength=NUM_CLASSES)
        n_events = 0
        for video_id, group in part.groupby("video_id"):
            ordered = group.sort_values("frame_index")
            n_events += len(
                fall_events_for_video(
                    str(video_id),
                    ordered["frame_index"].to_numpy(dtype=np.int64),
                    ordered["label"].to_numpy(dtype=np.int64),
                    BASELINE_A_ALARM_PROTOCOL,
                )
            )
        report[split] = {
            "videos": int(cast(int, part["video_id"].nunique())),
            "subjects": sorted(int(subject) for subject in part["subject"].unique()),
            "timesteps": int(len(part)),
            "ignored_timesteps": int((labels == IGNORE_LABEL).sum()),
            "source_gap_timesteps": int((part["gap_position"] == "source_gap").sum()),
            "timesteps_by_class": {name: int(counts[i]) for i, name in enumerate(LE2I_LABEL_NAMES)},
            "classes_without_support": [name for i, name in enumerate(LE2I_LABEL_NAMES) if counts[i] == 0],
            "fall_events": n_events,
            "hours": float(len(part) / TARGET_FPS / 3600.0),
        }
    return report


def run_audit(spec: StagedDatasetSpec) -> dict:
    paths = staged_paths(spec)
    manifest = load_staged_manifest(spec)
    identities = pd.read_parquet(paths.identities)
    sequence_index = pd.read_parquet(paths.sequence_index) if paths.sequence_index.exists() else None
    frames = read_frames(paths.frames)
    problems = leakage_problems(manifest, identities)
    missing_frames = sorted(set(manifest["video_id"]) - set(frames["video_id"]))
    if missing_frames:
        problems.append(f"vídeos sem grade temporal: {missing_frames[:10]}")
    if spec.role == "external_test" and set(manifest["split"]) != {"test"}:
        problems.append("teste externo contém vídeos fora de `test`")
    synchronization = view_synchronization(identities, sequence_index)
    if synchronization.get("recordings_above_tolerance"):
        problems.append("vistas com início dessincronizado acima da tolerância")
    report = {
        "dataset": spec.identifier,
        "role": spec.role,
        "split_version": spec.split.version,
        "split_sha256": split_sha256(manifest),
        "coverage": split_coverage(frames),
        "synchronization": synchronization,
        "omnifall_reference_split": reference_split_audit(spec),
        "problems": problems,
        "passed": not problems,
    }
    write_json(paths.coverage, report)
    return report
