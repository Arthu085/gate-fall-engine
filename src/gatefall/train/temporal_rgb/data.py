"""Reuso das dobras existentes e alinhamento causal dos clipes RGB."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, NUM_CLASSES, TRAIN_STRIDE
from gatefall.data.windowing import build_window_index, window_frame_indices
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.eval.analysis.internal_fold_feasibility import read_train_partition
from gatefall.hashing import sha256_dataframe
from gatefall.train.internal_folds import cli as cv_cli
from gatefall.train.internal_folds import folds as cv_folds
from gatefall.train.shared.metrics import RESTRICTED_CLASSES
from gatefall.train.subject_folds import cli as cs_cli
from gatefall.train.subject_folds import folds as cs_folds
from gatefall.train.temporal_rgb.contract import OFFSETS, digest

Partition = cs_folds.SubjectFoldPartition | cv_folds.FoldPartition


@dataclass
class Context:
    protocol: str
    adapter: Le2iDatasetAdapter
    frames: pd.DataFrame
    manifest: pd.DataFrame
    partitions: list[Partition]
    input_sha256: dict[str, str]
    membership_sha256: str

    @cached_property
    def train_frames_sha256(self) -> str:
        return sha256_dataframe(self.frames)


def load_context(protocol: str, fold_root: Path) -> Context:
    if protocol == "cs":
        adapter = cs_cli.ADAPTER
        frames, _, hashes = cs_cli._official_train(adapter.frames_path, adapter.manifest_path)
        manifest = read_train_partition(adapter.manifest_path)
        partitions: list[Partition] = list(cs_folds.build_subject_fold_partitions(frames))
        document = cs_folds.read_folds_document(fold_root)
    elif protocol == "cv":
        adapter = cv_cli.ADAPTER
        frames, manifest, _, hashes = cv_cli._official_train(adapter.frames_path, adapter.manifest_path)
        partitions = list(cv_folds.build_fold_partitions(frames))
        document = cv_folds.read_folds_document(fold_root)
    else:
        raise ValueError("protocolo deve ser cs ou cv")
    if document["input_sha256"] != hashes or document["folds"] != [p.membership_record for p in partitions]:
        raise ValueError("dobras preparadas divergem das entradas ou da composição atual")
    return Context(protocol, adapter, frames, manifest, partitions, hashes, cast(str, document["membership_sha256"]))


def fold_windows(partition: Partition, side: str) -> pd.DataFrame:
    if side not in (cv_folds.DEVELOPMENT_SPLIT, cv_folds.RETAINED_SPLIT):
        raise ValueError("lado inválido")
    stride = TRAIN_STRIDE if side == cv_folds.DEVELOPMENT_SPLIT else EVAL_STRIDE
    windows = build_window_index(partition.side_frames(side), stride)
    labels = windows["label"].to_numpy(dtype=np.int64)
    if len(labels) == 0 or np.any((labels < 0) | (labels >= NUM_CLASSES)):
        raise ValueError("janelas sem rótulos válidos")
    if not set(RESTRICTED_CLASSES) <= set(labels.tolist()):
        raise ValueError("as oito classes restritas precisam ter suporte em cada lado")
    return windows


def window_identity(windows: pd.DataFrame) -> str:
    selected = cast(pd.DataFrame, windows[["video_id", "k_end", "label", "n_frames"]])
    return digest(selected.to_dict("records"))


def align_video(frames: pd.DataFrame, fps: float, source_count: int) -> dict[str, np.ndarray]:
    frames = frames.sort_values("frame_index").reset_index(drop=True)
    if set(frames["split"].astype(str)) != {"train"} or frames["video_id"].nunique() != 1:
        raise ValueError("o alinhamento aceita exatamente um vídeo do train oficial")
    if not np.isfinite(fps) or fps <= 0 or source_count <= 0:
        raise ValueError("metadados de vídeo inválidos")
    windows = build_window_index(frames, EVAL_STRIDE)
    ends = windows["k_end"].to_numpy(dtype=np.int64)
    if not len(ends):
        raise ValueError("vídeo sem janelas rotuladas")
    times = frames["time_s"].to_numpy(dtype=np.float64)
    source = frames["src_index"].to_numpy(dtype=np.int64)
    if (not np.all(np.isfinite(times)) or times[0] < 0 or np.any(np.diff(times) <= 0)
            or source.min() < 0 or source.max() >= source_count or np.any(np.diff(source) < 0)):
        raise ValueError("grade temporal ou índices de origem inválidos")
    membership = np.stack([window_frame_indices(int(end), len(frames)) for end in ends])
    clip_grid = membership[:, OFFSETS]
    endpoint_limit = np.floor(times[ends] * fps + 1e-8).astype(np.int64)
    clip_source = np.minimum(source[clip_grid], endpoint_limit[:, None])
    if np.any(clip_source < 0) or np.any(clip_source / fps > times[ends, None] + 1e-8):
        raise ValueError("clipe contém quadro futuro")
    return {
        "k_end": ends, "labels": windows["label"].to_numpy(dtype=np.int64),
        "window_grid_indices": membership, "clip_grid_indices": clip_grid,
        "original_src_indices": source[clip_grid], "clip_src_indices": clip_source,
        "clip_src_times_s": clip_source.astype(np.float64) / fps,
        "endpoint_times_s": times[ends],
        "static_src_indices": np.repeat(clip_source[:, -1:], len(OFFSETS), axis=1),
    }


def context_record(context: Context) -> dict[str, Any]:
    return {
        "protocol": context.protocol, "input_sha256": context.input_sha256,
        "membership_sha256": context.membership_sha256,
        "train_frames_sha256": context.train_frames_sha256,
    }
