"""Análise pós-hoc de degradação isolada dos gates B1 e C1 congelados."""

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.data.video_io import decode_frames
from gatefall.data.windowing import build_window_index
from gatefall.datasets import get_dataset
from gatefall.datasets.base import DatasetAdapter
from gatefall.dinov3.backbone import (
    configure_deterministic_inference,
    load_backbone,
    read_dinov3_repo_commit,
    resolve_repo_dir,
    resolve_weights_path,
)
from gatefall.dinov3.quality import (
    DEGRADATION_SWEEPS as DINOV3_SWEEPS,
    _compute_feature_batch,
    apply_degradation,
    compute_visual_quality,
)
from gatefall.dinov3.preprocessing import resize_frames
from gatefall.dinov3.storage import dinov3_path, read_features
from gatefall.dinov3.features import Dinov3Backbone
from gatefall.eval.baseline_b1.cli import _load_run_assets as load_b1_assets
from gatefall.eval.baseline_c1.cli import (
    _load_run_assets as load_c1_assets,
    _quality_for_video as c1_quality_for_video,
)
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, load_alarm_protocol
from gatefall.eval.shared.events import split_event_report
from gatefall.eval.shared.orchestration import _n_fall_segments_in_annotation
from gatefall.features.dinov3_standardization import (
    Dinov3StandardizationStats,
    apply_standardization as standardize_dinov3,
)
from gatefall.features.quality_storage import quality_path, read_quality
from gatefall.features.standardize_dinov3 import dinov3_stats_path
from gatefall.features.standardize_sam3 import sam3_stats_path
from gatefall.features.sam3_standardization import (
    Sam3StandardizationStats,
    apply_standardization as standardize_sam3,
)
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization as standardize_pose,
)
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features, build_pose_features_from_arrays
from gatefall.pose.loading import PoseArrays, load_pose
from gatefall.pose.quality import pose_quality_from_arrays
from gatefall.runs import REPOSITORY_ROOT, default_run_dir_for_arm, validate_local_run_dir
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.sam3.extract import run_frames_through_segmenter
from gatefall.sam3.features import load_v_t
from gatefall.sam3.quality import (
    DEGRADATION_SWEEPS as SAM3_SWEEPS,
    apply_frame_degradation,
    compute_sam3_quality,
)
from gatefall.sam3.runtime import (
    Sam3RuntimeSegmenter,
    Sam3Segmenter,
    resolve_checkpoint_path,
    resolve_runtime_project_dir,
)
from gatefall.train.shared.gated_model import GatedFusionClassifier
from gatefall.train.shared.metrics import restricted_macro_f1

POSE_SEVERITIES = (0, 4, 8, 12, 16)
SPLITS = ("val", "test")
DEGRADATIONS = {"clean": "none", "pose": "keypoint_dropout", "visual": "blur"}
SHARD_ARTIFACT = "gate_degradation_shard"
SHARD_SCHEMA_VERSION = 1
SHARD_FIELDS = frozenset({
    "artifact", "shard_schema_version", "canonical", "partial", "shard_conditions",
    "trace_file", "trace_sha256", "report", "conditions",
})
REPORT_FIELDS = (
    "schema_version", "arm", "dataset", "run_dir", "checkpoint_path",
    "checkpoint_sha256", "config_sha256", "frames_sha256", "pose_stats_sha256",
    "visual_stats_sha256", "gate_parameters", "visual_quality_name",
    "alarm_protocol", "test_split_is_descriptive_only", "selection_performed",
    "pose_severities", "pose_dropout_selection", "numpy_version",
    "visual_blur_severities", "backbone_provenance",
)
REPORT_SHA256_FIELDS = (
    "checkpoint_sha256", "config_sha256", "frames_sha256", "pose_stats_sha256",
    "visual_stats_sha256",
)
BACKBONE_PROVENANCE_FIELDS = {
    "B1": frozenset({"weights_sha256", "dinov3_repo_commit"}),
    "C1": frozenset({
        "sam3_checkpoint_sha256", "sam3_runtime_lock_sha256",
        "sam3_inference_autocast_dtype", "sam3_source_revision",
    }),
}
GATE_PARAMETER_FIELDS = ("gate_weight_pose", "gate_weight_visual", "gate_bias")
POSE_DROPOUT_SELECTION = (
    "sha256(video_id:frame_index)[:8] big-endian seed; "
    "numpy.default_rng permutation of 17 keypoints"
)
CONDITION_FIELDS = (
    "arm", "dataset", "split", "modality", "degradation", "severity",
    "checkpoint_sha256", "gate_parameters", "n_windows", "n_labeled_windows",
    "n_frames", "gate_statistics_unit", "statistics", "macro_f1_restricted",
    "event_sensitivity", "false_alarms", "false_alarms_per_hour",
    "event_latency_seconds", "n_fall_events",
)
STATISTIC_NAMES = (
    "q_pose", "q_visual", "g_pose", "g_visual", "pose_effective_norm", "visual_effective_norm",
)
SUMMARY_FIELDS = ("mean", "median", "p05", "p95")
TRACE_IDENTITY_FIELDS = (
    "arm", "dataset", "split", "modality", "degradation", "severity",
    "checkpoint_sha256", "gate_weight_pose", "gate_weight_visual", "gate_bias",
)
TRACE_FIELDS = (
    "arm", "dataset", "split", "modality", "degradation", "severity",
    "checkpoint_sha256", "video_id", "frame_index", "q_pose", "q_visual",
    "g_pose", "g_visual", "pose_effective_norm", "visual_effective_norm",
    "gate_weight_pose", "gate_weight_visual", "gate_bias",
)


ConditionKey = tuple[str, str, int]


@dataclass(frozen=True)
class VideoInputs:
    pose: np.ndarray
    visual: np.ndarray
    quality: np.ndarray


@dataclass(frozen=True)
class Shard:
    path: Path
    report: dict[str, object]
    conditions: dict[ConditionKey, dict[str, object]]
    traces: dict[ConditionKey, list[dict[str, str]]]


def visual_severities(arm: str) -> tuple[int, ...]:
    if arm not in ("B1", "C1"):
        raise ValueError(f"arma inválida: {arm}")
    return tuple(int(value) for value in (DINOV3_SWEEPS if arm == "B1" else SAM3_SWEEPS)["blur"])


def canonical_conditions(arm: str) -> list[ConditionKey]:
    keys: list[ConditionKey] = []
    for split in SPLITS:
        keys.append((split, "clean", 0))
        keys.extend((split, "pose", severity) for severity in POSE_SEVERITIES)
        keys.extend((split, "visual", severity) for severity in visual_severities(arm))
    return keys


def condition_id(key: ConditionKey) -> str:
    return ":".join(str(part) for part in key)


def select_conditions(arm: str, patterns: list[str]) -> list[ConditionKey]:
    canonical = canonical_conditions(arm)
    selected: set[ConditionKey] = set()
    for pattern in patterns:
        parts = pattern.split(":")
        if len(parts) != 3:
            raise ValueError(f"condição deve ter o formato split:modalidade:severidade: {pattern}")
        matches = {
            key for key in canonical
            if all(part in ("*", str(value)) for part, value in zip(parts, key))
        }
        if not matches:
            raise ValueError(f"condição fora da grade canônica de {arm}: {pattern}")
        selected |= matches
    return [key for key in canonical if key in selected]


def drop_keypoints(pose: PoseArrays, video_id: str, severity: int) -> PoseArrays:
    if severity not in POSE_SEVERITIES:
        raise ValueError(f"severidade de pose inválida: {severity}")
    if severity == 0:
        return pose
    keypoints = pose.keypoints.copy()
    for frame_index in range(pose.k):
        seed = int.from_bytes(
            hashlib.sha256(f"{video_id}:{frame_index}".encode()).digest()[:8],
            "big",
        )
        dropped = np.random.default_rng(seed).permutation(17)[:severity]
        keypoints[frame_index, dropped] = 0.0
    return PoseArrays(keypoints, pose.bbox, pose.person_found, pose.k, pose.width, pose.height)


def replace_modality(
    clean: VideoInputs, modality: str, features: np.ndarray, quality: np.ndarray
) -> VideoInputs:
    if modality == "pose":
        if features.shape != clean.pose.shape or quality.shape != (len(clean.pose),):
            raise ValueError("shape de pose degradada inválido")
        combined = clean.quality.copy()
        combined[:, 0] = quality
        return VideoInputs(features, clean.visual, combined)
    if modality == "visual":
        if features.shape != clean.visual.shape or quality.shape != (len(clean.visual),):
            raise ValueError("shape visual degradado inválido")
        combined = clean.quality.copy()
        combined[:, 1] = quality
        return VideoInputs(clean.pose, features, combined)
    raise ValueError(f"modalidade inválida: {modality}")


def gate_coefficients(model: GatedFusionClassifier) -> dict[str, float]:
    weights = model.gate.linear.weight.detach().cpu().numpy()
    bias = model.gate.linear.bias.detach().cpu().numpy()
    if weights.shape != (1, 2) or bias.shape != (1,):
        raise ValueError("gate não é linear escalar com dois canais")
    return {
        "gate_weight_pose": float(weights[0, 0]),
        "gate_weight_visual": float(weights[0, 1]),
        "gate_bias": float(bias[0]),
    }


@torch.no_grad()
def gate_trace(
    model: GatedFusionClassifier,
    pose: np.ndarray,
    visual: np.ndarray,
    quality: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_tensor = torch.from_numpy(pose).to(device)
    visual_tensor = torch.from_numpy(visual).to(device)
    quality_tensor = torch.from_numpy(quality.astype(np.float32)).to(device)
    gate = model.gate(quality_tensor).squeeze(-1)
    pose_norm = torch.linalg.vector_norm(model.e_p(pose_tensor), dim=-1) * gate
    visual_norm = torch.linalg.vector_norm(model.e_v(visual_tensor), dim=-1) * (1 - gate)
    return tuple(
        tensor.cpu().numpy().astype(np.float32)
        for tensor in (gate, pose_norm, visual_norm)
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _clean_inputs(adapter: DatasetAdapter, arm: str, video_id: str) -> VideoInputs:
    pose = build_pose_features(video_id, pose_root=adapter.pose_root)[0]
    if arm == "B1":
        visual = read_features(dinov3_path(video_id, dinov3_root=adapter.dinov3_root))
        quality = read_quality(quality_path(video_id, quality_root=adapter.quality_root))
    else:
        visual = load_v_t(video_id, sam3_root=adapter.sam3_root)
        quality = c1_quality_for_video(video_id, adapter)
    if len(pose) != len(visual) or quality.shape != (len(pose), 2):
        raise ValueError(f"fontes desalinhadas: {video_id}")
    return VideoInputs(pose.astype(np.float32), visual.astype(np.float32), quality.astype(np.float32))


def _pose_condition(
    adapter: DatasetAdapter, clean: dict[str, VideoInputs], severity: int
) -> dict[str, VideoInputs]:
    if severity == 0:
        return clean
    result: dict[str, VideoInputs] = {}
    for video_id, original in clean.items():
        pose = drop_keypoints(load_pose(video_id, pose_root=adapter.pose_root), video_id, severity)
        features = build_pose_features_from_arrays(pose)[0]
        quality = pose_quality_from_arrays(pose.keypoints, pose.bbox, pose.person_found).q_pose
        result[video_id] = replace_modality(original, "pose", features, quality)
    return result


def _b1_visual_condition(
    adapter: DatasetAdapter,
    frames: pd.DataFrame,
    clean: dict[str, VideoInputs],
    severity: int,
    backbone: Dinov3Backbone,
    device: str,
    batch_size: int,
) -> dict[str, VideoInputs]:
    if severity == 0:
        return clean
    paths = adapter.video_paths()
    result: dict[str, VideoInputs] = {}
    for video_id, original in clean.items():
        rows = cast(pd.DataFrame, frames[frames["video_id"] == video_id]).sort_values("frame_index")
        decoded = decode_frames(paths[video_id], rows["src_index"].astype(int).tolist())
        visual_parts: list[np.ndarray] = []
        quality_parts: list[np.ndarray] = []
        for start in range(0, len(decoded), batch_size):
            resized = resize_frames(decoded[start : start + batch_size])
            degraded = apply_degradation(resized, "blur", severity)
            visual_parts.append(_compute_feature_batch(backbone, degraded, device=device))
            quality_parts.append(compute_visual_quality(degraded).q_visual)
        result[video_id] = replace_modality(
            original, "visual", np.concatenate(visual_parts), np.concatenate(quality_parts)
        )
    return result


def _check_dinov3_backbone(
    adapter: DatasetAdapter, frames: pd.DataFrame, repo_dir: Path, weights_path: Path
) -> dict[str, str]:
    expected = {
        "weights_sha256": sha256_file(weights_path),
        "dinov3_repo_commit": read_dinov3_repo_commit(repo_dir),
    }
    for video_id in sorted(str(value) for value in frames["video_id"].unique()):
        path = dinov3_path(video_id, dinov3_root=adapter.dinov3_root)
        with h5py.File(path, "r") as source:
            for name, value in expected.items():
                if source.attrs.get(name) != value:
                    raise ValueError(f"{path}: {name} difere do backbone congelado")
    return expected


def _c1_visual_condition(
    adapter: DatasetAdapter,
    frames: pd.DataFrame,
    clean: dict[str, VideoInputs],
    severity: int,
    segmenter: Sam3Segmenter,
) -> dict[str, VideoInputs]:
    if severity == 0:
        return clean
    paths = adapter.video_paths()
    result: dict[str, VideoInputs] = {}
    for video_id, original in clean.items():
        rows = cast(pd.DataFrame, frames[frames["video_id"] == video_id]).sort_values("frame_index")
        decoded = decode_frames(paths[video_id], rows["src_index"].astype(int).tolist())
        degraded = [apply_frame_degradation(frame, "blur", severity) for frame in decoded]
        height, width = decoded[0].shape[:2]
        visual, scores, _counts = run_frames_through_segmenter(
            degraded, segmenter=segmenter, width=width, height=height
        )
        quality = compute_sam3_quality(scores, visual[:, 0])
        result[video_id] = replace_modality(original, "visual", visual, quality)
    return result


@torch.no_grad()
def _predict(
    model: GatedFusionClassifier,
    source: GatedFusionWindowDataset,
    pose_stats: StandardizationStats,
    visual_stats: Dinov3StandardizationStats | Sam3StandardizationStats,
    arm: str,
    device: str,
    batch_size: int,
) -> tuple[list[str], list[int], list[int], list[int]]:
    identities: list[str] = []
    k_ends: list[int] = []
    labels: list[int] = []
    predictions: list[int] = []
    for start in range(0, len(source), batch_size):
        batch = [source[index] for index in range(start, min(start + batch_size, len(source)))]
        pose = standardize_pose(np.stack([item[0] for item in batch]), pose_stats)
        visual = _standardize_visual(
            arm, np.stack([item[1] for item in batch]), visual_stats
        )
        quality = np.stack([item[2] for item in batch])
        output = model(
            torch.from_numpy(pose).to(device),
            torch.from_numpy(visual).to(device),
            torch.from_numpy(quality).to(device),
        )
        predictions.extend(int(value) for value in output.argmax(dim=1).cpu().tolist())
        for _pose, _visual, _quality, label, (video_id, k_end) in batch:
            identities.append(video_id)
            k_ends.append(k_end)
            labels.append(label)
    return identities, k_ends, labels, predictions


def _standardize_visual(
    arm: str,
    visual: np.ndarray,
    stats: Dinov3StandardizationStats | Sam3StandardizationStats,
) -> np.ndarray:
    if arm == "B1" and isinstance(stats, Dinov3StandardizationStats):
        return standardize_dinov3(visual, stats)
    if arm == "C1" and isinstance(stats, Sam3StandardizationStats):
        return standardize_sam3(visual, stats)
    raise ValueError("estatísticas visuais incompatíveis com a arma")


def _condition_report(
    *,
    arm: str,
    adapter: DatasetAdapter,
    frames: pd.DataFrame,
    split: str,
    modality: str,
    degradation: str,
    severity: int,
    inputs: dict[str, VideoInputs],
    model: GatedFusionClassifier,
    pose_stats: StandardizationStats,
    visual_stats: Dinov3StandardizationStats | Sam3StandardizationStats,
    device: str,
    batch_size: int,
    checkpoint_sha256: str,
    writer: csv.DictWriter,
) -> dict[str, object]:
    split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
    source = GatedFusionWindowDataset(
        frames, split, EVAL_STRIDE,
        lambda video_id: inputs[video_id].pose,
        lambda video_id: inputs[video_id].visual,
        lambda video_id: inputs[video_id].quality,
        drop_ignored=False, visual_dim=1536 if arm == "B1" else V_T_DIM,
    )
    predictions = _predict(model, source, pose_stats, visual_stats, arm, device, batch_size)
    labeled = len(build_window_index(split_frames, stride=EVAL_STRIDE, drop_ignored=True))
    events = split_event_report(
        *predictions, BASELINE_A_ALARM_PROTOCOL, len(source), len(source), labeled
    )
    if events["n_fall_events"] != _n_fall_segments_in_annotation(frames, split):
        raise ValueError(f"split={split}: contagem de eventos diverge da anotação")
    true = np.asarray(predictions[2], dtype=np.int64)
    pred = np.asarray(predictions[3], dtype=np.int64)
    mask = true != IGNORE_LABEL
    macro_f1 = restricted_macro_f1(true[mask], pred[mask])[0]
    coefficients = gate_coefficients(model)
    statistics: dict[str, list[np.ndarray]] = {name: [] for name in STATISTIC_NAMES}
    for video_id, video in inputs.items():
        pose = standardize_pose(video.pose, pose_stats)
        visual = _standardize_visual(arm, video.visual, visual_stats)
        gate, pose_norm, visual_norm = gate_trace(model, pose, visual, video.quality, device)
        values = {
            "q_pose": video.quality[:, 0],
            "q_visual": video.quality[:, 1],
            "g_pose": gate,
            "g_visual": 1 - gate,
            "pose_effective_norm": pose_norm,
            "visual_effective_norm": visual_norm,
        }
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError(f"diagnóstico não finito: {video_id}")
        for name, value in values.items():
            statistics[name].append(value)
        for frame_index in range(len(video.pose)):
            writer.writerow({
                "arm": arm, "dataset": adapter.identifier, "split": split,
                "modality": modality, "degradation": degradation, "severity": severity,
                "checkpoint_sha256": checkpoint_sha256, "video_id": video_id,
                "frame_index": frame_index,
                **{name: float(value[frame_index]) for name, value in values.items()},
                **coefficients,
            })
    return {
        "arm": arm, "dataset": adapter.identifier, "split": split,
        "modality": modality, "degradation": degradation, "severity": severity,
        "checkpoint_sha256": checkpoint_sha256, "gate_parameters": coefficients,
        "n_windows": len(source), "n_labeled_windows": labeled,
        "n_frames": sum(len(video.pose) for video in inputs.values()),
        "gate_statistics_unit": "video_frame",
        "statistics": {name: _summary(np.concatenate(parts)) for name, parts in statistics.items()},
        "macro_f1_restricted": macro_f1,
        "event_sensitivity": events["sensitivity"],
        "false_alarms": events["n_false_alarms"],
        "false_alarms_per_hour": events["false_alarms_per_hour"],
        "event_latency_seconds": events["latency_seconds"],
        "n_fall_events": events["n_fall_events"],
    }


def _add_deltas(conditions: list[dict[str, object]]) -> None:
    baselines = {str(row["split"]): row for row in conditions if row["modality"] == "clean"}
    metrics = ("macro_f1_restricted", "event_sensitivity", "false_alarms", "false_alarms_per_hour")
    for row in conditions:
        baseline = baselines[str(row["split"])]
        row["delta_from_clean"] = {
            metric: float(cast(float, row[metric])) - float(cast(float, baseline[metric]))
            for metric in metrics
        }
        latency = cast(dict[str, object], row["event_latency_seconds"])["mean"]
        base_latency = cast(dict[str, object], baseline["event_latency_seconds"])["mean"]
        cast(dict[str, object], row["delta_from_clean"])["event_latency_mean_seconds"] = (
            None if latency is None or base_latency is None
            else float(cast(float, latency)) - float(cast(float, base_latency))
        )


def _check_output_dir(output_dir: Path, protected: list[Path]) -> None:
    resolved = output_dir.resolve()
    for path in protected:
        source = path.resolve()
        if resolved == source or source in resolved.parents or resolved in source.parents:
            raise ValueError(f"output-dir coincide com fonte canônica: {source}")


def _protected_paths(run_dir: Path, adapter: DatasetAdapter) -> list[Path]:
    return [
        run_dir, REPOSITORY_ROOT / "runs", REPOSITORY_ROOT / "data/raw",
        REPOSITORY_ROOT / "data/features", REPOSITORY_ROOT / "data/processed",
        adapter.raw_dir, adapter.frames_path.parent, adapter.pose_root,
        adapter.dinov3_root, adapter.sam3_root, adapter.quality_root,
        adapter.pose_stats_path.parent,
    ]


def _canonical_paths(output_dir: Path, arm: str) -> tuple[Path, Path]:
    stem = f"{arm.lower()}_gate_degradation"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.csv"


def _shard_paths(output_dir: Path, arm: str, selected: list[ConditionKey]) -> tuple[Path, Path]:
    digest = hashlib.sha256(",".join(condition_id(key) for key in selected).encode()).hexdigest()[:12]
    stem = f"{arm.lower()}_gate_degradation.shard-{digest}.partial"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.csv"


@contextmanager
def _staged_outputs(json_path: Path, csv_path: Path, force: bool) -> Iterator[tuple[Path, Path]]:
    if not force and (json_path.exists() or csv_path.exists()):
        raise FileExistsError(f"análise já existe em {json_path.parent}; use --force")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    csv_tmp = csv_path.parent / f".{csv_path.name}.{token}.tmp"
    json_tmp = json_path.parent / f".{json_path.name}.{token}.tmp"
    try:
        yield json_tmp, csv_tmp
        os.replace(csv_tmp, csv_path)
        os.replace(json_tmp, json_path)
    finally:
        csv_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run_analysis(
    arm: str,
    dataset_name: str,
    run_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
    conditions: list[str] | None = None,
    repo_dir: str | None = None,
    weights: str | None = None,
    runtime_dir: str | None = None,
    sam3_checkpoint: str | None = None,
) -> tuple[Path, Path]:
    validate_local_run_dir(run_dir, dataset_name)
    if arm == "B1":
        adapter, config, pose_stats, visual_stats, model = load_b1_assets(dataset_name, run_dir)
        frames = adapter.load_frames()
        sam3_provenance: dict[str, str] | None = None
    elif arm == "C1":
        adapter, config, pose_stats, visual_stats, model, frames = load_c1_assets(dataset_name, run_dir)
        sam3_provenance = config.sam3_provenance
    else:
        raise ValueError(f"arma inválida: {arm}")
    _check_output_dir(output_dir, _protected_paths(run_dir, adapter))
    selected = canonical_conditions(arm) if conditions is None else select_conditions(arm, conditions)
    json_path, csv_path = (
        _canonical_paths(output_dir, arm) if conditions is None
        else _shard_paths(output_dir, arm, selected)
    )
    alarm_path = run_dir / "alarm_protocol.yaml"
    if alarm_path.exists() and load_alarm_protocol(alarm_path) != BASELINE_A_ALARM_PROTOCOL:
        raise ValueError(f"protocolo de alarme divergente: {alarm_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    checkpoint_sha256 = sha256_file(run_dir / "checkpoint.pt")
    reports: list[dict[str, object]] = []
    backbone_provenance: dict[str, object] = {}
    with _staged_outputs(json_path, csv_path, force) as (json_tmp, csv_tmp):
        with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            for split in SPLITS:
                wanted = [key for key in selected if key[0] == split]
                if not wanted:
                    continue
                split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
                video_ids = sorted(str(value) for value in split_frames["video_id"].unique())
                clean = {video_id: _clean_inputs(adapter, arm, video_id) for video_id in video_ids}

                def evaluate(modality: str, severity: int, inputs: dict[str, VideoInputs]) -> None:
                    reports.append(_condition_report(
                        arm=arm, adapter=adapter, frames=frames, split=split,
                        modality=modality, degradation=DEGRADATIONS[modality], severity=severity,
                        inputs=inputs, model=model, pose_stats=pose_stats,
                        visual_stats=visual_stats, device=device,
                        batch_size=config.batch_size, checkpoint_sha256=checkpoint_sha256,
                        writer=writer,
                    ))

                if (split, "clean", 0) in wanted:
                    evaluate("clean", 0, clean)
                for severity in POSE_SEVERITIES:
                    if (split, "pose", severity) in wanted:
                        evaluate("pose", severity, _pose_condition(adapter, clean, severity))
                blur = [severity for _split, modality, severity in wanted if modality == "visual"]
                if arm == "B1":
                    repo_path = resolve_repo_dir(repo_dir)
                    weights_path = resolve_weights_path(weights)
                    backbone_provenance = dict(
                        _check_dinov3_backbone(adapter, frames, repo_path, weights_path)
                    )
                    if blur:
                        configure_deterministic_inference()
                        backbone = cast(Dinov3Backbone, load_backbone(repo_path, weights_path, device))
                        for severity in blur:
                            inputs = _b1_visual_condition(
                                adapter, split_frames, clean, severity, backbone, device, config.batch_size
                            )
                            evaluate("visual", severity, inputs)
                        del backbone
                else:
                    checkpoint = resolve_checkpoint_path(sam3_checkpoint)
                    runtime = resolve_runtime_project_dir(runtime_dir)
                    with Sam3RuntimeSegmenter(runtime_project_dir=runtime, checkpoint_path=checkpoint) as segmenter:
                        assert sam3_provenance is not None
                        provenance = sam3_provenance
                        if sha256_file(checkpoint) != provenance["sam3_checkpoint_sha256"]:
                            raise ValueError("checkpoint SAM 3 difere da fonte C1 congelada")
                        if segmenter.runtime_manifest.get("sam3_checkpoint_sha256") != provenance["sam3_checkpoint_sha256"]:
                            raise ValueError("checkpoint carregado pelo runtime SAM 3 difere da fonte C1")
                        if sha256_file(runtime / "uv.lock") != provenance["sam3_runtime_lock_sha256"]:
                            raise ValueError("runtime SAM 3 difere da fonte C1 congelada")
                        dtype = segmenter.runtime_manifest.get("sam3_inference_autocast_dtype")
                        if dtype != provenance["sam3_inference_autocast_dtype"]:
                            raise ValueError("dtype de inferência SAM 3 difere da fonte C1 congelada")
                        revision = segmenter.runtime_manifest.get("sam3_source_revision")
                        if revision != provenance["sam3_source_revision"]:
                            raise ValueError("revisão SAM 3 difere da fonte C1 congelada")
                        backbone_provenance = {
                            "sam3_checkpoint_sha256": sha256_file(checkpoint),
                            "sam3_runtime_lock_sha256": sha256_file(runtime / "uv.lock"),
                            "sam3_inference_autocast_dtype": dtype,
                            "sam3_source_revision": revision,
                        }
                        for severity in blur:
                            inputs = _c1_visual_condition(adapter, split_frames, clean, severity, segmenter)
                            evaluate("visual", severity, inputs)
        report: dict[str, object] = {
            "schema_version": 1,
            "arm": arm,
            "dataset": dataset_name,
            "run_dir": str(run_dir.resolve()),
            "checkpoint_path": str((run_dir / "checkpoint.pt").resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": sha256_file(run_dir / "config.yaml"),
            "frames_sha256": sha256_file(adapter.frames_path),
            "pose_stats_sha256": sha256_file(adapter.pose_stats_path),
            "visual_stats_sha256": sha256_file(
                dinov3_stats_path(dataset_name) if arm == "B1" else sam3_stats_path(dataset_name)
            ),
            "gate_parameters": gate_coefficients(model),
            "visual_quality_name": "q_visual" if arm == "B1" else "q_sam3",
            "alarm_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
            "test_split_is_descriptive_only": True,
            "selection_performed": False,
            "pose_severities": list(POSE_SEVERITIES),
            "pose_dropout_selection": POSE_DROPOUT_SELECTION,
            "numpy_version": np.__version__,
            "visual_blur_severities": list(visual_severities(arm)),
            "backbone_provenance": backbone_provenance,
        }
        if conditions is None:
            _add_deltas(reports)
            _write_json(json_tmp, {**report, "conditions": reports})
        else:
            _write_json(json_tmp, {
                "artifact": SHARD_ARTIFACT,
                "shard_schema_version": SHARD_SCHEMA_VERSION,
                "canonical": False,
                "partial": True,
                "shard_conditions": [condition_id(key) for key in selected],
                "trace_file": csv_path.name,
                "trace_sha256": sha256_file(csv_tmp),
                "report": report,
                "conditions": reports,
            })
    return json_path, csv_path


def _parse_condition_id(value: object, arm: str) -> ConditionKey:
    parts = str(value).split(":")
    if len(parts) != 3 or not parts[2].isdigit():
        raise ValueError(f"condição inválida: {value}")
    key = (parts[0], parts[1], int(parts[2]))
    if key not in canonical_conditions(arm):
        raise ValueError(f"condição fora da grade canônica de {arm}: {value}")
    return key


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_report(path: Path, report: object) -> dict[str, object]:
    if not isinstance(report, dict) or tuple(report) != REPORT_FIELDS:
        raise ValueError(f"{path}: campos do relatório divergem do contrato canônico")
    report = cast(dict[str, object], report)
    arm = report["arm"]
    if arm not in BACKBONE_PROVENANCE_FIELDS:
        raise ValueError(f"{path}: arma inválida no relatório: {arm}")
    arm = cast(str, arm)
    fixed: dict[str, object] = {
        "schema_version": 1,
        "visual_quality_name": "q_visual" if arm == "B1" else "q_sam3",
        "alarm_protocol": json.loads(json.dumps(BASELINE_A_ALARM_PROTOCOL.to_dict())),
        "test_split_is_descriptive_only": True,
        "selection_performed": False,
        "pose_severities": list(POSE_SEVERITIES),
        "pose_dropout_selection": POSE_DROPOUT_SELECTION,
        "visual_blur_severities": list(visual_severities(arm)),
    }
    differing = [name for name, value in fixed.items() if report[name] != value]
    differing += [name for name in REPORT_SHA256_FIELDS if not _is_sha256(report[name])]
    differing += [
        name for name in ("dataset", "run_dir", "checkpoint_path", "numpy_version")
        if not isinstance(report[name], str) or not report[name]
    ]
    gate = report["gate_parameters"]
    if not isinstance(gate, dict) or tuple(gate) != GATE_PARAMETER_FIELDS or not all(
        isinstance(value, float) for value in gate.values()
    ):
        differing.append("gate_parameters")
    provenance = report["backbone_provenance"]
    if not isinstance(provenance, dict) or set(provenance) != BACKBONE_PROVENANCE_FIELDS[arm] or not all(
        isinstance(value, str) and value for value in provenance.values()
    ):
        differing.append("backbone_provenance")
    if differing:
        raise ValueError(f"{path}: relatório fora do contrato canônico em {', '.join(differing)}")
    return report


def _validate_condition(path: Path, key: ConditionKey, row: object, report: dict[str, object]) -> None:
    if not isinstance(row, dict) or tuple(row) != CONDITION_FIELDS:
        raise ValueError(f"{path}: condição parcial {condition_id(key)}")
    statistics = row["statistics"]
    if not isinstance(statistics, dict) or tuple(statistics) != STATISTIC_NAMES or any(
        not isinstance(summary, dict) or tuple(summary) != SUMMARY_FIELDS
        for summary in statistics.values()
    ):
        raise ValueError(f"{path}: estatísticas parciais em {condition_id(key)}")
    if (
        row["arm"] != report["arm"] or row["dataset"] != report["dataset"]
        or row["checkpoint_sha256"] != report["checkpoint_sha256"]
        or row["gate_parameters"] != report["gate_parameters"]
        or row["degradation"] != DEGRADATIONS[key[1]]
        or not isinstance(row["n_frames"], int) or row["n_frames"] <= 0
    ):
        raise ValueError(f"{path}: identidade divergente em {condition_id(key)}")


def _trace_identity(key: ConditionKey, report: dict[str, object]) -> tuple[str, ...]:
    gate = cast(dict[str, float], report["gate_parameters"])
    return (
        str(report["arm"]), str(report["dataset"]), key[0], key[1], DEGRADATIONS[key[1]],
        str(key[2]), str(report["checkpoint_sha256"]),
        *(str(gate[name]) for name in GATE_PARAMETER_FIELDS),
    )


def _frame_sequence(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(row["video_id"], row["frame_index"]) for row in rows]


def _is_canonical_frame_order(sequence: list[tuple[str, str]]) -> bool:
    counts = Counter(video_id for video_id, _frame in sequence)
    videos = list(dict.fromkeys(video_id for video_id, _frame in sequence))
    if videos != sorted(videos):
        return False
    expected = [(video_id, str(index)) for video_id in videos for index in range(counts[video_id])]
    return sequence == expected


def read_shard(path: Path) -> Shard:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        not isinstance(payload, dict)
        or payload.get("artifact") != SHARD_ARTIFACT
        or payload.get("canonical") is not False
        or payload.get("partial") is not True
    ):
        raise ValueError(f"{path}: não é um shard de degradação do gate")
    if payload.get("shard_schema_version") != SHARD_SCHEMA_VERSION:
        raise ValueError(f"{path}: versão de shard incompatível: {payload.get('shard_schema_version')}")
    if set(payload) != SHARD_FIELDS:
        raise ValueError(f"{path}: campos do shard divergem do contrato")
    report = _validate_report(path, payload["report"])
    arm = str(report["arm"])
    declared = [_parse_condition_id(value, arm) for value in payload["shard_conditions"]]
    if not declared or len(set(declared)) != len(declared):
        raise ValueError(f"{path}: lista de condições vazia ou repetida")
    reports = cast(list[dict[str, object]], payload["conditions"])
    keys = [(str(row.get("split")), str(row.get("modality")), int(cast(int, row.get("severity", -1))))
            for row in reports]
    if keys != declared:
        raise ValueError(f"{path}: condições incompletas ou fora de ordem")
    for key, row in zip(keys, reports):
        _validate_condition(path, key, row, report)
    trace_name = str(payload["trace_file"])
    trace_path = path.parent / trace_name
    if Path(trace_name).name != trace_name or not trace_path.is_file():
        raise ValueError(f"{path}: trace ausente")
    if sha256_file(trace_path) != payload["trace_sha256"]:
        raise ValueError(f"{trace_path}: hash do trace diverge do shard")
    identities = {_trace_identity(key, report): key for key in declared}
    traces: dict[ConditionKey, list[dict[str, str]]] = {key: [] for key in declared}
    with trace_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRACE_FIELDS:
            raise ValueError(f"{trace_path}: cabeçalho do trace divergente")
        for row in reader:
            key = identities.get(tuple(row[name] for name in TRACE_IDENTITY_FIELDS))
            if key is None:
                raise ValueError(f"{trace_path}: identidade de linha fora das condições do shard")
            traces[key].append(row)
    conditions = dict(zip(keys, reports))
    for key, rows in traces.items():
        if len(rows) != conditions[key]["n_frames"]:
            raise ValueError(f"{trace_path}: trace parcial em {condition_id(key)}")
        if not _is_canonical_frame_order(_frame_sequence(rows)):
            raise ValueError(
                f"{trace_path}: quadros duplicados, ausentes ou fora da ordem canônica em {condition_id(key)}"
            )
    return Shard(path, report, conditions, traces)


def merge_shards(shard_paths: list[Path], output_dir: Path, *, force: bool = False) -> tuple[Path, Path]:
    shards = [read_shard(path) for path in shard_paths]
    if not shards:
        raise ValueError("nenhum shard informado")
    report = shards[0].report
    for shard in shards[1:]:
        differing = sorted(
            name for name in set(report) | set(shard.report)
            if report.get(name) != shard.report.get(name)
        )
        if differing:
            raise ValueError(f"{shard.path}: proveniência divergente em {', '.join(differing)}")
    arm = str(report["arm"])
    conditions: dict[ConditionKey, dict[str, object]] = {}
    traces: dict[ConditionKey, list[dict[str, str]]] = {}
    for shard in shards:
        for key, row in shard.conditions.items():
            if key in conditions and (conditions[key] != row or traces[key] != shard.traces[key]):
                raise ValueError(f"{shard.path}: condição duplicada conflitante {condition_id(key)}")
            conditions[key] = row
            traces[key] = shard.traces[key]
    canonical = canonical_conditions(arm)
    missing = [condition_id(key) for key in canonical if key not in conditions]
    if missing:
        raise ValueError(f"condições ausentes: {', '.join(missing)}")
    for key in canonical:
        if _frame_sequence(traces[key]) != _frame_sequence(traces[(key[0], "clean", 0)]):
            raise ValueError(f"quadros de {condition_id(key)} divergem da condição limpa do split")
    run_dir = Path(str(report["run_dir"]))
    _check_output_dir(output_dir, _protected_paths(run_dir, get_dataset(str(report["dataset"]))))
    reports = [dict(conditions[key]) for key in canonical]
    _add_deltas(reports)
    json_path, csv_path = _canonical_paths(output_dir, arm)
    with _staged_outputs(json_path, csv_path, force) as (json_tmp, csv_tmp):
        with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(TRACE_FIELDS)
            for key in canonical:
                writer.writerows([row[name] for name in TRACE_FIELDS] for row in traces[key])
        _write_json(json_tmp, {**report, "conditions": reports})
    return json_path, csv_path


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("le2i", "le2i-cv"), default="le2i")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-dir")
    parser.add_argument("--weights")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--sam3-checkpoint")
    parser.add_argument("--force", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Executa varreduras B1/C1 congeladas")
    analyze.add_argument("--arm", choices=("B1", "C1", "both"), default="both")
    analyze.add_argument("--b1-run-dir", type=Path)
    analyze.add_argument("--c1-run-dir", type=Path)
    _add_run_arguments(analyze)
    shard = subparsers.add_parser("shard", help="Executa um subconjunto não canônico de condições")
    shard.add_argument("--arm", choices=("B1", "C1"), required=True)
    shard.add_argument("--run-dir", type=Path)
    shard.add_argument("--conditions", nargs="+", required=True, metavar="SPLIT:MODALIDADE:SEVERIDADE")
    _add_run_arguments(shard)
    merge = subparsers.add_parser("merge", help="Une shards completos nas saídas canônicas")
    merge.add_argument("--shards", type=Path, nargs="+", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Verifica a análise com dados sintéticos")
    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.eval.analysis.selftests.gate_degradation import run_selftest

        if not run_selftest():
            raise SystemExit(1)
        return
    try:
        if args.command == "merge":
            paths = merge_shards(args.shards, args.output_dir, force=args.force)
            print(f"merge: {paths[0]} {paths[1]}")
            return
        if args.command == "shard":
            jobs = [(args.arm, args.run_dir, args.conditions)]
        else:
            arms = ("B1", "C1") if args.arm == "both" else (args.arm,)
            jobs = [(arm, args.b1_run_dir if arm == "B1" else args.c1_run_dir, None) for arm in arms]
        for arm, run_dir, conditions in jobs:
            if run_dir is None:
                run_dir = default_run_dir_for_arm(args.dataset, arm)
            paths = run_analysis(
                arm, args.dataset, run_dir, args.output_dir,
                force=args.force, conditions=conditions, repo_dir=args.repo_dir,
                weights=args.weights, runtime_dir=args.runtime_dir,
                sam3_checkpoint=args.sam3_checkpoint,
            )
            print(f"{arm}: {paths[0]} {paths[1]}")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"gate degradation FALHOU: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
