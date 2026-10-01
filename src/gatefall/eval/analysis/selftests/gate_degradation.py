"""Checagens sintéticas da análise pós-hoc de degradação do gate."""

import csv
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from gatefall.config import TARGET_FPS, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.datasets.base import DatasetAdapter
from gatefall.dinov3.quality import DEGRADATION_SWEEPS, apply_degradation
from gatefall.eval.analysis.gate_degradation import (
    POSE_SEVERITIES,
    TRACE_FIELDS,
    VideoInputs,
    _add_deltas,
    _c1_visual_condition,
    _condition_report,
    _predict,
    drop_keypoints,
    gate_coefficients,
    gate_trace,
    replace_modality,
)
from gatefall.eval.baseline_b1.cli import _predict_with_identity
from gatefall.features.dinov3_standardization import Dinov3StandardizationStats
from gatefall.features.standardization import StandardizationStats
from gatefall.pose.kinematics import feature_names
from gatefall.pose.loading import PoseArrays
from gatefall.pose.quality import pose_quality_from_arrays
from gatefall.sam3.quality import DEGRADATION_SWEEPS as SAM3_SWEEPS
from gatefall.sam3.quality import apply_frame_degradation
from gatefall.sam3.runtime import Sam3Instance
from gatefall.train.baseline_b1.model import B1AdaptiveGateClassifier


def _pose() -> PoseArrays:
    keypoints = np.ones((WINDOW_FRAMES, 17, 3), dtype=np.float32)
    keypoints[:, :, 0] = np.arange(17, dtype=np.float32)[None, :] + 1
    keypoints[:, :, 1] = 10
    bbox = np.tile(np.array([0, 0, 20, 20], dtype=np.float32), (WINDOW_FRAMES, 1))
    found = np.ones(WINDOW_FRAMES, dtype=bool)
    return PoseArrays(keypoints, bbox, found, WINDOW_FRAMES, 20, 20)


def _stats() -> tuple[StandardizationStats, Dinov3StandardizationStats]:
    names = feature_names()
    pose = StandardizationStats(
        "pose", "train", TARGET_FPS, WINDOW_FRAMES, TRAIN_STRIDE, 1, 134,
        names, [False] * 134, [0.0] * 134, [1.0] * 134, 0,
        [False] * 134, "synthetic",
    )
    visual = Dinov3StandardizationStats(
        "dinov3", "le2i", "train", TARGET_FPS, WINDOW_FRAMES,
        TRAIN_STRIDE, 1, 1536, [0.0] * 1536, [1.0] * 1536,
        0, [False] * 1536, "synthetic",
    )
    return pose, visual


def _sam3_path_check() -> bool:
    class FakeSegmenter:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []

        def segment_frame(self, frame_rgb: np.ndarray, text_prompt: str) -> list[Sam3Instance]:
            if text_prompt != "person":
                raise ValueError("prompt divergente")
            self.calls.append(frame_rgb.copy())
            left = np.zeros((8, 8), dtype=bool)
            right = np.zeros((8, 8), dtype=bool)
            left[2:6, 1:3] = True
            right[2:6, 5:7] = True
            score = 0.9 if len(self.calls) == 1 else 0.2
            return [Sam3Instance(left, score), Sam3Instance(right, 0.95 if len(self.calls) > 1 else 0.5)]

    frames_rgb = [
        np.tile(np.arange(8, dtype=np.uint8)[None, :, None], (8, 1, 3)) * 20
        for _ in range(2)
    ]
    frames = pd.DataFrame({
        "video_id": ["room/video"] * 2, "frame_index": [0, 1], "src_index": [0, 1]
    })
    clean = VideoInputs(
        np.ones((2, 134), dtype=np.float32),
        np.zeros((2, 10), dtype=np.float32),
        np.full((2, 2), 0.7, dtype=np.float32),
    )
    adapter = cast(DatasetAdapter, SimpleNamespace(video_paths=lambda: {"room/video": Path("video.avi")}))
    segmenter = FakeSegmenter()
    with patch("gatefall.eval.analysis.gate_degradation.decode_frames", return_value=frames_rgb):
        output = _c1_visual_condition(
            adapter, frames, {"room/video": clean}, 2,
            segmenter,
        )["room/video"]
    return bool(
        len(segmenter.calls) == 2
        and not np.array_equal(segmenter.calls[0], frames_rgb[0])
        and np.array_equal(output.pose, clean.pose)
        and np.array_equal(output.quality[:, 0], clean.quality[:, 0])
        and np.allclose(output.quality[:, 1], [0.9, 0.2])
        and np.all(output.visual[:, 2] < 0.5)
    )


def run_selftest() -> bool:
    raw = _pose()
    dropped = [drop_keypoints(raw, "room/video", severity) for severity in POSE_SEVERITIES]
    counts = [int(np.count_nonzero(item.keypoints[0, :, 2] == 0)) for item in dropped]
    deterministic = np.array_equal(
        dropped[2].keypoints, drop_keypoints(raw, "room/video", POSE_SEVERITIES[2]).keypoints
    )
    nested = all(
        set(np.flatnonzero(left.keypoints[0, :, 2] == 0)).issubset(
            np.flatnonzero(right.keypoints[0, :, 2] == 0)
        )
        for left, right in zip(dropped[:-1], dropped[1:])
    )
    q_values = [pose_quality_from_arrays(item.keypoints, item.bbox, item.person_found).q_pose
                for item in dropped]
    pose_order = counts == list(POSE_SEVERITIES) and deterministic and nested
    pose_order = pose_order and all(
        np.all(right <= left) for left, right in zip(q_values[:-1], q_values[1:])
    )

    clean = VideoInputs(
        np.ones((WINDOW_FRAMES, 134), dtype=np.float32),
        np.full((WINDOW_FRAMES, 1536), 0.25, dtype=np.float32),
        np.full((WINDOW_FRAMES, 2), 0.75, dtype=np.float32),
    )
    pose_changed = replace_modality(
        clean, "pose", clean.pose * 0.5, np.full(WINDOW_FRAMES, 0.5, np.float32)
    )
    visual_changed = replace_modality(
        clean, "visual", clean.visual * 0.5, np.full(WINDOW_FRAMES, 0.5, np.float32)
    )
    isolated = (
        np.array_equal(pose_changed.visual, clean.visual)
        and np.array_equal(pose_changed.quality[:, 1], clean.quality[:, 1])
        and np.array_equal(visual_changed.pose, clean.pose)
        and np.array_equal(visual_changed.quality[:, 0], clean.quality[:, 0])
        and np.array_equal(clean.quality, np.full((WINDOW_FRAMES, 2), 0.75))
    )

    pixels = np.arange(224, dtype=np.float32)
    grid = np.tile(pixels[None, None, None, :], (1, 3, 224, 1)) / 223
    blurred = [apply_degradation(torch.from_numpy(grid), "blur", level)
               for level in DEGRADATION_SWEEPS["blur"]]
    frame = np.tile(np.arange(32, dtype=np.uint8)[None, :, None], (32, 1, 3))
    sam_blurred = [apply_frame_degradation(frame, "blur", level)
                   for level in SAM3_SWEEPS["blur"]]
    severity_order = (
        tuple(DEGRADATION_SWEEPS["blur"]) == (0, 2, 3, 6, 12)
        and tuple(SAM3_SWEEPS["blur"]) == (0, 2, 3, 6, 12)
        and torch.equal(blurred[0], torch.from_numpy(grid))
        and np.array_equal(sam_blurred[0], frame)
        and all(
            float(torch.mean(torch.abs(right[:, :, :, 1:] - right[:, :, :, :-1])))
            <= float(torch.mean(torch.abs(left[:, :, :, 1:] - left[:, :, :, :-1]))) + 1e-6
            for left, right in zip(blurred[:-1], blurred[1:])
        )
        and all(
            float(np.mean(np.abs(np.diff(right.astype(np.float32), axis=1))))
            <= float(np.mean(np.abs(np.diff(left.astype(np.float32), axis=1)))) + 1e-6
            for left, right in zip(sam_blurred[:-1], sam_blurred[1:])
        )
    )

    torch.manual_seed(1)
    model = B1AdaptiveGateClassifier([4], 3, [1], 0.0).eval()
    with torch.no_grad():
        model.gate.linear.weight[:] = torch.tensor([[2.0, -3.0]])
        model.gate.linear.bias[:] = torch.tensor([0.5])
    coefficients = gate_coefficients(model)
    quality = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    pose = clean.pose[:2]
    visual = clean.visual[:2]
    gate, pose_norm, visual_norm = gate_trace(model, pose, visual, quality, "cpu")
    expected_gate = torch.sigmoid(torch.tensor([1.5, -1.5])).numpy()
    expected_pose = torch.linalg.vector_norm(model.e_p(torch.from_numpy(pose)), dim=-1).detach().numpy()
    expected_visual = torch.linalg.vector_norm(model.e_v(torch.from_numpy(visual)), dim=-1).detach().numpy()
    algebra = (
        coefficients == {"gate_weight_pose": 2.0, "gate_weight_visual": -3.0, "gate_bias": 0.5}
        and np.allclose(gate, expected_gate)
        and gate[0] > gate[1]
        and np.allclose(pose_norm, expected_gate * expected_pose)
        and np.allclose(visual_norm, (1 - expected_gate) * expected_visual)
    )

    frames = pd.DataFrame({
        "video_id": ["room/video"] * WINDOW_FRAMES,
        "split": ["val"] * WINDOW_FRAMES,
        "env": ["room"] * WINDOW_FRAMES,
        "subject": [1] * WINDOW_FRAMES,
        "frame_index": list(range(WINDOW_FRAMES)),
        "label": [0] * WINDOW_FRAMES,
    })
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=TRACE_FIELDS)
    writer.writeheader()
    pose_stats, visual_stats = _stats()
    source = GatedFusionWindowDataset(
        frames, "val", 1, lambda _video_id: clean.pose,
        lambda _video_id: clean.visual,
        lambda _video_id: clean.quality, drop_ignored=False,
    )
    parity = _predict(model, source, pose_stats, visual_stats, "B1", "cpu", 8) == (
        _predict_with_identity(model, source, pose_stats, visual_stats, "cpu", 8)
    )
    report = _condition_report(
        arm="B1", adapter=cast(DatasetAdapter, SimpleNamespace(identifier="le2i")),
        frames=frames, split="val", modality="clean", degradation="none", severity=0,
        inputs={"room/video": clean}, model=model, pose_stats=pose_stats,
        visual_stats=visual_stats, device="cpu", batch_size=8,
        checkpoint_sha256="synthetic", writer=writer,
    )
    rows = list(csv.DictReader(StringIO(output.getvalue())))
    baseline = dict(report)
    degraded_report = dict(report, modality="pose", degradation="keypoint_dropout", severity=4)
    reports = [baseline, degraded_report]
    _add_deltas(reports)
    schema = (
        len(rows) == WINDOW_FRAMES
        and tuple(rows[0]) == TRACE_FIELDS
        and rows[0]["checkpoint_sha256"] == "synthetic"
        and np.isclose(float(rows[0]["g_visual"]), 1 - float(rows[0]["g_pose"]))
        and report["n_frames"] == WINDOW_FRAMES
        and report["n_windows"] == WINDOW_FRAMES
        and report["gate_statistics_unit"] == "video_frame"
        and "macro_f1_restricted" in report
        and "event_latency_seconds" in report
        and cast(dict[str, float], degraded_report["delta_from_clean"])["macro_f1_restricted"] == 0
    )
    checks = {
        "pose severity and determinism": pose_order,
        "modality isolation": isolated,
        "visual blur grid": severity_order,
        "SAM 3 prompt, native blur, and continuity": _sam3_path_check(),
        "gate coefficients and effective norms": algebra,
        "clean inference parity": parity,
        "output schema and clean deltas": schema,
    }
    for name, valid in checks.items():
        print(f"[{'PASS' if valid else 'FAIL'}] {name}")
    return all(checks.values())
