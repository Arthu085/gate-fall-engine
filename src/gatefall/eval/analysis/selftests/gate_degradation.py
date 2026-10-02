"""Checagens sintéticas da análise pós-hoc de degradação do gate."""

import csv
import json
import shutil
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
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
    canonical_conditions,
    drop_keypoints,
    gate_coefficients,
    gate_trace,
    merge_shards,
    replace_modality,
    run_analysis,
    select_conditions,
)
from gatefall.eval.baseline_b1.cli import _predict_with_identity
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.features.dinov3_standardization import Dinov3StandardizationStats
from gatefall.features.sam3_standardization import Sam3StandardizationStats
from gatefall.features.standardization import StandardizationStats
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features_from_arrays, feature_names
from gatefall.pose.loading import PoseArrays
from gatefall.pose.quality import pose_quality_from_arrays
from gatefall.sam3.quality import DEGRADATION_SWEEPS as SAM3_SWEEPS
from gatefall.sam3.quality import apply_frame_degradation
from gatefall.sam3.descriptors import CHANNEL_NAMES, V_T_DIM
from gatefall.sam3.runtime import Sam3Instance
from gatefall.train.baseline_b1.model import B1AdaptiveGateClassifier
from gatefall.train.baseline_c1.model import C1AdaptiveGateClassifier

MODULE = "gatefall.eval.analysis.gate_degradation"
SYNTHETIC_FRAMES = WINDOW_FRAMES + 6
SYNTHETIC_VIDEOS = {"val": "room/val_video", "test": "room/test_video"}
SAM3_PROVENANCE_FIELDS = ("sam3_inference_autocast_dtype", "sam3_source_revision")


def _pose(k: int = WINDOW_FRAMES) -> PoseArrays:
    keypoints = np.ones((k, 17, 3), dtype=np.float32)
    keypoints[:, :, 0] = np.arange(17, dtype=np.float32)[None, :] + 1
    keypoints[:, :, 1] = 10 - np.linspace(0, 6, k, dtype=np.float32)[:, None]
    bbox = np.tile(np.array([0, 0, 20, 20], dtype=np.float32), (k, 1))
    found = np.ones(k, dtype=bool)
    return PoseArrays(keypoints, bbox, found, k, 20, 20)


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


class _SyntheticSam3:
    """Substitui o runtime SAM 3 por uma segmentação determinística do conteúdo do quadro."""

    calls = 0
    fail_after: int | None = None

    def __init__(self, runtime_project_dir: Path, checkpoint_path: Path) -> None:
        self.runtime_manifest: dict[str, object] = {
            "sam3_checkpoint_sha256": sha256_file(checkpoint_path),
            "sam3_inference_autocast_dtype": "bfloat16",
            "sam3_source_revision": "synthetic",
        }

    def __enter__(self) -> "_SyntheticSam3":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def segment_frame(self, frame_rgb: np.ndarray, text_prompt: str) -> list[Sam3Instance]:
        type(self).calls += 1
        if self.fail_after is not None and type(self).calls > self.fail_after:
            raise RuntimeError("interrupção sintética do SAM 3")
        gray = frame_rgb[..., 0].astype(np.float32)
        mask = gray > gray.mean()
        if text_prompt != "person" or not mask.any():
            return []
        return [Sam3Instance(mask, float(np.clip(gray.std() / 64, 0.05, 0.99)))]


def _synthetic_frames() -> pd.DataFrame:
    rows = []
    for split, video_id in SYNTHETIC_VIDEOS.items():
        for frame_index in range(SYNTHETIC_FRAMES):
            fall = 10 <= frame_index < 16
            rows.append({
                "video_id": video_id, "split": split, "env": "room", "subject": 1,
                "frame_index": frame_index, "src_index": frame_index,
                "label": BASELINE_A_ALARM_PROTOCOL.fall_label if fall else 0,
            })
    return pd.DataFrame(rows)


def _synthetic_clean(_adapter: DatasetAdapter, _arm: str, video_id: str) -> VideoInputs:
    rng = np.random.default_rng(len(video_id))
    pose = build_pose_features_from_arrays(_pose(SYNTHETIC_FRAMES))[0].astype(np.float32)
    visual = rng.uniform(0, 1, (SYNTHETIC_FRAMES, V_T_DIM)).astype(np.float32)
    quality = rng.uniform(0.2, 0.9, (SYNTHETIC_FRAMES, 2)).astype(np.float32)
    return VideoInputs(pose, visual, quality)


def _synthetic_decode(_path: Path, indices: list[int]) -> list[np.ndarray]:
    gradient = np.arange(32, dtype=np.int64)[None, :, None] * 6
    return [
        np.tile((gradient + 4 * index) % 256, (32, 1, 3)).astype(np.uint8)
        for index in indices
    ]


def _synthetic_environment(root: Path, stack: ExitStack) -> Path:
    for name in ("raw", "processed", "pose", "dinov3", "sam3", "quality", "stats", "runtime", "run"):
        (root / name).mkdir()
    run_dir = root / "run"
    for path in (run_dir / "checkpoint.pt", run_dir / "config.yaml", root / "processed/frames.parquet",
                 root / "stats/pose.json", root / "stats/sam3.json", root / "runtime/uv.lock",
                 root / "sam3.pt"):
        path.write_text(path.name, encoding="utf-8")
    adapter = SimpleNamespace(
        identifier="le2i", raw_dir=root / "raw", frames_path=root / "processed/frames.parquet",
        pose_root=root / "pose", dinov3_root=root / "dinov3", sam3_root=root / "sam3",
        quality_root=root / "quality", pose_stats_path=root / "stats/pose.json",
        video_paths=lambda: {video_id: root / "raw/video.avi" for video_id in SYNTHETIC_VIDEOS.values()},
    )
    config = SimpleNamespace(batch_size=8, sam3_provenance={
        "sam3_checkpoint_sha256": sha256_file(root / "sam3.pt"),
        "sam3_runtime_lock_sha256": sha256_file(root / "runtime/uv.lock"),
        "sam3_inference_autocast_dtype": "bfloat16",
        "sam3_source_revision": "synthetic",
    })
    pose_stats, _ = _stats()
    visual_stats = Sam3StandardizationStats(
        "sam3", "le2i", "train", TARGET_FPS, WINDOW_FRAMES, TRAIN_STRIDE, 1, V_T_DIM,
        list(CHANNEL_NAMES), [0.0] * V_T_DIM, [1.0] * V_T_DIM, 0, [False] * V_T_DIM,
        "synthetic", "synthetic",
    )
    # Restringe a cabeça às classes 0/1 para que as métricas variem entre condições
    # e os deltas contra a condição limpa não sejam trivialmente nulos.
    torch.manual_seed(5)
    model = C1AdaptiveGateClassifier([4], 3, [1], 0.0).eval()
    with torch.no_grad():
        model.classifier.weight[2:] = 0
        model.classifier.weight[:2] *= 8
        model.classifier.bias[2:] = -10
        model.classifier.bias[1] = model.classifier.bias[0] - 0.5
    assets = (adapter, config, pose_stats, visual_stats, model, _synthetic_frames())
    replacements: dict[str, object] = {
        "validate_local_run_dir": lambda *_args: None,
        "load_c1_assets": lambda *_args: assets,
        "_clean_inputs": _synthetic_clean,
        "load_pose": lambda *_args, **_kwargs: _pose(SYNTHETIC_FRAMES),
        "decode_frames": _synthetic_decode,
        "Sam3RuntimeSegmenter": _SyntheticSam3,
        "resolve_checkpoint_path": lambda _value: root / "sam3.pt",
        "resolve_runtime_project_dir": lambda _value: root / "runtime",
        "sam3_stats_path": lambda _dataset: root / "stats/sam3.json",
    }
    for name, value in replacements.items():
        stack.enter_context(patch(f"{MODULE}.{name}", value))
    return run_dir


def _copy_shard(
    source: Path,
    target_dir: Path,
    edit_report: Callable[[dict], None] | None = None,
    edit_trace: Callable[[list[str]], list[str]] | None = None,
    rehash: bool = True,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    trace = target_dir / payload["trace_file"]
    shutil.copyfile(source.parent / payload["trace_file"], trace)
    if edit_trace is not None:
        lines = trace.read_bytes().decode("utf-8").split("\r\n")
        trace.write_bytes("\r\n".join(edit_trace(lines)).encode("utf-8"))
        if rehash:
            payload["trace_sha256"] = sha256_file(trace)
    if edit_report is not None:
        edit_report(payload)
    target = target_dir / source.name
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _rejects(action: Callable[[], object]) -> bool:
    try:
        action()
    except ValueError:
        return True
    return False


def _shard_merge_checks() -> dict[str, bool]:
    cheap = ["*:clean:0", "*:pose:*", "*:visual:0", "*:visual:2", "*:visual:12"]
    expensive = ["*:visual:3", "*:visual:6"]
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        run_dir = _synthetic_environment(root, stack)
        _SyntheticSam3.calls = 0
        full_json, full_csv = run_analysis("C1", "le2i", run_dir, root / "full")
        full_calls = _SyntheticSam3.calls
        _SyntheticSam3.calls = 0
        shard_a = run_analysis("C1", "le2i", run_dir, root / "gpu0", conditions=cheap)[0]
        shard_b = run_analysis("C1", "le2i", run_dir, root / "gpu1", conditions=expensive)[0]
        shard_calls = _SyntheticSam3.calls
        merged_json, merged_csv = merge_shards([shard_b, shard_a], root / "merged")
        shard_payload = json.loads(shard_a.read_text(encoding="utf-8"))
        deltas = [
            value for row in json.loads(full_json.read_text(encoding="utf-8"))["conditions"]
            for value in row["delta_from_clean"].values()
        ]
        equivalence = (
            any(value not in (None, 0) for value in deltas)
            and merged_json.name == full_json.name and merged_csv.name == full_csv.name
            and merged_json.read_bytes() == full_json.read_bytes()
            and merged_csv.read_bytes() == full_csv.read_bytes()
            and full_calls == shard_calls == 8 * SYNTHETIC_FRAMES
        )
        partial_marker = (
            ".partial." in shard_a.name
            and shard_payload["canonical"] is False and shard_payload["partial"] is True
            and "delta_from_clean" not in shard_payload["conditions"][0]
            and _rejects(lambda: merge_shards([full_json], root / "rejected"))
        )
        selection = (
            select_conditions("C1", ["*:visual:2"]) == [("val", "visual", 2), ("test", "visual", 2)]
            and select_conditions("C1", ["*:*:*"]) == canonical_conditions("C1")
            and _rejects(lambda: select_conditions("C1", ["val:visual:5"]))
            and _rejects(lambda: select_conditions("C1", ["val:visual"]))
        )

        def drop_last_condition(payload: dict) -> None:
            payload["conditions"].pop()

        def drop_metric(payload: dict) -> None:
            del payload["conditions"][-1]["macro_f1_restricted"]

        def drop_last_row(lines: list[str]) -> list[str]:
            return lines[:-2] + lines[-1:]

        def alter_metric(payload: dict) -> None:
            payload["conditions"][0]["macro_f1_restricted"] += 0.125

        def alter_frames(payload: dict) -> None:
            payload["report"]["frames_sha256"] = "0" * 64

        def alter_sam3(payload: dict) -> None:
            payload["report"]["backbone_provenance"]["sam3_source_revision"] = "other"

        def unfinished(payload: dict) -> None:
            payload["partial"] = False
            payload["canonical"] = True

        _SyntheticSam3.fail_after = SYNTHETIC_FRAMES + 3
        _SyntheticSam3.calls = 0
        interrupted = _rejects_runtime(
            lambda: run_analysis("C1", "le2i", run_dir, root / "interrupted", conditions=expensive)
        )
        _SyntheticSam3.fail_after = None
        interrupted = interrupted and not any((root / "interrupted").iterdir())
        identical_json = merge_shards(
            [shard_a, shard_b, _copy_shard(shard_b, root / "copy")], root / "dedup"
        )[0]
        return {
            "shard merge equals unsharded output": equivalence,
            "shard artifacts are marked partial": partial_marker,
            "shard condition selection": selection,
            "merge rejects missing conditions": _rejects(lambda: merge_shards([shard_a], root / "missing")),
            "merge rejects partial conditions": all((
                interrupted,
                _rejects(lambda: merge_shards([shard_a, _copy_shard(shard_b, root / "p1", drop_last_condition)], root / "x")),
                _rejects(lambda: merge_shards([shard_a, _copy_shard(shard_b, root / "p2", drop_metric)], root / "x")),
                _rejects(lambda: merge_shards([shard_a, _copy_shard(shard_b, root / "p3", edit_trace=drop_last_row)], root / "x")),
                _rejects(lambda: merge_shards([shard_a, _copy_shard(shard_b, root / "p4", edit_trace=drop_last_row, rehash=False)], root / "x")),
                _rejects(lambda: merge_shards([shard_a, _copy_shard(shard_b, root / "p5", unfinished)], root / "x")),
            )),
            "merge deduplicates identical and rejects conflicting conditions": (
                identical_json.read_bytes() == full_json.read_bytes()
                and _rejects(lambda: merge_shards(
                    [shard_a, shard_b, _copy_shard(shard_b, root / "c1", alter_metric)], root / "x"
                ))
            ),
            "merge rejects provenance mismatch": all((
                _rejects(lambda: merge_shards([shard_a, _copy_shard(shard_b, root / "m1", alter_frames)], root / "x")),
                _rejects(lambda: merge_shards([shard_a, _copy_shard(shard_b, root / "m2", alter_sam3)], root / "x")),
            )),
            "merge keeps canonical sources protected": (
                _rejects(lambda: merge_shards([shard_a, shard_b], run_dir))
                and not (root / "x").exists()
            ),
        }


def _rejects_runtime(action: Callable[[], object]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


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
        **_shard_merge_checks(),
    }
    for name, valid in checks.items():
        print(f"[{'PASS' if valid else 'FAIL'}] {name}")
    return all(checks.values())
