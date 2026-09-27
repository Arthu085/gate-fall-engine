"""Checagens sintéticas da fusão C1, sem acesso aos vídeos reais."""

import tempfile
import shutil
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import cast
import yaml

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.data.windowing import window_frame_indices
from gatefall.datasets import DatasetAdapter
from gatefall.features.sam3_standardization import Sam3StandardizationStats
from gatefall.features.standardization import StandardizationStats, excluded_dimension_mask
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_names
from gatefall.runs import REFERENCE_RUN_ROOT, default_run_dir_for_arm
from gatefall.sam3.descriptors import CHANNEL_NAMES, V_T_DIM
from gatefall.sam3.quality import compute_sam3_quality
from gatefall.train.b1_config import B1_ADAPTIVE_GATE_CONFIG
from gatefall.train.b1_model import B1AdaptiveGateClassifier
from gatefall.train.c0_config import C0_FUSION_CONFIG
from gatefall.train.c0_model import C0FusionClassifier
from gatefall.train.config import BASELINE_A_CONFIG
from gatefall.train.c1_artifacts import load_compatible_c1_checkpoint, validate_c1_training_run
from gatefall.train.c1_config import C1_ADAPTIVE_GATE_CONFIG
from gatefall.train.c1_engine import run_c1_training
from gatefall.train.c1_gate import _guard_protected_output, _guard_run_dir
from gatefall.train.c1_model import C1AdaptiveGateClassifier
from gatefall.train.b1_engine import _StandardizedGatedFusionTorchDataset
import gatefall.train.c1_gate as c1_gate


def _check(name: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return condition


def _raises(callback) -> bool:
    try:
        callback()
    except (ValueError, RuntimeError):
        return True
    return False


def _sources() -> tuple[GatedFusionWindowDataset, GatedFusionWindowDataset, GatedFusionWindowDataset]:
    specs = {"train_video": ("train", 26), "val_video": ("val", 25), "test_video": ("test", 25)}
    frames = pd.concat([
        pd.DataFrame({"video_id": video_id, "split": split, "env": "room", "subject": index,
                      "frame_index": np.arange(k), "label": np.arange(k) % 2})
        for index, (video_id, (split, k)) in enumerate(specs.items())
    ], ignore_index=True)
    pose = {video_id: np.tile(np.arange(k, dtype=np.float32)[:, None], (1, POSE_FEATURE_DIM))
            for video_id, (_, k) in specs.items()}
    visual = {video_id: np.tile(np.arange(k, dtype=np.float32)[:, None], (1, V_T_DIM))
              for video_id, (_, k) in specs.items()}
    quality = {video_id: np.column_stack((np.arange(k) / k, np.arange(k) / (2 * k))).astype(np.float32)
               for video_id, (_, k) in specs.items()}
    def make(split: str, stride: int) -> GatedFusionWindowDataset:
        return GatedFusionWindowDataset(frames, split, stride, lambda key: pose[key],
                                        lambda key: visual[key], lambda key: quality[key], visual_dim=V_T_DIM)
    return make("train", TRAIN_STRIDE), make("val", EVAL_STRIDE), make("test", EVAL_STRIDE)


def _pose_stats() -> StandardizationStats:
    names = feature_names()
    return StandardizationStats("pose", "train", 10.0, WINDOW_FRAMES, TRAIN_STRIDE, 1,
                                POSE_FEATURE_DIM, names, excluded_dimension_mask(names).tolist(),
                                [0.0] * POSE_FEATURE_DIM, [1.0] * POSE_FEATURE_DIM,
                                0, [False] * POSE_FEATURE_DIM, "synthetic")


def _visual_stats() -> Sam3StandardizationStats:
    return Sam3StandardizationStats("sam3", "le2i", "train", 10.0, WINDOW_FRAMES,
                                    TRAIN_STRIDE, 1, V_T_DIM, list(CHANNEL_NAMES),
                                    [0.0] * V_T_DIM, [1.0] * V_T_DIM, 0,
                                    [False] * V_T_DIM, "synthetic", "synthetic")


def run_c1_selftest() -> bool:
    checks: list[bool] = []
    shared = ("seed", "window_frames", "train_stride", "eval_stride", "epochs", "batch_size",
              "channels", "dilations", "kernel_size", "dropout", "optimizer_name", "lr",
              "weight_decay", "grad_clip_norm", "lr_schedule_name", "loss_name",
              "class_weighted", "receptive_field", "num_classes")
    checks.append(_check("receita C1 idêntica a A, B1 e C0", all(
        getattr(C1_ADAPTIVE_GATE_CONFIG, key) == getattr(other, key)
        for other in (BASELINE_A_CONFIG, B1_ADAPTIVE_GATE_CONFIG, C0_FUSION_CONFIG)
        for key in shared
    )))
    checks.append(_check("dimensões do gate e SAM 3", C1_ADAPTIVE_GATE_CONFIG.visual_dim == V_T_DIM == 10
                         and C1_ADAPTIVE_GATE_CONFIG.pose_dim == 134
                         and C1_ADAPTIVE_GATE_CONFIG.projection_dim == 128
                         and C1_ADAPTIVE_GATE_CONFIG.fused_dim == 256
                         and C1_ADAPTIVE_GATE_CONFIG.gate_input_dim == 2))
    model = C1AdaptiveGateClassifier([8, 8], dilations=[1, 2]).eval()
    quality = torch.rand(2, WINDOW_FRAMES, 2)
    with torch.no_grad():
        weights = model.gate(quality)
        prediction = model(torch.rand(2, WINDOW_FRAMES, 134),
                           torch.rand(2, WINDOW_FRAMES, 10), quality)
    checks.append(_check("gate limitado e complementar; classificador usa 10 canais",
                         bool(torch.all((weights >= 0) & (weights <= 1)))
                         and bool(torch.all(weights + (1 - weights) == 1))
                         and tuple(prediction.shape) == (2, C1_ADAPTIVE_GATE_CONFIG.num_classes)
                         and cast(nn.Linear, model.e_v[0]).in_features == 10))
    score = np.array([0.8, 0.7], dtype=np.float32)
    checks.append(_check("q_sam3 usa score vezes presença", np.array_equal(
        compute_sam3_quality(score, np.array([1, 0])), np.array([0.8, 0], dtype=np.float32))))
    train, val, test = _sources()
    pose, visual, q, _, (_, end) = train[0]
    expected = window_frame_indices(end, 26).astype(np.float32)
    checks.append(_check("janelas alinhadas, causais e com replicação de borda",
                         np.array_equal(pose[:, 0], expected)
                         and np.array_equal(visual[:, 0], expected)
                         and np.allclose(q[:, 0], expected / 26)
                         and int(expected.max()) == end))
    standardized = _StandardizedGatedFusionTorchDataset(train, _pose_stats(), _visual_stats())
    checks.append(_check("qualidade entra crua após padronizar pose e SAM 3",
                         np.array_equal(standardized[0][2].numpy(), q)))
    k = 26
    frames = pd.DataFrame({
        "video_id": ["room/video"] * k,
        "split": ["train"] * k,
        "env": ["room"] * k,
        "subject": [1] * k,
        "frame_index": np.arange(k),
        "label": np.zeros(k, dtype=np.int8),
    })
    pose_values = np.zeros((k, POSE_FEATURE_DIM), dtype=np.float32)
    visual_values = np.zeros((k, V_T_DIM), dtype=np.float32)
    visual_values[:, 0] = np.arange(k) % 2
    pose_quality = np.arange(k, dtype=np.float32) / k
    scores = np.arange(k, dtype=np.float32) / k
    with (patch.object(c1_gate, "build_pose_features", return_value=(pose_values, None)),
          patch.object(c1_gate, "load_v_t", return_value=visual_values),
          patch.object(c1_gate, "compute_pose_quality", return_value=SimpleNamespace(q_pose=pose_quality)),
          patch.object(c1_gate, "read_sam_score", return_value=scores)):
        adapter = cast(DatasetAdapter, SimpleNamespace(pose_root=Path("pose"), sam3_root=Path("sam3")))
        source = c1_gate._split_sources(adapter, frames)["train"]
        _, _, wired_quality, _, (_, wired_end) = source[0]
    wired_indices = window_frame_indices(wired_end, k)
    checks.append(_check("C1 lê score SAM 3 e present de V_t na mesma janela da pose",
                         np.array_equal(wired_quality[:, 0], pose_quality[wired_indices])
                         and np.array_equal(wired_quality[:, 1],
                                            scores[wired_indices] * visual_values[wired_indices, 0])))
    checks.append(_check("run C1 isolado dos runs A/B0/B1/C0 e referência", all(
        _raises(lambda path=path: _guard_run_dir(path, "le2i"))
        for path in (default_run_dir_for_arm("le2i", name)
                     for name in ("baseline_a", "b0_fusion", "b1_adaptive_gate", "c0_fusion")))
        and _raises(lambda: _guard_run_dir(REFERENCE_RUN_ROOT, "le2i"))
        and _raises(lambda: _guard_run_dir(default_run_dir_for_arm("le2i-cv", "c1_adaptive_gate"), "le2i"))
        and _raises(lambda: _guard_protected_output(
            default_run_dir_for_arm("le2i", "c1_adaptive_gate"),
            default_run_dir_for_arm("le2i", "c0_fusion") / "checkpoint.pt", "le2i"))
        and _raises(lambda: _guard_protected_output(
            default_run_dir_for_arm("le2i", "c1_adaptive_gate"),
            default_run_dir_for_arm("le2i-cv", "baseline_a") / "metrics.json", "le2i"))))
    config = replace(C1_ADAPTIVE_GATE_CONFIG, epochs=1, batch_size=4, channels=[8, 8],
                     dilations=[1, 2], pose_standardization_stats_sha256="pose-stats",
                     visual_standardization_stats_sha256="sam3-stats",
                     pose_features_sha256="pose", sam3_features_sha256="sam3",
                     quality_features_sha256="quality", sam3_provenance={"model_name": "sam3"})
    with tempfile.TemporaryDirectory() as tmp:
        roots = [Path(tmp) / "first", Path(tmp) / "second"]
        for root in roots:
            run_c1_training(train, val, test, _pose_stats(), _visual_stats(), config,
                            root, False, tuple(f"class_{i}" for i in range(config.num_classes)))
        first = validate_c1_training_run(roots[0], expected_config=config,
                                         fields_allowed_to_differ=frozenset({"trainable_param_count"}))
        second = validate_c1_training_run(roots[1], expected_config=config,
                                          fields_allowed_to_differ=frozenset({"trainable_param_count"}))
        first_state = torch.load(roots[0] / "checkpoint.pt", weights_only=True)
        second_state = torch.load(roots[1] / "checkpoint.pt", weights_only=True)
        checks.append(_check("treino sintético determinístico e artefatos válidos",
                             first.arm == second.arm == "C1" and all(
                                 torch.equal(first_state[key], second_state[key]) for key in first_state)))
        incompatible = Path(tmp) / "incompatible.pt"
        b1_model = B1AdaptiveGateClassifier([8, 8], dilations=[1, 2])
        torch.save(b1_model.state_dict(), incompatible)
        rejects_b1 = _raises(lambda: load_compatible_c1_checkpoint(incompatible, first))
        torch.save(C0FusionClassifier([8, 8], dilations=[1, 2]).state_dict(), incompatible)
        rejects_c0 = _raises(lambda: load_compatible_c1_checkpoint(incompatible, first))
        checks.append(_check("checkpoints B1 e C0 incompatíveis recusados", rejects_b1 and rejects_c0))
        foreign = Path(tmp) / "foreign"
        shutil.copytree(roots[0], foreign)
        config_path = foreign / "config.yaml"
        with config_path.open(encoding="utf-8") as stream:
            foreign_config = yaml.safe_load(stream)
        foreign_config["arm"] = "B1"
        with config_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(foreign_config, stream)
        checks.append(_check("run de outra arma recusado", _raises(lambda: validate_c1_training_run(foreign))))
        checks.append(_check("hash de fonte obsoleta recusado", _raises(lambda: validate_c1_training_run(
            roots[0], expected_config=replace(config, sam3_features_sha256="changed"),
            fields_allowed_to_differ=frozenset({"trainable_param_count"})))))
        report_path = Path(tmp) / "classification_report.json"
        adapter = SimpleNamespace(label_names=tuple(f"class_{i}" for i in range(config.num_classes)))
        inputs = SimpleNamespace(pose_stats=_pose_stats(), visual_stats=_visual_stats(), frames=pd.DataFrame())
        with (patch.object(c1_gate, "get_dataset", return_value=adapter),
              patch.object(c1_gate, "ensure_sam3_dataset_supported"),
              patch.object(c1_gate, "_validated_inputs", return_value=inputs),
              patch.object(c1_gate, "_resolve_config", return_value=config),
              patch.object(c1_gate, "_split_sources", return_value={"train": train, "val": val, "test": test})):
            report_ok = c1_gate.run_report("le2i", roots[0], report_path, False)
        with report_path.open(encoding="utf-8") as stream:
            report = json.load(stream)
        checks.append(_check("report C1 revalida métricas do run e grava diagnóstico",
                             report_ok and report["verification_against_metrics_json"]["ok"]
                             and set(report["splits"]) == {"train", "val", "test"}))
    return all(checks)
