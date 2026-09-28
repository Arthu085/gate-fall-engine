"""Checagens sintéticas da avaliação de eventos da arma C1."""

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

import gatefall.eval.baseline_c1.cli as c1_events
from gatefall.config import IGNORE_LABEL
from gatefall.datasets import DatasetAdapter
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, load_alarm_protocol
from gatefall.eval.shared.event_artifacts import (
    EVENT_TRANSACTION_FILE,
    _promote_event_outputs,
    validate_event_metrics,
)
from gatefall.features.sam3_standardization import Sam3StandardizationStats
from gatefall.features.standardization import StandardizationStats, excluded_dimension_mask
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_names
from gatefall.runs import default_run_dir_for_arm
from gatefall.sam3.descriptors import CHANNEL_NAMES, V_T_DIM
from gatefall.train.baseline_c0.config import C0_FUSION_CONFIG
from gatefall.train.baseline_c1.artifacts import (
    load_compatible_c1_checkpoint,
    validate_c1_training_run,
)
from gatefall.train.baseline_c1.config import C1_ADAPTIVE_GATE_CONFIG
from gatefall.train.baseline_c1.model import C1AdaptiveGateClassifier
from gatefall.train.baseline_c1.run import comparison_run_dirs
from gatefall.train.baseline_b1.model import B1AdaptiveGateClassifier


def _check(name: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return condition


def _raises(callback, expected: type[Exception] = ValueError) -> bool:
    try:
        callback()
    except expected:
        return True
    return False


def _pose_stats() -> StandardizationStats:
    names = feature_names()
    return StandardizationStats(
        "pose", "train", 10.0, 24, 4, 1, POSE_FEATURE_DIM, names,
        excluded_dimension_mask(names).tolist(), [1.0] * POSE_FEATURE_DIM,
        [2.0] * POSE_FEATURE_DIM, 0, [False] * POSE_FEATURE_DIM, "synthetic",
    )


def _visual_stats() -> Sam3StandardizationStats:
    return Sam3StandardizationStats(
        "sam3", "le2i", "train", 10.0, 24, 4, 1, V_T_DIM,
        list(CHANNEL_NAMES), [4.0] * V_T_DIM, [3.0] * V_T_DIM,
        0, [False] * V_T_DIM, "synthetic", "synthetic",
    )


class _RecordingModel:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, int, int]] = []
        self.qualities: list[np.ndarray] = []
        self.pose_means: list[float] = []
        self.visual_means: list[float] = []

    def to(self, _device: str) -> "_RecordingModel":
        return self

    def eval(self) -> "_RecordingModel":
        return self

    def __call__(
        self, pose: torch.Tensor, visual: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        self.shapes.append((pose.shape[-1], visual.shape[-1], quality.shape[-1]))
        self.qualities.append(quality.cpu().numpy())
        self.pose_means.append(float(pose.mean()))
        self.visual_means.append(float(visual.mean()))
        logits = torch.zeros((len(pose), C1_ADAPTIVE_GATE_CONFIG.num_classes))
        logits[:, 0] = 1.0
        return logits


def _frames() -> pd.DataFrame:
    tables = []
    for split in ("val", "test"):
        labels = np.zeros(26, dtype=np.int8)
        labels[0] = IGNORE_LABEL
        labels[10:13] = 1
        tables.append(pd.DataFrame({
            "video_id": [f"{split}-video"] * 26,
            "split": [split] * 26,
            "env": ["room"] * 26,
            "subject": [split] * 26,
            "frame_index": np.arange(26),
            "label": labels,
        }))
    return pd.concat(tables, ignore_index=True)


def check_prediction_inputs() -> bool:
    class _Source:
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int):
            quality = np.tile(np.array([0.25, 0.75], dtype=np.float32), (24, 1))
            return (
                np.full((24, POSE_FEATURE_DIM), 5.0, dtype=np.float32),
                np.full((24, V_T_DIM), 10.0, dtype=np.float32),
                quality, index, ("video", index),
            )

    model = _RecordingModel()
    identity = c1_events._predict_with_identity(
        model, _Source(), _pose_stats(), _visual_stats(), "cpu", 2
    )
    return _check(
        "inferência C1: V_t de 10 canais, qualidade crua e identidade no batch parcial",
        model.shapes == [(POSE_FEATURE_DIM, V_T_DIM, 2)] * 2
        and model.pose_means == [2.0, 2.0]
        and model.visual_means == [2.0, 2.0]
        and all(np.all(values[:, :, 0] == 0.25) and
                np.all(values[:, :, 1] == 0.75) for values in model.qualities)
        and identity == (["video"] * 3, [0, 1, 2], [0, 1, 2], [0, 0, 0]),
    )


def check_source_and_run_validation() -> bool:
    adapter = SimpleNamespace(pose_root=Path("pose"), sam3_root=Path("sam3"))
    config = replace(C1_ADAPTIVE_GATE_CONFIG, seed=17)
    inputs = SimpleNamespace(frames=_frames(), pose_stats=_pose_stats(),
                             visual_stats=_visual_stats())
    model = C1AdaptiveGateClassifier([8, 8], dilations=[1, 2])
    run_dir = Path("synthetic-c1")
    with (patch.object(c1_events, "get_dataset", return_value=adapter),
          patch.object(c1_events, "ensure_sam3_dataset_supported") as support,
          patch.object(c1_events, "_validated_inputs", return_value=inputs) as validated,
          patch.object(c1_events, "_resolve_config", return_value=config) as resolved,
          patch.object(c1_events, "validate_c1_training_run", return_value=config) as run_check,
          patch.object(c1_events, "load_compatible_c1_checkpoint", return_value=model) as checkpoint):
        assets = c1_events._load_run_assets("le2i", run_dir)
    expected = run_check.call_args.kwargs["expected_config"]
    accepted = (assets == (adapter, config, inputs.pose_stats, inputs.visual_stats,
                           model, inputs.frames)
                and support.called and validated.called and resolved.called
                and expected is config
                and run_check.call_args.kwargs["fields_allowed_to_differ"]
                == frozenset({"seed", "trainable_param_count"})
                and checkpoint.call_args.args == (run_dir / "checkpoint.pt", config))
    bad_config = replace(config, eval_stride=2)
    with (patch.object(c1_events, "get_dataset", return_value=adapter),
          patch.object(c1_events, "ensure_sam3_dataset_supported"),
          patch.object(c1_events, "_validated_inputs", return_value=inputs),
          patch.object(c1_events, "_resolve_config", return_value=config),
          patch.object(c1_events, "validate_c1_training_run", return_value=bad_config)):
        rejects_stride = _raises(lambda: c1_events._load_run_assets("le2i", run_dir))
    return _check("C1 revalida fontes, config, métricas, checkpoint e eval_stride",
                  accepted and rejects_stride)


def check_real_run_and_checkpoint_guards() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        torch.save(B1AdaptiveGateClassifier([8, 8], dilations=[1, 2]).state_dict(),
                   run_dir / "checkpoint.pt")
        rejects_b1_checkpoint = _raises(
            lambda: load_compatible_c1_checkpoint(
                run_dir / "checkpoint.pt", C1_ADAPTIVE_GATE_CONFIG
            )
        )
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(C0_FUSION_CONFIG.to_dict()), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        rejects_foreign_run = _raises(
            lambda: validate_c1_training_run(run_dir), RuntimeError
        )
    return _check("checkpoint B1 e run C0 recusados pela validação C1 real",
                  rejects_b1_checkpoint and rejects_foreign_run)


def check_quality_formula_and_alignment() -> bool:
    adapter = SimpleNamespace(pose_root=Path("pose"), sam3_root=Path("sam3"))
    visual = np.zeros((26, V_T_DIM), dtype=np.float32)
    visual[:, 0] = np.arange(26) % 2
    q_pose = np.arange(26, dtype=np.float32) / 26
    score = np.full(26, 0.8, dtype=np.float32)
    with (patch.object(c1_events, "compute_pose_quality", return_value=SimpleNamespace(q_pose=q_pose)),
          patch.object(c1_events, "load_v_t", return_value=visual),
          patch.object(c1_events, "read_sam_score", return_value=score)):
        quality = c1_events._quality_for_video("val-video", cast(DatasetAdapter, adapter))
    from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
    frames = _frames()
    pose = np.tile(np.arange(26, dtype=np.float32)[:, None], (1, POSE_FEATURE_DIM))
    source = GatedFusionWindowDataset(
        frames, "val", 1, lambda _: pose, lambda _: visual, lambda _: quality,
        drop_ignored=False, visual_dim=V_T_DIM,
    )
    first_pose, first_visual, first_quality, first_label, first_identity = source[0]
    last_pose, last_visual, last_quality, _, last_identity = source[25]
    bad_dim = _raises(lambda: GatedFusionWindowDataset(
        frames, "val", 1, lambda _: pose,
        lambda _: np.zeros((26, 1536), dtype=np.float32), lambda _: quality,
        drop_ignored=False, visual_dim=V_T_DIM,
    ))
    return _check(
        "qualidade q_pose/q_sam3 e janelas alinhadas, inclusive IGNORE_LABEL e borda",
        quality.shape == (26, 2)
        and np.array_equal(quality[:, 0], q_pose)
        and np.array_equal(quality[:, 1], score * visual[:, 0])
        and len(source) == 26 and first_label == IGNORE_LABEL
        and first_identity == ("val-video", 0)
        and last_identity == ("val-video", 25)
        and np.array_equal(first_pose[:, 0], np.zeros(24))
        and np.array_equal(first_visual[:, 0], np.zeros(24))
        and np.array_equal(first_quality[:, 0], np.zeros(24))
        and np.array_equal(last_pose[:, 0], np.arange(2, 26))
        and np.array_equal(last_visual[:, 0], visual[2:26, 0])
        and np.array_equal(last_quality[:, 0], q_pose[2:26])
        and bad_dim,
    )


def check_evaluation_and_artifacts() -> bool:
    frames = _frames()
    adapter = SimpleNamespace(pose_root=Path("pose"), sam3_root=Path("sam3"))
    model = _RecordingModel()
    pose = np.full((26, POSE_FEATURE_DIM), 5.0, dtype=np.float32)
    visual = np.full((26, V_T_DIM), 10.0, dtype=np.float32)
    quality = np.tile(np.array([0.25, 0.75], dtype=np.float32), (26, 1))
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "c1"
        run_dir.mkdir()
        (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        with (patch.object(c1_events, "_load_run_assets", return_value=(
                  adapter, C1_ADAPTIVE_GATE_CONFIG, _pose_stats(), _visual_stats(),
                  model, frames)),
              patch.object(c1_events, "build_pose_features", return_value=(pose, None)),
              patch.object(c1_events, "load_v_t", return_value=visual),
              patch.object(c1_events, "_quality_for_video", return_value=quality)):
            c1_events.run_evaluate(False, run_dir=run_dir)
            protocol_path = run_dir / "alarm_protocol.yaml"
            metrics_path = run_dir / "event_metrics.json"
            first_bytes = metrics_path.read_bytes()
            report = json.loads(first_bytes)
            c1_events.run_evaluate(False, run_dir=run_dir)
            skipped = metrics_path.read_bytes() == first_bytes
            protocol_path.write_text("invalid: true", encoding="utf-8")
            rejects_corrupt = _raises(lambda: c1_events.run_evaluate(False, run_dir=run_dir), RuntimeError)
            c1_events.run_evaluate(True, run_dir=run_dir)
            recovered_bytes = metrics_path.read_bytes()

            def fail_after_protocol(step: str) -> None:
                if step == "publish_protocol":
                    raise RuntimeError("falha sintética")

            def failed_promotion(*args, **kwargs) -> None:
                _promote_event_outputs(*args, **kwargs, after_step=fail_after_protocol)

            with patch.object(c1_events, "_promote_event_outputs", side_effect=failed_promotion):
                rejects_failed_promotion = _raises(
                    lambda: c1_events.run_evaluate(True, run_dir=run_dir), RuntimeError
                )
            restored = metrics_path.read_bytes() == recovered_bytes
            c1_events.run_evaluate(False, run_dir=run_dir)
        repaired = load_alarm_protocol(protocol_path) == BASELINE_A_ALARM_PROTOCOL
        valid = True
        try:
            validate_event_metrics(report, C1_ADAPTIVE_GATE_CONFIG,
                                   run_dir / "checkpoint.pt", protocol_path,
                                   training_metrics_path=run_dir / "metrics.json",
                                   require_hashes=True)
        except ValueError:
            valid = False
        splits = report["splits"]
        complete = all(splits[name]["usable_windows"] == 26
                       and splits[name]["total_windows"] == 26
                       and splits[name]["labeled_windows"] == 25
                       and splits[name]["n_fall_events"] == 1
                       for name in ("val", "test"))
        clean = not (run_dir / EVENT_TRANSACTION_FILE).exists() and not list(run_dir.glob(".*.pending-*"))
        quality_raw = all(np.all(batch[:, :, 0] == 0.25) and
                          np.all(batch[:, :, 1] == 0.75) for batch in model.qualities)
        hashes = report["checkpoint_sha256"] == sha256_file(run_dir / "checkpoint.pt")
    return _check("grade val/test completa, protocolo congelado, integridade e --force",
                  complete and repaired and skipped and rejects_corrupt and valid
                  and rejects_failed_promotion and restored
                  and clean and quality_raw and hashes
                  and all(shape == (POSE_FEATURE_DIM, V_T_DIM, 2) for shape in model.shapes))


def check_isolation() -> bool:
    default = default_run_dir_for_arm("le2i", "c1_adaptive_gate")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        foreign = root / "foreign"
        foreign.mkdir()
        (foreign / "config.yaml").write_text(yaml.safe_dump(C0_FUSION_CONFIG.to_dict()), encoding="utf-8")
        foreign_rejected = _raises(lambda: c1_events.run_evaluate(False, run_dir=foreign))
        no_lock = not (foreign / ".event-evaluation.lock").exists()
    comparisons = all(
        _raises(lambda path=path: c1_events.run_evaluate(False, run_dir=path))
        for path in comparison_run_dirs("le2i").values()
    )
    return _check("run C1 padrão e isolamento de armas, referência e protocolo",
                  c1_events.ARM_NAME == default.name and foreign_rejected and no_lock
                  and comparisons
                  and _raises(lambda: c1_events.run_evaluate(False, run_dir=Path("runs/reference/le2i/c1")))
                  and _raises(lambda: c1_events.run_evaluate(False, run_dir=Path("runs/local/le2i_cv/c1")))
                  and _raises(lambda: c1_events.run_evaluate(False, dataset_name="le2i-cv")))


def run_c1_events_selftest() -> bool:
    checks = [check_prediction_inputs(), check_source_and_run_validation(),
              check_real_run_and_checkpoint_guards(),
              check_quality_formula_and_alignment(), check_evaluation_and_artifacts(),
              check_isolation()]
    ok = all(checks)
    print("c1 events selftest OK" if ok else "c1 events selftest FALHOU")
    return ok
