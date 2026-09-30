"""Checagens sintéticas da avaliação de eventos da arma C0."""

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

import gatefall.eval.baseline_c0.cli as c0_events
from gatefall.config import IGNORE_LABEL
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, load_alarm_protocol
from gatefall.eval.shared.event_artifacts import validate_event_metrics
from gatefall.features.sam3_standardization import Sam3StandardizationStats
from gatefall.features.standardization import StandardizationStats, excluded_dimension_mask
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_names
from gatefall.runs import default_run_dir_for_arm
from gatefall.sam3.descriptors import CHANNEL_NAMES, V_T_DIM
from gatefall.train.baseline_c0.artifacts import load_compatible_c0_checkpoint, validate_c0_training_run
from gatefall.train.baseline_c0.config import C0_FUSION_CONFIG
from gatefall.train.baseline_c1.config import C1_ADAPTIVE_GATE_CONFIG
from gatefall.train.baseline_c1.model import C1AdaptiveGateClassifier


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
        self.shapes: list[tuple[int, int]] = []
        self.pose_means: list[float] = []
        self.visual_means: list[float] = []
        self.batch_sizes: list[int] = []

    def to(self, _device: str) -> "_RecordingModel":
        return self

    def eval(self) -> "_RecordingModel":
        return self

    def __call__(self, pose: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        self.shapes.append((pose.shape[-1], visual.shape[-1]))
        self.pose_means.append(float(pose.mean()))
        self.visual_means.append(float(visual.mean()))
        self.batch_sizes.append(len(pose))
        logits = torch.zeros((len(pose), C0_FUSION_CONFIG.num_classes))
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
            return (
                np.full((24, POSE_FEATURE_DIM), 5.0, dtype=np.float32),
                np.full((24, V_T_DIM), 10.0, dtype=np.float32),
                index, ("video", 23 + index),
            )

    model = _RecordingModel()
    identity = c0_events._predict_with_identity(
        model, _Source(), _pose_stats(), _visual_stats(), "cpu", 2
    )
    return _check(
        "inferência C0: padronização, V_t e identidade no batch parcial",
        model.shapes == [(POSE_FEATURE_DIM, V_T_DIM)] * 2
        and model.pose_means == [2.0, 2.0]
        and model.visual_means == [2.0, 2.0]
        and model.batch_sizes == [2, 1]
        and identity == (["video"] * 3, [23, 24, 25], [0, 1, 2], [0, 0, 0]),
    )


def check_run_validation() -> bool:
    adapter = SimpleNamespace(pose_root=Path("pose"), sam3_root=Path("sam3"))
    inputs = SimpleNamespace(frames=_frames(), pose_stats=_pose_stats(), visual_stats=_visual_stats())
    config = replace(C0_FUSION_CONFIG, seed=17)
    model = _RecordingModel()
    run_dir = Path("synthetic-c0")
    with (patch.object(c0_events, "get_dataset", return_value=adapter),
          patch.object(c0_events, "ensure_sam3_dataset_supported") as support,
          patch.object(c0_events, "_validated_inputs", return_value=inputs) as validated,
          patch.object(c0_events, "resolve_c0_config", return_value=config),
          patch.object(c0_events, "validate_c0_training_run", return_value=config) as run_check,
          patch.object(c0_events, "load_compatible_c0_checkpoint", return_value=model) as checkpoint):
        assets = c0_events._load_run_assets("le2i", run_dir)
    accepted = (
        assets == (adapter, config, inputs.pose_stats, inputs.visual_stats, model, inputs.frames)
        and support.called and validated.called
        and run_check.call_args.kwargs["expected_config"] is config
        and run_check.call_args.kwargs["fields_allowed_to_differ"]
        == frozenset({"seed", "trainable_param_count"})
        and checkpoint.call_args.args == (run_dir / "checkpoint.pt", config)
    )
    with (patch.object(c0_events, "get_dataset", return_value=adapter),
          patch.object(c0_events, "ensure_sam3_dataset_supported"),
          patch.object(c0_events, "_validated_inputs", return_value=inputs),
          patch.object(c0_events, "resolve_c0_config", return_value=config),
          patch.object(c0_events, "validate_c0_training_run", return_value=replace(config, eval_stride=2))):
        rejects_stride = _raises(lambda: c0_events._load_run_assets("le2i", run_dir))
    return _check("run C0: fontes, config, checkpoint e eval_stride validados", accepted and rejects_stride)


def check_foreign_checkpoint_and_run() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        torch.save(
            C1AdaptiveGateClassifier([8, 8], dilations=[1, 2]).state_dict(),
            run_dir / "checkpoint.pt",
        )
        rejects_checkpoint = _raises(
            lambda: load_compatible_c0_checkpoint(run_dir / "checkpoint.pt", C0_FUSION_CONFIG)
        )
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(C1_ADAPTIVE_GATE_CONFIG.to_dict()), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        rejects_run = _raises(lambda: validate_c0_training_run(run_dir), RuntimeError)
    return _check("checkpoint e run C1 recusados pelo validador C0", rejects_checkpoint and rejects_run)


def check_evaluation_and_artifacts() -> bool:
    from gatefall.sam3 import extract as sam3_extract
    from gatefall.train.baseline_c0 import cli as c0_train

    adapter = SimpleNamespace(pose_root=Path("pose"), sam3_root=Path("sam3"))
    model = _RecordingModel()
    pose = np.full((26, POSE_FEATURE_DIM), 5.0, dtype=np.float32)
    visual = np.full((26, V_T_DIM), 10.0, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "c0"
        run_dir.mkdir()
        (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        with (patch.object(c0_events, "_load_run_assets", return_value=(
                  adapter, C0_FUSION_CONFIG, _pose_stats(), _visual_stats(), model, _frames())),
              patch.object(c0_events, "build_pose_features", return_value=(pose, None)) as pose_loader,
              patch.object(c0_events, "load_v_t", return_value=visual) as visual_loader,
              patch.object(c0_train, "run_train", side_effect=AssertionError("treino invocado")) as train,
              patch.object(sam3_extract, "run_sam3_extract_all", side_effect=AssertionError("extração invocada")) as extract):
            c0_events.run_evaluate(False, run_dir=run_dir)
            protocol_path = run_dir / "alarm_protocol.yaml"
            metrics_path = run_dir / "event_metrics.json"
            first_bytes = metrics_path.read_bytes()
            report = json.loads(first_bytes)
            c0_events.run_evaluate(False, run_dir=run_dir)
            skipped = metrics_path.read_bytes() == first_bytes
            protocol_path.write_text("invalid: true", encoding="utf-8")
            rejects_corrupt = _raises(lambda: c0_events.run_evaluate(False, run_dir=run_dir), RuntimeError)
            c0_events.run_evaluate(True, run_dir=run_dir)
            no_training_or_extraction = train.call_count == 0 and extract.call_count == 0
        valid = True
        try:
            validate_event_metrics(
                report, C0_FUSION_CONFIG, run_dir / "checkpoint.pt", protocol_path,
                training_metrics_path=run_dir / "metrics.json", require_hashes=True,
            )
        except ValueError:
            valid = False
        splits = report["splits"]
        complete = all(
            splits[name]["usable_windows"] == 26
            and splits[name]["total_windows"] == 26
            and splits[name]["labeled_windows"] == 25
            and splits[name]["n_fall_events"] == 1
            for name in ("val", "test")
        )
        sources_read = pose_loader.call_count == 4 and visual_loader.call_count == 4
        return _check(
            "grade completa, protocolo congelado, artefatos íntegros e --force",
            complete and skipped and rejects_corrupt and valid and sources_read
            and no_training_or_extraction
            and load_alarm_protocol(protocol_path) == BASELINE_A_ALARM_PROTOCOL
            and model.shapes == [(POSE_FEATURE_DIM, V_T_DIM)] * 4,
        )


def check_isolation() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        foreign = Path(tmp)
        (foreign / "config.yaml").write_text(
            yaml.safe_dump(C1_ADAPTIVE_GATE_CONFIG.to_dict()), encoding="utf-8"
        )
        foreign_rejected = _raises(lambda: c0_events.run_evaluate(False, run_dir=foreign))
        no_lock = not (foreign / ".event-evaluation.lock").exists()
    comparisons = all(
        _raises(lambda arm=arm: c0_events.run_evaluate(
            False, run_dir=default_run_dir_for_arm("le2i", arm)
        ))
        for arm in ("A", "B0", "B1", "C1")
    )
    return _check(
        "run C0: isolamento entre armas, referência e protocolos",
        foreign_rejected and no_lock and comparisons
        and _raises(lambda: c0_events.run_evaluate(False, run_dir=Path("runs/reference/le2i/baseline_c0")))
        and _raises(lambda: c0_events.run_evaluate(False, run_dir=Path("runs/local/le2i_cv/baseline_c0")))
        and default_run_dir_for_arm("le2i-cv", "C0") == Path("runs/local/le2i_cv/baseline_c0"),
    )


def run_c0_events_selftest() -> bool:
    checks = [check_prediction_inputs(), check_run_validation(),
              check_foreign_checkpoint_and_run(),
              check_evaluation_and_artifacts(), check_isolation()]
    ok = all(checks)
    print("c0 events selftest OK" if ok else "c0 events selftest FALHOU")
    return ok
