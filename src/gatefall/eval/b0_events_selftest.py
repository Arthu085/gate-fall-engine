"""Selftest sintético da avaliação de eventos da arma B0."""

import inspect
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from gatefall.config import TRAIN_STRIDE
from gatefall.eval.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.baseline_a_events import (
    EVENT_COUNT_FIELDS,
    EVENT_RATE_FIELDS,
    EVENT_SPLIT_FIELDS,
    _predict_with_identity as predict_arm_a_with_identity,
    validate_event_metrics,
)
from gatefall.features.dinov3_standardization import Dinov3StandardizationStats
from gatefall.features.standardization import (
    StandardizationStats,
    excluded_dimension_mask,
)
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_names
from gatefall.runs import default_run_dir, default_run_dir_for_arm
from gatefall.train.b0_config import B0_FUSION_CONFIG

_VISUAL_DIM = 1536


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _pose_stats(mean: float = 0.0, std: float = 1.0) -> StandardizationStats:
    names = feature_names()
    excluded = excluded_dimension_mask(names)
    return StandardizationStats(
        source="pose",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1,
        feature_dim=POSE_FEATURE_DIM,
        feature_names=names,
        excluded_mask=excluded.tolist(),
        mean=[mean] * POSE_FEATURE_DIM,
        std=[std] * POSE_FEATURE_DIM,
        guarded_count=0,
        guarded_mask=[False] * POSE_FEATURE_DIM,
        frames_hash="pose-frames",
    )


def _visual_stats(mean: float = 0.0, std: float = 1.0) -> Dinov3StandardizationStats:
    return Dinov3StandardizationStats(
        source="dinov3",
        dataset="le2i",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1,
        feature_dim=_VISUAL_DIM,
        mean=[mean] * _VISUAL_DIM,
        std=[std] * _VISUAL_DIM,
        guarded_count=0,
        guarded_mask=[False] * _VISUAL_DIM,
        frames_hash="visual-frames",
    )


class _FusionSource:
    def __init__(self) -> None:
        self.items = [
            (
                np.full((24, POSE_FEATURE_DIM), 5.0, dtype=np.float32),
                np.full((24, _VISUAL_DIM), 10.0, dtype=np.float32),
                index % 3,
                (f"video-{index // 3}", 23 + index),
            )
            for index in range(5)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, int, tuple[str, int]]:
        return self.items[index]


class _RecordingFusionModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.pose_means: list[float] = []
        self.visual_means: list[float] = []
        self.cursor = 0

    def __call__(
        self, pose: torch.Tensor, visual: torch.Tensor
    ) -> torch.Tensor:
        batch_size = pose.shape[0]
        self.batch_sizes.append(batch_size)
        self.pose_means.append(float(pose.mean()))
        self.visual_means.append(float(visual.mean()))
        logits = torch.full((batch_size, 7), -1.0)
        for offset in range(batch_size):
            logits[offset, self.cursor + offset] = 1.0
        self.cursor += batch_size
        return logits


class _PoseSource:
    def __init__(self) -> None:
        self.items = [
            (
                np.full((24, POSE_FEATURE_DIM), float(index), dtype=np.float32),
                index % 2,
                ("arm-a-video", 23 + index),
            )
            for index in range(3)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, tuple[str, int]]:
        return self.items[index]


class _RecordingPoseModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, pose: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(pose.shape[0])
        logits = torch.zeros((pose.shape[0], 7))
        logits[:, 1] = 1.0
        return logits


def check_b0_prediction_standardizes_both_inputs_and_preserves_identity() -> bool:
    from gatefall.eval.b0_events import _predict_with_identity

    source = _FusionSource()
    model = _RecordingFusionModel()
    video_ids, k_ends, true_labels, pred_labels = _predict_with_identity(
        model,
        source,
        _pose_stats(mean=1.0, std=2.0),
        _visual_stats(mean=4.0, std=3.0),
        device="cpu",
        batch_size=2,
    )
    return _check(
        "inferência B0: padroniza pose e DINOv3 separadamente, fornece as "
        "duas entradas ao modelo e preserva identidade/rótulo inclusive no "
        "batch final parcial",
        model.batch_sizes == [2, 2, 1]
        and np.allclose(model.pose_means, [2.0, 2.0, 2.0])
        and np.allclose(model.visual_means, [2.0, 2.0, 2.0])
        and video_ids == ["video-0", "video-0", "video-0", "video-1", "video-1"]
        and k_ends == [23, 24, 25, 26, 27]
        and true_labels == [0, 1, 2, 0, 1]
        and pred_labels == [0, 1, 2, 3, 4],
    )


def check_b0_defaults_to_own_run_and_rejects_arm_a() -> bool:
    import gatefall.eval.b0_events as b0_events

    captured: list[Path] = []

    class _Lock:
        def __init__(self, run_dir: Path) -> None:
            captured.append(run_dir)

        def __enter__(self) -> "_Lock":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    original_lock = b0_events.EventEvaluationLock
    original_locked = b0_events._run_evaluate_locked
    original_validate = b0_events.validate_local_run_dir
    setattr(b0_events, "EventEvaluationLock", cast(Any, _Lock))
    setattr(
        b0_events,
        "_run_evaluate_locked",
        cast(Any, lambda *_args, **_kwargs: None),
    )
    setattr(
        b0_events,
        "validate_local_run_dir",
        cast(Any, lambda *_args, **_kwargs: None),
    )
    try:
        b0_events.run_evaluate(force=False, dataset_name="le2i", run_dir=None)
        arm_a_rejected = False
        try:
            b0_events.run_evaluate(
                force=False,
                dataset_name="le2i",
                run_dir=default_run_dir("le2i"),
            )
        except ValueError:
            arm_a_rejected = True
    finally:
        setattr(b0_events, "EventEvaluationLock", original_lock)
        setattr(b0_events, "_run_evaluate_locked", original_locked)
        setattr(b0_events, "validate_local_run_dir", original_validate)

    default_is_optional = inspect.signature(b0_events.run_evaluate).parameters[
        "run_dir"
    ].default is None
    return _check(
        "run_evaluate B0: usa o run canônico b0_fusion por default e recusa "
        "explicitamente o run da arma A",
        default_is_optional
        and captured == [default_run_dir_for_arm("le2i", "b0_fusion")]
        and arm_a_rejected,
    )


def _empty_event_split() -> dict[str, object]:
    split: dict[str, object] = {}
    for field in EVENT_COUNT_FIELDS:
        split[field] = 0
    for field in EVENT_RATE_FIELDS:
        split[field] = 0.0
    split["latency_seconds"] = {"per_event": [], "mean": None, "median": None}
    assert set(split) == EVENT_SPLIT_FIELDS
    return split


def check_event_artifact_contract_accepts_b0_identity() -> bool:
    checkpoint_path = Path("runs/local/le2i/b0_fusion/checkpoint.pt")
    protocol_path = Path("runs/local/le2i/b0_fusion/alarm_protocol.yaml")
    report = {
        "run_name": B0_FUSION_CONFIG.run_name,
        "checkpoint_path": str(checkpoint_path),
        "alarm_protocol_path": str(protocol_path),
        "splits": {"val": _empty_event_split(), "test": _empty_event_split()},
    }
    accepted = True
    try:
        validate_event_metrics(
            report,
            cast(Any, B0_FUSION_CONFIG),
            checkpoint_path,
            protocol_path,
        )
    except ValueError:
        accepted = False
    return _check(
        "contrato de event_metrics.json compartilhado aceita run_name B0 sem "
        "alterar campos, protocolo ou semântica dos artefatos da arma A",
        accepted and BASELINE_A_ALARM_PROTOCOL.fall_label == 1,
    )


def check_arm_a_prediction_regression() -> bool:
    source = _PoseSource()
    model = _RecordingPoseModel()
    video_ids, k_ends, true_labels, pred_labels = predict_arm_a_with_identity(
        cast(Any, model),
        cast(Any, source),
        _pose_stats(),
        device="cpu",
        batch_size=2,
    )
    return _check(
        "regressão A: o avaliador original continua aceitando uma única entrada "
        "de pose e preserva seu contrato de identidade e batch parcial",
        model.batch_sizes == [2, 1]
        and video_ids == ["arm-a-video"] * 3
        and k_ends == [23, 24, 25]
        and true_labels == [0, 1, 0]
        and pred_labels == [1, 1, 1],
    )


def run_b0_events_selftest() -> bool:
    checks = [
        check_b0_prediction_standardizes_both_inputs_and_preserves_identity(),
        check_b0_defaults_to_own_run_and_rejects_arm_a(),
        check_event_artifact_contract_accepts_b0_identity(),
        check_arm_a_prediction_regression(),
    ]
    ok = all(checks)
    if not ok:
        print("\nb0 events selftest FALHOU", file=sys.stderr)
    else:
        print("\nb0 events selftest OK: todas as checagens passaram")
    return ok
