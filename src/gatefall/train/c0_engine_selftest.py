"""Selftest sintético do loop de treino da arma C0 (`c0_engine.py`). Não toca
em dados reais — um fixture minúsculo de janelas de pose+SAM 3 V_t sintéticas é
treinado por 1-2 épocas em um run_dir temporário, só para travar que os três
artefatos obrigatórios saem gravados e íntegros e que a perda é finita."""

import json
import math
import sys
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.fusion_dataset import FusionWindowDataset
from gatefall.features.sam3_standardization import Sam3StandardizationStats
from gatefall.features.standardization import StandardizationStats, excluded_dimension_mask
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_names
from gatefall.sam3.descriptors import CHANNEL_NAMES, V_T_DIM
from gatefall.train.c0_config import C0_FUSION_CONFIG
from gatefall.train.c0_engine import run_c0_training

_VISUAL_DIM = V_T_DIM


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _make_frames(video_specs: dict[str, tuple[str, int]]) -> pd.DataFrame:
    tables = []
    for i, (video_id, (split, n_frames)) in enumerate(video_specs.items()):
        tables.append(
            pd.DataFrame(
                {
                    "video_id": video_id,
                    "split": split,
                    "env": "coffee_room",
                    "subject": i + 1,
                    "frame_index": np.arange(n_frames, dtype=np.int32),
                    "label": (np.arange(n_frames, dtype=np.int8) % 2),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def _c0_source(
    frames: pd.DataFrame,
    split: str,
    stride: int,
    pose_loader: Callable[[str], np.ndarray],
    visual_loader: Callable[[str], np.ndarray],
) -> FusionWindowDataset:
    return FusionWindowDataset(
        frames, split, stride, pose_loader, visual_loader, visual_dim=_VISUAL_DIM
    )


def _identity_pose_stats() -> StandardizationStats:
    names = feature_names()
    excluded_mask = excluded_dimension_mask(names)
    return StandardizationStats(
        source="pose",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1,
        feature_dim=POSE_FEATURE_DIM,
        feature_names=names,
        excluded_mask=excluded_mask.tolist(),
        mean=[0.0] * POSE_FEATURE_DIM,
        std=[1.0] * POSE_FEATURE_DIM,
        guarded_count=0,
        guarded_mask=[False] * POSE_FEATURE_DIM,
        frames_hash="deadbeef",
    )


def _identity_visual_stats() -> Sam3StandardizationStats:
    return Sam3StandardizationStats(
        source="sam3",
        dataset="le2i",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1,
        feature_dim=_VISUAL_DIM,
        channel_names=list(CHANNEL_NAMES),
        mean=[0.0] * _VISUAL_DIM,
        std=[1.0] * _VISUAL_DIM,
        guarded_count=0,
        guarded_mask=[False] * _VISUAL_DIM,
        frames_hash="cafebabe",
        sam3_features_sha256="feedface",
    )


def check_training_run_writes_valid_artifacts() -> bool:
    video_specs = {
        "train_vid": ("train", 40),
        "val_vid": ("val", 30),
        "test_vid": ("test", 30),
    }
    frames = _make_frames(video_specs)
    rng = np.random.default_rng(0)

    def pose_loader(video_id: str) -> np.ndarray:
        _split, n_frames = video_specs[video_id]
        return rng.normal(size=(n_frames, POSE_FEATURE_DIM)).astype(np.float32)

    def visual_loader(video_id: str) -> np.ndarray:
        _split, n_frames = video_specs[video_id]
        return rng.normal(size=(n_frames, _VISUAL_DIM)).astype(np.float32)

    train_source = _c0_source(frames, "train", TRAIN_STRIDE, pose_loader, visual_loader)
    val_source = _c0_source(frames, "val", EVAL_STRIDE, pose_loader, visual_loader)
    test_source = _c0_source(frames, "test", EVAL_STRIDE, pose_loader, visual_loader)

    config = replace(
        C0_FUSION_CONFIG,
        run_name="c0_fusion_synthetic_selftest",
        epochs=2,
        batch_size=4,
        channels=[8, 8],
        dilations=[1, 2],
        kernel_size=3,
        seed=0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir) / "run"
        metrics = run_c0_training(
            train_source=train_source,
            val_source=val_source,
            test_source=test_source,
            pose_stats=_identity_pose_stats(),
            visual_stats=_identity_visual_stats(),
            config=config,
            run_dir=run_dir,
            force=True,
            label_names=tuple(f"class_{i}" for i in range(config.num_classes)),
        )

        artifacts_present = (
            (run_dir / "config.yaml").is_file()
            and (run_dir / "metrics.json").is_file()
            and (run_dir / "checkpoint.pt").is_file()
        )

        with (run_dir / "metrics.json").open(encoding="utf-8") as stream:
            metrics_on_disk = json.load(stream)

        losses = [entry["train_loss"] for entry in metrics_on_disk["history"]]
        losses_finite = all(isinstance(loss, (int, float)) and math.isfinite(loss) for loss in losses)
        epochs_match = len(losses) == config.epochs

        metrics_return_consistent = metrics is not None and metrics.get("run_name") == config.run_name

    ok = artifacts_present and losses_finite and epochs_match and metrics_return_consistent
    return _check(
        "run_c0_training grava config.yaml/metrics.json/checkpoint.pt em "
        "run_dir, com history de tamanho epochs e train_loss finito em "
        "cada época, sobre um fixture sintético minúsculo de pose+SAM 3 V_t",
        ok,
    )


def check_second_invocation_without_force_skips_valid_run() -> bool:
    video_specs = {
        "train_vid": ("train", 40),
        "val_vid": ("val", 30),
        "test_vid": ("test", 30),
    }
    frames = _make_frames(video_specs)
    rng = np.random.default_rng(1)

    def pose_loader(video_id: str) -> np.ndarray:
        _split, n_frames = video_specs[video_id]
        return rng.normal(size=(n_frames, POSE_FEATURE_DIM)).astype(np.float32)

    def visual_loader(video_id: str) -> np.ndarray:
        _split, n_frames = video_specs[video_id]
        return rng.normal(size=(n_frames, _VISUAL_DIM)).astype(np.float32)

    def _make_sources() -> tuple[FusionWindowDataset, FusionWindowDataset, FusionWindowDataset]:
        return (
            _c0_source(frames, "train", TRAIN_STRIDE, pose_loader, visual_loader),
            _c0_source(frames, "val", EVAL_STRIDE, pose_loader, visual_loader),
            _c0_source(frames, "test", EVAL_STRIDE, pose_loader, visual_loader),
        )

    config = replace(
        C0_FUSION_CONFIG,
        run_name="c0_fusion_synthetic_selftest_idempotent",
        epochs=1,
        batch_size=4,
        channels=[8, 8],
        dilations=[1, 2],
        kernel_size=3,
        seed=0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir) / "run"

        train_source, val_source, test_source = _make_sources()
        run_c0_training(
            train_source=train_source,
            val_source=val_source,
            test_source=test_source,
            pose_stats=_identity_pose_stats(),
            visual_stats=_identity_visual_stats(),
            config=config,
            run_dir=run_dir,
            force=True,
            label_names=tuple(f"class_{i}" for i in range(config.num_classes)),
        )

        # `run_c0_training` reatribui `config` a uma variável local via
        # `replace(...)`; o objeto do chamador não é mutado e continua com
        # trainable_param_count=0, igual a um config recém-resolvido.
        config_unchanged = config.trainable_param_count == 0

        train_source, val_source, test_source = _make_sources()
        raised: Exception | None = None
        try:
            run_c0_training(
                train_source=train_source,
                val_source=val_source,
                test_source=test_source,
                pose_stats=_identity_pose_stats(),
                visual_stats=_identity_visual_stats(),
                config=config,
                run_dir=run_dir,
                force=False,
                label_names=tuple(f"class_{i}" for i in range(config.num_classes)),
            )
        except RuntimeError as exc:
            raised = exc

    ok = config_unchanged and raised is None
    return _check(
        "segunda invocação de run_c0_training sem --force sobre um run já "
        "completo e íntegro não levanta RuntimeError, mesmo recebendo o "
        "mesmo objeto de config original (trainable_param_count=0, "
        "divergente do valor real persistido no run)",
        ok,
    )


def run_c0_engine_selftest() -> bool:
    checks = [
        check_training_run_writes_valid_artifacts(),
        check_second_invocation_without_force_skips_valid_run(),
    ]
    ok = all(checks)
    if not ok:
        print("\nc0 engine selftest FALHOU", file=sys.stderr)
    else:
        print("\nc0 engine selftest OK: todas as checagens passaram")
    return ok
