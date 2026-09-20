"""Selftest sintético do loop de treino da arma B1 (`b1_engine.py`). Não toca
em dados reais — um fixture minúsculo de janelas pose+DINOv3+qualidade é
treinado por 1-2 épocas em um run_dir temporário, só para travar que os três
artefatos obrigatórios saem gravados e íntegros, que a perda é finita e que
dois treinos com a mesma seed produzem o mesmo checkpoint."""

import json
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.features.dinov3_standardization import Dinov3StandardizationStats
from gatefall.features.quality_storage import QUALITY_CHANNELS
from gatefall.features.standardization import StandardizationStats, excluded_dimension_mask
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_names
from gatefall.train.b1_config import B1_ADAPTIVE_GATE_CONFIG, B1TrainConfig
from gatefall.train.b1_engine import run_b1_training

_VISUAL_DIM = 1536

_VIDEO_SPECS = {
    "train_vid": ("train", 40),
    "val_vid": ("val", 30),
    "test_vid": ("test", 30),
}


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


def _identity_pose_stats() -> StandardizationStats:
    names = feature_names()
    return StandardizationStats(
        source="pose",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1,
        feature_dim=POSE_FEATURE_DIM,
        feature_names=names,
        excluded_mask=excluded_dimension_mask(names).tolist(),
        mean=[0.0] * POSE_FEATURE_DIM,
        std=[1.0] * POSE_FEATURE_DIM,
        guarded_count=0,
        guarded_mask=[False] * POSE_FEATURE_DIM,
        frames_hash="deadbeef",
    )


def _identity_visual_stats() -> Dinov3StandardizationStats:
    return Dinov3StandardizationStats(
        source="dinov3",
        dataset="le2i",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1,
        feature_dim=_VISUAL_DIM,
        mean=[0.0] * _VISUAL_DIM,
        std=[1.0] * _VISUAL_DIM,
        guarded_count=0,
        guarded_mask=[False] * _VISUAL_DIM,
        frames_hash="cafebabe",
    )


def _make_sources(
    frames: pd.DataFrame, seed: int
) -> tuple[
    GatedFusionWindowDataset, GatedFusionWindowDataset, GatedFusionWindowDataset
]:
    rng = np.random.default_rng(seed)
    pose_by_video = {
        video_id: rng.normal(size=(n_frames, POSE_FEATURE_DIM)).astype(np.float32)
        for video_id, (_split, n_frames) in _VIDEO_SPECS.items()
    }
    visual_by_video = {
        video_id: rng.normal(size=(n_frames, _VISUAL_DIM)).astype(np.float32)
        for video_id, (_split, n_frames) in _VIDEO_SPECS.items()
    }
    quality_by_video = {
        video_id: rng.random((n_frames, QUALITY_CHANNELS)).astype(np.float32)
        for video_id, (_split, n_frames) in _VIDEO_SPECS.items()
    }

    def make(split: str, stride: int) -> GatedFusionWindowDataset:
        return GatedFusionWindowDataset(
            frames,
            split=split,
            stride=stride,
            pose_loader=lambda video_id: pose_by_video[video_id],
            visual_loader=lambda video_id: visual_by_video[video_id],
            quality_loader=lambda video_id: quality_by_video[video_id],
        )

    return (
        make("train", TRAIN_STRIDE),
        make("val", EVAL_STRIDE),
        make("test", EVAL_STRIDE),
    )


def _tiny_config(run_name: str, epochs: int) -> B1TrainConfig:
    return replace(
        B1_ADAPTIVE_GATE_CONFIG,
        run_name=run_name,
        epochs=epochs,
        batch_size=4,
        channels=[8, 8],
        dilations=[1, 2],
        kernel_size=3,
        seed=0,
    )


def check_training_run_writes_valid_artifacts() -> bool:
    frames = _make_frames(_VIDEO_SPECS)
    train_source, val_source, test_source = _make_sources(frames, seed=0)
    config = _tiny_config("b1_adaptive_gate_synthetic_selftest", epochs=2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir) / "run"
        metrics = run_b1_training(
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

        artifacts_present = all(
            (run_dir / name).is_file()
            for name in ("config.yaml", "metrics.json", "checkpoint.pt")
        )
        with (run_dir / "metrics.json").open(encoding="utf-8") as stream:
            metrics_on_disk = json.load(stream)
        losses = [entry["train_loss"] for entry in metrics_on_disk["history"]]
        losses_finite = all(
            isinstance(loss, (int, float)) and math.isfinite(loss) for loss in losses
        )
        epochs_match = len(losses) == config.epochs
        returned_consistent = (
            metrics is not None and metrics.get("run_name") == config.run_name
        )

    ok = artifacts_present and losses_finite and epochs_match and returned_consistent
    return _check(
        "run_b1_training grava config.yaml/metrics.json/checkpoint.pt em "
        "run_dir, com history de tamanho epochs e train_loss finito em cada "
        "época, sobre um fixture sintético de pose+DINOv3+qualidade",
        ok,
    )


def check_same_seed_produces_identical_checkpoint() -> bool:
    frames = _make_frames(_VIDEO_SPECS)
    config = _tiny_config("b1_adaptive_gate_synthetic_determinism", epochs=1)

    digests: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for attempt in range(2):
            run_dir = Path(tmp_dir) / f"run_{attempt}"
            train_source, val_source, test_source = _make_sources(frames, seed=0)
            run_b1_training(
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
            digests.append(sha256_file(run_dir / "checkpoint.pt"))

            state = torch.load(
                run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            gate_present = {"gate.linear.weight", "gate.linear.bias"} <= set(state)

    return _check(
        "dois treinos B1 com a mesma seed produzem checkpoints idênticos "
        "(sha256) e o checkpoint carrega os pesos do gate",
        digests[0] == digests[1] and gate_present,
    )


def check_second_invocation_without_force_skips_valid_run() -> bool:
    frames = _make_frames(_VIDEO_SPECS)
    config = _tiny_config("b1_adaptive_gate_synthetic_idempotent", epochs=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir) / "run"
        train_source, val_source, test_source = _make_sources(frames, seed=1)
        run_b1_training(
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
        config_unchanged = config.trainable_param_count == 0

        train_source, val_source, test_source = _make_sources(frames, seed=1)
        raised: Exception | None = None
        try:
            run_b1_training(
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

    return _check(
        "segunda invocação de run_b1_training sem --force sobre um run já "
        "completo e íntegro não levanta RuntimeError, mesmo recebendo o mesmo "
        "objeto de config original (trainable_param_count=0)",
        config_unchanged and raised is None,
    )


def check_undecodable_config_reaches_the_run_validators() -> bool:
    from gatefall.train.b1_engine import guard_not_foreign_arm_run_dir

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir)
        # config.yaml binário: `open(encoding="utf-8")` levanta UnicodeDecodeError,
        # que é ValueError e não YAMLError. A guarda precisa devolver o controle
        # aos validadores do run, que emitem a mensagem acionável.
        (run_dir / "config.yaml").write_bytes(b"\xff\xfe\x00arm: B0")
        passed_through = True
        try:
            guard_not_foreign_arm_run_dir(run_dir, "B1")
        except Exception:
            passed_through = False

    return _check(
        "guard_not_foreign_arm_run_dir não escapa com UnicodeDecodeError num "
        "config.yaml não-UTF-8: devolve o controle aos validadores do run",
        passed_through,
    )


def check_force_refuses_run_dir_of_another_arm() -> bool:
    frames = _make_frames(_VIDEO_SPECS)
    config = _tiny_config("b1_adaptive_gate_synthetic_foreign_arm", epochs=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # run não canônico da arma B0 (ex.: b0_fusion_seed7): as guardas de CLI
        # só cobrem os run_dirs canônicos.
        run_dir = Path(tmp_dir) / "b0_fusion_seed7"
        run_dir.mkdir()
        config_path = run_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump({"run_name": "b0_fusion", "arm": "B0"}, stream)
        digest_before = sha256_file(config_path)

        train_source, val_source, test_source = _make_sources(frames, seed=2)
        raised: Exception | None = None
        try:
            run_b1_training(
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
        except RuntimeError as exc:
            raised = exc

        preserved = (
            config_path.is_file() and sha256_file(config_path) == digest_before
        )

    return _check(
        "run_b1_training recusa com --force um run_dir não canônico cujo "
        "config.yaml declara outra arma, preservando o run intacto",
        raised is not None and "B0" in str(raised) and preserved,
    )


def run_b1_engine_selftest() -> bool:
    checks = [
        check_training_run_writes_valid_artifacts(),
        check_same_seed_produces_identical_checkpoint(),
        check_second_invocation_without_force_skips_valid_run(),
        check_force_refuses_run_dir_of_another_arm(),
        check_undecodable_config_reaches_the_run_validators(),
    ]
    ok = all(checks)
    if not ok:
        print("\nb1 engine selftest FALHOU", file=sys.stderr)
    else:
        print("\nb1 engine selftest OK: todas as checagens passaram")
    return ok
