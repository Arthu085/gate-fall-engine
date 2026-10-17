"""Contrato fixo, identidade e isolamento da investigação RGB temporal."""

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.hashing import sha256_file
from gatefall.runs import REPOSITORY_ROOT
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG
from gatefall.train.shared.determinism import configure_determinism
from gatefall.train.shared.metrics import RESTRICTED_CLASSES

OUTPUT_ROOT = REPOSITORY_ROOT / "runs/local/temporal_rgb"
WEIGHT_FILENAME = "r3d_18-b3b3357e.pth"
FEATURE_DIM = 512
CLIP_FRAMES = 16
SEED = 42
MODES = ("dynamic", "static")
OFFSETS = np.linspace(0, WINDOW_FRAMES - 1, CLIP_FRAMES).astype(np.int64)
PILOT_CRITERIA = {
    "mean_delta_vs_a_min": 0.030,
    "positive_folds_vs_a_min": 2,
    "mean_delta_vs_static_min": 0.010,
    "positive_folds_vs_static_min": 2,
    "fall_fallen_recall_delta_min_each_fold": -0.050,
    "other_walk_fp_delta_max_each_fold": 0,
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_output(root: Path) -> Path:
    if OUTPUT_ROOT.resolve() != OUTPUT_ROOT:
        raise ValueError("a raiz experimental não pode atravessar links simbólicos")
    resolved = root.resolve()
    if resolved != OUTPUT_ROOT.resolve():
        raise ValueError(f"a saída deve ser exatamente {OUTPUT_ROOT}; links para outros destinos são recusados")
    return resolved


def validate_persistent_checkout() -> None:
    git_dir = REPOSITORY_ROOT / ".git"
    if not git_dir.is_dir():
        raise RuntimeError("extração e treino reais exigem o checkout persistente, fora de worktree vinculado")


def guard_destination(path: Path) -> None:
    validate_output(OUTPUT_ROOT)
    if not path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"destino escapa da investigação: {path}")


def implementation_sha256() -> str:
    return digest({path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))})


def setup(device: str) -> dict[str, Any]:
    configure_determinism(SEED)
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"dispositivo indisponível: {device}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": SEED, "device": device, "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "dtype": "float32", "autocast": False, "tf32": False,
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
        "lock_sha256": sha256_file(REPOSITORY_ROOT / "uv.lock"),
        "implementation_sha256": implementation_sha256(),
    }


def contract() -> dict[str, Any]:
    config = BASELINE_A_CONFIG
    return {
        "schema_version": 1, "canonical": False, "official_splits_read": ["train"],
        "model": "torchvision.r3d_18", "weights": "R3D_18_Weights.KINETICS400_V1",
        "feature_dim": FEATURE_DIM, "window_frames": WINDOW_FRAMES,
        "clip_frames": CLIP_FRAMES, "sample_offsets": OFFSETS.tolist(),
        "train_stride": TRAIN_STRIDE, "eval_stride": EVAL_STRIDE,
        "alignment": "existing_src_index_clamped_to_floor_endpoint_time_times_source_fps",
        "padding": "left_edge", "static_control": "repeat_final_causal_frame_16_times",
        "preprocessing": "R3D_18_Weights.KINETICS400_V1.transforms()",
        "restricted_classes": RESTRICTED_CLASSES, "seed": SEED,
        "head": "standardize_development_only_then_linear_512_to_10",
        "epochs": config.epochs, "batch_size": config.batch_size,
        "lr": config.lr, "weight_decay": config.weight_decay,
        "grad_clip_norm": config.grad_clip_norm, "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR", "loss": "inverse_frequency_weighted_cross_entropy",
        "checkpoint_policy": "last_epoch", "pilot_criteria": PILOT_CRITERIA,
    }
