"""Selftest sintético da extração DINOv3 (pré-processamento, features, armazenamento).

Não toca em GPU, dataset real ou pesos do backbone — todas as entradas são
sintéticas.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from gatefall.dinov3.backbone import FEATURE_DIM, RESIZE_SIZE
from gatefall.dinov3.features import compute_features
from gatefall.dinov3.preprocessing import preprocess_frames
from gatefall.dinov3.storage import (
    Dinov3StorageError,
    verify_written_file,
    write_dinov3_features_atomic,
)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


class _FakeBackbone:
    def __init__(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> None:
        self._cls_token = cls_token
        self._patch_tokens = patch_tokens

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "x_norm_clstoken": self._cls_token,
            "x_norm_patchtokens": self._patch_tokens,
        }


def _check_preprocess_frames_shape_and_dtype() -> bool:
    ok = True
    for height, width in [(64, 64), (100, 200), (300, 150)]:
        frames = [
            np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        out = preprocess_frames(frames)
        ok = ok and (
            out.shape == (3, 3, RESIZE_SIZE, RESIZE_SIZE)
            and out.dtype == torch.float32
        )
    return _check(
        "preprocess_frames: shape [K,3,224,224] float32 para H,W variados", ok
    )


def _check_compute_features_formula() -> bool:
    batch_size = 2
    cls_token = torch.arange(batch_size * 768, dtype=torch.float32).reshape(
        batch_size, 768
    )
    patch_tokens = torch.zeros((batch_size, 4, 768), dtype=torch.float32)
    patch_tokens[0] = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(4, 1).expand(4, 768)
    patch_tokens[1] = torch.tensor([5.0, 6.0, 7.0, 8.0]).view(4, 1).expand(4, 768)
    expected_mean_patch = torch.tensor(
        [[2.5] * 768, [6.5] * 768], dtype=torch.float32
    )

    backbone = _FakeBackbone(cls_token, patch_tokens)
    features = compute_features(backbone, torch.zeros((batch_size, 3, 8, 8)))

    ok = (
        features.shape == (batch_size, FEATURE_DIM)
        and features.dtype == np.float16
        and np.allclose(
            features[:, :768], cls_token.numpy(), atol=1e-2
        )
        and np.allclose(
            features[:, 768:], expected_mean_patch.numpy(), atol=1e-2
        )
    )
    return _check(
        "compute_features: [B,1536] float16 com CLS nas primeiras 768 colunas "
        "e média dos patch tokens nas últimas 768",
        ok,
    )


def _check_storage_round_trip() -> bool:
    features = np.random.randn(5, FEATURE_DIM).astype(np.float16)
    attrs: dict[str, object] = {
        "video_id": "synthetic_env/synthetic_video",
        "K": 5,
        "normalize_mean": np.array([0.485, 0.456, 0.406], dtype=np.float64),
        "normalize_std": np.array([0.229, 0.224, 0.225], dtype=np.float64),
    }

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "synthetic_env" / "synthetic_video.h5"
        write_dinov3_features_atomic(path, features, attrs)
        try:
            verify_written_file(path, features=features, attrs=attrs)
            round_trip_ok = True
        except Dinov3StorageError:
            round_trip_ok = False

        mismatched_features = np.random.randn(5, FEATURE_DIM).astype(np.float16)
        try:
            verify_written_file(path, features=mismatched_features, attrs=attrs)
            mismatch_caught = False
        except Dinov3StorageError:
            mismatch_caught = True

    ok = round_trip_ok and mismatch_caught
    return _check(
        "storage: escrita atômica + releitura confere features e atributos "
        "(inclusive array-valued), e divergência é detectada",
        ok,
    )


def run_dinov3_selftest() -> None:
    checks = [
        _check_preprocess_frames_shape_and_dtype(),
        _check_compute_features_formula(),
        _check_storage_round_trip(),
    ]
    if not all(checks):
        print("\ndinov3 extract selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\ndinov3 extract selftest OK: todas as checagens passaram")
