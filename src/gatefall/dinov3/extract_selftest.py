"""Selftest sintético da extração DINOv3 (pré-processamento, features, armazenamento).

Não toca em GPU, dataset real ou pesos do backbone — todas as entradas são
sintéticas.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from gatefall.dinov3.audit import (
    DimensionStatsAccumulator,
    count_duplicate_consecutive_rows,
    count_non_finite,
    frame_index_is_contiguous,
    max_abs_and_headroom,
)
from gatefall.dinov3.backbone import (
    FEATURE_DIM,
    RESIZE_SIZE,
    configure_deterministic_inference,
)
from gatefall.dinov3.features import compute_features
from gatefall.dinov3.preprocessing import preprocess_frames
from gatefall.dinov3.report import find_provenance_divergences
from gatefall.dinov3.storage import (
    Dinov3StorageError,
    validate_existing_file,
    verify_written_file,
    write_dinov3_features_atomic,
)
from gatefall.hashing import sha256_array


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


def _check_determinism_plumbing() -> bool:
    try:
        configure_deterministic_inference()
        configure_deterministic_inference()
        no_exception = True
    except Exception:
        no_exception = False

    array = np.random.randn(5, FEATURE_DIM).astype(np.float16)
    hash_a = sha256_array(array)
    hash_b = sha256_array(array)
    mutated = array.copy()
    mutated[0, 0] += 1.0
    hash_mutated = sha256_array(mutated)

    cls_token = torch.arange(2 * 768, dtype=torch.float32).reshape(2, 768)
    patch_tokens = torch.zeros((2, 4, 768), dtype=torch.float32)
    backbone = _FakeBackbone(cls_token, patch_tokens)
    batch = torch.zeros((2, 3, 8, 8))
    features_a = compute_features(backbone, batch)
    features_b = compute_features(backbone, batch)
    pipeline_hash_a = sha256_array(features_a)
    pipeline_hash_b = sha256_array(features_b)

    ok = (
        no_exception
        and hash_a == hash_b
        and hash_a != hash_mutated
        and pipeline_hash_a == pipeline_hash_b
    )
    return _check(
        "determinismo: configure_deterministic_inference não lança, "
        "sha256_array é estável e sensível a mutação, pipeline fake repete hash",
        ok,
    )


def _check_audit_helpers() -> bool:
    with_nan = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    with_nan[1, 0] = np.nan
    non_finite_ok = count_non_finite(with_nan) == 1

    max_abs_array = np.zeros((2, 2), dtype=np.float32)
    max_abs_array[0, 0] = 100.0
    max_abs, headroom = max_abs_and_headroom(max_abs_array, ceiling=200.0)
    max_abs_ok = max_abs == 100.0 and headroom == 100.0

    duplicate_array = np.array(
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32
    )
    duplicate_ok = count_duplicate_consecutive_rows(duplicate_array) == 1

    contiguous_ok = frame_index_is_contiguous([0, 1, 2, 3]) and not frame_index_is_contiguous(
        [0, 2, 3, 5]
    )

    accumulator = DimensionStatsAccumulator()
    dead_dim_array = np.zeros((10, 3), dtype=np.float32)
    dead_dim_array[:, 0] = 5.0
    dead_dim_array[:, 1] = np.arange(10, dtype=np.float32)
    dead_dim_array[:, 2] = -3.0
    accumulator.update(dead_dim_array)
    dead = accumulator.dead_dimensions()
    dead_ok = set(dead) == {0, 2}

    ok = non_finite_ok and max_abs_ok and duplicate_ok and contiguous_ok and dead_ok
    return _check(
        "audit: count_non_finite, max_abs_and_headroom, "
        "count_duplicate_consecutive_rows, frame_index_is_contiguous e "
        "DimensionStatsAccumulator dão os resultados esperados", ok
    )


def _check_provenance_divergences() -> bool:
    homogeneous: dict[str, dict[str, object]] = {
        "env1/v1": {"weights_sha256": "abc", "model_name": "dinov3_vitb16"},
        "env1/v2": {"weights_sha256": "abc", "model_name": "dinov3_vitb16"},
    }
    homogeneous_ok = find_provenance_divergences(homogeneous) == []

    heterogeneous: dict[str, dict[str, object]] = {
        "env1/v1": {"weights_sha256": "abc", "model_name": "dinov3_vitb16"},
        "env1/v2": {"weights_sha256": "different", "model_name": "dinov3_vitb16"},
    }
    divergences = find_provenance_divergences(heterogeneous)
    heterogeneous_ok = len(divergences) == 1 and "env1/v2" in divergences[0]

    ok = homogeneous_ok and heterogeneous_ok
    return _check(
        "find_provenance_divergences: dataset homogêneo dá [] e vídeo com "
        "atributo divergente é nomeado", ok
    )


def _check_validate_existing_file() -> bool:
    features = np.random.randn(5, FEATURE_DIM).astype(np.float16)
    expected_attrs: dict[str, object] = {
        "model_name": "dinov3_vitb16",
        "normalize_mean": np.array([0.485, 0.456, 0.406], dtype=np.float64),
    }
    attrs: dict[str, object] = {"K": 5, **expected_attrs}

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "synthetic_env" / "synthetic_video.h5"
        write_dinov3_features_atomic(path, features, attrs)

        valid_reasons = validate_existing_file(
            path, expected_k=5, feature_dim=FEATURE_DIM, expected_attrs=expected_attrs
        )
        valid_ok = valid_reasons == []

        wrong_k_reasons = validate_existing_file(
            path, expected_k=6, feature_dim=FEATURE_DIM, expected_attrs=expected_attrs
        )
        wrong_k_ok = len(wrong_k_reasons) > 0

        wrong_attr_reasons = validate_existing_file(
            path,
            expected_k=5,
            feature_dim=FEATURE_DIM,
            expected_attrs={"model_name": "different_model"},
        )
        wrong_attr_ok = (
            len(wrong_attr_reasons) > 0 and "model_name" in wrong_attr_reasons[0]
        )

    ok = valid_ok and wrong_k_ok and wrong_attr_ok
    return _check(
        "validate_existing_file: arquivo válido dá [], shape divergente e "
        "atributo divergente dão razões nomeadas", ok
    )


def run_dinov3_selftest() -> None:
    checks = [
        _check_preprocess_frames_shape_and_dtype(),
        _check_compute_features_formula(),
        _check_storage_round_trip(),
        _check_determinism_plumbing(),
        _check_audit_helpers(),
        _check_provenance_divergences(),
        _check_validate_existing_file(),
    ]
    if not all(checks):
        print("\ndinov3 extract selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\ndinov3 extract selftest OK: todas as checagens passaram")
