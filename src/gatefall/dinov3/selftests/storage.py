import tempfile
from pathlib import Path

import numpy as np

from gatefall.dinov3.backbone import FEATURE_DIM
from gatefall.dinov3.selftests.fixtures import _check
from gatefall.dinov3.storage import (
    Dinov3StorageError,
    validate_existing_file,
    verify_written_file,
    write_dinov3_features_atomic,
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
