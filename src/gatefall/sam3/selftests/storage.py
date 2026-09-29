"""Checagens sintéticas do armazenamento SAM 3."""

import tempfile
from pathlib import Path

import numpy as np

from gatefall.sam3.descriptors import V_T_DIM
from gatefall.sam3.storage import (
    Sam3StorageError,
    validate_existing_file,
    verify_written_file,
    write_sam3_atomic,
)
from gatefall.sam3.selftests.fixtures import _check


def _check_storage_round_trip() -> bool:
    v_t = np.random.randn(5, V_T_DIM).astype(np.float32)
    sam_score = np.random.rand(5).astype(np.float32)
    n_instances = np.array([1, 2, 0, 1, 3], dtype=np.int16)
    attrs: dict[str, object] = {
        "video_id": "synthetic_env/synthetic_video",
        "K": 5,
        "model_name": "facebook/sam3",
    }

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "synthetic_env" / "synthetic_video.h5"
        write_sam3_atomic(path, v_t, sam_score, n_instances, attrs)
        try:
            verify_written_file(
                path, v_t=v_t, sam_score=sam_score, n_instances=n_instances, attrs=attrs
            )
            round_trip_ok = True
        except Sam3StorageError:
            round_trip_ok = False

        mismatched_v_t = np.random.randn(5, V_T_DIM).astype(np.float32)
        try:
            verify_written_file(
                path, v_t=mismatched_v_t, sam_score=sam_score, n_instances=n_instances, attrs=attrs
            )
            mismatch_caught = False
        except Sam3StorageError:
            mismatch_caught = True

    ok = round_trip_ok and mismatch_caught
    return _check(
        "storage: escrita atômica + releitura confere v_t/sam_score/"
        "n_instances e atributos bit-a-bit, e divergência é detectada", ok
    )


def _check_validate_existing_file() -> bool:
    v_t = np.random.randn(5, V_T_DIM).astype(np.float32)
    sam_score = np.zeros(5, dtype=np.float32)
    n_instances = np.zeros(5, dtype=np.int16)
    expected_attrs: dict[str, object] = {"model_name": "facebook/sam3"}
    attrs: dict[str, object] = {"K": 5, **expected_attrs}

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "synthetic_env" / "synthetic_video.h5"
        write_sam3_atomic(path, v_t, sam_score, n_instances, attrs)

        valid_reasons = validate_existing_file(
            path, expected_k=5, v_t_dim=V_T_DIM, expected_attrs=expected_attrs
        )
        valid_ok = valid_reasons == []

        wrong_k_reasons = validate_existing_file(
            path, expected_k=6, v_t_dim=V_T_DIM, expected_attrs=expected_attrs
        )
        wrong_k_ok = len(wrong_k_reasons) > 0

        wrong_attr_reasons = validate_existing_file(
            path,
            expected_k=5,
            v_t_dim=V_T_DIM,
            expected_attrs={"model_name": "different_model"},
        )
        wrong_attr_ok = (
            len(wrong_attr_reasons) > 0 and "model_name" in wrong_attr_reasons[0]
        )

    with tempfile.TemporaryDirectory() as temporary_dir:
        wrong_n_instances_dtype_path = Path(temporary_dir) / "synthetic_env" / "video.h5"
        write_sam3_atomic(
            wrong_n_instances_dtype_path,
            v_t,
            sam_score,
            n_instances.astype(np.int32),
            attrs,
        )
        wrong_n_instances_dtype_reasons = validate_existing_file(
            wrong_n_instances_dtype_path,
            expected_k=5,
            v_t_dim=V_T_DIM,
            expected_attrs=expected_attrs,
        )
        wrong_n_instances_dtype_ok = (
            len(wrong_n_instances_dtype_reasons) > 0
            and any("n_instances" in reason for reason in wrong_n_instances_dtype_reasons)
        )

    with tempfile.TemporaryDirectory() as temporary_dir:
        wrong_sam_score_dtype_path = Path(temporary_dir) / "synthetic_env" / "video.h5"
        write_sam3_atomic(
            wrong_sam_score_dtype_path,
            v_t,
            sam_score.astype(np.float64),
            n_instances,
            attrs,
        )
        wrong_sam_score_dtype_reasons = validate_existing_file(
            wrong_sam_score_dtype_path,
            expected_k=5,
            v_t_dim=V_T_DIM,
            expected_attrs=expected_attrs,
        )
        wrong_sam_score_dtype_ok = (
            len(wrong_sam_score_dtype_reasons) > 0
            and any("sam_score" in reason for reason in wrong_sam_score_dtype_reasons)
        )

    ok = (
        valid_ok
        and wrong_k_ok
        and wrong_attr_ok
        and wrong_n_instances_dtype_ok
        and wrong_sam_score_dtype_ok
    )
    return _check(
        "validate_existing_file: arquivo válido dá [], shape divergente, "
        "atributo divergente e dtype divergente de sam_score/n_instances "
        "dão razões nomeadas", ok
    )
