"""Selftest sintético da extração DINOv3 (pré-processamento, features, armazenamento).

Não toca em GPU, dataset real ou pesos do backbone — todas as entradas são
sintéticas.
"""

import sys

from gatefall.dinov3.selftests.preprocessing_features import (
    _check_preprocess_frames_shape_and_dtype,
    _check_compute_features_formula,
)
from gatefall.dinov3.selftests.storage import (
    _check_storage_round_trip,
    _check_validate_existing_file,
)
from gatefall.dinov3.selftests.audit_provenance import (
    _check_audit_helpers,
    _check_provenance_divergences,
    _check_provenance_divergences_missing_attribute,
    _check_provenance_divergences_missing_from_every_file,
)
from gatefall.dinov3.selftests.frame_alignment import (
    _check_grid_positions,
    _check_max_abs_diff_within_tolerance,
    _check_check_discriminative_match,
    _check_run_dinov3_verify_frame_alignment_happy_path,
    _check_run_dinov3_verify_frame_alignment_shifted,
    _check_run_dinov3_verify_frame_alignment_shifted_within_tolerance,
    _check_run_dinov3_verify_frame_alignment_k_mismatch,
    _check_run_dinov3_verify_frame_alignment_missing_manifest_row,
    _check_run_dinov3_verify_frame_alignment_missing_h5,
)
from gatefall.dinov3.selftests.determinism_guard import (
    _check_determinism_plumbing,
    _check_configure_deterministic_inference_does_not_seed_rng,
    _check_adapter_with_dinov3_root,
    _check_dinov3_entry_points_reject_le2i_cv,
    _check_resolve_verify_determinism_output_root,
)


def run_dinov3_selftest() -> None:
    checks = [
        _check_preprocess_frames_shape_and_dtype(),
        _check_compute_features_formula(),
        _check_storage_round_trip(),
        _check_determinism_plumbing(),
        _check_audit_helpers(),
        _check_provenance_divergences(),
        _check_provenance_divergences_missing_attribute(),
        _check_provenance_divergences_missing_from_every_file(),
        _check_validate_existing_file(),
        _check_configure_deterministic_inference_does_not_seed_rng(),
        _check_grid_positions(),
        _check_max_abs_diff_within_tolerance(),
        _check_check_discriminative_match(),
        _check_run_dinov3_verify_frame_alignment_happy_path(),
        _check_run_dinov3_verify_frame_alignment_shifted(),
        _check_run_dinov3_verify_frame_alignment_shifted_within_tolerance(),
        _check_run_dinov3_verify_frame_alignment_k_mismatch(),
        _check_run_dinov3_verify_frame_alignment_missing_manifest_row(),
        _check_run_dinov3_verify_frame_alignment_missing_h5(),
        _check_adapter_with_dinov3_root(),
        _check_dinov3_entry_points_reject_le2i_cv(),
        _check_resolve_verify_determinism_output_root(),
    ]
    if not all(checks):
        print("\ndinov3 extract selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\ndinov3 extract selftest OK: todas as checagens passaram")
