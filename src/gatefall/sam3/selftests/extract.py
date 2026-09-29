"""Selftest sintético da extração SAM 3 (descritores, seleção, armazenamento).

Não toca em GPU, dataset real nem no runtime isolado `sam3_runtime/` — todas
as entradas são sintéticas e o segmentador é sempre um fake injetado via
`segmenter=`, nunca o `Sam3RuntimeSegmenter` real.
"""

import sys

from gatefall.sam3.selftests.descriptor_extraction import (
    _check_compute_descriptor_rectangle,
    _check_compute_descriptor_square_is_isotropic,
    _check_compute_descriptor_empty_mask,
    _check_missing_frame_is_isolated_zero_no_forward_fill,
    _check_end_to_end_continuity_tracks_moving_instance,
    _check_dataset_guard_rejects_le2i_cv,
    _check_extract_persists_inference_autocast_dtype_from_runtime_manifest,
    _check_extract_force_rejects_second_inference_autocast_dtype,
    _check_missing_video_id_raises_extract_error,
)
from gatefall.sam3.selftests.storage import (
    _check_storage_round_trip,
    _check_validate_existing_file,
)
from gatefall.sam3.selftests.provenance_report import (
    _check_provenance_divergence_heterogeneous_checkpoint,
    _check_provenance_divergence_mixed_inference_autocast_dtype,
    _check_find_invalid_required_provenance,
    _check_inference_autocast_dtype_provenance_format,
    _check_report_detects_structurally_invalid_h5,
    _check_report_rejects_malformed_source_revision,
    _check_report_rejects_malformed_checkpoint_digest,
    _check_report_rejects_malformed_inference_autocast_dtype,
)
from gatefall.sam3.selftests.runtime import (
    _check_build_worker_invocation_project_equals_cwd,
    _check_build_worker_invocation_checkpoint_absolute_independent_of_cwd,
    _check_resolve_precedence_and_absolute_paths,
    _check_ensure_sam3_runtime_available_requires_uv_lock,
    _check_select_inference_autocast_dtype_policy,
)
from gatefall.sam3.selftests.runtime_compatibility import (
    _check_setuptools_ceiling_matches_distribution_name_not_substring,
    _check_setuptools_ceiling_rejects_environment_markers_and_trailing_garbage,
    _check_setuptools_lock_version_boundary_and_corruption,
    _check_normalize_distribution_name_collapses_separator_runs,
    _check_sam3_runtime_lock_pins_setuptools_below_pkg_resources_removal,
    _check_sam3_runtime_worker_declares_same_inference_autocast_policy,
)


def run_sam3_selftest() -> None:
    checks = [
        _check_compute_descriptor_rectangle(),
        _check_compute_descriptor_square_is_isotropic(),
        _check_compute_descriptor_empty_mask(),
        _check_missing_frame_is_isolated_zero_no_forward_fill(),
        _check_end_to_end_continuity_tracks_moving_instance(),
        _check_storage_round_trip(),
        _check_validate_existing_file(),
        _check_provenance_divergence_heterogeneous_checkpoint(),
        _check_provenance_divergence_mixed_inference_autocast_dtype(),
        _check_dataset_guard_rejects_le2i_cv(),
        _check_build_worker_invocation_project_equals_cwd(),
        _check_build_worker_invocation_checkpoint_absolute_independent_of_cwd(),
        _check_resolve_precedence_and_absolute_paths(),
        _check_ensure_sam3_runtime_available_requires_uv_lock(),
        _check_select_inference_autocast_dtype_policy(),
        _check_find_invalid_required_provenance(),
        _check_inference_autocast_dtype_provenance_format(),
        _check_report_detects_structurally_invalid_h5(),
        _check_report_rejects_malformed_source_revision(),
        _check_report_rejects_malformed_checkpoint_digest(),
        _check_report_rejects_malformed_inference_autocast_dtype(),
        _check_extract_persists_inference_autocast_dtype_from_runtime_manifest(),
        _check_extract_force_rejects_second_inference_autocast_dtype(),
        _check_missing_video_id_raises_extract_error(),
        _check_setuptools_ceiling_matches_distribution_name_not_substring(),
        _check_setuptools_ceiling_rejects_environment_markers_and_trailing_garbage(),
        _check_setuptools_lock_version_boundary_and_corruption(),
        _check_normalize_distribution_name_collapses_separator_runs(),
        _check_sam3_runtime_lock_pins_setuptools_below_pkg_resources_removal(),
        _check_sam3_runtime_worker_declares_same_inference_autocast_policy(),
    ]
    if not all(checks):
        print("\nsam3 extract selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nsam3 extract selftest OK: todas as checagens passaram")
