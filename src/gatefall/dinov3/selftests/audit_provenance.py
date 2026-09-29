import numpy as np

from gatefall.dinov3 import storage
from gatefall.dinov3.audit import (
    DimensionStatsAccumulator,
    count_duplicate_consecutive_rows,
    count_non_finite,
    frame_index_is_contiguous,
    max_abs_and_headroom,
)
from gatefall.dinov3.backbone import FEATURE_DIM
from gatefall.dinov3.report import find_provenance_divergences
from gatefall.dinov3.selftests.fixtures import _check


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


def _full_provenance_attrs(**overrides: object) -> dict[str, object]:
    attrs: dict[str, object] = {
        name: f"valor-{name}" for name in storage.PROVENANCE_ATTR_NAMES
    }
    attrs.update(overrides)
    return attrs


def _check_provenance_divergences() -> bool:
    homogeneous: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(),
        "env1/v2": _full_provenance_attrs(),
    }
    homogeneous_ok = find_provenance_divergences(homogeneous) == []

    heterogeneous: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(),
        "env1/v2": _full_provenance_attrs(weights_sha256="different"),
    }
    divergences = find_provenance_divergences(heterogeneous)
    heterogeneous_ok = len(divergences) == 1 and "env1/v2" in divergences[0]

    ok = homogeneous_ok and heterogeneous_ok
    return _check(
        "find_provenance_divergences: dataset homogêneo dá [] e vídeo com "
        "atributo divergente é nomeado", ok
    )


def _check_provenance_divergences_missing_attribute() -> bool:
    missing_in_reference: dict[str, dict[str, object]] = {
        "env1/v1": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
        "env1/v2": _full_provenance_attrs(),
    }
    divergences_missing_in_reference = find_provenance_divergences(missing_in_reference)
    missing_in_reference_ok = (
        len(divergences_missing_in_reference) == 1
        and "weights_sha256" in divergences_missing_in_reference[0]
        and "env1/v1" in divergences_missing_in_reference[0]
    )

    missing_in_candidate: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(),
        "env1/v2": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
    }
    divergences_missing_in_candidate = find_provenance_divergences(missing_in_candidate)
    missing_in_candidate_ok = (
        len(divergences_missing_in_candidate) == 1
        and "weights_sha256" in divergences_missing_in_candidate[0]
        and "env1/v2" in divergences_missing_in_candidate[0]
    )

    ok = missing_in_reference_ok and missing_in_candidate_ok
    return _check(
        "find_provenance_divergences: atributo ausente em apenas um dos "
        "arquivos (referência ou candidato) é reportado como divergência", ok
    )


def _check_provenance_divergences_missing_from_every_file() -> bool:
    missing_everywhere: dict[str, dict[str, object]] = {
        "env1/v1": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
        "env1/v2": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
    }
    divergences = find_provenance_divergences(missing_everywhere)
    missing_everywhere_ok = (
        len(divergences) == 2
        and all("weights_sha256" in message for message in divergences)
    )

    single_video_with_missing_attribute: dict[str, dict[str, object]] = {
        "env1/v1": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
    }
    single_video_divergences = find_provenance_divergences(
        single_video_with_missing_attribute
    )
    single_video_ok = (
        len(single_video_divergences) == 1
        and "weights_sha256" in single_video_divergences[0]
        and "env1/v1" in single_video_divergences[0]
    )

    ok = missing_everywhere_ok and single_video_ok
    return _check(
        "find_provenance_divergences: atributo ausente em todos os arquivos "
        "(inclusive a referência) e dataset de um único vídeo são reportados", ok
    )
