"""Selftest sintético da grade de reamostragem temporal (`resampling.py`).

Não toca no dataset real — todas as entradas são sintéticas, para travar o
comportamento da grade e da resolução de rótulo contra futuras mudanças
(inclusive uma troca de versão do numpy que altere o arredondamento).
"""

import sys

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, TARGET_FPS
from gatefall.data.resampling import build_time_grid, labels_for_grid


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_basic_grid() -> bool:
    times, src_indices = build_time_grid(n_frames=250, fps_src=25.0, fps_target=10.0)
    ok = times.shape[0] == 100
    ok = ok and bool(
        np.array_equal(src_indices[:5], np.array([0, 2, 5, 8, 10], dtype=np.int32))
    )
    return _check(
        "grade básica: 25fps/250 frames -> K=100, src_indices[:5]=[0,2,5,8,10]", ok
    )


def check_floor_boundary() -> bool:
    times, _ = build_time_grid(n_frames=240, fps_src=24.000384, fps_target=10.0)
    return _check(
        "fronteira de floor: 24.000384fps/240 frames -> K=99 (não 100)",
        times.shape[0] == 99,
    )


def check_half_open_boundary() -> bool:
    segments = pd.DataFrame({"start": [1.0, 2.0], "end": [2.0, 3.0], "label": [1, 2]})
    times = np.array([1.5, 2.0, 2.5], dtype=np.float64)
    labels, _ = labels_for_grid(segments, times, IGNORE_LABEL)
    return _check(
        "fronteira semiaberta [start, end): t=2.0 pertence ao segundo segmento",
        int(labels[1]) == 2,
    )


def check_subframe_seam() -> bool:
    segments = pd.DataFrame({"start": [0.0, 0.9502], "end": [0.95, 2.0], "label": [1, 2]})

    times_inside = np.array([0.0, 0.9501, 2.0], dtype=np.float64)
    labels_inside, _ = labels_for_grid(segments, times_inside, IGNORE_LABEL)
    has_ignore_inside = bool(np.any(labels_inside == IGNORE_LABEL))

    times_outside = np.array([0.0, 1.0], dtype=np.float64)
    labels_outside, _ = labels_for_grid(segments, times_outside, IGNORE_LABEL)
    has_ignore_outside = bool(np.any(labels_outside == IGNORE_LABEL))

    ok = has_ignore_inside and not has_ignore_outside
    return _check(
        "seam sub-quadro entre segmentos: IGNORE só quando um ponto da grade "
        "cai dentro dele — labels_for_grid não preenche gaps sub-quadro",
        ok,
    )


def check_overlap_resolution() -> bool:
    segments = pd.DataFrame({"start": [0.0, 1.0], "end": [2.0, 3.0], "label": [1, 2]})
    times = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    labels, n_overlap_resolved = labels_for_grid(segments, times, IGNORE_LABEL)
    ok = int(labels[1]) == 1 and n_overlap_resolved > 0
    return _check(
        "sobreposição: segmento de menor `start` vence e n_overlap_resolved > 0",
        ok,
    )


def check_leading_gap() -> bool:
    times = np.arange(40, dtype=np.float64) / TARGET_FPS
    segments = pd.DataFrame({"start": [2.8], "end": [4.0], "label": [1]})
    labels, _ = labels_for_grid(segments, times, IGNORE_LABEL)
    ok = bool(np.all(labels[:28] == IGNORE_LABEL)) and int(labels[28]) == 1
    return _check(
        "gap inicial: exatamente 28 quadros IGNORE antes de t=2.8s", ok
    )


def check_empty_grid() -> bool:
    times, src_indices = build_time_grid(n_frames=2, fps_src=25.0, fps_target=10.0)
    ok = times.shape == (0,) and src_indices.shape == (0,)

    empty_segments = pd.DataFrame(
        {
            "start": pd.Series(dtype="float64"),
            "end": pd.Series(dtype="float64"),
            "label": pd.Series(dtype="int64"),
        }
    )
    labels, n_overlap_resolved = labels_for_grid(empty_segments, times, IGNORE_LABEL)
    ok = ok and labels.shape == (0,) and n_overlap_resolved == 0
    return _check(
        "K=0 (n_frames=2, fps_src=25, fps_target=10): sem crash, arrays vazios", ok
    )


def check_src_index_clip() -> bool:
    _, src_indices_basic = build_time_grid(n_frames=250, fps_src=25.0, fps_target=10.0)
    ok = bool(np.all(src_indices_basic <= 249))

    _, src_indices_stress = build_time_grid(n_frames=1, fps_src=1.0, fps_target=10.0)
    ok = ok and bool(np.all(src_indices_stress <= 0))
    return _check(
        "src_indices nunca excede n_frames - 1 (inclui caso de estresse do clip)",
        ok,
    )


def run_resampling_selftest() -> None:
    checks = [
        check_basic_grid(),
        check_floor_boundary(),
        check_half_open_boundary(),
        check_subframe_seam(),
        check_overlap_resolution(),
        check_leading_gap(),
        check_empty_grid(),
        check_src_index_clip(),
    ]
    if not all(checks):
        print("\nselftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: todos os casos passaram")
