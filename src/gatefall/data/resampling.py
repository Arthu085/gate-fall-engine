"""Grade de reamostragem temporal e definição do rótulo por quadro (agnóstico de dataset)."""

import numpy as np
import pandas as pd


def build_time_grid(
    n_frames: int, fps_src: float, fps_target: float
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (times_s float64[K], src_indices int32[K]).
    duration = n_frames / fps_src ; K = floor(duration * fps_target)
    times[k] = k / fps_target
    src_indices = clip(round(times * fps_src), 0, n_frames - 1)"""
    duration_s = n_frames / fps_src
    k = int(np.floor(duration_s * fps_target))
    times = np.arange(k, dtype=np.float64) / fps_target
    src_indices = np.clip(np.round(times * fps_src), 0, n_frames - 1).astype(
        np.int32
    )
    return times, src_indices


def labels_for_grid(
    segments: pd.DataFrame, times: np.ndarray, ignore_label: int
) -> tuple[np.ndarray, int]:
    """Retorna (labels int8[K], n_overlap_resolved).
    Intervalo semiaberto [start, end). Timestamps não cobertos por nenhum segmento
    recebem ignore_label. Em sobreposição vence o segmento de menor `start`;
    devolve quantos timestamps foram afetados."""
    k = times.shape[0]
    labels = np.full(k, ignore_label, dtype=np.int8)
    filled = np.zeros(k, dtype=bool)
    overlapped = np.zeros(k, dtype=bool)

    ordered = segments.sort_values("start", kind="stable")
    for start, end, label in zip(
        ordered["start"], ordered["end"], ordered["label"]
    ):
        lo = int(np.searchsorted(times, start, side="left"))
        hi = int(np.searchsorted(times, end, side="left"))
        if hi <= lo:
            continue
        window_filled = filled[lo:hi]
        overlapped[lo:hi] |= window_filled
        to_write = ~window_filled
        labels[lo:hi] = np.where(to_write, label, labels[lo:hi])
        filled[lo:hi] = True

    return labels, int(overlapped.sum())
