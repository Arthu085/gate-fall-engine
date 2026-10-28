"""Grade causal de 10 fps e rótulo por quadro para fontes com tempos explícitos."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, TARGET_FPS
from gatefall.data.frames import apply_frame_schema
from gatefall.data.intervals import sweep_gaps_and_overlap, tag_gap_positions
from gatefall.data.resampling import build_time_grid, labels_for_grid
from gatefall.data.staged.labels import UnverifiableLabelsError

# Um passo da grade: além disso o quadro de origem mais próximo pertence a
# outro instante da grade e o timestep não tem observação própria.
MAX_ALIGNMENT_OFFSET_S = 1.0 / TARGET_FPS
# Anotação que termina depois do fim da fonte por mais que isso não é
# verificável contra os quadros disponíveis.
LABEL_END_TOLERANCE_S = 0.5
SOURCE_GAP_POSITION = "source_gap"


@dataclass(frozen=True)
class VideoGrid:
    times_s: np.ndarray
    src_indices: np.ndarray
    offsets_s: np.ndarray


def grid_from_timestamps(timestamps_s: np.ndarray, duration_s: float) -> VideoGrid:
    if timestamps_s.ndim != 1 or len(timestamps_s) < 2 or np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("timestamps devem ser 1-D, com dois ou mais valores estritamente crescentes")
    k = int(np.floor(duration_s * TARGET_FPS))
    times = np.arange(k, dtype=np.float64) / TARGET_FPS
    right = np.clip(np.searchsorted(timestamps_s, times, side="left"), 0, len(timestamps_s) - 1)
    left = np.clip(right - 1, 0, len(timestamps_s) - 1)
    # Mesma regra do Le2i (quadro de origem mais próximo); empate fica com o anterior.
    choose_left = np.abs(times - timestamps_s[left]) <= np.abs(timestamps_s[right] - times)
    # Numa lacuna de quadros maior que um passo, o quadro futuro mais próximo
    # anteciparia a observação: usa-se o último quadro em ou antes de t.
    at_or_before = np.where(timestamps_s[right] <= times, right, left)
    future_gap = (~choose_left) & (timestamps_s[right] - times > MAX_ALIGNMENT_OFFSET_S + 1e-9)
    src = np.where(choose_left, left, right)
    src = np.where(future_gap, at_or_before, src).astype(np.int32)
    return VideoGrid(times, src, np.abs(timestamps_s[src] - times))


def grid_from_video(n_frames: int, fps: float) -> VideoGrid:
    times, src = build_time_grid(n_frames, fps, TARGET_FPS)
    return VideoGrid(times, src, np.abs(src / fps - times))


def build_video_frames(
    *,
    video_id: str,
    split: str,
    env: str,
    subject: int,
    grid: VideoGrid,
    duration_s: float,
    segments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if segments.empty:
        raise UnverifiableLabelsError(f"{video_id}: nenhum segmento anotado")
    latest_end = float(segments["end"].to_numpy(dtype=np.float64).max())
    if latest_end > duration_s + LABEL_END_TOLERANCE_S:
        raise UnverifiableLabelsError(
            f"{video_id}: anotação termina em {latest_end:.3f}s, além da fonte "
            f"({duration_s:.3f}s + {LABEL_END_TOLERANCE_S}s)"
        )
    k = len(grid.times_s)
    if k == 0:
        raise UnverifiableLabelsError(f"{video_id}: fonte mais curta que um passo da grade")
    grid_labels = labels_for_grid(
        cast(pd.DataFrame, segments[["start", "end", "label"]]), grid.times_s, IGNORE_LABEL
    )
    labels = grid_labels.labels.copy()

    gap_position = np.full(k, None, dtype=object)
    tuples = list(zip(segments["start"].astype(float), segments["end"].astype(float)))
    _, _, gaps = sweep_gaps_and_overlap(tuples, duration_s)
    for start, end, position in tag_gap_positions(gaps, duration_s):
        lo = int(np.searchsorted(grid.times_s, start, side="left"))
        hi = int(np.searchsorted(grid.times_s, end, side="left"))
        gap_position[lo:hi] = position

    # O rótulo continua definido pelo tempo anotado; só a observação é antiga.
    # Ignorar esses timesteps partiria um evento de queda em dois.
    misaligned = grid.offsets_s > MAX_ALIGNMENT_OFFSET_S + 1e-9
    gap_position[misaligned] = SOURCE_GAP_POSITION

    frames = apply_frame_schema(
        pd.DataFrame(
            {
                "video_id": video_id,
                "split": split,
                "env": env,
                "subject": subject,
                "frame_index": np.arange(k, dtype=np.int32),
                "time_s": grid.times_s,
                "src_index": grid.src_indices,
                "label": labels,
                "gap_position": gap_position,
            }
        )
    )
    alignment = {
        "video_id": video_id,
        "k": k,
        "duration_s": duration_s,
        "max_offset_s": float(grid.offsets_s.max()),
        "n_misaligned": int(misaligned.sum()),
        "n_overlap_resolved": grid_labels.n_overlap_resolved,
        "n_segments_skipped": grid_labels.n_segments_skipped,
        "n_ignored": int((labels == IGNORE_LABEL).sum()),
    }
    return frames, alignment
