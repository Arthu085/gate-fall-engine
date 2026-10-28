"""Grade causal de 10 fps e rótulo por quadro para fontes com tempos explícitos."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, TARGET_FPS
from gatefall.data.frames import apply_frame_schema
from gatefall.data.intervals import sweep_gaps_and_overlap, tag_gap_positions
from gatefall.data.resampling import labels_for_grid
from gatefall.data.staged.labels import UnverifiableLabelsError

# Contrato causal dos datasets encenados: o quadro de origem do instante t é o
# último com tempo <= t; nenhum quadro posterior a t é usado. O Le2i mantém o
# quadro mais próximo (`build_time_grid`), que pode ficar até meio intervalo de
# origem à frente de t, para preservar as features e referências congeladas.
CAUSAL_TOLERANCE_S = 1e-9
# Observação mais antiga que um passo da grade não pertence ao instante t.
MAX_OBSERVATION_LAG_S = 1.0 / TARGET_FPS
# Anotação que termina depois do fim da fonte por mais que isso não é
# verificável contra os quadros disponíveis.
LABEL_END_TOLERANCE_S = 0.5
SOURCE_GAP_POSITION = "source_gap"


class FutureObservationError(ValueError):
    """Quadro de origem posterior ao instante de predição."""


@dataclass(frozen=True)
class VideoGrid:
    times_s: np.ndarray
    src_indices: np.ndarray
    source_times_s: np.ndarray

    @property
    def lag_s(self) -> np.ndarray:
        return self.times_s - self.source_times_s


def _grid_times(duration_s: float) -> np.ndarray:
    return np.arange(int(np.floor(duration_s * TARGET_FPS)), dtype=np.float64) / TARGET_FPS


def grid_from_timestamps(timestamps_s: np.ndarray, duration_s: float) -> VideoGrid:
    """Grade de 10 fps sobre tempos explícitos (relativos ao primeiro quadro)."""
    if timestamps_s.ndim != 1 or len(timestamps_s) < 2 or np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("timestamps devem ser 1-D, com dois ou mais valores estritamente crescentes")
    if timestamps_s[0] != 0:
        raise ValueError("timestamps devem ser relativos ao primeiro quadro (primeiro = 0)")
    times = _grid_times(duration_s)
    # Empate (quadro exatamente em t, a menos do erro de ponto flutuante) conta como passado.
    src = np.searchsorted(timestamps_s, times + CAUSAL_TOLERANCE_S, side="right") - 1
    return VideoGrid(times, src.astype(np.int32), timestamps_s[src])


def grid_from_video(n_frames: int, fps: float) -> VideoGrid:
    """Grade de 10 fps sobre vídeo de taxa constante: o quadro j é exibido em j / fps."""
    if n_frames < 1 or not fps > 0:
        raise ValueError("vídeo exige quadros e fps positivos")
    times = _grid_times(n_frames / fps)
    src = np.clip(np.floor((times + CAUSAL_TOLERANCE_S) * fps), 0, n_frames - 1).astype(np.int32)
    return VideoGrid(times, src, src / fps)


def verify_causal_grid(grid: VideoGrid, video_id: str) -> None:
    future = grid.lag_s < -CAUSAL_TOLERANCE_S
    if future.any():
        first = int(np.flatnonzero(future)[0])
        raise FutureObservationError(
            f"{video_id}: passo {first} (t={grid.times_s[first]:.6f}s) usaria o quadro "
            f"{int(grid.src_indices[first])} em {grid.source_times_s[first]:.6f}s")


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
    verify_causal_grid(grid, video_id)
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

    # O rótulo continua definido pelo tempo anotado; só a observação está atrasada.
    # Ignorar esses timesteps partiria um evento de queda em dois.
    misaligned = grid.lag_s > MAX_OBSERVATION_LAG_S + CAUSAL_TOLERANCE_S
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
        "max_lag_s": float(grid.lag_s.max()),
        "n_misaligned": int(misaligned.sum()),
        "n_overlap_resolved": grid_labels.n_overlap_resolved,
        "n_segments_skipped": grid_labels.n_segments_skipped,
        "n_ignored": int((labels == IGNORE_LABEL).sum()),
    }
    return frames, alignment
