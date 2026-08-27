"""Janelamento deslizante sobre a grade de reamostragem temporal (agnóstico de dataset)."""

from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, WINDOW_FRAMES


def window_frame_indices(
    k_end: int, n_frames: int, window_frames: int = WINDOW_FRAMES
) -> np.ndarray:
    """Retorna os `window_frames` índices de quadro que terminam em `k_end`.

    `raw` cobre `k_end - window_frames + 1 .. k_end`; o `clip` inferior em 0
    É a própria replicação de borda (edge padding) para janelas próximas do
    início do vídeo — não existe um caminho de código separado para padding."""
    raw = np.arange(k_end - window_frames + 1, k_end + 1, dtype=np.int64)
    return np.clip(raw, 0, n_frames - 1)


def window_end_indices(n_frames: int, stride: int) -> np.ndarray:
    """Retorna os índices de fim de janela `0, stride, 2*stride, ...` abaixo de `n_frames`.

    O último fim NÃO é forçado para `n_frames - 1` de propósito: em
    EVAL_STRIDE=1 todo quadro já é um fim de janela, e em TRAIN_STRIDE=4
    perder no máximo 3 quadros finais por vídeo é irrelevante. Não "corrigir"
    isso para alinhar o último fim ao último quadro."""
    return np.arange(0, n_frames, stride, dtype=np.int64)


def build_window_index(
    frames: pd.DataFrame, stride: int, drop_ignored: bool = True
) -> pd.DataFrame:
    """Constrói o índice de janelas a partir da tabela de quadros genérica.

    Uma linha por janela, colunas `video_id, split, env, subject, k_end,
    label, n_frames`. `n_frames` é o K daquele vídeo, necessário para quem só
    tem o índice de janela reconstruir os `window_frames` índices de quadro a
    partir de `k_end`. Nada aqui faz I/O."""
    window_tables: list[pd.DataFrame] = []
    for _video_id, group in frames.groupby("video_id", sort=False):
        # frame_index é contíguo 0..K-1 por vídeo — garantido a montante pela
        # grade de reamostragem — então k_end pode indexar `label` por posição.
        ordered = cast(
            pd.DataFrame, group.sort_values("frame_index")
        ).reset_index(drop=True)
        n_frames = len(ordered)
        if not np.array_equal(
            ordered["frame_index"].to_numpy(), np.arange(n_frames, dtype=np.int64)
        ):
            raise ValueError(
                f"video_id={_video_id!r}: frame_index não é contíguo 0..K-1 "
                "— a indexação posicional de `label` por `k_end` corromperia "
                "o rótulo de toda janela deste vídeo"
            )
        k_end = window_end_indices(n_frames, stride)
        labels = ordered["label"].to_numpy()[k_end]

        window_tables.append(
            pd.DataFrame(
                {
                    "video_id": ordered["video_id"].iloc[0],
                    "split": ordered["split"].iloc[0],
                    "env": ordered["env"].iloc[0],
                    "subject": ordered["subject"].iloc[0],
                    "k_end": k_end,
                    "label": labels,
                    "n_frames": n_frames,
                }
            )
        )

    if not window_tables:
        windows = pd.DataFrame(
            columns=[
                "video_id",
                "split",
                "env",
                "subject",
                "k_end",
                "label",
                "n_frames",
            ]
        )
    else:
        windows = pd.concat(window_tables, ignore_index=True)

    if drop_ignored:
        windows = cast(
            pd.DataFrame, windows[windows["label"] != IGNORE_LABEL]
        )

    ordered_windows = cast(
        pd.DataFrame, windows.sort_values(["video_id", "k_end"])
    ).reset_index(drop=True)
    return ordered_windows
