"""Sequência causal `[K, 2]` de proxies de qualidade por quadro.

Nenhuma fórmula é redefinida aqui: `q_pose` vem de
`gatefall.pose.quality.compute_pose_quality` e `q_visual` vem de
`gatefall.dinov3.quality.compute_visual_quality`, aplicado exatamente sobre o
mesmo pré-processamento de quadro usado na extração DINOv3
(`resize_frames(decode_frames(...))`). Este módulo só alinha as duas
sequências quadro a quadro e as empilha na ordem de canal persistida
(0 = q_pose, 1 = q_visual).

Os dois valores são proxies operacionais de qualidade em [0, 1], não
probabilidades calibradas nem escores de confiança.
"""

from pathlib import Path

import numpy as np

from gatefall.data.video_io import decode_frames
from gatefall.dinov3.preprocessing import resize_frames
from gatefall.dinov3.quality import compute_visual_quality
from gatefall.features.quality_storage import QUALITY_CHANNELS
from gatefall.pose.quality import compute_pose_quality

DEFAULT_BATCH_SIZE = 32


class QualitySequenceError(ValueError):
    """Sequências de qualidade desalinhadas, fora de faixa ou não finitas."""


def assemble_quality_sequence(
    q_pose: np.ndarray, q_visual: np.ndarray
) -> np.ndarray:
    if q_pose.ndim != 1 or q_visual.ndim != 1:
        raise QualitySequenceError(
            f"q_pose e q_visual devem ser 1D; recebidos {q_pose.shape} e "
            f"{q_visual.shape}"
        )
    if q_pose.shape[0] != q_visual.shape[0]:
        raise QualitySequenceError(
            f"q_pose tem K={q_pose.shape[0]} e q_visual tem "
            f"K={q_visual.shape[0]}: as duas fontes precisam cobrir "
            "exatamente os mesmos quadros"
        )
    stacked = np.stack([q_pose, q_visual], axis=1).astype(np.float32)
    if not np.isfinite(stacked).all():
        raise QualitySequenceError("sequência de qualidade tem valor não finito")
    if not bool(np.all((stacked >= 0.0) & (stacked <= 1.0))):
        raise QualitySequenceError(
            "sequência de qualidade fora do intervalo [0, 1]"
        )
    if stacked.shape[1] != QUALITY_CHANNELS:
        raise QualitySequenceError(
            f"sequência de qualidade deve ter {QUALITY_CHANNELS} canais"
        )
    return stacked


def compute_visual_quality_sequence(
    video_path: Path, src_indices: list[int], *, batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    if batch_size <= 0:
        raise QualitySequenceError("batch_size deve ser positivo")
    frames_rgb = decode_frames(video_path, src_indices)
    if len(frames_rgb) != len(src_indices):
        raise QualitySequenceError(
            f"{video_path}: decodificação retornou {len(frames_rgb)} quadros; "
            f"esperado {len(src_indices)}"
        )
    values = np.zeros(len(frames_rgb), dtype=np.float32)
    for start in range(0, len(frames_rgb), batch_size):
        end = min(start + batch_size, len(frames_rgb))
        resized = resize_frames(frames_rgb[start:end])
        values[start:end] = compute_visual_quality(resized).q_visual
    return values


def compute_quality_sequence(
    video_id: str,
    *,
    pose_root: Path,
    video_path: Path,
    src_indices: list[int],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    q_pose = compute_pose_quality(video_id, pose_root=pose_root).q_pose
    if q_pose.shape[0] != len(src_indices):
        raise QualitySequenceError(
            f"video_id={video_id!r}: q_pose tem K={q_pose.shape[0]}, esperado "
            f"K={len(src_indices)} (grade temporal de frames.parquet)"
        )
    q_visual = compute_visual_quality_sequence(
        video_path, src_indices, batch_size=batch_size
    )
    return assemble_quality_sequence(q_pose, q_visual)
