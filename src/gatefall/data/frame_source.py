"""Decodificação de quadros pela fonte do dataset: arquivo de vídeo ou sequência de imagens."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from gatefall.datasets import DatasetAdapter

VideoDecoder = Callable[[Path, Sequence[int]], list[np.ndarray]]


@runtime_checkable
class ImageSequenceSource(Protocol):
    def decode_image_frames(
        self, video_id: str, src_indices: Sequence[int]
    ) -> list[np.ndarray]: ...


def decode_dataset_frames(
    adapter: DatasetAdapter,
    video_id: str,
    src_indices: Sequence[int],
    *,
    decode_video: VideoDecoder,
) -> list[np.ndarray]:
    """Devolve quadros RGB uint8 [H, W, 3] na ordem de `src_indices`.

    `decode_video` é recebido do módulo chamador para que selftests que
    substituem o decodificador de vídeo daquele módulo continuem valendo.
    """
    if isinstance(adapter, ImageSequenceSource):
        return adapter.decode_image_frames(video_id, src_indices)
    video_paths = adapter.video_paths()
    if video_id not in video_paths:
        raise KeyError(f"video_id {video_id!r} não encontrado no manifesto")
    return decode_video(video_paths[video_id], list(src_indices))
