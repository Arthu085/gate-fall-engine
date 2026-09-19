"""Pré-processamento de quadros RGB para o backbone DINOv3."""

import numpy as np
import torch
from torchvision.transforms.v2.functional import InterpolationMode, resize

from gatefall.dinov3.backbone import NORMALIZE_MEAN, NORMALIZE_STD, RESIZE_SIZE


def resize_frames(frames_rgb: list[np.ndarray]) -> torch.Tensor:
    if not frames_rgb:
        raise ValueError("frames_rgb não pode ser vazio")
    if any(frame.ndim != 3 or frame.shape[2] != 3 for frame in frames_rgb):
        raise ValueError("cada quadro deve ter shape [H, W, 3]")

    indices_by_shape: dict[tuple[int, ...], list[int]] = {}
    for index, frame in enumerate(frames_rgb):
        indices_by_shape.setdefault(frame.shape, []).append(index)

    resized_frames: list[torch.Tensor | None] = [None] * len(frames_rgb)
    for indices in indices_by_shape.values():
        stacked = torch.from_numpy(
            np.stack([frames_rgb[index] for index in indices], axis=0)
        )
        chw = stacked.permute(0, 3, 1, 2)
        resized = resize(
            chw,
            size=[RESIZE_SIZE, RESIZE_SIZE],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).to(torch.float32) / 255.0
        for batch_index, original_index in enumerate(indices):
            resized_frames[original_index] = resized[batch_index : batch_index + 1]

    return torch.cat([frame for frame in resized_frames if frame is not None], dim=0)


def normalize_resized_frames(resized_frames: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (resized_frames - mean) / std


def preprocess_frames(frames_rgb: list[np.ndarray]) -> torch.Tensor:
    return normalize_resized_frames(resize_frames(frames_rgb))
