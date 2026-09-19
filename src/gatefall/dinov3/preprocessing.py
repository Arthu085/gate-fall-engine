"""Pré-processamento de quadros RGB para o backbone DINOv3."""

import numpy as np
import torch
from torchvision.transforms.v2.functional import InterpolationMode, resize

from gatefall.dinov3.backbone import NORMALIZE_MEAN, NORMALIZE_STD, RESIZE_SIZE


def preprocess_frames(frames_rgb: list[np.ndarray]) -> torch.Tensor:
    stacked = torch.from_numpy(np.stack(frames_rgb, axis=0))
    chw = stacked.permute(0, 3, 1, 2)
    resized = resize(
        chw,
        size=[RESIZE_SIZE, RESIZE_SIZE],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    normalized = resized.to(torch.float32) / 255.0
    mean = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (normalized - mean) / std
