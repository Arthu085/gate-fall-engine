"""Cômputo do vetor de features DINOv3 (CLS + média dos patch tokens)."""

from typing import Protocol, cast

import numpy as np
import torch


class Dinov3Backbone(Protocol):
    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]: ...


def compute_features(backbone: Dinov3Backbone, batch: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        out = backbone.forward_features(batch)
        cls_token = cast(torch.Tensor, out["x_norm_clstoken"])
        patch_tokens = cast(torch.Tensor, out["x_norm_patchtokens"])
        # Descritor por quadro = token CLS ([B,768]) concatenado com a média
        # dos patch tokens ao longo do eixo espacial ([B,768]), formando um
        # vetor [B,1536] com contexto global (CLS) e contexto espacial médio
        # (patches) — mesma convenção usada por outros consumidores de
        # features DINOv3 congeladas.
        features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
    return features.cpu().numpy().astype(np.float16)
