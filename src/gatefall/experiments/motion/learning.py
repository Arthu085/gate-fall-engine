"""Duas previsões independentes acopladas pela identidade cinemática."""

import torch
from torch import nn
from torch.nn import functional as F

from gatefall.experiments.motion import JOINTS, SPATIAL_SCALE
from gatefall.experiments.physics.model import FusionClassifier


class MotionClassifier(FusionClassifier):
    def __init__(self, visual_dim: int, fusion: str, horizons: list[int]) -> None:
        super().__init__(visual_dim, fusion)
        self.horizons = horizons
        # Linear nasce em CPU: preservar RNG mantém os logits e dropout da referência.
        with torch.random.fork_rng(devices=[]):
            self.position_head = nn.Linear(32, len(horizons) * len(JOINTS) * 2)
            self.velocity_head = nn.Linear(32, len(horizons) * len(JOINTS) * 2)

    def forward_with_motion(
        self, pose: torch.Tensor, visual: torch.Tensor, evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.encode(pose, visual, evidence)
        shape = (len(state), len(self.horizons), len(JOINTS), 2)
        return (self.classifier(state), self.position_head(state).reshape(shape),
                self.velocity_head(state).reshape(shape))


def masked_huber(error: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.unsqueeze(-1).expand_as(error)
    safe = torch.where(mask, error / SPATIAL_SCALE, torch.zeros_like(error))
    terms = F.smooth_l1_loss(safe, torch.zeros_like(safe), reduction="none")
    return terms.sum() / mask.sum().clamp_min(1)


def motion_losses(
    position: torch.Tensor, velocity: torch.Tensor, batch: dict[str, torch.Tensor],
    intervals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = intervals.reshape(1, -1, 1, 1)
    valid = batch["motion_valid"]
    supervised = masked_huber(position - batch["motion_future"], valid)
    supervised = supervised + masked_huber(dt * (velocity - batch["motion_velocity"]), valid)
    # v é média do mesmo intervalo: não se pressupõe velocidade/aceleração constante.
    residual = position - batch["motion_current"].unsqueeze(1) - dt * velocity
    consistency = masked_huber(residual, valid)
    return supervised, consistency
