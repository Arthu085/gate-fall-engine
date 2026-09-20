"""Classificador de fusão adaptativa B1: gate escalar por timestep sobre as
projeções de pose e DINOv3, seguido da mesma TCN do braço A."""

import torch
from torch import nn

from gatefall.config import NUM_CLASSES, WINDOW_FRAMES
from gatefall.train.tcn import TCNEncoder

POSE_DIM = 134
VISUAL_DIM = 1536
PROJECTION_DIM = 128
FUSED_DIM = PROJECTION_DIM * 2
# Entrada do gate: [q_pose, q_visual] por quadro, nessa ordem de canal.
GATE_INPUT_DIM = 2
GATE_OUTPUT_DIM = 1


class AdaptiveGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(GATE_INPUT_DIM, GATE_OUTPUT_DIM)

    def forward(self, x_quality: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x_quality))


class B1AdaptiveGateClassifier(nn.Module):
    def __init__(
        self,
        channels: list[int],
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.e_p = nn.Sequential(
            nn.Linear(POSE_DIM, PROJECTION_DIM),
            nn.LayerNorm(PROJECTION_DIM),
            nn.ReLU(),
        )
        self.e_v = nn.Sequential(
            nn.Linear(VISUAL_DIM, PROJECTION_DIM),
            nn.LayerNorm(PROJECTION_DIM),
            nn.ReLU(),
        )
        self.gate = AdaptiveGate()
        self.encoder = TCNEncoder(FUSED_DIM, channels, kernel_size, dilations, dropout)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(
        self, x_pose: torch.Tensor, x_visual: torch.Tensor, x_quality: torch.Tensor
    ) -> torch.Tensor:
        if x_pose.shape[1] != WINDOW_FRAMES:
            raise ValueError(
                f"x_pose com {x_pose.shape[1]} quadros; esperado {WINDOW_FRAMES}"
            )
        if x_visual.shape[1] != WINDOW_FRAMES:
            raise ValueError(
                f"x_visual com {x_visual.shape[1]} quadros; esperado {WINDOW_FRAMES}"
            )
        if x_quality.shape[1] != WINDOW_FRAMES:
            raise ValueError(
                f"x_quality com {x_quality.shape[1]} quadros; esperado {WINDOW_FRAMES}"
            )
        if x_quality.shape[-1] != GATE_INPUT_DIM:
            raise ValueError(
                f"x_quality com {x_quality.shape[-1]} canais; esperado "
                f"{GATE_INPUT_DIM} (q_pose, q_visual)"
            )
        projected_pose = self.e_p(x_pose)
        projected_visual = self.e_v(x_visual)
        # g pondera a pose e 1 - g pondera o visual: pesos complementares por
        # timestep, sem alterar a dimensão fundida de 256.
        g = self.gate(x_quality)
        fused = torch.cat([g * projected_pose, (1.0 - g) * projected_visual], dim=-1)
        # [B, T, FUSED_DIM] -> [B, FUSED_DIM, T] para Conv1d.
        fused = fused.permute(0, 2, 1)
        encoded = self.encoder(fused)
        last_timestep = encoded[:, :, -1]
        return self.classifier(last_timestep)
