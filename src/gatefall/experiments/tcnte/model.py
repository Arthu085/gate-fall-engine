"""Rede TCNTE e perda focal, com a mesma interface dos classificadores do estudo."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from gatefall.config import NUM_CLASSES, WINDOW_FRAMES
from gatefall.experiments.physics.kinematics import POSE_DIM

# COCO-17 sem os cinco pontos da face (nariz, olhos, orelhas): ombros a
# tornozelos, como nas 12 articulações × (x, y) = 24 entradas do artigo.
BODY_JOINTS = tuple(range(5, 17))
INPUT_DIM = len(BODY_JOINTS) * 2
KP_XY_END = 34
POSE_PRESENT_CHANNEL = 134
TCN_CHANNELS = (8, 16, 32)
TCN_DILATIONS = (1, 2, 4)
TCN_KERNEL = 3
TCN_DROPOUT = 0.2
ENCODER_HEADS = 4
ENCODER_FEEDFORWARD = 32
ENCODER_DROPOUT = 0.1
MINMAX_EPSILON = 1e-4
FOCAL_GAMMA = 3.0


def tcnte_input(pose: torch.Tensor) -> torch.Tensor:
    """[B, T, 138] pose138 bruta -> [B, T, 24] normalizada por quadro.

    Min–max por eixo entre as 12 articulações de cada quadro. O bloco kp_xy do
    pose138 é uma transformação afim isotrópica por quadro dos pixels, então o
    resultado coincide com o min–max sobre coordenadas de imagem. Quadros sem
    pessoa viram zero (o artigo não define tratamento de ausência).
    """
    batch, frames = pose.shape[:2]
    xy = pose[..., :KP_XY_END].reshape(batch, frames, 17, 2)[:, :, BODY_JOINTS, :]
    low = xy.amin(dim=2, keepdim=True)
    high = xy.amax(dim=2, keepdim=True)
    normalized = (xy - low) / (high - low + MINMAX_EPSILON)
    present = pose[..., POSE_PRESENT_CHANNEL].reshape(batch, frames, 1, 1)
    return (normalized * present).reshape(batch, frames, INPUT_DIM)


class _CausalWeightNormBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.padding = (TCN_KERNEL - 1) * dilation
        self.first = weight_norm(nn.Conv1d(in_channels, out_channels, TCN_KERNEL, dilation=dilation))
        self.second = weight_norm(nn.Conv1d(out_channels, out_channels, TCN_KERNEL, dilation=dilation))
        self.dropout = nn.Dropout(TCN_DROPOUT)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preenchimento só à esquerda: a saída em t depende apenas de t' <= t.
        out = self.dropout(F.relu(self.first(F.pad(x, (self.padding, 0)))))
        out = self.dropout(F.relu(self.second(F.pad(out, (self.padding, 0)))))
        return F.relu(out + self.residual(x))


class TCNTEClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, window_frames: int = WINDOW_FRAMES) -> None:
        super().__init__()
        blocks = []
        in_channels = INPUT_DIM
        for out_channels, dilation in zip(TCN_CHANNELS, TCN_DILATIONS):
            blocks.append(_CausalWeightNormBlock(in_channels, out_channels, dilation))
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        width = TCN_CHANNELS[-1]
        self.positional = nn.Parameter(torch.zeros(window_frames, width))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=ENCODER_HEADS,
                dim_feedforward=ENCODER_FEEDFORWARD,
                dropout=ENCODER_DROPOUT,
                batch_first=True,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(width, num_classes)
        self.window_frames = window_frames

    def encode(self, pose: torch.Tensor, visual: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        del visual, evidence
        if pose.ndim != 3 or pose.shape[1:] != (self.window_frames, POSE_DIM):
            raise ValueError(f"TCNTE exige janelas [B, {self.window_frames}, {POSE_DIM}]")
        features = self.tcn(tcnte_input(pose).transpose(1, 2)).transpose(1, 2)
        # A janela já termina em t; atenção e média ficam restritas a ela.
        encoded = self.encoder(features + self.positional)
        return encoded.mean(dim=1)

    def forward(self, pose: torch.Tensor, visual: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(pose, visual, evidence))


class FocalLoss(nn.Module):
    """FL = -α_c (1 - p_t)^γ log p_t, com α normalizado para somar 1 (Lin et al., 2017)."""

    def __init__(self, class_weights: torch.Tensor, gamma: float = FOCAL_GAMMA) -> None:
        super().__init__()
        if class_weights.ndim != 1 or bool((class_weights < 0).any()) or float(class_weights.sum()) <= 0:
            raise ValueError("pesos de classe devem ser 1-D, não negativos e com soma positiva")
        self.alpha: torch.Tensor
        self.register_buffer("alpha", class_weights / class_weights.sum())
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probability = F.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
        probability = log_probability.exp()
        alpha = self.alpha[target]
        return (-alpha * (1 - probability) ** self.gamma * log_probability).mean()
