"""Controles de fusão e gate com restrições explícitas de sinais."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from gatefall.config import NUM_CLASSES, WINDOW_FRAMES
from gatefall.experiments.physics.kinematics import POSE_DIM
from gatefall.train.shared.tcn import TCNEncoder

FUSIONS = ("pose", "concat", "constant", "quality_concat", "learned", "monotonic", "physics")
TRAINABLE_GATES = ("learned", "monotonic", "physics")


class EvidenceGate(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in (*TRAINABLE_GATES, "constant"):
            raise ValueError(f"gate desconhecido: {mode}")
        self.mode = mode
        if mode == "learned":
            self.linear = nn.Linear(2, 1)
        elif mode in ("monotonic", "physics"):
            initial = [math.log(math.expm1(1.0))] * 2
            if mode == "physics":
                initial += [math.log(math.expm1(.1))] * 2
            self.raw_weights = nn.Parameter(torch.tensor(initial))
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.shape[-1] != 8:
            raise ValueError("gate exige o esquema de evidência [K,8]")
        if self.mode == "constant":
            return torch.full_like(evidence[..., :1], .5)
        if self.mode == "learned":
            return torch.sigmoid(self.linear(evidence[..., :2]))
        weights = F.softplus(self.raw_weights)
        score = self.bias + weights[0] * evidence[..., 0] - weights[1] * evidence[..., 1]
        if self.mode == "physics":
            score = (score - weights[2] * evidence[..., 2] * evidence[..., 4]
                     + weights[3] * evidence[..., 3] * evidence[..., 5])
        return torch.sigmoid(score).unsqueeze(-1)


class FusionClassifier(nn.Module):
    def __init__(self, visual_dim: int, fusion: str) -> None:
        super().__init__()
        if fusion not in FUSIONS or (fusion != "pose" and visual_dim <= 0):
            raise ValueError("modo de fusão ou dimensão visual inválidos")
        self.fusion = fusion
        self.visual_dim = visual_dim
        if fusion == "pose":
            input_dim = POSE_DIM
        else:
            self.e_p = nn.Sequential(nn.Linear(POSE_DIM, 128), nn.LayerNorm(128), nn.ReLU())
            self.e_v = nn.Sequential(nn.Linear(visual_dim, 128), nn.LayerNorm(128), nn.ReLU())
            input_dim = 256 + (8 if fusion == "quality_concat" else 0)
        self.gate = EvidenceGate(fusion) if fusion in (*TRAINABLE_GATES, "constant") else None
        self.encoder = TCNEncoder(input_dim, [32, 32, 32], 3, [1, 2, 4], .3)
        self.classifier = nn.Linear(32, NUM_CLASSES)

    def encode(self, pose: torch.Tensor, visual: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        if (pose.ndim != 3 or pose.shape[1:] != (WINDOW_FRAMES, POSE_DIM)
                or visual.shape != (*pose.shape[:2], self.visual_dim)
                or evidence.shape != (*pose.shape[:2], 8)):
            raise ValueError("janelas incompatíveis com pose138/evidência8/janela24")
        if self.fusion == "pose":
            fused = pose
        else:
            # Aplicar presença depois da projeção evita que o viés represente uma modalidade ausente.
            p = self.e_p(pose) * evidence[..., 6:7]
            v = self.e_v(visual) * evidence[..., 7:8]
            if self.gate is not None:
                g = self.gate(evidence)
                p, v = g * p, (1 - g) * v
            fused = torch.cat((p, v), dim=-1)
            if self.fusion == "quality_concat":
                fused = torch.cat((fused, evidence), dim=-1)
        encoded = self.encoder(fused.transpose(1, 2))
        return encoded[:, :, -1]

    def forward(self, pose: torch.Tensor, visual: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(pose, visual, evidence))


def ordering_loss(
    clean: torch.Tensor, pose_corrupt: torch.Tensor, visual_corrupt: torch.Tensor,
    pose_changed: torch.Tensor, visual_changed: torch.Tensor, margin: float,
) -> torch.Tensor:
    if not math.isfinite(margin) or not 0 < margin < 1:
        raise ValueError("margem deve estar em (0,1)")
    p_mask = pose_changed.to(clean.dtype).unsqueeze(-1)
    v_mask = visual_changed.to(clean.dtype).unsqueeze(-1)
    p_loss = F.relu(margin + pose_corrupt - clean) * p_mask
    v_loss = F.relu(margin + clean - visual_corrupt) * v_mask
    return (p_loss.sum() + v_loss.sum()) / (p_mask.sum() + v_mask.sum()).clamp_min(1)
