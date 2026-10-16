"""Receitas opcionais de perda para a investigação não canônica de desbalanceamento."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gatefall.runs import REPOSITORY_ROOT


@dataclass(frozen=True)
class ImbalanceRecipe:
    name: str
    weight_power: float
    hard_scale: float

    def __post_init__(self) -> None:
        if not 0 < self.weight_power <= 1 or not 0 <= self.hard_scale <= 0.5:
            raise ValueError("expoente de peso ou intensidade de exemplos difíceis fora do limite")


RECIPES = {
    "baseline": ImbalanceRecipe("baseline", 1.0, 0.0),
    "soft": ImbalanceRecipe("soft", 0.5, 0.0),
    "hard": ImbalanceRecipe("hard", 1.0, 0.5),
    "combined": ImbalanceRecipe("combined", 0.5, 0.5),
}


def validate_imbalance_run_dir(path: Path) -> None:
    root = (REPOSITORY_ROOT / "runs/local/imbalance_experiment").resolve()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"receita experimental exige run dentro de {root}")


def class_weights(labels: np.ndarray, num_classes: int, recipe: ImbalanceRecipe) -> torch.Tensor:
    if labels.ndim != 1 or len(labels) == 0 or np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("rótulos de treino inválidos para os pesos de classe")
    counts = np.bincount(labels, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / np.power(counts[present], recipe.weight_power)
    return torch.from_numpy(weights)


def training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    recipe: ImbalanceRecipe | None,
    other_id: int,
    walk_id: int,
) -> torch.Tensor:
    base = criterion(logits, targets)
    if recipe is None or recipe.hard_scale == 0:
        return base
    predictions = logits.argmax(dim=1)
    hard = ((targets == other_id) & (predictions == walk_id)) | (
        (targets == walk_id) & (predictions == other_id)
    )
    if not bool(hard.any()):
        return base
    losses = F.cross_entropy(logits, targets, reduction="none")
    if criterion.weight is None:
        extra = losses[hard].sum() / targets.numel()
    else:
        sample_weights = criterion.weight[targets]
        extra = (losses[hard] * sample_weights[hard]).sum() / sample_weights.sum()
    return base + recipe.hard_scale * extra
