"""Classificador C1 com gate compartilhado e descritor SAM 3 de 10 canais."""

from gatefall.config import NUM_CLASSES
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.train.shared.gated_model import GatedFusionClassifier


class C1AdaptiveGateClassifier(GatedFusionClassifier):
    def __init__(
        self,
        channels: list[int],
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            channels, kernel_size, dilations, dropout, num_classes, visual_dim=V_T_DIM
        )
