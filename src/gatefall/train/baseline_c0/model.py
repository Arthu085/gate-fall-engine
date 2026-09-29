"""Classificador C0 de fusão por concatenação."""

from gatefall.config import NUM_CLASSES
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.train.shared.concat_model import ConcatFusionClassifier
from gatefall.train.shared.concat_model import FUSED_DIM, POSE_DIM, PROJECTION_DIM

VISUAL_DIM = V_T_DIM


class C0FusionClassifier(ConcatFusionClassifier):
    def __init__(
        self,
        channels: list[int],
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(VISUAL_DIM, channels, kernel_size, dilations, dropout, num_classes)
