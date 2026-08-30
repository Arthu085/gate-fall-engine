"""TCN dilatada causal para classificação de janelas de pose do GateFall."""

import torch
from torch import nn

from gatefall.config import NUM_CLASSES, WINDOW_FRAMES


def receptive_field(kernel_size: int, dilations: list[int], convs_per_block: int = 2) -> int:
    # Campo receptivo de uma pilha de blocos residuais causais (Bai, Kolter e
    # Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent
    # Networks for Sequence Modeling", arXiv:1803.01271, eq. 4): cada bloco
    # contribui (kernel_size - 1) * dilation por convolução causal.
    return 1 + convs_per_block * (kernel_size - 1) * sum(dilations)


class _CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self._padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self._padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self._padding > 0:
            out = out[:, :, : -self._padding]
        return out


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + residual)


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: list[int],
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]
        if len(dilations) != len(channels):
            raise ValueError(
                f"dilations ({len(dilations)}) e channels ({len(channels)}) devem "
                "ter o mesmo comprimento"
            )
        blocks: list[nn.Module] = []
        in_channels = input_dim
        for out_channels, dilation in zip(channels, dilations):
            blocks.append(
                _TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: list[int],
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.encoder = TCNEncoder(input_dim, channels, kernel_size, dilations, dropout)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, WINDOW_FRAMES, input_dim] -> [B, input_dim, T] para Conv1d.
        assert x.shape[1] == WINDOW_FRAMES
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        last_timestep = encoded[:, :, -1]
        return self.classifier(last_timestep)
