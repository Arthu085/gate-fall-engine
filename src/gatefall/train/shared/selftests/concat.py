"""Caracterização do layout de checkpoints das armas de concatenação."""

from gatefall.train.shared.concat_model import ConcatFusionClassifier

_STATE_KEYS = (
    "e_p.0.weight", "e_p.0.bias", "e_p.1.weight", "e_p.1.bias",
    "e_v.0.weight", "e_v.0.bias", "e_v.1.weight", "e_v.1.bias",
    "encoder.blocks.0.conv1.conv.weight", "encoder.blocks.0.conv1.conv.bias",
    "encoder.blocks.0.conv2.conv.weight", "encoder.blocks.0.conv2.conv.bias",
    "encoder.blocks.0.downsample.weight", "encoder.blocks.0.downsample.bias",
    "encoder.blocks.1.conv1.conv.weight", "encoder.blocks.1.conv1.conv.bias",
    "encoder.blocks.1.conv2.conv.weight", "encoder.blocks.1.conv2.conv.bias",
    "encoder.blocks.2.conv1.conv.weight", "encoder.blocks.2.conv1.conv.bias",
    "encoder.blocks.2.conv2.conv.weight", "encoder.blocks.2.conv2.conv.bias",
    "classifier.weight", "classifier.bias",
)


def check_concat_checkpoint_layout(
    model: ConcatFusionClassifier, visual_dim: int, expected_params: int
) -> bool:
    state = model.state_dict()
    count = sum(parameter.numel() for parameter in model.parameters())
    ok = (
        tuple(state) == _STATE_KEYS
        and tuple(state["e_v.0.weight"].shape) == (128, visual_dim)
        and count == expected_params
    )
    print(f"[{'PASS' if ok else 'FAIL'}] layout e contagem do checkpoint de concatenação")
    return ok
