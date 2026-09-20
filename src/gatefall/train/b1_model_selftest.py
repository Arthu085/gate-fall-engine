"""Selftest sintético do `B1AdaptiveGateClassifier` (`b1_model.py`). Não toca
em dados reais: cobre shape e faixa do gate, a ponderação complementar entre
os dois ramos e o determinismo da construção."""

import sys

import torch

from gatefall.config import NUM_CLASSES, WINDOW_FRAMES
from gatefall.train.b1_model import (
    GATE_INPUT_DIM,
    AdaptiveGate,
    B1AdaptiveGateClassifier,
)

_POSE_DIM = 134
_VISUAL_DIM = 1536
_PROJECTION_DIM = 128
_FUSED_DIM = 256


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _build_model() -> B1AdaptiveGateClassifier:
    return B1AdaptiveGateClassifier(
        channels=[32, 32, 32],
        kernel_size=3,
        dilations=[1, 2, 4],
        dropout=0.3,
        num_classes=NUM_CLASSES,
    )


def check_gate_shape_and_closed_range() -> bool:
    torch.manual_seed(0)
    gate = AdaptiveGate()
    gate.eval()
    quality = torch.rand(4, WINDOW_FRAMES, GATE_INPUT_DIM, dtype=torch.float32)
    with torch.no_grad():
        g = gate(quality)
    # Faixa fechada: a sigmoide satura exatamente em 0.0 e 1.0 em float32.
    ok = (
        tuple(g.shape) == (4, WINDOW_FRAMES, 1)
        and g.dtype == torch.float32
        and bool(torch.all((g >= 0.0) & (g <= 1.0)))
    )
    return _check(
        f"gate produz um escalar por timestep [B,{WINDOW_FRAMES},1] em [0,1] a "
        "partir de [q_pose, q_visual]",
        ok,
    )


def check_gate_weights_are_complementary() -> bool:
    torch.manual_seed(1)
    gate = AdaptiveGate()
    gate.eval()
    quality = torch.rand(8, WINDOW_FRAMES, GATE_INPUT_DIM, dtype=torch.float32)
    with torch.no_grad():
        g = gate(quality)
    # g + (1 - g) é exato em float32 para todo g representável.
    exact = bool(torch.all(g + (1.0 - g) == 1.0))
    return _check(
        "peso da pose (g) e peso do visual (1-g) somam exatamente 1.0 em "
        "float32 em todo timestep",
        exact,
    )


def check_gate_direction_pose_versus_visual() -> bool:
    torch.manual_seed(2)
    model = _build_model()
    model.eval()
    x_pose = torch.randn(2, WINDOW_FRAMES, _POSE_DIM, dtype=torch.float32)
    x_visual = torch.randn(2, WINDOW_FRAMES, _VISUAL_DIM, dtype=torch.float32)
    x_quality = torch.rand(2, WINDOW_FRAMES, GATE_INPUT_DIM, dtype=torch.float32)

    with torch.no_grad():
        projected_pose = model.e_p(x_pose)
        projected_visual = model.e_v(x_visual)

        # Saturação forçada da sigmoide: em float32, sigmoid(+200) == 1.0 e
        # sigmoid(-200) == 0.0 exatamente.
        model.gate.linear.weight.zero_()
        model.gate.linear.bias.fill_(200.0)
        g_high = model.gate(x_quality)
        fused_high = torch.cat(
            [g_high * projected_pose, (1.0 - g_high) * projected_visual], dim=-1
        )

        model.gate.linear.bias.fill_(-200.0)
        g_low = model.gate(x_quality)
        fused_low = torch.cat(
            [g_low * projected_pose, (1.0 - g_low) * projected_visual], dim=-1
        )

    visual_zeroed = bool(torch.all(fused_high[..., _PROJECTION_DIM:] == 0.0))
    pose_kept = bool(torch.allclose(fused_high[..., :_PROJECTION_DIM], projected_pose))
    pose_zeroed = bool(torch.all(fused_low[..., :_PROJECTION_DIM] == 0.0))
    visual_kept = bool(
        torch.allclose(fused_low[..., _PROJECTION_DIM:], projected_visual)
    )

    return _check(
        "g -> 1 zera a metade visual do vetor fundido e preserva a de pose; "
        "g -> 0 zera a metade de pose e preserva a visual (g pondera a POSE)",
        visual_zeroed and pose_kept and pose_zeroed and visual_kept,
    )


def check_fused_representation_is_256_dim() -> bool:
    torch.manual_seed(3)
    model = _build_model()
    model.eval()
    x_pose = torch.randn(2, WINDOW_FRAMES, _POSE_DIM, dtype=torch.float32)
    x_visual = torch.randn(2, WINDOW_FRAMES, _VISUAL_DIM, dtype=torch.float32)
    x_quality = torch.rand(2, WINDOW_FRAMES, GATE_INPUT_DIM, dtype=torch.float32)
    with torch.no_grad():
        g = model.gate(x_quality)
        fused = torch.cat(
            [g * model.e_p(x_pose), (1.0 - g) * model.e_v(x_visual)], dim=-1
        )
    ok = tuple(fused.shape) == (2, WINDOW_FRAMES, _FUSED_DIM)
    return _check(
        f"o gate não altera a dimensão fundida: a TCN continua recebendo "
        f"{_FUSED_DIM} dimensões por timestep (obtido {tuple(fused.shape)})",
        ok,
    )


def check_forward_logits_shape() -> bool:
    torch.manual_seed(4)
    model = _build_model()
    model.eval()
    x_pose = torch.randn(4, WINDOW_FRAMES, _POSE_DIM, dtype=torch.float32)
    x_visual = torch.randn(4, WINDOW_FRAMES, _VISUAL_DIM, dtype=torch.float32)
    x_quality = torch.rand(4, WINDOW_FRAMES, GATE_INPUT_DIM, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_pose, x_visual, x_quality)
    ok = tuple(logits.shape) == (4, NUM_CLASSES) and logits.dtype == torch.float32
    return _check(
        f"forward(x_pose, x_visual, x_quality) produz logits [4, {NUM_CLASSES}] "
        f"float32 (obtido shape={tuple(logits.shape)}, dtype={logits.dtype})",
        ok,
    )


def check_forward_rejects_wrong_shapes() -> bool:
    torch.manual_seed(5)
    model = _build_model()
    model.eval()
    good_pose = torch.randn(2, WINDOW_FRAMES, _POSE_DIM, dtype=torch.float32)
    good_visual = torch.randn(2, WINDOW_FRAMES, _VISUAL_DIM, dtype=torch.float32)
    good_quality = torch.rand(2, WINDOW_FRAMES, GATE_INPUT_DIM, dtype=torch.float32)

    cases = (
        (torch.randn(2, WINDOW_FRAMES + 1, _POSE_DIM), good_visual, good_quality),
        (good_pose, torch.randn(2, WINDOW_FRAMES - 1, _VISUAL_DIM), good_quality),
        (good_pose, good_visual, torch.rand(2, WINDOW_FRAMES + 1, GATE_INPUT_DIM)),
        (good_pose, good_visual, torch.rand(2, WINDOW_FRAMES, GATE_INPUT_DIM + 1)),
    )
    all_raised = True
    for pose, visual, quality in cases:
        raised = False
        try:
            with torch.no_grad():
                model(pose, visual, quality)
        except ValueError:
            raised = True
        all_raised = all_raised and raised

    return _check(
        f"forward rejeita comprimento temporal != {WINDOW_FRAMES} em qualquer "
        f"entrada e x_quality com número de canais != {GATE_INPUT_DIM}",
        all_raised,
    )


def check_construction_is_deterministic_and_all_trainable() -> bool:
    torch.manual_seed(6)
    model_a = _build_model()
    torch.manual_seed(6)
    model_b = _build_model()

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    identical_state = set(state_a) == set(state_b) and all(
        torch.equal(state_a[key], state_b[key]) for key in state_a
    )

    model_a.eval()
    model_b.eval()
    x_pose = torch.randn(3, WINDOW_FRAMES, _POSE_DIM, dtype=torch.float32)
    x_visual = torch.randn(3, WINDOW_FRAMES, _VISUAL_DIM, dtype=torch.float32)
    x_quality = torch.rand(3, WINDOW_FRAMES, GATE_INPUT_DIM, dtype=torch.float32)
    with torch.no_grad():
        identical_forward = torch.equal(
            model_a(x_pose, x_visual, x_quality),
            model_b(x_pose, x_visual, x_quality),
        )

    count_a = sum(p.numel() for p in model_a.parameters())
    trainable_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    gate_params = sum(p.numel() for p in model_a.gate.parameters())

    ok = (
        identical_state
        and identical_forward
        and count_a > 0
        and trainable_a == count_a
        and all(p.requires_grad for p in model_a.parameters())
        # nn.Linear(2, 1): dois pesos e um bias.
        and gate_params == GATE_INPUT_DIM + 1
    )
    return _check(
        "gêmeos construídos com a mesma seed têm state_dict e forward "
        "idênticos bit a bit, todos os parâmetros são treináveis e o gate "
        "acrescenta exatamente uma Linear(2,1)",
        ok,
    )


def run_b1_model_selftest() -> bool:
    checks = [
        check_gate_shape_and_closed_range(),
        check_gate_weights_are_complementary(),
        check_gate_direction_pose_versus_visual(),
        check_fused_representation_is_256_dim(),
        check_forward_logits_shape(),
        check_forward_rejects_wrong_shapes(),
        check_construction_is_deterministic_and_all_trainable(),
    ]
    ok = all(checks)
    if not ok:
        print("\nb1 model selftest FALHOU", file=sys.stderr)
    else:
        print("\nb1 model selftest OK: todas as checagens passaram")
    return ok
