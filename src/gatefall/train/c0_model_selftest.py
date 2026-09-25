"""Selftest sintético do `C0FusionClassifier` (`c0_model.py`). Não toca em dados reais."""

import sys

import torch

from gatefall.config import NUM_CLASSES, WINDOW_FRAMES
from gatefall.train.c0_model import C0FusionClassifier

_POSE_DIM = 134
_VISUAL_DIM = 10
_PROJECTION_DIM = 128
_FUSED_DIM = 256


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _build_model() -> C0FusionClassifier:
    return C0FusionClassifier(
        channels=[32, 32, 32],
        kernel_size=3,
        dilations=[1, 2, 4],
        dropout=0.3,
        num_classes=NUM_CLASSES,
    )


def check_pose_projection_shape() -> bool:
    torch.manual_seed(0)
    model = _build_model()
    model.eval()
    x_pose = torch.randn(4, WINDOW_FRAMES, _POSE_DIM, dtype=torch.float32)
    with torch.no_grad():
        projected = model.e_p(x_pose)
    ok = tuple(projected.shape) == (4, WINDOW_FRAMES, _PROJECTION_DIM)
    return _check(
        f"E_P projeta [B,{WINDOW_FRAMES},{_POSE_DIM}] -> "
        f"[B,{WINDOW_FRAMES},{_PROJECTION_DIM}] (obtido {tuple(projected.shape)})",
        ok,
    )


def check_visual_projection_shape() -> bool:
    torch.manual_seed(1)
    model = _build_model()
    model.eval()
    x_visual = torch.randn(4, WINDOW_FRAMES, _VISUAL_DIM, dtype=torch.float32)
    with torch.no_grad():
        projected = model.e_v(x_visual)
    ok = tuple(projected.shape) == (4, WINDOW_FRAMES, _PROJECTION_DIM)
    return _check(
        f"E_V projeta [B,{WINDOW_FRAMES},{_VISUAL_DIM}] -> "
        f"[B,{WINDOW_FRAMES},{_PROJECTION_DIM}] (obtido {tuple(projected.shape)})",
        ok,
    )


def check_fused_representation_is_256_dim() -> bool:
    torch.manual_seed(2)
    model = _build_model()
    model.eval()
    x_pose = torch.randn(2, WINDOW_FRAMES, _POSE_DIM, dtype=torch.float32)
    x_visual = torch.randn(2, WINDOW_FRAMES, _VISUAL_DIM, dtype=torch.float32)
    with torch.no_grad():
        projected_pose = model.e_p(x_pose)
        projected_visual = model.e_v(x_visual)
    fused_dim = projected_pose.shape[-1] + projected_visual.shape[-1]
    ok = fused_dim == _FUSED_DIM
    return _check(
        f"representação fundida (concat E_P || E_V) tem {_FUSED_DIM} dimensões "
        f"(obtido {fused_dim})",
        ok,
    )


def check_forward_logits_shape() -> bool:
    torch.manual_seed(3)
    model = _build_model()
    model.eval()
    x_pose = torch.randn(4, WINDOW_FRAMES, _POSE_DIM, dtype=torch.float32)
    x_visual = torch.randn(4, WINDOW_FRAMES, _VISUAL_DIM, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_pose, x_visual)
    ok = tuple(logits.shape) == (4, NUM_CLASSES) and logits.dtype == torch.float32
    return _check(
        f"forward(x_pose, x_visual) produz logits [4, {NUM_CLASSES}] float32 "
        f"(obtido shape={tuple(logits.shape)}, dtype={logits.dtype})",
        ok,
    )


def check_forward_rejects_wrong_temporal_length() -> bool:
    torch.manual_seed(4)
    model = _build_model()
    model.eval()
    good_pose = torch.randn(2, WINDOW_FRAMES, _POSE_DIM, dtype=torch.float32)
    good_visual = torch.randn(2, WINDOW_FRAMES, _VISUAL_DIM, dtype=torch.float32)
    bad_pose = torch.randn(2, WINDOW_FRAMES + 1, _POSE_DIM, dtype=torch.float32)
    bad_visual = torch.randn(2, WINDOW_FRAMES - 1, _VISUAL_DIM, dtype=torch.float32)

    raised_on_pose = False
    try:
        with torch.no_grad():
            model(bad_pose, good_visual)
    except ValueError:
        raised_on_pose = True

    raised_on_visual = False
    try:
        with torch.no_grad():
            model(good_pose, bad_visual)
    except ValueError:
        raised_on_visual = True

    ok = raised_on_pose and raised_on_visual
    return _check(
        f"forward rejeita comprimento temporal != {WINDOW_FRAMES} em x_pose ou x_visual",
        ok,
    )


def check_parameter_count_deterministic_and_all_trainable() -> bool:
    torch.manual_seed(5)
    model_a = _build_model()
    torch.manual_seed(5)
    model_b = _build_model()

    count_a = sum(p.numel() for p in model_a.parameters())
    count_b = sum(p.numel() for p in model_b.parameters())
    trainable_count_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    all_require_grad = all(p.requires_grad for p in model_a.parameters())

    ok = count_a > 0 and count_a == count_b and trainable_count_a == count_a and all_require_grad
    return _check(
        "contagem de parâmetros é determinística entre gêmeos construídos com "
        "a mesma seed, positiva, e nenhum parâmetro está presente sem "
        "requires_grad=True (nada congelado-mas-presente)",
        ok,
    )


def run_c0_model_selftest() -> bool:
    checks = [
        check_pose_projection_shape(),
        check_visual_projection_shape(),
        check_fused_representation_is_256_dim(),
        check_forward_logits_shape(),
        check_forward_rejects_wrong_temporal_length(),
        check_parameter_count_deterministic_and_all_trainable(),
    ]
    ok = all(checks)
    if not ok:
        print("\nc0 model selftest FALHOU", file=sys.stderr)
    else:
        print("\nc0 model selftest OK: todas as checagens passaram")
    return ok
