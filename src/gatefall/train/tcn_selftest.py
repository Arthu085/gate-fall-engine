"""Selftest sintético da TCN (`tcn.py`). Não toca em dados reais."""

import sys

import numpy as np
import torch

from gatefall.config import NUM_CLASSES, WINDOW_FRAMES
from gatefall.train.tcn import TCNClassifier, TCNEncoder, receptive_field


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_receptive_field_formula() -> bool:
    value = receptive_field(3, [1, 2, 4])
    return _check(f"receptive_field(3, [1, 2, 4]) == 29 (obtido {value})", value == 29)


def check_forward_pass_shape() -> bool:
    torch.manual_seed(0)
    model = TCNClassifier(input_dim=134, channels=[32, 32, 32], kernel_size=3, dilations=[1, 2, 4])
    model.eval()
    x = torch.randn(4, WINDOW_FRAMES, 134, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    ok = tuple(out.shape) == (4, NUM_CLASSES) and out.dtype == torch.float32
    return _check(
        f"forward em [4, {WINDOW_FRAMES}, 134] produz saída (4, {NUM_CLASSES}) float32 "
        f"(obtido shape={tuple(out.shape)}, dtype={out.dtype})",
        ok,
    )


def check_causal_no_leakage() -> bool:
    torch.manual_seed(1)
    encoder = TCNEncoder(input_dim=134, channels=[32, 32, 32], kernel_size=3, dilations=[1, 2, 4])
    encoder.eval()

    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.normal(size=(2, 134, WINDOW_FRAMES)).astype(np.float32))

    t = WINDOW_FRAMES // 2

    with torch.no_grad():
        out_before = encoder(x)

    x_perturbed = x.clone()
    x_perturbed[:, :, t + 1 :] += 100.0

    with torch.no_grad():
        out_after = encoder(x_perturbed)

    unchanged_prefix = bool(torch.allclose(out_before[:, :, : t + 1], out_after[:, :, : t + 1]))
    changed_suffix = bool(
        not torch.allclose(out_before[:, :, t + 1 :], out_after[:, :, t + 1 :])
    )
    return _check(
        "causalidade: perturbar quadros futuros (t+1..) não muda a saída da TCN em "
        "posições <= t, mas muda em posições futuras",
        unchanged_prefix and changed_suffix,
    )


def run_tcn_selftest() -> bool:
    checks = [
        check_receptive_field_formula(),
        check_forward_pass_shape(),
        check_causal_no_leakage(),
    ]
    ok = all(checks)
    if not ok:
        print("\ntcn selftest FALHOU", file=sys.stderr)
    else:
        print("\ntcn selftest OK: todas as checagens passaram")
    return ok
