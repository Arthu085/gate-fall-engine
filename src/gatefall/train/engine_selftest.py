"""Selftest sintético do determinismo de treino (`engine.py`). Não toca em dados reais."""

import os
import sys

import torch

from gatefall.train.engine import _configure_determinism


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_determinism_flags_enabled() -> bool:
    _configure_determinism(seed=42)
    ok = (
        torch.backends.cudnn.deterministic is True
        and torch.backends.cudnn.benchmark is False
        and torch.are_deterministic_algorithms_enabled()
    )
    return _check(
        "_configure_determinism ativa cudnn.deterministic, desativa "
        "cudnn.benchmark e ativa use_deterministic_algorithms",
        ok,
    )


def check_cublas_workspace_config_set() -> bool:
    _configure_determinism(seed=42)
    ok = os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None
    return _check("CUBLAS_WORKSPACE_CONFIG está definido no ambiente do processo", ok)


def run_engine_selftest() -> bool:
    checks = [check_determinism_flags_enabled(), check_cublas_workspace_config_set()]
    ok = all(checks)
    if not ok:
        print("\nengine selftest FALHOU", file=sys.stderr)
    else:
        print("\nengine selftest OK: todas as checagens passaram")
    return ok
