"""Selftest sintético do determinismo de treino (`engine.py`). Não toca em dados reais."""

import os
import sys

import torch

from gatefall.train.engine import (
    _CUBLAS_DETERMINISTIC_WORKSPACE_CONFIGS,
    _CUBLAS_WORKSPACE_CONFIG_DEFAULT,
    _configure_determinism,
)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _restore_cublas_workspace_config(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous


def check_determinism_flags_enabled() -> bool:
    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG_DEFAULT
    try:
        _configure_determinism(seed=42)
        ok = (
            torch.backends.cudnn.deterministic is True
            and torch.backends.cudnn.benchmark is False
            and torch.are_deterministic_algorithms_enabled()
        )
    finally:
        _restore_cublas_workspace_config(previous)
    return _check(
        "_configure_determinism ativa cudnn.deterministic, desativa "
        "cudnn.benchmark e ativa use_deterministic_algorithms",
        ok,
    )


def check_cublas_workspace_config_default() -> bool:
    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    try:
        _configure_determinism(seed=42)
        ok = os.environ.get("CUBLAS_WORKSPACE_CONFIG") == _CUBLAS_WORKSPACE_CONFIG_DEFAULT
    finally:
        _restore_cublas_workspace_config(previous)
    return _check(
        "CUBLAS_WORKSPACE_CONFIG ausente vira o padrão do projeto "
        f"({_CUBLAS_WORKSPACE_CONFIG_DEFAULT})",
        ok,
    )


def check_cublas_workspace_config_rejects_unsupported_value() -> bool:
    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":1:1"
    try:
        raised = False
        try:
            _configure_determinism(seed=42)
        except ValueError:
            raised = True
        ok = raised
    finally:
        _restore_cublas_workspace_config(previous)
    return _check(
        "_configure_determinism recusa CUBLAS_WORKSPACE_CONFIG=:1:1 "
        f"(fora de {sorted(_CUBLAS_DETERMINISTIC_WORKSPACE_CONFIGS)})",
        ok,
    )


def run_engine_selftest() -> bool:
    checks = [
        check_determinism_flags_enabled(),
        check_cublas_workspace_config_default(),
        check_cublas_workspace_config_rejects_unsupported_value(),
    ]
    ok = all(checks)
    if not ok:
        print("\nengine selftest FALHOU", file=sys.stderr)
    else:
        print("\nengine selftest OK: todas as checagens passaram")
    return ok
