"""Selftest sintético do determinismo de treino (`engine.py`). Não toca em dados reais."""

import os
import sys

import numpy as np
import torch

from gatefall.train.b0_engine import _collect_labels as collect_b0_labels
from gatefall.train.b1_engine import _collect_labels as collect_b1_labels
from gatefall.train.c0_engine import _collect_labels as collect_c0_labels
from gatefall.train.engine import (
    _CUBLAS_DETERMINISTIC_WORKSPACE_CONFIGS,
    _CUBLAS_WORKSPACE_CONFIG_DEFAULT,
    _collect_labels as collect_pose_labels,
    configure_determinism,
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
        configure_determinism(seed=42)
        ok = (
            torch.backends.cudnn.deterministic is True
            and torch.backends.cudnn.benchmark is False
            and torch.are_deterministic_algorithms_enabled()
        )
    finally:
        _restore_cublas_workspace_config(previous)
    return _check(
        "configure_determinism ativa cudnn.deterministic, desativa "
        "cudnn.benchmark e ativa use_deterministic_algorithms",
        ok,
    )


def check_cublas_workspace_config_default() -> bool:
    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    try:
        configure_determinism(seed=42)
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
            configure_determinism(seed=42)
        except ValueError:
            raised = True
        ok = raised
    finally:
        _restore_cublas_workspace_config(previous)
    return _check(
        "configure_determinism recusa CUBLAS_WORKSPACE_CONFIG=:1:1 "
        f"(fora de {sorted(_CUBLAS_DETERMINISTIC_WORKSPACE_CONFIGS)})",
        ok,
    )


def check_legacy_sources_keep_item_fallback() -> bool:
    labels = (2, 0, 1)
    window = np.zeros((1, 1), dtype=np.float32)

    class PoseSource:
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(self, index: int) -> tuple[np.ndarray, int, object]:
            return window, labels[index], index

    class FusionSource:
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(
            self, index: int
        ) -> tuple[np.ndarray, np.ndarray, int, object]:
            return window, window, labels[index], index

    class GatedSource:
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(
            self, index: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, object]:
            return window, window, window, labels[index], index

    expected = np.array(labels, dtype=np.int64)
    fusion = FusionSource()
    return _check(
        "fontes sintéticas sem label_at mantêm a coleta por __getitem__",
        all(
            np.array_equal(observed, expected)
            for observed in (
                collect_pose_labels(PoseSource()),
                collect_b0_labels(fusion),
                collect_b1_labels(GatedSource()),
                collect_c0_labels(fusion),
            )
        ),
    )


def run_engine_selftest() -> bool:
    checks = [
        check_determinism_flags_enabled(),
        check_cublas_workspace_config_default(),
        check_cublas_workspace_config_rejects_unsupported_value(),
        check_legacy_sources_keep_item_fallback(),
    ]
    ok = all(checks)
    if not ok:
        print("\nengine selftest FALHOU", file=sys.stderr)
    else:
        print("\nengine selftest OK: todas as checagens passaram")
    return ok
