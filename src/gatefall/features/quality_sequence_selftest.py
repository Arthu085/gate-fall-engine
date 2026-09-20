"""Selftest sintético da montagem da sequência de qualidade
(`quality_sequence.py`). Não toca em dados reais nem em backbone: só verifica
o alinhamento das duas sequências e a ordem de canal persistida."""

import sys

import numpy as np

from gatefall.features.quality_sequence import (
    QualitySequenceError,
    assemble_quality_sequence,
)
from gatefall.features.quality_storage import QUALITY_CHANNEL_NAMES


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_channel_order_is_pose_then_visual() -> bool:
    q_pose = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    q_visual = np.linspace(1.0, 0.0, 7, dtype=np.float32)
    quality = assemble_quality_sequence(q_pose, q_visual)
    ok = (
        quality.shape == (7, 2)
        and quality.dtype == np.float32
        and bool(np.array_equal(quality[:, 0], q_pose))
        and bool(np.array_equal(quality[:, 1], q_visual))
        and QUALITY_CHANNEL_NAMES == ("q_pose", "q_visual")
    )
    return _check(
        "assemble_quality_sequence produz [K,2] float32 com canal 0 = q_pose e "
        "canal 1 = q_visual, na mesma ordem de QUALITY_CHANNEL_NAMES",
        ok,
    )


def check_mismatched_lengths_rejected() -> bool:
    raised = False
    try:
        assemble_quality_sequence(
            np.zeros(10, dtype=np.float32), np.zeros(9, dtype=np.float32)
        )
    except QualitySequenceError:
        raised = True
    return _check(
        "q_pose e q_visual com K diferente levantam QualitySequenceError",
        raised,
    )


def check_out_of_range_rejected() -> bool:
    raised_high = False
    try:
        assemble_quality_sequence(
            np.full(4, 1.5, dtype=np.float32), np.zeros(4, dtype=np.float32)
        )
    except QualitySequenceError:
        raised_high = True

    raised_nan = False
    try:
        assemble_quality_sequence(
            np.zeros(4, dtype=np.float32), np.full(4, np.nan, dtype=np.float32)
        )
    except QualitySequenceError:
        raised_nan = True

    return _check(
        "valor fora de [0,1] ou não finito em qualquer canal levanta "
        "QualitySequenceError",
        raised_high and raised_nan,
    )


def check_assembly_is_deterministic() -> bool:
    rng = np.random.default_rng(0)
    q_pose = rng.random(16).astype(np.float32)
    q_visual = rng.random(16).astype(np.float32)
    first = assemble_quality_sequence(q_pose, q_visual)
    second = assemble_quality_sequence(q_pose, q_visual)
    return _check(
        "assemble_quality_sequence é determinística: duas montagens da mesma "
        "entrada batem bit a bit",
        bool(np.array_equal(first, second)),
    )


def run_quality_sequence_selftest() -> bool:
    checks = [
        check_channel_order_is_pose_then_visual(),
        check_mismatched_lengths_rejected(),
        check_out_of_range_rejected(),
        check_assembly_is_deterministic(),
    ]
    ok = all(checks)
    if not ok:
        print("\nquality sequence selftest FALHOU", file=sys.stderr)
    else:
        print("\nquality sequence selftest OK: todas as checagens passaram")
    return ok
