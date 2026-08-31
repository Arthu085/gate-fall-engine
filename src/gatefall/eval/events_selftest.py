"""Selftest sintético do protocolo de eventos (`events.py`). Não toca em dados reais."""

import sys

import numpy as np

from gatefall.eval.alarm_protocol import AlarmProtocol
from gatefall.eval.events import (
    associate_events_and_alarms,
    detect_alarms_for_video,
    fall_events_for_video,
)

_PROTOCOL = AlarmProtocol(
    fall_label=1,
    fallen_label=2,
    positive_labels=[1, 2],
    trigger_consecutive=3,
    refractory_period_s=5.0,
    association_end_offset_s=2.0,
    fallback_association_uses_fall_end=True,
    eval_stride=1,
    target_fps=10.0,
    latency_decimal_places=1,
)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_no_trigger_below_threshold() -> bool:
    k_ends = np.array([0, 1, 2, 3], dtype=np.int64)
    preds = np.array([0, 1, 1, 0], dtype=np.int64)
    alarms = detect_alarms_for_video("video_a", k_ends, preds, _PROTOCOL)
    return _check(
        "run de apenas 2 positivos consecutivos não gera alarme",
        alarms == [],
    )


def check_single_alarm_from_long_run() -> bool:
    k_ends = np.arange(8, dtype=np.int64)
    preds = np.array([0, 0, 1, 1, 1, 1, 1, 0], dtype=np.int64)
    alarms = detect_alarms_for_video("video_b", k_ends, preds, _PROTOCOL)
    ok = len(alarms) == 1 and alarms[0].trigger_k == 4
    return _check(
        "run longo de positivos colapsa em exatamente 1 alarme, disparado no "
        "k_end onde a contagem consecutiva primeiro atinge 3",
        ok,
    )


def check_refractory_suppresses_second_alarm() -> bool:
    # Runs positivos em k=[2,3,4] (dispara em k=4, t=0.4s), k=[6,7,8]
    # (dispararia em k=8, t=0.8s, dentro do refratário de 5s -> suprimido) e
    # k=[60,61,62] (dispara em k=62, t=6.2s, >=5s depois do 1o alarme -> soa).
    n = 63
    k_ends = np.arange(n, dtype=np.int64)
    preds = np.zeros(n, dtype=np.int64)
    preds[2:5] = 1
    preds[6:9] = 1
    preds[60:63] = 1

    alarms = detect_alarms_for_video("video_c", k_ends, preds, _PROTOCOL)
    ok = len(alarms) == 2 and alarms[0].trigger_k == 4 and alarms[1].trigger_k == 62
    return _check(
        "segundo run dentro do período refratário é suprimido e não é "
        "retentado depois; terceiro run após o refratário soa normalmente",
        ok,
    )


def check_false_alarm_outside_association_window() -> bool:
    k_ends = np.array([0, 1, 2, 3], dtype=np.int64)
    true_labels = np.array([0, 0, 0, 0], dtype=np.int64)
    preds = np.array([1, 1, 1, 0], dtype=np.int64)

    events = fall_events_for_video("video_d", k_ends, true_labels, _PROTOCOL)
    events_empty = events == []

    alarms = detect_alarms_for_video("video_d", k_ends, preds, _PROTOCOL)
    outcomes, false_alarms = associate_events_and_alarms(events, alarms)
    alarm_is_false = len(alarms) == 1 and false_alarms == alarms and outcomes == []

    return _check(
        "vídeo sem segmentos fall/fallen não gera eventos e o alarme "
        "correspondente entra em false_alarms, não associado a nenhum evento",
        events_empty and alarm_is_false,
    )


def run_events_selftest() -> bool:
    checks = [
        check_no_trigger_below_threshold(),
        check_single_alarm_from_long_run(),
        check_refractory_suppresses_second_alarm(),
        check_false_alarm_outside_association_window(),
    ]
    ok = all(checks)
    if not ok:
        print("\nevents selftest FALHOU", file=sys.stderr)
    else:
        print("\nevents selftest OK: todas as checagens passaram")
    return ok
