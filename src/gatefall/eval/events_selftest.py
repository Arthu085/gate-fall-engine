"""Selftest sintético do protocolo de eventos (`events.py`). Não toca em dados reais."""

import sys

import numpy as np

from gatefall.config import IGNORE_LABEL
from gatefall.eval.alarm_protocol import AlarmProtocol
from gatefall.eval.events import (
    associate_events_and_alarms,
    count_pre_fall_false_alarms,
    detect_alarms_for_video,
    fall_events_for_video,
    split_event_report,
    window_level_binary_metrics,
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
    pre_fall_diagnostic_window_s=1.0,
    pre_fall_alarms_count_as_false_alarms=True,
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
    outcomes, false_alarms = associate_events_and_alarms(events, alarms, _PROTOCOL)
    alarm_is_false = len(alarms) == 1 and false_alarms == alarms and outcomes == []

    return _check(
        "vídeo sem segmentos fall/fallen não gera eventos e o alarme "
        "correspondente entra em false_alarms, não associado a nenhum evento",
        events_empty and alarm_is_false,
    )


def check_normal_association_path() -> bool:
    # fall em k=[2,3] (t=0.2..0.3s), fallen em k=[5,6,7] (t=0.5..0.7s) ->
    # start_time_s=0.2, association_end_time_s = 0.7 + 2.0 = 2.7s.
    # Alarme dispara em run positivo k=[10,11,12], t=1.2s (dentro da janela).
    n = 13
    k_ends = np.arange(n, dtype=np.int64)
    true_labels = np.zeros(n, dtype=np.int64)
    true_labels[2:4] = 1
    true_labels[5:8] = 2
    preds = np.zeros(n, dtype=np.int64)
    preds[10:13] = 1

    events = fall_events_for_video("video_e", k_ends, true_labels, _PROTOCOL)
    alarms = detect_alarms_for_video("video_e", k_ends, preds, _PROTOCOL)
    outcomes, false_alarms = associate_events_and_alarms(events, alarms, _PROTOCOL)

    expected_latency = 1.0
    ok = (
        len(outcomes) == 1
        and outcomes[0].detected
        and outcomes[0].latency_s == expected_latency
        and false_alarms == []
    )
    return _check(
        "caminho normal de associação: evento fall->fallen com alarme dentro "
        "da janela é detected=True com latency_s exato calculado à mão",
        ok,
    )


def check_fallback_association_path() -> bool:
    # fall em k=[2,3,4] (t=0.2..0.4s), sem fallen seguinte.
    # fallback: association_end_time_s = fall.end_k/fps + offset = 0.4+2.0=2.4s.
    # Alarme dispara em run positivo k=[8,9,10], t=1.0s (dentro do fallback).
    n = 11
    k_ends = np.arange(n, dtype=np.int64)
    true_labels = np.zeros(n, dtype=np.int64)
    true_labels[2:5] = 1
    preds = np.zeros(n, dtype=np.int64)
    preds[8:11] = 1

    events_fallback_on = fall_events_for_video("video_f", k_ends, true_labels, _PROTOCOL)
    alarms = detect_alarms_for_video("video_f", k_ends, preds, _PROTOCOL)
    outcomes, _false_alarms = associate_events_and_alarms(events_fallback_on, alarms, _PROTOCOL)
    fallback_detected = (
        len(events_fallback_on) == 1
        and not events_fallback_on[0].has_following_fallen
        and len(outcomes) == 1
        and outcomes[0].detected
    )

    protocol_no_fallback = AlarmProtocol(
        fall_label=_PROTOCOL.fall_label,
        fallen_label=_PROTOCOL.fallen_label,
        positive_labels=_PROTOCOL.positive_labels,
        trigger_consecutive=_PROTOCOL.trigger_consecutive,
        refractory_period_s=_PROTOCOL.refractory_period_s,
        association_end_offset_s=_PROTOCOL.association_end_offset_s,
        fallback_association_uses_fall_end=False,
        eval_stride=_PROTOCOL.eval_stride,
        target_fps=_PROTOCOL.target_fps,
        latency_decimal_places=_PROTOCOL.latency_decimal_places,
        pre_fall_diagnostic_window_s=_PROTOCOL.pre_fall_diagnostic_window_s,
        pre_fall_alarms_count_as_false_alarms=_PROTOCOL.pre_fall_alarms_count_as_false_alarms,
    )
    raised = False
    try:
        fall_events_for_video("video_f", k_ends, true_labels, protocol_no_fallback)
    except ValueError:
        raised = True

    return _check(
        "fallback_association_uses_fall_end=True usa a janela de fallback e "
        "detecta o evento; =False levanta exceção em vez de degradar em "
        "silêncio",
        fallback_detected and raised,
    )


def check_alarm_before_event_start_is_false_alarm() -> bool:
    # fall em k=[5,6] (start_time_s=0.5s), sem fallen seguinte.
    # Alarme dispara em run positivo k=[0,1,2], t=0.2s, estritamente antes do
    # início do evento -> não deve casar com o evento.
    n = 9
    k_ends = np.arange(n, dtype=np.int64)
    true_labels = np.zeros(n, dtype=np.int64)
    true_labels[5:7] = 1
    preds = np.zeros(n, dtype=np.int64)
    preds[0:3] = 1

    events = fall_events_for_video("video_g", k_ends, true_labels, _PROTOCOL)
    alarms = detect_alarms_for_video("video_g", k_ends, preds, _PROTOCOL)
    outcomes, false_alarms = associate_events_and_alarms(events, alarms, _PROTOCOL)

    ok = (
        len(alarms) == 1
        and false_alarms == alarms
        and len(outcomes) == 1
        and not outcomes[0].detected
    )
    return _check(
        "alarme disparado antes do start_time_s do evento conta como "
        "false_alarm, não é associado ao evento",
        ok,
    )


def check_k_end_discontinuity_breaks_run() -> bool:
    # Grade com um gap entre k=1 e k=3 (falta k=2): k_ends=[0,1,3,4,5,6],
    # todos positivos. Se a contiguidade fosse adjacência de array (posição
    # no vetor) em vez de adjacência na grade k_end, os 3 primeiros elementos
    # (k=0,1,3) contariam como 3 consecutivos e disparariam em k=3. Com a
    # contiguidade correta (k_end == prev_k + 1), o gap 1->3 quebra o run:
    # ele só reacumula 3 consecutivos reais em k=3,4,5, disparando em k=5.
    k_ends = np.array([0, 1, 3, 4, 5, 6], dtype=np.int64)
    preds = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)

    alarms = detect_alarms_for_video("video_h", k_ends, preds, _PROTOCOL)
    ok = len(alarms) == 1 and alarms[0].trigger_k == 5
    return _check(
        "descontinuidade em k_end (gap na grade, faltando k=2) quebra o run: "
        "o alarme só dispara em k=5, depois de reacumular 3 consecutivos "
        "reais após o gap, nunca em k=3 como aconteceria com adjacência de "
        "array",
        ok,
    )


def check_pre_fall_false_alarm_counter() -> bool:
    # fall em k=[5,6] (start_time_s=0.5s), sem fallen seguinte -> fallback:
    # association_end_time_s = fall.end_k/fps + offset = 6/10 + 2.0 = 2.6s.
    # Alarme 1 dispara em run k=[1,2,3], t=0.3s: dentro da janela de
    # diagnóstico pré-queda [0.5-1.0, 0.5) = [-0.5, 0.5), e antes do
    # start_time_s -> false_alarm, contado por count_pre_fall_false_alarms.
    # Alarme 2 dispara em run k=[58,59,60], t=6.0s: refratário >=5s desde o
    # alarme 1 (5.7s), fora da janela de diagnóstico e fora da janela de
    # associação -> false_alarm, mas NÃO contado (fora de [-0.5, 0.5)).
    n = 61
    k_ends = np.arange(n, dtype=np.int64)
    true_labels = np.zeros(n, dtype=np.int64)
    true_labels[5:7] = 1
    preds = np.zeros(n, dtype=np.int64)
    preds[1:4] = 1
    preds[58:61] = 1

    events = fall_events_for_video("video_i", k_ends, true_labels, _PROTOCOL)
    alarms = detect_alarms_for_video("video_i", k_ends, preds, _PROTOCOL)
    outcomes, false_alarms = associate_events_and_alarms(events, alarms, _PROTOCOL)

    setup_ok = (
        len(events) == 1
        and len(alarms) == 2
        and len(outcomes) == 1
        and not outcomes[0].detected
        and len(false_alarms) == 2
    )

    n_counted = count_pre_fall_false_alarms(events, false_alarms, _PROTOCOL)

    return _check(
        "count_pre_fall_false_alarms conta só o alarme dentro da janela de "
        "diagnóstico pré-queda, não o alarme distante fora dela",
        setup_ok and n_counted == 1,
    )


def check_window_level_binary_metrics() -> bool:
    # 7 janelas: uma com true_label=IGNORE_LABEL (excluída), e entre as
    # demais 6: 2 TP, 1 FN, 2 TN, 1 FP (calculado à mão contra
    # positive_labels=[1, 2]).
    true_labels = np.array([1, -1, 0, 2, 0, 1, 0], dtype=np.int64)
    pred_labels = np.array([1, 0, 0, 0, 1, 2, 0], dtype=np.int64)

    sensitivity, specificity = window_level_binary_metrics(
        true_labels, pred_labels, _PROTOCOL, ignore_label=-1
    )

    expected_sensitivity = 2 / 3
    expected_specificity = 2 / 3

    ok = (
        abs(sensitivity - expected_sensitivity) < 1e-9
        and abs(specificity - expected_specificity) < 1e-9
    )
    return _check(
        "window_level_binary_metrics exclui janelas IGNORE_LABEL e calcula "
        "sensitivity/specificity exatas sobre tp/fn/tn/fp calculados à mão",
        ok,
    )


def check_split_event_report_labeled_time_rates() -> bool:
    # 61 janelas de um único vídeo, sem nenhum segmento fall/fallen (todo
    # true_label é IGNORE_LABEL ou 0), então todo alarme vira false_alarm.
    # k=[0..4]: IGNORE_LABEL, com run positivo em k=[2,3,4] -> dispara em
    # k=4 (t=0.4s), dentro de um trecho não rotulado.
    # k=[5..54]: IGNORE_LABEL, sem positivos.
    # k=[55..60]: true_label=0 (rotulado), com run positivo em k=[58,59,60]
    # -> dispara em k=60 (t=6.0s, >=5s do 1o alarme, refratário não suprime),
    # dentro de um trecho rotulado.
    n = 61
    k_ends = list(range(n))
    true_labels = [IGNORE_LABEL] * 55 + [0] * 6
    pred_labels = [0] * n
    pred_labels[2] = 1
    pred_labels[3] = 1
    pred_labels[4] = 1
    pred_labels[58] = 1
    pred_labels[59] = 1
    pred_labels[60] = 1
    video_ids = ["video_j"] * n

    labeled_windows = sum(1 for label in true_labels if label != IGNORE_LABEL)
    total_windows = n
    usable_windows = n

    report = split_event_report(
        video_ids,
        k_ends,
        true_labels,
        pred_labels,
        _PROTOCOL,
        usable_windows,
        total_windows,
        labeled_windows,
    )

    labeled_time_hours = labeled_windows / _PROTOCOL.target_fps / 3600
    ok = (
        report["n_false_alarms"] == 2
        and report["labeled_windows"] == labeled_windows
        and abs(report["labeled_time_hours"] - labeled_time_hours) < 1e-12
        and report["false_alarms_per_hour"] != report["false_alarms_per_hour_labeled_time"]
    )
    return _check(
        "split_event_report: false_alarms_per_hour (denominador de tempo "
        "total) e false_alarms_per_hour_labeled_time (denominador de tempo "
        "rotulado, contando só false alarms em janelas rotuladas) divergem "
        "quando há um false alarm em trecho IGNORE_LABEL e outro em trecho "
        "rotulado; labeled_windows/labeled_time_hours aparecem com os "
        "valores passados",
        ok,
    )


def run_events_selftest() -> bool:
    checks = [
        check_no_trigger_below_threshold(),
        check_single_alarm_from_long_run(),
        check_refractory_suppresses_second_alarm(),
        check_false_alarm_outside_association_window(),
        check_normal_association_path(),
        check_fallback_association_path(),
        check_alarm_before_event_start_is_false_alarm(),
        check_k_end_discontinuity_breaks_run(),
        check_pre_fall_false_alarm_counter(),
        check_window_level_binary_metrics(),
        check_split_event_report_labeled_time_rates(),
    ]
    ok = all(checks)
    if not ok:
        print("\nevents selftest FALHOU", file=sys.stderr)
    else:
        print("\nevents selftest OK: todas as checagens passaram")
    return ok
