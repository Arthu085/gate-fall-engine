"""Detecção de alarmes por FSM de gatilho/refratário e associação alarme-evento de queda.

A FSM de gatilho usa uma comparação estrita (`==`, não `>=`) no ponto em que a
contagem de positivos consecutivos primeiro atinge `trigger_consecutive`: isso
faz uma sequência longa de predições positivas colapsar em exatamente um
candidato a alarme, em vez de um candidato por quadro depois do limiar.
"""

from dataclasses import dataclass

import numpy as np

from gatefall.config import IGNORE_LABEL
from gatefall.eval.alarm_protocol import AlarmProtocol


@dataclass(frozen=True)
class Segment:
    label: int
    start_k: int
    end_k: int


@dataclass(frozen=True)
class FallEvent:
    video_id: str
    start_time_s: float
    association_end_time_s: float
    has_following_fallen: bool


@dataclass(frozen=True)
class Alarm:
    video_id: str
    trigger_k: int
    trigger_time_s: float


@dataclass(frozen=True)
class EventOutcome:
    event: FallEvent
    detected: bool
    latency_s: float | None


def extract_label_segments(
    k_ends: np.ndarray, labels: np.ndarray, target_label: int
) -> list[Segment]:
    segments: list[Segment] = []
    start_k: int | None = None
    prev_k: int | None = None

    for k_end, label in zip(k_ends.tolist(), labels.tolist()):
        is_target = label == target_label
        contiguous = prev_k is not None and k_end == prev_k + 1

        if is_target and start_k is not None and contiguous:
            prev_k = k_end
            continue

        if start_k is not None:
            assert prev_k is not None
            segments.append(Segment(label=target_label, start_k=start_k, end_k=prev_k))
            start_k = None

        if is_target:
            start_k = k_end

        prev_k = k_end

    if start_k is not None:
        assert prev_k is not None
        segments.append(Segment(label=target_label, start_k=start_k, end_k=prev_k))

    return segments


def fall_events_for_video(
    video_id: str,
    k_ends: np.ndarray,
    true_labels: np.ndarray,
    protocol: AlarmProtocol,
) -> list[FallEvent]:
    fall_segments = extract_label_segments(k_ends, true_labels, protocol.fall_label)
    fallen_segments = extract_label_segments(k_ends, true_labels, protocol.fallen_label)

    events: list[FallEvent] = []
    for fall in fall_segments:
        following_fallen = min(
            (fallen for fallen in fallen_segments if fallen.start_k > fall.end_k),
            key=lambda fallen: fallen.start_k,
            default=None,
        )
        start_time_s = fall.start_k / protocol.target_fps
        if following_fallen is not None:
            association_end_time_s = (
                following_fallen.end_k / protocol.target_fps + protocol.association_end_offset_s
            )
            has_following_fallen = True
        elif protocol.fallback_association_uses_fall_end:
            association_end_time_s = (
                fall.end_k / protocol.target_fps + protocol.association_end_offset_s
            )
            has_following_fallen = False
        else:
            raise ValueError(
                f"video_id={video_id!r}: segmento fall em start_k={fall.start_k} "
                "não tem um segmento fallen seguinte e "
                "protocol.fallback_association_uses_fall_end é False — não há "
                "fallback de associação permitido para este evento"
            )

        events.append(
            FallEvent(
                video_id=video_id,
                start_time_s=start_time_s,
                association_end_time_s=association_end_time_s,
                has_following_fallen=has_following_fallen,
            )
        )

    return events


def detect_alarms_for_video(
    video_id: str,
    k_ends: np.ndarray,
    pred_labels: np.ndarray,
    protocol: AlarmProtocol,
) -> list[Alarm]:
    positive_labels = frozenset(protocol.positive_labels)

    alarms: list[Alarm] = []
    last_alarm_time_s: float | None = None
    consecutive = 0
    prev_k: int | None = None

    for k_end, pred in zip(k_ends.tolist(), pred_labels.tolist()):
        is_positive = pred in positive_labels
        contiguous = prev_k is not None and k_end == prev_k + 1

        if is_positive and consecutive > 0 and contiguous:
            consecutive += 1
        elif is_positive:
            consecutive = 1
        else:
            consecutive = 0

        if consecutive == protocol.trigger_consecutive:
            trigger_time_s = k_end / protocol.target_fps
            suppressed = (
                last_alarm_time_s is not None
                and trigger_time_s - last_alarm_time_s < protocol.refractory_period_s
            )
            if not suppressed:
                alarms.append(
                    Alarm(video_id=video_id, trigger_k=k_end, trigger_time_s=trigger_time_s)
                )
                last_alarm_time_s = trigger_time_s

        prev_k = k_end

    return alarms


def associate_events_and_alarms(
    events: list[FallEvent], alarms: list[Alarm], protocol: AlarmProtocol
) -> tuple[list[EventOutcome], list[Alarm]]:
    matched_alarm_ids: set[int] = set()
    outcomes: list[EventOutcome] = []

    for event in events:
        matches = [
            (alarm_id, alarm)
            for alarm_id, alarm in enumerate(alarms)
            if alarm.video_id == event.video_id
            and event.start_time_s <= alarm.trigger_time_s <= event.association_end_time_s
        ]
        if matches:
            _earliest_alarm_id, earliest_alarm = min(
                matches, key=lambda pair: pair[1].trigger_time_s
            )
            for alarm_id, _alarm in matches:
                matched_alarm_ids.add(alarm_id)
            latency_s = round(
                earliest_alarm.trigger_time_s - event.start_time_s,
                ndigits=protocol.latency_decimal_places,
            )
            outcomes.append(EventOutcome(event=event, detected=True, latency_s=latency_s))
        else:
            outcomes.append(EventOutcome(event=event, detected=False, latency_s=None))

    false_alarms = [
        alarm for alarm_id, alarm in enumerate(alarms) if alarm_id not in matched_alarm_ids
    ]
    return outcomes, false_alarms


def window_level_binary_metrics(
    true_labels: np.ndarray, pred_labels: np.ndarray, protocol: AlarmProtocol, ignore_label: int
) -> tuple[float, float]:
    mask = true_labels != ignore_label
    true_masked = true_labels[mask]
    pred_masked = pred_labels[mask]

    positive_labels = frozenset(protocol.positive_labels)
    true_positive_mask = np.isin(true_masked, list(positive_labels))
    pred_positive_mask = np.isin(pred_masked, list(positive_labels))

    tp = int(np.sum(true_positive_mask & pred_positive_mask))
    fn = int(np.sum(true_positive_mask & ~pred_positive_mask))
    tn = int(np.sum(~true_positive_mask & ~pred_positive_mask))
    fp = int(np.sum(~true_positive_mask & pred_positive_mask))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sensitivity), float(specificity)


def count_pre_fall_false_alarms(
    events: list[FallEvent], false_alarms: list[Alarm], protocol: AlarmProtocol
) -> int:
    count = 0
    for alarm in false_alarms:
        for event in events:
            if event.video_id != alarm.video_id:
                continue
            window_start = event.start_time_s - protocol.pre_fall_diagnostic_window_s
            if window_start <= alarm.trigger_time_s < event.start_time_s:
                count += 1
                break
    return count


def split_event_report(
    video_ids: list[str],
    k_ends: list[int],
    true_labels: list[int],
    pred_labels: list[int],
    protocol: AlarmProtocol,
    usable_windows: int,
    total_windows: int,
    labeled_windows: int,
) -> dict:
    assert protocol.eval_stride == 1, (
        f"protocol.eval_stride ({protocol.eval_stride}) != 1 — a contiguidade "
        "de k_end usada por extract_label_segments/detect_alarms_for_video "
        "(k_end == prev_k + 1) só é válida em stride 1"
    )

    grouped: dict[str, list[int]] = {}
    for index, video_id in enumerate(video_ids):
        grouped.setdefault(video_id, []).append(index)

    all_events: list[FallEvent] = []
    all_alarms: list[Alarm] = []
    true_label_by_video_k: dict[tuple[str, int], int] = {}

    for video_id, indices in grouped.items():
        order = sorted(indices, key=lambda i: k_ends[i])
        video_k_ends = np.array([k_ends[i] for i in order], dtype=np.int64)
        video_true = np.array([true_labels[i] for i in order], dtype=np.int64)
        video_pred = np.array([pred_labels[i] for i in order], dtype=np.int64)

        all_events.extend(fall_events_for_video(video_id, video_k_ends, video_true, protocol))
        all_alarms.extend(detect_alarms_for_video(video_id, video_k_ends, video_pred, protocol))

        for k_end, true_label in zip(video_k_ends.tolist(), video_true.tolist()):
            true_label_by_video_k[(video_id, k_end)] = true_label

    outcomes, false_alarms = associate_events_and_alarms(all_events, all_alarms, protocol)
    n_pre_fall_false_alarms = count_pre_fall_false_alarms(all_events, false_alarms, protocol)

    n_fall_events = len(all_events)
    n_detected_events = sum(1 for outcome in outcomes if outcome.detected)
    n_missed_events = n_fall_events - n_detected_events

    total_video_time_hours = total_windows / protocol.target_fps / 3600
    labeled_time_hours = labeled_windows / protocol.target_fps / 3600

    n_false_alarms = len(false_alarms)
    n_false_alarms_labeled = sum(
        1
        for alarm in false_alarms
        if true_label_by_video_k.get((alarm.video_id, alarm.trigger_k), IGNORE_LABEL)
        != IGNORE_LABEL
    )
    false_alarms_per_hour = (
        n_false_alarms / total_video_time_hours if total_video_time_hours > 0 else 0.0
    )
    false_alarms_per_hour_labeled_time = (
        n_false_alarms_labeled / labeled_time_hours if labeled_time_hours > 0 else 0.0
    )

    window_binary_sensitivity, window_binary_specificity = window_level_binary_metrics(
        np.array(true_labels, dtype=np.int64),
        np.array(pred_labels, dtype=np.int64),
        protocol,
        IGNORE_LABEL,
    )

    per_event_latency = [
        outcome.latency_s for outcome in outcomes if outcome.detected and outcome.latency_s is not None
    ]

    latency_seconds: dict[str, object] = {"per_event": per_event_latency}
    if per_event_latency:
        latency_seconds["mean"] = round(
            float(np.mean(per_event_latency)), protocol.latency_decimal_places
        )
        latency_seconds["median"] = round(
            float(np.median(per_event_latency)), protocol.latency_decimal_places
        )
    else:
        latency_seconds["mean"] = None
        latency_seconds["median"] = None

    return {
        "usable_windows": usable_windows,
        "total_windows": total_windows,
        "labeled_windows": labeled_windows,
        "total_video_time_hours": float(total_video_time_hours),
        "labeled_time_hours": float(labeled_time_hours),
        "n_fall_events": n_fall_events,
        "n_detected_events": n_detected_events,
        "n_missed_events": n_missed_events,
        "sensitivity": float(n_detected_events / n_fall_events) if n_fall_events else 0.0,
        "n_alarms_total": len(all_alarms),
        "n_false_alarms": n_false_alarms,
        "n_pre_fall_false_alarms": n_pre_fall_false_alarms,
        "false_alarms_per_hour": float(false_alarms_per_hour),
        "false_alarms_per_hour_labeled_time": float(false_alarms_per_hour_labeled_time),
        "window_binary_sensitivity": window_binary_sensitivity,
        "window_binary_specificity": window_binary_specificity,
        "latency_seconds": latency_seconds,
    }
