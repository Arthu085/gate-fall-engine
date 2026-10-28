"""Métricas do estudo recalculadas a partir das predições por janela de cada avaliação."""

from collections.abc import Callable

import numpy as np

from gatefall.config import IGNORE_LABEL, NUM_CLASSES, TARGET_FPS
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.shared.events import split_event_report
from gatefall.train.shared.metrics import RESTRICTED_CLASSES

ClusterOf = Callable[[str], str]


def _arrays(predictions: list[dict]) -> tuple[list[str], list[int], np.ndarray, np.ndarray]:
    ids = [row["video_id"] for row in predictions]
    ends = [int(row["k_end"]) for row in predictions]
    true = np.array([row["label"] for row in predictions], dtype=np.int64)
    pred = np.array([row["prediction"] for row in predictions], dtype=np.int64)
    return ids, ends, true, pred


def confusion(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    keep = true != IGNORE_LABEL
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (true[keep], pred[keep]), 1)
    return matrix


def per_class(matrix: np.ndarray, label_names: tuple[str, ...]) -> list[dict]:
    rows = []
    for index, name in enumerate(label_names):
        tp = int(matrix[index, index]); support = int(matrix[index].sum()); predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class_id": index, "class": name, "support": support, "predicted": predicted,
                     "precision": precision, "recall": recall, "f1": f1})
    return rows


def supported_classes(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int(index) for index in np.flatnonzero(matrix.sum(axis=1) > 0))


def macro_f1(matrix: np.ndarray, classes: tuple[int, ...]) -> float:
    """Média simples do F1 sobre um conjunto de classes fixado antes de olhar as predições."""
    if not classes:
        raise ValueError("macro-F1 exige ao menos uma classe com suporte")
    rows = per_class(matrix, tuple(str(index) for index in range(NUM_CLASSES)))
    return float(np.mean([rows[index]["f1"] for index in classes]))


def events(predictions: list[dict]) -> dict:
    ids, ends, true, pred = _arrays(predictions)
    report = split_event_report(ids, ends, true.tolist(), pred.tolist(), BASELINE_A_ALARM_PROTOCOL,
                                len(ids), len(ids), int(np.sum(true != IGNORE_LABEL)), round_latency=False)
    total = report["n_fall_events"]; detected = report["n_detected_events"]
    during = report["n_events_detected_in_fall"]
    latencies = sorted(report["latency_seconds"]["per_event"])
    # Perdas entram como latência infinita: a mediana sobre todos os eventos só
    # existe quando mais da metade foi detectada.
    censored = latencies + [float("inf")] * (total - detected)
    median_all = float(np.median(censored)) if total and detected * 2 > total else None
    return {
        "n_events": total, "n_detected": detected, "n_missed": report["n_missed_events"],
        "sensitivity": detected / total if total else None,
        "n_detected_during_fall": during, "n_detected_only_after_fall": detected - during,
        "sensitivity_during_fall": during / total if total else None,
        "sensitivity_only_after_fall": (detected - during) / total if total else None,
        "latency_detected_s": {"n": len(latencies), "mean": float(np.mean(latencies)) if latencies else None,
                               "median": float(np.median(latencies)) if latencies else None,
                               "per_event": latencies},
        "latency_median_all_events_s": median_all, "latency_missed_events": total - detected,
        "false_alarms": {"n": report["n_false_alarms"], "n_pre_fall": report["n_pre_fall_false_alarms"],
                         "hours_total": report["total_video_time_hours"],
                         "hours_labeled": report["labeled_time_hours"],
                         "per_hour": report["false_alarms_per_hour"] if report["total_video_time_hours"] else None,
                         "per_hour_labeled": report["false_alarms_per_hour_labeled_time"]
                         if report["labeled_time_hours"] else None,
                         "denominator": "horas de vídeo avaliadas (janelas de passo 1 / 10 fps)"},
        "n_alarms": report["n_alarms_total"],
    }


def evaluation_metrics(report: dict, label_names: tuple[str, ...]) -> dict:
    _, _, true, pred = _arrays(report["predictions"])
    matrix = confusion(true, pred)
    classes = supported_classes(matrix)
    return {
        "per_class": per_class(matrix, label_names),
        "supported_classes": list(classes),
        "macro_f1_supported": macro_f1(matrix, classes),
        "macro_f1_restricted8": macro_f1(matrix, tuple(RESTRICTED_CLASSES)),
        "labeled_windows": int(matrix.sum()), "ignored_windows": int(np.sum(true == IGNORE_LABEL)),
        "events": events(report["predictions"]),
        "timing": report.get("timing"),
    }


def cluster_statistics(predictions: list[dict], cluster_of: ClusterOf) -> dict[str, dict]:
    """Estatísticas suficientes por cluster (sujeito ou gravação) para o bootstrap."""
    groups: dict[str, list[dict]] = {}
    for row in predictions:
        groups.setdefault(cluster_of(row["video_id"]), []).append(row)
    result = {}
    for cluster, rows in sorted(groups.items()):
        _, _, true, pred = _arrays(rows)
        summary = events(rows)
        result[cluster] = {"confusion": confusion(true, pred), "events": summary["n_events"],
                           "detected": summary["n_detected"], "during": summary["n_detected_during_fall"],
                           "false_alarms": summary["false_alarms"]["n"], "hours": len(rows) / TARGET_FPS / 3600}
    return result
