"""Bootstrap agrupado por sujeito (IC percentil) das métricas congeladas das armas.

Ferramenta independente de estágio, deliberadamente fora do pipeline padrão
(`gatefall.pipeline`) e da suíte de lifecycle dos avaliadores de eventos
(`gatefall.eval.baseline_{a,b0,b1,c0,c1}.cli`): não abre o lock, não escreve o
journal e não toca `alarm_protocol.yaml`/`event_metrics.json`/`metrics.json`/
`checkpoint.pt`. Reusa o validador de treino, os loaders de estatísticas e
features, o modelo e o caminho de inferência do avaliador de eventos de cada
arma (`load_event_evaluation`), roda essa inferência uma única vez por split
(val e test) e bootstrapa as predições/identidades já cacheadas — nunca
reexecuta o modelo por réplica.

Janelas de avaliação em `EVAL_STRIDE=1` se sobrepõem em 96% e não são
observações independentes; tratá-las como unidade de reamostragem infla
artificialmente a precisão do intervalo de confiança. Este módulo reamostra
por **sujeito** (unidade de cluster): cada réplica sorteia sujeitos com
reposição e arrasta consigo todos os vídeos e todas as janelas daquele
sujeito, preservando a estrutura de dependência dentro de cada vídeo/sujeito.

`analyze` produz o IC de um único run de qualquer arma (A, B0, B1, C0, C1).
`compare` produz o IC pareado de `adaptativa - baseline` exclusivamente para
B1 - B0 e C1 - C0: em cada réplica o mesmo sorteio ordenado de sujeitos é
aplicado aos dois runs, cada métrica é calculada separadamente em cada arma e
só então o delta é registrado. Variação entre seeds de treino é assunto de
`gatefall.eval.analysis.multiseed_summary`, nunca combinada aqui.

Este módulo nunca seleciona, ranqueia ou promove um modelo/protocolo:
`BASELINE_A_ALARM_PROTOCOL` permanece a única configuração congelada usada
pelos avaliadores de eventos. O bootstrap aqui produzido é estritamente
descritivo — apenas expõe incerteza em torno das métricas já congeladas, não
implementa nenhum teste de hipótese em nível de janela nem p-valor.
"""

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL
from gatefall.data.windowing import build_window_index
from gatefall.datasets import get_dataset
from gatefall.eval.analysis.multiseed_summary import (
    ARMS,
    RunConfig,
    validate_run_event_artifacts,
)
from gatefall.eval.baseline_a import cli as baseline_a_events
from gatefall.eval.baseline_b0 import cli as baseline_b0_events
from gatefall.eval.baseline_b1 import cli as baseline_b1_events
from gatefall.eval.baseline_c0 import cli as baseline_c0_events
from gatefall.eval.baseline_c1 import cli as baseline_c1_events
from gatefall.eval.shared.alarm_protocol import AlarmProtocol, BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.shared.events import split_event_report
from gatefall.eval.shared.orchestration import EventEvaluation
from gatefall.hashing import sha256_file
from gatefall.runs import REFERENCE_RUN_ROOT, default_run_dir_for_arm, validate_local_run_dir
from gatefall.train.shared.metrics import (
    BINARY_POSITIVE_LABELS,
    binary_projection_summary,
    restricted_macro_f1,
)

GROUPED_BOOTSTRAP_JSON_FILE = "grouped_bootstrap.json"
GROUPED_BOOTSTRAP_CSV_FILE = "grouped_bootstrap.csv"

DEFAULT_SEED = 42
DEFAULT_N_REPLICATES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95

SPLITS: tuple[str, ...] = ("val", "test")
DATASETS: tuple[str, ...] = ("le2i", "le2i-cv")

# Pares (adaptativa, baseline): o delta é sempre adaptativa - baseline.
PAIRED_COMPARISONS: tuple[tuple[str, str], ...] = (("B1", "B0"), ("C1", "C0"))
# Campos que identificam a arma em vez de descrever o contrato experimental;
# todo outro campo presente nas duas configs precisa coincidir no par.
ARM_IDENTITY_FIELDS = frozenset({"run_name", "arm", "trainable_param_count"})

CLASSIFICATION_METRICS: tuple[str, ...] = (
    "macro_f1_restricted",
    "precision",
    "recall",
    "specificity",
    "f1",
    "accuracy",
)
EVENT_METRICS: tuple[str, ...] = (
    "sensitivity",
    "fall_sensitivity",
    "fall_or_fallen_sensitivity",
    "detected_events_alarm_within_fall_rate",
    "false_alarms_per_hour",
    "false_alarms_per_hour_labeled_time",
    "window_binary_sensitivity",
    "window_binary_specificity",
    "latency_mean_s",
    "latency_median_s",
)
METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("classification", CLASSIFICATION_METRICS),
    ("events", EVENT_METRICS),
)

METHOD_NOTE = (
    "janelas de avaliação em EVAL_STRIDE=1 se sobrepõem fortemente e NÃO "
    "foram tratadas como amostras independentes; a unidade de reamostragem é "
    "o sujeito (cluster), com reposição, arrastando todos os vídeos e todas "
    "as janelas de cada sujeito sorteado"
)
PAIRED_METHOD_NOTE = (
    METHOD_NOTE
    + "; em cada split e réplica o mesmo sorteio ordenado de sujeitos (mesmas "
    "multiplicidades, mesmas posições de sorteio e logo os mesmos IDs "
    "virtuais de vídeo) é aplicado às duas armas, cada métrica é calculada "
    "separadamente em cada arma e só então delta = adaptativa - baseline; a "
    "réplica só é válida para a métrica quando ela é válida nas duas armas; o "
    "IC percentil vem dos deltas por réplica, não da subtração de limites "
    "independentes; não é teste de hipótese nem variação entre seeds de treino"
)

CSV_COLUMNS = [
    "split",
    "metric_group",
    "metric",
    "point_estimate",
    "ci_lower",
    "ci_upper",
    "valid_replicates",
    "undefined_replicates",
]
PAIRED_CSV_COLUMNS = [
    "split",
    "metric_group",
    "metric",
    "baseline_point_estimate",
    "adaptive_point_estimate",
    "delta_point_estimate",
    "delta_ci_lower",
    "delta_ci_upper",
    "valid_replicates",
    "undefined_replicates",
]

MetricValues = dict[str, tuple[float, bool]]


@dataclass(frozen=True)
class SplitPredictions:
    video_ids: list[str]
    k_ends: list[int]
    true_labels: list[int]
    pred_labels: list[int]
    usable_windows: int
    total_windows: int
    labeled_windows: int


@dataclass(frozen=True)
class RunPredictions:
    arm: str
    run_dir: Path
    config: RunConfig
    splits: dict[str, SplitPredictions]
    subject_maps: dict[str, dict[int, list[str]]]


@dataclass(frozen=True)
class _ReplicateSource:
    windows_by_video: dict[str, np.ndarray]
    k_ends: np.ndarray
    true_labels: np.ndarray
    pred_labels: np.ndarray


def build_subject_to_videos(frames: pd.DataFrame, split: str) -> dict[int, list[str]]:
    split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
    pairs = cast(
        pd.DataFrame, split_frames[["video_id", "subject"]].drop_duplicates()
    )

    subject_counts = cast(pd.Series, pairs.groupby("video_id")["subject"].nunique())
    offending_counts = cast(pd.Series, subject_counts[subject_counts > 1])
    offenders = sorted(offending_counts.index.tolist())
    if offenders:
        raise ValueError(
            f"split={split!r}: vídeo(s) mapeando para mais de um subject "
            f"distinto: {offenders}"
        )

    mapping: dict[int, list[str]] = {}
    for video_id, subject in zip(pairs["video_id"], pairs["subject"]):
        mapping.setdefault(int(subject), []).append(str(video_id))
    for subject in mapping:
        mapping[subject] = sorted(mapping[subject])
    return mapping


def _validate_subject_video_mapping(
    subject_to_videos: dict[int, list[str]], video_ids: Sequence[str], split: str
) -> None:
    mapped_videos = {video for videos in subject_to_videos.values() for video in videos}
    predicted_videos = set(video_ids)

    missing_from_map = sorted(predicted_videos - mapped_videos)
    missing_from_predictions = sorted(mapped_videos - predicted_videos)
    if missing_from_map or missing_from_predictions:
        raise ValueError(
            f"split={split!r}: mapeamento vídeo->subject inconsistente com "
            f"as predições; vídeo(s) nas predições ausentes do mapeamento: "
            f"{missing_from_map}; vídeo(s) no mapeamento ausentes das "
            f"predições: {missing_from_predictions}"
        )


def _index_windows_by_video(
    video_ids: Sequence[str], k_ends: Sequence[int]
) -> dict[str, np.ndarray]:
    positions = pd.DataFrame(
        {
            "video_id": list(video_ids),
            "k_end": list(k_ends),
            "position": np.arange(len(video_ids)),
        }
    )
    result: dict[str, np.ndarray] = {}
    for video_id, group in positions.groupby("video_id", sort=False):
        ordered = cast(pd.DataFrame, group.sort_values("k_end"))
        result[str(video_id)] = ordered["position"].to_numpy()
    return result


def _build_replicate_arrays(
    drawn_subjects: np.ndarray,
    subject_to_videos: dict[int, list[str]],
    windows_by_video: dict[str, np.ndarray],
    k_ends: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    video_id_parts: list[np.ndarray] = []
    k_end_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []

    for draw_position, subject in enumerate(drawn_subjects.tolist()):
        for video_id in subject_to_videos[int(subject)]:
            idx = windows_by_video[video_id]
            n = len(idx)
            if n == 0:
                continue
            virtual_video_id = f"{video_id}__draw{draw_position:06d}"
            video_id_parts.append(np.full(n, virtual_video_id, dtype=object))
            k_end_parts.append(k_ends[idx])
            true_parts.append(true_labels[idx])
            pred_parts.append(pred_labels[idx])

    if not video_id_parts:
        return (
            np.array([], dtype=object),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    return (
        np.concatenate(video_id_parts),
        np.concatenate(k_end_parts),
        np.concatenate(true_parts),
        np.concatenate(pred_parts),
    )


def _classification_metrics(
    true_labels: np.ndarray, pred_labels: np.ndarray, num_classes: int
) -> dict[str, tuple[float, bool]]:
    mask = true_labels != IGNORE_LABEL
    y_true = true_labels[mask]
    y_pred = pred_labels[mask]

    macro_f1, _ = restricted_macro_f1(y_true, y_pred, num_classes)
    binary = binary_projection_summary(y_true, y_pred, BINARY_POSITIVE_LABELS)
    tp, tn, fp, fn = binary["tp"], binary["tn"], binary["fp"], binary["fn"]
    n_samples = len(y_true)

    return {
        "macro_f1_restricted": (macro_f1, True),
        "precision": (binary["precision"], (tp + fp) > 0),
        "recall": (binary["recall"], (tp + fn) > 0),
        "specificity": (binary["specificity"], (tn + fp) > 0),
        "f1": (binary["f1"], (2 * tp + fp + fn) > 0),
        "accuracy": (binary["accuracy"], n_samples > 0),
    }


def _window_binary_counts(
    true_labels: np.ndarray, pred_labels: np.ndarray, protocol: AlarmProtocol
) -> tuple[int, int, int, int]:
    # Reproduz exatamente a máscara/positive_labels usados por
    # events.window_level_binary_metrics; usado apenas para decidir
    # indefinição, não para recomputar o valor (que vem de
    # split_event_report para permanecer uma única fonte de verdade).
    mask = true_labels != IGNORE_LABEL
    true_masked = true_labels[mask]
    pred_masked = pred_labels[mask]

    positive_labels = frozenset(protocol.positive_labels)
    true_positive_mask = np.isin(true_masked, list(positive_labels))
    pred_positive_mask = np.isin(pred_masked, list(positive_labels))

    tp = int(np.sum(true_positive_mask & pred_positive_mask))
    fn = int(np.sum(true_positive_mask & ~pred_positive_mask))
    tn = int(np.sum(~true_positive_mask & ~pred_positive_mask))
    fp = int(np.sum(~true_positive_mask & pred_positive_mask))
    return tp, fn, tn, fp


def _event_metrics(
    video_ids: Sequence[str],
    k_ends: Sequence[int],
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    protocol: AlarmProtocol,
    usable_windows: int,
    total_windows: int,
    labeled_windows: int,
) -> tuple[dict, dict[str, tuple[float, bool]]]:
    report = split_event_report(
        list(video_ids),
        list(k_ends),
        list(true_labels),
        list(pred_labels),
        protocol,
        usable_windows,
        total_windows,
        labeled_windows,
    )
    n_fall_events = report["n_fall_events"]
    n_detected_events = report["n_detected_events"]
    latency = report["latency_seconds"]

    tp, fn, tn, fp = _window_binary_counts(
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(pred_labels, dtype=np.int64),
        protocol,
    )

    metrics = {
        "sensitivity": (report["sensitivity"], n_fall_events > 0),
        "fall_sensitivity": (report["fall_sensitivity"], n_fall_events > 0),
        "fall_or_fallen_sensitivity": (
            report["fall_or_fallen_sensitivity"],
            n_fall_events > 0,
        ),
        "detected_events_alarm_within_fall_rate": (
            report["detected_events_alarm_within_fall_rate"],
            n_detected_events > 0,
        ),
        "false_alarms_per_hour": (report["false_alarms_per_hour"], total_windows > 0),
        "false_alarms_per_hour_labeled_time": (
            report["false_alarms_per_hour_labeled_time"],
            labeled_windows > 0,
        ),
        "window_binary_sensitivity": (
            report["window_binary_sensitivity"],
            (tp + fn) > 0,
        ),
        "window_binary_specificity": (
            report["window_binary_specificity"],
            (tn + fp) > 0,
        ),
        "latency_mean_s": (
            latency["mean"] if latency["mean"] is not None else 0.0,
            latency["mean"] is not None,
        ),
        "latency_median_s": (
            latency["median"] if latency["median"] is not None else 0.0,
            latency["median"] is not None,
        ),
    }
    return report, metrics


def _percentile_ci(
    values: list[float], confidence_level: float
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(values, alpha / 2 * 100))
    upper = float(np.percentile(values, (1 - alpha / 2) * 100))
    return lower, upper


def _replicate_source(predictions: SplitPredictions) -> _ReplicateSource:
    return _ReplicateSource(
        windows_by_video=_index_windows_by_video(predictions.video_ids, predictions.k_ends),
        k_ends=np.array(predictions.k_ends, dtype=np.int64),
        true_labels=np.array(predictions.true_labels, dtype=np.int64),
        pred_labels=np.array(predictions.pred_labels, dtype=np.int64),
    )


def _point_metrics(
    predictions: SplitPredictions, protocol: AlarmProtocol, num_classes: int
) -> MetricValues:
    classification = _classification_metrics(
        np.array(predictions.true_labels, dtype=np.int64),
        np.array(predictions.pred_labels, dtype=np.int64),
        num_classes,
    )
    _report, events = _event_metrics(
        predictions.video_ids,
        predictions.k_ends,
        predictions.true_labels,
        predictions.pred_labels,
        protocol,
        predictions.usable_windows,
        predictions.total_windows,
        predictions.labeled_windows,
    )
    return {**classification, **events}


def _draw_subjects(rng: np.random.Generator, subjects: np.ndarray) -> np.ndarray:
    return rng.choice(subjects, size=len(subjects), replace=True)


def _replicate_metrics(
    drawn_subjects: np.ndarray,
    subject_to_videos: dict[int, list[str]],
    source: _ReplicateSource,
    protocol: AlarmProtocol,
    num_classes: int,
) -> MetricValues:
    rep_video_ids, rep_k_ends, rep_true, rep_pred = _build_replicate_arrays(
        drawn_subjects,
        subject_to_videos,
        source.windows_by_video,
        source.k_ends,
        source.true_labels,
        source.pred_labels,
    )
    total_windows = len(rep_video_ids)
    labeled_windows = int(np.sum(rep_true != IGNORE_LABEL))

    classification = _classification_metrics(rep_true, rep_pred, num_classes)
    _rep_report, events = _event_metrics(
        rep_video_ids.tolist(),
        rep_k_ends.tolist(),
        rep_true.tolist(),
        rep_pred.tolist(),
        protocol,
        total_windows,
        total_windows,
        labeled_windows,
    )
    return {**classification, **events}


def _grouped_metric_reports(build_report: Callable[[str], dict]) -> dict[str, dict]:
    return {
        group_name: {name: build_report(name) for name in metric_names}
        for group_name, metric_names in METRIC_GROUPS
    }


def run_grouped_bootstrap_for_split(
    predictions: SplitPredictions,
    subject_to_videos: dict[int, list[str]],
    protocol: AlarmProtocol,
    num_classes: int,
    n_replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict:
    source = _replicate_source(predictions)
    subjects = np.array(sorted(subject_to_videos), dtype=np.int64)
    point = _point_metrics(predictions, protocol, num_classes)

    valid_values: dict[str, list[float]] = {name: [] for name in point}
    for _replicate in range(n_replicates):
        drawn = _draw_subjects(rng, subjects)
        metrics = _replicate_metrics(drawn, subject_to_videos, source, protocol, num_classes)
        for name, (value, is_valid) in metrics.items():
            if is_valid:
                valid_values[name].append(value)

    def _metric_report(name: str) -> dict:
        valid = len(valid_values[name])
        ci_lower, ci_upper = _percentile_ci(valid_values[name], confidence_level)
        return {
            "point_estimate": point[name][0],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "valid_replicates": valid,
            "undefined_replicates": n_replicates - valid,
        }

    return {"n_unique_clusters": len(subjects), **_grouped_metric_reports(_metric_report)}


def run_paired_bootstrap_for_split(
    baseline: SplitPredictions,
    adaptive: SplitPredictions,
    subject_to_videos: dict[int, list[str]],
    protocol: AlarmProtocol,
    num_classes: int,
    n_replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict:
    baseline_source = _replicate_source(baseline)
    adaptive_source = _replicate_source(adaptive)
    subjects = np.array(sorted(subject_to_videos), dtype=np.int64)
    baseline_point = _point_metrics(baseline, protocol, num_classes)
    adaptive_point = _point_metrics(adaptive, protocol, num_classes)

    deltas: dict[str, list[float]] = {name: [] for name in baseline_point}
    for _replicate in range(n_replicates):
        drawn = _draw_subjects(rng, subjects)
        baseline_metrics = _replicate_metrics(
            drawn, subject_to_videos, baseline_source, protocol, num_classes
        )
        adaptive_metrics = _replicate_metrics(
            drawn, subject_to_videos, adaptive_source, protocol, num_classes
        )
        for name, (baseline_value, baseline_valid) in baseline_metrics.items():
            adaptive_value, adaptive_valid = adaptive_metrics[name]
            if baseline_valid and adaptive_valid:
                deltas[name].append(adaptive_value - baseline_value)

    def _metric_report(name: str) -> dict:
        baseline_value, baseline_valid = baseline_point[name]
        adaptive_value, adaptive_valid = adaptive_point[name]
        valid = len(deltas[name])
        ci_lower, ci_upper = _percentile_ci(deltas[name], confidence_level)
        return {
            "baseline_point_estimate": baseline_value if baseline_valid else None,
            "adaptive_point_estimate": adaptive_value if adaptive_valid else None,
            "delta_point_estimate": (
                adaptive_value - baseline_value
                if baseline_valid and adaptive_valid
                else None
            ),
            "delta_ci_lower": ci_lower,
            "delta_ci_upper": ci_upper,
            "valid_replicates": valid,
            "undefined_replicates": n_replicates - valid,
        }

    return {"n_unique_clusters": len(subjects), **_grouped_metric_reports(_metric_report)}


def _csv_rows_from_report(report: dict) -> list[dict]:
    return [
        {"split": split_name, "metric_group": group_name, "metric": metric_name, **metric_report}
        for split_name, split_report in report["splits"].items()
        for group_name, _metric_names in METRIC_GROUPS
        for metric_name, metric_report in split_report[group_name].items()
    ]


def _write_bootstrap_outputs(
    json_path: Path,
    csv_path: Path,
    report: dict,
    csv_rows: list[dict],
    csv_columns: list[str],
    force: bool,
    description: str,
) -> bool:
    if not force and (json_path.is_file() or csv_path.is_file()):
        present = [str(path) for path in (json_path, csv_path) if path.is_file()]
        print(f"skip {', '.join(present)} (já existe, use --force para sobrescrever)")
        return False

    output_dir = json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    json_tmp = output_dir / f".{json_path.name}.tmp-{token}"
    csv_tmp = output_dir / f".{csv_path.name}.tmp-{token}"

    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    csv_text = pd.DataFrame(csv_rows, columns=csv_columns).to_csv(index=False)

    try:
        json_tmp.write_text(json_text, encoding="utf-8")
        csv_tmp.write_text(csv_text, encoding="utf-8")
        # As duas promoções ficam adjacentes de propósito: nada falível corre
        # entre elas, então uma falha não pode promover um artefato sem o
        # outro. É um melhor esforço de pareamento, não uma transação com
        # journal.
        os.replace(json_tmp, json_path)
        os.replace(csv_tmp, csv_path)
    except BaseException:
        json_tmp.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
        raise

    print(f"{json_path}: {description} gravado")
    print(f"{csv_path}: {len(csv_rows)} linhas")
    return True


def _validate_bootstrap_parameters(n_replicates: int, confidence_level: float) -> None:
    if n_replicates <= 0:
        raise ValueError(f"--n-replicates deve ser positivo: {n_replicates!r}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"--confidence-level deve estar em (0, 1): {confidence_level!r}")


def _split_rngs(seed: int) -> dict[str, np.random.Generator]:
    seed_sequences = np.random.SeedSequence(seed).spawn(len(SPLITS))
    return {
        split: np.random.default_rng(seed_sequence)
        for split, seed_sequence in zip(SPLITS, seed_sequences)
    }


def load_arm_evaluation(arm: str, dataset_name: str, run_dir: Path) -> EventEvaluation:
    if arm == "A":
        # A análise de A sempre exigiu a seed congelada; preservado.
        evaluation = baseline_a_events.load_event_evaluation(
            dataset_name, run_dir, fields_allowed_to_differ=frozenset()
        )
    elif arm == "B0":
        evaluation = baseline_b0_events.load_event_evaluation(dataset_name, run_dir)
    elif arm == "B1":
        evaluation = baseline_b1_events.load_event_evaluation(dataset_name, run_dir)
    elif arm == "C0":
        evaluation = baseline_c0_events.load_event_evaluation(dataset_name, run_dir)
    elif arm == "C1":
        evaluation = baseline_c1_events.load_event_evaluation(dataset_name, run_dir)
    else:
        raise ValueError(f"arma não suportada: {arm!r}; opções: {', '.join(ARMS)}")
    declared_arm = cast(RunConfig, evaluation.config).arm
    if declared_arm != arm:
        raise ValueError(
            f"run_dir {run_dir} pertence à arma {declared_arm!r}, não à arma pedida {arm!r}"
        )
    return evaluation


def collect_run_predictions(
    arm: str, run_dir: Path, evaluation: EventEvaluation
) -> RunPredictions:
    frames, evaluate_split = evaluation.prepare()
    splits: dict[str, SplitPredictions] = {}
    subject_maps: dict[str, dict[int, list[str]]] = {}
    for split in SPLITS:
        usable_windows, (video_ids, k_ends, true_labels, pred_labels) = evaluate_split(split)
        split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
        total_windows = len(
            build_window_index(split_frames, stride=EVAL_STRIDE, drop_ignored=False)
        )
        if usable_windows != total_windows:
            raise RuntimeError(
                f"split={split!r}: usable_windows ({usable_windows}) != "
                f"total_windows ({total_windows}) apesar de drop_ignored=False"
            )
        labeled_windows = len(
            build_window_index(split_frames, stride=EVAL_STRIDE, drop_ignored=True)
        )
        splits[split] = SplitPredictions(
            video_ids=video_ids,
            k_ends=k_ends,
            true_labels=true_labels,
            pred_labels=pred_labels,
            usable_windows=usable_windows,
            total_windows=total_windows,
            labeled_windows=labeled_windows,
        )
        subject_to_videos = build_subject_to_videos(frames, split)
        _validate_subject_video_mapping(subject_to_videos, video_ids, split)
        subject_maps[split] = subject_to_videos
    return RunPredictions(
        arm=arm,
        run_dir=run_dir,
        config=cast(RunConfig, evaluation.config),
        splits=splits,
        subject_maps=subject_maps,
    )


def _method(n_replicates: int, confidence_level: float, seed: int, note: str) -> dict:
    return {
        "cluster_unit": "subject",
        "n_replicates": n_replicates,
        "confidence_level": confidence_level,
        "seed": seed,
        "ci_method": "percentile",
        "note": note,
    }


def run_analyze(
    force: bool,
    dataset_name: str = "le2i",
    run_dir: Path | None = None,
    n_replicates: int = DEFAULT_N_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
    arm: str = "A",
) -> bool:
    _validate_bootstrap_parameters(n_replicates, confidence_level)
    if run_dir is None:
        run_dir = default_run_dir_for_arm(dataset_name, arm)
    validate_local_run_dir(run_dir, dataset_name)

    evaluation = load_arm_evaluation(arm, dataset_name, run_dir)
    run = collect_run_predictions(arm, run_dir, evaluation)
    rngs = _split_rngs(seed)
    splits_report = {
        split: run_grouped_bootstrap_for_split(
            run.splits[split],
            run.subject_maps[split],
            BASELINE_A_ALARM_PROTOCOL,
            run.config.num_classes,
            n_replicates,
            confidence_level,
            rngs[split],
        )
        for split in SPLITS
    }

    checkpoint_path = run_dir / "checkpoint.pt"
    training_metrics_path = run_dir / "metrics.json"
    report = {
        "arm": arm,
        "dataset": dataset_name,
        "run_name": run.config.run_name,
        "training_seed": run.config.seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_metrics_path": str(training_metrics_path),
        "training_metrics_sha256": sha256_file(training_metrics_path),
        "alarm_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
        "method": _method(n_replicates, confidence_level, seed, METHOD_NOTE),
        "splits": splits_report,
    }
    return _write_bootstrap_outputs(
        run_dir / GROUPED_BOOTSTRAP_JSON_FILE,
        run_dir / GROUPED_BOOTSTRAP_CSV_FILE,
        report,
        _csv_rows_from_report(report),
        CSV_COLUMNS,
        force,
        "bootstrap agrupado por sujeito",
    )


def paired_output_paths(output_dir: Path, adaptive_arm: str, baseline_arm: str) -> tuple[Path, Path]:
    stem = f"paired_bootstrap_{adaptive_arm.lower()}_minus_{baseline_arm.lower()}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.csv"


def _validate_pair(adaptive_arm: str, baseline_arm: str) -> None:
    if (adaptive_arm, baseline_arm) not in PAIRED_COMPARISONS:
        supported = ", ".join(f"{adaptive} - {baseline}" for adaptive, baseline in PAIRED_COMPARISONS)
        raise ValueError(
            f"comparação pareada {adaptive_arm} - {baseline_arm} não suportada; "
            f"apenas (adaptativa - baseline): {supported}"
        )


def _is_same_or_nested(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def _validate_paired_paths(
    baseline_run_dir: Path, adaptive_run_dir: Path, output_dir: Path
) -> None:
    if _is_same_or_nested(baseline_run_dir, adaptive_run_dir):
        raise ValueError(
            f"run_dirs da comparação coincidem ou se aninham: {baseline_run_dir} "
            f"e {adaptive_run_dir}"
        )
    resolved_output = output_dir.resolve()
    for run_dir in (baseline_run_dir, adaptive_run_dir, REFERENCE_RUN_ROOT):
        resolved_run_dir = run_dir.resolve()
        if resolved_output == resolved_run_dir or resolved_run_dir in resolved_output.parents:
            raise ValueError(
                f"--output-dir {output_dir} coincide com ou está dentro de {run_dir}; "
                "a comparação pareada nunca grava em um run de entrada nem em referência"
            )


def _shared_contract(baseline_config: RunConfig, adaptive_config: RunConfig) -> dict:
    if baseline_config.seed != adaptive_config.seed:
        raise ValueError(
            f"seeds de treino divergentes: {baseline_config.arm} seed="
            f"{baseline_config.seed}, {adaptive_config.arm} seed={adaptive_config.seed}"
        )
    baseline_fields = baseline_config.to_dict()
    adaptive_fields = adaptive_config.to_dict()
    shared = sorted((baseline_fields.keys() & adaptive_fields.keys()) - ARM_IDENTITY_FIELDS)
    divergent = [field for field in shared if baseline_fields[field] != adaptive_fields[field]]
    if divergent:
        raise ValueError(
            f"contrato experimental compartilhado diverge entre {adaptive_config.arm} "
            f"e {baseline_config.arm}: {', '.join(divergent)}"
        )
    return {field: baseline_fields[field] for field in shared}


def _first_divergence(first: Sequence, second: Sequence) -> int:
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return index
    return min(len(first), len(second))


def _support_sha256(predictions: SplitPredictions) -> str:
    payload = json.dumps(
        [predictions.video_ids, predictions.k_ends, predictions.true_labels]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_paired_support(baseline: RunPredictions, adaptive: RunPredictions) -> dict:
    support: dict[str, dict] = {}
    for split in SPLITS:
        subject_to_videos = baseline.subject_maps[split]
        if subject_to_videos != adaptive.subject_maps[split]:
            raise ValueError(
                f"split={split!r}: sujeitos ou mapeamento sujeito->vídeos divergem "
                f"entre {adaptive.arm} e {baseline.arm}"
            )
        baseline_split = baseline.splits[split]
        adaptive_split = adaptive.splits[split]
        for field in ("video_ids", "k_ends", "true_labels"):
            baseline_values = getattr(baseline_split, field)
            adaptive_values = getattr(adaptive_split, field)
            if baseline_values != adaptive_values:
                raise ValueError(
                    f"split={split!r}: {field} ordenado diverge entre "
                    f"{adaptive.arm} e {baseline.arm} na posição "
                    f"{_first_divergence(baseline_values, adaptive_values)} "
                    f"(tamanhos {len(adaptive_values)} e {len(baseline_values)})"
                )
        for field in ("usable_windows", "total_windows", "labeled_windows"):
            if getattr(baseline_split, field) != getattr(adaptive_split, field):
                raise ValueError(
                    f"split={split!r}: {field} diverge entre {adaptive.arm} e {baseline.arm}"
                )
        support[split] = {
            "n_subjects": len(subject_to_videos),
            "subject_to_videos": {
                str(subject): videos for subject, videos in sorted(subject_to_videos.items())
            },
            "n_videos": len(set(baseline_split.video_ids)),
            "total_windows": baseline_split.total_windows,
            "labeled_windows": baseline_split.labeled_windows,
            "support_sha256": _support_sha256(baseline_split),
        }
    return support


def _paired_run_provenance(arm: str, run_dir: Path, config: RunConfig) -> dict:
    paths = {
        "config": run_dir / "config.yaml",
        "checkpoint": run_dir / "checkpoint.pt",
        "training_metrics": run_dir / "metrics.json",
        "event_metrics": run_dir / "event_metrics.json",
        "alarm_protocol": run_dir / "alarm_protocol.yaml",
    }
    provenance: dict = {
        "arm": arm,
        "run_name": config.run_name,
        "run_dir": str(run_dir),
        "training_seed": config.seed,
    }
    for name, path in paths.items():
        provenance[f"{name}_path"] = str(path)
        provenance[f"{name}_sha256"] = sha256_file(path)
    return provenance


def run_compare(
    adaptive_arm: str,
    baseline_arm: str,
    dataset_name: str,
    adaptive_run_dir: Path,
    baseline_run_dir: Path,
    output_dir: Path,
    force: bool,
    n_replicates: int = DEFAULT_N_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
) -> bool:
    _validate_bootstrap_parameters(n_replicates, confidence_level)
    _validate_pair(adaptive_arm, baseline_arm)
    validate_local_run_dir(baseline_run_dir, dataset_name)
    validate_local_run_dir(adaptive_run_dir, dataset_name)
    _validate_paired_paths(baseline_run_dir, adaptive_run_dir, output_dir)
    json_path, csv_path = paired_output_paths(output_dir, adaptive_arm, baseline_arm)

    baseline_evaluation = load_arm_evaluation(baseline_arm, dataset_name, baseline_run_dir)
    adaptive_evaluation = load_arm_evaluation(adaptive_arm, dataset_name, adaptive_run_dir)
    baseline_config = cast(RunConfig, baseline_evaluation.config)
    adaptive_config = cast(RunConfig, adaptive_evaluation.config)
    shared_config = _shared_contract(baseline_config, adaptive_config)
    validate_run_event_artifacts(baseline_run_dir, baseline_config)
    validate_run_event_artifacts(adaptive_run_dir, adaptive_config)
    provenance = {
        "adaptive": _paired_run_provenance(adaptive_arm, adaptive_run_dir, adaptive_config),
        "baseline": _paired_run_provenance(baseline_arm, baseline_run_dir, baseline_config),
    }

    if not force and (json_path.is_file() or csv_path.is_file()):
        present = [str(path) for path in (json_path, csv_path) if path.is_file()]
        print(f"skip {', '.join(present)} (já existe, use --force para sobrescrever)")
        return False

    baseline = collect_run_predictions(baseline_arm, baseline_run_dir, baseline_evaluation)
    adaptive = collect_run_predictions(adaptive_arm, adaptive_run_dir, adaptive_evaluation)
    support = _validate_paired_support(baseline, adaptive)

    rngs = _split_rngs(seed)
    splits_report = {
        split: run_paired_bootstrap_for_split(
            baseline.splits[split],
            adaptive.splits[split],
            baseline.subject_maps[split],
            BASELINE_A_ALARM_PROTOCOL,
            baseline_config.num_classes,
            n_replicates,
            confidence_level,
            rngs[split],
        )
        for split in SPLITS
    }

    report = {
        "comparison": f"{adaptive_arm} - {baseline_arm}",
        "dataset": dataset_name,
        "training_seed": baseline_config.seed,
        **provenance,
        "alarm_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
        "method": {
            **_method(n_replicates, confidence_level, seed, PAIRED_METHOD_NOTE),
            "paired": True,
            "delta": "adaptive - baseline",
        },
        "compatibility": {
            "label_names": list(get_dataset(dataset_name).label_names),
            "shared_config": shared_config,
            "splits": support,
        },
        "splits": splits_report,
    }
    return _write_bootstrap_outputs(
        json_path,
        csv_path,
        report,
        _csv_rows_from_report(report),
        PAIRED_CSV_COLUMNS,
        force,
        f"bootstrap pareado {adaptive_arm} - {baseline_arm}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_bootstrap_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset", default="le2i", choices=DATASETS)
        subparser.add_argument("--n-replicates", type=int, default=DEFAULT_N_REPLICATES)
        subparser.add_argument(
            "--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL
        )
        subparser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help=(
            "Bootstrap agrupado por sujeito sobre o checkpoint treinado de uma "
            "arma; produz IC percentil das métricas congeladas"
        ),
    )
    analyze_parser.add_argument("--arm", default="A", choices=ARMS)
    analyze_parser.add_argument(
        "--run-dir", type=Path, default=None, help="Padrão: run_dir local da arma/dataset"
    )
    add_bootstrap_arguments(analyze_parser)
    analyze_parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve grouped_bootstrap.json/.csv já existentes",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help=(
            "Bootstrap pareado por sujeito de adaptativa - baseline (B1 - B0 ou "
            "C1 - C0) sobre os mesmos sorteios de sujeitos"
        ),
    )
    compare_parser.add_argument("--adaptive-arm", required=True, choices=ARMS)
    compare_parser.add_argument("--baseline-arm", required=True, choices=ARMS)
    compare_parser.add_argument(
        "--adaptive-run-dir", type=Path, default=None, help="Padrão: run_dir local da arma"
    )
    compare_parser.add_argument(
        "--baseline-run-dir", type=Path, default=None, help="Padrão: run_dir local da arma"
    )
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    add_bootstrap_arguments(compare_parser)
    compare_parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve paired_bootstrap_*.json/.csv já existentes",
    )

    subparsers.add_parser(
        "selftest",
        help="Roda checagens sintéticas do bootstrap agrupado por sujeito",
    )

    args = parser.parse_args()
    if args.command == "analyze":
        run_analyze(
            force=args.force,
            dataset_name=args.dataset,
            run_dir=args.run_dir,
            n_replicates=args.n_replicates,
            confidence_level=args.confidence_level,
            seed=args.seed,
            arm=args.arm,
        )
    elif args.command == "compare":
        run_compare(
            adaptive_arm=args.adaptive_arm,
            baseline_arm=args.baseline_arm,
            dataset_name=args.dataset,
            adaptive_run_dir=(
                args.adaptive_run_dir
                or default_run_dir_for_arm(args.dataset, args.adaptive_arm)
            ),
            baseline_run_dir=(
                args.baseline_run_dir
                or default_run_dir_for_arm(args.dataset, args.baseline_arm)
            ),
            output_dir=args.output_dir,
            force=args.force,
            n_replicates=args.n_replicates,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.grouped_bootstrap import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
