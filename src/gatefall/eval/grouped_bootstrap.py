"""Bootstrap agrupado por sujeito (IC percentil) das métricas congeladas da arma A.

Ferramenta independente de estágio, deliberadamente fora do pipeline padrão
(`gatefall.pipeline`) e da suíte de lifecycle de
`gatefall.eval.baseline_a_events` (não abre o lock, não escreve o journal e
não toca `alarm_protocol.yaml`/`event_metrics.json`/`metrics.json`/
`checkpoint.pt`). Roda a inferência local uma única vez por split (val e
test) e bootstrapa as predições/identidades já cacheadas — nunca reexecuta o
modelo por réplica.

Janelas de avaliação em `EVAL_STRIDE=1` se sobrepõem em 96% e não são
observações independentes; tratá-las como unidade de reamostragem infla
artificialmente a precisão do intervalo de confiança. Este módulo reamostra
por **sujeito** (unidade de cluster): cada réplica sorteia sujeitos com
reposição e arrasta consigo todos os vídeos e todas as janelas daquele
sujeito, preservando a estrutura de dependência dentro de cada vídeo/sujeito.

Este módulo nunca seleciona, ranqueia ou promove um modelo/protocolo:
`BASELINE_A_ALARM_PROTOCOL` permanece a única configuração congelada usada em
`gatefall.eval.baseline_a_events`. O bootstrap aqui produzido é estritamente
descritivo — apenas expõe incerteza em torno das métricas já congeladas, não
implementa nenhum teste de hipótese em nível de janela nem p-valor.
"""

import argparse
import json
import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import build_window_index
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.eval.alarm_protocol import AlarmProtocol, BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.events import split_event_report
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization,
    load_stats,
    validate_stats_layout,
)
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import validate_local_run_dir
from gatefall.train.artifacts import load_compatible_checkpoint, validate_training_run
from gatefall.train.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.metrics import (
    BINARY_POSITIVE_LABELS,
    binary_projection_summary,
    restricted_macro_f1,
)
from gatefall.train.tcn import TCNClassifier

RUN_DIR = Path("runs/local/le2i/baseline_a")
GROUPED_BOOTSTRAP_JSON_FILE = "grouped_bootstrap.json"
GROUPED_BOOTSTRAP_CSV_FILE = "grouped_bootstrap.csv"

DEFAULT_SEED = 42
DEFAULT_N_REPLICATES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95

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

METHOD_NOTE = (
    "janelas de avaliação em EVAL_STRIDE=1 se sobrepõem fortemente e NÃO "
    "foram tratadas como amostras independentes; a unidade de reamostragem é "
    "o sujeito (cluster), com reposição, arrastando todos os vídeos e todas "
    "as janelas de cada sujeito sorteado"
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


def _load_model(
    config: TrainConfig, checkpoint_path: Path, device: str
) -> TCNClassifier:
    model = load_compatible_checkpoint(checkpoint_path, config).to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict_with_identity(
    model: TCNClassifier,
    source: PoseWindowDataset,
    stats: StandardizationStats,
    device: str,
    batch_size: int,
) -> tuple[list[str], list[int], list[int], list[int]]:
    video_ids: list[str] = []
    k_ends: list[int] = []
    true_labels: list[int] = []
    pred_labels: list[int] = []

    batch_windows: list[np.ndarray] = []
    batch_labels: list[int] = []
    batch_identity: list[tuple[str, int]] = []

    def flush() -> None:
        if not batch_windows:
            return
        stacked = np.stack(batch_windows, axis=0)
        standardized = apply_standardization(stacked, stats)
        x = torch.from_numpy(standardized).to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        for (video_id, k_end), label, pred in zip(batch_identity, batch_labels, preds):
            video_ids.append(video_id)
            k_ends.append(k_end)
            true_labels.append(label)
            pred_labels.append(int(pred))

        batch_windows.clear()
        batch_labels.clear()
        batch_identity.clear()

    for i in range(len(source)):
        window, label, (video_id, k_end) = source[i]
        batch_windows.append(window)
        batch_labels.append(label)
        batch_identity.append((video_id, k_end))
        if len(batch_windows) == batch_size:
            flush()
    flush()

    return video_ids, k_ends, true_labels, pred_labels


@dataclass(frozen=True)
class SplitPredictions:
    video_ids: list[str]
    k_ends: list[int]
    true_labels: list[int]
    pred_labels: list[int]
    usable_windows: int
    total_windows: int
    labeled_windows: int


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


def run_grouped_bootstrap_for_split(
    predictions: SplitPredictions,
    subject_to_videos: dict[int, list[str]],
    protocol: AlarmProtocol,
    num_classes: int,
    n_replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict:
    video_ids_arr = np.array(predictions.video_ids, dtype=object)
    k_ends_arr = np.array(predictions.k_ends, dtype=np.int64)
    true_arr = np.array(predictions.true_labels, dtype=np.int64)
    pred_arr = np.array(predictions.pred_labels, dtype=np.int64)

    windows_by_video = _index_windows_by_video(predictions.video_ids, predictions.k_ends)
    subjects = sorted(subject_to_videos)
    subjects_arr = np.array(subjects, dtype=np.int64)
    n_unique_clusters = len(subjects)

    point_classification = _classification_metrics(true_arr, pred_arr, num_classes)
    _point_report, point_events = _event_metrics(
        predictions.video_ids,
        predictions.k_ends,
        predictions.true_labels,
        predictions.pred_labels,
        protocol,
        predictions.usable_windows,
        predictions.total_windows,
        predictions.labeled_windows,
    )

    metric_names = CLASSIFICATION_METRICS + EVENT_METRICS
    valid_values: dict[str, list[float]] = {name: [] for name in metric_names}
    valid_counts: dict[str, int] = {name: 0 for name in metric_names}

    for _replicate in range(n_replicates):
        drawn = rng.choice(subjects_arr, size=n_unique_clusters, replace=True)
        rep_video_ids, rep_k_ends, rep_true, rep_pred = _build_replicate_arrays(
            drawn, subject_to_videos, windows_by_video, k_ends_arr, true_arr, pred_arr
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

        for name, (value, is_valid) in classification.items():
            if is_valid:
                valid_values[name].append(value)
                valid_counts[name] += 1
        for name, (value, is_valid) in events.items():
            if is_valid:
                valid_values[name].append(value)
                valid_counts[name] += 1

    def _metric_report(name: str, point_value: float) -> dict:
        valid = valid_counts[name]
        ci_lower, ci_upper = _percentile_ci(valid_values[name], confidence_level)
        return {
            "point_estimate": point_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "valid_replicates": valid,
            "undefined_replicates": n_replicates - valid,
        }

    classification_out = {
        name: _metric_report(name, point_classification[name][0])
        for name in CLASSIFICATION_METRICS
    }
    events_out = {
        name: _metric_report(name, point_events[name][0]) for name in EVENT_METRICS
    }

    return {
        "n_unique_clusters": n_unique_clusters,
        "classification": classification_out,
        "events": events_out,
    }


def _csv_rows_from_report(report: dict) -> list[dict]:
    csv_rows: list[dict] = []
    for split_name, split_report in report["splits"].items():
        for group_name in ("classification", "events"):
            for metric_name, metric_report in split_report[group_name].items():
                csv_rows.append(
                    {
                        "split": split_name,
                        "metric_group": group_name,
                        "metric": metric_name,
                        "point_estimate": metric_report["point_estimate"],
                        "ci_lower": metric_report["ci_lower"],
                        "ci_upper": metric_report["ci_upper"],
                        "valid_replicates": metric_report["valid_replicates"],
                        "undefined_replicates": metric_report["undefined_replicates"],
                    }
                )
    return csv_rows


def _write_grouped_bootstrap_outputs(
    run_dir: Path, report: dict, csv_rows: list[dict], force: bool
) -> bool:
    json_path = run_dir / GROUPED_BOOTSTRAP_JSON_FILE
    csv_path = run_dir / GROUPED_BOOTSTRAP_CSV_FILE
    if not force and (json_path.is_file() or csv_path.is_file()):
        present = [str(path) for path in (json_path, csv_path) if path.is_file()]
        print(f"skip {', '.join(present)} (já existe, use --force para sobrescrever)")
        return False

    run_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    json_tmp = run_dir / f".{GROUPED_BOOTSTRAP_JSON_FILE}.tmp-{token}"
    csv_tmp = run_dir / f".{GROUPED_BOOTSTRAP_CSV_FILE}.tmp-{token}"

    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    csv_text = pd.DataFrame(csv_rows, columns=CSV_COLUMNS).to_csv(index=False)

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

    print(f"{json_path}: bootstrap agrupado por sujeito gravado")
    print(f"{csv_path}: {len(csv_rows)} linhas")
    return True


def run_analyze(
    force: bool,
    dataset_name: str = "le2i",
    run_dir: Path = RUN_DIR,
    n_replicates: int = DEFAULT_N_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
) -> None:
    if n_replicates <= 0:
        raise ValueError(f"--n-replicates deve ser positivo: {n_replicates!r}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"--confidence-level deve estar em (0, 1): {confidence_level!r}")
    validate_local_run_dir(run_dir)

    adapter = get_dataset(dataset_name)
    checkpoint_path = run_dir / "checkpoint.pt"
    training_metrics_path = run_dir / "metrics.json"

    expected_config = replace(
        BASELINE_A_CONFIG,
        standardization_stats_path=str(adapter.pose_stats_path),
        standardization_stats_sha256=sha256_file(adapter.pose_stats_path),
    )
    config = validate_training_run(run_dir, expected_config=expected_config)
    if config.eval_stride != EVAL_STRIDE:
        raise ValueError(
            f"config.eval_stride ({config.eval_stride}) diverge de "
            f"EVAL_STRIDE ({EVAL_STRIDE})"
        )

    stats = load_stats(adapter.pose_stats_path)
    validate_stats_layout(stats)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(config, checkpoint_path, device)

    frames = adapter.load_frames()

    seed_sequences = np.random.SeedSequence(seed).spawn(2)
    split_seed_sequences = dict(zip(("val", "test"), seed_sequences))

    split_predictions: dict[str, SplitPredictions] = {}
    subject_maps: dict[str, dict[int, list[str]]] = {}

    for split in ("val", "test"):
        source = PoseWindowDataset(
            frames,
            split,
            EVAL_STRIDE,
            lambda video_id: build_pose_features(
                video_id, pose_root=adapter.pose_root
            )[0],
            drop_ignored=False,
        )
        video_ids, k_ends, true_labels, pred_labels = _predict_with_identity(
            model, source, stats, device, batch_size=config.batch_size
        )
        usable_windows = len(source)
        total_windows = len(
            build_window_index(
                cast(pd.DataFrame, frames[frames["split"] == split]),
                stride=EVAL_STRIDE,
                drop_ignored=False,
            )
        )
        if usable_windows != total_windows:
            raise RuntimeError(
                f"split={split!r}: usable_windows ({usable_windows}) != "
                f"total_windows ({total_windows}) apesar de drop_ignored=False"
            )
        labeled_windows = len(
            build_window_index(
                cast(pd.DataFrame, frames[frames["split"] == split]),
                stride=EVAL_STRIDE,
                drop_ignored=True,
            )
        )

        split_predictions[split] = SplitPredictions(
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

    splits_report: dict[str, dict] = {}
    for split in ("val", "test"):
        rng = np.random.default_rng(split_seed_sequences[split])
        splits_report[split] = run_grouped_bootstrap_for_split(
            split_predictions[split],
            subject_maps[split],
            BASELINE_A_ALARM_PROTOCOL,
            config.num_classes,
            n_replicates,
            confidence_level,
            rng,
        )

    report = {
        "run_name": config.run_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_metrics_path": str(training_metrics_path),
        "training_metrics_sha256": sha256_file(training_metrics_path),
        "alarm_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
        "method": {
            "cluster_unit": "subject",
            "n_replicates": n_replicates,
            "confidence_level": confidence_level,
            "seed": seed,
            "ci_method": "percentile",
            "note": METHOD_NOTE,
        },
        "splits": splits_report,
    }
    csv_rows = _csv_rows_from_report(report)

    _write_grouped_bootstrap_outputs(run_dir, report, csv_rows, force)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _fixture_single_video_subject(
    subject: int, video_id: str, n: int, true_labels: list[int], pred_labels: list[int]
) -> tuple[SplitPredictions, dict[int, list[str]]]:
    predictions = SplitPredictions(
        video_ids=[video_id] * n,
        k_ends=list(range(n)),
        true_labels=true_labels,
        pred_labels=pred_labels,
        usable_windows=n,
        total_windows=n,
        labeled_windows=n,
    )
    return predictions, {subject: [video_id]}


def _selftest_replicate_window_counts_are_cluster_multiples() -> bool:
    # Sujeito 1 -> vídeo com 5 janelas; sujeito 2 -> vídeo com 3 janelas.
    video_ids = ["v1"] * 5 + ["v2"] * 3
    k_ends = list(range(5)) + list(range(3))
    true_labels = [0] * 8
    pred_labels = [0] * 8
    predictions = SplitPredictions(
        video_ids=video_ids,
        k_ends=k_ends,
        true_labels=true_labels,
        pred_labels=pred_labels,
        usable_windows=8,
        total_windows=8,
        labeled_windows=8,
    )
    subject_to_videos = {1: ["v1"], 2: ["v2"]}
    windows_by_video = _index_windows_by_video(video_ids, k_ends)

    drawn = np.array([1, 1, 2], dtype=np.int64)
    rep_video_ids, _rep_k_ends, _rep_true, _rep_pred = _build_replicate_arrays(
        drawn,
        subject_to_videos,
        windows_by_video,
        np.array(k_ends, dtype=np.int64),
        np.array(true_labels, dtype=np.int64),
        np.array(pred_labels, dtype=np.int64),
    )

    real_video_of = lambda virtual_id: virtual_id.split("__draw")[0]
    from collections import Counter

    counts_by_real_video: dict[str, int] = Counter(
        real_video_of(vid) for vid in rep_video_ids.tolist()
    )
    ok = (
        len(rep_video_ids) == 13
        and counts_by_real_video["v1"] % 5 == 0
        and counts_by_real_video["v2"] % 3 == 0
        and counts_by_real_video["v1"] == 10
        and counts_by_real_video["v2"] == 3
    )
    return _check(
        "contagem de janelas por vídeo real na réplica é sempre múltiplo "
        "exato da contagem verdadeira daquele vídeo (clusters, não janelas, "
        "são reamostrados)",
        ok,
    )


def _selftest_multi_video_subject_moves_together() -> bool:
    video_ids = ["va"] * 4 + ["vb"] * 2
    k_ends = list(range(4)) + list(range(2))
    true_labels = [0] * 6
    pred_labels = [0] * 6
    subject_to_videos = {1: ["va", "vb"]}
    windows_by_video = _index_windows_by_video(video_ids, k_ends)

    drawn = np.array([1], dtype=np.int64)
    rep_video_ids, _rep_k_ends, _rep_true, _rep_pred = _build_replicate_arrays(
        drawn,
        subject_to_videos,
        windows_by_video,
        np.array(k_ends, dtype=np.int64),
        np.array(true_labels, dtype=np.int64),
        np.array(pred_labels, dtype=np.int64),
    )
    real_videos_present = {vid.split("__draw")[0] for vid in rep_video_ids.tolist()}
    ok = len(rep_video_ids) == 6 and real_videos_present == {"va", "vb"}
    return _check(
        "sortear um sujeito com múltiplos vídeos arrasta todos os seus "
        "vídeos e todas as suas janelas juntos",
        ok,
    )


def _selftest_repeated_draws_stay_separate_in_events() -> bool:
    # Geometria adaptada de alarm_protocol_sensitivity: fall em k=[2,3],
    # fallen em k=[5,6,7], alarme dispara em k=[10,11,12] (trigger=3 do
    # protocolo congelado) -> exatamente 1 evento fall detectado no vídeo
    # base.
    n = 13
    video_id = "video_fixture"
    k_ends = list(range(n))
    true_labels = [0] * n
    true_labels[2:4] = [1, 1]
    true_labels[5:8] = [2, 2, 2]
    pred_labels = [0] * n
    pred_labels[10:13] = [1, 1, 1]

    subject_to_videos = {1: [video_id]}
    windows_by_video = _index_windows_by_video([video_id] * n, k_ends)

    for multiplicity in (1, 2, 3):
        drawn = np.array([1] * multiplicity, dtype=np.int64)
        rep_video_ids, rep_k_ends, rep_true, rep_pred = _build_replicate_arrays(
            drawn,
            subject_to_videos,
            windows_by_video,
            np.array(k_ends, dtype=np.int64),
            np.array(true_labels, dtype=np.int64),
            np.array(pred_labels, dtype=np.int64),
        )
        total_windows = len(rep_video_ids)
        labeled_windows = int(np.sum(rep_true != IGNORE_LABEL))
        report, _metrics = _event_metrics(
            rep_video_ids.tolist(),
            rep_k_ends.tolist(),
            rep_true.tolist(),
            rep_pred.tolist(),
            BASELINE_A_ALARM_PROTOCOL,
            total_windows,
            total_windows,
            labeled_windows,
        )
        if report["n_fall_events"] != multiplicity or report["n_detected_events"] != multiplicity:
            return _check(
                "sorteios repetidos do mesmo cluster permanecem estatisticamente "
                "separados na avaliação de eventos: n_fall_events/n_detected_events "
                "escalam exatamente com a multiplicidade do sorteio",
                False,
            )
    return _check(
        "sorteios repetidos do mesmo cluster permanecem estatisticamente "
        "separados na avaliação de eventos: n_fall_events/n_detected_events "
        "escalam exatamente com a multiplicidade do sorteio",
        True,
    )


def _selftest_identical_seeds_identical_results() -> bool:
    n = 13
    video_id = "video_fixture"
    k_ends = list(range(n))
    true_labels = [0] * n
    true_labels[2:4] = [1, 1]
    true_labels[5:8] = [2, 2, 2]
    pred_labels = [0] * n
    pred_labels[10:13] = [1, 1, 1]
    predictions = SplitPredictions(
        video_ids=[video_id] * n,
        k_ends=k_ends,
        true_labels=true_labels,
        pred_labels=pred_labels,
        usable_windows=n,
        total_windows=n,
        labeled_windows=n,
    )
    subject_to_videos = {1: [video_id]}

    seed_sequence = np.random.SeedSequence(123)
    rng_a = np.random.default_rng(seed_sequence)
    result_a = run_grouped_bootstrap_for_split(
        predictions, subject_to_videos, BASELINE_A_ALARM_PROTOCOL, 10, 200, 0.95, rng_a
    )
    seed_sequence_repeat = np.random.SeedSequence(123)
    rng_b = np.random.default_rng(seed_sequence_repeat)
    result_b = run_grouped_bootstrap_for_split(
        predictions, subject_to_videos, BASELINE_A_ALARM_PROTOCOL, 10, 200, 0.95, rng_b
    )
    return _check(
        "seeds idênticas produzem resultados idênticos (igualdade de dict "
        "completa do relatório por split)",
        result_a == result_b,
    )


def _selftest_percentile_bounds_known_array() -> bool:
    values = [float(v) for v in range(1, 101)]
    ci_lower, ci_upper = _percentile_ci(values, 0.95)
    ok = (
        ci_lower is not None
        and ci_upper is not None
        and abs(ci_lower - 3.475) < 1e-9
        and abs(ci_upper - 97.525) < 1e-9
    )
    return _check(
        "percentil 2.5/97.5 sobre 1..100 reproduz os limites conhecidos "
        "(3.475, 97.525) via interpolação linear",
        ok,
    )


def _selftest_subject_split_mapping_validation() -> bool:
    frames_two_subjects = pd.DataFrame(
        {
            "video_id": ["v1", "v1"],
            "split": ["val", "val"],
            "subject": [1, 2],
        }
    )
    raised_two_subjects = False
    try:
        build_subject_to_videos(frames_two_subjects, "val")
    except ValueError:
        raised_two_subjects = True

    subject_to_videos = {1: ["v1"]}
    raised_missing_from_map = False
    try:
        _validate_subject_video_mapping(subject_to_videos, ["v1", "v2"], "val")
    except ValueError:
        raised_missing_from_map = True

    raised_missing_from_predictions = False
    try:
        _validate_subject_video_mapping(subject_to_videos, [], "val")
    except ValueError:
        raised_missing_from_predictions = True

    ok = raised_two_subjects and raised_missing_from_map and raised_missing_from_predictions
    return _check(
        "vídeo com dois subjects distintos e vídeo ausente do mapeamento (em "
        "qualquer direção) levantam ValueError",
        ok,
    )


def _selftest_undefined_replicates_handling() -> bool:
    # Todos os replicates sem nenhum evento fall (true_labels/pred_labels
    # constantes em 0) -> sensitivity indefinida em 100% das réplicas.
    n = 5
    video_id = "video_no_fall"
    predictions_no_fall = SplitPredictions(
        video_ids=[video_id] * n,
        k_ends=list(range(n)),
        true_labels=[0] * n,
        pred_labels=[0] * n,
        usable_windows=n,
        total_windows=n,
        labeled_windows=n,
    )
    subject_to_videos_no_fall = {1: [video_id]}
    rng_no_fall = np.random.default_rng(np.random.SeedSequence(7))
    result_no_fall = run_grouped_bootstrap_for_split(
        predictions_no_fall,
        subject_to_videos_no_fall,
        BASELINE_A_ALARM_PROTOCOL,
        10,
        50,
        0.95,
        rng_no_fall,
    )
    sensitivity_no_fall = result_no_fall["events"]["sensitivity"]
    all_no_fall_ok = (
        sensitivity_no_fall["ci_lower"] is None
        and sensitivity_no_fall["ci_upper"] is None
        and sensitivity_no_fall["valid_replicates"] == 0
        and sensitivity_no_fall["undefined_replicates"] == 50
    )

    # Dois sujeitos: um sem eventos fall, outro com um evento fall -> as
    # réplicas alternam entre definidas e indefinidas para sensitivity.
    n2 = 13
    video_with_fall = "video_with_fall"
    true_with_fall = [0] * n2
    true_with_fall[2:4] = [1, 1]
    true_with_fall[5:8] = [2, 2, 2]
    pred_with_fall = [0] * n2
    pred_with_fall[10:13] = [1, 1, 1]

    video_ids_mixed = [video_id] * n + [video_with_fall] * n2
    k_ends_mixed = list(range(n)) + list(range(n2))
    true_mixed = [0] * n + true_with_fall
    pred_mixed = [0] * n + pred_with_fall
    predictions_mixed = SplitPredictions(
        video_ids=video_ids_mixed,
        k_ends=k_ends_mixed,
        true_labels=true_mixed,
        pred_labels=pred_mixed,
        usable_windows=len(video_ids_mixed),
        total_windows=len(video_ids_mixed),
        labeled_windows=len(video_ids_mixed),
    )
    subject_to_videos_mixed = {1: [video_id], 2: [video_with_fall]}
    rng_mixed = np.random.default_rng(np.random.SeedSequence(11))
    result_mixed = run_grouped_bootstrap_for_split(
        predictions_mixed,
        subject_to_videos_mixed,
        BASELINE_A_ALARM_PROTOCOL,
        10,
        200,
        0.95,
        rng_mixed,
    )
    sensitivity_mixed = result_mixed["events"]["sensitivity"]
    mixed_ok = (
        sensitivity_mixed["valid_replicates"] + sensitivity_mixed["undefined_replicates"] == 200
        and 0 < sensitivity_mixed["valid_replicates"] < 200
        and sensitivity_mixed["ci_lower"] is not None
        and sensitivity_mixed["ci_upper"] is not None
    )

    ok = all_no_fall_ok and mixed_ok
    check_no_fall_and_mixed = _check(
        "réplicas com denominador zero nunca substituem por zero: fixture "
        "sem nenhum evento fall produz limites nulos com "
        "valid_replicates=0/undefined_replicates=n_replicates; fixture mista "
        "produz valid+undefined=n_replicates com CI calculado só sobre o "
        "subconjunto válido",
        ok,
    )

    # Sujeito mapeado para uma lista de vídeos vazia -> toda réplica sorteia
    # zero janelas, total_windows == 0 em 100% das réplicas ->
    # false_alarms_per_hour indefinida (denominador de horas de vídeo é
    # zero), nunca substituída silenciosamente por 0.0.
    n3 = 5
    predictions_zero_windows = SplitPredictions(
        video_ids=["video_unreachable"] * n3,
        k_ends=list(range(n3)),
        true_labels=[0] * n3,
        pred_labels=[0] * n3,
        usable_windows=n3,
        total_windows=n3,
        labeled_windows=n3,
    )
    subject_to_videos_zero_windows: dict[int, list[str]] = {1: []}
    rng_zero_windows = np.random.default_rng(np.random.SeedSequence(13))
    result_zero_windows = run_grouped_bootstrap_for_split(
        predictions_zero_windows,
        subject_to_videos_zero_windows,
        BASELINE_A_ALARM_PROTOCOL,
        10,
        50,
        0.95,
        rng_zero_windows,
    )
    false_alarms_per_hour_zero_windows = result_zero_windows["events"][
        "false_alarms_per_hour"
    ]
    zero_windows_ok = (
        false_alarms_per_hour_zero_windows["ci_lower"] is None
        and false_alarms_per_hour_zero_windows["ci_upper"] is None
        and false_alarms_per_hour_zero_windows["valid_replicates"] == 0
        and false_alarms_per_hour_zero_windows["undefined_replicates"] == 50
    )
    check_zero_windows = _check(
        "réplicas cujo sorteio de subjects resulta em zero janelas (subject "
        "sem vídeos) nunca contam false_alarms_per_hour como válida com "
        "0.0: valid_replicates=0/undefined_replicates=n_replicates e "
        "limites nulos",
        zero_windows_ok,
    )

    return check_no_fall_and_mixed and check_zero_windows


def _selftest_binary_f1_validity_rule() -> bool:
    # Caso (a): existe positivo verdadeiro (fall/fallen), mas nenhuma
    # predição positiva -> tp=0, fp=0, fn>0. Denominador do F1 próprio
    # (2*tp+fp+fn) é fn>0, logo é uma réplica VÁLIDA com F1=0.0, não uma
    # réplica indefinida.
    n_a = 5
    video_a = "video_all_predictions_negative"
    predictions_a = SplitPredictions(
        video_ids=[video_a] * n_a,
        k_ends=list(range(n_a)),
        true_labels=[1, 1, 0, 0, 0],
        pred_labels=[0, 0, 0, 0, 0],
        usable_windows=n_a,
        total_windows=n_a,
        labeled_windows=n_a,
    )
    subject_to_videos_a = {1: [video_a]}
    rng_a = np.random.default_rng(np.random.SeedSequence(101))
    result_a = run_grouped_bootstrap_for_split(
        predictions_a, subject_to_videos_a, BASELINE_A_ALARM_PROTOCOL, 10, 50, 0.95, rng_a
    )
    f1_a = result_a["classification"]["f1"]
    case_a_ok = (
        f1_a["valid_replicates"] == 50
        and f1_a["undefined_replicates"] == 0
        and f1_a["point_estimate"] == 0.0
        and f1_a["ci_lower"] == 0.0
        and f1_a["ci_upper"] == 0.0
    )

    # Caso (b): predições incluem positivos, mas nenhum positivo verdadeiro
    # -> tp=0, fn=0, fp>0. Denominador do F1 próprio é fp>0, logo também é
    # uma réplica VÁLIDA com F1=0.0.
    n_b = 5
    video_b = "video_no_true_positives"
    predictions_b = SplitPredictions(
        video_ids=[video_b] * n_b,
        k_ends=list(range(n_b)),
        true_labels=[0, 0, 0, 0, 0],
        pred_labels=[1, 1, 0, 0, 0],
        usable_windows=n_b,
        total_windows=n_b,
        labeled_windows=n_b,
    )
    subject_to_videos_b = {1: [video_b]}
    rng_b = np.random.default_rng(np.random.SeedSequence(102))
    result_b = run_grouped_bootstrap_for_split(
        predictions_b, subject_to_videos_b, BASELINE_A_ALARM_PROTOCOL, 10, 50, 0.95, rng_b
    )
    f1_b = result_b["classification"]["f1"]
    case_b_ok = (
        f1_b["valid_replicates"] == 50
        and f1_b["undefined_replicates"] == 0
        and f1_b["point_estimate"] == 0.0
        and f1_b["ci_lower"] == 0.0
        and f1_b["ci_upper"] == 0.0
    )

    # Caso genuinamente indefinido: nem positivo verdadeiro, nem positivo
    # predito -> tp=fp=fn=0, denominador do F1 próprio é zero -> réplica
    # indefinida, limites nulos.
    n_c = 5
    video_c = "video_no_positives_at_all"
    predictions_c = SplitPredictions(
        video_ids=[video_c] * n_c,
        k_ends=list(range(n_c)),
        true_labels=[0, 0, 0, 0, 0],
        pred_labels=[0, 0, 0, 0, 0],
        usable_windows=n_c,
        total_windows=n_c,
        labeled_windows=n_c,
    )
    subject_to_videos_c = {1: [video_c]}
    rng_c = np.random.default_rng(np.random.SeedSequence(103))
    result_c = run_grouped_bootstrap_for_split(
        predictions_c, subject_to_videos_c, BASELINE_A_ALARM_PROTOCOL, 10, 50, 0.95, rng_c
    )
    f1_c = result_c["classification"]["f1"]
    case_c_ok = (
        f1_c["valid_replicates"] == 0
        and f1_c["undefined_replicates"] == 50
        and f1_c["ci_lower"] is None
        and f1_c["ci_upper"] is None
    )

    ok = case_a_ok and case_b_ok and case_c_ok
    return _check(
        "F1 binário é válido (F1=0.0) sempre que seu próprio denominador "
        "(2*tp+fp+fn) é não nulo, mesmo quando precision ou recall têm "
        "denominador zero isoladamente; só é indefinido quando tp=fp=fn=0",
        ok,
    )


def _selftest_alarm_protocol_recorded_in_report() -> bool:
    n = 5
    video_id = "video_alarm_protocol_check"
    predictions = SplitPredictions(
        video_ids=[video_id] * n,
        k_ends=list(range(n)),
        true_labels=[0] * n,
        pred_labels=[0] * n,
        usable_windows=n,
        total_windows=n,
        labeled_windows=n,
    )
    subject_to_videos = {1: [video_id]}
    rng = np.random.default_rng(np.random.SeedSequence(104))
    split_report = run_grouped_bootstrap_for_split(
        predictions, subject_to_videos, BASELINE_A_ALARM_PROTOCOL, 10, 10, 0.95, rng
    )
    report = {
        "run_name": "selftest_run",
        "checkpoint_path": "unused",
        "checkpoint_sha256": "unused",
        "training_metrics_path": "unused",
        "training_metrics_sha256": "unused",
        "alarm_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
        "method": {
            "cluster_unit": "subject",
            "n_replicates": 10,
            "confidence_level": 0.95,
            "seed": 104,
            "ci_method": "percentile",
            "note": METHOD_NOTE,
        },
        "splits": {"val": split_report},
    }
    ok = (
        "alarm_protocol" in report
        and report["alarm_protocol"] == BASELINE_A_ALARM_PROTOCOL.to_dict()
    )
    return _check(
        "relatório do bootstrap agrupado carrega alarm_protocol == "
        "BASELINE_A_ALARM_PROTOCOL.to_dict()",
        ok,
    )


def run_grouped_bootstrap_selftest() -> bool:
    checks = [
        _selftest_replicate_window_counts_are_cluster_multiples(),
        _selftest_multi_video_subject_moves_together(),
        _selftest_repeated_draws_stay_separate_in_events(),
        _selftest_identical_seeds_identical_results(),
        _selftest_percentile_bounds_known_array(),
        _selftest_subject_split_mapping_validation(),
        _selftest_undefined_replicates_handling(),
        _selftest_binary_f1_validity_rule(),
        _selftest_alarm_protocol_recorded_in_report(),
    ]
    ok = all(checks)
    if not ok:
        print("\ngrouped_bootstrap selftest FALHOU", file=sys.stderr)
    else:
        print("\ngrouped_bootstrap selftest OK: todas as checagens passaram")
    return ok


def run_selftest() -> None:
    if not run_grouped_bootstrap_selftest():
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help=(
            "Bootstrap agrupado por sujeito sobre o checkpoint treinado; "
            "produz IC percentil das métricas congeladas da arma A"
        ),
    )
    analyze_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)
    analyze_parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    analyze_parser.add_argument(
        "--n-replicates", type=int, default=DEFAULT_N_REPLICATES
    )
    analyze_parser.add_argument(
        "--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL
    )
    analyze_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    analyze_parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve grouped_bootstrap.json/.csv já existentes",
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
        )
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
