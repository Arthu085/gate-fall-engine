"""Sumário multi-seed das armas A, B0, B1, C0 e C1.

Ferramenta somente leitura, conceitualmente separada do bootstrap agrupado por
sujeito (`gatefall.eval.grouped_bootstrap`): aqui a unidade agregada é o
**treino independente** (uma seed, um checkpoint, um `run_dir` completo), não
a réplica de reamostragem sobre um único checkpoint fixo. Este módulo nunca
mistura as duas noções de variação — jamais combina réplicas de bootstrap com
seeds de treino na mesma estatística.

Cada `--run-dir` deve ser um run local completo da arma selecionada,
diferindo apenas na seed e nos campos de auditoria permitidos pelo validador.
As armas A, B0, B1 e C1 exigem avaliação de evento íntegra; C0 agrega apenas
classificação. O módulo valida um fingerprint sha256 da configuração
normalizada, guarda por seed os blocos de classificação e, quando aplicável,
evento validados na íntegra (confusion_matrix, per_class, latências por evento) e
agrega estatísticas descritivas (n/mean/std/min/max) sobre `macro_f1_restricted`
e `f1_by_class` (splits train/val/test), `per_class` (splits train/val/test),
a projeção binária queda/caído derivada da confusion_matrix (splits
train/val/test) e todo campo escalar das métricas de evento (splits val/test).
Não seleciona, ranqueia nem promove nenhum run.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from gatefall.datasets import get_dataset
from gatefall.datasets.base import DatasetAdapter
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.dinov3.dataset_guard import ensure_dinov3_dataset_supported
from gatefall.eval.alarm_protocol import (
    BASELINE_A_ALARM_PROTOCOL,
    load_alarm_protocol,
    save_alarm_protocol,
)
from gatefall.eval.baseline_a_events import EVENT_SPLIT_FIELDS, validate_event_metrics
from gatefall.features.dinov3_standardization import load_stats as load_visual_stats
from gatefall.features.dinov3_standardization import (
    validate_stats_freshness as validate_visual_stats_freshness,
)
from gatefall.features.dinov3_standardization import (
    validate_stats_layout as validate_visual_stats_layout,
)
from gatefall.features.quality_storage import quality_set_sha256
from gatefall.features.standardization import load_stats as load_pose_stats
from gatefall.features.standardization import validate_stats_layout as validate_pose_stats_layout
from gatefall.features.standardize_dinov3 import DINOV3_STATS_PATH
from gatefall.hashing import sha256_file
from gatefall.runs import validate_local_run_dir
from gatefall.sam3.dataset_guard import ensure_sam3_dataset_supported
from gatefall.train.baseline_a.artifacts import validate_training_run
from gatefall.train.baseline_b0.artifacts import validate_b0_training_run
from gatefall.train.baseline_b0.config import B0_FUSION_CONFIG, B0TrainConfig
from gatefall.train.baseline_b0.config import save_config as save_b0_config
from gatefall.train.baseline_b0.model import B0FusionClassifier
from gatefall.train.baseline_b0.run import resolve_b0_config
from gatefall.train.baseline_b1.artifacts import validate_b1_training_run
from gatefall.train.baseline_b1.config import B1_ADAPTIVE_GATE_CONFIG, B1TrainConfig
from gatefall.train.baseline_b1.config import save_config as save_b1_config
from gatefall.train.baseline_b1.model import B1AdaptiveGateClassifier
from gatefall.train.baseline_b1.run import resolve_b1_config
from gatefall.train.baseline_c0.artifacts import validate_c0_training_run
from gatefall.train.baseline_c0.config import C0_FUSION_CONFIG, C0TrainConfig
from gatefall.train.baseline_c0.config import save_config as save_c0_config
from gatefall.train.baseline_c0.run import resolve_c0_config_for_inputs as resolve_c0_config
from gatefall.train.shared.sam3_inputs import _validated_inputs as validated_sam3_inputs
from gatefall.train.baseline_c0.model import C0FusionClassifier
from gatefall.train.baseline_c1.artifacts import validate_c1_training_run
from gatefall.train.baseline_c1.config import C1_ADAPTIVE_GATE_CONFIG, C1TrainConfig
from gatefall.train.baseline_c1.config import save_config as save_c1_config
from gatefall.train.baseline_c1.run import resolve_c1_config_for_inputs as resolve_c1_config
from gatefall.train.baseline_c1.model import C1AdaptiveGateClassifier
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig, save_config
from gatefall.train.shared.metrics import (
    BINARY_POSITIVE_LABELS,
    RESTRICTED_CLASSES,
    binary_projection_from_confusion_matrix,
    classification_summary,
    restricted_macro_f1,
)
from gatefall.train.shared.tcn import TCNClassifier

MULTISEED_SUMMARY_JSON_FILE = "multiseed_summary.json"
MULTISEED_SUMMARY_CSV_FILE = "multiseed_summary.csv"

MIN_SEEDS = 2
ARMS = ("A", "B0", "B1", "C0", "C1")
EVENT_ARMS = frozenset({"A", "B0", "B1", "C1"})
RunConfig = TrainConfig | B0TrainConfig | B1TrainConfig | C0TrainConfig | C1TrainConfig
AUDIT_FIELDS = frozenset({"seed", "trainable_param_count"})

CLASSIFICATION_SPLITS: tuple[str, ...] = ("train", "val", "test")
EVENT_SPLITS: tuple[str, ...] = ("val", "test")

# Todo campo escalar de `event_metrics.json[splits][split]` é agregado;
# `latency_seconds` não é escalar (é um objeto com `per_event`/`mean`/`median`)
# e é substituído por `latency_seconds_mean`/`latency_seconds_median`. A
# regra é "todo campo escalar", não uma lista escolhida a dedo — por isso a
# derivação a partir de `EVENT_SPLIT_FIELDS` em vez de uma tupla própria.
EVENT_SCALAR_FIELDS: tuple[str, ...] = tuple(
    sorted(EVENT_SPLIT_FIELDS - {"latency_seconds"})
) + ("latency_seconds_mean", "latency_seconds_median")

PER_CLASS_METRIC_FIELDS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "tp",
    "tn",
    "fp",
    "fn",
    "support",
)

# Ordem de campos de `binary_projection_from_confusion_matrix`/
# `binary_projection_summary`.
BINARY_FIELDS: tuple[str, ...] = (
    "tp",
    "tn",
    "fp",
    "fn",
    "precision",
    "recall",
    "specificity",
    "f1",
    "accuracy",
)

CSV_COLUMNS = ["split", "metric_group", "entity", "metric", "n", "mean", "std", "min", "max"]


def _config_fingerprint(config: RunConfig) -> str:
    data = config.to_dict()
    data.pop("seed", None)
    if config.arm != "A":
        data.pop("trainable_param_count", None)
    payload = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_event_scalar(split_data: dict, field: str) -> float | int | None:
    if field == "latency_seconds_mean":
        return split_data["latency_seconds"]["mean"]
    if field == "latency_seconds_median":
        return split_data["latency_seconds"]["median"]
    return split_data[field]


def _aggregate_stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": n,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if n >= 2 else None,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_all(seed_reports: list[dict], adapter: DatasetAdapter) -> dict:
    label_names = adapter.label_names
    restricted_label_names = [label_names[c] for c in RESTRICTED_CLASSES]

    classification = {
        split: {
            "macro_f1_restricted": _aggregate_stats(
                [
                    report["classification"][split]["macro_f1_restricted"]
                    for report in seed_reports
                    if report["classification"][split]["macro_f1_restricted"] is not None
                ]
            ),
            "f1_by_class": {
                name: _aggregate_stats(
                    [
                        report["classification"][split]["f1_by_class"][str(class_id)]
                        for report in seed_reports
                        if report["classification"][split]["f1_by_class"][str(class_id)]
                        is not None
                    ]
                )
                for class_id, name in zip(RESTRICTED_CLASSES, restricted_label_names)
            },
        }
        for split in CLASSIFICATION_SPLITS
    }

    per_class = {
        split: {
            name: {
                metric: _aggregate_stats(
                    [
                        report["classification"][split]["per_class"][name][metric]
                        for report in seed_reports
                        if report["classification"][split]["per_class"][name][metric]
                        is not None
                    ]
                )
                for metric in PER_CLASS_METRIC_FIELDS
            }
            for name in label_names
        }
        for split in CLASSIFICATION_SPLITS
    }

    binary_fall_fallen = {
        split: {
            field: _aggregate_stats(
                [
                    report["binary_fall_fallen"][split][field]
                    for report in seed_reports
                    if report["binary_fall_fallen"][split][field] is not None
                ]
            )
            for field in BINARY_FIELDS
        }
        for split in CLASSIFICATION_SPLITS
    }

    aggregate = {
        "classification": classification,
        "per_class": per_class,
        "binary_fall_fallen": binary_fall_fallen,
    }
    if "events" in seed_reports[0]:
        aggregate["events"] = {
            split: {
                field: _aggregate_stats(
                    [
                        value
                        for report in seed_reports
                        if (value := _extract_event_scalar(report["events"][split], field))
                        is not None
                    ]
                )
                for field in EVENT_SCALAR_FIELDS
            }
            for split in EVENT_SPLITS
        }
    return aggregate


def _csv_rows_from_aggregate(aggregate: dict, adapter: DatasetAdapter) -> list[dict]:
    label_names = adapter.label_names
    restricted_label_names = [label_names[c] for c in RESTRICTED_CLASSES]
    rows: list[dict] = []

    for split in CLASSIFICATION_SPLITS:
        split_aggregate = aggregate["classification"][split]
        rows.append(
            {
                "split": split,
                "metric_group": "classification",
                "entity": "",
                "metric": "macro_f1_restricted",
                **split_aggregate["macro_f1_restricted"],
            }
        )
        for name in restricted_label_names:
            rows.append(
                {
                    "split": split,
                    "metric_group": "classification",
                    "entity": name,
                    "metric": "f1_by_class",
                    **split_aggregate["f1_by_class"][name],
                }
            )

    for split in CLASSIFICATION_SPLITS:
        for name in label_names:
            for metric in PER_CLASS_METRIC_FIELDS:
                rows.append(
                    {
                        "split": split,
                        "metric_group": "per_class",
                        "entity": name,
                        "metric": metric,
                        **aggregate["per_class"][split][name][metric],
                    }
                )

    for split in CLASSIFICATION_SPLITS:
        for field in BINARY_FIELDS:
            rows.append(
                {
                    "split": split,
                    "metric_group": "binary",
                    "entity": "",
                    "metric": field,
                    **aggregate["binary_fall_fallen"][split][field],
                }
            )

    if "events" in aggregate:
        for split in EVENT_SPLITS:
            for field in EVENT_SCALAR_FIELDS:
                rows.append(
                    {
                        "split": split,
                        "metric_group": "events",
                        "entity": "",
                        "metric": field,
                        **aggregate["events"][split][field],
                    }
                )

    return rows


def _resolve_shared_expected(dataset_name: str, arm: str = "A") -> RunConfig:
    if arm not in ARMS:
        raise ValueError(f"arma não suportada: {arm!r}")
    if arm != "A" and dataset_name != "le2i":
        raise ValueError(f"arma {arm} suporta somente le2i")
    adapter = get_dataset(dataset_name)
    stats_path = adapter.pose_stats_path
    if arm == "A":
        return replace(
            BASELINE_A_CONFIG,
            standardization_stats_path=str(stats_path),
            standardization_stats_sha256=sha256_file(stats_path),
        )
    if arm in {"B0", "B1"}:
        ensure_dinov3_dataset_supported(adapter)
        pose_stats = load_pose_stats(stats_path)
        validate_pose_stats_layout(pose_stats)
        visual_stats = load_visual_stats(DINOV3_STATS_PATH)
        validate_visual_stats_layout(visual_stats, dataset_name=dataset_name)
        validate_visual_stats_freshness(visual_stats, adapter.frames_path)
        common = (
            stats_path,
            sha256_file(stats_path),
            DINOV3_STATS_PATH,
            sha256_file(DINOV3_STATS_PATH),
        )
        if arm == "B0":
            return resolve_b0_config(B0_FUSION_CONFIG.seed, *common)
        return resolve_b1_config(
            B1_ADAPTIVE_GATE_CONFIG.seed,
            *common,
            adapter.quality_root,
            quality_set_sha256(
                [str(video_id) for video_id in adapter.load_frames()["video_id"].unique()],
                quality_root=adapter.quality_root,
            ),
        )
    ensure_sam3_dataset_supported(adapter)
    inputs = validated_sam3_inputs(adapter, dataset_name)
    if arm == "C0":
        return resolve_c0_config(C0_FUSION_CONFIG.seed, adapter, inputs)
    return resolve_c1_config(C1_ADAPTIVE_GATE_CONFIG.seed, adapter, inputs)


def _validate_arm_training_run(run_dir: Path, expected: RunConfig) -> RunConfig:
    if expected.arm == "A":
        assert isinstance(expected, TrainConfig)
        return validate_training_run(
            run_dir, expected_config=expected, fields_allowed_to_differ=frozenset({"seed"})
        )
    allowed = AUDIT_FIELDS
    if expected.arm == "B0":
        assert isinstance(expected, B0TrainConfig)
        return validate_b0_training_run(run_dir, expected, allowed)
    if expected.arm == "B1":
        assert isinstance(expected, B1TrainConfig)
        return validate_b1_training_run(run_dir, expected, allowed)
    if expected.arm == "C0":
        assert isinstance(expected, C0TrainConfig)
        return validate_c0_training_run(run_dir, expected, allowed)
    assert isinstance(expected, C1TrainConfig)
    return validate_c1_training_run(run_dir, expected, allowed)


def _reject_duplicate_run_dirs(run_dirs: list[Path]) -> None:
    resolved_seen: dict[Path, Path] = {}
    duplicates: list[str] = []
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved in resolved_seen:
            duplicates.append(str(run_dir))
        else:
            resolved_seen[resolved] = run_dir
    if duplicates:
        raise ValueError(
            f"--run-dir duplicado(s) (mesmo path resolvido): {', '.join(duplicates)}"
        )


def _require_classification_diagnostics(final_split: dict, run_dir: Path, split: str) -> None:
    if "confusion_matrix" not in final_split or "per_class" not in final_split:
        raise RuntimeError(
            f"metrics.json em {run_dir}: final.{split} sem confusion_matrix/"
            "per_class, necessários para o sumário multi-seed de fidelidade "
            "completa"
        )


def _summarize(
    run_dirs: list[Path], shared_expected: RunConfig, adapter: DatasetAdapter
) -> tuple[dict, list[dict]]:
    if len(run_dirs) < MIN_SEEDS:
        raise ValueError(
            f"são necessárias ao menos {MIN_SEEDS} seeds distintas (--run-dir "
            f"repetível); recebido(s) {len(run_dirs)}"
        )
    _reject_duplicate_run_dirs(run_dirs)

    seed_reports: list[dict] = []
    seen_seeds: dict[int, Path] = {}
    fingerprint: str | None = None
    fingerprint_run_dir: Path | None = None

    for run_dir in run_dirs:
        validate_local_run_dir(run_dir, adapter.identifier)
        config = _validate_arm_training_run(run_dir, shared_expected)

        if config.seed in seen_seeds:
            raise RuntimeError(
                f"seed duplicada entre runs: seed={config.seed} em "
                f"{seen_seeds[config.seed]} e {run_dir}"
            )
        seen_seeds[config.seed] = run_dir

        run_fingerprint = _config_fingerprint(config)
        if fingerprint is None:
            fingerprint = run_fingerprint
            fingerprint_run_dir = run_dir
        elif run_fingerprint != fingerprint:
            raise RuntimeError(
                f"config.yaml em {run_dir} diverge (fora dos campos permitidos) da "
                f"configuração de {fingerprint_run_dir}"
            )

        checkpoint_path = run_dir / "checkpoint.pt"
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)

        classification: dict[str, dict] = {}
        binary_fall_fallen: dict[str, dict] = {}
        for split in CLASSIFICATION_SPLITS:
            final_split = metrics["final"][split]
            _require_classification_diagnostics(final_split, run_dir, split)
            classification[split] = final_split
            binary_fall_fallen[split] = binary_projection_from_confusion_matrix(
                final_split["confusion_matrix"], BINARY_POSITIVE_LABELS
            )

        seed_report = {
            "seed": config.seed,
            "run_dir": str(run_dir),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "classification": classification,
            "binary_fall_fallen": binary_fall_fallen,
        }
        if config.arm in EVENT_ARMS:
            alarm_protocol_path = run_dir / "alarm_protocol.yaml"
            if not alarm_protocol_path.is_file():
                raise RuntimeError(f"alarm_protocol.yaml ausente em {run_dir}")
            protocol = load_alarm_protocol(alarm_protocol_path)
            if protocol != BASELINE_A_ALARM_PROTOCOL:
                raise RuntimeError(
                    f"alarm_protocol.yaml em {run_dir} incompatível com o "
                    f"protocolo congelado da arma {config.arm}"
                )

            event_metrics_path = run_dir / "event_metrics.json"
            if not event_metrics_path.is_file():
                raise RuntimeError(f"event_metrics.json ausente em {run_dir}")
            with event_metrics_path.open(encoding="utf-8") as stream:
                event_metrics = json.load(stream)
            try:
                validate_event_metrics(
                    event_metrics,
                    config,
                    checkpoint_path,
                    alarm_protocol_path,
                    training_metrics_path=metrics_path,
                    require_hashes=True,
                )
            except (ValueError, OSError, TypeError, KeyError) as exc:
                raise RuntimeError(
                    f"event_metrics.json inválido em {run_dir}: {exc}"
                ) from exc
            seed_report["events"] = {
                split: event_metrics["splits"][split] for split in EVENT_SPLITS
            }
        seed_reports.append(seed_report)

    assert fingerprint is not None
    aggregate = _aggregate_all(seed_reports, adapter)

    report = {
        "arm": shared_expected.arm,
        "config_fingerprint_sha256": fingerprint,
        "n_seeds": len(seed_reports),
        "seeds": seed_reports,
        "aggregate": aggregate,
    }
    csv_rows = _csv_rows_from_aggregate(aggregate, adapter)
    return report, csv_rows


def _write_multiseed_summary_outputs(
    output_dir: Path, report: dict, csv_rows: list[dict], force: bool
) -> bool:
    json_path = output_dir / MULTISEED_SUMMARY_JSON_FILE
    csv_path = output_dir / MULTISEED_SUMMARY_CSV_FILE
    if not force and (json_path.is_file() or csv_path.is_file()):
        present = [str(path) for path in (json_path, csv_path) if path.is_file()]
        print(f"skip {', '.join(present)} (já existe, use --force para sobrescrever)")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    json_tmp = output_dir / f".{MULTISEED_SUMMARY_JSON_FILE}.tmp-{token}"
    csv_tmp = output_dir / f".{MULTISEED_SUMMARY_CSV_FILE}.tmp-{token}"

    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    csv_text = pd.DataFrame(csv_rows, columns=CSV_COLUMNS).to_csv(index=False)

    try:
        json_tmp.write_text(json_text, encoding="utf-8")
        csv_tmp.write_text(csv_text, encoding="utf-8")
        os.replace(json_tmp, json_path)
        os.replace(csv_tmp, csv_path)
    except BaseException:
        json_tmp.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
        raise

    print(f"{json_path}: sumário multi-seed gravado ({report['n_seeds']} seeds)")
    print(f"{csv_path}: {len(csv_rows)} linhas")
    return True


def run_summarize(
    dataset_name: str,
    run_dirs: list[Path],
    output_dir: Path,
    force: bool,
    arm: str = "A",
) -> bool:
    adapter = get_dataset(dataset_name)
    shared_expected = _resolve_shared_expected(dataset_name, arm)
    report, csv_rows = _summarize(run_dirs, shared_expected, adapter)
    return _write_multiseed_summary_outputs(output_dir, report, csv_rows, force)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _build_event_split(
    n_fall_events: int,
    n_detected_events: int,
    sensitivity: float,
    fall_sensitivity: float,
    fall_or_fallen_sensitivity: float,
    false_alarms_per_hour: float,
    n_false_alarms: int,
    latency_mean: float | None,
    latency_median: float | None,
) -> dict:
    n_missed_events = n_fall_events - n_detected_events
    n_events_detected_in_fall = min(1, n_detected_events)
    n_events_detected_in_fall_or_fallen = min(1, n_detected_events)
    per_event = [latency_mean] * n_detected_events if n_detected_events > 0 else []
    return {
        "usable_windows": 100,
        "total_windows": 100,
        "labeled_windows": 80,
        "total_video_time_hours": 1.0,
        "labeled_time_hours": 0.8,
        "n_fall_events": n_fall_events,
        "n_detected_events": n_detected_events,
        "n_missed_events": n_missed_events,
        "sensitivity": sensitivity,
        "n_events_detected_in_fall": n_events_detected_in_fall,
        "n_events_detected_in_fall_or_fallen": n_events_detected_in_fall_or_fallen,
        "fall_sensitivity": fall_sensitivity,
        "fall_or_fallen_sensitivity": fall_or_fallen_sensitivity,
        "detected_events_alarm_within_fall_rate": 1.0 if n_detected_events > 0 else 0.0,
        "n_alarms_total": n_detected_events + n_false_alarms,
        "n_false_alarms": n_false_alarms,
        "n_pre_fall_false_alarms": 0,
        "false_alarms_per_hour": false_alarms_per_hour,
        "false_alarms_per_hour_labeled_time": false_alarms_per_hour,
        "window_binary_sensitivity": sensitivity,
        "window_binary_specificity": 0.9,
        "latency_seconds": {
            "per_event": per_event,
            "mean": latency_mean if per_event else None,
            "median": latency_median if per_event else None,
        },
    }


def _synthetic_classification_arrays(offset: int) -> tuple[np.ndarray, np.ndarray]:
    # 16 amostras cobrindo as 8 classes restritas com suporte 2 cada; as
    # classes 5 (lie_down) e 6 (lying) ficam sem suporte real, replicando o
    # cenário real do Le2i em stride 4.
    y_true = np.array(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 7, 7, 8, 8, 9, 9], dtype=np.int64
    )
    y_pred = y_true.copy()
    for i in range(offset):
        idx = (2 * i) % len(y_true)
        true_class = int(y_true[idx])
        position = RESTRICTED_CLASSES.index(true_class)
        wrong_class = RESTRICTED_CLASSES[(position + 1) % len(RESTRICTED_CLASSES)]
        y_pred[idx] = wrong_class
    return y_true, y_pred


def _synthetic_arm_config(arm: str, seed: int, epochs: int) -> RunConfig:
    if arm == "A":
        return replace(
            BASELINE_A_CONFIG,
            seed=seed,
            epochs=epochs,
            standardization_stats_path="synthetic",
            standardization_stats_sha256="synthetic",
        )
    common = dict(
        seed=seed,
        epochs=epochs,
        pose_standardization_stats_path="synthetic",
        pose_standardization_stats_sha256="synthetic",
        visual_standardization_stats_path="synthetic",
        visual_standardization_stats_sha256="synthetic",
    )
    if arm == "B0":
        return replace(B0_FUSION_CONFIG, **common)
    if arm == "B1":
        return replace(
            B1_ADAPTIVE_GATE_CONFIG,
            **common,
            quality_features_path="synthetic",
            quality_features_sha256="synthetic",
        )
    sam3 = dict(
        sam3_features_path="synthetic",
        sam3_features_sha256="synthetic",
        sam3_provenance={"synthetic": "synthetic"},
    )
    if arm == "C0":
        return replace(C0_FUSION_CONFIG, **common, **sam3)
    if arm == "C1":
        return replace(
            C1_ADAPTIVE_GATE_CONFIG,
            **common,
            **sam3,
            quality_features_path="synthetic",
            quality_features_sha256="synthetic",
            pose_features_sha256="synthetic",
        )
    raise ValueError(f"arma não suportada: {arm!r}")


def _write_synthetic_seed_run(
    run_dir: Path,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_event: dict,
    test_event: dict,
    epochs: int = 1,
    arm: str = "A",
) -> RunConfig:
    config = _synthetic_arm_config(arm, seed, epochs)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoint.pt"
    if isinstance(config, C1TrainConfig):
        save_c1_config(config, config_path, force=True)
        model = C1AdaptiveGateClassifier(
            config.channels, config.kernel_size, config.dilations,
            config.dropout, config.num_classes,
        )
    elif isinstance(config, B1TrainConfig):
        save_b1_config(config, config_path, force=True)
        model = B1AdaptiveGateClassifier(
            config.channels, config.kernel_size, config.dilations,
            config.dropout, config.num_classes,
        )
    elif isinstance(config, C0TrainConfig):
        save_c0_config(config, config_path, force=True)
        model = C0FusionClassifier(
            config.channels, config.kernel_size, config.dilations,
            config.dropout, config.num_classes,
        )
    elif isinstance(config, B0TrainConfig):
        save_b0_config(config, config_path, force=True)
        model = B0FusionClassifier(
            config.channels, config.kernel_size, config.dilations,
            config.dropout, config.num_classes,
        )
    else:
        save_config(config, config_path, force=True)
        model = TCNClassifier(
            config.input_dim, config.channels, config.kernel_size,
            config.dilations, config.dropout, config.num_classes,
        )
    torch.save(model.state_dict(), checkpoint_path)

    macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred, config.num_classes)
    summary = classification_summary(y_true, y_pred, LE2I_LABEL_NAMES, config.num_classes)

    split = {
        "macro_f1_restricted": macro_f1,
        "f1_by_class": {str(index): f1_by_class[index] for index in RESTRICTED_CLASSES},
        "support": {
            name: summary["per_class"][name]["support"] for name in LE2I_LABEL_NAMES
        },
        "confusion_matrix": summary["confusion_matrix"],
        "per_class": summary["per_class"],
    }
    metrics = {
        "run_name": config.run_name,
        "epochs_trained": config.epochs,
        "history": [
            {"epoch": epoch, "train_loss": 0.0, "val_macro_f1_restricted": macro_f1}
            for epoch in range(1, config.epochs + 1)
        ],
        "final": {name: dict(split) for name in ("train", "val", "test")},
        "restricted_classes": RESTRICTED_CLASSES,
        "excluded_classes": [
            index for index in range(config.num_classes) if index not in RESTRICTED_CLASSES
        ],
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream)

    if arm in EVENT_ARMS:
        alarm_protocol_path = run_dir / "alarm_protocol.yaml"
        save_alarm_protocol(BASELINE_A_ALARM_PROTOCOL, alarm_protocol_path, force=True)

        event_metrics_path = run_dir / "event_metrics.json"
        event_report = {
            "run_name": config.run_name,
            "checkpoint_path": str(checkpoint_path),
            "alarm_protocol_path": str(alarm_protocol_path),
            "splits": {"val": val_event, "test": test_event},
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_metrics_sha256": sha256_file(metrics_path),
            "alarm_protocol_sha256": sha256_file(alarm_protocol_path),
        }
        with event_metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(event_report, stream)

    return config


def _selftest_aggregate_stats_known_array() -> bool:
    stats = _aggregate_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    ok = (
        stats["n"] == 5
        and stats["mean"] == 3.0
        and stats["std"] is not None
        and abs(stats["std"] - statistics.stdev([1.0, 2.0, 3.0, 4.0, 5.0])) < 1e-12
        and stats["min"] == 1.0
        and stats["max"] == 5.0
    )
    single = _aggregate_stats([7.0])
    single_ok = single["n"] == 1 and single["mean"] == 7.0 and single["std"] is None
    empty = _aggregate_stats([])
    empty_ok = (
        empty["n"] == 0
        and empty["mean"] is None
        and empty["std"] is None
        and empty["min"] is None
        and empty["max"] is None
    )
    return _check(
        "_aggregate_stats reproduz n/mean/std/min/max sobre um array "
        "conhecido, com std=None para n<2 e todos os campos None para n=0",
        ok and single_ok and empty_ok,
    )


def _selftest_two_valid_seed_runs_aggregate() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "seed1"
        run2 = root / "seed2"

        y_true_1, y_pred_1 = _synthetic_classification_arrays(offset=1)
        y_true_2, y_pred_2 = _synthetic_classification_arrays(offset=3)
        expected_macro_f1_1, _ = restricted_macro_f1(y_true_1, y_pred_1)
        expected_macro_f1_2, _ = restricted_macro_f1(y_true_2, y_pred_2)
        expected_summary_1 = classification_summary(y_true_1, y_pred_1, LE2I_LABEL_NAMES)
        expected_binary_1 = binary_projection_from_confusion_matrix(
            expected_summary_1["confusion_matrix"], BINARY_POSITIVE_LABELS
        )

        val_event_1 = _build_event_split(
            n_fall_events=2,
            n_detected_events=1,
            sensitivity=0.5,
            fall_sensitivity=0.5,
            fall_or_fallen_sensitivity=0.5,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        test_event_1 = val_event_1
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true_1, y_pred=y_pred_1, val_event=val_event_1, test_event=test_event_1
        )

        val_event_2 = _build_event_split(
            n_fall_events=2,
            n_detected_events=2,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=2.0,
            n_false_alarms=2,
            latency_mean=3.0,
            latency_median=3.0,
        )
        test_event_2 = val_event_2
        _write_synthetic_seed_run(
            run2, seed=2, y_true=y_true_2, y_pred=y_pred_2, val_event=val_event_2, test_event=test_event_2
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        report, csv_rows = _summarize([run1, run2], shared_expected, adapter)

        expected_macro_f1_values = [expected_macro_f1_1, expected_macro_f1_2]
        expected_stats = _aggregate_stats(expected_macro_f1_values)

        macro_f1_val = report["aggregate"]["classification"]["val"]["macro_f1_restricted"]
        sensitivity_val = report["aggregate"]["events"]["val"]["sensitivity"]
        false_alarms_val = report["aggregate"]["events"]["val"]["false_alarms_per_hour"]
        latency_mean_val = report["aggregate"]["events"]["val"]["latency_seconds_mean"]

        ok = (
            report["n_seeds"] == 2
            and len(report["seeds"]) == 2
            and macro_f1_val["n"] == expected_stats["n"]
            and abs(macro_f1_val["mean"] - expected_stats["mean"]) < 1e-12
            and abs(macro_f1_val["std"] - expected_stats["std"]) < 1e-12
            and macro_f1_val["min"] == expected_stats["min"]
            and macro_f1_val["max"] == expected_stats["max"]
            and sensitivity_val["n"] == 2
            and abs(sensitivity_val["mean"] - 0.75) < 1e-12
            and false_alarms_val["n"] == 2
            and abs(false_alarms_val["mean"] - 1.0) < 1e-12
            and latency_mean_val["n"] == 2
            and abs(latency_mean_val["mean"] - 2.0) < 1e-12
            and len(csv_rows) > 0
        )
        return _check(
            "duas seeds válidas produzem n/mean/std/min/max corretos, "
            "agregados sobre macro_f1_restricted e métricas de evento",
            ok,
        )


def _selftest_seed_blocks_round_trip_and_new_aggregates_exist() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "seed1"
        run2 = root / "seed2"

        y_true_1, y_pred_1 = _synthetic_classification_arrays(offset=1)
        y_true_2, y_pred_2 = _synthetic_classification_arrays(offset=3)
        expected_summary_1 = classification_summary(y_true_1, y_pred_1, LE2I_LABEL_NAMES)

        val_event_1 = _build_event_split(
            n_fall_events=2,
            n_detected_events=1,
            sensitivity=0.5,
            fall_sensitivity=0.5,
            fall_or_fallen_sensitivity=0.5,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        test_event_1 = val_event_1
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true_1, y_pred=y_pred_1, val_event=val_event_1, test_event=test_event_1
        )

        val_event_2 = _build_event_split(
            n_fall_events=2,
            n_detected_events=2,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=2.0,
            n_false_alarms=2,
            latency_mean=3.0,
            latency_median=3.0,
        )
        test_event_2 = val_event_2
        _write_synthetic_seed_run(
            run2, seed=2, y_true=y_true_2, y_pred=y_pred_2, val_event=val_event_2, test_event=test_event_2
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        report, _csv_rows = _summarize([run1, run2], shared_expected, adapter)

        seed1_report = next(r for r in report["seeds"] if r["seed"] == 1)
        round_trip_ok = (
            seed1_report["classification"]["train"]["confusion_matrix"]
            == expected_summary_1["confusion_matrix"]
            and seed1_report["classification"]["train"]["per_class"]
            == expected_summary_1["per_class"]
            and seed1_report["events"]["val"]["latency_seconds"]["per_event"]
            == val_event_1["latency_seconds"]["per_event"]
        )

        aggregate = report["aggregate"]
        new_leaves_ok = (
            "per_class" in aggregate
            and all(
                label in aggregate["per_class"]["val"] for label in LE2I_LABEL_NAMES
            )
            and "binary_fall_fallen" in aggregate
            and all(field in aggregate["binary_fall_fallen"]["val"] for field in BINARY_FIELDS)
            and all(field in aggregate["events"]["val"] for field in EVENT_SCALAR_FIELDS)
        )

        return _check(
            "blocos por seed (confusion_matrix, per_class, latência per_event) "
            "chegam verbatim no relatório e as novas folhas de agregado "
            "(per_class, binary_fall_fallen, campos de evento ampliados) existem",
            round_trip_ok and new_leaves_ok,
        )


def _selftest_binary_fall_fallen_matches_independent_recomputation() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "seed1"
        run2 = root / "seed2"

        y_true_1, y_pred_1 = _synthetic_classification_arrays(offset=1)
        y_true_2, y_pred_2 = _synthetic_classification_arrays(offset=3)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true_1, y_pred=y_pred_1, val_event=event, test_event=event
        )
        _write_synthetic_seed_run(
            run2, seed=2, y_true=y_true_2, y_pred=y_pred_2, val_event=event, test_event=event
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        report, _csv_rows = _summarize([run1, run2], shared_expected, adapter)

        ok = True
        for seed_report in report["seeds"]:
            for split in CLASSIFICATION_SPLITS:
                matrix = seed_report["classification"][split]["confusion_matrix"]
                expected = binary_projection_from_confusion_matrix(matrix, BINARY_POSITIVE_LABELS)
                ok = ok and seed_report["binary_fall_fallen"][split] == expected

        return _check(
            "binary_fall_fallen[split] armazenado é idêntico a uma "
            "recomputação independente a partir da confusion_matrix armazenada",
            ok,
        )


def _selftest_missing_diagnostics_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_ok = root / "run_ok"
        run_missing_diagnostics = root / "run_missing_diagnostics"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run_ok, seed=1, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )
        _write_synthetic_seed_run(
            run_missing_diagnostics, seed=2, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )

        metrics_path = run_missing_diagnostics / "metrics.json"
        with metrics_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        del metrics["final"]["val"]["confusion_matrix"]
        del metrics["final"]["val"]["per_class"]
        with metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream)

        event_metrics_path = run_missing_diagnostics / "event_metrics.json"
        with event_metrics_path.open(encoding="utf-8") as stream:
            event_metrics = json.load(stream)
        event_metrics["training_metrics_sha256"] = sha256_file(metrics_path)
        with event_metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(event_metrics, stream)

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run_ok, run_missing_diagnostics], shared_expected, adapter)
        except RuntimeError as exc:
            raised = str(run_missing_diagnostics) in str(exc) and "val" in str(exc)
        return _check(
            "final.<split> sem confusion_matrix/per_class levanta RuntimeError "
            "nomeando o run_dir e o split",
            raised,
        )


def _selftest_csv_row_inventory_matches_schema() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "seed1"
        run2 = root / "seed2"
        y_true_1, y_pred_1 = _synthetic_classification_arrays(offset=1)
        y_true_2, y_pred_2 = _synthetic_classification_arrays(offset=3)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true_1, y_pred=y_pred_1, val_event=event, test_event=event
        )
        _write_synthetic_seed_run(
            run2, seed=2, y_true=y_true_2, y_pred=y_pred_2, val_event=event, test_event=event
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        _report, csv_rows = _summarize([run1, run2], shared_expected, adapter)

        n_restricted = len(RESTRICTED_CLASSES)
        n_labels = len(LE2I_LABEL_NAMES)
        expected_n_rows = (
            len(CLASSIFICATION_SPLITS) * (1 + n_restricted)
            + len(CLASSIFICATION_SPLITS) * n_labels * len(PER_CLASS_METRIC_FIELDS)
            + len(CLASSIFICATION_SPLITS) * len(BINARY_FIELDS)
            + len(EVENT_SPLITS) * len(EVENT_SCALAR_FIELDS)
        )
        count_ok = len(csv_rows) == expected_n_rows

        rows_by_key = {
            (row["split"], row["metric_group"], row["entity"], row["metric"]): row
            for row in csv_rows
        }
        expected_present = [
            ("val", "classification", "", "macro_f1_restricted"),
            ("val", "classification", "fall", "f1_by_class"),
            ("val", "per_class", "fall", "precision"),
            ("val", "binary", "", "tp"),
            ("val", "events", "", "sensitivity"),
            ("val", "events", "", "latency_seconds_mean"),
        ]
        presence_ok = all(key in rows_by_key for key in expected_present)
        n_ok = all(rows_by_key[key]["n"] == 2 for key in expected_present)

        return _check(
            "inventário de linhas do CSV bate com a regra do esquema "
            "(contagem derivada da regra, independente do número de seeds) "
            "e tuplas (split, metric_group, entity, metric) esperadas estão "
            "presentes com n==2",
            count_ok and presence_ok and n_ok,
        )


def _selftest_duplicate_seed_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "run_a"
        run2 = root / "run_b"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run1, seed=5, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )
        _write_synthetic_seed_run(
            run2, seed=5, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run1, run2], shared_expected, adapter)
        except RuntimeError as exc:
            raised = "seed" in str(exc) and str(run1) in str(exc) and str(run2) in str(exc)
        return _check(
            "seed duplicada entre run_dirs levanta RuntimeError nomeando "
            "os dois run_dirs envolvidos",
            raised,
        )


def _selftest_non_seed_config_divergence_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "run_a"
        run2 = root / "run_b"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event, epochs=1
        )
        _write_synthetic_seed_run(
            run2, seed=2, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event, epochs=2
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run1, run2], shared_expected, adapter)
        except RuntimeError as exc:
            raised = str(run2) in str(exc)
        return _check(
            "divergência de configuração fora do campo seed (epochs) "
            "levanta RuntimeError nomeando o run_dir divergente",
            raised,
        )


def _selftest_malformed_run_dir_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_ok = root / "run_ok"
        run_missing_event_metrics = root / "run_missing_event_metrics"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run_ok, seed=1, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )
        _write_synthetic_seed_run(
            run_missing_event_metrics,
            seed=2,
            y_true=y_true,
            y_pred=y_pred,
            val_event=event,
            test_event=event,
        )
        (run_missing_event_metrics / "event_metrics.json").unlink()

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run_ok, run_missing_event_metrics], shared_expected, adapter)
        except RuntimeError as exc:
            raised = str(run_missing_event_metrics) in str(exc)
        return _check(
            "run_dir malformado (event_metrics.json ausente) levanta "
            "RuntimeError nomeando o run_dir",
            raised,
        )


def _selftest_fewer_than_min_seeds_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "run_a"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )
        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run1], shared_expected, adapter)
        except ValueError:
            raised = True
        return _check(
            f"menos de MIN_SEEDS={MIN_SEEDS} run_dirs levanta ValueError",
            raised,
        )


def _selftest_writer_honors_force() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        report = {"n_seeds": 1, "seeds": [], "aggregate": {}, "arm": "A", "config_fingerprint_sha256": "x"}
        csv_rows: list[dict] = []

        wrote_first = _write_multiseed_summary_outputs(output_dir, report, csv_rows, force=False)
        json_path = output_dir / MULTISEED_SUMMARY_JSON_FILE
        original_text = json_path.read_text(encoding="utf-8")

        changed_report = dict(report)
        changed_report["n_seeds"] = 999
        wrote_second_without_force = _write_multiseed_summary_outputs(
            output_dir, changed_report, csv_rows, force=False
        )
        unchanged = json_path.read_text(encoding="utf-8") == original_text

        wrote_third_with_force = _write_multiseed_summary_outputs(
            output_dir, changed_report, csv_rows, force=True
        )
        overwritten = "999" in json_path.read_text(encoding="utf-8")

        canonical_artifacts = (
            "config.yaml",
            "metrics.json",
            "checkpoint.pt",
            "alarm_protocol.yaml",
            "event_metrics.json",
        )
        no_canonical_artifacts = not any(
            (output_dir / name).exists() for name in canonical_artifacts
        )

        ok = (
            wrote_first
            and not wrote_second_without_force
            and unchanged
            and wrote_third_with_force
            and overwritten
            and no_canonical_artifacts
        )
        return _check(
            "writer honra --force: sem --force não sobrescreve nem sinaliza "
            "sucesso; com --force sobrescreve; nunca escreve artefatos "
            "canônicos de treino/avaliação",
            ok,
        )


def _synthetic_pair(root: Path, arm: str) -> tuple[list[Path], RunConfig]:
    y_true, y_pred = _synthetic_classification_arrays(offset=0)
    event = _build_event_split(1, 1, 1.0, 1.0, 1.0, 0.0, 0, 1.0, 1.0)
    run_dirs = [root / "seed1", root / "seed2"]
    first = _write_synthetic_seed_run(
        run_dirs[0], 1, y_true, y_pred, event, event, arm=arm
    )
    _write_synthetic_seed_run(run_dirs[1], 2, y_true, y_pred, event, event, arm=arm)
    return run_dirs, replace(first, seed=42)


def _rewrite_synthetic_config(run_dir: Path, changes: dict) -> None:
    config_path = run_dir / "config.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config.update(changes)
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    metrics["config_sha256"] = sha256_file(config_path)
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream)
    event_path = run_dir / "event_metrics.json"
    if event_path.is_file():
        with event_path.open(encoding="utf-8") as stream:
            events = json.load(stream)
        events["training_metrics_sha256"] = sha256_file(metrics_path)
        with event_path.open("w", encoding="utf-8") as stream:
            json.dump(events, stream)


def _selftest_all_arms() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm in ARMS:
            run_dirs, expected = _synthetic_pair(root / arm, arm)
            report, rows = _summarize(run_dirs, expected, adapter)
            has_events = arm in EVENT_ARMS
            if (
                report["arm"] != arm
                or report["n_seeds"] != 2
                or ("events" in report["aggregate"]) != has_events
                or any(("events" in seed) != has_events for seed in report["seeds"])
                or any(row["metric_group"] == "events" for row in rows) != has_events
            ):
                return _check("todas as armas preservam seu contrato de métricas", False)
            _rewrite_synthetic_config(run_dirs[1], {"lr": 0.5})
            try:
                _summarize(run_dirs, expected, adapter)
            except RuntimeError as exc:
                if "lr" not in str(exc):
                    return _check("configuração científica divergente é rejeitada", False)
            else:
                return _check("configuração científica divergente é rejeitada", False)
            _rewrite_synthetic_config(run_dirs[1], {"lr": expected.lr})
            if has_events:
                event_path = run_dirs[1] / "event_metrics.json"
                with event_path.open(encoding="utf-8") as stream:
                    event_report = json.load(stream)
                event_report["checkpoint_sha256"] = "incompatível"
                with event_path.open("w", encoding="utf-8") as stream:
                    json.dump(event_report, stream)
                try:
                    _summarize(run_dirs, expected, adapter)
                except RuntimeError as exc:
                    if "checkpoint_sha256" not in str(exc):
                        return _check("hash de evento inválido é rejeitado", False)
                else:
                    return _check("hash de evento inválido é rejeitado", False)
                (run_dirs[1] / "event_metrics.json").unlink()
                try:
                    _summarize(run_dirs, expected, adapter)
                except RuntimeError as exc:
                    if "event_metrics.json ausente" not in str(exc):
                        return _check("armas com evento exigem evidência completa", False)
                else:
                    return _check("armas com evento exigem evidência completa", False)
            else:
                if (run_dirs[1] / "event_metrics.json").exists():
                    return _check("C0 dispensa artefatos de evento", False)
                (run_dirs[1] / "event_metrics.json").write_text("{}", encoding="utf-8")
                c0_report, c0_rows = _summarize(run_dirs, expected, adapter)
                if "events" in c0_report["aggregate"] or any(
                    row["metric_group"] == "events" for row in c0_rows
                ):
                    return _check("C0 omite evento mesmo se houver arquivo extra", False)
        return _check(
            "A/B0/B1/C0/C1: classificação validada; evento obrigatório exceto C0",
            True,
        )


def _selftest_arm_and_config_guards() -> bool:
    adapter = get_dataset("le2i")
    cv_rejected = True
    for arm in ARMS[1:]:
        try:
            _resolve_shared_expected("le2i-cv", arm)
        except ValueError:
            continue
        cv_rejected = False
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs, expected = _synthetic_pair(root / "b0", "B0")
        duplicate_path = False
        try:
            _summarize([runs[0], runs[0]], expected, adapter)
        except ValueError as exc:
            duplicate_path = "duplicado" in str(exc)
        _rewrite_synthetic_config(runs[1], {"seed": 1})
        duplicate_seed = False
        try:
            _summarize(runs, expected, adapter)
        except RuntimeError as exc:
            duplicate_seed = "seed duplicada" in str(exc)
        _rewrite_synthetic_config(runs[1], {"seed": 2, "lr": 0.5})
        mismatch = False
        try:
            _summarize(runs, expected, adapter)
        except RuntimeError as exc:
            mismatch = "lr" in str(exc)
        _rewrite_synthetic_config(runs[1], {"lr": expected.lr, "trainable_param_count": 123})
        allowed_report, _ = _summarize(runs, expected, adapter)
        audit_allowed = allowed_report["n_seeds"] == 2

        c0_runs, _ = _synthetic_pair(root / "c0", "C0")
        mixed = False
        try:
            _summarize([runs[0], c0_runs[0]], expected, adapter)
        except RuntimeError as exc:
            mixed = str(c0_runs[0]) in str(exc)
        return _check(
            "paths/seeds duplicados, arma mista e configuração científica divergente rejeitados; campo de auditoria permitido",
            duplicate_path and duplicate_seed and mismatch and mixed and audit_allowed and cv_rejected,
        )


def _selftest_arm_a_output_compatibility() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs, expected = _synthetic_pair(root / "a", "A")
        default_out = root / "default"
        explicit_out = root / "explicit"
        with patch.object(sys.modules[__name__], "_resolve_shared_expected", return_value=expected):
            run_summarize("le2i", runs, default_out, False)
            run_summarize("le2i", runs, explicit_out, False, arm="A")
        with (default_out / MULTISEED_SUMMARY_JSON_FILE).open(encoding="utf-8") as stream:
            report = json.load(stream)
        rows = pd.read_csv(default_out / MULTISEED_SUMMARY_CSV_FILE)
        valid = (
            list(report) == ["arm", "config_fingerprint_sha256", "n_seeds", "seeds", "aggregate"]
            and list(report["aggregate"]) == ["classification", "per_class", "binary_fall_fallen", "events"]
            and len(rows) == 340
            and list(rows) == CSV_COLUMNS
            and all(
                (default_out / name).read_bytes() == (explicit_out / name).read_bytes()
                for name in (MULTISEED_SUMMARY_JSON_FILE, MULTISEED_SUMMARY_CSV_FILE)
            )
        )
        return _check("A mantém esquema e bytes JSON/CSV do caminho padrão", valid)


def run_multiseed_summary_selftest() -> bool:
    checks = [
        _selftest_aggregate_stats_known_array(),
        _selftest_two_valid_seed_runs_aggregate(),
        _selftest_seed_blocks_round_trip_and_new_aggregates_exist(),
        _selftest_binary_fall_fallen_matches_independent_recomputation(),
        _selftest_missing_diagnostics_raises(),
        _selftest_csv_row_inventory_matches_schema(),
        _selftest_duplicate_seed_raises(),
        _selftest_non_seed_config_divergence_raises(),
        _selftest_malformed_run_dir_raises(),
        _selftest_fewer_than_min_seeds_raises(),
        _selftest_writer_honors_force(),
        _selftest_all_arms(),
        _selftest_arm_and_config_guards(),
        _selftest_arm_a_output_compatibility(),
    ]
    ok = all(checks)
    if not ok:
        print("\nmultiseed_summary selftest FALHOU", file=sys.stderr)
    else:
        print("\nmultiseed_summary selftest OK: todas as checagens passaram")
    return ok


def run_selftest() -> None:
    if not run_multiseed_summary_selftest():
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help=(
            "Agrega classificação e, quando aplicável, eventos de múltiplos "
            "runs da mesma arma com seeds distintas"
        ),
    )
    summarize_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    summarize_parser.add_argument("--arm", default="A", choices=ARMS)
    summarize_parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        dest="run_dirs",
        required=True,
        help="Repetível: um run_dir por seed (mínimo de %d)" % MIN_SEEDS,
    )
    summarize_parser.add_argument("--output-dir", type=Path, required=True)
    summarize_parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve multiseed_summary.json/.csv já existentes",
    )
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas do sumário multi-seed"
    )

    args = parser.parse_args()
    if args.command == "summarize":
        run_summarize(
            dataset_name=args.dataset,
            run_dirs=args.run_dirs,
            output_dir=args.output_dir,
            force=args.force,
            arm=args.arm,
        )
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
