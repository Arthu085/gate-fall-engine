"""Diagnóstico somente leitura dos erros de classificação das armas A, B0, B1, C0 e C1.

Consome artefatos experimentais já existentes — referências canônicas do
Le2i-CS, runs locais e sumários multi-seed fornecidos explicitamente — sem
rodar inferência, treino, ajuste, seleção de checkpoint ou promoção. Toda
métrica é recalculada a partir da `confusion_matrix` persistida com as mesmas
funções de `gatefall.train.shared.metrics` e conferida contra os valores
gravados; a Macro-F1 restrita usa exatamente `RESTRICTED_CLASSES`. Os
protocolos `le2i` (Le2i-CS) e `le2i-cv` (Le2i-CV) são diagnosticados em blocos
separados e nunca agregados entre si. O teste é usado apenas descritivamente.
"""

import argparse
import json
import math
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gatefall.config import NUM_CLASSES
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.multiseed_summary import (
    ARMS,
    CLASSIFICATION_SPLITS,
    PER_CLASS_METRIC_FIELDS,
    _aggregate_stats,
)
from gatefall.hashing import sha256_file
from gatefall.runs import (
    ARM_RUN_DIR_NAMES,
    LOCAL_RUN_ROOTS,
    REFERENCE_RUN_ROOT,
    REPOSITORY_ROOT,
    validate_local_run_dir,
)
from gatefall.train.shared.metrics import (
    RESTRICTED_CLASSES,
    classification_summary,
    restricted_macro_f1,
)

DIAGNOSIS_JSON_FILE = "classification_diagnosis.json"
DIAGNOSIS_CSV_FILE = "classification_diagnosis.csv"
DIAGNOSIS_SUMMARY_FILE = "classification_diagnosis.md"

PROTOCOL_NAMES = {"le2i": "Le2i-CS", "le2i-cv": "Le2i-CV"}
PROTOCOL_HELD_OUT_FACTOR = {"le2i": "sujeitos", "le2i-cv": "ambientes"}
CANONICAL_REFERENCE_DATASET = "le2i"
DEFAULT_REFERENCE_ROOT = REFERENCE_RUN_ROOT / "le2i"

CANONICAL_REFERENCE = "canonical_reference"
LOCAL_RUN = "local_run"
MULTISEED_SUMMARY = "multiseed_summary"

EXPECTED_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
GAP_PAIRS: tuple[tuple[str, str], ...] = (("train", "val"), ("train", "test"), ("val", "test"))
TOP_CONFUSIONS = 5
REPORTED_ITEMS = 3
METRIC_TOLERANCE = 1e-9

# Limiares que apenas disparam hipóteses no resumo; não filtram nem
# selecionam nenhum resultado.
LOW_SUPPORT_FRACTION = 0.02
LARGE_GAP = 0.2
UNSTABLE_SEED_STD = 0.05
# Cada classe de transição do Le2i com os estados que a precedem ou sucedem
# imediatamente na mesma ação.
TRANSITION_ADJACENT_STATES: dict[str, frozenset[str]] = {
    "fall": frozenset({"walk", "standing", "fallen"}),
    "sit_down": frozenset({"walk", "standing", "sitting"}),
    "stand_up": frozenset({"sitting", "fallen", "lying", "standing", "walk"}),
    "lie_down": frozenset({"sitting", "standing", "lying"}),
}
RESIDUAL_LABEL = "other"

CSV_COLUMNS = [
    "dataset",
    "protocol",
    "arm",
    "source_kind",
    "source",
    "seed",
    "split",
    "record",
    "entity",
    "metric",
    "value",
]


@dataclass(frozen=True)
class DiagnosisInput:
    dataset: str
    arm: str
    path: Path


def display_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT):
        return str(resolved.relative_to(REPOSITORY_ROOT))
    return str(path)


def _context(dataset: str, arm: str, path: Path) -> str:
    return f"{PROTOCOL_NAMES[dataset]}/{arm} ({path})"


def _validate_dataset_and_arm(dataset: str, arm: str) -> None:
    if dataset not in PROTOCOL_NAMES:
        raise ValueError(
            f"dataset não suportado: {dataset!r}; opções: {', '.join(SUPPORTED_DATASET_IDENTIFIERS)}"
        )
    if arm not in ARMS:
        raise ValueError(f"arma não suportada: {arm!r}; opções: {', '.join(ARMS)}")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"arquivo ausente: {path}")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: esperado objeto JSON")
    return data


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"arquivo ausente: {path}")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: esperado mapeamento YAML")
    return data


def validate_confusion_matrix(matrix: object, context: str) -> list[list[int]]:
    if not isinstance(matrix, list) or len(matrix) != NUM_CLASSES:
        raise ValueError(f"{context}: confusion_matrix deve ter {NUM_CLASSES} linhas")
    for row in matrix:
        if not isinstance(row, list) or len(row) != NUM_CLASSES:
            raise ValueError(f"{context}: confusion_matrix deve ser {NUM_CLASSES}x{NUM_CLASSES}")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{context}: confusion_matrix com célula inválida {value!r} "
                    "(esperado inteiro não negativo)"
                )
    return matrix


def labels_from_confusion_matrix(matrix: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(matrix, dtype=np.int64)
    true_ids, predicted_ids = np.indices(counts.shape)
    flat_counts = counts.ravel()
    return np.repeat(true_ids.ravel(), flat_counts), np.repeat(predicted_ids.ravel(), flat_counts)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _stored_mismatches(
    stored: dict,
    macro_f1: float,
    f1_by_class: dict[int, float],
    per_class: dict[str, dict],
    label_names: tuple[str, ...],
) -> list[str]:
    mismatches: list[str] = []

    def compare(field: str, stored_value: object, recomputed: float) -> None:
        stored_number = _finite_number(stored_value)
        if stored_number is None:
            mismatches.append(f"{field}: valor gravado ausente, não numérico ou não finito ({stored_value!r})")
        elif abs(stored_number - recomputed) > METRIC_TOLERANCE:
            mismatches.append(f"{field}: gravado {stored_value}, recalculado {recomputed}")

    compare("macro_f1_restricted", stored.get("macro_f1_restricted"), macro_f1)
    stored_f1_by_class = stored.get("f1_by_class") or {}
    for class_id in RESTRICTED_CLASSES:
        compare(f"f1_by_class[{class_id}]", stored_f1_by_class.get(str(class_id)), f1_by_class[class_id])
    stored_support = stored.get("support") or {}
    stored_per_class = stored.get("per_class") or {}
    for label in label_names:
        compare(f"support[{label}]", stored_support.get(label), per_class[label]["support"])
        stored_class = stored_per_class.get(label) or {}
        for field in PER_CLASS_METRIC_FIELDS:
            compare(f"per_class[{label}].{field}", stored_class.get(field), per_class[label][field])
    return mismatches


def dominant_confusions(
    matrix: list[list[int]], label_names: tuple[str, ...], top_k: int = TOP_CONFUSIONS
) -> list[dict]:
    counts = np.asarray(matrix, dtype=np.int64)
    true_totals = counts.sum(axis=1)
    predicted_totals = counts.sum(axis=0)
    total_errors = int(counts.sum() - np.trace(counts))
    cells = sorted(
        (
            (-int(counts[true_id, predicted_id]), true_id, predicted_id)
            for true_id in range(counts.shape[0])
            for predicted_id in range(counts.shape[1])
            if true_id != predicted_id and counts[true_id, predicted_id] > 0
        )
    )
    return [
        {
            "true_id": true_id,
            "true_label": label_names[true_id],
            "predicted_id": predicted_id,
            "predicted_label": label_names[predicted_id],
            "count": -negative_count,
            "share_of_true_support": -negative_count / int(true_totals[true_id]),
            "share_of_predicted": -negative_count / int(predicted_totals[predicted_id]),
            "share_of_all_errors": -negative_count / total_errors,
        }
        for negative_count, true_id, predicted_id in cells[:top_k]
    ]


def macro_f1_deficit(f1_by_class: dict[int, float], label_names: tuple[str, ...]) -> list[dict]:
    # 1 − Macro-F1 restrita = Σ_c (1 − F1_c) / |RESTRICTED_CLASSES|: cada classe
    # restrita contribui (1 − F1_c) / |RESTRICTED_CLASSES| para o déficit.
    n_classes = len(RESTRICTED_CLASSES)
    deficits = {class_id: (1.0 - f1_by_class[class_id]) / n_classes for class_id in RESTRICTED_CLASSES}
    total = sum(deficits.values())
    ranked = sorted(RESTRICTED_CLASSES, key=lambda class_id: (-deficits[class_id], class_id))
    return [
        {
            "id": class_id,
            "label": label_names[class_id],
            "f1": f1_by_class[class_id],
            "deficit": deficits[class_id],
            "share_of_total_deficit": deficits[class_id] / total if total > 0 else None,
        }
        for class_id in ranked
    ]


def diagnose_split(split_block: object, label_names: tuple[str, ...], context: str) -> dict:
    if not isinstance(split_block, dict):
        raise ValueError(f"{context}: bloco de split ausente ou inválido")
    missing = [
        field
        for field in ("macro_f1_restricted", "f1_by_class", "support", "confusion_matrix", "per_class")
        if field not in split_block
    ]
    if missing:
        raise ValueError(f"{context}: campos ausentes {missing}")
    matrix = validate_confusion_matrix(split_block["confusion_matrix"], context)
    y_true, y_pred = labels_from_confusion_matrix(matrix)
    summary = classification_summary(y_true, y_pred, label_names, NUM_CLASSES)
    macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred, NUM_CLASSES)
    per_class: dict[str, dict] = summary["per_class"]

    mismatches = _stored_mismatches(split_block, macro_f1, f1_by_class, per_class, label_names)
    if mismatches:
        raise ValueError(
            f"{context}: métricas gravadas divergem do recálculo a partir da "
            f"confusion_matrix: {'; '.join(mismatches[:5])}"
        )

    n_windows = len(y_true)
    restricted = set(RESTRICTED_CLASSES)
    for label, row in per_class.items():
        row["included_in_macro_f1"] = row["id"] in restricted
        row["support_fraction"] = row["support"] / n_windows if n_windows > 0 else None
    return {
        "n_windows": n_windows,
        "macro_f1_restricted": macro_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "dominant_confusions": dominant_confusions(matrix, label_names),
        "macro_f1_deficit": macro_f1_deficit(f1_by_class, label_names),
    }


def generalization_gaps(splits: dict[str, dict], label_names: tuple[str, ...]) -> dict:
    return {
        "macro_f1_restricted": {
            f"{first}_minus_{second}": splits[first]["macro_f1_restricted"]
            - splits[second]["macro_f1_restricted"]
            for first, second in GAP_PAIRS
        },
        "f1_by_class": {
            label_names[class_id]: {
                f"{first}_minus_{second}": splits[first]["per_class"][label_names[class_id]]["f1"]
                - splits[second]["per_class"][label_names[class_id]]["f1"]
                for first, second in GAP_PAIRS
            }
            for class_id in RESTRICTED_CLASSES
        },
    }


def _diagnose_classification_splits(
    final: object, label_names: tuple[str, ...], context: str
) -> dict[str, dict]:
    if not isinstance(final, dict):
        raise ValueError(f"{context}: bloco final de classificação ausente")
    splits: dict[str, dict] = {}
    for split in CLASSIFICATION_SPLITS:
        if split not in final:
            raise ValueError(f"{context}: split {split!r} ausente")
        splits[split] = diagnose_split(final[split], label_names, f"{context} [{split}]")
    return splits


def _pose_stats_name(config: dict) -> str | None:
    value = config.get("standardization_stats_path") or config.get("pose_standardization_stats_path")
    return Path(value).name if isinstance(value, str) else None


def _validate_run_identity(config: dict, dataset: str, arm: str, context: str) -> None:
    if config.get("arm") != arm:
        raise ValueError(f"{context}: config.yaml declara arma {config.get('arm')!r}, esperado {arm!r}")
    expected_stats = get_dataset(dataset).pose_stats_path.name
    stats_name = _pose_stats_name(config)
    if stats_name != expected_stats:
        raise ValueError(
            f"{context}: estatísticas de pose {stats_name!r} não pertencem ao protocolo "
            f"{PROTOCOL_NAMES[dataset]} (esperado {expected_stats!r})"
        )


def _validate_classification_report(
    report: dict, splits: dict[str, dict], context: str
) -> dict | None:
    report_splits = report.get("splits") or {}
    for split in CLASSIFICATION_SPLITS:
        report_matrix = (report_splits.get(split) or {}).get("confusion_matrix")
        if report_matrix != splits[split]["confusion_matrix"]:
            raise ValueError(
                f"{context}: classification_report.json diverge de metrics.json em {split}"
            )
    verification = report.get("verification_against_metrics_json") or {}
    if verification.get("ok") is not True:
        raise ValueError(f"{context}: classification_report.json sem verificação ok contra metrics.json")
    return report.get("macro_f1_policy")


def diagnose_run(
    run_dir: Path, dataset: str, arm: str, source_kind: str, label_names: tuple[str, ...]
) -> tuple[dict, list[Path]]:
    context = _context(dataset, arm, run_dir)
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "classification_report.json"
    config = _read_yaml(config_path)
    metrics = _read_json(metrics_path)
    _validate_run_identity(config, dataset, arm, context)
    splits = _diagnose_classification_splits(metrics.get("final"), label_names, context)

    files_read = [config_path, metrics_path]
    macro_f1_policy = None
    if report_path.is_file():
        macro_f1_policy = _validate_classification_report(_read_json(report_path), splits, context)
        files_read.append(report_path)

    entry = {
        "dataset": dataset,
        "protocol": PROTOCOL_NAMES[dataset],
        "arm": arm,
        "source_kind": source_kind,
        "source": display_path(run_dir),
        "seed": config.get("seed"),
        "run_name": metrics.get("run_name"),
        "checkpoint_sha256": metrics.get("checkpoint_sha256"),
        "macro_f1_policy": macro_f1_policy,
        "splits": splits,
        "generalization_gaps": generalization_gaps(splits, label_names),
    }
    return entry, files_read


def run_dir_protocol(run_dir: str) -> str | None:
    path = Path(run_dir)
    resolved = (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()
    roots = [
        *((dataset, (REPOSITORY_ROOT / root).resolve()) for dataset, root in LOCAL_RUN_ROOTS.items()),
        (CANONICAL_REFERENCE_DATASET, DEFAULT_REFERENCE_ROOT.resolve()),
    ]
    for dataset, root in roots:
        if resolved == root or root in resolved.parents:
            return dataset
    return None


def _seed_report_fields(seed_report: object, context: str) -> tuple[int, str, dict]:
    if not isinstance(seed_report, dict):
        raise ValueError(f"{context}: entrada de seed inválida")
    seed = seed_report.get("seed")
    run_dir = seed_report.get("run_dir")
    classification = seed_report.get("classification")
    if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(run_dir, str):
        raise ValueError(f"{context}: entrada de seed sem seed/run_dir válidos")
    if not isinstance(classification, dict):
        raise ValueError(f"{context}: seed {seed} sem bloco classification")
    return seed, run_dir, classification


def _seed_variation(seed_entries: list[dict], label_names: tuple[str, ...]) -> dict:
    variation: dict[str, dict] = {}
    for split in CLASSIFICATION_SPLITS:
        split_entries = [entry["splits"][split] for entry in seed_entries]
        mean_f1 = {
            class_id: float(np.mean([s["per_class"][label_names[class_id]]["f1"] for s in split_entries]))
            for class_id in RESTRICTED_CLASSES
        }
        pooled = np.sum([np.asarray(s["confusion_matrix"]) for s in split_entries], axis=0)
        recurrence = Counter(
            (item["true_label"], item["predicted_label"])
            for s in split_entries
            for item in s["dominant_confusions"]
        )
        variation[split] = {
            "macro_f1_restricted": _aggregate_stats([s["macro_f1_restricted"] for s in split_entries]),
            "per_class": {
                label: {
                    metric: _aggregate_stats([s["per_class"][label][metric] for s in split_entries])
                    for metric in ("precision", "recall", "f1", "support")
                }
                for label in label_names
            },
            "macro_f1_deficit_from_mean_f1": macro_f1_deficit(mean_f1, label_names),
            "pooled_dominant_confusions": dominant_confusions(pooled.tolist(), label_names),
            "confusion_recurrence_in_top_k": [
                {"true_label": true_label, "predicted_label": predicted_label, "n_seeds": n_seeds}
                for (true_label, predicted_label), n_seeds in sorted(
                    recurrence.items(), key=lambda item: (-item[1], item[0])
                )
            ],
        }
    variation["generalization_gaps"] = {
        pair: _aggregate_stats([entry["generalization_gaps"]["macro_f1_restricted"][pair] for entry in seed_entries])
        for pair in seed_entries[0]["generalization_gaps"]["macro_f1_restricted"]
    }
    return variation


def _verify_summary_aggregate(summary: dict, variation: dict, context: str) -> None:
    for split in CLASSIFICATION_SPLITS:
        try:
            stored = summary["aggregate"]["classification"][split]["macro_f1_restricted"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{context}: aggregate.classification.{split} ausente") from exc
        recomputed = variation[split]["macro_f1_restricted"]
        for field in ("n", "mean"):
            stored_value = stored.get(field) if isinstance(stored, dict) else None
            stored_number = _finite_number(stored_value)
            if stored_number is None or abs(stored_number - float(recomputed[field])) > METRIC_TOLERANCE:
                raise ValueError(
                    f"{context}: aggregate.{split}.macro_f1_restricted.{field} gravado "
                    f"{stored_value} diverge do recálculo {recomputed[field]}"
                )


def diagnose_multiseed_summary(
    summary_path: Path, dataset: str, arm: str, label_names: tuple[str, ...]
) -> tuple[dict, list[Path]]:
    context = _context(dataset, arm, summary_path)
    summary = _read_json(summary_path)
    if summary.get("arm") != arm:
        raise ValueError(f"{context}: sumário declara arma {summary.get('arm')!r}, esperado {arm!r}")
    seed_reports = summary.get("seeds")
    if not isinstance(seed_reports, list) or not seed_reports:
        raise ValueError(f"{context}: lista de seeds ausente ou vazia")
    if summary.get("n_seeds") != len(seed_reports):
        raise ValueError(f"{context}: n_seeds={summary.get('n_seeds')} diverge de {len(seed_reports)} seeds")

    seed_entries: list[dict] = []
    unconfirmed_run_dirs: list[str] = []
    seen: set[int] = set()
    for seed_report in seed_reports:
        seed, run_dir, classification = _seed_report_fields(seed_report, context)
        if seed in seen:
            raise ValueError(f"{context}: seed duplicada {seed}")
        seen.add(seed)
        run_protocol = run_dir_protocol(run_dir)
        if run_protocol is None:
            unconfirmed_run_dirs.append(run_dir)
        elif run_protocol != dataset:
            raise ValueError(
                f"{context}: seed {seed} aponta para {run_dir}, do protocolo "
                f"{PROTOCOL_NAMES[run_protocol]}"
            )
        splits = _diagnose_classification_splits(classification, label_names, f"{context} seed {seed}")
        seed_entries.append(
            {
                "seed": seed,
                "run_dir": run_dir,
                "checkpoint_sha256": seed_report.get("checkpoint_sha256"),
                "splits": splits,
                "generalization_gaps": generalization_gaps(splits, label_names),
            }
        )

    seed_entries.sort(key=lambda entry: entry["seed"])
    variation = _seed_variation(seed_entries, label_names)
    _verify_summary_aggregate(summary, variation, context)
    seeds = [entry["seed"] for entry in seed_entries]
    entry = {
        "dataset": dataset,
        "protocol": PROTOCOL_NAMES[dataset],
        "arm": arm,
        "source_kind": MULTISEED_SUMMARY,
        "source": display_path(summary_path),
        "config_fingerprint_sha256": summary.get("config_fingerprint_sha256"),
        "n_seeds": len(seed_entries),
        "seeds": seeds,
        "missing_expected_seeds": [seed for seed in EXPECTED_SEEDS if seed not in seen],
        "unexpected_seeds": [seed for seed in seeds if seed not in EXPECTED_SEEDS],
        "protocol_unconfirmed_run_dirs": unconfirmed_run_dirs,
        "seed_variation": variation,
        "per_seed": seed_entries,
    }
    return entry, [summary_path]


def _empty_protocol_block(dataset: str) -> dict:
    return {
        "name": PROTOCOL_NAMES[dataset],
        "held_out_factor": PROTOCOL_HELD_OUT_FACTOR[dataset],
        "arms": {arm: {"runs": [], "multiseed_summaries": []} for arm in ARMS},
    }


def _missing_inputs(protocols: dict, reference_root: Path | None) -> list[dict]:
    missing: list[dict] = []
    for dataset, block in protocols.items():
        for arm, sources in block["arms"].items():
            has_reference = any(run["source_kind"] == CANONICAL_REFERENCE for run in sources["runs"])
            if dataset == CANONICAL_REFERENCE_DATASET and not has_reference:
                reason = (
                    "referências canônicas desabilitadas"
                    if reference_root is None
                    else f"metrics.json ausente em {display_path(reference_root / ARM_RUN_DIR_NAMES[arm])}"
                )
                missing.append({"dataset": dataset, "arm": arm, "input": CANONICAL_REFERENCE, "reason": reason})
            if dataset != CANONICAL_REFERENCE_DATASET and not sources["runs"]:
                missing.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "input": LOCAL_RUN,
                        "reason": "nenhum run local fornecido; não há referência canônica versionada para Le2i-CV",
                    }
                )
            if not sources["multiseed_summaries"]:
                missing.append(
                    {"dataset": dataset, "arm": arm, "input": MULTISEED_SUMMARY, "reason": "nenhum sumário multi-seed fornecido"}
                )
            for summary in sources["multiseed_summaries"]:
                if summary["missing_expected_seeds"]:
                    missing.append(
                        {
                            "dataset": dataset,
                            "arm": arm,
                            "input": "seeds",
                            "reason": f"seeds esperadas ausentes em {summary['source']}: {summary['missing_expected_seeds']}",
                        }
                    )
    return missing


def _reference_matches(protocols: dict) -> list[dict]:
    matches: list[dict] = []
    for dataset, block in protocols.items():
        for arm, sources in block["arms"].items():
            references = [run for run in sources["runs"] if run["source_kind"] == CANONICAL_REFERENCE]
            for reference in references:
                for summary in sources["multiseed_summaries"]:
                    for seed_entry in summary["per_seed"]:
                        same_checkpoint = (
                            reference["checkpoint_sha256"] is not None
                            and reference["checkpoint_sha256"] == seed_entry["checkpoint_sha256"]
                        )
                        same_matrices = all(
                            reference["splits"][split]["confusion_matrix"]
                            == seed_entry["splits"][split]["confusion_matrix"]
                            for split in CLASSIFICATION_SPLITS
                        )
                        if same_checkpoint or same_matrices:
                            matches.append(
                                {
                                    "dataset": dataset,
                                    "arm": arm,
                                    "reference": reference["source"],
                                    "summary": summary["source"],
                                    "seed": seed_entry["seed"],
                                    "same_checkpoint_sha256": same_checkpoint,
                                    "same_confusion_matrices": same_matrices,
                                }
                            )
    return matches


def _fmt(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.4f}"


def _source_label(entry: dict) -> str:
    if entry["source_kind"] == MULTISEED_SUMMARY:
        return f"sumário multi-seed ({entry['n_seeds']} seeds: {entry['seeds']})"
    kind = "referência canônica" if entry["source_kind"] == CANONICAL_REFERENCE else "run local"
    return f"{kind} (seed {entry['seed']})"


def _split_view(entry: dict, split: str) -> tuple[float | None, list[dict], list[dict], dict]:
    if entry["source_kind"] == MULTISEED_SUMMARY:
        variation = entry["seed_variation"][split]
        per_class = {label: {"f1": stats["f1"]["mean"], "support": stats["support"]["mean"]}
                     for label, stats in variation["per_class"].items()}
        return (
            variation["macro_f1_restricted"]["mean"],
            variation["macro_f1_deficit_from_mean_f1"],
            variation["pooled_dominant_confusions"],
            per_class,
        )
    split_diagnosis = entry["splits"][split]
    return (
        split_diagnosis["macro_f1_restricted"],
        split_diagnosis["macro_f1_deficit"],
        split_diagnosis["dominant_confusions"],
        split_diagnosis["per_class"],
    )


def _train_minus_test(entry: dict) -> float | None:
    if entry["source_kind"] == MULTISEED_SUMMARY:
        return entry["seed_variation"]["generalization_gaps"]["train_minus_test"]["mean"]
    return entry["generalization_gaps"]["macro_f1_restricted"]["train_minus_test"]


def _entry_observations(entry: dict) -> list[str]:
    observations: list[str] = []
    macro = {split: _split_view(entry, split)[0] for split in CLASSIFICATION_SPLITS}
    prefix = "média entre seeds de " if entry["source_kind"] == MULTISEED_SUMMARY else ""
    observations.append(
        f"Macro-F1 restrita ({prefix}train/val/test): "
        + " / ".join(_fmt(macro[split]) for split in CLASSIFICATION_SPLITS)
        + f"; lacuna train−test {_fmt(_train_minus_test(entry))}."
    )
    for split in ("val", "test"):
        _, deficit, confusions, _ = _split_view(entry, split)
        top_deficit = ", ".join(
            f"{item['label']} (F1 {_fmt(item['f1'])}, {_fmt(item['share_of_total_deficit'])} do déficit)"
            for item in deficit[:REPORTED_ITEMS]
        )
        observations.append(f"Maiores contribuições ao déficit de Macro-F1 em {split}: {top_deficit}.")
        if confusions:
            top_confusions = ", ".join(
                f"{item['true_label']}→{item['predicted_label']} ({item['count']} janelas, "
                f"{_fmt(item['share_of_true_support'])} do suporte de {item['true_label']})"
                for item in confusions[:REPORTED_ITEMS]
            )
            pooled = " (soma entre seeds)" if entry["source_kind"] == MULTISEED_SUMMARY else ""
            observations.append(f"Confusões dominantes em {split}{pooled}: {top_confusions}.")
    if entry["source_kind"] == MULTISEED_SUMMARY:
        test_variation = entry["seed_variation"]["test"]
        macro_stats = test_variation["macro_f1_restricted"]
        observations.append(
            f"Variação entre seeds da Macro-F1 restrita em test: desvio-padrão {_fmt(macro_stats['std'])}, "
            f"faixa {_fmt(macro_stats['min'])}–{_fmt(macro_stats['max'])} (n={macro_stats['n']})."
        )
        restricted_labels = [LE2I_LABEL_NAMES[class_id] for class_id in RESTRICTED_CLASSES]
        stds = {
            label: test_variation["per_class"][label]["f1"]["std"]
            for label in restricted_labels
            if test_variation["per_class"][label]["f1"]["std"] is not None
        }
        if stds:
            most_variable = max(stds, key=lambda label: stds[label])
            observations.append(
                f"Classe restrita com maior desvio-padrão de F1 em test entre seeds: "
                f"{most_variable} ({_fmt(stds[most_variable])})."
            )
        persistent = [
            f"{item['true_label']}→{item['predicted_label']}"
            for item in test_variation["confusion_recurrence_in_top_k"]
            if item["n_seeds"] == entry["n_seeds"]
        ]
        observations.append(
            f"Confusões no top-{TOP_CONFUSIONS} de test em todas as {entry['n_seeds']} seeds: "
            + (", ".join(persistent) if persistent else "nenhuma")
            + "."
        )
    return observations


def _temporally_adjacent(first: str, second: str) -> bool:
    return second in TRANSITION_ADJACENT_STATES.get(first, frozenset()) or first in TRANSITION_ADJACENT_STATES.get(
        second, frozenset()
    )


def _entry_hypotheses(entry: dict, held_out_factor: str) -> list[str]:
    hypotheses: list[str] = []
    _, test_deficit, test_confusions, test_per_class = _split_view(entry, "test")
    _, _, val_confusions, _ = _split_view(entry, "val")
    val_pairs = {(item["true_label"], item["predicted_label"]) for item in val_confusions}
    temporal: list[str] = []
    residual: list[str] = []
    for item in test_confusions[:REPORTED_ITEMS]:
        true_label, predicted_label = item["true_label"], item["predicted_label"]
        in_val = "também no top de val" if (true_label, predicted_label) in val_pairs else "fora do top de val"
        described = f"{true_label}→{predicted_label} ({in_val})"
        if _temporally_adjacent(true_label, predicted_label):
            temporal.append(described)
        elif RESIDUAL_LABEL in (true_label, predicted_label):
            residual.append(described)
    if temporal:
        hypotheses.append(
            f"{', '.join(temporal)} envolve(m) fase de transição e estado contíguo; pode refletir "
            "ambiguidade de fronteira temporal dentro da janela de 2,4 s."
        )
    if residual:
        hypotheses.append(
            f"{', '.join(residual)} envolve(m) a classe residual {RESIDUAL_LABEL}, que agrupa ações "
            "heterogêneas; parte das janelas pode ser indistinguível da outra classe em pose e aparência."
        )
    test_windows = sum(float(row["support"]) for row in test_per_class.values())
    for item in test_deficit[:REPORTED_ITEMS]:
        support_value = float(test_per_class[item["label"]]["support"])
        fraction = support_value / test_windows if test_windows > 0 else 0.0
        if fraction < LOW_SUPPORT_FRACTION:
            hypotheses.append(
                f"{item['label']} tem {support_value:.0f} janelas em test ({fraction:.2%}); o F1 baixo "
                "pode ser dominado por suporte escasso e concentrado em poucos vídeos, não só por erro do modelo."
            )
    gap = _train_minus_test(entry)
    if gap is not None and gap > LARGE_GAP:
        hypotheses.append(
            f"Lacuna train−test de {_fmt(gap)} sugere sobreajuste e/ou dificuldade de generalizar a "
            f"{held_out_factor} não vistos; separar as duas causas exige análise por sujeito/ambiente, fora deste escopo."
        )
    if entry["source_kind"] == MULTISEED_SUMMARY:
        per_class = entry["seed_variation"]["test"]["per_class"]
        unstable = [
            LE2I_LABEL_NAMES[class_id]
            for class_id in RESTRICTED_CLASSES
            if (std := per_class[LE2I_LABEL_NAMES[class_id]]["f1"]["std"]) is not None
            and std > UNSTABLE_SEED_STD
        ]
        if unstable:
            hypotheses.append(
                f"F1 em test instável entre seeds (desvio-padrão > {UNSTABLE_SEED_STD}) em {', '.join(unstable)}; "
                "pode indicar fronteira de decisão sensível à inicialização para essas classes."
            )
    return hypotheses


def _limitations(protocols: dict) -> list[str]:
    limitations = [
        "Métricas em nível de janela com eval_stride 1: janelas vizinhas se sobrepõem e não são "
        "observações independentes; contagens não equivalem a eventos, vídeos ou sujeitos.",
        "A variação entre seeds descreve inicialização/otimização sob a mesma receita; não é incerteza "
        "amostral sobre sujeitos (ver bootstrap agrupado por sujeito).",
        "Le2i-CS e Le2i-CV têm splits e fatores retidos diferentes (sujeitos vs. ambientes); seus números "
        "não são agregados nem diretamente comparáveis.",
        "O split test é usado apenas descritivamente; nenhum run, seed, checkpoint, limiar ou hiperparâmetro "
        "é selecionado ou promovido por este diagnóstico.",
        "Não há quebra por ambiente/sujeito, redução de dimensionalidade, aumento de dados nem nova "
        "inferência; os confusion_matrix persistidos são a única fonte.",
        f"lie_down e lying ficam fora da Macro-F1 restrita ({RESTRICTED_CLASSES}); seus F1 por classe "
        "são reportados, mas não entram no déficit.",
        "Hipóteses são geradas por regras simples sobre os números confirmados e não foram testadas.",
    ]
    for dataset, block in protocols.items():
        for arm, sources in block["arms"].items():
            for summary in sources["multiseed_summaries"]:
                if summary["protocol_unconfirmed_run_dirs"]:
                    limitations.append(
                        f"{PROTOCOL_NAMES[dataset]}/{arm}: protocolo do sumário {summary['source']} declarado "
                        "pelo usuário; run_dirs fora das raízes locais conhecidas não confirmam o protocolo."
                    )
            for run in sources["runs"]:
                policy = run.get("macro_f1_policy")
                if policy is not None and not policy.get("matches_configured_restriction", True):
                    limitations.append(
                        f"{PROTOCOL_NAMES[dataset]}/{arm}: classes com suporte de treino positivo divergem "
                        f"de RESTRICTED_CLASSES em {run['source']}; a definição restrita foi mantida."
                    )
    return limitations


def build_findings(protocols: dict) -> tuple[list[dict], list[dict]]:
    observations: list[dict] = []
    hypotheses: list[dict] = []
    for dataset, block in protocols.items():
        for arm, sources in block["arms"].items():
            for entry in [*sources["runs"], *sources["multiseed_summaries"]]:
                base = {"dataset": dataset, "protocol": block["name"], "arm": arm, "source": entry["source"]}
                observations.extend({**base, "text": text} for text in _entry_observations(entry))
                hypotheses.extend(
                    {**base, "text": text} for text in _entry_hypotheses(entry, block["held_out_factor"])
                )
    return observations, hypotheses


def build_diagnosis(
    runs: list[DiagnosisInput],
    multiseed_summaries: list[DiagnosisInput],
    reference_root: Path | None = DEFAULT_REFERENCE_ROOT,
) -> tuple[dict, list[Path]]:
    label_names = LE2I_LABEL_NAMES
    protocols = {dataset: _empty_protocol_block(dataset) for dataset in PROTOCOL_NAMES}
    files_read: list[Path] = []

    if reference_root is not None:
        for arm in ARMS:
            reference_dir = reference_root / ARM_RUN_DIR_NAMES[arm]
            if not (reference_dir / "metrics.json").is_file():
                continue
            entry, read = diagnose_run(
                reference_dir, CANONICAL_REFERENCE_DATASET, arm, CANONICAL_REFERENCE, label_names
            )
            protocols[CANONICAL_REFERENCE_DATASET]["arms"][arm]["runs"].append(entry)
            files_read.extend(read)

    for item in runs:
        _validate_dataset_and_arm(item.dataset, item.arm)
        validate_local_run_dir(item.path, item.dataset)
        if reference_root is not None and _overlaps(item.path, reference_root):
            raise ValueError(f"--run {item.path} aponta para referência canônica; ela é lida automaticamente")
        entry, read = diagnose_run(item.path, item.dataset, item.arm, LOCAL_RUN, label_names)
        protocols[item.dataset]["arms"][item.arm]["runs"].append(entry)
        files_read.extend(read)

    for item in multiseed_summaries:
        _validate_dataset_and_arm(item.dataset, item.arm)
        entry, read = diagnose_multiseed_summary(item.path, item.dataset, item.arm, label_names)
        protocols[item.dataset]["arms"][item.arm]["multiseed_summaries"].append(entry)
        files_read.extend(read)

    resolved_reads = [path.resolve() for path in files_read]
    if len(set(resolved_reads)) != len(resolved_reads):
        raise ValueError("a mesma fonte foi fornecida mais de uma vez")

    observations, hypotheses = build_findings(protocols)
    report = {
        "analysis": "classification_diagnosis",
        "read_only": True,
        "macro_f1_definition": {
            "restricted_classes": list(RESTRICTED_CLASSES),
            "restricted_labels": [label_names[class_id] for class_id in RESTRICTED_CLASSES],
            "excluded_classes": [c for c in range(NUM_CLASSES) if c not in RESTRICTED_CLASSES],
            "excluded_labels": [label_names[c] for c in range(NUM_CLASSES) if c not in RESTRICTED_CLASSES],
            "source": "gatefall.train.shared.metrics.restricted_macro_f1",
        },
        "label_names": list(label_names),
        "expected_seeds": list(EXPECTED_SEEDS),
        "reference_root": None if reference_root is None else display_path(reference_root),
        "protocols": protocols,
        "canonical_reference_matches": _reference_matches(protocols),
        "missing_inputs": _missing_inputs(protocols, reference_root),
        "observations": observations,
        "hypotheses": hypotheses,
        "limitations": _limitations(protocols),
    }
    return report, files_read


def _per_class_csv_rows(base: dict, split: str, per_class: dict, record: str) -> list[dict]:
    return [
        {**base, "split": split, "record": record, "entity": label, "metric": metric, "value": value}
        for label, row in per_class.items()
        for metric, value in row.items()
        if metric in ("precision", "recall", "f1", "support")
    ]


def csv_rows(report: dict) -> list[dict]:
    rows: list[dict] = []
    for dataset, block in report["protocols"].items():
        for arm, sources in block["arms"].items():
            for entry in sources["runs"]:
                base = {"dataset": dataset, "protocol": block["name"], "arm": arm,
                        "source_kind": entry["source_kind"], "source": entry["source"], "seed": entry["seed"]}
                rows.extend(_diagnosis_csv_rows(base, entry))
            for summary in sources["multiseed_summaries"]:
                summary_base = {"dataset": dataset, "protocol": block["name"], "arm": arm,
                                "source_kind": MULTISEED_SUMMARY, "source": summary["source"]}
                for seed_entry in summary["per_seed"]:
                    rows.extend(_diagnosis_csv_rows({**summary_base, "seed": seed_entry["seed"]}, seed_entry))
                rows.extend(_seed_variation_csv_rows({**summary_base, "seed": "all"}, summary["seed_variation"]))
    return rows


def _diagnosis_csv_rows(base: dict, entry: dict) -> list[dict]:
    rows: list[dict] = []
    for split, diagnosis in entry["splits"].items():
        rows.append({**base, "split": split, "record": "macro_f1_restricted", "entity": "",
                     "metric": "value", "value": diagnosis["macro_f1_restricted"]})
        rows.extend(_per_class_csv_rows(base, split, diagnosis["per_class"], "per_class"))
        for item in diagnosis["macro_f1_deficit"]:
            rows.append({**base, "split": split, "record": "macro_f1_deficit", "entity": item["label"],
                         "metric": "deficit", "value": item["deficit"]})
        for item in diagnosis["dominant_confusions"]:
            rows.append({**base, "split": split, "record": "confusion",
                         "entity": f"{item['true_label']}->{item['predicted_label']}",
                         "metric": "count", "value": item["count"]})
    for pair, value in entry["generalization_gaps"]["macro_f1_restricted"].items():
        rows.append({**base, "split": "", "record": "generalization_gap", "entity": "",
                     "metric": pair, "value": value})
    return rows


def _seed_variation_csv_rows(base: dict, variation: dict) -> list[dict]:
    rows: list[dict] = []
    for split in CLASSIFICATION_SPLITS:
        for stat, value in variation[split]["macro_f1_restricted"].items():
            rows.append({**base, "split": split, "record": "macro_f1_restricted", "entity": "",
                         "metric": stat, "value": value})
        for label, metrics in variation[split]["per_class"].items():
            for metric, stats in metrics.items():
                for stat, value in stats.items():
                    rows.append({**base, "split": split, "record": "per_class", "entity": label,
                                 "metric": f"{metric}.{stat}", "value": value})
    for pair, stats in variation["generalization_gaps"].items():
        for stat, value in stats.items():
            rows.append({**base, "split": "", "record": "generalization_gap", "entity": "",
                         "metric": f"{pair}.{stat}", "value": value})
    return rows


def _primary_entries(sources: dict) -> list[dict]:
    return sources["multiseed_summaries"] or sources["runs"]


def _findings_for(findings: list[dict], entry: dict) -> list[str]:
    return [
        item["text"]
        for item in findings
        if (item["dataset"], item["arm"], item["source"]) == (entry["dataset"], entry["arm"], entry["source"])
    ]


def _deficit_table(block: dict) -> list[str]:
    primaries = {arm: _primary_entries(sources)[0] for arm, sources in block["arms"].items() if _primary_entries(sources)}
    if not primaries:
        return []
    deficits = {arm: {item["label"]: item for item in _split_view(entry, "test")[1]} for arm, entry in primaries.items()}
    labels = [LE2I_LABEL_NAMES[class_id] for class_id in RESTRICTED_CLASSES]
    labels.sort(key=lambda label: -sum(deficits[arm][label]["deficit"] for arm in primaries))
    lines = [
        "",
        "Contribuição de cada classe restrita ao déficit da Macro-F1 em test — F1 (parcela do déficit):",
        "",
        "| Classe | " + " | ".join(primaries) + " |",
        "| --- |" + " --- |" * len(primaries),
    ]
    for label in labels:
        cells = [
            f"{_fmt(deficits[arm][label]['f1'])} ({_fmt(deficits[arm][label]['share_of_total_deficit'])})"
            for arm in primaries
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def render_summary(report: dict) -> str:
    definition = report["macro_f1_definition"]
    lines = [
        "# Diagnóstico de erros de classificação",
        "",
        "Análise somente leitura de artefatos existentes; nenhuma inferência, treino ou seleção. "
        f"Macro-F1 restrita = média simples do F1 sobre {definition['restricted_labels']} "
        f"(excluídas: {definition['excluded_labels']}). Quando há sumário multi-seed, os detalhes por "
        "arma usam a média entre seeds; os demais runs e os valores por seed estão no JSON e no CSV.",
    ]
    observations = report["observations"]
    hypotheses = report["hypotheses"]
    for dataset, block in report["protocols"].items():
        lines += ["", f"## {block['name']} (`{dataset}`, {block['held_out_factor']} retidos)", ""]
        lines += ["| Arma | Fonte | Macro-F1 val | Macro-F1 test | Lacuna train−test |", "| --- | --- | --- | --- | --- |"]
        for arm, sources in block["arms"].items():
            for entry in [*sources["runs"], *sources["multiseed_summaries"]]:
                lines.append(
                    f"| {arm} | {_source_label(entry)} | {_fmt(_split_view(entry, 'val')[0])} | "
                    f"{_fmt(_split_view(entry, 'test')[0])} | {_fmt(_train_minus_test(entry))} |"
                )
        lines += _deficit_table(block)
        for arm, sources in block["arms"].items():
            for entry in _primary_entries(sources):
                lines += ["", f"### {arm} — {_source_label(entry)}", "", f"Fonte: `{entry['source']}`", ""]
                lines += ["Observações confirmadas:", ""]
                lines += [f"- {text}" for text in _findings_for(observations, entry)]
                lines += ["", "Hipóteses (não verificadas):", ""]
                lines += [f"- {text}" for text in _findings_for(hypotheses, entry)] or [
                    "- Nenhuma regra de hipótese disparada."
                ]
    lines += ["", "## Correspondência com referências canônicas", ""]
    matches = report["canonical_reference_matches"]
    lines += [
        f"- {PROTOCOL_NAMES[item['dataset']]}/{item['arm']}: referência `{item['reference']}` = seed "
        f"{item['seed']} de `{item['summary']}` (checkpoint igual: {item['same_checkpoint_sha256']}, "
        f"matrizes iguais: {item['same_confusion_matrices']})"
        for item in matches
    ] or ["- Nenhuma correspondência encontrada entre referências e sumários fornecidos."]
    lines += ["", "## Entradas ausentes", ""]
    lines += [
        f"- {PROTOCOL_NAMES[item['dataset']]}/{item['arm']} — {item['input']}: {item['reason']}"
        for item in report["missing_inputs"]
    ] or ["- Nenhuma."]
    lines += ["", "## Limitações", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def _overlaps(first: Path, second: Path) -> bool:
    a = first.resolve()
    b = second.resolve()
    return a == b or b in a.parents or a in b.parents


def check_output_dir(output_dir: Path, protected: list[Path]) -> None:
    for path in protected:
        if _overlaps(output_dir, path):
            raise ValueError(f"--output-dir {output_dir} se sobrepõe a artefato protegido: {path}")


def protected_paths(
    runs: list[DiagnosisInput], multiseed_summaries: list[DiagnosisInput], reference_root: Path | None
) -> list[Path]:
    paths = [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        *(item.path for item in runs),
        *(item.path.parent for item in multiseed_summaries),
    ]
    if reference_root is not None:
        paths.append(reference_root)
    return paths


def snapshot_inputs(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def verify_inputs_unchanged(snapshot: dict[str, str]) -> None:
    changed = [path for path, digest in snapshot.items() if sha256_file(Path(path)) != digest]
    if changed:
        raise RuntimeError(f"artefatos de entrada mudaram durante o diagnóstico: {changed}")


def write_outputs(output_dir: Path, report: dict, rows: list[dict], summary: str, force: bool) -> bool:
    targets = {
        output_dir / DIAGNOSIS_JSON_FILE: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        output_dir / DIAGNOSIS_CSV_FILE: pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(index=False),
        output_dir / DIAGNOSIS_SUMMARY_FILE: summary,
    }
    existing = [str(path) for path in targets if path.exists()]
    if existing and not force:
        print(f"skip {', '.join(existing)} (já existe, use --force para sobrescrever)")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged = {path: path.with_name(f".{path.name}.tmp-{token}") for path in targets}
    try:
        for path, text in targets.items():
            staged[path].write_text(text, encoding="utf-8")
        for path, temporary in staged.items():
            os.replace(temporary, path)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    for path in targets:
        print(f"{path}: gravado")
    return True


def run_diagnosis(
    runs: list[DiagnosisInput],
    multiseed_summaries: list[DiagnosisInput],
    output_dir: Path,
    reference_root: Path | None = DEFAULT_REFERENCE_ROOT,
    force: bool = False,
) -> bool:
    check_output_dir(output_dir, protected_paths(runs, multiseed_summaries, reference_root))
    report, files_read = build_diagnosis(runs, multiseed_summaries, reference_root)
    snapshot = snapshot_inputs(files_read)
    report["input_sha256"] = snapshot
    written = write_outputs(output_dir, report, csv_rows(report), render_summary(report), force)
    verify_inputs_unchanged(snapshot)
    return written


def _parse_inputs(values: list[list[str]] | None) -> list[DiagnosisInput]:
    inputs: list[DiagnosisInput] = []
    for dataset, arm, path in values or []:
        _validate_dataset_and_arm(dataset, arm)
        inputs.append(DiagnosisInput(dataset=dataset, arm=arm, path=Path(path)))
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Diagnostica erros de classificação a partir de artefatos existentes"
    )
    diagnose_parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("DATASET", "ARM", "RUN_DIR"),
        help="Repetível: run local com config.yaml e metrics.json",
    )
    diagnose_parser.add_argument(
        "--multiseed-summary",
        nargs=3,
        action="append",
        metavar=("DATASET", "ARM", "JSON"),
        help="Repetível: multiseed_summary.json do protocolo e arma declarados",
    )
    diagnose_parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    diagnose_parser.add_argument(
        "--no-canonical-references",
        action="store_true",
        help="Não lê as referências canônicas do Le2i-CS",
    )
    diagnose_parser.add_argument("--output-dir", type=Path, required=True)
    diagnose_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do diagnóstico")

    args = parser.parse_args()
    if args.command == "diagnose":
        run_diagnosis(
            runs=_parse_inputs(args.run),
            multiseed_summaries=_parse_inputs(args.multiseed_summary),
            output_dir=args.output_dir,
            reference_root=None if args.no_canonical_references else args.reference_root,
            force=args.force,
        )
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.classification_diagnosis import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
