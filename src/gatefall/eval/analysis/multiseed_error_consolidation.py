"""Consolidação multi-seed, somente leitura, do diagnóstico estratificado de erros de classificação.

Consome relatórios `stratified_error_diagnosis.json` já gravados (Etapa 1B)
com runs das armas A, B0, B1, C0 e C1 em Le2i-CS e Le2i-CV e resume, por
ambiente e por (ambiente, sujeito), a variação entre seeds de treino da
Macro-F1 restrita, os padrões de erro por classe e a consistência das
confusões dominantes. Não roda inferência, treino, ajuste, seleção de
checkpoint nem promoção; não lê checkpoints nem features.

A unidade agregada é a seed de treino independente: cada estatística
(n/mean/std/min/max) é calculada sobre os valores por seed, e contagens de
janelas nunca são somadas entre seeds. Le2i-CS e Le2i-CV ficam em blocos
separados e nunca são agregados entre si. O split `test` é usado só de forma
descritiva.

As predições de cada seed são as da reexecução validada da Etapa 1B. A
reconciliação dela com `metrics.json` não prova a identidade das predições
históricas por janela, e matrizes globais iguais entre relatórios também não
provam que as mesmas janelas receberam os mesmos rótulos.
"""

import argparse
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeGuard

import numpy as np
import pandas as pd

from gatefall.config import NUM_CLASSES
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import (
    EXPECTED_SEEDS,
    METRIC_TOLERANCE,
    PROTOCOL_HELD_OUT_FACTOR,
    PROTOCOL_NAMES,
    RESIDUAL_LABEL,
    TOP_CONFUSIONS,
    UNSTABLE_SEED_STD,
    _finite_number,
    _read_json,
    check_output_dir,
    display_path,
    dominant_confusions,
    labels_from_confusion_matrix,
    run_dir_protocol,
    snapshot_inputs,
    validate_confusion_matrix,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.multiseed_summary import ARMS, MIN_SEEDS, _aggregate_stats
from gatefall.eval.analysis.stratified_error_diagnosis import LEVELS, SPLITS
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.shared.metrics import RESTRICTED_CLASSES, classification_summary, restricted_macro_f1

CONSOLIDATION_JSON_FILE = "multiseed_error_consolidation.json"
CONSOLIDATION_CSV_FILE = "multiseed_error_consolidation.csv"
CONSOLIDATION_SUMMARY_FILE = "multiseed_error_consolidation.md"

SOURCE_ANALYSIS = "stratified_error_diagnosis"
NOT_PROVIDED_REASON = "nenhum run fornecido"
SPLIT_GROUP = "all"
AGGREGATED_LEVELS: tuple[str, ...] = ("split", "environment", "environment_subject")
COMPATIBILITY_FIELDS: tuple[str, ...] = (
    "macro_f1_definition",
    "label_names",
    "ignore_label",
    "eval_stride",
    "splits",
    "levels",
)
VALIDATED_PER_CLASS_FIELDS: tuple[str, ...] = ("support", "precision", "recall", "f1", "tp", "fp", "fn")
AGGREGATED_PER_CLASS_FIELDS: tuple[str, ...] = ("recall", "precision", "f1")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# Seed do smoke descritivo da Etapa 1B, cujos padrões de erro são conferidos
# nas demais seeds; não é escolhida por desempenho.
REFERENCE_SEED = 42
FOCUS_ARMS: tuple[str, ...] = ("B0", "B1")
COMPARISON_ARM = "A"

STABLE = "stable"
STABLE_IN_AVAILABLE_SEEDS = "stable_in_available_seeds"
MAJORITY = "majority"
UNSTABLE = "unstable"
INCONCLUSIVE = "inconclusive"
SUPPORTED_STATUSES = frozenset({STABLE, STABLE_IN_AVAILABLE_SEEDS})
STATUS_TEXT = {
    STABLE: "estável em todas as seeds esperadas",
    STABLE_IN_AVAILABLE_SEEDS: "estável nas seeds disponíveis (cobertura incompleta)",
    MAJORITY: "dominante na maioria das seeds",
    UNSTABLE: "instável",
    INCONCLUSIVE: "inconclusiva",
}

CSV_COLUMNS = [
    "dataset",
    "protocol",
    "arm",
    "split",
    "level",
    "group",
    "env",
    "subject",
    "entity",
    "metric",
    "n",
    "mean",
    "std",
    "min",
    "max",
    "value",
]


def _pair(true_label: str, predicted_label: str) -> str:
    return f"{true_label}->{predicted_label}"


def _overall_group(analysis: dict) -> dict:
    return {"group": SPLIT_GROUP, "env": None, "subject": None, "video_id": None, **analysis["overall"]}


def _level_groups(analysis: dict, level: str) -> list[dict]:
    return [_overall_group(analysis)] if level == "split" else analysis["levels"][level]


def _is_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _metric_mismatch(field: str, stored: object, recomputed: float) -> str | None:
    number = _finite_number(stored)
    if number is None:
        return f"{field}: valor gravado ausente, não numérico ou não finito ({stored!r})"
    if abs(number - recomputed) > METRIC_TOLERANCE:
        return f"{field}: gravado {stored}, recalculado {recomputed}"
    return None


def _count_mismatch(field: str, stored: object, expected: int) -> str | None:
    if not _is_count(stored):
        return f"{field}: contagem gravada ausente ou não inteira ({stored!r})"
    if stored != expected:
        return f"{field}: gravado {stored}, recalculado {expected}"
    return None


def _window_count_mismatches(group: dict, n_labeled: int) -> list[str]:
    total = group.get("n_windows")
    ignored = group.get("n_ignored_windows")
    if not _is_count(total) or not _is_count(ignored):
        return [f"n_windows/n_ignored_windows: contagens ausentes ou não inteiras ({total!r}, {ignored!r})"]
    if total != n_labeled + ignored:
        return [f"n_windows: gravado {total}, esperado n_labeled_windows + n_ignored_windows = {n_labeled + ignored}"]
    return []


def _dominant_confusion_mismatches(stored: object, expected: list[dict]) -> list[str]:
    if not isinstance(stored, list):
        return [f"dominant_confusions: esperado lista, gravado {type(stored).__name__}"]
    if len(stored) != len(expected):
        return [f"dominant_confusions: esperado lista com {len(expected)} item(ns), gravado {len(stored)}"]
    mismatches: list[str] = []
    for index, (item, recomputed) in enumerate(zip(stored, expected)):
        field = f"dominant_confusions[{index}]"
        if not isinstance(item, dict):
            mismatches.append(f"{field}: item inválido ({item!r})")
            continue
        for key in ("true_id", "true_label", "predicted_id", "predicted_label", "count"):
            if item.get(key) != recomputed[key] or isinstance(item.get(key), bool):
                mismatches.append(f"{field}.{key}: gravado {item.get(key)!r}, recalculado {recomputed[key]!r}")
        for key in ("share_of_true_support", "share_of_predicted", "share_of_all_errors"):
            mismatch = _metric_mismatch(f"{field}.{key}", item.get(key), recomputed[key])
            if mismatch:
                mismatches.append(mismatch)
    return mismatches


def _validate_group(group: dict, context: str) -> np.ndarray:
    matrix = np.asarray(validate_confusion_matrix(group.get("confusion_matrix"), context), dtype=np.int64)
    y_true, y_pred = labels_from_confusion_matrix(matrix.tolist())
    summary = classification_summary(y_true, y_pred, LE2I_LABEL_NAMES, NUM_CLASSES)
    n_labeled = int(matrix.sum())
    mismatches = [
        message
        for message in (
            _count_mismatch("n_labeled_windows", group.get("n_labeled_windows"), n_labeled),
            _count_mismatch("n_errors", group.get("n_errors"), n_labeled - int(np.trace(matrix))),
        )
        if message
    ]
    mismatches += _window_count_mismatches(group, n_labeled)
    stored_macro = group.get("macro_f1_restricted")
    if n_labeled == 0:
        if stored_macro is not None:
            mismatches.append(f"macro_f1_restricted: esperado null sem janelas rotuladas, gravado {stored_macro!r}")
    else:
        mismatch = _metric_mismatch(
            "macro_f1_restricted", stored_macro, restricted_macro_f1(y_true, y_pred, NUM_CLASSES)[0]
        )
        if mismatch:
            mismatches.append(mismatch)
    stored_per_class = group.get("per_class") or {}
    for label in LE2I_LABEL_NAMES:
        stored = stored_per_class.get(label) or {}
        recomputed = summary["per_class"][label]
        for field in VALIDATED_PER_CLASS_FIELDS:
            name = f"per_class[{label}].{field}"
            mismatch = (
                _metric_mismatch(name, stored.get(field), recomputed[field])
                if field in AGGREGATED_PER_CLASS_FIELDS
                else _count_mismatch(name, stored.get(field), recomputed[field])
            )
            if mismatch:
                mismatches.append(mismatch)
    restricted_labels = [LE2I_LABEL_NAMES[class_id] for class_id in RESTRICTED_CLASSES]
    absent = [label for label in LE2I_LABEL_NAMES if summary["per_class"][label]["support"] == 0]
    expected_labels = {
        "absent_labels": absent,
        "absent_restricted_labels": [label for label in absent if label in restricted_labels],
        "predicted_absent_labels": [label for label in absent if summary["per_class"][label]["fp"] > 0],
    }
    mismatches += [
        f"{field}: gravado {group.get(field)!r}, recalculado {expected!r}"
        for field, expected in expected_labels.items()
        if group.get(field) != expected
    ]
    mismatches += _dominant_confusion_mismatches(
        group.get("dominant_confusions"), dominant_confusions(matrix.tolist(), LE2I_LABEL_NAMES)
    )
    if mismatches:
        raise ValueError(f"{context}: relatório inconsistente com a própria confusion_matrix: {mismatches[:5]}")
    return matrix


def _validate_split(analysis: object, context: str) -> None:
    if not isinstance(analysis, dict) or not isinstance(analysis.get("levels"), dict):
        raise ValueError(f"{context}: split sem overall/levels")
    reconciliation = analysis.get("reconciliation") or {}
    if (
        reconciliation.get("ok") is not True
        or reconciliation.get("prediction_source") != "validated_rerun"
        or reconciliation.get("proves_historical_window_predictions") is not False
    ):
        raise ValueError(
            f"{context}: reconciliação ausente, falha ou sem procedência validated_rerun "
            "(relatório de versão incompatível da Etapa 1B?)"
        )
    overall = _validate_group(_overall_group(analysis), f"{context} [split]")
    for level in LEVELS:
        groups = analysis["levels"].get(level)
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"{context}: nível {level} ausente ou vazio")
        names = [group.get("group") for group in groups]
        if len(set(names)) != len(names):
            raise ValueError(f"{context}: grupos duplicados no nível {level}")
        matrices = [_validate_group(group, f"{context} [{level}/{group.get('group')}]") for group in groups]
        if not np.array_equal(np.sum(matrices, axis=0), overall):
            raise ValueError(f"{context}: soma das matrizes por {level} diverge da matriz do split")


def _frames_digest(entry: dict, context: str) -> str:
    digests = [digest for path, digest in entry["input_sha256"].items() if path.endswith("frames.parquet")]
    if len(digests) != 1:
        raise ValueError(f"{context}: input_sha256 deve registrar exatamente um frames.parquet")
    return digests[0]


def validate_entry(entry: object, dataset: str, arm: str, expected_seeds: tuple[int, ...], context: str) -> dict:
    if not isinstance(entry, dict):
        raise ValueError(f"{context}: entrada de run inválida")
    seed = entry.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{context}: seed ausente ou não inteira ({seed!r})")
    context = f"{context} seed {seed}"
    if seed not in expected_seeds:
        raise ValueError(f"{context}: seed fora do conjunto esperado {list(expected_seeds)}")
    if (entry.get("dataset"), entry.get("protocol"), entry.get("arm")) != (dataset, PROTOCOL_NAMES[dataset], arm):
        raise ValueError(f"{context}: dataset/protocolo/arma da entrada divergem do bloco do relatório")
    source = entry.get("source")
    checkpoint = entry.get("checkpoint_sha256")
    inputs = entry.get("input_sha256")
    if not isinstance(source, str) or not isinstance(checkpoint, str) or not SHA256_PATTERN.fullmatch(checkpoint):
        raise ValueError(f"{context}: source ou checkpoint_sha256 ausente ou inválido")
    if run_dir_protocol(source) not in (None, dataset):
        raise ValueError(f"{context}: source {source} pertence a outro protocolo")
    if not isinstance(inputs, dict) or inputs.get(str(Path(source) / "checkpoint.pt")) != checkpoint:
        raise ValueError(f"{context}: input_sha256 não registra o checkpoint.pt do source com o mesmo SHA-256")
    _frames_digest(entry, context)
    splits = entry.get("splits")
    if not isinstance(splits, dict) or sorted(splits) != sorted(SPLITS):
        raise ValueError(f"{context}: splits devem ser exatamente {list(SPLITS)}")
    for split in SPLITS:
        _validate_split(splits[split], f"{context} [{split}]")
    return entry


def validate_report(path: Path, report: dict) -> None:
    context = display_path(path)
    if report.get("analysis") != SOURCE_ANALYSIS:
        raise ValueError(f"{context}: esperado relatório {SOURCE_ANALYSIS}")
    if report.get("canonical") is not False or report.get("read_only") is not True:
        raise ValueError(f"{context}: relatório deve ser não canônico e somente leitura")
    protocols = report.get("protocols")
    if not isinstance(protocols, dict) or set(protocols) - set(PROTOCOL_NAMES):
        raise ValueError(f"{context}: protocols ausente ou com protocolo desconhecido")
    for dataset, block in protocols.items():
        arms = block.get("arms") if isinstance(block, dict) else None
        if not isinstance(arms, dict) or set(arms) - set(ARMS):
            raise ValueError(f"{context}: {dataset} sem arms ou com arma desconhecida")


def _validate_compatibility(reports: list[tuple[Path, dict]]) -> dict:
    first_path, first = reports[0]
    signature = {field: first.get(field) for field in COMPATIBILITY_FIELDS}
    definition = signature["macro_f1_definition"] or {}
    if (
        signature["label_names"] != list(LE2I_LABEL_NAMES)
        or signature["splits"] != list(SPLITS)
        or definition.get("restricted_classes") != list(RESTRICTED_CLASSES)
    ):
        raise ValueError(f"{display_path(first_path)}: rótulos, splits ou Macro-F1 restrita incompatíveis")
    for path, report in reports[1:]:
        different = [field for field in COMPATIBILITY_FIELDS if report.get(field) != signature[field]]
        if different:
            raise ValueError(
                f"{display_path(path)}: incompatível com {display_path(first_path)} em {different}"
            )
    return signature


def _support_signature(entry: dict) -> dict:
    return {
        (split, level, group["group"]): (
            group.get("env"),
            group.get("subject"),
            group.get("n_videos"),
            group["n_windows"],
            group["n_labeled_windows"],
            group["n_ignored_windows"],
            tuple(group["per_class"][label]["support"] for label in LE2I_LABEL_NAMES),
        )
        for split in SPLITS
        for level in ("split", *LEVELS)
        for group in _level_groups(entry["splits"][split], level)
    }


def _validate_support(entries: list[dict], dataset: str) -> None:
    if not entries:
        return
    first = entries[0]
    first_frames = _frames_digest(first, dataset)
    first_signature = _support_signature(first)
    for entry in entries[1:]:
        context = f"{PROTOCOL_NAMES[dataset]}/{entry['arm']} seed {entry['seed']} ({entry['source']})"
        reference = f"{first['arm']} seed {first['seed']}"
        if _frames_digest(entry, context) != first_frames:
            raise ValueError(f"{context}: frames.parquet diverge do usado por {reference}")
        signature = _support_signature(entry)
        if signature.keys() != first_signature.keys():
            missing = sorted(set(first_signature) ^ set(signature), key=str)[:3]
            raise ValueError(f"{context}: cobertura de grupos incompatível com {reference}: {missing}")
        different = [key for key in first_signature if signature[key] != first_signature[key]]
        if different:
            raise ValueError(f"{context}: suporte incompatível com {reference} em {different[:3]}")


def collect_entries(
    reports: list[tuple[Path, dict]], expected_seeds: tuple[int, ...]
) -> tuple[dict[str, dict[str, list[dict]]], list[dict]]:
    entries: dict[str, dict[str, list[dict]]] = {dataset: {arm: [] for arm in ARMS} for dataset in PROTOCOL_NAMES}
    upstream_missing: list[dict] = []
    identities: dict[tuple[str, str, int], str] = {}
    checkpoints: dict[str, str] = {}
    for path, report in reports:
        report_path = display_path(path)
        for dataset, block in report["protocols"].items():
            for arm, arm_entries in block["arms"].items():
                for entry in arm_entries:
                    context = f"{report_path}: {PROTOCOL_NAMES[dataset]}/{arm}"
                    validate_entry(entry, dataset, arm, expected_seeds, context)
                    label = f"{PROTOCOL_NAMES[dataset]}/{arm} seed {entry['seed']} ({report_path})"
                    identity = (dataset, arm, entry["seed"])
                    if identity in identities:
                        raise ValueError(f"identidade de seed duplicada: {label} e {identities[identity]}")
                    if entry["checkpoint_sha256"] in checkpoints:
                        raise ValueError(
                            f"checkpoint_sha256 duplicado: {label} e {checkpoints[entry['checkpoint_sha256']]}"
                        )
                    identities[identity] = label
                    checkpoints[entry["checkpoint_sha256"]] = label
                    entries[dataset][arm].append({**entry, "report": report_path})
        upstream_missing.extend(
            {**item, "report": report_path}
            for item in report.get("missing_inputs", [])
            if item.get("reason") != NOT_PROVIDED_REASON
        )
    for dataset, arms in entries.items():
        for arm_entries in arms.values():
            arm_entries.sort(key=lambda entry: entry["seed"])
        _validate_support([entry for arm_entries in arms.values() for entry in arm_entries], dataset)
    return entries, upstream_missing


def _stats(values: list[float | int | None]) -> dict:
    return _aggregate_stats([value for value in values if value is not None])


def _by_seed(groups: list[tuple[int, dict]], value: Callable[[dict], object]) -> dict[str, object]:
    return {str(seed): value(group) for seed, group in groups}


def _main_error_target(group: dict, class_id: int) -> str | None:
    row = group["confusion_matrix"][class_id]
    errors = [(-count, predicted_id) for predicted_id, count in enumerate(row) if predicted_id != class_id and count > 0]
    return LE2I_LABEL_NAMES[min(errors)[1]] if errors else None


def _modal(values: Sequence[object]) -> tuple[str | None, int]:
    counts = Counter(value for value in values if isinstance(value, str))
    if not counts:
        return None, 0
    value, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return value, count


def per_class_pattern(groups: list[tuple[int, dict]], label: str) -> dict:
    class_id = LE2I_LABEL_NAMES.index(label)
    support = groups[0][1]["per_class"][label]["support"]
    if support == 0:
        false_positives = [group["per_class"][label]["fp"] for _seed, group in groups]
        return {
            "support": 0,
            "absent": True,
            "false_positives": {**_stats(false_positives), "by_seed": _by_seed(groups, lambda g: g["per_class"][label]["fp"])},
            "seeds_predicting_absent_class": [seed for seed, group in groups if group["per_class"][label]["fp"] > 0],
        }
    targets = _by_seed(groups, lambda group: _main_error_target(group, class_id))
    modal, modal_count = _modal(list(targets.values()))
    return {
        "support": support,
        "absent": False,
        **{
            field: {
                **_stats([group["per_class"][label][field] for _seed, group in groups]),
                "by_seed": _by_seed(groups, lambda group, field=field: group["per_class"][label][field]),
            }
            for field in AGGREGATED_PER_CLASS_FIELDS
        },
        "main_error_target": {"by_seed": targets, "modal": modal, "modal_seeds": modal_count},
    }


def _status(top1_seeds: int, n_seeds: int, complete: bool) -> str:
    if n_seeds < MIN_SEEDS:
        return INCONCLUSIVE
    if top1_seeds == n_seeds:
        return STABLE if complete else STABLE_IN_AVAILABLE_SEEDS
    return MAJORITY if 2 * top1_seeds > n_seeds else UNSTABLE


def confusion_consistency(groups: list[tuple[int, dict]], complete: bool) -> dict:
    top_pairs = {
        seed: [_pair(item["true_label"], item["predicted_label"]) for item in group["dominant_confusions"]]
        for seed, group in groups
    }
    top1 = {seed: pairs[0] if pairs else None for seed, pairs in top_pairs.items()}
    modal, modal_count = _modal(list(top1.values()))
    reference = top1.get(REFERENCE_SEED)
    consistency = {
        "n_seeds": len(groups),
        "top1_by_seed": {str(seed): pair for seed, pair in top1.items()},
        "modal_top1": modal,
        "modal_top1_seeds": modal_count,
        "distinct_top1": len({pair for pair in top1.values() if pair is not None}),
        "reference_seed": REFERENCE_SEED,
        "reference_top1": reference,
        "reference_top1_seeds": None,
        "reference_in_top_k_seeds": None,
        "reference_share_of_true_support": None,
        "status": INCONCLUSIVE,
    }
    if reference is None:
        return consistency
    true_label, predicted_label = reference.split("->")
    true_id = LE2I_LABEL_NAMES.index(true_label)
    predicted_id = LE2I_LABEL_NAMES.index(predicted_label)
    shares = {
        str(seed): group["confusion_matrix"][true_id][predicted_id] / sum(group["confusion_matrix"][true_id])
        for seed, group in groups
    }
    top1_seeds = sum(pair == reference for pair in top1.values())
    return {
        **consistency,
        "reference_top1_seeds": top1_seeds,
        "reference_in_top_k_seeds": sum(reference in pairs for pairs in top_pairs.values()),
        "reference_share_of_true_support": {**_stats(list(shares.values())), "by_seed": shares},
        "status": _status(top1_seeds, len(groups), complete),
    }


def _residual_predictions(group: dict) -> tuple[float | None, float | None]:
    residual_id = LE2I_LABEL_NAMES.index(RESIDUAL_LABEL)
    matrix = group["confusion_matrix"]
    predicted = sum(row[residual_id] for true_id, row in enumerate(matrix) if true_id != residual_id)
    non_residual = group["n_labeled_windows"] - group["per_class"][RESIDUAL_LABEL]["support"]
    return (
        predicted / non_residual if non_residual else None,
        predicted / group["n_errors"] if group["n_errors"] else None,
    )


def residual_prediction(groups: list[tuple[int, dict]]) -> dict:
    rates = {str(seed): _residual_predictions(group) for seed, group in groups}
    return {
        "label": RESIDUAL_LABEL,
        "rate_of_non_residual_windows": {
            **_stats([rate for rate, _share in rates.values()]),
            "by_seed": {seed: rate for seed, (rate, _share) in rates.items()},
        },
        "share_of_errors": {
            **_stats([share for _rate, share in rates.values()]),
            "by_seed": {seed: share for seed, (_rate, share) in rates.items()},
        },
    }


def _group_warnings(first: dict, predicted_absent: dict[str, list[int]], macro: dict) -> list[str]:
    warnings: list[str] = []
    if first["absent_restricted_labels"]:
        warnings.append(
            f"classes restritas ausentes ({', '.join(first['absent_restricted_labels'])}) entram com F1 0 "
            "na média fixa de oito classes; a Macro-F1 do grupo é mecanicamente menor"
        )
    warnings.extend(
        f"{label} ausente no grupo, mas predita nas seeds {seeds}" for label, seeds in predicted_absent.items()
    )
    if macro["n"] == 0:
        warnings.append("grupo sem janelas rotuladas; Macro-F1 indefinida")
    return warnings


def aggregate_group(groups: list[tuple[int, dict]], complete: bool) -> dict:
    first = groups[0][1]
    macro = {**_stats([group["macro_f1_restricted"] for _seed, group in groups]),
             "by_seed": _by_seed(groups, lambda group: group["macro_f1_restricted"])}
    per_class = {label: per_class_pattern(groups, label) for label in LE2I_LABEL_NAMES}
    predicted_absent = {
        label: pattern["seeds_predicting_absent_class"]
        for label, pattern in per_class.items()
        if pattern["absent"] and pattern["seeds_predicting_absent_class"]
    }
    return {
        "group": first["group"],
        "env": first["env"],
        "subject": first["subject"],
        "n_videos": first.get("n_videos"),
        "n_windows": first["n_windows"],
        "n_labeled_windows": first["n_labeled_windows"],
        "n_ignored_windows": first["n_ignored_windows"],
        "absent_labels": first["absent_labels"],
        "absent_restricted_labels": first["absent_restricted_labels"],
        "macro_f1_restricted": macro,
        "n_errors": {**_stats([group["n_errors"] for _seed, group in groups]),
                     "by_seed": _by_seed(groups, lambda group: group["n_errors"])},
        "per_class": per_class,
        "dominant_confusion_consistency": confusion_consistency(groups, complete),
        "residual_prediction": residual_prediction(groups),
        "warnings": _group_warnings(first, predicted_absent, macro),
    }


def aggregate_arm(entries: list[dict], expected_seeds: tuple[int, ...]) -> dict:
    seeds = [entry["seed"] for entry in entries]
    missing = [seed for seed in expected_seeds if seed not in seeds]
    coverage = {
        "expected_seeds": list(expected_seeds),
        "seeds": seeds,
        "missing_seeds": missing,
        "n_seeds": len(seeds),
        "complete": not missing,
    }
    runs = [
        {key: entry[key] for key in ("seed", "run_name", "source", "report", "checkpoint_sha256")}
        for entry in entries
    ]
    splits = {}
    if entries:
        for split in SPLITS:
            splits[split] = {}
            for level in AGGREGATED_LEVELS:
                names = [group["group"] for group in _level_groups(entries[0]["splits"][split], level)]
                lookup = [
                    (entry["seed"], {group["group"]: group for group in _level_groups(entry["splits"][split], level)})
                    for entry in entries
                ]
                splits[split][level] = [
                    aggregate_group([(seed, groups[name]) for seed, groups in lookup], coverage["complete"])
                    for name in names
                ]
    return {"coverage": coverage, "runs": runs, "splits": splits}


def _fmt(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.4f}"


def _mean_std(stats: dict) -> str:
    if stats["n"] == 0:
        return "n/d"
    std = "n/d" if stats["std"] is None else f"{stats['std']:.4f}"
    return f"{stats['mean']:.4f} ± {std} (n={stats['n']})"


def _arrow(pair: str) -> str:
    return pair.replace("->", "→")


def _group_label(level: str, group: dict) -> str:
    return "split inteiro" if level == "split" else group["group"]


def _find_group(arm_block: dict, split: str, level: str, name: str) -> dict | None:
    for group in arm_block["splits"].get(split, {}).get(level, []):
        if group["group"] == name:
            return group
    return None


def _require_group(arm_block: dict, split: str, level: str, name: str) -> dict:
    group = _find_group(arm_block, split, level, name)
    if group is None:
        raise KeyError(f"grupo {name} ausente em {split}/{level}")
    return group


def _residual_rate(group: dict) -> dict:
    return group["residual_prediction"]["rate_of_non_residual_windows"]


def _rate_range(group: dict) -> str:
    rate = _residual_rate(group)
    if rate["n"] == 0:
        return "n/d"
    return f"{_mean_std(rate)}, de {rate['min']:.4f} a {rate['max']:.4f}"


def _residual_above_comparison(group: dict, comparison: dict | None) -> bool | None:
    if comparison is None:
        return None
    rate = _residual_rate(group)
    comparison_rate = _residual_rate(comparison)
    if rate["n"] < MIN_SEEDS or comparison_rate["n"] < MIN_SEEDS:
        return None
    return rate["min"] > comparison_rate["max"]


def _focus_observation(protocol: str, arm: str, split: str, level: str, group: dict, comparison: dict | None) -> str:
    context = f"{protocol}/{arm} {split}, {_group_label(level, group)}"
    text = f"{context}: Macro-F1 restrita {_mean_std(group['macro_f1_restricted'])}"
    if comparison is not None:
        text += f" ({COMPARISON_ARM}: {_mean_std(comparison['macro_f1_restricted'])})"
    text += f"; janelas fora de {RESIDUAL_LABEL} preditas como {RESIDUAL_LABEL}: " + _rate_range(group)
    if comparison is not None:
        text += f" ({COMPARISON_ARM}: {_rate_range(comparison)})"
    consistency = group["dominant_confusion_consistency"]
    reference = consistency["reference_top1"]
    if reference is None:
        return text + f"; sem confusão dominante na seed {REFERENCE_SEED} para conferir."
    true_label = reference.split("->")[0]
    return (
        text
        + f"; a confusão dominante da seed {REFERENCE_SEED}, {_arrow(reference)}, é a dominante em "
        f"{consistency['reference_top1_seeds']} de {consistency['n_seeds']} seeds e está entre as "
        f"{TOP_CONFUSIONS} maiores em {consistency['reference_in_top_k_seeds']}; fração do suporte de "
        f"{true_label}: {_mean_std(consistency['reference_share_of_true_support'])}; "
        f"{STATUS_TEXT[consistency['status']]}."
    )


def _focus_hypotheses(
    protocol: str, arm: str, split: str, level: str, group: dict, comparison: dict | None
) -> list[str]:
    context = f"{protocol}/{arm} {split}, {_group_label(level, group)}"
    consistency = group["dominant_confusion_consistency"]
    reference = consistency["reference_top1"]
    hypotheses: list[str] = []
    if comparison is not None and _residual_above_comparison(group, comparison):
        rate = _residual_rate(group)
        comparison_rate = _residual_rate(comparison)
        hypotheses.append(
            f"{context}: em todas as seeds, a fração de janelas fora de {RESIDUAL_LABEL} preditas como "
            f"{RESIDUAL_LABEL} ({rate['min']:.4f} a {rate['max']:.4f}) excede a de qualquer seed de "
            f"{COMPARISON_ARM} ({comparison_rate['min']:.4f} a {comparison_rate['max']:.4f}); o deslocamento "
            "para a classe residual se repete entre seeds, mas a magnitude varia com a seed e a causa "
            "(features visuais, ambiente, sujeitos ou suporte por classe) não está isolada."
        )
    if reference is not None and consistency["status"] in SUPPORTED_STATUSES:
        if reference.endswith(f"->{RESIDUAL_LABEL}"):
            hypotheses.append(
                f"{context}: o deslocamento {_arrow(reference)} para a classe residual se repete em todas as "
                "seeds disponíveis; é compatível com um efeito sistemático da configuração da arma neste "
                "protocolo, e não com ruído de inicialização, mas não isola a causa (features visuais, "
                "ambiente, sujeitos ou suporte por classe)."
            )
        else:
            hypotheses.append(
                f"{context}: {_arrow(reference)} é um erro sistemático entre seeds; pode refletir ambiguidade "
                "entre estados vizinhos ou de rótulo, o que não foi testado."
            )
    elif reference is not None and consistency["status"] in (MAJORITY, UNSTABLE):
        hypotheses.append(
            f"{context}: a confusão dominante da seed {REFERENCE_SEED} ({_arrow(reference)}) não se repete em "
            f"todas as seeds ({consistency['reference_top1_seeds']}/{consistency['n_seeds']}); a leitura da "
            "Etapa 1B para este grupo depende da seed."
        )
    std = group["macro_f1_restricted"]["std"]
    if std is not None and std > UNSTABLE_SEED_STD:
        hypotheses.append(
            f"{context}: desvio-padrão da Macro-F1 entre seeds {std:.4f} > {UNSTABLE_SEED_STD}; o resultado "
            "do grupo é sensível à seed."
        )
    return hypotheses


def build_findings(protocols: dict) -> tuple[list[dict], list[dict], list[dict]]:
    observations: list[dict] = []
    hypotheses: list[dict] = []
    stability: list[dict] = []
    for dataset, block in protocols.items():
        comparison_block = block["arms"][COMPARISON_ARM]
        for arm in FOCUS_ARMS:
            arm_block = block["arms"][arm]
            for split, levels in arm_block["splits"].items():
                for level in ("split", "environment"):
                    for group in levels[level]:
                        base = {"dataset": dataset, "protocol": block["name"], "arm": arm, "split": split,
                                "level": level, "group": group["group"]}
                        consistency = group["dominant_confusion_consistency"]
                        comparison = _find_group(comparison_block, split, level, group["group"])
                        stability.append({
                            **base,
                            **{key: consistency[key] for key in (
                                "n_seeds", "reference_seed", "reference_top1", "reference_top1_seeds",
                                "reference_in_top_k_seeds", "modal_top1", "modal_top1_seeds", "status")},
                            "supported": consistency["status"] in SUPPORTED_STATUSES,
                            "residual_prediction_rate": {
                                key: _residual_rate(group)[key] for key in ("n", "mean", "std", "min", "max")
                            },
                            "residual_above_comparison_in_all_seeds": _residual_above_comparison(group, comparison),
                        })
                        observations.append({**base, "text": _focus_observation(
                            block["name"], arm, split, level, group, comparison)})
                        hypotheses.extend({**base, "text": text} for text in _focus_hypotheses(
                            block["name"], arm, split, level, group, comparison))
    return observations, hypotheses, stability


def _missing_inputs(protocols: dict, upstream_missing: list[dict]) -> list[dict]:
    missing = [
        {"dataset": dataset, "arm": arm, "missing_seeds": arm_block["coverage"]["missing_seeds"],
         "reason": "nenhuma seed fornecida" if not arm_block["coverage"]["seeds"] else "cobertura de seeds incompleta",
         "report": None}
        for dataset, block in protocols.items()
        for arm, arm_block in block["arms"].items()
        if arm_block["coverage"]["missing_seeds"]
    ]
    missing.extend(
        {"dataset": item.get("dataset"), "arm": item.get("arm"), "missing_seeds": [],
         "reason": f"ausente no relatório de origem: {item.get('reason')}", "report": item["report"],
         "source": item.get("source"), "missing_files": item.get("missing_files", [])}
        for item in upstream_missing
    )
    return missing


def _limitations(missing_inputs: list[dict]) -> list[str]:
    limitations = [
        "Análise exploratória e não canônica: nada aqui substitui runs/reference nem foi revisado; o split "
        "test é descritivo e nada aqui seleciona run, seed, checkpoint, limiar, modificação ou hiperparâmetro.",
        "A unidade agregada é a seed de treino independente: mean/std/min/max são calculados sobre o valor de "
        "cada seed (std amostral, nulo com menos de duas seeds). Contagens de janelas nunca são somadas entre "
        "seeds; o suporte é o mesmo em todas as seeds de um protocolo e é reportado uma vez. Com até cinco "
        "seeds não há intervalo de confiança.",
        "Cada seed traz as predições da reexecução validada da Etapa 1B. A reconciliação com metrics.json não "
        "prova que elas coincidem janela a janela com as predições históricas, e matrizes globais iguais "
        "entre relatórios também não provam identidade por janela.",
        "Métricas por janela com eval_stride 1: janelas vizinhas se sobrepõem e não são independentes.",
        f"A Macro-F1 restrita usa sempre as oito classes de RESTRICTED_CLASSES {RESTRICTED_CLASSES}; num grupo "
        "com classe restrita ausente, ela entra com F1 0 e a Macro-F1 do grupo não é comparável à do split.",
        "Le2i-CS e Le2i-CV nunca são agregados entre si. Grupos de sujeito usam o par (ambiente, sujeito).",
        "Seeds de armas diferentes são treinos independentes; a mesma seed em duas armas não forma par.",
        f"A seed {REFERENCE_SEED} serve de referência só porque o smoke da Etapa 1B foi feito nela; não foi "
        "escolhida por desempenho.",
        "Os relatórios de entrada são conferidos por SHA-256; checkpoints, métricas e features dos runs não "
        "são relidos, e a procedência de cada seed é a registrada pela Etapa 1B.",
        "Hipóteses vêm de regras fixas sobre os números confirmados e não foram testadas.",
    ]
    if missing_inputs:
        limitations.append(
            "Há seeds ou runs ausentes (ver missing_inputs); as estatísticas das armas afetadas usam só as "
            "seeds disponíveis e não equivalem às de cobertura completa."
        )
    return limitations


def build_consolidation(reports: list[tuple[Path, dict]], expected_seeds: tuple[int, ...]) -> dict:
    if not reports:
        raise ValueError("nenhum --report fornecido")
    reports = sorted(reports, key=lambda item: display_path(item[0]))
    for path, report in reports:
        validate_report(path, report)
    signature = _validate_compatibility(reports)
    entries, upstream_missing = collect_entries(reports, expected_seeds)
    protocols = {
        dataset: {
            "name": name,
            "held_out_factor": PROTOCOL_HELD_OUT_FACTOR[dataset],
            "arms": {arm: aggregate_arm(entries[dataset][arm], expected_seeds) for arm in ARMS},
        }
        for dataset, name in PROTOCOL_NAMES.items()
    }
    missing_inputs = _missing_inputs(protocols, upstream_missing)
    observations, hypotheses, stability = build_findings(protocols)
    return {
        "analysis": "multiseed_error_consolidation",
        "source_analysis": SOURCE_ANALYSIS,
        "canonical": False,
        "read_only": True,
        **signature,
        "aggregated_levels": list(AGGREGATED_LEVELS),
        "aggregation": {
            "unit": "seed de treino independente",
            "statistics": "n/mean/std/min/max sobre valores por seed; std amostral; contagens nunca somadas",
        },
        "expected_seeds": list(expected_seeds),
        "reference_seed": REFERENCE_SEED,
        "focus_arms": list(FOCUS_ARMS),
        "reports": [display_path(path) for path, _report in reports],
        "protocols": protocols,
        "missing_inputs": missing_inputs,
        "focus_stability": stability,
        "observations": observations,
        "hypotheses": hypotheses,
        "limitations": _limitations(missing_inputs),
    }


def _stat_row(base: dict, entity: str, metric: str, stats: dict) -> dict:
    return {**base, "entity": entity, "metric": metric,
            **{key: stats[key] for key in ("n", "mean", "std", "min", "max")}, "value": None}


def _value_row(base: dict, entity: str, metric: str, value: object) -> dict:
    return {**base, "entity": entity, "metric": metric, "value": value}


def _aggregated_group_rows(base: dict, group: dict) -> list[dict]:
    rows = [
        _stat_row(base, "", "macro_f1_restricted", group["macro_f1_restricted"]),
        _stat_row(base, "", "n_errors", group["n_errors"]),
        _value_row(base, "", "n_labeled_windows", group["n_labeled_windows"]),
        _value_row(base, "", "n_ignored_windows", group["n_ignored_windows"]),
    ]
    for label, pattern in group["per_class"].items():
        rows.append(_value_row(base, label, "support", pattern["support"]))
        if pattern["absent"]:
            rows.append(_value_row(base, label, "absent", 1))
            rows.append(_stat_row(base, label, "false_positives", pattern["false_positives"]))
            continue
        rows.extend(_stat_row(base, label, field, pattern[field]) for field in AGGREGATED_PER_CLASS_FIELDS)
        target = pattern["main_error_target"]
        if target["modal"] is not None:
            rows.append(_value_row(base, _pair(label, target["modal"]), "main_error_target_modal_seeds",
                                   target["modal_seeds"]))
    consistency = group["dominant_confusion_consistency"]
    if consistency["modal_top1"] is not None:
        rows.append(_value_row(base, consistency["modal_top1"], "modal_top1_seeds", consistency["modal_top1_seeds"]))
    reference = consistency["reference_top1"]
    if reference is not None:
        rows += [
            _value_row(base, reference, "reference_top1_seeds", consistency["reference_top1_seeds"]),
            _value_row(base, reference, "reference_in_top_k_seeds", consistency["reference_in_top_k_seeds"]),
            _stat_row(base, reference, "reference_share_of_true_support",
                      consistency["reference_share_of_true_support"]),
        ]
    rows.append(_value_row(base, reference or "", "dominance_status", consistency["status"]))
    residual = group["residual_prediction"]
    rows += [
        _stat_row(base, residual["label"], "residual_prediction_rate", residual["rate_of_non_residual_windows"]),
        _stat_row(base, residual["label"], "residual_share_of_errors", residual["share_of_errors"]),
    ]
    return rows


def csv_rows(report: dict) -> list[dict]:
    rows: list[dict] = []
    for dataset, block in report["protocols"].items():
        for arm, arm_block in block["arms"].items():
            coverage_base = {"dataset": dataset, "protocol": block["name"], "arm": arm, "split": None,
                             "level": "coverage", "group": None, "env": None, "subject": None}
            rows.append(_value_row(coverage_base, "", "n_seeds", arm_block["coverage"]["n_seeds"]))
            rows.extend(_value_row(coverage_base, str(seed), "missing_seed", 1)
                        for seed in arm_block["coverage"]["missing_seeds"])
            for split, levels in arm_block["splits"].items():
                for level, groups in levels.items():
                    for group in groups:
                        base = {"dataset": dataset, "protocol": block["name"], "arm": arm, "split": split,
                                "level": level, "group": group["group"], "env": group["env"],
                                "subject": group["subject"]}
                        rows.extend(_aggregated_group_rows(base, group))
    return rows


def _arm_columns(block: dict) -> list[str]:
    return [arm for arm in ARMS if block["arms"][arm]["splits"]]


def _macro_table(block: dict, split: str, level: str, title: str) -> list[str]:
    arms = _arm_columns(block)
    first = block["arms"][arms[0]]["splits"][split][level]
    lines = [
        "",
        f"{title} — Macro-F1 restrita, média ± desvio entre seeds:",
        "",
        "| Grupo | Janelas rotuladas | Classes restritas ausentes | " + " | ".join(arms) + " |",
        "| --- | --- | --- |" + " --- |" * len(arms),
    ]
    for group in first:
        cells = [_mean_std(_require_group(block["arms"][arm], split, level, group["group"])["macro_f1_restricted"])
                 for arm in arms]
        absent = ", ".join(group["absent_restricted_labels"]) or "—"
        lines.append(f"| {_group_label(level, group)} | {group['n_labeled_windows']} | {absent} | "
                     + " | ".join(cells) + " |")
    return lines


def _confusion_cell(group: dict) -> str:
    consistency = group["dominant_confusion_consistency"]
    reference = consistency["reference_top1"]
    if reference is None:
        modal = consistency["modal_top1"]
        return "n/d" if modal is None else f"(modal) {_arrow(modal)} {consistency['modal_top1_seeds']}/{consistency['n_seeds']}"
    return f"{_arrow(reference)} {consistency['reference_top1_seeds']}/{consistency['n_seeds']}"


def _confusion_table(block: dict, split: str) -> list[str]:
    arms = _arm_columns(block)
    lines = [
        "",
        f"Confusão dominante da seed {REFERENCE_SEED} e em quantas seeds ela é a dominante:",
        "",
        "| Grupo | " + " | ".join(arms) + " |",
        "| --- |" + " --- |" * len(arms),
    ]
    for level in ("split", "environment"):
        for group in block["arms"][arms[0]]["splits"][split][level]:
            cells = [_confusion_cell(_require_group(block["arms"][arm], split, level, group["group"]))
                     for arm in arms]
            lines.append(f"| {_group_label(level, group)} | " + " | ".join(cells) + " |")
    return lines


def _residual_table(block: dict, split: str) -> list[str]:
    arms = _arm_columns(block)
    lines = [
        "",
        f"Fração das janelas fora de {RESIDUAL_LABEL} preditas como {RESIDUAL_LABEL}, média ± desvio entre seeds "
        "(mínimo a máximo):",
        "",
        "| Grupo | " + " | ".join(arms) + " |",
        "| --- |" + " --- |" * len(arms),
    ]
    for level in ("split", "environment"):
        for group in block["arms"][arms[0]]["splits"][split][level]:
            cells = [_rate_range(_require_group(block["arms"][arm], split, level, group["group"])) for arm in arms]
            lines.append(f"| {_group_label(level, group)} | " + " | ".join(cells) + " |")
    return lines


def _recall_table(block: dict, split: str) -> list[str]:
    arms = _arm_columns(block)
    overall = {arm: block["arms"][arm]["splits"][split]["split"][0] for arm in arms}
    lines = [
        "",
        "Recall por classe no split inteiro, média ± desvio entre seeds (alvo de erro modal):",
        "",
        "| Classe | Suporte | " + " | ".join(arms) + " |",
        "| --- | --- |" + " --- |" * len(arms),
    ]
    for label in LE2I_LABEL_NAMES:
        first = overall[arms[0]]["per_class"][label]
        if first["absent"]:
            continue
        cells = []
        for arm in arms:
            pattern = overall[arm]["per_class"][label]
            target = pattern["main_error_target"]["modal"]
            cells.append(_mean_std(pattern["recall"]) + (f" ({target})" if target else ""))
        lines.append(f"| {label} | {first['support']} | " + " | ".join(cells) + " |")
    return lines


def render_summary(report: dict) -> str:
    definition = report["macro_f1_definition"]
    lines = [
        "# Consolidação multi-seed do diagnóstico estratificado de erros (não canônico)",
        "",
        "Consolidação exploratória e somente leitura de relatórios da Etapa 1B, sem inferência, treino, "
        "ajuste ou seleção. Cada estatística é calculada sobre os valores por seed (std amostral); contagens "
        "de janelas nunca são somadas entre seeds. Macro-F1 restrita = média simples do F1 sobre "
        f"{definition['restricted_labels']}, mesmo em grupos sem alguma dessas classes. As predições são as da "
        "reexecução validada, e a reconciliação com metrics.json não prova identidade com as predições "
        "históricas por janela. Detalhes por classe e por seed estão no JSON e no CSV.",
        "",
        "## Cobertura de seeds",
        "",
        "| Protocolo | Arma | Seeds presentes | Seeds ausentes |",
        "| --- | --- | --- | --- |",
    ]
    for block in report["protocols"].values():
        for arm, arm_block in block["arms"].items():
            coverage = arm_block["coverage"]
            lines.append(
                f"| {block['name']} | {arm} | {', '.join(map(str, coverage['seeds'])) or '—'} | "
                f"{', '.join(map(str, coverage['missing_seeds'])) or '—'} |"
            )
    for dataset, block in report["protocols"].items():
        lines += ["", f"## {block['name']} (`{dataset}`, {block['held_out_factor']} retidos)"]
        if not _arm_columns(block):
            lines += ["", "Nenhuma seed disponível."]
            continue
        for split in report["splits"]:
            lines += ["", f"### {split}"]
            lines += _macro_table(block, split, "environment", "Por ambiente")
            lines += _macro_table(block, split, "environment_subject", "Por ambiente e sujeito")
            lines += _confusion_table(block, split)
            lines += _residual_table(block, split)
            lines += _recall_table(block, split)
    lines += [
        "",
        f"## Estabilidade dos erros de {'/'.join(report['focus_arms'])} observados na seed {report['reference_seed']}",
        "",
        "Observações confirmadas (valores calculados):",
        "",
    ]
    lines += [f"- {item['text']}" for item in report["observations"]] or ["- Nenhuma seed disponível."]
    lines += ["", "Hipóteses (não verificadas):", ""]
    lines += [f"- {item['text']}" for item in report["hypotheses"]] or ["- Nenhuma regra disparada."]
    lines += ["", "## Entradas ausentes", ""]
    lines += [
        f"- {PROTOCOL_NAMES.get(item['dataset'], item['dataset'])}/{item['arm']}: {item['reason']}"
        + (f" — seeds {', '.join(map(str, item['missing_seeds']))}" if item["missing_seeds"] else "")
        + (f" (`{item['report']}`)" if item["report"] else "")
        for item in report["missing_inputs"]
    ] or ["- Nenhuma."]
    lines += ["", "## Limitações", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def _source_path(source: str) -> Path:
    path = Path(source)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def protected_paths(report_paths: list[Path], consolidation: dict) -> list[Path]:
    return [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        *(path.parent for path in report_paths),
        *(
            _source_path(run["source"])
            for block in consolidation["protocols"].values()
            for arm_block in block["arms"].values()
            for run in arm_block["runs"]
        ),
    ]


def _validate_report_paths(report_paths: list[Path]) -> None:
    if not report_paths:
        raise ValueError("nenhum --report fornecido")
    resolved = [path.resolve() for path in report_paths]
    duplicates = sorted({str(path) for path in resolved if resolved.count(path) > 1})
    if duplicates:
        raise ValueError(f"--report fornecido mais de uma vez: {duplicates}")


def _validate_expected_seeds(expected_seeds: tuple[int, ...]) -> None:
    if len(expected_seeds) < MIN_SEEDS or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError(f"--seed exige ao menos {MIN_SEEDS} seeds distintas; recebido {list(expected_seeds)}")


def run_consolidation(
    report_paths: list[Path],
    output_dir: Path,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
    force: bool = False,
) -> bool:
    _validate_report_paths(report_paths)
    _validate_expected_seeds(expected_seeds)
    snapshot = snapshot_inputs(sorted(path.resolve() for path in report_paths))
    consolidation = build_consolidation([(path, _read_json(path)) for path in report_paths], expected_seeds)
    check_output_dir(output_dir, protected_paths(report_paths, consolidation))
    consolidation["input_sha256"] = {display_path(Path(path)): digest for path, digest in snapshot.items()}
    targets = {
        output_dir / CONSOLIDATION_JSON_FILE: json.dumps(consolidation, indent=2, ensure_ascii=False, allow_nan=False),
        output_dir / CONSOLIDATION_CSV_FILE: pd.DataFrame(csv_rows(consolidation), columns=CSV_COLUMNS).to_csv(
            index=False
        ),
        output_dir / CONSOLIDATION_SUMMARY_FILE: render_summary(consolidation),
    }
    written = write_text_outputs(output_dir, targets, force)
    verify_inputs_unchanged(snapshot)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    consolidate_parser = subparsers.add_parser(
        "consolidate", help="Consolida entre seeds relatórios do diagnóstico estratificado de erros"
    )
    consolidate_parser.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="Repetível: stratified_error_diagnosis.json gravado pela Etapa 1B",
    )
    consolidate_parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help=f"Repetível: seeds esperadas (padrão: {', '.join(map(str, EXPECTED_SEEDS))})",
    )
    consolidate_parser.add_argument("--output-dir", type=Path, required=True)
    consolidate_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da consolidação multi-seed")

    args = parser.parse_args()
    if args.command == "consolidate":
        run_consolidation(
            args.report,
            args.output_dir,
            expected_seeds=tuple(args.seeds) if args.seeds else EXPECTED_SEEDS,
            force=args.force,
        )
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.multiseed_error_consolidation import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
