"""Montagem e publicação dos relatórios de classificação dos braços de treino."""

import json
import math
import os
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from gatefall.hashing import sha256_file
from gatefall.train.shared.metrics import (
    BINARY_POSITIVE_LABELS,
    RESTRICTED_CLASSES,
    binary_projection_summary,
    class_support_table,
    classification_summary,
    macro_f1_policy_summary,
    restricted_macro_f1,
    support,
)


def _print_class_support_table(rows: list[dict]) -> None:
    columns = (
        "id",
        "label",
        "train_support",
        "val_support",
        "test_support",
        "included_in_macro_f1",
    )
    formatted_rows = [
        {column: str(row[column]) for column in columns} for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in formatted_rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def _print_macro_f1_policy_summary(policy_summary: dict) -> None:
    restricted = policy_summary["restricted_classes"]
    excluded = policy_summary["excluded_classes"]
    with_support = policy_summary["classes_with_positive_train_support"]
    if policy_summary["matches_configured_restriction"]:
        print(
            f"política de macro-F1: classes restritas {restricted}, "
            f"classes excluídas {excluded}, classes com suporte de treino "
            f"positivo {with_support} (conjuntos coincidem)"
        )
    else:
        print(
            f"ATENÇÃO: DESCASAMENTO na política de macro-F1: classes restritas "
            f"configuradas {restricted}, classes excluídas {excluded}, mas "
            f"classes com suporte de treino positivo {with_support} "
            f"(conjuntos NÃO coincidem)"
        )


def publish_classification_report(
    *,
    predictions: Iterable[tuple[str, np.ndarray, np.ndarray]],
    label_names: tuple[str, ...],
    num_classes: int,
    run_name: str,
    dataset_name: str,
    run_dir: Path,
    checkpoint_path: Path,
    device: str,
    output_path: Path,
) -> bool:
    splits_report: dict[str, dict] = {}
    mismatches: list[dict] = []
    support_by_split: dict[str, dict[int, int]] = {}

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open(encoding="utf-8") as f:
        stored_metrics = json.load(f)
    stored_final = stored_metrics["final"]

    for split_name, y_true, y_pred in predictions:
        summary = classification_summary(y_true, y_pred, label_names, num_classes)
        binary = binary_projection_summary(y_true, y_pred, BINARY_POSITIVE_LABELS)
        splits_report[split_name] = {
            "confusion_matrix": summary["confusion_matrix"],
            "per_class": summary["per_class"],
            "binary_fall_fallen": binary,
        }

        stored_split = stored_final[split_name]
        macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred, num_classes)
        stored_macro_f1 = stored_split["macro_f1_restricted"]
        if not math.isclose(stored_macro_f1, macro_f1, abs_tol=1e-12, rel_tol=0):
            mismatches.append(
                {
                    "split": split_name,
                    "field": "macro_f1_restricted",
                    "stored": stored_macro_f1,
                    "recomputed": macro_f1,
                }
            )
        stored_f1_by_class = stored_split["f1_by_class"]
        for c in RESTRICTED_CLASSES:
            stored_value = stored_f1_by_class[str(c)]
            recomputed_value = f1_by_class[c]
            if not math.isclose(stored_value, recomputed_value, abs_tol=1e-12, rel_tol=0):
                mismatches.append(
                    {
                        "split": split_name,
                        "field": f"f1_by_class[{c}]",
                        "stored": stored_value,
                        "recomputed": recomputed_value,
                    }
                )
        recomputed_support = support(y_true, num_classes)
        support_by_split[split_name] = recomputed_support
        stored_support = stored_split["support"]
        for c in range(num_classes):
            label_name = label_names[c]
            stored_count = stored_support[label_name]
            recomputed_count = recomputed_support[c]
            if stored_count != recomputed_count:
                mismatches.append(
                    {
                        "split": split_name,
                        "field": f"support[{label_name}]",
                        "stored": stored_count,
                        "recomputed": recomputed_count,
                    }
                )

    ok = len(mismatches) == 0
    support_table = class_support_table(
        label_names,
        support_by_split["train"],
        support_by_split["val"],
        support_by_split["test"],
    )
    policy_summary = macro_f1_policy_summary(support_by_split["train"])

    report = {
        "run_name": run_name,
        "dataset": dataset_name,
        "run_dir": str(run_dir),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": device,
        "splits": splits_report,
        "verification_against_metrics_json": {"ok": ok, "mismatches": mismatches},
        "class_support_table": support_table,
        "macro_f1_policy": policy_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)

    print(f"{output_path}: relatório de classificação gravado (run_name={run_name})")
    _print_class_support_table(support_table)
    _print_macro_f1_policy_summary(policy_summary)
    if not ok:
        print(
            f"verificação contra metrics.json falhou: {len(mismatches)} divergência(s)",
            file=sys.stderr,
        )
    return ok
