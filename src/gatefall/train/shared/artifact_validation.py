"""Validação dos diagnósticos de classificação compartilhados entre armas."""

import math
from collections.abc import Mapping
from typing import Any


def validate_classification_diagnostics(
    split_metrics: Mapping[str, Any],
    num_classes: int,
    split: str,
    total_support: int,
) -> None:
    prefix = f"metrics.json.final.{split}"

    matrix = split_metrics.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != num_classes
        or any(
            not isinstance(row, list)
            or len(row) != num_classes
            or any(not isinstance(value, int) or value < 0 for value in row)
            for row in matrix
        )
    ):
        raise ValueError(
            f"{prefix}.confusion_matrix deve ser {num_classes}x{num_classes} "
            "de inteiros não negativos"
        )
    matrix_total = sum(sum(row) for row in matrix)
    if matrix_total != total_support:
        raise ValueError(
            f"{prefix}.confusion_matrix: soma total ({matrix_total}) diverge "
            f"do suporte total do split ({total_support})"
        )

    per_class = split_metrics.get("per_class")
    if not isinstance(per_class, Mapping) or len(per_class) != num_classes:
        raise ValueError(f"{prefix}.per_class deve conter {num_classes} entradas")

    seen_ids: set[int] = set()
    for name, entry in per_class.items():
        entry_prefix = f"{prefix}.per_class[{name!r}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{entry_prefix} deve ser um objeto")

        class_id = entry.get("id")
        if (
            not isinstance(class_id, int)
            or isinstance(class_id, bool)
            or not (0 <= class_id < num_classes)
        ):
            raise ValueError(f"{entry_prefix}.id deve ser um inteiro em [0, {num_classes})")
        if class_id in seen_ids:
            raise ValueError(f"{entry_prefix}.id repetido: {class_id}")
        seen_ids.add(class_id)

        int_fields = ("tp", "tn", "fp", "fn", "support")
        for field in int_fields:
            value = entry.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{entry_prefix}.{field} deve ser um inteiro não negativo")

        tp, tn, fp, fn, class_support = (entry[field] for field in int_fields)
        if tp + fn != class_support:
            raise ValueError(f"{entry_prefix}: tp + fn != support")
        if tn != total_support - tp - fp - fn:
            raise ValueError(f"{entry_prefix}: tn != N - tp - fp - fn")

        row_sum = sum(matrix[class_id])
        if row_sum != class_support:
            raise ValueError(
                f"{entry_prefix}: soma da linha {class_id} da confusion_matrix "
                f"({row_sum}) diverge de support ({class_support})"
            )

        matrix_tp = matrix[class_id][class_id]
        if tp != matrix_tp:
            raise ValueError(
                f"{entry_prefix}: tp ({tp}) diverge de confusion_matrix[{class_id}][{class_id}] "
                f"({matrix_tp})"
            )
        matrix_fn = row_sum - matrix_tp
        if fn != matrix_fn:
            raise ValueError(
                f"{entry_prefix}: fn ({fn}) diverge da confusion_matrix "
                f"(soma da linha {class_id} menos a diagonal = {matrix_fn})"
            )
        column_sum = sum(row[class_id] for row in matrix)
        matrix_fp = column_sum - matrix_tp
        if fp != matrix_fp:
            raise ValueError(
                f"{entry_prefix}: fp ({fp}) diverge da confusion_matrix "
                f"(soma da coluna {class_id} menos a diagonal = {matrix_fp})"
            )

        float_fields = ("precision", "recall", "f1")
        for field in float_fields:
            value = entry.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{entry_prefix}.{field} deve ser numérico")
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"{entry_prefix}.{field} deve ser finito e estar em [0,1]")

        precision_denom = tp + fp
        recall_denom = tp + fn
        expected_precision = tp / precision_denom if precision_denom > 0 else 0.0
        expected_recall = tp / recall_denom if recall_denom > 0 else 0.0
        if expected_precision + expected_recall == 0:
            expected_f1 = 0.0
        else:
            expected_f1 = (
                2 * expected_precision * expected_recall / (expected_precision + expected_recall)
            )
        for field, expected in (
            ("precision", expected_precision),
            ("recall", expected_recall),
            ("f1", expected_f1),
        ):
            if not math.isclose(entry[field], expected, abs_tol=1e-9, rel_tol=0):
                raise ValueError(
                    f"{entry_prefix}.{field} ({entry[field]}) diverge do valor "
                    f"recalculado ({expected})"
                )

    if seen_ids != set(range(num_classes)):
        raise ValueError(f"{prefix}.per_class: ids devem cobrir range(0, {num_classes})")
