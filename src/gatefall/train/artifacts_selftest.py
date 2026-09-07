"""Selftest sintético da validação per_class vs confusion_matrix (`artifacts.py`). Não toca em dados reais."""

import sys
from dataclasses import replace

from gatefall.train.artifacts import validate_training_metrics
from gatefall.train.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.metrics import RESTRICTED_CLASSES


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _class_counts_from_matrix(
    matrix: list[list[int]], class_id: int
) -> tuple[int, int, int, int]:
    total = sum(sum(row) for row in matrix)
    tp = matrix[class_id][class_id]
    row_sum = sum(matrix[class_id])
    col_sum = sum(row[class_id] for row in matrix)
    fn = row_sum - tp
    fp = col_sum - tp
    tn = total - tp - fp - fn
    return tp, fp, fn, tn


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _build_confusion_matrix(num_classes: int) -> list[list[int]]:
    # Classes 0 e 1 trocam massa fora da diagonal (fp/fn reais não nulos);
    # as demais classes só têm a diagonal, para manter o restante do
    # metrics.json sintético trivialmente consistente.
    matrix = [[0] * num_classes for _ in range(num_classes)]
    matrix[0][0] = 5
    matrix[0][1] = 3
    matrix[1][0] = 2
    matrix[1][1] = 4
    for c in range(2, num_classes):
        matrix[c][c] = 1
    return matrix


def _build_split_metrics(num_classes: int, matrix: list[list[int]]) -> dict:
    per_class: dict[str, dict] = {}
    support: dict[str, int] = {}
    for c in range(num_classes):
        tp, fp, fn, tn = _class_counts_from_matrix(matrix, c)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        class_support = tp + fn
        support[str(c)] = class_support
        per_class[str(c)] = {
            "id": c,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "support": class_support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    f1_by_class = {str(c): per_class[str(c)]["f1"] for c in RESTRICTED_CLASSES}
    macro_f1 = sum(f1_by_class.values()) / len(f1_by_class)
    return {
        "macro_f1_restricted": macro_f1,
        "f1_by_class": f1_by_class,
        "support": support,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def _build_metrics_payload(num_classes: int) -> tuple[dict, TrainConfig]:
    config = replace(BASELINE_A_CONFIG, epochs=1)
    matrix = _build_confusion_matrix(num_classes)
    split_metrics = _build_split_metrics(num_classes, matrix)
    excluded_classes = [c for c in range(num_classes) if c not in RESTRICTED_CLASSES]
    data = {
        "run_name": config.run_name,
        "epochs_trained": config.epochs,
        "history": [
            {"epoch": 1, "train_loss": 0.1, "val_macro_f1_restricted": 0.5},
        ],
        "restricted_classes": RESTRICTED_CLASSES,
        "excluded_classes": excluded_classes,
        "final": {
            "train": split_metrics,
            "val": split_metrics,
            "test": split_metrics,
        },
    }
    return data, config


def check_per_class_rejected_when_matrix_inconsistent() -> bool:
    num_classes = BASELINE_A_CONFIG.num_classes
    data, config = _build_metrics_payload(num_classes)

    original_test = data["final"]["test"]
    entry = dict(original_test["per_class"]["0"])
    tp = entry["tp"]
    fn = entry["fn"]
    true_fp = entry["fp"]

    # Desloca a contagem de fp da classe 0 para longe do valor real
    # (col_sum(matriz, 0) - tp) mantendo tp/fn/support inalterados e
    # recalculando tn/precision/recall/f1 a partir do próprio bloco
    # per_class corrompido — ou seja, o bloco continua autoconsistente,
    # apenas divergindo da confusion_matrix.
    corrupted_fp = true_fp + 2
    total_support = sum(int(value) for value in original_test["support"].values())
    corrupted_tn = total_support - tp - corrupted_fp - fn
    precision, recall, f1 = _precision_recall_f1(tp, corrupted_fp, fn)
    entry.update(
        fp=corrupted_fp,
        tn=corrupted_tn,
        precision=precision,
        recall=recall,
        f1=f1,
    )

    per_class = dict(original_test["per_class"])
    per_class["0"] = entry
    test_split = dict(original_test)
    test_split["per_class"] = per_class
    data["final"] = dict(data["final"])
    data["final"]["test"] = test_split

    raised = False
    try:
        validate_training_metrics(data, config, config_path=None, checkpoint_path=None)
    except ValueError:
        raised = True

    return _check(
        "validate_training_metrics rejeita per_class[0].fp que diverge de "
        "confusion_matrix (col_sum - tp) mesmo quando o bloco per_class "
        "permanece autoconsistente (tp+fn==support, tn==N-tp-fp-fn, "
        "precision/recall/f1 recalculados a partir dos próprios contadores)",
        raised,
    )


def run_artifacts_selftest() -> bool:
    checks = [
        check_per_class_rejected_when_matrix_inconsistent(),
    ]
    ok = all(checks)
    if not ok:
        print("\nartifacts selftest FALHOU", file=sys.stderr)
    else:
        print("\nartifacts selftest OK: todas as checagens passaram")
    return ok
