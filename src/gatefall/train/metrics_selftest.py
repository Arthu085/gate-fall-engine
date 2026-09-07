"""Selftest sintético das métricas (`metrics.py`). Não toca em dados reais."""

import math
import sys

import numpy as np

from gatefall.config import NUM_CLASSES
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.train.metrics import (
    RESTRICTED_CLASSES,
    binary_projection_summary,
    classification_summary,
    restricted_macro_f1,
    support,
)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_restricted_classes_set() -> bool:
    ok = set(RESTRICTED_CLASSES) == {0, 1, 2, 3, 4, 7, 8, 9} and len(RESTRICTED_CLASSES) == 8
    return _check("RESTRICTED_CLASSES == {0,1,2,3,4,7,8,9}, comprimento 8", ok)


def check_perfect_prediction_gives_f1_1() -> bool:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, NUM_CLASSES, size=200)
    y_true[y_true == 5] = 0
    y_true[y_true == 6] = 0
    y_pred = y_true.copy()

    macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred)
    ok = macro_f1 == 1.0 and all(f1_by_class[c] == 1.0 for c in RESTRICTED_CLASSES)

    y_pred_corrupted = y_pred.copy()
    corrupt_mask = np.isin(y_true, [5, 6])
    y_pred_corrupted[corrupt_mask] = 5

    y_true_with_56 = y_true.copy()
    y_true_with_56[:5] = 5
    y_true_with_56[5:10] = 6
    y_pred_with_56 = y_true_with_56.copy()
    macro_f1_with_56, f1_by_class_with_56 = restricted_macro_f1(y_true_with_56, y_pred_with_56)

    y_pred_with_56_corrupted = y_pred_with_56.copy()
    y_pred_with_56_corrupted[y_true_with_56 == 5] = 6
    y_pred_with_56_corrupted[y_true_with_56 == 6] = 5
    macro_f1_corrupted, f1_by_class_corrupted = restricted_macro_f1(
        y_true_with_56, y_pred_with_56_corrupted
    )

    bit_identical = macro_f1_with_56 == macro_f1_corrupted and f1_by_class_with_56 == (
        f1_by_class_corrupted
    )

    return _check(
        "predição perfeita nas 8 classes restritas dá macro_f1==1.0 e permanece "
        "bit-idêntica quando predições das classes 5/6 são corrompidas",
        ok and bit_identical,
    )


def check_support_all_10_keys() -> bool:
    y_true = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    counts = support(y_true)
    ok = set(counts.keys()) == set(range(NUM_CLASSES))
    ok = ok and counts[0] == 2 and counts[1] == 1 and counts[2] == 3
    ok = ok and all(counts[c] == 0 for c in range(NUM_CLASSES) if c not in {0, 1, 2})
    return _check(
        "support(): reporta as 10 chaves, incluindo classes de contagem zero",
        ok,
    )


def check_confusion_matrix_multiclass() -> bool:
    # y_true/y_pred exercitam as classes 0, 1 e 2 com acertos e erros conhecidos.
    # idx: 0    1    2    3    4    5    6
    y_true = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 2, 2, 2, 0], dtype=np.int64)

    # Matriz esperada (linha=true, coluna=pred), calculada à mão:
    # true=0: 1x pred=0, 1x pred=1
    # true=1: 1x pred=1, 1x pred=2
    # true=2: 1x pred=0, 2x pred=2
    expected = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    expected[0][0] = 1
    expected[0][1] = 1
    expected[1][1] = 1
    expected[1][2] = 1
    expected[2][0] = 1
    expected[2][2] = 2

    result = classification_summary(y_true, y_pred, LE2I_LABEL_NAMES)
    matrix = result["confusion_matrix"]
    total = sum(sum(row) for row in matrix)
    ok = matrix == expected and total == len(y_true)
    return _check(
        "classification_summary(): confusion_matrix bate com o cálculo manual "
        "e soma total == len(y_true)",
        ok,
    )


def check_classification_summary_zero_support_class() -> bool:
    lying_id = LE2I_LABEL_NAMES.index("lying")
    assert lying_id == 6

    # y_true nunca contém a classe "lying" (id 6), mas y_pred a prediz 2x
    # (índices 1 e 3), gerando falsos positivos sem nenhum suporte real.
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    y_pred = np.array([0, 6, 1, 6, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    n_samples = len(y_true)

    result = classification_summary(y_true, y_pred, LE2I_LABEL_NAMES)
    lying_stats = result["per_class"]["lying"]

    fp = lying_stats["fp"]
    ok = (
        lying_stats["tp"] == 0
        and lying_stats["fn"] == 0
        and fp == 2
        and lying_stats["tn"] == n_samples - fp
        and lying_stats["precision"] == 0.0
    )
    return _check(
        "classification_summary(): classe sem suporte real (lying) com falsos "
        "positivos tem tp=0, fn=0, tn==N-fp e precision==0.0",
        ok,
    )


def check_classification_summary_matches_restricted_f1() -> bool:
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, NUM_CLASSES, size=300)
    y_pred = rng.integers(0, NUM_CLASSES, size=300)
    # Força suporte real zero para as classes 5 (lie_down) e 6 (lying),
    # replicando o cenário real do Le2i em stride 4.
    y_true[y_true == 5] = 0
    y_true[y_true == 6] = 0

    _, f1_by_class = restricted_macro_f1(y_true, y_pred)
    result = classification_summary(y_true, y_pred, LE2I_LABEL_NAMES)

    ok = all(
        result["per_class"][LE2I_LABEL_NAMES[c]]["f1"] == f1_by_class[c]
        for c in RESTRICTED_CLASSES
    )
    return _check(
        "classification_summary(): f1 por classe restrita é idêntico ao "
        "calculado por restricted_macro_f1() para os mesmos dados",
        ok,
    )


def check_binary_projection_fall_fallen() -> bool:
    # ids 1=fall, 2=fallen em LE2I_LABEL_NAMES.
    assert LE2I_LABEL_NAMES[1] == "fall" and LE2I_LABEL_NAMES[2] == "fallen"

    # idx:      0    1    2    3    4    5    6    7    8    9
    y_true = np.array([0, 1, 1, 2, 2, 3, 4, 7, 8, 9], dtype=np.int64)
    y_pred = np.array([0, 1, 2, 2, 3, 3, 4, 7, 1, 9], dtype=np.int64)

    # Positivos verdadeiros (true em {1,2}): índices 1,2,3,4.
    # idx1: true=1,pred=1 -> tp; idx2: true=1,pred=2 -> tp (pred positivo);
    # idx3: true=2,pred=2 -> tp; idx4: true=2,pred=3 -> fn (pred negativo).
    # idx8: true=8 (negativo),pred=1 (positivo) -> fp.
    # Demais negativos (0,5,6,7,9) predizem negativo -> tn.
    result = binary_projection_summary(y_true, y_pred, positive_labels=frozenset({1, 2}))

    expected_tp, expected_fn, expected_fp, expected_tn = 3, 1, 1, 5
    expected_precision = 3 / 4
    expected_recall = 3 / 4
    expected_specificity = 5 / 6
    expected_f1 = 0.75
    expected_accuracy = 0.8

    ok = (
        result["tp"] == expected_tp
        and result["fn"] == expected_fn
        and result["fp"] == expected_fp
        and result["tn"] == expected_tn
        and math.isclose(result["precision"], expected_precision, abs_tol=1e-9)
        and math.isclose(result["recall"], expected_recall, abs_tol=1e-9)
        and math.isclose(result["specificity"], expected_specificity, abs_tol=1e-9)
        and math.isclose(result["f1"], expected_f1, abs_tol=1e-9)
        and math.isclose(result["accuracy"], expected_accuracy, abs_tol=1e-9)
    )
    return _check(
        "binary_projection_summary(): tp/tn/fp/fn e métricas derivadas batem "
        "com o cálculo manual para positive_labels={fall,fallen}",
        ok,
    )


def run_metrics_selftest() -> bool:
    checks = [
        check_restricted_classes_set(),
        check_perfect_prediction_gives_f1_1(),
        check_support_all_10_keys(),
        check_confusion_matrix_multiclass(),
        check_classification_summary_zero_support_class(),
        check_classification_summary_matches_restricted_f1(),
        check_binary_projection_fall_fallen(),
    ]
    ok = all(checks)
    if not ok:
        print("\nmetrics selftest FALHOU", file=sys.stderr)
    else:
        print("\nmetrics selftest OK: todas as checagens passaram")
    return ok
