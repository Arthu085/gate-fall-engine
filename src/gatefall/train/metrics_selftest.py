"""Selftest sintético das métricas (`metrics.py`). Não toca em dados reais."""

import sys

import numpy as np

from gatefall.config import NUM_CLASSES
from gatefall.train.metrics import RESTRICTED_CLASSES, restricted_macro_f1, support


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


def run_metrics_selftest() -> bool:
    checks = [
        check_restricted_classes_set(),
        check_perfect_prediction_gives_f1_1(),
        check_support_all_10_keys(),
    ]
    ok = all(checks)
    if not ok:
        print("\nmetrics selftest FALHOU", file=sys.stderr)
    else:
        print("\nmetrics selftest OK: todas as checagens passaram")
    return ok
