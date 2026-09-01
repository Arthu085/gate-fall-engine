"""Métricas de classificação restritas às classes com suporte real no Le2i."""

import numpy as np

from gatefall.config import NUM_CLASSES

# 5=lie_down tem suporte zero no treino em stride 4 (ver
# EXPECTED_USABLE_WINDOWS_BY_LABEL_STRIDE4 em gatefall.data.le2i.pose_dataset),
# então seu F1 é sempre 0 e derrubaria a média macro em ~1/9; 6=lying nunca
# ocorre no Le2i. Ambas ficam fora da média macro.
RESTRICTED_CLASSES: list[int] = [0, 1, 2, 3, 4, 7, 8, 9]


def per_class_counts(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        pred_c = y_pred == c
        true_c = y_true == c
        tp[c] = int(np.sum(pred_c & true_c))
        fp[c] = int(np.sum(pred_c & ~true_c))
        fn[c] = int(np.sum(~pred_c & true_c))
    return tp, fp, fn


def restricted_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> tuple[float, dict[int, float]]:
    tp, fp, fn = per_class_counts(y_true, y_pred, num_classes)

    f1_by_class: dict[int, float] = {}
    for c in RESTRICTED_CLASSES:
        denom = tp[c] + fp[c] + fn[c]
        if denom == 0:
            f1_by_class[c] = 0.0
            continue
        precision_denom = tp[c] + fp[c]
        recall_denom = tp[c] + fn[c]
        precision = tp[c] / precision_denom if precision_denom > 0 else 0.0
        recall = tp[c] / recall_denom if recall_denom > 0 else 0.0
        if precision + recall == 0:
            f1_by_class[c] = 0.0
        else:
            f1_by_class[c] = 2 * precision * recall / (precision + recall)

    macro_f1 = float(np.mean([f1_by_class[c] for c in RESTRICTED_CLASSES]))
    return macro_f1, f1_by_class


def support(y_true: np.ndarray, num_classes: int = NUM_CLASSES) -> dict[int, int]:
    return {c: int(np.sum(y_true == c)) for c in range(num_classes)}


def support_by_name(
    y_true: np.ndarray, label_names: tuple[str, ...], num_classes: int = NUM_CLASSES
) -> dict[str, int]:
    counts = support(y_true, num_classes)
    return {label_names[c]: counts[c] for c in range(num_classes)}
