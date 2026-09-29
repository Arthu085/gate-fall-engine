"""Selftest sintético da política de seleção contínua do SAM 3 (`selection.py`).

Não toca no dataset real nem no runtime SAM 3 — todas as máscaras são
sintéticas, para travar a ordem de decisão (continuidade por IoU, fallback
por score) e a invariante de cobertura contra futuras mudanças.
"""

import sys

import numpy as np

from gatefall.sam3.runtime import Sam3Instance
from gatefall.sam3.selection import InstanceSelector, bbox_iou


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _rect_mask(
    x_min: int, y_min: int, x_max: int, y_max: int, *, height: int = 50, width: int = 50
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[y_min : y_max + 1, x_min : x_max + 1] = True
    return mask


EMPTY_MASK = np.zeros((50, 50), dtype=bool)

LEFT_MASK = _rect_mask(0, 0, 9, 19)
RIGHT_MASK = _rect_mask(30, 0, 39, 19)
SHIFTED_RIGHT_MASK = _rect_mask(32, 0, 41, 19)
FAR_MASK = _rect_mask(45, 45, 49, 49, height=60, width=60)


def check_acquisition_by_score() -> bool:
    selector = InstanceSelector()
    index = selector.select(
        [Sam3Instance(mask=LEFT_MASK, score=0.6), Sam3Instance(mask=RIGHT_MASK, score=0.9)]
    )
    ok = index == 1 and selector.last_bbox is not None
    return _check("aquisição inicial escolhe o maior score do SAM", ok)


def check_continuity_beats_higher_score_distractor() -> bool:
    selector = InstanceSelector()
    selector.select(
        [Sam3Instance(mask=LEFT_MASK, score=0.6), Sam3Instance(mask=RIGHT_MASK, score=0.9)]
    )
    index = selector.select(
        [
            Sam3Instance(mask=SHIFTED_RIGHT_MASK, score=0.4),
            Sam3Instance(mask=FAR_MASK, score=0.99),
        ]
    )
    ok = index == 0
    return _check(
        "continuidade por IoU vence um distrator de score maior sem overlap", ok
    )


def check_reacquisition_when_tracked_mask_disappears() -> bool:
    selector = InstanceSelector()
    selector.select(
        [Sam3Instance(mask=LEFT_MASK, score=0.6), Sam3Instance(mask=RIGHT_MASK, score=0.9)]
    )
    index = selector.select(
        [Sam3Instance(mask=LEFT_MASK, score=0.95), Sam3Instance(mask=FAR_MASK, score=0.3)]
    )
    ok = index == 0
    return _check(
        "instância seguida some do quadro: reancora pelo maior score restante",
        ok,
    )


def check_gap_preserves_state_and_resumes_continuity() -> bool:
    selector = InstanceSelector()
    selector.select(
        [Sam3Instance(mask=LEFT_MASK, score=0.6), Sam3Instance(mask=RIGHT_MASK, score=0.9)]
    )
    bbox_before = selector.last_bbox
    gap_results = [selector.select([]), selector.select([])]
    state_survived = (
        selector.last_bbox is not None
        and bbox_before is not None
        and bool(np.array_equal(selector.last_bbox, bbox_before))
    )
    index = selector.select(
        [
            Sam3Instance(mask=SHIFTED_RIGHT_MASK, score=0.99),
            Sam3Instance(mask=FAR_MASK, score=0.4),
        ]
    )
    ok = all(result is None for result in gap_results) and state_survived and index == 0
    return _check(
        "lacuna de detecção não muta o estado e um quadro futuro retoma a "
        "continuidade",
        ok,
    )


def check_all_zero_iou_falls_back_to_score() -> bool:
    selector = InstanceSelector()
    selector.select([Sam3Instance(mask=_rect_mask(0, 0, 4, 4), score=0.9)])
    index = selector.select(
        [
            Sam3Instance(mask=_rect_mask(20, 20, 24, 24, height=60, width=60), score=0.3),
            Sam3Instance(mask=_rect_mask(30, 30, 34, 34, height=60, width=60), score=0.8),
        ]
    )
    ok = index == 1
    return _check("IoU zero em todas as instâncias cai no fallback por score", ok)


def check_exact_iou_tie_broken_by_score() -> bool:
    selector = InstanceSelector()
    selector.select([Sam3Instance(mask=RIGHT_MASK, score=0.9)])
    index = selector.select(
        [Sam3Instance(mask=RIGHT_MASK, score=0.4), Sam3Instance(mask=RIGHT_MASK, score=0.9)]
    )
    ok = index == 1
    return _check("empate exato de IoU é desfeito pelo score do SAM", ok)


def check_select_returns_none_iff_zero_instances() -> bool:
    selector = InstanceSelector()
    ok = selector.select([]) is None
    ok = ok and selector.select([Sam3Instance(mask=LEFT_MASK, score=0.5)]) is not None
    return _check("select devolve None se e somente se a lista de instâncias é vazia", ok)


def check_empty_mask_candidates_are_excluded() -> bool:
    selector = InstanceSelector()
    index = selector.select(
        [
            Sam3Instance(mask=EMPTY_MASK, score=0.99),
            Sam3Instance(mask=RIGHT_MASK, score=0.2),
        ]
    )
    ok = index == 1
    return _check(
        "instância de máscara vazia é excluída mesmo tendo o maior score", ok
    )


def check_all_masks_empty_still_returns_index() -> bool:
    selector = InstanceSelector()
    index = selector.select(
        [Sam3Instance(mask=EMPTY_MASK, score=0.3), Sam3Instance(mask=EMPTY_MASK, score=0.7)]
    )
    ok = index == 1
    return _check(
        "lista não vazia com todas as máscaras degeneradas ainda devolve um "
        "índice (cai no score bruto)",
        ok,
    )


def check_bbox_iou_numerics() -> bool:
    previous = np.array((0.0, 0.0, 10.0, 10.0), dtype=np.float32)
    candidates = np.array(
        [
            (0.0, 0.0, 10.0, 10.0),
            (5.0, 0.0, 15.0, 10.0),
            (20.0, 20.0, 30.0, 30.0),
            (5.0, 5.0, 5.0, 5.0),
            (np.nan, 0.0, 10.0, 10.0),
            (10.0, 0.0, 5.0, 10.0),
        ],
        dtype=np.float32,
    )
    ious = bbox_iou(previous, candidates)
    expected = np.array([1.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.0])
    ok = bool(np.allclose(ious, expected)) and bool(np.isfinite(ious).all())
    return _check(
        "bbox_iou: valores esperados, finitos mesmo com NaN, área nula ou "
        "negativa",
        ok,
    )


def run_sam3_selection_selftest() -> None:
    checks = [
        check_acquisition_by_score(),
        check_continuity_beats_higher_score_distractor(),
        check_reacquisition_when_tracked_mask_disappears(),
        check_gap_preserves_state_and_resumes_continuity(),
        check_all_zero_iou_falls_back_to_score(),
        check_exact_iou_tie_broken_by_score(),
        check_select_returns_none_iff_zero_instances(),
        check_empty_mask_candidates_are_excluded(),
        check_all_masks_empty_still_returns_index(),
        check_bbox_iou_numerics(),
    ]
    if not all(checks):
        print("\nsam3 selection selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nsam3 selection selftest OK: todas as checagens passaram")
