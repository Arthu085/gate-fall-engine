"""Selftest sintético da política de seleção contínua (`selection.py`).

Não toca no dataset real, no Ultralytics nem no disco — todos os quadros são
sintéticos, para travar a ordem de decisão (track ativa, reancoragem por IoU,
fallback por confiança) e a invariante de cobertura contra futuras mudanças.
"""

import itertools
import sys

import numpy as np

from gatefall.pose.selection import PersonSelector, bbox_iou


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _frame(
    boxes: list[tuple[float, float, float, float]],
    confs: list[float],
    ids: list[int] | None,
) -> tuple[int, np.ndarray, np.ndarray, list[int] | None]:
    return (
        len(boxes),
        np.array(confs, dtype=np.float32),
        np.array(boxes, dtype=np.float32),
        ids,
    )


LEFT_BOX = (0.0, 0.0, 10.0, 20.0)
RIGHT_BOX = (100.0, 0.0, 110.0, 20.0)
SHIFTED_RIGHT_BOX = (102.0, 0.0, 112.0, 20.0)
# Alvo deslocado para longe: separa a resposta da track ativa da resposta por
# IoU, que ficaria com o distrator sobreposto à bbox antiga.
FAR_BOX = (500.0, 300.0, 510.0, 320.0)
ZERO_AREA_BOX = (5.0, 5.0, 5.0, 5.0)


def check_initial_acquisition_by_confidence() -> bool:
    selector = PersonSelector()
    index = selector.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.6, 0.9], [1, 2]))
    ok = (
        index == 1
        and selector.active_track_id == 2
        and selector.last_bbox is not None
        and bool(np.array_equal(selector.last_bbox, np.array(RIGHT_BOX, np.float32)))
    )
    return _check("aquisição inicial escolhe a maior confiança e fixa a track", ok)


def check_active_track_beats_higher_confidence() -> bool:
    selector = PersonSelector()
    selector.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.6, 0.9], [1, 2]))
    index = selector.select(*_frame([SHIFTED_RIGHT_BOX, FAR_BOX], [0.99, 0.50], [1, 2]))
    ok = (
        index == 1
        and selector.active_track_id == 2
        and selector.last_bbox is not None
        and bool(np.array_equal(selector.last_bbox, np.array(FAR_BOX, np.float32)))
    )
    return _check("track ativa vence distrator de confiança e de IoU maiores", ok)


def check_iou_recovery_when_active_track_disappears() -> bool:
    selector = PersonSelector()
    selector.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.6, 0.9], [1, 2]))
    index = selector.select(
        *_frame([LEFT_BOX, SHIFTED_RIGHT_BOX], [0.95, 0.30], [1, 7])
    )
    ok = index == 1 and selector.active_track_id == 7
    return _check("track ativa some em quadro multi-pessoa: reancora por IoU", ok)


def check_gap_then_same_id_resumes() -> bool:
    selector = PersonSelector()
    selector.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.6, 0.9], [1, 2]))
    bbox_before = selector.last_bbox
    gaps = [
        selector.select(0, None, None, None),
        selector.select(0, None, None, None),
    ]
    state_survived = (
        selector.active_track_id == 2
        and selector.last_bbox is not None
        and bbox_before is not None
        and bool(np.array_equal(selector.last_bbox, bbox_before))
    )
    index = selector.select(*_frame([SHIFTED_RIGHT_BOX, FAR_BOX], [0.99, 0.4], [3, 2]))
    ok = (
        all(gap is None for gap in gaps)
        and state_survived
        and index == 1
        and selector.active_track_id == 2
    )
    return _check("perda não muta o estado e o mesmo ID retoma a seleção", ok)


def check_gap_then_new_id_on_continuous_person() -> bool:
    selector = PersonSelector()
    selector.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.6, 0.9], [1, 2]))
    gap = selector.select(0, None, None, None)
    index = selector.select(
        *_frame([LEFT_BOX, SHIFTED_RIGHT_BOX], [0.95, 0.30], [11, 12])
    )
    ok = gap is None and index == 1 and selector.active_track_id == 12
    return _check("após perda, reaquisição segue a pessoa contínua com novo ID", ok)


def check_missing_tracker_ids() -> bool:
    selector = PersonSelector()
    first = selector.select(
        len([LEFT_BOX, RIGHT_BOX]),
        np.array([0.3, 0.8], dtype=np.float32),
        np.array([LEFT_BOX, RIGHT_BOX], dtype=np.float32),
        None,
    )
    fresh_ok = first == 1 and selector.active_track_id is None
    second = selector.select(
        2,
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([LEFT_BOX, SHIFTED_RIGHT_BOX], dtype=np.float32),
        None,
    )
    ok = fresh_ok and second == 1 and selector.active_track_id is None
    return _check("sem track_ids, seleciona por confiança e depois por IoU", ok)


def check_single_detection_with_new_id() -> bool:
    selector = PersonSelector()
    selector.select(*_frame([LEFT_BOX], [0.9], [2]))
    index = selector.select(*_frame([FAR_BOX], [0.4], [99]))
    ok = (
        index == 0
        and selector.active_track_id == 99
        and selector.last_bbox is not None
        and bool(np.array_equal(selector.last_bbox, np.array(FAR_BOX, np.float32)))
    )
    return _check("detecção única com ID novo e sem overlap ainda é aceita", ok)


def check_immediate_track_change_without_gap() -> bool:
    selector = PersonSelector()
    selector.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.9, 0.5], [5, 9]))
    index = selector.select(*_frame([RIGHT_BOX], [0.5], [9]))
    ok = (
        index == 0
        and selector.active_track_id == 9
        and selector.last_bbox is not None
        and bool(np.array_equal(selector.last_bbox, np.array(RIGHT_BOX, np.float32)))
    )
    return _check("troca imediata de track sem gap não devolve None", ok)


def check_idless_frame_preserves_active_track() -> bool:
    selector = PersonSelector()
    selector.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.6, 0.9], [1, 2]))
    idless = selector.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.9, 0.5], None))
    survived = selector.active_track_id == 2
    index = selector.select(*_frame([SHIFTED_RIGHT_BOX, FAR_BOX], [0.99, 0.4], [1, 2]))
    ok = (
        idless == 1
        and survived
        and index == 1
        and selector.active_track_id == 2
    )
    return _check("quadro sem IDs não apaga a track ativa", ok)


def check_all_zero_iou_falls_back_to_confidence() -> bool:
    selector = PersonSelector()
    selector.select(*_frame([(0.0, 0.0, 5.0, 5.0)], [0.9], [2]))
    index = selector.select(
        *_frame([(200.0, 200.0, 210.0, 220.0), (300.0, 300.0, 310.0, 320.0)],
                [0.3, 0.8], [20, 21])
    )
    ok = index == 1 and selector.active_track_id == 21
    return _check("IoU zero em todas as detecções cai no fallback por confiança", ok)


def check_exact_iou_tie_broken_by_confidence() -> bool:
    selector = PersonSelector()
    selector.select(*_frame([RIGHT_BOX], [0.9], [2]))
    index = selector.select(*_frame([RIGHT_BOX, RIGHT_BOX], [0.4, 0.9], [30, 31]))
    ok = index == 1 and selector.active_track_id == 31
    return _check("empate exato de IoU é desfeito pela confiança da caixa", ok)


def check_state_is_per_selector_instance() -> bool:
    warm = PersonSelector()
    warm.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.6, 0.9], [1, 2]))
    fresh = PersonSelector()
    fresh_is_empty = fresh.active_track_id is None and fresh.last_bbox is None
    index = fresh.select(*_frame([RIGHT_BOX, LEFT_BOX], [0.2, 0.9], [2, 5]))
    ok = warm.active_track_id == 2 and fresh_is_empty and index == 1
    return _check("estado é por instância: um seletor novo não herda a track", ok)


def check_coverage_invariant() -> bool:
    box_layouts = {
        0: [],
        1: [LEFT_BOX],
        2: [LEFT_BOX, RIGHT_BOX],
        3: [LEFT_BOX, RIGHT_BOX, SHIFTED_RIGHT_BOX],
    }
    conf_layouts = {
        0: [],
        1: [0.7],
        2: [0.4, 0.8],
        3: [0.4, 0.8, 0.6],
    }
    # Cada variante ataca um ramo defensivo de `select`: ausência de arrays,
    # comprimento incompatível com `n_det` e caixas de área nula.
    variants = ("full", "no_conf", "no_boxes", "short_conf", "short_ids", "zero_area")
    ok = True
    for warm, n_det, with_ids, variant in itertools.product(
        (False, True), (0, 1, 2, 3), (False, True), variants
    ):
        selector = PersonSelector()
        if warm:
            selector.select(*_frame([LEFT_BOX, RIGHT_BOX], [0.6, 0.9], [1, 2]))
        boxes = box_layouts[n_det]
        confs = conf_layouts[n_det]
        ids = list(range(50, 50 + n_det)) if with_ids else None
        if variant == "zero_area":
            boxes = [ZERO_AREA_BOX] * n_det
        elif variant == "short_conf":
            confs = confs[:-1]
        elif variant == "short_ids" and ids is not None:
            ids = ids[:-1]
        box_conf = (
            None if variant == "no_conf" else np.array(confs, dtype=np.float32)
        )
        box_xyxy = (
            None
            if variant == "no_boxes"
            else np.array(boxes, dtype=np.float32).reshape(len(boxes), 4)
        )
        index = selector.select(n_det, box_conf, box_xyxy, ids)
        if n_det == 0:
            ok = ok and index is None
        else:
            ok = ok and index is not None and 0 <= index < n_det
    return _check("select devolve None se e somente se n_det == 0", ok)


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
        "bbox_iou: valores esperados, finitos mesmo com NaN, área nula ou negativa",
        ok,
    )


def run_selftest() -> None:
    checks = [
        check_initial_acquisition_by_confidence(),
        check_active_track_beats_higher_confidence(),
        check_iou_recovery_when_active_track_disappears(),
        check_gap_then_same_id_resumes(),
        check_gap_then_new_id_on_continuous_person(),
        check_missing_tracker_ids(),
        check_single_detection_with_new_id(),
        check_immediate_track_change_without_gap(),
        check_idless_frame_preserves_active_track(),
        check_all_zero_iou_falls_back_to_confidence(),
        check_exact_iou_tie_broken_by_confidence(),
        check_state_is_per_selector_instance(),
        check_coverage_invariant(),
        check_bbox_iou_numerics(),
    ]
    if not all(checks):
        print("\npose selection selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose selection selftest OK: todas as checagens passaram")
