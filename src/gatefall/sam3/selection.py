"""Seleção contínua da instância-alvo por quadro, sem IDs de track.

SAM 3 não devolve identidade persistente entre quadros — tracking de vídeo é
proibido pelo escopo desta fundação (ver plano), então a política de
continuidade se apoia só na última bbox selecionada, nunca em um índice de
track. `select` devolve `None` apenas quando o quadro não tem instância
alguma; um quadro sem detecção não apaga `_last_bbox`, para que uma lacuna
ainda deixe a continuidade retomar num quadro futuro. Aquisição e empates de
IoU são resolvidos pelo score do SAM, nunca por pose — este módulo não
importa `gatefall.pose`.
"""

import argparse

import numpy as np

from gatefall.sam3.descriptors import bbox_from_mask
from gatefall.sam3.runtime import Sam3Instance


def bbox_iou(previous: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """IoU de uma bbox xyxy contra um conjunto [N, 4] de bboxes xyxy.

    Cópia privada de `pose/selection.py::bbox_iou` (decisão do plano: SAM 3
    não deve importar `gatefall.pose`, nem o inverso). Coordenada não finita
    (NaN/inf) devolve IoU 0,0 para aquela candidata, assim como caixa de área
    nula ou negativa.
    """
    inter_x1 = np.maximum(previous[0], candidates[:, 0])
    inter_y1 = np.maximum(previous[1], candidates[:, 1])
    inter_x2 = np.minimum(previous[2], candidates[:, 2])
    inter_y2 = np.minimum(previous[3], candidates[:, 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    intersection = inter_w * inter_h

    previous_area = max(float(previous[2] - previous[0]), 0.0) * max(
        float(previous[3] - previous[1]), 0.0
    )
    candidate_areas = np.clip(
        candidates[:, 2] - candidates[:, 0], 0.0, None
    ) * np.clip(candidates[:, 3] - candidates[:, 1], 0.0, None)

    union = previous_area + candidate_areas - intersection
    return np.where(union > 0.0, intersection / np.where(union > 0.0, union, 1.0), 0.0)


def _to_half_open_xyxy(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    return np.array([x_min, y_min, x_max + 1, y_max + 1], dtype=np.float32)


class InstanceSelector:
    """Estado de seleção de um único vídeo. Instancie um por vídeo; não há reset."""

    def __init__(self) -> None:
        self._last_bbox: np.ndarray | None = None

    @property
    def last_bbox(self) -> np.ndarray | None:
        return self._last_bbox

    def select(self, instances: list[Sam3Instance]) -> int | None:
        if not instances:
            return None

        valid_indices: list[int] = []
        valid_boxes: list[np.ndarray] = []
        for index, instance in enumerate(instances):
            bbox = bbox_from_mask(instance.mask)
            if bbox is not None:
                valid_indices.append(index)
                valid_boxes.append(_to_half_open_xyxy(bbox))

        if not valid_indices:
            # Nenhuma instância tem um pixel de primeiro plano sequer: não há
            # geometria para reancorar ou atualizar `_last_bbox`, então cai no
            # score bruto sobre todas as instâncias como último recurso.
            scores = np.array([instance.score for instance in instances], dtype=np.float32)
            return int(np.argmax(scores))

        candidate_boxes = np.stack(valid_boxes, axis=0)
        candidate_scores = np.array(
            [instances[i].score for i in valid_indices], dtype=np.float32
        )

        if self._last_bbox is not None:
            ious = bbox_iou(self._last_bbox, candidate_boxes)
            best = float(ious.max())
            if best > 0.0:
                tied = np.flatnonzero(ious == best)
                chosen = (
                    int(tied[0])
                    if tied.shape[0] == 1
                    else int(tied[np.argmax(candidate_scores[tied])])
                )
                self._last_bbox = candidate_boxes[chosen].copy()
                return valid_indices[chosen]

        chosen = int(np.argmax(candidate_scores))
        self._last_bbox = candidate_boxes[chosen].copy()
        return valid_indices[chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "selftest",
        help="Roda os casos sintéticos da política de seleção contínua do SAM 3",
    )

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.sam3.selection_selftest import run_sam3_selection_selftest

        run_sam3_selection_selftest()


if __name__ == "__main__":
    main()
