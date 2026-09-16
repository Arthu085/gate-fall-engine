"""Seleção contínua da pessoa-alvo, com estado por vídeo, na extração de pose.

A política é de continuidade primeiro: uma vez adquirida, a pessoa-alvo é
seguida pela track do ByteTrack e, quando o ID se perde, reancorada pela
maior sobreposição (IoU) com a última bbox selecionada; a confiança da caixa
só decide a aquisição inicial e os empates. `select` devolve `None` apenas
quando o quadro não tem detecção alguma, de modo que a cobertura de
`person_found` não muda por causa da seleção.

`PersonSelector` carrega o estado de um único vídeo e não tem `reset`:
instancie um seletor por vídeo, que é o que torna a garantia estrutural.
"""

import argparse

import numpy as np


def bbox_iou(previous: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """IoU de uma bbox xyxy contra um conjunto [N, 4] de bboxes xyxy.

    Coordenada não finita (NaN/inf) devolve IoU 0,0 para aquela candidata,
    assim como caixa de área nula ou negativa.
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


class PersonSelector:
    """Estado de seleção de um único vídeo. Instancie um por vídeo; não há reset."""

    def __init__(self) -> None:
        self._active_track_id: int | None = None
        self._last_bbox: np.ndarray | None = None

    @property
    def active_track_id(self) -> int | None:
        return self._active_track_id

    @property
    def last_bbox(self) -> np.ndarray | None:
        return self._last_bbox

    def select(
        self,
        n_det: int,
        box_conf: np.ndarray | None,
        box_xyxy: np.ndarray | None,
        track_ids: list[int] | None,
    ) -> int | None:
        if n_det == 0:
            return None

        ids = (
            track_ids
            if (track_ids is not None and len(track_ids) == n_det)
            else None
        )
        confs = (
            box_conf if (box_conf is not None and box_conf.shape[0] == n_det) else None
        )
        boxes = (
            box_xyxy if (box_xyxy is not None and box_xyxy.shape[0] == n_det) else None
        )

        # A track ativa vence mesmo contra uma detecção de confiança maior; e,
        # depois de uma perda, um ID diferente não é prova de outra pessoa, por
        # isso o quadro continua sendo aceito em vez de descartado.
        if self._active_track_id is not None and ids is not None:
            if self._active_track_id in ids:
                index = ids.index(self._active_track_id)
                if boxes is not None:
                    self._last_bbox = boxes[index].copy()
                return index

        index = self._reacquire(confs, boxes)

        # Um quadro sem IDs utilizáveis não é evidência de troca de pessoa: a
        # track ativa sobrevive a ele do mesmo jeito que a última bbox.
        if ids is not None:
            self._active_track_id = ids[index]
        if boxes is not None:
            self._last_bbox = boxes[index].copy()
        return index

    def _reacquire(
        self, confs: np.ndarray | None, boxes: np.ndarray | None
    ) -> int:
        if self._last_bbox is not None and boxes is not None:
            ious = bbox_iou(self._last_bbox, boxes)
            best = float(ious.max())
            if best > 0.0:
                candidates = np.flatnonzero(ious == best)
                if candidates.shape[0] == 1:
                    return int(candidates[0])
                if confs is not None:
                    return int(candidates[np.argmax(confs[candidates])])
                return int(candidates[0])

        return int(np.argmax(confs)) if confs is not None else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "selftest",
        help="Roda os casos sintéticos da política de seleção contínua",
    )

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.pose.selection_selftest import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
