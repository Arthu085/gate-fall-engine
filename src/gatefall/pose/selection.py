"""Regra de seleção de pessoa por quadro (não por track) para pose."""

import numpy as np


def select_person_index(n_det: int, box_conf: np.ndarray | None) -> int | None:
    # Seleção de pessoa por detecção do quadro, não por track: no Le2i (ator
    # único) o track_id serve só como diagnóstico e descartaria quadros
    # válidos sempre que o tracker perde e recupera a identidade.
    if n_det == 1:
        return 0
    if n_det > 1 and box_conf is not None:
        return int(np.argmax(box_conf))
    return None
