"""Descritor V_t em R^10 derivado apenas da máscara binária selecionada.

Nenhum canal depende de pose ou de qualquer outro backbone: tudo vem dos
momentos de imagem da própria máscara (contrato do plano SAM 3, invariante 3
de `CLAUDE.md` — pipeline monocular RGB, sem descritor derivado de
profundidade).
"""

import numpy as np

CHANNEL_NAMES: tuple[str, ...] = (
    "present",
    "mask_area_norm",
    "centroid_x_norm",
    "centroid_y_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "fill_ratio",
    "eccentricity",
    "sin_2theta",
    "cos_2theta",
)
V_T_DIM = len(CHANNEL_NAMES)

ZERO_DESCRIPTOR: np.ndarray = np.zeros(V_T_DIM, dtype=np.float32)
ZERO_DESCRIPTOR.setflags(write=False)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bbox xyxy inclusiva, em espaço de pixel, da extensão da máscara.

    Devolve `None` quando a máscara não tem nenhum pixel de primeiro plano.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y_indices = np.flatnonzero(rows)
    x_indices = np.flatnonzero(cols)
    y_min, y_max = int(y_indices[0]), int(y_indices[-1])
    x_min, x_max = int(x_indices[0]), int(x_indices[-1])
    return x_min, y_min, x_max, y_max


def compute_descriptor(
    mask: np.ndarray | None, *, frame_width: int, frame_height: int
) -> np.ndarray:
    if mask is None:
        return ZERO_DESCRIPTOR.copy()

    bbox = bbox_from_mask(mask)
    if bbox is None:
        return ZERO_DESCRIPTOR.copy()
    x_min, y_min, x_max, y_max = bbox

    y_coords, x_coords = np.nonzero(mask)
    m00 = float(x_coords.shape[0])

    xbar = float(x_coords.mean())
    ybar = float(y_coords.mean())

    dx = x_coords.astype(np.float64) - xbar
    dy = y_coords.astype(np.float64) - ybar
    mu20 = float(np.mean(dx * dx))
    mu02 = float(np.mean(dy * dy))
    mu11 = float(np.mean(dx * dy))

    # theta = 0.5*atan2(2*mu11, mu20-mu02) é o ângulo do eixo principal da
    # máscara; codificar como (sin_2theta, cos_2theta) em vez de theta remove
    # a ambiguidade pi-periódica de um eixo sem sentido de seta (theta e
    # theta+pi descrevem o mesmo eixo).
    r = float(np.sqrt((mu20 - mu02) ** 2 + (2.0 * mu11) ** 2))
    if r == 0.0:
        sin_2theta = 0.0
        cos_2theta = 0.0
    else:
        sin_2theta = (2.0 * mu11) / r
        cos_2theta = (mu20 - mu02) / r

    lambda1 = ((mu20 + mu02) + r) / 2.0
    lambda2 = ((mu20 + mu02) - r) / 2.0
    if lambda1 > 0.0:
        # Convenção do skimage.measure.regionprops: excentricidade da elipse
        # de mesmos momentos de segunda ordem que a máscara.
        radicand = max(0.0, 1.0 - lambda2 / lambda1)
        eccentricity = float(np.sqrt(radicand))
    else:
        eccentricity = 0.0

    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1

    descriptor = np.array(
        [
            1.0,
            m00 / (frame_width * frame_height),
            xbar / frame_width,
            ybar / frame_height,
            bbox_w / frame_width,
            bbox_h / frame_height,
            m00 / (bbox_w * bbox_h),
            eccentricity,
            sin_2theta,
            cos_2theta,
        ],
        dtype=np.float32,
    )
    return descriptor
