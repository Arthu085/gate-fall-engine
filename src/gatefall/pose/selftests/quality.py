"""Selftest sintético do índice de qualidade de pose (`quality.py`).

Não toca no dataset real: todo quadro é sintético, para travar a semântica
dos quatro componentes (q_conf, q_valid, q_temporal, q_pose) e evitar
regressões silenciosas quando o cálculo evoluir.

Os quadros usados aqui têm keypoints em posições fixas e distintas (não
degeneradas), para que qualquer conjunto razoável de arestas do esqueleto
COCO17 produza arestas de comprimento não nulo — os testes de corrupção
temporal não assumem qual lista exata de arestas `quality.py` usa
internamente.
"""

import sys

import numpy as np

from gatefall.pose.quality import (
    CONF_CLIP_MAX,
    CONF_CLIP_MIN,
    MIN_VALID_EDGES,
    N_KEYPOINTS,
    QualityComponents,
    TEMPORAL_LENGTH_EPS,
    VALID_CONF_THRESHOLD,
    clip_confidence,
    compute_q_conf,
    compute_q_temporal,
    compute_q_valid,
    pose_quality_from_arrays,
)

SEED = 20260916  # data do desenho do selftest, só para fixar a sequência

DEFAULT_BBOX = (0.0, 0.0, 100.0, 200.0)

# Pose humana plausível (ombros, cotovelos, punhos, quadris, joelhos,
# tornozelos com coordenadas distintas), só para que toda aresta do
# esqueleto tenha comprimento não nulo.
BASE_XY = np.array(
    [
        (50.0, 10.0),  # nose
        (55.0, 8.0),  # left_eye
        (45.0, 8.0),  # right_eye
        (60.0, 10.0),  # left_ear
        (40.0, 10.0),  # right_ear
        (65.0, 40.0),  # left_shoulder
        (35.0, 40.0),  # right_shoulder
        (75.0, 70.0),  # left_elbow
        (25.0, 70.0),  # right_elbow
        (80.0, 100.0),  # left_wrist
        (20.0, 100.0),  # right_wrist
        (60.0, 110.0),  # left_hip
        (40.0, 110.0),  # right_hip
        (62.0, 150.0),  # left_knee
        (38.0, 150.0),  # right_knee
        (64.0, 190.0),  # left_ankle
        (36.0, 190.0),  # right_ankle
    ],
    dtype=np.float32,
)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _bbox_sequence(k: int, bbox: tuple[float, float, float, float] = DEFAULT_BBOX) -> np.ndarray:
    return np.tile(np.array(bbox, dtype=np.float32), (k, 1))


def _keypoints_frame(xy: np.ndarray, conf: np.ndarray | float) -> np.ndarray:
    frame = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    frame[:, :2] = xy
    frame[:, 2] = conf
    return frame


def _static_sequence(k: int, *, conf: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequência de `k` quadros com a mesma pose e a mesma confiança (caso perfeito)."""
    frame = _keypoints_frame(BASE_XY, conf)
    keypoints = np.tile(frame, (k, 1, 1))
    bbox = _bbox_sequence(k)
    person_found = np.ones(k, dtype=bool)
    return keypoints, bbox, person_found


def _radial_scale(xy: np.ndarray, bbox: np.ndarray, scale: float) -> np.ndarray:
    """Escala a pose radialmente a partir do centro da bbox.

    Uma transformação de similaridade: multiplica TODA distância par a par
    (logo, toda aresta do esqueleto, seja qual for a lista usada) pelo mesmo
    fator `scale`, o que torna a mudança relativa esperada previsível sem
    depender de qual conjunto de arestas `quality.py` usa internamente:
    para escala s >= 1, mudança relativa = (s - 1) / s; para s < 1,
    mudança relativa = 1 - s.
    """
    center = np.array(
        [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32
    )
    return center + scale * (xy - center)


def _expected_relative_change(scale: float) -> float:
    if scale >= 1.0:
        return (scale - 1.0) / scale
    return 1.0 - scale


# --------------------------------------------------------------------------
# Funções unitárias
# --------------------------------------------------------------------------


def check_clip_confidence_bounds() -> bool:
    raw = np.array([-5.0, -0.001, 0.0, 0.3, 0.5, 0.9999, 1.0, 1.0001, 8.0], dtype=np.float32)
    clipped = clip_confidence(raw)
    expected = np.clip(raw, CONF_CLIP_MIN, CONF_CLIP_MAX)
    ok = (
        bool(np.allclose(clipped, expected))
        and bool(np.all(clipped >= CONF_CLIP_MIN))
        and bool(np.all(clipped <= CONF_CLIP_MAX))
    )
    return _check("clip_confidence: satura em [CONF_CLIP_MIN, CONF_CLIP_MAX]", ok)


def check_compute_q_conf_is_mean() -> bool:
    all_ones = np.ones((1, N_KEYPOINTS), dtype=np.float32)
    mixed = np.zeros((1, N_KEYPOINTS), dtype=np.float32)
    mixed[0, :9] = 1.0  # 9 de 17 em 1.0, resto em 0.0
    q_ones = compute_q_conf(all_ones)
    q_mixed = compute_q_conf(mixed)
    ok = (
        bool(np.allclose(q_ones, 1.0))
        and bool(np.allclose(q_mixed, 9.0 / N_KEYPOINTS))
    )
    return _check("compute_q_conf: média das 17 confianças clipadas", ok)


def check_compute_q_valid_threshold_is_inclusive() -> bool:
    at_threshold = np.full((1, N_KEYPOINTS), VALID_CONF_THRESHOLD, dtype=np.float32)
    just_below = np.full((1, N_KEYPOINTS), VALID_CONF_THRESHOLD - 1e-4, dtype=np.float32)
    half = np.zeros((1, N_KEYPOINTS), dtype=np.float32)
    half[0, :5] = 1.0
    q_at = compute_q_valid(at_threshold)
    q_below = compute_q_valid(just_below)
    q_half = compute_q_valid(half)
    ok = (
        bool(np.allclose(q_at, 1.0))
        and bool(np.allclose(q_below, 0.0))
        and bool(np.allclose(q_half, 5.0 / N_KEYPOINTS))
    )
    return _check(
        "compute_q_valid: fração com confiança clipada >= 0.5 (limiar inclusivo)", ok
    )


def check_compute_q_temporal_direct_call() -> bool:
    scale = 1.5
    xy_prev = BASE_XY
    xy_curr = _radial_scale(BASE_XY, np.array(DEFAULT_BBOX, dtype=np.float32), scale)
    xy = np.stack([xy_prev, xy_curr], axis=0)
    conf_clipped = np.ones((2, N_KEYPOINTS), dtype=np.float32)
    person_found = np.array([True, True])

    q_temporal = compute_q_temporal(xy, conf_clipped, person_found)
    expected_first = 1.0
    expected_second = 1.0 - _expected_relative_change(scale)
    ok = (
        q_temporal.shape == (2,)
        and bool(np.isclose(q_temporal[0], expected_first))
        and bool(np.isclose(q_temporal[1], expected_second, atol=1e-4))
    )
    return _check(
        "compute_q_temporal: escala radial uniforme dá 1 - (s-1)/s na chamada direta",
        ok,
    )


# --------------------------------------------------------------------------
# pose_quality_from_arrays: casos de ponta a ponta
# --------------------------------------------------------------------------


def check_bounds_finiteness_and_multiplicative_identity() -> bool:
    rng = np.random.default_rng(SEED)
    k = 14
    keypoints = np.zeros((k, N_KEYPOINTS, 3), dtype=np.float32)
    person_found = rng.random(k) > 0.3
    for i in range(k):
        jitter = rng.normal(scale=5.0, size=(N_KEYPOINTS, 2)).astype(np.float32)
        keypoints[i, :, :2] = BASE_XY + jitter
        keypoints[i, :, 2] = rng.uniform(-0.3, 1.3, size=N_KEYPOINTS).astype(np.float32)
    bbox = _bbox_sequence(k)

    components = pose_quality_from_arrays(keypoints, bbox, person_found)

    shapes_ok = (
        components.q_conf.shape == (k,)
        and components.q_valid.shape == (k,)
        and components.q_temporal.shape == (k,)
        and components.q_pose.shape == (k,)
    )
    dtypes_ok = (
        components.q_conf.dtype == np.float32
        and components.q_valid.dtype == np.float32
        and components.q_temporal.dtype == np.float32
        and components.q_pose.dtype == np.float32
    )
    bounded_ok = all(
        bool(np.all(arr >= 0.0)) and bool(np.all(arr <= 1.0)) and bool(np.isfinite(arr).all())
        for arr in (
            components.q_conf,
            components.q_valid,
            components.q_temporal,
            components.q_pose,
        )
    )
    product_ok = bool(
        np.allclose(
            components.q_pose,
            components.q_conf * components.q_valid * components.q_temporal,
            atol=1e-5,
        )
    )
    ok = isinstance(components, QualityComponents) and shapes_ok and dtypes_ok and bounded_ok and product_ok
    return _check(
        "pose_quality_from_arrays: formato [K] float32, limitado a [0,1], finito, "
        "e q_pose == q_conf * q_valid * q_temporal",
        ok,
    )


def check_perfect_static_pose_gives_quality_one() -> bool:
    keypoints, bbox, person_found = _static_sequence(5, conf=1.0)
    components = pose_quality_from_arrays(keypoints, bbox, person_found)
    ok = (
        bool(np.allclose(components.q_conf, 1.0))
        and bool(np.allclose(components.q_valid, 1.0))
        and bool(np.allclose(components.q_temporal, 1.0))
        and bool(np.allclose(components.q_pose, 1.0))
    )
    return _check(
        "pose estática perfeita (confiança 1.0, coordenadas idênticas) dá q_pose == 1.0",
        ok,
    )


def check_uniform_confidence_reduction_lowers_q_conf_and_pose() -> bool:
    baseline_kp, bbox, person_found = _static_sequence(3, conf=1.0)
    reduced_kp, _, _ = _static_sequence(3, conf=0.6)

    baseline = pose_quality_from_arrays(baseline_kp, bbox, person_found)
    reduced = pose_quality_from_arrays(reduced_kp, bbox, person_found)

    ok = (
        bool(np.allclose(baseline.q_conf, 1.0))
        and bool(np.allclose(reduced.q_conf, 0.6, atol=1e-4))
        and bool(np.all(reduced.q_conf < baseline.q_conf))
        and bool(np.all(reduced.q_pose < baseline.q_pose))
        # confiança 0.6 continua acima do limiar: q_valid não deveria mudar
        and bool(np.allclose(reduced.q_valid, baseline.q_valid))
    )
    return _check(
        "redução uniforme de confiança (acima do limiar) reduz q_conf e q_pose "
        "sem mudar q_valid",
        ok,
    )


def check_keypoint_dropout_lowers_q_valid() -> bool:
    baseline_kp, bbox, person_found = _static_sequence(1, conf=1.0)
    dropout_kp = baseline_kp.copy()
    dropped = [1, 2, 3, 4, 7, 8, 9, 10]  # 8 de 17 keypoints derrubados
    dropout_kp[0, dropped, 2] = 0.1

    baseline = pose_quality_from_arrays(baseline_kp, bbox, person_found)
    dropout = pose_quality_from_arrays(dropout_kp, bbox, person_found)

    expected_valid = (N_KEYPOINTS - len(dropped)) / N_KEYPOINTS
    ok = (
        bool(np.allclose(baseline.q_valid, 1.0))
        and bool(np.isclose(dropout.q_valid[0], expected_valid))
        and bool(dropout.q_valid[0] < baseline.q_valid[0])
        and bool(dropout.q_pose[0] < baseline.q_pose[0])
    )
    return _check("dropout de keypoints reduz q_valid e, com isso, q_pose", ok)


def check_temporal_structural_corruption_lowers_q_temporal() -> bool:
    scale = 2.0
    bbox_arr = np.array(DEFAULT_BBOX, dtype=np.float32)
    frame0 = _keypoints_frame(BASE_XY, 1.0)
    frame1_same = _keypoints_frame(BASE_XY, 1.0)
    frame1_corrupt = _keypoints_frame(_radial_scale(BASE_XY, bbox_arr, scale), 1.0)

    keypoints_same = np.stack([frame0, frame1_same], axis=0)
    keypoints_corrupt = np.stack([frame0, frame1_corrupt], axis=0)
    bbox = _bbox_sequence(2)
    person_found = np.array([True, True])

    same = pose_quality_from_arrays(keypoints_same, bbox, person_found)
    corrupt = pose_quality_from_arrays(keypoints_corrupt, bbox, person_found)

    expected_corrupt = 1.0 - _expected_relative_change(scale)
    ok = (
        bool(np.allclose(same.q_temporal[1], 1.0))
        and bool(np.isclose(corrupt.q_temporal[1], expected_corrupt, atol=1e-4))
        and bool(corrupt.q_temporal[1] < same.q_temporal[1])
        and bool(corrupt.q_pose[1] < same.q_pose[1])
    )
    return _check(
        "corrupção estrutural entre quadros consecutivos reduz q_temporal e q_pose",
        ok,
    )


def check_first_observed_frame_is_neutral() -> bool:
    keypoints, bbox, person_found = _static_sequence(1, conf=1.0)
    components = pose_quality_from_arrays(keypoints, bbox, person_found)
    ok = bool(np.isclose(components.q_temporal[0], 1.0))
    return _check("primeiro quadro observado (sem predecessor) tem q_temporal neutro 1.0", ok)


def check_leading_absence_is_neutral() -> bool:
    frame_absent = _keypoints_frame(np.zeros_like(BASE_XY), 0.0)
    frame_observed = _keypoints_frame(BASE_XY, 1.0)
    keypoints = np.stack([frame_absent, frame_absent, frame_observed], axis=0)
    bbox = _bbox_sequence(3)
    person_found = np.array([False, False, True])

    components = pose_quality_from_arrays(keypoints, bbox, person_found)
    ok = bool(np.isclose(components.q_temporal[2], 1.0))
    return _check(
        "quadro cujo predecessor é ausência (sem observação anterior alguma) "
        "tem q_temporal neutro 1.0",
        ok,
    )


def check_forward_fillable_gap_does_not_fake_perfect_match() -> bool:
    scale = 2.0
    bbox_arr = np.array(DEFAULT_BBOX, dtype=np.float32)
    frame_a = _keypoints_frame(BASE_XY, 1.0)
    frame_gap = _keypoints_frame(np.zeros_like(BASE_XY), 0.0)  # ausente, valores irrelevantes
    frame_b = _keypoints_frame(_radial_scale(BASE_XY, bbox_arr, scale), 1.0)

    keypoints = np.stack([frame_a, frame_gap, frame_b], axis=0)
    bbox = _bbox_sequence(3)
    person_found = np.array([True, False, True])

    components = pose_quality_from_arrays(keypoints, bbox, person_found)
    expected = 1.0 - _expected_relative_change(scale)
    ok = (
        bool(np.isclose(components.q_temporal[2], expected, atol=1e-4))
        and not bool(np.isclose(components.q_temporal[2], 1.0, atol=1e-3))
    )
    return _check(
        "gap forward-fillable não mascara a mudança real: o quadro seguinte é "
        "comparado com o último quadro OBSERVADO, não com um preenchimento",
        ok,
    )


def check_absent_frames_are_exactly_zero() -> bool:
    frame_observed = _keypoints_frame(BASE_XY, 1.0)
    frame_absent = _keypoints_frame(np.full_like(BASE_XY, 999.0), 0.9)  # valores "sujos" propositais
    keypoints = np.stack(
        [frame_observed, frame_absent, frame_observed, frame_absent], axis=0
    )
    bbox = _bbox_sequence(4)
    person_found = np.array([True, False, True, False])

    components = pose_quality_from_arrays(keypoints, bbox, person_found)
    absent_idx = np.array([1, 3])
    ok = (
        bool(np.allclose(components.q_pose[absent_idx], 0.0))
        and bool(np.allclose(components.q_conf[absent_idx], 0.0))
        and bool(np.allclose(components.q_valid[absent_idx], 0.0))
        and bool(np.allclose(components.q_temporal[absent_idx], 0.0))
    )
    return _check(
        "quadros com person_found == False têm os quatro componentes exatamente 0.0",
        ok,
    )


def check_insufficient_valid_edges_falls_back_to_neutral() -> bool:
    scale = 3.0  # corrupção grande, mas com poucas arestas utilizáveis
    bbox_arr = np.array(DEFAULT_BBOX, dtype=np.float32)
    conf0 = np.full(N_KEYPOINTS, 0.1, dtype=np.float32)
    conf0[[5, 6]] = 1.0  # só ombros válidos: no máximo 1 aresta (5,6)

    frame0 = _keypoints_frame(BASE_XY, conf0)
    frame1 = _keypoints_frame(_radial_scale(BASE_XY, bbox_arr, scale), conf0)
    keypoints = np.stack([frame0, frame1], axis=0)
    bbox = _bbox_sequence(2)
    person_found = np.array([True, True])

    components = pose_quality_from_arrays(keypoints, bbox, person_found)
    ok = bool(np.isclose(components.q_temporal[1], 1.0))
    return _check(
        f"menos que MIN_VALID_EDGES ({MIN_VALID_EDGES}) arestas utilizáveis cai no "
        "neutro 1.0, mesmo com corrupção grande",
        ok,
    )


def run_selftest() -> None:
    checks = [
        check_clip_confidence_bounds(),
        check_compute_q_conf_is_mean(),
        check_compute_q_valid_threshold_is_inclusive(),
        check_compute_q_temporal_direct_call(),
        check_bounds_finiteness_and_multiplicative_identity(),
        check_perfect_static_pose_gives_quality_one(),
        check_uniform_confidence_reduction_lowers_q_conf_and_pose(),
        check_keypoint_dropout_lowers_q_valid(),
        check_temporal_structural_corruption_lowers_q_temporal(),
        check_first_observed_frame_is_neutral(),
        check_leading_absence_is_neutral(),
        check_forward_fillable_gap_does_not_fake_perfect_match(),
        check_absent_frames_are_exactly_zero(),
        check_insufficient_valid_edges_falls_back_to_neutral(),
    ]
    if not all(checks):
        print("\npose quality selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose quality selftest OK: todas as checagens passaram")


# --------------------------------------------------------------------------
# Varredura determinística de severidade (run_degradation)
# --------------------------------------------------------------------------


def _assert_non_increasing(name: str, values: list[float], *, atol: float = 1e-4) -> bool:
    diffs = np.diff(np.array(values, dtype=np.float64))
    ok = bool(np.all(diffs <= atol))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {['%.4f' % v for v in values]}")
    return ok


def run_confidence_attenuation_channel() -> bool:
    # Fica sempre >= VALID_CONF_THRESHOLD: isola o efeito em q_conf, sem
    # derrubar keypoints do lado de q_valid nem mexer em q_temporal (pose
    # estática, quadro único sem predecessor => neutro).
    severities = [1.0, 0.85, 0.70, 0.55]
    q_conf_by_level = []
    q_pose_by_level = []
    for conf in severities:
        keypoints, bbox, person_found = _static_sequence(1, conf=conf)
        components = pose_quality_from_arrays(keypoints, bbox, person_found)
        q_conf_by_level.append(float(components.q_conf[0]))
        q_pose_by_level.append(float(components.q_pose[0]))

    ok_conf = _assert_non_increasing(
        "atenuação de confiança: q_conf não cresce", q_conf_by_level
    )
    ok_pose = _assert_non_increasing(
        "atenuação de confiança: q_pose não cresce", q_pose_by_level
    )
    return ok_conf and ok_pose


def run_keypoint_dropout_channel() -> bool:
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(N_KEYPOINTS)
    prefix_lengths = [0, 4, 8, 12, 16]  # cada nível é superconjunto estrito do anterior

    q_valid_by_level = []
    q_pose_by_level = []
    for prefix_len in prefix_lengths:
        keypoints, bbox, person_found = _static_sequence(1, conf=1.0)
        dropped = permutation[:prefix_len]
        keypoints[0, dropped, 2] = 0.05
        components = pose_quality_from_arrays(keypoints, bbox, person_found)
        q_valid_by_level.append(float(components.q_valid[0]))
        q_pose_by_level.append(float(components.q_pose[0]))

    ok_valid = _assert_non_increasing(
        "dropout de keypoints (superconjunto crescente): q_valid não cresce",
        q_valid_by_level,
    )
    ok_pose = _assert_non_increasing(
        "dropout de keypoints (superconjunto crescente): q_pose não cresce",
        q_pose_by_level,
    )
    return ok_valid and ok_pose


def run_temporal_corruption_channel() -> bool:
    rng = np.random.default_rng(SEED)
    # Direção fixa (expandir ou encolher a pose): sorteada uma única vez e
    # aplicada com magnitude crescente, o que preserva a monotonicidade
    # analítica de 1 - relative_change(scale) discutida em `_radial_scale`.
    direction = 1.0 if rng.random() < 0.5 else -1.0
    fractions = [0.0, 0.10, 0.25, 0.45]  # fração crescente da diagonal da bbox
    bbox_arr = np.array(DEFAULT_BBOX, dtype=np.float32)
    diagonal = float(np.hypot(bbox_arr[2] - bbox_arr[0], bbox_arr[3] - bbox_arr[1]))

    q_temporal_by_level = []
    q_pose_by_level = []
    for fraction in fractions:
        # A fração ancora a magnitude do deslocamento na diagonal da bbox;
        # convertida aqui num fator de escala radial equivalente.
        scale = 1.0 + direction * fraction
        frame0 = _keypoints_frame(BASE_XY, 1.0)
        frame1 = _keypoints_frame(_radial_scale(BASE_XY, bbox_arr, scale), 1.0)
        keypoints = np.stack([frame0, frame1], axis=0)
        bbox = _bbox_sequence(2)
        person_found = np.array([True, True])

        components = pose_quality_from_arrays(keypoints, bbox, person_found)
        q_temporal_by_level.append(float(components.q_temporal[1]))
        q_pose_by_level.append(float(components.q_pose[1]))

    assert diagonal > 0.0  # só documenta a grandeza usada para derivar `fractions`
    ok_temporal = _assert_non_increasing(
        "corrupção temporal (deslocamento crescente): q_temporal não cresce",
        q_temporal_by_level,
    )
    ok_pose = _assert_non_increasing(
        "corrupção temporal (deslocamento crescente): q_pose não cresce",
        q_pose_by_level,
    )
    return ok_temporal and ok_pose


def run_complete_absence_channel() -> bool:
    k = 6
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(k)
    absent_counts = [0, 2, 4, 6]  # superconjunto estrito de quadros ausentes

    mean_pose_by_level = []
    per_level_absent_zero_ok = True
    for count in absent_counts:
        keypoints, bbox, person_found = _static_sequence(k, conf=1.0)
        absent_idx = permutation[:count]
        person_found[absent_idx] = False
        components = pose_quality_from_arrays(keypoints, bbox, person_found)
        mean_pose_by_level.append(float(np.mean(components.q_pose)))
        if count > 0:
            per_level_absent_zero_ok = per_level_absent_zero_ok and bool(
                np.allclose(components.q_pose[absent_idx], 0.0)
            )

    ok_mean = _assert_non_increasing(
        "ausência completa (mais quadros sem pessoa): média de q_pose não cresce",
        mean_pose_by_level,
    )
    ok_zero = _check(
        "ausência completa: todo quadro marcado ausente tem q_pose == 0.0", per_level_absent_zero_ok
    )
    return ok_mean and ok_zero


def run_degradation() -> None:
    """Varredura determinística de severidade sobre quatro canais de degradação.

    Cada canal é isolado dos demais (só um tipo de corrupção por vez) para
    que a queda monotônica do componente-alvo — e de q_pose — não dependa de
    efeitos colaterais entre canais.
    """
    results = [
        run_confidence_attenuation_channel(),
        run_keypoint_dropout_channel(),
        run_temporal_corruption_channel(),
        run_complete_absence_channel(),
    ]
    if not all(results):
        print("\npose quality degradation sweep FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npose quality degradation sweep OK: todos os canais não-crescentes")


if __name__ == "__main__":
    run_selftest()
    run_degradation()
