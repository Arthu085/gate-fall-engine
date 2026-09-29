"""Selftest sintético da proxy de qualidade `q_sam3` da fonte SAM 3.

Não acessa vídeos, checkpoint, runtime isolado nem GPU: um segmentador falso,
cujo score cai com o contraste e cuja máscara encolhe com o borrão, substitui
o SAM 3 real para travar fórmula, causalidade, degradações, seleção contínua
e agregações da validação.
"""

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.sam3.quality import (
    DEGRADATION_SWEEPS,
    Sam3QualityValidationError,
    apply_frame_degradation,
    clean_quality_rows,
    compute_sam3_quality,
    mask_iou,
    observe_clean_then_degraded,
    summarize_degradation,
    sweep_rows,
)
from gatefall.sam3.runtime import Sam3Instance
from gatefall.sam3.storage import sam3_path, write_sam3_atomic

SEED = 20260925
HEIGHT = 48
WIDTH = 64


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _raises(function: Callable[..., object], *args: object, **kwargs: object) -> bool:
    try:
        function(*args, **kwargs)
    except Sam3QualityValidationError:
        return True
    return False


def _synthetic_frame() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    frame = rng.integers(40, 216, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[12:36, 20:44] = 230
    return frame


def _rect_mask(x_min: int, y_min: int, x_max: int, y_max: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y_min : y_max + 1, x_min : x_max + 1] = True
    return mask


class _ContrastSegmenter:
    """Score = desvio da luminância normalizado; máscara cobre pixels claros.

    Um distrator de score baixo no canto testa que a seleção segue a
    continuidade do quadro limpo.
    """

    def __init__(self) -> None:
        self.calls = 0

    def segment_frame(self, frame_rgb: np.ndarray, text_prompt: str) -> list[Sam3Instance]:
        self.calls += 1
        luminance = frame_rgb.astype(np.float64).mean(axis=2) / 255.0
        score = float(np.clip(luminance.std() / 0.25, 0.0, 1.0))
        if score < 0.05:
            return []
        mask = luminance > luminance.mean() + 0.5 * luminance.std()
        distractor = _rect_mask(0, 0, 3, 3)
        return [Sam3Instance(mask=mask, score=score), Sam3Instance(mask=distractor, score=0.3)]


def check_quality_contract_and_formula() -> bool:
    scores = np.array([0.0, 0.5, 0.73, 1.0, 0.91, 1.0 + 1e-7], dtype=np.float32)
    present = np.array([0, 1, 1, 1, 0, 1], dtype=np.float32)
    quality = compute_sam3_quality(scores, present)
    expected = np.array([0.0, 0.5, 0.73, 1.0, 0.0, 1.0], dtype=np.float32)
    rejects = all(
        (
            _raises(compute_sam3_quality, np.array([np.nan]), np.array([1.0])),
            _raises(compute_sam3_quality, np.array([1.2]), np.array([1.0])),
            _raises(compute_sam3_quality, np.array([-0.1]), np.array([0.0])),
            _raises(compute_sam3_quality, np.array([0.7]), np.array([0.5])),
            _raises(compute_sam3_quality, np.array([0.7, 0.8]), np.array([1.0])),
            _raises(compute_sam3_quality, np.zeros((2, 2)), np.zeros((2, 2))),
        )
    )
    ok = bool(
        quality.dtype == np.float32
        and quality.shape == scores.shape
        and bool(np.allclose(quality, expected))
        and bool(np.all((quality >= 0.0) & (quality <= 1.0)))
        and rejects
    )
    return _check(
        "fórmula: q_sam3 = clip(sam_score) * present, float32 em [0, 1], "
        "rejeita não finito, fora de faixa, present não binário e shape inválido",
        ok,
    )


def check_quality_is_frame_causal_and_deterministic() -> bool:
    rng = np.random.default_rng(SEED)
    scores = rng.uniform(0.5, 1.0, size=32).astype(np.float32)
    present = (rng.uniform(size=32) > 0.2).astype(np.float32)
    full = compute_sam3_quality(scores, present)
    prefix_ok = all(
        np.array_equal(compute_sam3_quality(scores[:end], present[:end]), full[:end])
        for end in range(1, 33)
    )
    permutation = rng.permutation(32)
    permuted = compute_sam3_quality(scores[permutation], present[permutation])
    future_changed = scores.copy()
    future_changed[20:] = 0.5
    ok = bool(
        prefix_ok
        and np.array_equal(permuted, full[permutation])
        and np.array_equal(compute_sam3_quality(scores, present), full)
        and np.array_equal(compute_sam3_quality(future_changed, present)[:20], full[:20])
    )
    return _check(
        "causalidade: cada quadro depende só do próprio score/present; "
        "prefixo, permutação e repetição preservam o resultado",
        ok,
    )


def _naive_box_blur(frame: np.ndarray, radius: int) -> np.ndarray:
    values = frame.astype(np.float64) / 255.0
    result = np.empty_like(values)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            window = values[
                max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1
            ]
            result[y, x] = window.reshape(-1, 3).mean(axis=0)
    return np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)


def _high_frequency(frame: np.ndarray) -> float:
    luminance = frame.astype(np.float64).mean(axis=2)
    return float(np.abs(np.diff(luminance, axis=1)).mean())


def check_degradations_are_deterministic_bounded_and_monotonic() -> bool:
    frame = _synthetic_frame()
    identity_ok = all(
        np.array_equal(apply_frame_degradation(frame, name, severities[0]), frame)
        for name, severities in DEGRADATION_SWEEPS.items()
    )
    # Imagem integral e média ingênua podem arredondar para lados opostos em x.5.
    blur_matches_naive = (
        int(
            np.abs(
                apply_frame_degradation(frame, "blur", 2).astype(np.int16)
                - _naive_box_blur(frame, 2).astype(np.int16)
            ).max()
        )
        <= 1
    )
    sweeps = {
        name: [apply_frame_degradation(frame, name, severity) for severity in severities]
        for name, severities in DEGRADATION_SWEEPS.items()
    }
    shapes_ok = all(
        degraded.shape == frame.shape and degraded.dtype == np.uint8
        for series in sweeps.values()
        for degraded in series
    )
    deterministic = all(
        np.array_equal(apply_frame_degradation(frame, name, severity), degraded)
        for name, severities in DEGRADATION_SWEEPS.items()
        for severity, degraded in zip(severities, sweeps[name])
    )
    blur_hf = [_high_frequency(degraded) for degraded in sweeps["blur"]]
    under_mean = [float(degraded.mean()) for degraded in sweeps["underexposure"]]
    over_mean = [float(degraded.mean()) for degraded in sweeps["overexposure"]]
    contrast_std = [float(degraded.astype(np.float64).std()) for degraded in sweeps["contrast"]]
    monotonic = (
        all(right < left for left, right in zip(blur_hf, blur_hf[1:]))
        and all(right < left for left, right in zip(under_mean, under_mean[1:]))
        and all(right > left for left, right in zip(over_mean, over_mean[1:]))
        and all(right < left for left, right in zip(contrast_std, contrast_std[1:]))
    )
    rejects = all(
        (
            _raises(apply_frame_degradation, frame, "blur", 1.5),
            _raises(apply_frame_degradation, frame, "blur", -1),
            _raises(apply_frame_degradation, frame, "contrast", 1.5),
            _raises(apply_frame_degradation, frame, "noise", 0.5),
            _raises(apply_frame_degradation, frame.astype(np.float32), "blur", 2),
            _raises(apply_frame_degradation, frame[:, :, 0], "blur", 2),
        )
    )
    ok = bool(
        identity_ok
        and blur_matches_naive
        and shapes_ok
        and deterministic
        and monotonic
        and rejects
    )
    return _check(
        "degradações: identidade na severidade 0, box blur igual à referência "
        "ingênua, uint8 [H, W, 3], determinísticas, monotônicas e com rejeições",
        ok,
    )


def check_mask_iou_edge_cases() -> bool:
    left = _rect_mask(0, 0, 9, 9)
    right = _rect_mask(5, 0, 14, 9)
    empty = np.zeros((HEIGHT, WIDTH), dtype=bool)
    ok = bool(
        mask_iou(None, None) == 1.0
        and mask_iou(empty, empty) == 1.0
        and mask_iou(left, None) == 0.0
        and mask_iou(None, left) == 0.0
        and mask_iou(left, left) == 1.0
        and np.isclose(mask_iou(left, right), 50.0 / 150.0)
        and _raises(mask_iou, left, np.zeros((2, 2), dtype=bool))
    )
    return _check(
        "mask_iou: ausências, máscaras vazias, identidade, sobreposição parcial "
        "e rejeição de shape divergente",
        ok,
    )


def check_selection_uses_clean_frame_as_continuity_history() -> bool:
    target = _rect_mask(20, 10, 40, 40)
    shifted_target = _rect_mask(22, 10, 42, 40)
    distractor = _rect_mask(50, 0, 63, 12)
    clean = [Sam3Instance(target, 0.9), Sam3Instance(distractor, 0.6)]
    degraded = [Sam3Instance(shifted_target, 0.55), Sam3Instance(distractor, 0.95)]
    clean_observation, observation = observe_clean_then_degraded(
        clean, degraded, width=WIDTH, height=HEIGHT
    )
    empty_clean, empty_degraded = observe_clean_then_degraded(
        [], [], width=WIDTH, height=HEIGHT
    )
    degenerate_clean, _ = observe_clean_then_degraded(
        [Sam3Instance(np.zeros((HEIGHT, WIDTH), dtype=bool), 0.8)],
        [],
        width=WIDTH,
        height=HEIGHT,
    )
    ok = bool(
        np.isclose(clean_observation.q_sam3, 0.9)
        and np.isclose(observation.q_sam3, 0.55)
        and observation.present == 1
        and observation.n_instances == 2
        and observation.mask is not None
        and bool(np.array_equal(observation.mask, shifted_target))
        and empty_clean.q_sam3 == 0.0
        and empty_degraded.present == 0
        and degenerate_clean.q_sam3 == 0.0
        and degenerate_clean.mask is None
    )
    return _check(
        "seleção: quadro degradado segue a continuidade do limpo, não o maior "
        "score; ausência e máscara degenerada dão q_sam3 = 0",
        ok,
    )


def _selected_samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "val"],
            "label": [0, 1],
            "video_id": ["env/video_1", "env/video_2"],
            "frame_index": [3, 7],
            "src_index": [3, 7],
        },
        index=[10, 11],
    )


def check_sweep_reuses_clean_inference_and_degrades_proxy() -> bool:
    frame = _synthetic_frame()
    segmenter = _ContrastSegmenter()
    rows = sweep_rows(
        _selected_samples(),
        frames_by_position={10: frame, 11: frame.copy()},
        segmenter=segmenter,
    )
    expected_calls = 2 * (1 + sum(len(values) - 1 for values in DEGRADATION_SWEEPS.values()))
    contrast = cast(pd.DataFrame, rows[rows["degradation"] == "contrast"]).sort_values(
        ["video_id", "severity_index"]
    )
    per_event = [
        group["q_sam3"].to_numpy(dtype=np.float64)
        for _, group in contrast.groupby("video_id", sort=True)
    ]
    clean_rows = rows[rows["severity_index"] == 0]
    repeated = sweep_rows(
        _selected_samples(),
        frames_by_position={10: frame, 11: frame.copy()},
        segmenter=_ContrastSegmenter(),
    )
    ok = bool(
        len(rows) == 2 * sum(len(values) for values in DEGRADATION_SWEEPS.values())
        and segmenter.calls == expected_calls
        and set(rows["split"]) == {"train", "val"}
        and bool(np.all((rows["q_sam3"] >= 0.0) & (rows["q_sam3"] <= 1.0)))
        and bool(np.allclose(clean_rows["mask_iou"], 1.0))
        and bool(np.allclose(clean_rows["q_sam3"], clean_rows["clean_q_sam3"]))
        and all(bool(np.all(np.diff(values) < 0.0)) for values in per_event)
        and rows.equals(repeated)
    )
    return _check(
        "sweep: severidade 0 reaproveita a inferência limpa, q_sam3 cai com o "
        "contraste, IoU limpo = 1 e resultado determinístico",
        ok,
    )


def _sweep_group(q_values: list[list[float]], iou_values: list[list[float]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for event, (qualities, ious) in enumerate(zip(q_values, iou_values)):
        for severity_index, (quality, iou) in enumerate(zip(qualities, ious)):
            records.append(
                {
                    "split": "train",
                    "label": 0,
                    "video_id": f"env/video_{event}",
                    "frame_index": 0,
                    "severity_index": severity_index,
                    "q_sam3": quality,
                    "mask_iou": iou,
                }
            )
    return pd.DataFrame(records)


def check_degradation_summary_handles_constant_events() -> bool:
    group = _sweep_group(
        [
            [0.9, 0.8, 0.7, 0.6, 0.0],
            [0.95, 0.96, 0.9, 0.7, 0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, 0.9, 0.8, 0.5, 0.0],
            [1.0, 0.98, 0.9, 0.6, 0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    )
    summary = cast(dict[str, float], summarize_degradation(group))
    incomplete = group.iloc[:-1]
    ok = bool(
        summary["events"] == 3
        and summary["constant_q_events"] == 1
        and summary["within_event_severity_n"] == 2
        and summary["within_event_mask_iou_n"] == 2
        and np.isclose(summary["non_increasing_fraction"], 11.0 / 12.0)
        and np.isclose(summary["endpoint_drop_fraction"], 2.0 / 3.0)
        and summary["within_event_spearman_severity_q_median"] < 0.0
        and summary["within_event_spearman_q_mask_iou_median"] > 0.0
        and summary["pooled_spearman_severity_q"] < 0.0
        and _raises(summarize_degradation, incomplete)
    )
    return _check(
        "resumo: eventos constantes ficam fora das correlações intraevento, "
        "monotonicidade e queda final contadas por evento, sweep incompleto falha",
        ok,
    )


def check_clean_rows_read_only_train_and_val() -> bool:
    frames = pd.DataFrame(
        {
            "split": ["train"] * 3 + ["val"] * 2 + ["test"] * 2,
            "label": [0, 0, 1, IGNORE_LABEL, 2, 0, 0],
            "video_id": ["env/a"] * 3 + ["env/b"] * 2 + ["env/c"] * 2,
            "frame_index": [0, 1, 2, 0, 1, 0, 1],
            "src_index": [0, 1, 2, 0, 1, 0, 1],
        }
    )
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        for video_id, scores, present, counts in (
            ("env/a", [0.8, 0.9, 0.6], [1, 1, 0], [1, 2, 1]),
            ("env/b", [0.0, 0.7], [0, 1], [0, 1]),
        ):
            v_t = np.zeros((len(scores), 10), dtype=np.float32)
            v_t[:, 0] = present
            write_sam3_atomic(
                sam3_path(video_id, sam3_root=root),
                v_t,
                np.array(scores, dtype=np.float32),
                np.array(counts, dtype=np.int16),
                {"video_id": video_id},
            )
        rows = clean_quality_rows(frames, sam3_root=root)
        write_sam3_atomic(
            sam3_path("env/b", sam3_root=root),
            np.zeros((3, 10), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.int16),
            {"video_id": "env/b"},
        )
        mismatch_rejected = _raises(clean_quality_rows, frames, sam3_root=root)
    ok = bool(
        set(rows["split"]) == {"train", "val"}
        and rows["q_sam3"].tolist() == [np.float32(0.8), np.float32(0.9), 0.0, 0.0, np.float32(0.7)]
        and rows["instance_bucket"].tolist() == ["1", "2+", "1", "0", "1"]
        and mismatch_rejected
    )
    return _check(
        "distribuição limpa: lê só .h5 de train/val (test nem precisa existir), "
        "zera q sem present e rejeita K divergente da grade",
        ok,
    )


def check_validation_cli_is_restricted_to_le2i_cs() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "gatefall.sam3.quality", "validate", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    ok = result.returncode == 0 and "--dataset {le2i}" in output
    return _check(
        "CLI validate: escolha de dataset fica restrita ao protocolo Le2i CS",
        ok,
    )


def run_selftest() -> None:
    checks = [
        check_quality_contract_and_formula(),
        check_quality_is_frame_causal_and_deterministic(),
        check_degradations_are_deterministic_bounded_and_monotonic(),
        check_mask_iou_edge_cases(),
        check_selection_uses_clean_frame_as_continuity_history(),
        check_sweep_reuses_clean_inference_and_degrades_proxy(),
        check_degradation_summary_handles_constant_events(),
        check_clean_rows_read_only_train_and_val(),
        check_validation_cli_is_restricted_to_le2i_cs(),
    ]
    if not all(checks):
        print("\nsam3 quality selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nsam3 quality selftest OK: todas as checagens passaram")
