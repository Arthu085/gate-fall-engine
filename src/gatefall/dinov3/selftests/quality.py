"""Selftest sintético da qualidade visual usada pela fonte DINOv3.

Não acessa vídeos, pesos do backbone ou artefatos reais. Os casos fixam o
contrato numérico, a causalidade por quadro e a preparação determinística da
validação de degradação.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torchvision.transforms.v2.functional import InterpolationMode, resize

from gatefall.config import IGNORE_LABEL
from gatefall.dinov3.backbone import NORMALIZE_MEAN, NORMALIZE_STD, RESIZE_SIZE
from gatefall.dinov3.preprocessing import (
    normalize_resized_frames,
    preprocess_frames,
    resize_frames,
)
from gatefall.dinov3.quality import (
    VisualQualityComponents,
    aggregate_metrics,
    apply_degradation,
    compute_visual_quality,
    cosine_similarity,
    pearson_correlation,
    select_validation_samples,
    spearman_correlation,
    summarize_within_event_correlations,
)

SEED = 20260919


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _synthetic_batch() -> torch.Tensor:
    height = width = RESIZE_SIZE
    y, x = torch.meshgrid(
        torch.linspace(0.05, 0.95, height),
        torch.linspace(0.05, 0.95, width),
        indexing="ij",
    )
    checker = ((torch.arange(height)[:, None] + torch.arange(width)[None, :]) % 2).float()
    textured = torch.stack((x, y, 0.25 + 0.5 * checker), dim=0)
    return textured.unsqueeze(0).to(torch.float32)


def _all_component_arrays(components: VisualQualityComponents) -> list[np.ndarray]:
    return [
        components.q_exposure,
        components.q_contrast,
        components.q_sharpness,
        components.q_visual,
    ]


def check_quality_contract_and_formula() -> bool:
    black = torch.zeros((1, 3, RESIZE_SIZE, RESIZE_SIZE), dtype=torch.float32)
    white = torch.ones_like(black)
    middle = torch.full_like(black, 0.5)
    textured = _synthetic_batch()
    batch = torch.cat((black, white, middle, textured), dim=0)

    components = compute_visual_quality(batch)
    arrays = _all_component_arrays(components)
    arrays_ok = all(
        value.shape == (4,)
        and value.dtype == np.float32
        and bool(np.isfinite(value).all())
        and bool(np.all((value >= 0.0) & (value <= 1.0)))
        for value in arrays
    )

    formula = np.cbrt(
        components.q_exposure
        * components.q_contrast
        * components.q_sharpness
    ).astype(np.float32)
    edge_cases_ok = (
        bool(np.allclose(components.q_visual, formula, atol=1e-6))
        and components.q_exposure[0] == 0.0
        and components.q_exposure[1] == 0.0
        and bool(np.isclose(components.q_exposure[2], 1.0, atol=1e-6))
        and components.q_contrast[0] == 0.0
        and components.q_contrast[1] == 0.0
        and components.q_contrast[2] == 0.0
        and components.q_sharpness[0] == 0.0
        and components.q_sharpness[1] == 0.0
        and components.q_sharpness[2] == 0.0
        and components.q_visual[0] == 0.0
        and components.q_visual[1] == 0.0
        and components.q_visual[2] == 0.0
        and components.q_visual[3] > 0.0
    )
    return _check(
        "qualidade visual: componentes float32 finitos em [0,1], média constante "
        "tem exposição máxima, extremos/constantes têm q_visual zero e a média "
        "geométrica segue a fórmula",
        arrays_ok and edge_cases_ok,
    )


def check_quality_is_frame_causal_and_order_independent() -> bool:
    generator = torch.Generator().manual_seed(SEED)
    batch = torch.rand(
        (5, 3, RESIZE_SIZE, RESIZE_SIZE), generator=generator, dtype=torch.float32
    )
    together = compute_visual_quality(batch)
    reversed_batch = compute_visual_quality(batch.flip(0))

    ok = True
    for index in range(batch.shape[0]):
        alone = compute_visual_quality(batch[index : index + 1])
        for together_value, reversed_value, alone_value in zip(
            _all_component_arrays(together),
            _all_component_arrays(reversed_batch),
            _all_component_arrays(alone),
        ):
            ok = ok and bool(np.allclose(together_value[index], alone_value[0], atol=1e-7))
            ok = ok and bool(
                np.allclose(together_value[index], reversed_value[-1 - index], atol=1e-7)
            )
    return _check(
        "compute_visual_quality: cada quadro independe do batch e da ordem", ok
    )


def check_degradations_are_deterministic_bounded_and_monotonic() -> bool:
    clean = _synthetic_batch()
    sweeps = {
        "blur": [0, 2, 3, 6, 12],
        "underexposure": [1.0, 0.75, 0.5, 0.25, 0.125],
        "overexposure": [1.0, 0.75, 0.5, 0.25, 0.125],
        "contrast": [1.0, 0.75, 0.5, 0.25, 0.125],
    }

    deterministic_and_bounded = True
    qualities: dict[str, list[VisualQualityComponents]] = {}
    for degradation, severities in sweeps.items():
        qualities[degradation] = []
        for severity in severities:
            first = apply_degradation(clean, degradation, severity)
            second = apply_degradation(clean, degradation, severity)
            deterministic_and_bounded = deterministic_and_bounded and (
                first.dtype == torch.float32
                and first.shape == clean.shape
                and bool(torch.equal(first, second))
                and bool(torch.isfinite(first).all())
                and bool(torch.all((first >= 0.0) & (first <= 1.0)))
            )
            qualities[degradation].append(compute_visual_quality(first))

    blur_values = [float(item.q_sharpness[0]) for item in qualities["blur"]]
    under_values = [float(item.q_exposure[0]) for item in qualities["underexposure"]]
    over_values = [float(item.q_exposure[0]) for item in qualities["overexposure"]]
    contrast_values = [float(item.q_contrast[0]) for item in qualities["contrast"]]
    visual_values = {
        name: [float(item.q_visual[0]) for item in items]
        for name, items in qualities.items()
    }

    monotonic = all(
        values[index + 1] <= values[index] + 1e-6
        for values in (
            blur_values,
            under_values,
            over_values,
            contrast_values,
            *visual_values.values(),
        )
        for index in range(len(values) - 1)
    )
    endpoints_degrade = all(
        values[-1] < values[0]
        for values in (blur_values, under_values, over_values, contrast_values)
    )
    identities_ok = all(
        bool(torch.equal(apply_degradation(clean, name, levels[0]), clean))
        for name, levels in sweeps.items()
    )
    return _check(
        "degradações: blur/subexposição/sobre-exposição/contraste são "
        "determinísticos, limitados, identidade na severidade limpa e degradam "
        "monotonicamente o componente correspondente",
        deterministic_and_bounded and monotonic and endpoints_degrade and identities_ok,
    )


def check_preprocessing_refactor_preserves_output() -> bool:
    generator = np.random.default_rng(SEED)
    frames = [
        generator.integers(0, 256, size=(73, 119, 3), dtype=np.uint8),
        generator.integers(0, 256, size=(73, 119, 3), dtype=np.uint8),
    ]
    different_shape = generator.integers(
        0, 256, size=(91, 67, 3), dtype=np.uint8
    )
    stacked = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
    legacy_resized = resize(
        stacked,
        size=[RESIZE_SIZE, RESIZE_SIZE],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).to(torch.float32) / 255.0
    mean = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(1, 3, 1, 1)
    legacy = (legacy_resized - mean) / std

    resized = resize_frames(frames)
    normalized = normalize_resized_frames(resized)
    composed = preprocess_frames(frames)
    mixed_resized = resize_frames([frames[0], different_shape, frames[1]])
    mixed_expected = torch.cat(
        [
            resize_frames([frames[0]]),
            resize_frames([different_shape]),
            resize_frames([frames[1]]),
        ],
        dim=0,
    )
    ok = (
        resized.dtype == torch.float32
        and resized.shape == (2, 3, RESIZE_SIZE, RESIZE_SIZE)
        and bool(torch.all((resized >= 0.0) & (resized <= 1.0)))
        and bool(torch.equal(resized, legacy_resized))
        and bool(torch.equal(normalized, legacy))
        and bool(torch.equal(composed, legacy))
        and bool(torch.equal(mixed_resized, mixed_expected))
    )
    return _check(
        "pré-processamento: resize em [0,1] + normalização preserva exatamente "
        "a saída anterior de preprocess_frames e aceita shapes mistos",
        ok,
    )


def _selection_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("train", 0, "room/a", 0, 10),
            ("train", 0, "room/a", 1, 11),
            ("train", 1, "room/a", 2, 12),
            ("train", 1, "room/a", 3, 13),
            ("train", 0, "room/a", 4, 14),
            ("train", 0, "room/a", 5, 15),
            ("val", 0, "office/b", 0, 20),
            ("val", 0, "office/b", 1, 21),
            ("val", 1, "office/b", 2, 22),
            ("val", 1, "office/b", 3, 23),
            ("val", IGNORE_LABEL, "office/b", 4, 24),
            ("test", 0, "home/c", 0, 30),
        ],
        columns=["split", "label", "video_id", "frame_index", "src_index"],
    )


def check_validation_selection_is_deterministic_and_isolated() -> bool:
    frames = _selection_fixture()
    shuffled = frames.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    selected = select_validation_samples(frames, events_per_class=1)
    selected_shuffled = select_validation_samples(shuffled, events_per_class=1)

    required = {"split", "label", "video_id", "frame_index", "src_index"}
    normalized = selected.sort_values(sorted(required)).reset_index(drop=True)
    normalized_shuffled = selected_shuffled.sort_values(sorted(required)).reset_index(drop=True)
    group_counts = selected.groupby(["split", "label"]).size()
    ok = (
        required.issubset(selected.columns)
        and set(selected["split"]) <= {"train", "val"}
        and IGNORE_LABEL not in set(selected["label"])
        and bool((group_counts <= 1).all())
        and set(selected["split"]) == {"train", "val"}
        and set(selected["label"]) == {0, 1}
        and normalized.equals(normalized_shuffled)
    )
    return _check(
        "seleção real: determinística e independente da ordem, uma amostra por "
        "evento/classe solicitada, apenas train/val e nunca IGNORE_LABEL",
        ok,
    )


def check_similarity_correlation_and_aggregation_helpers() -> bool:
    clean = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    corrupted = np.array([[1.0, 0.0], [-1.0, -1.0], [0.0, 0.0]], dtype=np.float32)
    cosine = cosine_similarity(clean, corrupted)
    cosine_ok = (
        cosine.shape == (3,)
        and cosine.dtype == np.float32
        and bool(np.isfinite(cosine).all())
        and bool(np.allclose(cosine, np.array([1.0, -1.0, 0.0], dtype=np.float32)))
    )

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    increasing = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    decreasing = increasing[::-1].copy()
    constant = np.ones(4, dtype=np.float32)
    correlations_ok = bool(
        np.isclose(pearson_correlation(x, increasing), 1.0)
        and np.isclose(spearman_correlation(x, decreasing), -1.0)
        and pearson_correlation(x, constant) == 0.0
        and spearman_correlation(x, constant) == 0.0
    )

    rows = pd.DataFrame(
        {
            "split": ["train"] * 4 + ["val"] * 2,
            "label": [0] * 4 + [1] * 2,
            "q_visual": [0.1, 0.3, 0.5, 0.7, 0.2, 0.6],
            "similarity": [0.2, 0.4, 0.6, 0.8, 0.3, 0.9],
        }
    )
    summary = aggregate_metrics(
        rows,
        group_columns=["split", "label"],
        metric_columns=["q_visual", "similarity"],
    )
    train = summary[(summary["split"] == "train") & (summary["label"] == 0)].iloc[0]
    aggregation_ok = bool(
        set(
            [
                "split",
                "label",
                "n",
                "q_visual_median",
                "q_visual_q25",
                "q_visual_q75",
                "similarity_median",
                "similarity_q25",
                "similarity_q75",
            ]
        ).issubset(summary.columns)
        and int(train["n"]) == 4
        and np.isclose(float(train["q_visual_median"]), 0.4)
        and np.isclose(float(train["q_visual_q25"]), 0.25)
        and np.isclose(float(train["q_visual_q75"]), 0.55)
    )
    return _check(
        "helpers: cosseno por linha é finito, correlações tratam constantes e "
        "agregação reporta n, mediana e IQR por grupo",
        cosine_ok and correlations_ok and aggregation_ok,
    )


def check_within_event_correlations_resist_pooling_reversal() -> bool:
    rows = pd.DataFrame(
        {
            "split": ["train"] * 10,
            "label": [0] * 5 + [1] * 5,
            "video_id": ["a"] * 5 + ["b"] * 5,
            "frame_index": [10] * 5 + [20] * 5,
            "severity_index": list(range(5)) * 2,
            "q_visual": [0, 1, 2, 3, 4, 100, 101, 102, 103, 104],
            "similarity": [100, 101, 102, 103, 104, 0, 1, 2, 3, 4],
        }
    )
    identity_columns = ["split", "label", "video_id", "frame_index"]
    summary = summarize_within_event_correlations(
        rows, identity_columns=identity_columns
    )
    pooled_pearson = pearson_correlation(
        rows["q_visual"].to_numpy(), rows["similarity"].to_numpy()
    )
    pooled_spearman = spearman_correlation(
        rows["q_visual"].to_numpy(), rows["similarity"].to_numpy()
    )
    ok = bool(
        int(summary["within_event_n"]) == 2
        and np.isclose(summary["within_event_pearson_median"], 1.0)
        and np.isclose(summary["within_event_pearson_q25"], 1.0)
        and np.isclose(summary["within_event_pearson_q75"], 1.0)
        and summary["within_event_pearson_fraction_positive"] == 1.0
        and np.isclose(summary["within_event_spearman_median"], 1.0)
        and np.isclose(summary["within_event_spearman_q25"], 1.0)
        and np.isclose(summary["within_event_spearman_q75"], 1.0)
        and summary["within_event_spearman_fraction_positive"] == 1.0
        and pooled_pearson < 0.0
        and pooled_spearman < 0.0
    )
    return _check(
        "correlações: resumo intraevento preserva relações positivas mesmo "
        "quando o agrupamento pooled sofre reversão de Simpson",
        ok,
    )


def check_validation_cli_is_restricted_to_le2i_cs() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "gatefall.dinov3.quality", "validate", "--help"],
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
        check_quality_is_frame_causal_and_order_independent(),
        check_degradations_are_deterministic_bounded_and_monotonic(),
        check_preprocessing_refactor_preserves_output(),
        check_validation_selection_is_deterministic_and_isolated(),
        check_similarity_correlation_and_aggregation_helpers(),
        check_within_event_correlations_resist_pooling_reversal(),
        check_validation_cli_is_restricted_to_le2i_cs(),
    ]
    if not all(checks):
        print("\ndinov3 quality selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\ndinov3 quality selftest OK: todas as checagens passaram")
