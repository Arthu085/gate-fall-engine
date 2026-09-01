"""Selftest sintético da padronização (`standardization.py`).

Não toca no dataset real — todas as entradas são sintéticas, para travar o
comportamento da padronização, do passthrough de `kp_conf`, da guarda de
dimensão degenerada e do round-trip de persistência contra futuras mudanças.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from gatefall.config import TRAIN_STRIDE
from gatefall.features.standardization import (
    GUARD_STD_THRESHOLD,
    SOURCE_NAME,
    TRAIN_SPLIT,
    StandardizationStats,
    apply_standardization,
    excluded_dimension_mask,
    load_stats,
    mean_std_from_accumulators,
    save_stats,
    stale_stats_mismatches,
    validate_stats_layout,
)
from gatefall.pose.kinematics import EXPECTED_D, feature_blocks, feature_names


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_known_input_mean0_std1() -> bool:
    rng = np.random.default_rng(0)
    names = feature_names()
    excluded_mask = excluded_dimension_mask(names)
    n = 200
    x = rng.normal(loc=5.0, scale=3.0, size=(n, EXPECTED_D)).astype(np.float32)
    x[:, excluded_mask] = rng.uniform(
        0.0, 1.0, size=(n, int(excluded_mask.sum()))
    ).astype(np.float32)

    mean = x.astype(np.float64).mean(axis=0)
    std = x.astype(np.float64).std(axis=0, ddof=0)
    mean[excluded_mask] = 0.0
    std[excluded_mask] = 1.0

    stats = StandardizationStats(
        source="pose",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=4,
        window_count=n,
        feature_dim=EXPECTED_D,
        feature_names=names,
        excluded_mask=excluded_mask.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
        guarded_count=0,
        guarded_mask=[False] * EXPECTED_D,
        frames_hash="deadbeef",
    )

    standardized = apply_standardization(x, stats)
    standardized_mean = standardized[:, ~excluded_mask].astype(np.float64).mean(axis=0)
    standardized_std = standardized[:, ~excluded_mask].astype(np.float64).std(axis=0, ddof=0)

    ok = bool(np.allclose(standardized_mean, 0.0, atol=1e-4)) and bool(
        np.allclose(standardized_std, 1.0, atol=1e-4)
    )
    return _check(
        "entrada sintética conhecida: dimensões padronizadas dão mean~0, std~1", ok
    )


def check_kp_conf_byte_identical() -> bool:
    rng = np.random.default_rng(1)
    names = feature_names()
    excluded_mask = excluded_dimension_mask(names)
    x = rng.normal(size=(50, EXPECTED_D)).astype(np.float32)
    x[:, excluded_mask] = rng.uniform(0.0, 1.0, size=(50, int(excluded_mask.sum()))).astype(
        np.float32
    )
    x[0, excluded_mask] = 0.0

    mean = np.zeros(EXPECTED_D, dtype=np.float64)
    std = np.ones(EXPECTED_D, dtype=np.float64)
    stats = StandardizationStats(
        source="pose",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=4,
        window_count=x.shape[0],
        feature_dim=EXPECTED_D,
        feature_names=names,
        excluded_mask=excluded_mask.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
        guarded_count=0,
        guarded_mask=[False] * EXPECTED_D,
        frames_hash="deadbeef",
    )

    standardized = apply_standardization(x, stats)
    ok = bool(np.array_equal(standardized[:, excluded_mask], x[:, excluded_mask]))
    ok = ok and int(excluded_mask.sum()) == 17
    ok = ok and bool(np.all(standardized[0, excluded_mask] == 0.0))
    return _check(
        "kp_conf: as 17 colunas saem byte a byte idênticas à entrada, incluindo 0.0 exato",
        ok,
    )


def check_mask_length_and_exclusion() -> bool:
    names = feature_names()
    mask = excluded_dimension_mask(names)
    kp_conf_start, kp_conf_end = next(
        (start, end) for name, start, end in feature_blocks() if name == "kp_conf"
    )
    expected = np.zeros(EXPECTED_D, dtype=bool)
    expected[kp_conf_start:kp_conf_end] = True
    ok = mask.shape[0] == EXPECTED_D and bool(np.array_equal(mask, expected))
    return _check(
        "máscara de exclusão: comprimento 134 e exclui exatamente as colunas kp_conf",
        ok,
    )


def check_degenerate_dimension_guarded() -> bool:
    rng = np.random.default_rng(2)
    names = feature_names()
    excluded_mask = excluded_dimension_mask(names)
    n = 100
    x = rng.normal(size=(n, EXPECTED_D)).astype(np.float32)
    x[:, excluded_mask] = 0.5
    constant_dim = int(np.argmax(~excluded_mask))
    x[:, constant_dim] = 7.0

    mean_raw = x.astype(np.float64).mean(axis=0)
    std_raw = x.astype(np.float64).std(axis=0, ddof=0)
    guarded_mask = (std_raw < GUARD_STD_THRESHOLD) & ~excluded_mask

    mean = mean_raw.copy()
    std = std_raw.copy()
    mean[guarded_mask] = 0.0
    std[guarded_mask] = 1.0
    mean[excluded_mask] = 0.0
    std[excluded_mask] = 1.0

    stats = StandardizationStats(
        source="pose",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=4,
        window_count=n,
        feature_dim=EXPECTED_D,
        feature_names=names,
        excluded_mask=excluded_mask.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
        guarded_count=int(guarded_mask.sum()),
        guarded_mask=guarded_mask.tolist(),
        frames_hash="deadbeef",
    )

    standardized = apply_standardization(x, stats)
    ok = bool(guarded_mask[constant_dim])
    ok = ok and bool(np.isfinite(standardized).all())
    ok = ok and bool(np.allclose(standardized[:, constant_dim], 7.0))
    return _check(
        "dimensão constante: sem inf/NaN após padronizar e contada como guardada",
        ok,
    )


def check_save_load_round_trip(tmp_path: Path) -> bool:
    rng = np.random.default_rng(3)
    names = feature_names()
    excluded_mask = excluded_dimension_mask(names)
    stats = StandardizationStats(
        source="pose",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=4,
        window_count=1234,
        feature_dim=EXPECTED_D,
        feature_names=names,
        excluded_mask=excluded_mask.tolist(),
        mean=rng.normal(size=EXPECTED_D).tolist(),
        std=np.abs(rng.normal(size=EXPECTED_D)).tolist(),
        guarded_count=3,
        guarded_mask=([True] * 3 + [False] * (EXPECTED_D - 3)),
        frames_hash="cafebabe",
    )

    path = tmp_path / "roundtrip_stats.json"
    save_stats(stats, path, force=True)
    loaded = load_stats(path)

    ok = loaded.to_dict() == stats.to_dict()
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    ok = ok and raw == stats.to_dict()
    return _check("save/load: round-trip bit-idêntico ao objeto original", ok)


def check_streaming_matches_batch() -> bool:
    rng = np.random.default_rng(4)
    n_windows = 37
    window_frames = 24
    windows = rng.normal(loc=2.0, scale=1.5, size=(n_windows, window_frames, EXPECTED_D)).astype(
        np.float32
    )

    count = 0
    sum_ = np.zeros(EXPECTED_D, dtype=np.float64)
    sumsq = np.zeros(EXPECTED_D, dtype=np.float64)
    for i in range(n_windows):
        w64 = windows[i].astype(np.float64)
        sum_ += w64.sum(axis=0)
        sumsq += np.square(w64).sum(axis=0)
        count += w64.shape[0]
    streaming_mean, streaming_std = mean_std_from_accumulators(count, sum_, sumsq)

    flat = windows.reshape(-1, EXPECTED_D).astype(np.float64)
    batch_mean = flat.mean(axis=0)
    batch_std = flat.std(axis=0, ddof=0)

    ok = bool(np.allclose(streaming_mean, batch_mean, atol=1e-8)) and bool(
        np.allclose(streaming_std, batch_std, atol=1e-8)
    )
    return _check(
        "acumulação em streaming bate com o cálculo em lote sobre os mesmos dados sintéticos",
        ok,
    )


def check_stale_stats_rejected() -> bool:
    names = feature_names()
    excluded_mask = excluded_dimension_mask(names)
    stats = StandardizationStats(
        source=SOURCE_NAME,
        split=TRAIN_SPLIT,
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1234,
        feature_dim=EXPECTED_D,
        feature_names=names,
        excluded_mask=excluded_mask.tolist(),
        mean=[0.0] * EXPECTED_D,
        std=[1.0] * EXPECTED_D,
        guarded_count=0,
        guarded_mask=[False] * EXPECTED_D,
        frames_hash="deadbeef",
    )

    fresh_ok = stale_stats_mismatches(stats) == []

    stale_names = list(names)
    stale_names[0] = f"{stale_names[0]}_stale"
    stale_stats = StandardizationStats(**{**stats.to_dict(), "feature_names": stale_names})
    stale_ok = stale_stats_mismatches(stale_stats) == ["feature_names"]

    stale_stride_stats = StandardizationStats(
        **{**stats.to_dict(), "stride": TRAIN_STRIDE + 1}
    )
    stale_stride_ok = stale_stats_mismatches(stale_stride_stats) == ["stride"]

    ok = fresh_ok and stale_ok and stale_stride_ok
    return _check(
        "checagem de estatísticas obsoletas: aceita stats atuais e rejeita "
        "feature_names/stride divergentes de gatefall.pose.kinematics",
        ok,
    )


def check_stale_feature_layout_fails_explicitly() -> bool:
    names = feature_names()
    stale_dim = EXPECTED_D - 1
    stats = StandardizationStats(
        source=SOURCE_NAME,
        split=TRAIN_SPLIT,
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1234,
        feature_dim=stale_dim,
        feature_names=names[:stale_dim],
        excluded_mask=[False] * stale_dim,
        mean=[0.0] * stale_dim,
        std=[1.0] * stale_dim,
        guarded_count=0,
        guarded_mask=[False] * stale_dim,
        frames_hash="deadbeef",
    )

    try:
        validate_stats_layout(stats)
    except ValueError as exc:
        message = str(exc)
        ok = "feature_dim" in message and "feature_names" in message
    except IndexError:
        ok = False
    else:
        ok = False
    return _check(
        "layout obsoleto: falha explicitamente antes de construir/indexar máscaras",
        ok,
    )


def run_standardization_selftest() -> None:
    checks = [
        check_known_input_mean0_std1(),
        check_kp_conf_byte_identical(),
        check_mask_length_and_exclusion(),
        check_degenerate_dimension_guarded(),
        check_streaming_matches_batch(),
        check_stale_stats_rejected(),
        check_stale_feature_layout_fails_explicitly(),
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        checks.append(check_save_load_round_trip(Path(tmp_dir)))

    if not all(checks):
        print("\nstandardization selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nstandardization selftest OK: todas as checagens passaram")
