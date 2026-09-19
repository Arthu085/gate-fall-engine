"""Selftest sintético da padronização (z-score) das features DINOv3
(`dinov3_standardization.py`). Não toca no dataset real — todas as entradas
são sintéticas, para travar o comportamento de treino-apenas-no-train (sem
vazamento), da guarda de dimensão degenerada e do round-trip de persistência.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from gatefall.features.dinov3_standardization import (
    SOURCE_NAME,
    TRAIN_SPLIT,
    Dinov3StandardizationStats,
    apply_standardization,
    compute_train_stats,
    load_stats,
    save_stats,
    stale_stats_mismatches,
    validate_stats_layout,
)

FEATURE_DIM = 1536
WINDOW_FRAMES = 24
STRIDE = 4


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


class _SyntheticWindowSource:
    def __init__(self, windows: np.ndarray) -> None:
        self._windows = windows

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, object]:
        return self._windows[index], 0, ("synthetic_video", index)


def _make_windows(
    rng: np.random.Generator, n_windows: int, loc: float, scale: float
) -> np.ndarray:
    return rng.normal(
        loc=loc, scale=scale, size=(n_windows, WINDOW_FRAMES, FEATURE_DIM)
    ).astype(np.float32)


def _identity_stats(feature_dim: int) -> Dinov3StandardizationStats:
    return Dinov3StandardizationStats(
        source=SOURCE_NAME,
        dataset="le2i",
        split=TRAIN_SPLIT,
        target_fps=10.0,
        window_frames=WINDOW_FRAMES,
        stride=STRIDE,
        window_count=1,
        feature_dim=feature_dim,
        mean=[0.0] * feature_dim,
        std=[1.0] * feature_dim,
        guarded_count=0,
        guarded_mask=[False] * feature_dim,
        frames_hash="deadbeef",
    )


def check_layout_rejects_wrong_feature_dim() -> bool:
    stats = _identity_stats(FEATURE_DIM - 1)
    raised = False
    try:
        validate_stats_layout(stats)
    except ValueError:
        raised = True
    return _check(
        "validate_stats_layout rejeita feature_dim != 1536",
        raised,
    )


def check_layout_rejects_wrong_source_name() -> bool:
    stats = _identity_stats(FEATURE_DIM)
    stats = Dinov3StandardizationStats(**{**stats.to_dict(), "source": "pose"})
    raised = False
    try:
        validate_stats_layout(stats)
    except ValueError:
        raised = True
    return _check(
        f"validate_stats_layout rejeita source != {SOURCE_NAME!r}",
        raised,
    )


def check_train_only_no_leakage() -> bool:
    rng = np.random.default_rng(0)
    train_windows = _make_windows(rng, 40, loc=5.0, scale=2.0)
    source = _SyntheticWindowSource(train_windows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_path = Path(tmp_dir) / "frames.parquet"
        frames_path.write_bytes(b"synthetic-frames-fixture")
        stats = compute_train_stats(source, frames_path, "le2i", stride=STRIDE)

    train_standardized = apply_standardization(train_windows, stats)
    flat_train = train_standardized.reshape(-1, FEATURE_DIM).astype(np.float64)
    train_mean = flat_train.mean(axis=0)
    train_std = flat_train.std(axis=0)
    train_ok = bool(np.allclose(train_mean, 0.0, atol=1e-3)) and bool(
        np.allclose(train_std, 1.0, atol=1e-3)
    )

    # val/test amostrados de uma distribuição bem diferente do train: se as
    # estatísticas vazassem informação de val/test, o resultado padronizado
    # ficaria artificialmente centrado também aqui.
    val_windows = _make_windows(rng, 20, loc=50.0, scale=2.0)
    val_standardized = apply_standardization(val_windows, stats)
    val_mean = val_standardized.reshape(-1, FEATURE_DIM).astype(np.float64).mean(axis=0)
    no_leakage_ok = bool(np.all(np.abs(val_mean) > 1.0))

    return _check(
        "estatísticas treinadas só em TRAIN sintético dão mean~0/std~1 no "
        "próprio train, mas NÃO centralizam val de distribuição diferente "
        "(prova de ausência de vazamento)",
        train_ok and no_leakage_ok,
    )


def check_constant_dimension_guarded() -> bool:
    rng = np.random.default_rng(1)
    windows = _make_windows(rng, 30, loc=0.0, scale=1.0)
    constant_dim = 7
    windows[:, :, constant_dim] = 3.0
    source = _SyntheticWindowSource(windows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_path = Path(tmp_dir) / "frames.parquet"
        frames_path.write_bytes(b"synthetic-frames-fixture-constant")
        stats = compute_train_stats(source, frames_path, "le2i", stride=STRIDE)

    standardized = apply_standardization(windows, stats)
    ok = bool(stats.guarded_mask[constant_dim])
    ok = ok and bool(np.isfinite(standardized).all())
    ok = ok and bool(np.allclose(standardized[:, :, constant_dim], 3.0))
    return _check(
        "dimensão constante sintética é guardada (piso de std) em vez de "
        "dividir por ~0, e a saída permanece finita",
        ok,
    )


def check_save_load_round_trip() -> bool:
    rng = np.random.default_rng(2)
    stats = Dinov3StandardizationStats(
        source=SOURCE_NAME,
        dataset="le2i",
        split=TRAIN_SPLIT,
        target_fps=10.0,
        window_frames=WINDOW_FRAMES,
        stride=STRIDE,
        window_count=1234,
        feature_dim=FEATURE_DIM,
        mean=rng.normal(size=FEATURE_DIM).tolist(),
        std=np.abs(rng.normal(size=FEATURE_DIM)).tolist(),
        guarded_count=3,
        guarded_mask=([True] * 3 + [False] * (FEATURE_DIM - 3)),
        frames_hash="cafebabe",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "dinov3_roundtrip_stats.json"
        save_stats(stats, path, force=True)
        loaded = load_stats(path)

        ok = loaded.to_dict() == stats.to_dict()
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        ok = ok and raw == stats.to_dict()
    return _check("save/load: round-trip bit-idêntico ao objeto original", ok)


def check_streaming_matches_batch() -> bool:
    rng = np.random.default_rng(3)
    n_windows = 37
    windows = rng.normal(
        loc=2.0, scale=1.5, size=(n_windows, WINDOW_FRAMES, FEATURE_DIM)
    ).astype(np.float32)
    source = _SyntheticWindowSource(windows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_path = Path(tmp_dir) / "frames.parquet"
        frames_path.write_bytes(b"synthetic-frames-fixture-streaming")
        stats = compute_train_stats(source, frames_path, "le2i", stride=STRIDE)

    flat = windows.reshape(-1, FEATURE_DIM).astype(np.float64)
    batch_mean = flat.mean(axis=0)
    batch_std = flat.std(axis=0)

    ok = bool(
        np.allclose(np.asarray(stats.mean), batch_mean, atol=1e-6)
    ) and bool(np.allclose(np.asarray(stats.std), batch_std, atol=1e-6))
    return _check(
        "acumulação em streaming de compute_train_stats bate com o cálculo "
        "em lote sobre os mesmos dados sintéticos",
        ok,
    )


def check_stale_stats_rejected() -> bool:
    stats = _identity_stats(FEATURE_DIM)
    fresh_ok = stale_stats_mismatches(stats) == []

    stale_dim_stats = Dinov3StandardizationStats(
        **{**stats.to_dict(), "feature_dim": FEATURE_DIM - 1}
    )
    stale_ok = "feature_dim" in stale_stats_mismatches(stale_dim_stats)

    ok = fresh_ok and stale_ok
    return _check(
        "checagem de estatísticas obsoletas: aceita stats atuais e sinaliza "
        "feature_dim divergente do layout DINOv3 atual",
        ok,
    )


def check_stale_stats_rejects_wrong_dataset() -> bool:
    stats = _identity_stats(FEATURE_DIM)
    matching_dataset_ok = stale_stats_mismatches(stats, dataset_name="le2i") == []
    mismatched_dataset_ok = "dataset" in stale_stats_mismatches(
        stats, dataset_name="outro-dataset"
    )
    no_dataset_arg_ok = stale_stats_mismatches(stats) == []

    ok = matching_dataset_ok and mismatched_dataset_ok and no_dataset_arg_ok
    return _check(
        "stale_stats_mismatches sinaliza 'dataset' quando dataset_name "
        "informado diverge de stats.dataset, e não checa dataset quando "
        "dataset_name é omitido",
        ok,
    )


def check_freshness_rejects_stale_frames_hash() -> bool:
    from gatefall.features.dinov3_standardization import validate_stats_freshness

    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_path = Path(tmp_dir) / "frames.parquet"
        frames_path.write_bytes(b"frames-original")

        source = _SyntheticWindowSource(
            _make_windows(np.random.default_rng(4), 10, loc=0.0, scale=1.0)
        )
        stats = compute_train_stats(source, frames_path, "le2i", stride=STRIDE)

        fresh_ok = True
        try:
            validate_stats_freshness(stats, frames_path)
        except ValueError:
            fresh_ok = False

        frames_path.write_bytes(b"frames-changed-after-stats-were-fit")
        stale_raised = False
        try:
            validate_stats_freshness(stats, frames_path)
        except ValueError:
            stale_raised = True

    ok = fresh_ok and stale_raised
    return _check(
        "validate_stats_freshness aceita frames_hash atual e recusa "
        "frames.csv/parquet alterado depois que as estatísticas foram "
        "ajustadas",
        ok,
    )


def run_dinov3_standardization_selftest() -> bool:
    checks = [
        check_layout_rejects_wrong_feature_dim(),
        check_layout_rejects_wrong_source_name(),
        check_train_only_no_leakage(),
        check_constant_dimension_guarded(),
        check_save_load_round_trip(),
        check_streaming_matches_batch(),
        check_stale_stats_rejected(),
        check_stale_stats_rejects_wrong_dataset(),
        check_freshness_rejects_stale_frames_hash(),
    ]
    ok = all(checks)
    if not ok:
        print("\ndinov3 standardization selftest FALHOU", file=sys.stderr)
    else:
        print("\ndinov3 standardization selftest OK: todas as checagens passaram")
    return ok
