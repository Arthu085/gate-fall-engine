"""Selftest sintético da padronização (z-score) do descritor SAM 3 V_t
(`sam3_standardization.py`). Não toca no dataset real — trava o ajuste só no
train (sem vazamento), a guarda de canal degenerado, o round-trip de
persistência e o frescor contra `frames.parquet` e o conjunto de `.h5`."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from gatefall.features.sam3_standardization import (
    FEATURE_DIM,
    SOURCE_NAME,
    Sam3StandardizationStats,
    apply_standardization,
    compute_train_stats,
    load_stats,
    save_stats,
    stale_stats_mismatches,
    validate_stats_freshness,
    validate_stats_layout,
)

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


def _fit(windows: np.ndarray, frames_bytes: bytes = b"frames") -> Sam3StandardizationStats:
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_path = Path(tmp_dir) / "frames.parquet"
        frames_path.write_bytes(frames_bytes)
        return compute_train_stats(
            _SyntheticWindowSource(windows), frames_path, "le2i", "feedface", stride=STRIDE
        )


def check_layout_rejects_wrong_feature_dim_and_source() -> bool:
    stats = _fit(_make_windows(np.random.default_rng(0), 4, 0.0, 1.0))
    wrong_dim = Sam3StandardizationStats(
        **{**stats.to_dict(), "feature_dim": FEATURE_DIM + 1}
    )
    wrong_source = Sam3StandardizationStats(**{**stats.to_dict(), "source": "dinov3"})
    wrong_channels = Sam3StandardizationStats(
        **{**stats.to_dict(), "channel_names": list(reversed(stats.channel_names))}
    )
    rejected = []
    for candidate in (wrong_dim, wrong_source, wrong_channels):
        try:
            validate_stats_layout(candidate)
            rejected.append(False)
        except ValueError:
            rejected.append(True)
    return _check(
        f"validate_stats_layout rejeita feature_dim != {FEATURE_DIM}, source != "
        f"{SOURCE_NAME!r} e ordem de canais divergente de CHANNEL_NAMES",
        all(rejected) and stale_stats_mismatches(stats, dataset_name="le2i") == [],
    )


def check_train_only_no_leakage() -> bool:
    rng = np.random.default_rng(1)
    train_windows = _make_windows(rng, 40, loc=0.3, scale=0.1)
    stats = _fit(train_windows)

    flat_train = apply_standardization(train_windows, stats).reshape(-1, FEATURE_DIM)
    train_ok = bool(np.allclose(flat_train.astype(np.float64).mean(axis=0), 0.0, atol=1e-3)) and bool(
        np.allclose(flat_train.astype(np.float64).std(axis=0), 1.0, atol=1e-3)
    )
    val_windows = _make_windows(rng, 20, loc=5.0, scale=0.1)
    val_mean = apply_standardization(val_windows, stats).reshape(-1, FEATURE_DIM).mean(axis=0)
    return _check(
        "estatísticas ajustadas só no train dão mean~0/std~1 no train e NÃO "
        "centralizam val de distribuição diferente (sem vazamento)",
        train_ok and bool(np.all(np.abs(val_mean) > 1.0)),
    )


def check_constant_channel_guarded() -> bool:
    windows = _make_windows(np.random.default_rng(2), 30, loc=0.0, scale=1.0)
    present_channel = 0
    windows[:, :, present_channel] = 1.0
    stats = _fit(windows)
    standardized = apply_standardization(windows, stats)
    ok = (
        bool(stats.guarded_mask[present_channel])
        and stats.guarded_count == 1
        and bool(np.isfinite(standardized).all())
        and bool(np.allclose(standardized[:, :, present_channel], 1.0))
    )
    return _check(
        "canal constante (ex.: present sempre 1) é guardado com mean=0/std=1 "
        "em vez de dividir por ~0",
        ok,
    )


def check_save_load_round_trip() -> bool:
    stats = _fit(_make_windows(np.random.default_rng(3), 8, 0.5, 0.2))
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "sam3_stats.json"
        save_stats(stats, path, force=True)
        loaded = load_stats(path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    return _check(
        "save/load: round-trip bit-idêntico ao objeto original",
        loaded.to_dict() == stats.to_dict() and raw == stats.to_dict(),
    )


def check_freshness_rejects_stale_frames_and_features() -> bool:
    windows = _make_windows(np.random.default_rng(4), 6, 0.0, 1.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_path = Path(tmp_dir) / "frames.parquet"
        frames_path.write_bytes(b"frames-original")
        stats = compute_train_stats(
            _SyntheticWindowSource(windows), frames_path, "le2i", "feedface", stride=STRIDE
        )

        fresh_ok = True
        try:
            validate_stats_freshness(stats, frames_path, "feedface")
        except ValueError:
            fresh_ok = False

        features_rejected = False
        try:
            validate_stats_freshness(stats, frames_path, "reextracted")
        except ValueError as exc:
            features_rejected = "sam3_features_sha256" in str(exc)

        frames_path.write_bytes(b"frames-changed")
        frames_rejected = False
        try:
            validate_stats_freshness(stats, frames_path, "feedface")
        except ValueError as exc:
            frames_rejected = "frames_hash" in str(exc)

    return _check(
        "validate_stats_freshness aceita o estado atual e recusa frames.parquet "
        "alterado ou conjunto de .h5 do SAM 3 reextraído",
        fresh_ok and features_rejected and frames_rejected,
    )


def run_sam3_standardization_selftest() -> bool:
    checks = [
        check_layout_rejects_wrong_feature_dim_and_source(),
        check_train_only_no_leakage(),
        check_constant_channel_guarded(),
        check_save_load_round_trip(),
        check_freshness_rejects_stale_frames_and_features(),
    ]
    ok = all(checks)
    if not ok:
        print("\nsam3 standardization selftest FALHOU", file=sys.stderr)
    else:
        print("\nsam3 standardization selftest OK: todas as checagens passaram")
    return ok
