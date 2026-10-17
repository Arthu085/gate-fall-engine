"""Provas sintéticas de causalidade, isolamento, cache, treino e decisão."""

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.windowing import build_window_index
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.hashing import sha256_file
from gatefall.train.internal_folds.folds import DEVELOPMENT_SPLIT, RETAINED_SPLIT, build_fold_partitions
from gatefall.train.internal_folds.selftests import fixtures as cv_fixtures
from gatefall.train.subject_folds.folds import build_subject_fold_partitions
from gatefall.train.subject_folds.selftests import fixtures as cs_fixtures
from gatefall.train.temporal_rgb import contract as configuration
from gatefall.train.temporal_rgb import extraction, training
from gatefall.train.temporal_rgb.contract import FEATURE_DIM, MODES, contract, validate_output, validate_persistent_checkout
from gatefall.train.temporal_rgb.data import Context, align_video, fold_windows, window_identity
from gatefall.train.temporal_rgb.report import build_report, paired, pilot_decision
from gatefall.train.temporal_rgb.storage import cache_path, publish_cache, read_cache, write_json


class SyntheticBackbone(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.mean(dim=(1, 2, 3, 4))[:, None].repeat(1, FEATURE_DIM)


def rejects(action: Any, errors: Any = (ValueError, RuntimeError, OSError, KeyError)) -> None:
    try:
        action()
    except errors:
        return
    raise AssertionError("operação inválida foi aceita")


def video_frames() -> pd.DataFrame:
    count = 40
    times = np.arange(count) / 10
    return pd.DataFrame({
        "video_id": ["synthetic/video"] * count, "split": ["train"] * count,
        "env": ["EnvA"] * count, "subject": [0] * count,
        "frame_index": np.arange(count), "time_s": times,
        "src_index": np.round(times * 25).astype(np.int64),
        "label": np.resize([0, 1, 2, 3, 4, 7, 8, 9], count),
    })


def test_alignment_and_cache(root: Path) -> None:
    frames = video_frames()
    alignment = align_video(frames, 25, 100)
    assert alignment["window_grid_indices"].shape == (40, 24)
    assert alignment["clip_src_indices"].shape == (40, 16)
    assert not alignment["clip_src_indices"][0].any()
    assert np.all(alignment["clip_src_times_s"] <= alignment["endpoint_times_s"][:, None] + 1e-8)
    assert alignment["original_src_indices"][3, -1] == 8
    assert alignment["clip_src_indices"][3, -1] == 7
    assert np.array_equal(alignment["static_src_indices"], np.repeat(alignment["clip_src_indices"][:, -1:], 16, axis=1))
    prefix = align_video(frames.iloc[:25].copy(), 25, 100)
    assert all(np.array_equal(value[:25], prefix[name]) for name, value in alignment.items())
    changed_future = frames.copy()
    changed_future.loc[25:, "src_index"] += 1
    changed_future.loc[25:, "label"] = 9
    altered = align_video(changed_future, 25, 102)
    assert all(np.array_equal(value[:25], altered[name][:25]) for name, value in alignment.items())
    ignored = frames.copy()
    ignored.loc[4, "label"] = -1
    assert 4 not in align_video(ignored, 25, 100)["k_end"]
    rejects(lambda: align_video(frames.assign(split="test"), 25, 100))
    rejects(lambda: align_video(frames.assign(frame_index=0), 25, 100))
    rejects(lambda: align_video(frames.assign(src_index=-1), 25, 100))
    rejects(lambda: align_video(frames, 0, 100))
    features = {mode: np.full((40, FEATURE_DIM), index, dtype=np.float32) for index, mode in enumerate(MODES)}
    cache = root / "cache.h5"
    expected = {"source_sha256": "a" * 64}
    result = publish_cache(cache, expected, alignment, lambda: features)
    assert np.array_equal(result["dynamic"], features["dynamic"])
    publish_cache(cache, expected, alignment, lambda: (_ for _ in ()).throw(AssertionError("cache não reutilizado")))
    rejects(lambda: read_cache(cache, {"source_sha256": "b" * 64}, alignment))
    with h5py.File(cache, "r+") as stream:
        cast(h5py.Dataset, stream["dynamic"])[0, 0] = 123
    rejects(lambda: read_cache(cache, expected, alignment))
    partial = root / "partial.h5"
    rejects(lambda: publish_cache(partial, expected, alignment, lambda: {"dynamic": features["dynamic"]}))
    assert not partial.exists() and not list(root.glob(".partial-*"))
    rejects(lambda: publish_cache(partial, expected, alignment, lambda: (_ for _ in ()).throw(RuntimeError("interrupção"))))
    assert not partial.exists()
    print("[PASS] causalidade, prefixo, padding, rótulos e publicação íntegra do cache")


def test_folds() -> None:
    cs = cs_fixtures.official_train_frames()
    partitions = build_subject_fold_partitions(cs)
    for partition in partitions:
        assert not set(partition.development_video_ids) & set(partition.retained_video_ids)
        assert not set(partition.development_subjects) & set(partition.retained_subjects)
        for side, stride in ((DEVELOPMENT_SPLIT, TRAIN_STRIDE), (RETAINED_SPLIT, EVAL_STRIDE)):
            expected = build_window_index(partition.side_frames(side), stride)
            assert window_identity(fold_windows(partition, side)) == window_identity(expected)
    rejects(lambda: build_subject_fold_partitions(cs_fixtures.frames_table()))
    cv = cv_fixtures.official_train_frames()
    for partition in build_fold_partitions(cv):
        assert partition.retained_environment not in partition.development_environments
        assert not set(partition.development_video_ids) & set(partition.retained_video_ids)
    rejects(lambda: build_fold_partitions(cv_fixtures.frames_table()))
    rejects(lambda: validate_output(Path("runs/reference")))
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(configuration, "REPOSITORY_ROOT", Path(directory)):
            (Path(directory) / ".git").write_text("gitdir: disposable", encoding="utf-8")
            rejects(validate_persistent_checkout)
    print("[PASS] dobras e strides existentes, isolamento de sujeitos/ambientes e checkout persistente")


def test_features_and_head() -> None:
    rgb = np.zeros((16, 128, 171, 3), dtype=np.uint8)
    rgb[-1] = 255
    model = SyntheticBackbone().eval()
    dynamic = extraction.embed(model, rgb, "cpu")
    static = extraction.embed(model, np.repeat(rgb[-1:], 16, axis=0), "cpu")
    assert dynamic.shape == static.shape == (FEATURE_DIM,) and not np.array_equal(dynamic, static)
    assert np.array_equal(dynamic, extraction.embed(model, rgb, "cpu"))
    rng = np.random.default_rng(42)
    train = rng.normal(size=(32, FEATURE_DIM)).astype(np.float32)
    retained = rng.normal(size=(16, FEATURE_DIM)).astype(np.float32) + 100
    labels = np.resize(np.array([0, 1, 2, 3, 4, 7, 8, 9]), 32)
    _, _, mean, std = training.normalize(train, retained)
    _, _, changed_mean, changed_std = training.normalize(train, retained * 100)
    assert np.array_equal(mean, changed_mean) and np.array_equal(std, changed_std)
    first = training.fit_head(train, labels, retained, "cpu", 2)
    second = training.fit_head(train, labels, retained, "cpu", 2)
    assert training.checkpoint_digest(first[0]) == training.checkpoint_digest(second[0])
    assert np.array_equal(first[1], second[1]) and np.array_equal(first[2], second[2]) and first[3] == second[3]
    print("[PASS] preprocessing oficial, controle estático, estatística só de treino e repetição do head")


def test_decision() -> None:
    perfect = training.summary(np.array([0, 1, 2, 9]), np.array([0, 1, 2, 9]))
    assert perfect["restricted_macro_f1"] == 0.5
    delta = paired(perfect, perfect)
    rows = []
    for index in range(3):
        vs_a = copy.deepcopy(delta)
        vs_static = copy.deepcopy(delta)
        vs_a["restricted_macro_f1"] = 0.031
        vs_static["restricted_macro_f1"] = 0.011
        rows.append({"fold": str(index), "dynamic_minus_a": vs_a, "dynamic_minus_static": vs_static})
    assert pilot_decision(rows, "cs")["advance"]
    assert not pilot_decision(rows, "cv")["advance"]
    assert not pilot_decision(rows[:2], "cs")["advance"]
    for metric, value in (("restricted_macro_f1", 0.029), ("other_walk_fp", 1)):
        rejected = copy.deepcopy(rows)
        for row in rejected:
            row["dynamic_minus_a"][metric] = value
        assert not pilot_decision(rejected, "cs")["advance"]
    rejected = copy.deepcopy(rows)
    rejected[0]["dynamic_minus_a"]["per_class"]["fall"]["recall"] = -0.051
    assert not pilot_decision(rejected, "cs")["advance"]
    for row in rows:
        row["dynamic_minus_static"]["restricted_macro_f1"] = 0
    assert not pilot_decision(rows, "cs")["advance"]
    print("[PASS] métrica de oito classes, controles e critérios completos de progressão")


def test_orchestration(root: Path) -> None:
    frames = cs_fixtures.official_train_frames()
    partitions = build_subject_fold_partitions(frames)
    context = Context("cs", Le2iDatasetAdapter(), frames, pd.DataFrame(), list(partitions), {}, "synthetic")
    def fake_features(
        _context: Context, _root: Path, windows: pd.DataFrame, _mode: str, **_kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, str]]:
        generator = np.random.default_rng(int(window_identity(windows)[:8], 16) + MODES.index(_mode))
        features = generator.normal(size=(len(windows), FEATURE_DIM)).astype(np.float32)
        return features, {}

    def fake_control(_context: Context, partition: Any, _fold_root: Path) -> dict[str, Any]:
        labels = fold_windows(partition, RETAINED_SPLIT)["label"].to_numpy(dtype=np.int64)
        return training.summary(labels, labels)

    with (patch.object(training, "guard_destination"), patch.object(training, "baseline_control", side_effect=fake_control),
          patch.object(training, "load_features", side_effect=fake_features)):
        for mode in MODES:
            training.run_probe(context, partitions[0], root, root, mode, "cpu")
        report = build_report(context, root, root, "cpu")
        assert len(report["runs"]) == 1 and len(report["missing"]) == 4
        assert not report["pilot_decision"]["advance"]
        assert report["event_metrics"]["fall_event_sensitivity"] is None
        run = root / "cs/runs" / partitions[0].fold / "dynamic/seed_42"
        checkpoint_hash = sha256_file(run / "checkpoint.pt")
        training.run_probe(context, partitions[0], root, root, "dynamic", "cpu", verify_only=True)
        assert sha256_file(run / "checkpoint.pt") == checkpoint_hash
        metrics = json.loads((run / "metrics.json").read_text())
        metrics["retained"]["restricted_macro_f1"] = 1
        (run / "metrics.json").write_text(json.dumps(metrics))
        rejects(lambda: training.run_probe(context, partitions[0], root, root, "dynamic", "cpu", verify_only=True))
    print("[PASS] treino sintético completo, cache de run, relatório parcial e recusa de métricas adulteradas")


def test_resources(root: Path) -> None:
    frames = video_frames()
    manifest = pd.DataFrame([{"video_id": "synthetic/video", "relative_path": "video.avi", "fps": 25,
                              "n_frames_counted": 100, "width": 320, "height": 240}])
    (root / "video.avi").touch()
    context = Context("cs", Le2iDatasetAdapter(raw_dir=root), frames, manifest, [], {}, "synthetic")
    benchmark = {"seconds_per_forward": 0.01, "settings": {"device": "cpu"}}
    with patch.object(extraction, "available_memory_bytes", return_value=16 * 1024**3):
        assert extraction.resource_plan(context, root, benchmark)["feasible"]
        with patch.object(extraction, "MAX_DECODED_BYTES", 1):
            assert not extraction.resource_plan(context, root, benchmark)["feasible"]
        with patch.object(extraction.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()):
            assert not extraction.resource_plan(context, root, benchmark)["feasible"]
        assert not extraction.resource_plan(context, root, {**benchmark, "seconds_per_forward": 1e8})["feasible"]
    print("[PASS] guardas de RAM, disco e orçamento de inferência")


def test_extraction(root: Path) -> None:
    frames = video_frames()
    source = root / "synthetic.avi"
    source.write_bytes(b"synthetic_video_identity")
    weights = root / "synthetic.pth"
    weights.write_bytes(b"synthetic_weight_identity")
    manifest = pd.DataFrame([{"video_id": "synthetic/video", "relative_path": source.name, "fps": 25,
                              "n_frames_counted": 100, "width": 32, "height": 32}])
    context = Context("cs", Le2iDatasetAdapter(raw_dir=root), frames, manifest, [], {}, "synthetic")
    write_json(root / "smoke_cpu.json", {
        "settings": configuration.setup("cpu"), "weight_sha256": sha256_file(weights),
        "contract": contract(), "seconds_per_forward": 0.001,
    })
    calls = []

    def decode(_path: Path, indices: list[int]) -> list[np.ndarray]:
        calls.append(indices)
        return [np.full((32, 32, 3), index, dtype=np.uint8) for index in indices]

    with (patch.object(extraction, "guard_destination"), patch.object(extraction, "weight_path", return_value=weights),
          patch.object(extraction, "load_backbone", return_value=SyntheticBackbone().eval()),
          patch.object(extraction, "decode_frames", side_effect=decode),
          patch.object(extraction, "available_memory_bytes", return_value=16 * 1024**3)):
        extraction.extract(context, root, "cpu")
        assert len(calls) == 1 and max(calls[0]) == 97
        before = sha256_file(cache_path(root / "cs/features", "synthetic/video"))
        extraction.extract(context, root, "cpu")
        assert len(calls) == 1
        windows = build_window_index(frames, TRAIN_STRIDE)
        dynamic, hashes = extraction.load_features(context, root, windows, "dynamic")
        static, _ = extraction.load_features(context, root, windows, "static")
        assert dynamic.shape == static.shape == (10, FEATURE_DIM)
        assert not np.array_equal(dynamic, static) and hashes["synthetic/video"] == before
        rejects(lambda: extraction.load_features(context, root, windows, "dynamic", expected_settings={"device": "changed"}))
        source.write_bytes(b"changed_video_identity")
        rejects(lambda: extraction.extract(context, root, "cpu"))
        assert sha256_file(cache_path(root / "cs/features", "synthetic/video")) == before
    with patch.object(configuration, "OUTPUT_ROOT", root / "outputs"):
        (root / "outputs").symlink_to(root / "cs", target_is_directory=True)
        rejects(lambda: configuration.validate_output(root / "outputs"))
    print("[PASS] extração sintética completa, reuso sem decode, leitura por janela e fonte RGB alterada")


def run_selftests() -> None:
    from gatefall.train.temporal_rgb.control_selftests import run_control_selftests

    run_control_selftests()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_alignment_and_cache(root)
        test_folds()
        test_features_and_head()
        test_decision()
        test_orchestration(root)
        test_resources(root)
        test_extraction(root)
    print("selftest RGB temporal OK; nenhum dado real ou peso baixado")


if __name__ == "__main__":
    run_selftests()
