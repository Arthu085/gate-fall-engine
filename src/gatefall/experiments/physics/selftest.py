"""Contratos analíticos, causais e de integração; sem dados reais."""

import numpy as np
import torch
import tempfile
from pathlib import Path

from gatefall.pose.loading import PoseArrays

from gatefall.experiments.physics.kinematics import observed_derivatives


def test_kinematics() -> None:
    times = np.arange(9, dtype=np.float64) / 10
    found = np.array([False, True, True, False, True, True, False, True, True])
    for acceleration in (0.0, 2.0):
        x = (3 * times + .5 * acceleration * times**2)[:, None]
        v, a, vv, av = observed_derivatives(x, found, times, max_gap_seconds=.5)
        assert np.allclose(a[av], acceleration, atol=1e-4)
        assert not vv[0] and not vv[1] and not av[2]
        assert np.all(v[~vv] == 0) and np.all(a[~av] == 0)
        for end in range(1, len(times) + 1):
            prefix = observed_derivatives(x[:end], found[:end], times[:end], max_gap_seconds=.5)
            for full, part in zip((v, a, vv, av), prefix):
                assert np.array_equal(full[:end], part)
    _, _, vv, av = observed_derivatives(times[:, None], found, times, max_gap_seconds=.15)
    assert not vv[4] and not av[5]
    for presence in (np.zeros(9, dtype=bool), np.arange(9) == 4):
        v, a, _, _ = observed_derivatives(times[:, None], presence, times, max_gap_seconds=.5)
        assert not v.any() and not a.any()
    print("[PASS] derivadas válidas, aquisição, gaps e causalidade")


def run_selftest() -> None:
    test_kinematics()
    test_consistency_and_gate()
    test_corruption_and_loss()
    test_latency_precision()
    test_source_preparation()
    test_dinov3_source_preparation()
    test_training_pipeline()


def fixture_pose(k: int = 32) -> PoseArrays:
    rng = np.random.default_rng(14)
    base = rng.uniform([60, 40], [120, 160], (17, 2))
    points = np.zeros((k, 17, 3), dtype=np.float32)
    boxes = np.zeros((k, 4), dtype=np.float32)
    for i in range(k):
        angle = i * .04
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points[i, :, :2] = base @ rotation + np.array([i, i * i * .15])
        points[i, :, 2] = .95
        boxes[i, :2] = points[i, :, :2].min(axis=0) - 2
        boxes[i, 2:] = points[i, :, :2].max(axis=0) + 2
    return PoseArrays(points, boxes, np.ones(k, dtype=bool), k, 320, 240)


def test_consistency_and_gate() -> None:
    from gatefall.experiments.physics.consistency import pose_residual, sam3_residual
    from gatefall.experiments.physics.model import EvidenceGate
    pose = fixture_pose()
    r, valid = pose_residual(pose, .5)
    assert valid[1:].all() and not valid[0] and r.max() < 1e-5
    bad = fixture_pose()
    bad.keypoints[10, 0, 0] += 100
    rb, _ = pose_residual(bad, .5)
    assert rb[10] > .03
    for end in (1, 10, 11, 20):
        prefix = PoseArrays(bad.keypoints[:end], bad.bbox[:end], bad.person_found[:end], end, 320, 240)
        assert np.allclose(pose_residual(prefix, .5)[0], rb[:end])
    masks = np.zeros((20, 10), dtype=np.float32)
    masks[:, 0] = 1; masks[:, 1] = .2; masks[:, 6] = .8
    assert sam3_residual(masks, .5)[0].max() == 0
    masks[10, 1] = .01
    assert sam3_residual(masks, .5)[0][10] > 0
    gate = EvidenceGate("physics")
    e = torch.full((2, 24, 8), .5); e[:, :, 4:] = 1
    original = gate(e)
    for channel, direction in [(0, 1), (1, -1), (2, -1), (3, 1)]:
        changed = e.clone(); changed[:, :, channel] += .2
        assert torch.all(direction * (gate(changed) - original) > 0)
    assert torch.all((original > 0) & (original < 1))
    e[:, :, 5] = 0
    changed = e.clone(); changed[:, :, 3] = 1
    assert torch.equal(gate(e), gate(changed))
    with torch.no_grad():
        gate.raw_weights.copy_(torch.tensor([-100.0, 100.0, -10.0, 10.0]))
        gate.bias.fill_(100)
    saturated = gate(e)
    assert torch.isfinite(saturated).all() and torch.all((saturated >= 0) & (saturated <= 1))
    assert torch.all(torch.nn.functional.softplus(gate.raw_weights) >= 0)
    print("[PASS] consistência causal, movimento global e sinais do gate")


def test_corruption_and_loss() -> None:
    from gatefall.experiments.physics.data import RawVideo, build_variants
    from gatefall.experiments.physics.model import EvidenceGate, ordering_loss
    pose = fixture_pose()
    visual = np.zeros((pose.k, 10), dtype=np.float32)
    visual[:, 0] = 1; visual[:, 1] = .2; visual[:, 6] = .8
    record = RawVideo("room/video", "train", np.ones(pose.k, dtype=np.int64), pose, visual,
                      np.full(pose.k, .9, dtype=np.float32))
    first = build_variants(record, "sam3", 42, .5, .25, 3)
    second = build_variants(record, "sam3", 42, .5, .25, 3)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert first['pose_changed'].any() and first['visual_changed'].any()
    assert np.array_equal(first['evidence'][:, [1, 3, 5, 7]],
                          first['evidence_pose_corrupt'][:, [1, 3, 5, 7]])
    assert np.array_equal(first['evidence'][:, [0, 2, 4, 6]],
                          first['evidence_visual_corrupt'][:, [0, 2, 4, 6]])
    gate = EvidenceGate("physics")
    e = torch.full((1, 24, 8), .5); e[:, :, 4:] = 1
    ep = e.clone(); ep[:, :, 0] -= .01
    ev = e.clone(); ev[:, :, 1] -= .01
    mask = torch.ones((1, 24), dtype=torch.bool)
    loss = ordering_loss(gate(e), gate(ep), gate(ev), mask, mask, .05)
    assert loss.item() > 0
    loss.backward()
    assert gate.raw_weights.grad is not None
    assert torch.isfinite(gate.raw_weights.grad).all() and gate.raw_weights.grad.abs().sum() > 0
    assert ordering_loss(gate(e), gate(ep), gate(ev), ~mask, ~mask, .05).item() == 0
    for margin in (0, 1, float("nan")):
        try:
            ordering_loss(gate(e), gate(ep), gate(ev), mask, mask, margin)
        except ValueError:
            pass
        else:
            raise AssertionError("margem inválida aceita")
    print("[PASS] corrupção reproduzível e perda de margem com gradiente")


def test_latency_precision() -> None:
    from dataclasses import replace
    from gatefall.eval.shared.events import FallEvent, Alarm, associate_events_and_alarms, split_event_report
    from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
    event = FallEvent("v", 0.0, 2.0, None, None, 2.0, False)
    alarm = Alarm("v", 1, .123456789)
    rounded = associate_events_and_alarms([event], [alarm], BASELINE_A_ALARM_PROTOCOL)[0][0]
    exact = associate_events_and_alarms([event], [alarm], BASELINE_A_ALARM_PROTOCOL,
                                      round_latency=False)[0][0]
    assert rounded.latency_s == round(alarm.trigger_time_s, BASELINE_A_ALARM_PROTOCOL.latency_decimal_places)
    assert exact.latency_s == alarm.trigger_time_s and rounded.latency_s != exact.latency_s
    # Média de latências diferentes não deve ser arredondada para uma casa.
    protocol = replace(BASELINE_A_ALARM_PROTOCOL, trigger_consecutive=1)
    report = split_event_report(["a", "a", "b", "b"], [0, 1, 0, 1], [1, 1, 1, 1],
                                [1, 0, 0, 1], protocol, 4, 4, 4, round_latency=False)
    assert abs(report["latency_seconds"]["mean"] - .05) < 1e-12
    print("[PASS] precisão de latência opt-in e comportamento legado")


def test_training_pipeline() -> None:
    from gatefall.experiments.physics.data import RawVideo, prepare_records, load_bundle
    from gatefall.experiments.physics.training import train, evaluate, checkpoint_is_better, fit_stats
    assert checkpoint_is_better(.7, .6, "best_val")
    assert not checkpoint_is_better(.7, .7, "best_val")
    assert checkpoint_is_better(.1, .9, "last")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); records = []
        for split in ("train", "val", "test"):
            pose = fixture_pose(32)
            visual = np.zeros((pose.k, 10), dtype=np.float32)
            visual[:, 0] = 1; visual[:, 1] = .2; visual[:, 6] = .8
            labels = np.zeros(pose.k, dtype=np.int64); labels[12:20] = 1; labels[20:28] = 2
            records.append(RawVideo(split, split, labels, pose, visual, np.full(pose.k, .9, dtype=np.float32)))
        bundle = prepare_records(records, root / "bundle", dataset="synthetic", visual_kind="sam3",
                                 label_names=tuple(str(i) for i in range(10)), sources={}, outage_probability=.2)
        manifest, contents = load_bundle(bundle)
        assert "pose_corrupt" not in contents["test"]
        stats = fit_stats(manifest, contents)
        changed = {identity: {name: value.copy() for name, value in data.items()}
                   for identity, data in contents.items()}
        changed["test"]["pose"] *= 1000
        changed["test"]["visual"] += 1000
        changed["test"]["labels"][:] = 9
        assert fit_stats(manifest, changed) == stats
        for fusion in ("pose", "concat", "constant", "quality_concat", "learned", "monotonic", "physics"):
            arm = "A" if fusion == "pose" else "C0" if fusion == "concat" else "C1"
            run = root / fusion
            metrics = train(bundle, run, arm=arm, fusion=fusion, epochs=2, batch_size=8,
                            augment=True, order_weight=.1 if fusion in ("learned", "monotonic", "physics") else 0)
            assert 1 <= metrics["selected_epoch"] <= 2
            assert (run / "last.pt").is_file() and (run / "checkpoint.pt").is_file()
            report = evaluate(bundle, run, "test")
            assert report["events"]["n_fall_events"] == 1
            assert report["events"]["total_windows"] == 32
            assert "test" not in metrics["final"]
            assert len(report["predictions"]) == 32
        for kind, dimension, recipes in (("none", 0, (("A", "pose"),)),
                                         ("dinov3", 1536, (("B0", "concat"), ("B1", "physics")))):
            variants = [RawVideo(r.video_id, r.split, r.labels, r.pose,
                                np.ones((r.pose.k, dimension), dtype=np.float32),
                                np.full(r.pose.k, .9, dtype=np.float32)) for r in records]
            other_bundle = prepare_records(variants, root / kind, dataset="synthetic", visual_kind=kind,
                                            label_names=tuple(str(i) for i in range(10)), sources={})
            for arm, fusion in recipes:
                run = root / f"{kind}-{arm}"
                train(other_bundle, run, arm=arm, fusion=fusion, epochs=1, batch_size=8)
                assert evaluate(other_bundle, run)["events"]["total_windows"] == 32
        # Uma fonte modificada deve ser recusada antes de qualquer treino.
        import h5py
        with h5py.File(bundle / manifest["videos"][0]["path"], "r+") as handle:
            handle.attrs["tampered"] = True
        try:
            load_bundle(bundle)
        except ValueError as error:
            assert "hash" in str(error)
        else:
            raise AssertionError("pacote alterado foi aceito")
        print("[PASS] sete controles, cinco braços, fontes SAM/DINO/pose, isolamento e integridade")


def _pose_world(root: Path):
    import h5py
    import pandas as pd

    rows = []
    for split in ("train", "val", "test"):
        pose = fixture_pose(8)
        identity = "room/" + split
        path = root / "pose" / "room" / f"{split}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            for name in ("keypoints", "bbox", "person_found"):
                handle.create_dataset(name, data=getattr(pose, name))
            for name, value in {"video_id": identity, "split": split, "K": pose.k,
                                "width": pose.width, "height": pose.height}.items():
                handle.attrs[name] = value
        rows.extend({"video_id": identity, "split": split, "frame_index": i, "label": 0}
                    for i in range(pose.k))
    frames = pd.DataFrame(rows)
    frames_path = root / "frames.parquet"; frames.to_parquet(frames_path)
    return frames, frames_path


def test_source_preparation() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch
    from gatefall.experiments.physics.data import prepare_dataset, source_attrs, load_bundle
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frames, frames_path = _pose_world(root)
        adapter = SimpleNamespace(identifier="le2i", frames_path=frames_path, pose_root=root / "pose",
                                  load_frames=lambda: frames, label_names=tuple(str(i) for i in range(10)))
        with patch("gatefall.datasets.get_dataset", return_value=adapter):
            bundle = prepare_dataset("le2i", "none", root / "bundle")
        manifest, _ = load_bundle(bundle)
        assert len(manifest["sources"]["files"]) == 3
        try:
            source_attrs(root / "pose/room/train.h5", "room/test", 8, "train")
        except ValueError:
            pass
        else:
            raise AssertionError("identidade divergente aceita")
    print("[PASS] preparação por adapter e validação da identidade HDF5")


def test_dinov3_source_preparation() -> None:
    import json
    from types import SimpleNamespace
    from unittest.mock import patch
    import h5py
    from gatefall.config import TARGET_FPS
    from gatefall.dinov3.extract import NORMALIZE_MEAN, NORMALIZE_STD
    from gatefall.dinov3.storage import dinov3_path, write_dinov3_features_atomic
    from gatefall.experiments.physics.data import json_compatible, load_bundle, prepare_dataset
    from gatefall.features.quality_extract import current_provenance
    from gatefall.features.quality_storage import QUALITY_CHANNELS, quality_path, write_quality_atomic
    from gatefall.hashing import sha256_file
    from gatefall.pose.loading import pose_path

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frames, frames_path = _pose_world(root)
        adapter = SimpleNamespace(identifier="le2i", frames_path=frames_path, pose_root=root / "pose",
                                  dinov3_root=root / "dinov3", quality_root=root / "quality",
                                  load_frames=lambda: frames, label_names=tuple(str(i) for i in range(10)))
        # Mesmos tipos do extrator real: o HDF5 devolve int64, float64 e arrays NumPy.
        provenance = {"model_name": "dinov3_vitb16", "feature_dim": 1536, "weights_sha256": "a" * 64,
                      "dinov3_repo_commit": 1j, "resize_height": 224, "resize_width": 224,
                      "normalize_mean": np.array(NORMALIZE_MEAN, dtype=np.float64),
                      "normalize_std": np.array(NORMALIZE_STD, dtype=np.float64), "target_fps": TARGET_FPS}
        rng = np.random.default_rng(3)
        for identity, group in frames.groupby("video_id"):
            identity, k, split = str(identity), len(group), str(group["split"].iloc[0])
            identity_attrs = {"video_id": identity, "K": k, "split": split}
            write_dinov3_features_atomic(dinov3_path(identity, dinov3_root=adapter.dinov3_root),
                                         rng.normal(size=(k, 1536)).astype(np.float16),
                                         {**identity_attrs, **provenance})
            digest = sha256_file(pose_path(identity, pose_root=adapter.pose_root))
            write_quality_atomic(quality_path(identity, quality_root=adapter.quality_root),
                                 rng.uniform(size=(k, QUALITY_CHANNELS)).astype(np.float32),
                                 {**identity_attrs, **current_provenance(pose_source_digest=digest)})
        with h5py.File(dinov3_path("room/train", dinov3_root=adapter.dinov3_root), "r") as handle:
            assert isinstance(handle.attrs["feature_dim"], np.integer), "fixture sem inteiro NumPy"
            assert isinstance(handle.attrs["normalize_mean"], np.ndarray), "fixture sem array NumPy"

        def leftovers() -> list[Path]:
            return sorted(root.glob(".physics-prepare-*"))

        with patch("gatefall.datasets.get_dataset", return_value=adapter):
            try:
                prepare_dataset("le2i", "dinov3", root / "rejected")
            except TypeError as error:
                assert "complex" in str(error)
            else:
                raise AssertionError("metadado sem representação JSON exata aceito")
            assert not (root / "rejected").exists() and not leftovers(), "falha deixou pacote parcial"
            for identity in ("room/train", "room/val", "room/test"):
                with h5py.File(dinov3_path(identity, dinov3_root=adapter.dinov3_root), "r+") as handle:
                    handle.attrs["dinov3_repo_commit"] = "b" * 40
            bundle = prepare_dataset("le2i", "dinov3", root / "bundle")
            published = (bundle / "manifest.json").read_bytes()
            try:
                prepare_dataset("le2i", "dinov3", root / "bundle")
            except FileExistsError:
                pass
            else:
                raise AssertionError("pacote existente sobrescrito")
        assert (bundle / "manifest.json").read_bytes() == published and not leftovers()
        manifest, videos = load_bundle(bundle)
        recorded = manifest["sources"]["dinov3_extractor"]
        expected = {**provenance, "dinov3_repo_commit": "b" * 40,
                    "normalize_mean": list(NORMALIZE_MEAN), "normalize_std": list(NORMALIZE_STD)}
        assert recorded == expected, recorded
        assert all(type(recorded[name]) is int for name in ("feature_dim", "resize_height", "resize_width"))
        assert all(type(value) is float for value in recorded["normalize_mean"] + recorded["normalize_std"])
        assert json.loads(json.dumps(recorded)) == recorded and len(videos) == 3
        assert len(manifest["sources"]["files"]["room/train"]) == 3
        for unsupported in (np.complex64(1j), np.array([1j]), {1: "x"}, object()):
            try:
                json_compatible(unsupported)
            except TypeError:
                continue
            raise AssertionError(f"metadado não suportado aceito: {unsupported!r}")
    print("[PASS] pacote dinov3: proveniência HDF5 NumPy convertida sem perda, publicação atômica e sem sobrescrita")


if __name__ == "__main__":
    run_selftest()
