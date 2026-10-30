"""Verificações do experimento: contratos reais, sem medir eficácia em quedas."""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from gatefall.config import TARGET_FPS
from gatefall.experiments.motion import RECIPES
from gatefall.experiments.motion.data import build_targets, load_targets, prepare
from gatefall.experiments.motion.learning import MotionClassifier, motion_losses
from gatefall.experiments.motion.report import summarize
from gatefall.experiments.physics.data import RawVideo, load_bundle, prepare_records
from gatefall.experiments.physics.model import FusionClassifier
from gatefall.experiments.physics.selftest import fixture_pose
from gatefall.experiments.physics.training import Windows, evaluate, fit_stats, train
from gatefall.hashing import sha256_file
from gatefall.pose.loading import pose_path


def expect_error(action) -> None:
    try:
        action()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("contrato inválido foi aceito")


def test_targets_and_losses() -> None:
    pose = fixture_pose(12)
    for k in range(pose.k):
        pose.keypoints[k, :, 0] = 50 + 2 * k
        pose.keypoints[k, :, 1] = 60 + k
    targets = build_targets(pose, [1, 3], .5)
    assert np.allclose(targets["velocity"][:-3, 1, :, 0], 20 / 400)
    assert not targets["valid"][-3:, 1].any()
    pose.person_found[4] = False
    pose.keypoints[6, 5, 2] = .1
    masked = build_targets(pose, [1, 3], .5)
    assert not masked["valid"][4].any()
    assert not masked["valid"][3, 1, 0]
    assert not masked["valid"][5, 0, 0]
    assert masked["valid"][5, 0, 1:].all()
    assert not build_targets(pose, [20], .5)["valid"].any()
    expect_error(lambda: build_targets(pose, [0], .5))
    expect_error(lambda: build_targets(pose, [3, 1], .5))
    batch = {"motion_" + key: torch.from_numpy(value[:2]) for key, value in targets.items()}
    intervals = torch.tensor([1, 3]) / TARGET_FPS
    position = batch["motion_future"].clone().requires_grad_()
    velocity = batch["motion_velocity"].clone().requires_grad_()
    supervised, physical = motion_losses(position, velocity, batch, intervals)
    assert supervised == 0 and physical < 1e-10
    wrong = (position.detach() + .1).requires_grad_()
    _, physical = motion_losses(wrong, velocity, batch, intervals)
    physical.backward()
    assert physical > 0 and wrong.grad is not None and wrong.grad.abs().sum() > 0
    assert velocity.grad is not None and velocity.grad.abs().sum() > 0
    batch["motion_valid"][:] = False
    supervised, physical = motion_losses(position, velocity, batch, intervals)
    assert supervised == 0 and physical == 0
    (supervised + physical).backward()
    print("[PASS] unidades, horizontes, máscaras, coerência, contradição e batch vazio")


def test_model() -> None:
    for fusion, dimension in (("pose", 0), ("concat", 10), ("learned", 1536), ("physics", 10)):
        torch.manual_seed(42)
        reference = FusionClassifier(dimension, fusion)
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        model = MotionClassifier(dimension, fusion, [1, 3])
        assert torch.equal(reference_rng, torch.get_rng_state())
        for name, value in reference.state_dict().items():
            assert torch.equal(value, model.state_dict()[name])
        pose = torch.randn(2, 24, 138)
        visual = torch.randn(2, 24, dimension)
        evidence = torch.full((2, 24, 8), .8)
        model.eval(); reference.eval()
        logits, position, velocity = model.forward_with_motion(pose, visual, evidence)
        assert torch.equal(logits, reference(pose, visual, evidence))
        batch = {"motion_current": torch.zeros(2, 4, 2), "motion_future": torch.ones(2, 2, 4, 2),
                 "motion_velocity": torch.zeros(2, 2, 4, 2),
                 "motion_valid": torch.ones(2, 2, 4, dtype=torch.bool)}
        _, physical = motion_losses(position, velocity, batch, torch.tensor([.1, .3]))
        physical.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        assert model.position_head.weight.grad is not None and model.position_head.weight.grad.abs().sum() > 0
        assert model.velocity_head.weight.grad is not None and model.velocity_head.weight.grad.abs().sum() > 0
        with torch.no_grad():
            model.position_head.weight.add_(1000)
            model.velocity_head.weight.add_(1000)
        assert torch.equal(logits, model(pose, visual, evidence))
    print("[PASS] inicialização/RNG preservados, gradientes na TCN e logits independentes dos cabeçotes")


def fixture(root: Path, kind: str = "sam3") -> tuple[Path, Path]:
    records = []
    sources: dict = {"files": {}}
    pose_root = root / "poses"
    for split in ("train", "val", "test"):
        pose = fixture_pose(32)
        identity = "room/" + split
        path = pose_path(identity, pose_root=pose_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            for name in ("keypoints", "bbox", "person_found"):
                handle.create_dataset(name, data=getattr(pose, name))
            for name, value in {"video_id": identity, "split": split, "K": pose.k,
                                "width": pose.width, "height": pose.height}.items():
                handle.attrs[name] = value
        sources["files"][identity] = {str(path): sha256_file(path)}
        dim = {"sam3": 10, "none": 0, "dinov3": 1536}[kind]
        visual = np.full((pose.k, dim), .2, dtype=np.float32)
        if kind == "sam3":
            visual[:, 0] = 1
            visual[:, 6] = .8
        labels = np.zeros(pose.k, dtype=np.int64)
        labels[12:20] = 1
        labels[20:28] = 2
        records.append(RawVideo(identity, split, labels, pose, visual,
                                np.full(pose.k, .9, dtype=np.float32)))
    bundle = prepare_records(records, root / "bundle", dataset="synthetic", visual_kind=kind,
                             label_names=tuple(str(i) for i in range(10)), sources=sources)
    targets = prepare(bundle, pose_root, root / "targets")
    return bundle, targets


def test_pipeline() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle, targets = fixture(root)
        manifest, videos = load_bundle(bundle)
        _, arrays, coverage = load_targets(targets, bundle, manifest, videos)
        assert set(arrays) == {"room/train"} and coverage["total_valid"] > 0
        stats = fit_stats(manifest, videos)
        plain = Windows(manifest, videos, stats, "train", 1)
        with_targets = Windows(manifest, videos, stats, "train", 1, motion_targets=arrays)
        assert plain.index == with_targets.index
        for index in range(len(plain)):
            a, b = plain[index], with_targets[index]
            assert all(torch.equal(value, b[key]) for key, value in a.items())
        assert not with_targets[len(with_targets)-1]["motion_valid"].any()
        assert not any(key.startswith("motion_") for key in plain[0])
        # Alterar futuro só muda a supervisão: entradas e logits não recebem alvos.
        arrays["room/train"]["future"] += 100
        assert torch.equal(plain[0]["pose"], with_targets[0]["pose"])
        _, arrays, _ = load_targets(targets, bundle, manifest, videos)
        runs = []
        for recipe in RECIPES:
            run = root / recipe
            metrics = train(bundle, run, arm="C0", fusion="concat", epochs=2, batch_size=8,
                            motion_recipe=recipe, targets=targets)
            assert "test" not in metrics["final"]
            assert all(np.isfinite(item["train_loss"]) for item in metrics["history"])
            if recipe == "baseline":
                assert metrics["history"][0]["motion_loss"] == 0
            else:
                assert metrics["history"][0]["motion_loss"] > 0
            runs.append(run)
        # A inferência funciona mesmo quando o pacote auxiliar não está disponível.
        hidden = root / "hidden-targets"
        targets.rename(hidden)
        for run in runs:
            report = evaluate(bundle, run)
            assert report["events"]["total_windows"] == 32
        hidden.rename(targets)
        report = summarize(runs, root / "summary")
        assert len(report["paired_deltas"]) == 3
        expect_error(lambda: summarize(runs[:-1], root / "incomplete"))
        expect_error(lambda: summarize(runs + runs[:1], root / "duplicate"))
        # Avaliações e ambientes heterogêneos não formam uma comparação controlada.
        for filename, field in (("metrics.json", "device"), ("metrics.json", "torch_version"),
                                ("metrics.json", "numpy_version"),
                                ("evaluation-test.json", "latency_precision"),
                                ("evaluation-test.json", "code_provenance")):
            path = runs[0] / filename
            original = path.read_bytes()
            value = json.loads(original)
            if field == "code_provenance":
                value[field]["python_sources_sha256"]["changed.py"] = "different"
            else:
                value[field] = "different"
            path.write_text(json.dumps(value), encoding="utf-8")
            expect_error(lambda: summarize(runs, root / "mixed"))
            path.write_bytes(original)
        # Controle opt-in deve reproduzir o treino original com a mesma seed.
        train(bundle, root / "original", arm="C0", fusion="concat", epochs=2, batch_size=8)
        a = torch.load(root / "original/checkpoint.pt", weights_only=True)
        b = torch.load(root / "baseline/checkpoint.pt", weights_only=True)
        assert all(torch.equal(value, b[key]) for key, value in a.items())
        # Integração em todos os braços, mantendo augmentation e order opcionais.
        for kind, arms in (("none", [("A", "pose")]),
                           ("dinov3", [("B0", "concat"), ("B1", "physics")]),
                           ("sam3", [("C1", "physics")])):
            other, extra = fixture(root / kind, kind)
            for arm, fusion in arms:
                for recipe in RECIPES:
                    run = root / f"{arm}-{recipe}"
                    train(other, run, arm=arm, fusion=fusion, epochs=1, batch_size=8,
                          motion_recipe=recipe, targets=extra, augment=True,
                          order_weight=.1 if fusion == "physics" else 0)
                    assert evaluate(other, run)["events"]["total_windows"] == 32
        # Adulteração de sidecar e pose original precisa ser detectada.
        meta = json.loads((targets / "motion-manifest.json").read_text())
        path = targets / meta["videos"][0]["path"]
        with path.open("ab") as handle:
            handle.write(b"tampered")
        expect_error(lambda: load_targets(targets, bundle, manifest, videos))
        raw = pose_path("room/train", pose_root=root / "poses")
        with h5py.File(raw, "r+") as handle:
            handle.attrs["tampered"] = True
        expect_error(lambda: prepare(bundle, root / "poses", root / "bad-targets"))
        print("[PASS] três receitas/cinco braços, treino original idêntico, "
              "inferência sem alvos, pareamento e hashes")


def run_selftest() -> None:
    torch.set_num_threads(1)
    test_targets_and_losses()
    test_model()
    test_pipeline()
    print("Motion selftest PASS. Isto não mede melhoria em dados reais.")
