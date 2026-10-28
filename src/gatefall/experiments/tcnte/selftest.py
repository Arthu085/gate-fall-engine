"""Contratos sintéticos do TCNTE adaptado e da avaliação externa congelada."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gatefall.config import WINDOW_FRAMES
from gatefall.experiments.physics.kinematics import corrected_pose_features
from gatefall.experiments.physics.selftest import fixture_pose
from gatefall.experiments.tcnte.model import (
    BODY_JOINTS,
    MINMAX_EPSILON,
    FocalLoss,
    TCNTEClassifier,
    tcnte_input,
)

LABEL_NAMES = tuple(str(i) for i in range(10))


def _expect(error: type[BaseException], action, message: str) -> None:
    try:
        action()
    except error:
        return
    raise AssertionError(message)


def test_input_representation() -> None:
    pose = fixture_pose(WINDOW_FRAMES)
    pose.person_found[5] = False
    features = torch.from_numpy(corrected_pose_features(pose))[None]
    pixels = pose.keypoints[:, BODY_JOINTS, :2].astype(np.float64)
    low, high = pixels.min(axis=1, keepdims=True), pixels.max(axis=1, keepdims=True)
    scale = high - low
    expected = (pixels - low) / scale
    actual = tcnte_input(features)[0].numpy().reshape(WINDOW_FRAMES, len(BODY_JOINTS), 2)
    observed = pose.person_found
    # O épsilon do min–max é aplicado em unidades normalizadas; a tolerância o cobre.
    tolerance = MINMAX_EPSILON * 50
    assert np.allclose(actual[observed], expected[observed], atol=tolerance)
    assert np.all(actual[~observed] == 0), "quadro sem pessoa deve virar zero"
    print("[PASS] min–max por quadro sobre pose138 equivale ao min–max em pixels; ausência zerada")


def test_model_and_loss() -> None:
    torch.manual_seed(0)
    model = TCNTEClassifier().eval()
    pose = torch.from_numpy(corrected_pose_features(fixture_pose(WINDOW_FRAMES)))[None].repeat(3, 1, 1)
    logits = model(pose, torch.zeros(3, WINDOW_FRAMES, 0), torch.zeros(3, WINDOW_FRAMES, 8))
    assert logits.shape == (3, 10) and torch.isfinite(logits).all()
    _expect(ValueError, lambda: model(pose[:, :-1], torch.zeros(0), torch.zeros(0)), "janela curta aceita")
    target = torch.tensor([0, 1, 2])
    uniform = FocalLoss(torch.ones(10), gamma=0.0)
    assert torch.allclose(uniform(logits, target), F.cross_entropy(logits, target) / 10)
    focal = FocalLoss(torch.ones(10))
    loss = focal(logits.detach().requires_grad_(True), target)
    loss.backward()
    assert torch.isfinite(loss) and float(loss.detach()) <= float(F.cross_entropy(logits, target) / 10)
    _expect(ValueError, lambda: FocalLoss(torch.zeros(10)), "pesos nulos aceitos")
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters < 20000, parameters
    print(f"[PASS] TCNTE dez classes, {parameters} parâmetros, perda focal reduz a CE com γ=0")


def _records(dataset_splits: tuple[str, ...]):
    from gatefall.experiments.physics.data import RawVideo

    records = []
    for split in dataset_splits:
        pose = fixture_pose(32)
        labels = np.zeros(pose.k, dtype=np.int64)
        labels[12:20] = 1
        labels[20:28] = 2
        records.append(RawVideo(f"room/{split}", split, labels, pose,
                                np.zeros((pose.k, 0), dtype=np.float32), np.zeros(pose.k, dtype=np.float32)))
    return records


def test_training_and_external_evaluation() -> None:
    from gatefall.experiments.physics.data import prepare_records
    from gatefall.experiments.physics.training import evaluate, evaluate_external, train

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="gatefall-tcnte-") as temporary:
        root = Path(temporary)
        source = prepare_records(_records(("train", "val", "test")), root / "source", dataset="synthetic-src",
                                 visual_kind="none", label_names=LABEL_NAMES, sources={})
        external = prepare_records(_records(("test",)), root / "external", dataset="synthetic-ext",
                                   visual_kind="none", label_names=LABEL_NAMES, sources={},
                                   required_splits=("test",))
        _expect(ValueError, lambda: train(source, root / "bad", arm="TCNTE", fusion="pose", epochs=1),
                "TCNTE com fusão errada aceito")
        _expect(ValueError, lambda: train(source, root / "bad", arm="A", fusion="tcnte", epochs=1),
                "fusão tcnte em outro braço aceita")
        run = root / "runs" / "tcnte"
        metrics = train(source, run, arm="TCNTE", fusion="tcnte", epochs=2, batch_size=8)
        config = (run / "config.json").read_text()
        assert '"loss": "focal_gamma3"' in config and '"lr": 0.0001' in config
        assert metrics["timing"]["train_seconds"] > 0 and len(metrics["timing"]["epoch_seconds"]) == 2
        report = evaluate(source, run, "test")
        assert report["events"]["n_fall_events"] == 1 and len(report["predictions"]) == 32
        assert len(report["alarms"]) == report["events"]["n_alarms_total"]
        transfer = evaluate_external(source, run, external)
        assert transfer["adaptation"] == "none" and transfer["target_dataset"] == "synthetic-ext"
        assert transfer["timing"]["windows"] == 32
        _expect(FileExistsError, lambda: evaluate_external(source, run, external), "avaliação externa sobrescrita")
        _expect(ValueError, lambda: evaluate_external(source, run, source), "pacote de treino aceito como externo")
        pose_run = root / "runs" / "pose"
        train(source, pose_run, arm="A", fusion="pose", epochs=1, batch_size=8)
        evaluate_external(source, pose_run, external)
    print("[PASS] treino/avaliação TCNTE e transferência congelada sem adaptação nem sobrescrita")


def run_selftest() -> None:
    test_input_representation()
    test_model_and_loss()
    test_training_and_external_evaluation()
    print("tcnte selftest PASS")


if __name__ == "__main__":
    run_selftest()
