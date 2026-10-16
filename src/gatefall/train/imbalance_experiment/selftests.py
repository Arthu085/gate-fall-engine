"""Checagens sintéticas da perda, do pareamento e da separação de saídas."""

import numpy as np
import torch
from torch import nn
from typing import cast

from gatefall.train.imbalance_experiment.cli import OUTPUT_ROOT, _output_root, run_dir
from gatefall.train.imbalance_experiment.report import _paired
from gatefall.train.shared.imbalance import (
    RECIPES,
    ImbalanceRecipe,
    class_weights,
    training_loss,
    validate_imbalance_run_dir,
)


def run_selftests() -> bool:
    labels = np.array([0, 0, 0, 9], dtype=np.int64)
    baseline = class_weights(labels, 10, RECIPES["baseline"])
    soft = class_weights(labels, 10, RECIPES["soft"])
    assert torch.isclose(baseline[0], torch.tensor(1 / 3))
    assert torch.isclose(soft[0], torch.tensor(float(1 / np.sqrt(3))))
    assert baseline[9] == soft[9] == 1
    assert torch.count_nonzero(baseline).item() == 2
    try:
        class_weights(np.array([0, 10]), 10, RECIPES["soft"])
    except ValueError:
        pass
    else:
        raise AssertionError("rótulo inválido aceito")
    try:
        ImbalanceRecipe("invalid", 0.5, 0.75)
    except ValueError:
        pass
    else:
        raise AssertionError("intensidade acima do limite aceita")

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    targets = torch.tensor([1, 0])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.5]))
    base = criterion(logits, targets)
    hard = training_loss(logits, targets, criterion, RECIPES["hard"], 1, 0)
    assert torch.isclose(hard, 1.5 * base)
    hard.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    correct = training_loss(-logits.detach(), targets, criterion, RECIPES["hard"], 1, 0)
    assert torch.isclose(correct, criterion(-logits.detach(), targets))
    assert torch.isclose(training_loss(logits.detach(), targets, criterion, RECIPES["baseline"], 1, 0), base)
    unrelated = torch.tensor([[0.0, 3.0, 1.0]])
    unrelated_target = torch.tensor([2])
    assert torch.isclose(
        training_loss(unrelated, unrelated_target, nn.CrossEntropyLoss(), RECIPES["hard"], 2, 0),
        nn.CrossEntropyLoss()(unrelated, unrelated_target),
    )

    metadata = {
        "protocol": "cs", "fold": "retain_s0", "membership_sha256": "0" * 64,
        "input_sha256": {"frames": "1" * 64}, "standardization_stats_sha256": "2" * 64,
        "development_videos": ["train"], "retained_videos": ["retained"],
        "seed": 42, "epochs": 30, "checkpoint_policy": "last_epoch",
    }
    classes = {name: {"support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0, "fp": 0, "fn": 0}
               for name in ("walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
                            "stand_up", "standing", "other")}
    control = {"provenance": metadata, "restricted_macro_f1": 0.5, "per_class": classes}
    candidate = {**control, "restricted_macro_f1": 0.6}
    assert abs(cast(float, _paired(control, candidate)["restricted_macro_f1"]) - 0.1) < 1e-12
    try:
        _paired(control, {**candidate, "provenance": {**metadata, "seed": 43}})
    except RuntimeError:
        pass
    else:
        raise AssertionError("seeds diferentes foram pareadas")
    try:
        _output_root(run_dir(OUTPUT_ROOT.parent, "cs", "retain_s0", "soft", 42))
    except ValueError:
        pass
    else:
        raise AssertionError("saída fora da raiz experimental foi aceita")
    try:
        validate_imbalance_run_dir(OUTPUT_ROOT.parent / "baseline_a")
    except ValueError:
        pass
    else:
        raise AssertionError("receita experimental aceita run canônico")
    print("[PASS] pesos, perda limitada, gradiente, pareamento e isolamento de saída")
    return True


if __name__ == "__main__":
    run_selftests()
