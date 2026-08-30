"""Loop de treino e avaliação da TCN sobre janelas de pose padronizadas."""

import json
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gatefall.data.le2i.pose_dataset import LABEL_NAMES
from gatefall.features.standardization import StandardizationStats, apply_standardization
from gatefall.train.config import TrainConfig, save_config
from gatefall.train.metrics import RESTRICTED_CLASSES, restricted_macro_f1, support
from gatefall.train.tcn import TCNClassifier


class _WindowSource(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[np.ndarray, int, object]: ...


class _StandardizedTorchDataset(Dataset):
    def __init__(self, source: _WindowSource, stats: StandardizationStats) -> None:
        self._source = source
        self._stats = stats

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        window, label, _diag = self._source[index]
        standardized = apply_standardization(window, self._stats)
        return torch.from_numpy(standardized), label


def _collect_labels(source: _WindowSource) -> np.ndarray:
    labels = np.empty(len(source), dtype=np.int64)
    for i in range(len(source)):
        _, label, _diag = source[i]
        labels[i] = label
    return labels


def _class_weights(train_labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = support(train_labels, num_classes)
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = 1.0 / counts[c]
    return torch.from_numpy(weights)


@torch.no_grad()
def _predict(model: TCNClassifier, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred)


def _evaluate_split(model: TCNClassifier, loader: DataLoader, device: str, num_classes: int) -> dict:
    y_true, y_pred = _predict(model, loader, device)
    macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred, num_classes)
    split_support = support(y_true, num_classes)
    return {
        "macro_f1_restricted": macro_f1,
        "f1_by_class": {str(c): f1_by_class[c] for c in RESTRICTED_CLASSES},
        "support": {LABEL_NAMES[c]: split_support[c] for c in range(num_classes)},
    }


def run_training(
    input_dim: int,
    train_source: _WindowSource,
    val_source: _WindowSource,
    test_source: _WindowSource,
    stats: StandardizationStats,
    config: TrainConfig,
    run_dir: Path,
    force: bool,
) -> dict:
    config_path = run_dir / "config.yaml"
    if config_path.exists() and not force:
        print(f"skip {run_dir} (já existe, use --force para sobrescrever)")
        sys.exit(1)

    save_config(config, config_path, force=True)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = _StandardizedTorchDataset(train_source, stats)
    val_dataset = _StandardizedTorchDataset(val_source, stats)
    test_dataset = _StandardizedTorchDataset(test_source, stats)

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = TCNClassifier(
        input_dim=input_dim,
        channels=config.channels,
        kernel_size=config.kernel_size,
        dilations=config.dilations,
        dropout=config.dropout,
        num_classes=config.num_classes,
    ).to(device)

    train_labels = _collect_labels(train_source)
    if config.class_weighted:
        weights = _class_weights(train_labels, config.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history: list[dict] = []
    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        n_examples = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            loss_sum += float(loss.item()) * x.size(0)
            n_examples += x.size(0)
        scheduler.step()

        train_loss = loss_sum / n_examples
        val_y_true, val_y_pred = _predict(model, val_loader, device)
        val_macro_f1, _ = restricted_macro_f1(val_y_true, val_y_pred, config.num_classes)
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_macro_f1_restricted": val_macro_f1}
        )
        print(
            f"epoch {epoch + 1}/{config.epochs}: train_loss={train_loss:.4f}, "
            f"val_macro_f1_restricted={val_macro_f1:.4f}"
        )

    final = {
        "train": _evaluate_split(model, train_loader, device, config.num_classes),
        "val": _evaluate_split(model, val_loader, device, config.num_classes),
        "test": _evaluate_split(model, test_loader, device, config.num_classes),
    }

    metrics = {
        "run_name": config.run_name,
        "epochs_trained": config.epochs,
        "device": device,
        "torch_version": torch.__version__,
        "history": history,
        "final": final,
        "restricted_classes": RESTRICTED_CLASSES,
        "excluded_classes": [c for c in range(config.num_classes) if c not in RESTRICTED_CLASSES],
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    torch.save(model.state_dict(), run_dir / "checkpoint.pt")

    return metrics
