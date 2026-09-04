"""Loop de treino e avaliação da TCN sobre janelas de pose padronizadas."""

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gatefall.features.standardization import StandardizationStats, apply_standardization
from gatefall.hashing import sha256_file
from gatefall.runs import validate_local_run_dir
from gatefall.train.artifacts import REQUIRED_TRAINING_ARTIFACTS, validate_training_run
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
    weights = np.zeros(num_classes, dtype=np.float32)
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


def _evaluate_split(
    model: TCNClassifier,
    loader: DataLoader,
    device: str,
    num_classes: int,
    label_names: tuple[str, ...],
) -> dict:
    y_true, y_pred = _predict(model, loader, device)
    macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred, num_classes)
    split_support = support(y_true, num_classes)
    return {
        "macro_f1_restricted": macro_f1,
        "f1_by_class": {str(c): f1_by_class[c] for c in RESTRICTED_CLASSES},
        "support": {label_names[c]: split_support[c] for c in range(num_classes)},
    }


def _configure_determinism(seed: int) -> str:
    # CUBLAS_WORKSPACE_CONFIG precisa estar no ambiente do processo antes da
    # primeira chamada CUDA (abaixo) para que o cuBLAS use um algoritmo
    # determinístico; nada neste processo toca CUDA antes daqui.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Retreinos com a mesma seed e as mesmas features devem produzir o
    # mesmo checkpoint nesta máquina/GPU/driver/cuDNN.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return device


def run_training(
    input_dim: int,
    train_source: _WindowSource,
    val_source: _WindowSource,
    test_source: _WindowSource,
    stats: StandardizationStats,
    config: TrainConfig,
    run_dir: Path,
    force: bool,
    label_names: tuple[str, ...],
) -> dict | None:
    validate_local_run_dir(run_dir)
    required = REQUIRED_TRAINING_ARTIFACTS
    present = [name for name in required if (run_dir / name).is_file()]
    if run_dir.exists() and not force:
        if len(present) == len(required):
            try:
                validate_training_run(run_dir, expected_config=config)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"run inconsistente em {run_dir}: artefato inválido ({exc}); "
                    "use --force para reconstruir"
                ) from exc
            print(f"skip {run_dir} (treino completo e íntegro)")
            return None
        missing = [name for name in required if name not in present]
        raise RuntimeError(
            f"run parcial em {run_dir}: artefatos ausentes: {', '.join(missing)}; "
            "use --force para reconstruir"
        )

    temporary_dir = run_dir.with_name(f".{run_dir.name}.tmp-{uuid.uuid4().hex}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    device = _configure_determinism(config.seed)

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
        "train": _evaluate_split(model, train_loader, device, config.num_classes, label_names),
        "val": _evaluate_split(model, val_loader, device, config.num_classes, label_names),
        "test": _evaluate_split(model, test_loader, device, config.num_classes, label_names),
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

    config_path = temporary_dir / "config.yaml"
    checkpoint_path = temporary_dir / "checkpoint.pt"
    save_config(config, config_path, force=True)
    torch.save(model.state_dict(), checkpoint_path)
    metrics["config_sha256"] = sha256_file(config_path)
    metrics["checkpoint_sha256"] = sha256_file(checkpoint_path)
    with (temporary_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    for name in required:
        if not (temporary_dir / name).is_file():
            raise RuntimeError(f"treino não produziu o artefato obrigatório: {name}")
    validate_training_run(temporary_dir, expected_config=config)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        if not force:
            raise RuntimeError(f"run surgiu durante a execução: {run_dir}")
        backup_dir = run_dir.with_name(f".{run_dir.name}.old-{uuid.uuid4().hex}")
        os.replace(run_dir, backup_dir)
        try:
            os.replace(temporary_dir, run_dir)
        except BaseException:
            os.replace(backup_dir, run_dir)
            raise
        shutil.rmtree(backup_dir)
    else:
        os.replace(temporary_dir, run_dir)

    return metrics
