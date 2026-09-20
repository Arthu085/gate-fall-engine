"""Loop de treino e avaliação do B1AdaptiveGateClassifier sobre janelas
pose+DINOv3 padronizadas e qualidade crua."""

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gatefall.features.dinov3_standardization import Dinov3StandardizationStats
from gatefall.features.dinov3_standardization import apply_standardization as apply_visual_standardization
from gatefall.features.standardization import StandardizationStats
from gatefall.features.standardization import apply_standardization as apply_pose_standardization
from gatefall.hashing import sha256_file
from gatefall.runs import validate_local_run_dir
from gatefall.train.b1_artifacts import REQUIRED_B1_TRAINING_ARTIFACTS, validate_b1_training_run
from gatefall.train.b1_config import B1TrainConfig, save_config
from gatefall.train.b1_model import B1AdaptiveGateClassifier
from gatefall.train.engine import configure_determinism
from gatefall.train.metrics import (
    RESTRICTED_CLASSES,
    classification_summary,
    restricted_macro_f1,
    support,
)


class _GatedFusionWindowSource(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, object]: ...


class _StandardizedGatedFusionTorchDataset(Dataset):
    def __init__(
        self,
        source: _GatedFusionWindowSource,
        pose_stats: StandardizationStats,
        visual_stats: Dinov3StandardizationStats,
    ) -> None:
        self._source = source
        self._pose_stats = pose_stats
        self._visual_stats = visual_stats

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        pose_window, visual_window, quality_window, label, _diag = self._source[index]
        standardized_pose = apply_pose_standardization(pose_window, self._pose_stats)
        standardized_visual = apply_visual_standardization(visual_window, self._visual_stats)
        # A qualidade NÃO é padronizada: q_pose e q_visual entram no gate como
        # proxies operacionais crus em [0, 1].
        return (
            torch.from_numpy(standardized_pose),
            torch.from_numpy(standardized_visual),
            torch.from_numpy(quality_window),
            label,
        )


def _collect_labels(source: _GatedFusionWindowSource) -> np.ndarray:
    labels = np.empty(len(source), dtype=np.int64)
    for i in range(len(source)):
        _, _, _, label, _diag = source[i]
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
def _predict(
    model: B1AdaptiveGateClassifier, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for x_pose, x_visual, x_quality, y in loader:
        x_pose = x_pose.to(device)
        x_visual = x_visual.to(device)
        x_quality = x_quality.to(device)
        logits = model(x_pose, x_visual, x_quality)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred)


def _evaluate_split(
    model: B1AdaptiveGateClassifier,
    loader: DataLoader,
    device: str,
    num_classes: int,
    label_names: tuple[str, ...],
) -> dict:
    y_true, y_pred = _predict(model, loader, device)
    macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred, num_classes)
    split_support = support(y_true, num_classes)
    summary = classification_summary(y_true, y_pred, label_names, num_classes)
    return {
        "macro_f1_restricted": macro_f1,
        "f1_by_class": {str(c): f1_by_class[c] for c in RESTRICTED_CLASSES},
        "support": {label_names[c]: split_support[c] for c in range(num_classes)},
        "confusion_matrix": summary["confusion_matrix"],
        "per_class": summary["per_class"],
    }


def guard_not_foreign_arm_run_dir(run_dir: Path, arm: str) -> None:
    """Recusa sobrescrever um run_dir que já pertence a outra arma.

    As guardas de CLI só cobrem os run_dirs canônicos; um run não canônico de
    outra arma (por exemplo `b0_fusion_seed7`) passaria por elas e seria
    substituído pelo promote com `--force`.
    """
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return
    try:
        with config_path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, ValueError, yaml.YAMLError):
        return
    if not isinstance(data, Mapping):
        return
    declared_arm = data.get("arm")
    if declared_arm is not None and declared_arm != arm:
        raise RuntimeError(
            f"run_dir {run_dir} já contém um run da arma {declared_arm!r}, "
            f"incompatível com a arma {arm!r}; escolha um run_dir próprio — "
            "nem --force sobrescreve o run de outra arma"
        )


def run_b1_training(
    train_source: _GatedFusionWindowSource,
    val_source: _GatedFusionWindowSource,
    test_source: _GatedFusionWindowSource,
    pose_stats: StandardizationStats,
    visual_stats: Dinov3StandardizationStats,
    config: B1TrainConfig,
    run_dir: Path,
    force: bool,
    label_names: tuple[str, ...],
) -> dict | None:
    validate_local_run_dir(run_dir)
    guard_not_foreign_arm_run_dir(run_dir, config.arm)
    required = REQUIRED_B1_TRAINING_ARTIFACTS
    present = [name for name in required if (run_dir / name).is_file()]
    if run_dir.exists() and not force:
        if len(present) == len(required):
            try:
                validate_b1_training_run(
                    run_dir,
                    expected_config=config,
                    fields_allowed_to_differ=frozenset({"trainable_param_count"}),
                )
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

    device = configure_determinism(config.seed)

    train_dataset = _StandardizedGatedFusionTorchDataset(train_source, pose_stats, visual_stats)
    val_dataset = _StandardizedGatedFusionTorchDataset(val_source, pose_stats, visual_stats)
    test_dataset = _StandardizedGatedFusionTorchDataset(test_source, pose_stats, visual_stats)

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

    model = B1AdaptiveGateClassifier(
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
        for x_pose, x_visual, x_quality, y in train_loader:
            x_pose = x_pose.to(device)
            x_visual = x_visual.to(device)
            x_quality = x_quality.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x_pose, x_visual, x_quality)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            loss_sum += float(loss.item()) * x_pose.size(0)
            n_examples += x_pose.size(0)
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

    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = replace(config, trainable_param_count=trainable_param_count)

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
    validate_b1_training_run(temporary_dir, expected_config=config)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        if not force:
            raise RuntimeError(f"run surgiu durante a execução: {run_dir}")
        guard_not_foreign_arm_run_dir(run_dir, config.arm)
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
