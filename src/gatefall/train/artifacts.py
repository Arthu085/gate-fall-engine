"""Validação semântica dos artefatos persistidos de treino."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from gatefall.hashing import sha256_file
from gatefall.train.config import TrainConfig, load_config
from gatefall.train.metrics import RESTRICTED_CLASSES
from gatefall.train.tcn import TCNClassifier

REQUIRED_TRAINING_ARTIFACTS = ("config.yaml", "metrics.json", "checkpoint.pt")


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} deve ser um objeto")
    return value


def validate_training_metrics(
    data: object,
    config: TrainConfig,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> None:
    metrics = _require_mapping(data, "metrics.json")
    if metrics.get("run_name") != config.run_name:
        raise ValueError("metrics.json: run_name diverge de config.yaml")
    if metrics.get("epochs_trained") != config.epochs:
        raise ValueError("metrics.json: epochs_trained diverge de config.yaml")
    history = metrics.get("history")
    if not isinstance(history, list) or len(history) != config.epochs:
        raise ValueError(
            f"metrics.json: history deve conter {config.epochs} épocas"
        )
    for expected_epoch, entry in enumerate(history, start=1):
        epoch = _require_mapping(entry, f"metrics.json.history[{expected_epoch - 1}]")
        if epoch.get("epoch") != expected_epoch:
            raise ValueError("metrics.json: sequência de épocas incompatível")
        for field in ("train_loss", "val_macro_f1_restricted"):
            value = epoch.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"metrics.json: {field} deve ser finito")
    if metrics.get("restricted_classes") != RESTRICTED_CLASSES:
        raise ValueError("metrics.json: restricted_classes incompatível")
    expected_excluded = [
        index for index in range(config.num_classes) if index not in RESTRICTED_CLASSES
    ]
    if metrics.get("excluded_classes") != expected_excluded:
        raise ValueError("metrics.json: excluded_classes incompatível")
    if config_path is not None and metrics.get("config_sha256") != sha256_file(
        config_path
    ):
        raise ValueError("metrics.json: config_sha256 incompatível")
    if checkpoint_path is not None and metrics.get(
        "checkpoint_sha256"
    ) != sha256_file(checkpoint_path):
        raise ValueError("metrics.json: checkpoint_sha256 incompatível")

    final = _require_mapping(metrics.get("final"), "metrics.json.final")
    for split in ("train", "val", "test"):
        split_metrics = _require_mapping(
            final.get(split), f"metrics.json.final.{split}"
        )
        macro_f1 = split_metrics.get("macro_f1_restricted")
        if not isinstance(macro_f1, (int, float)) or not math.isfinite(macro_f1):
            raise ValueError(
                f"metrics.json.final.{split}.macro_f1_restricted deve ser numérico"
            )
        f1_by_class = _require_mapping(
            split_metrics.get("f1_by_class"),
            f"metrics.json.final.{split}.f1_by_class",
        )
        if set(f1_by_class) != {str(index) for index in RESTRICTED_CLASSES}:
            raise ValueError(
                f"metrics.json.final.{split}.f1_by_class tem classes incompatíveis"
            )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in f1_by_class.values()
        ):
            raise ValueError(
                f"metrics.json.final.{split}.f1_by_class deve ser numérico e finito"
            )
        support = _require_mapping(
            split_metrics.get("support"), f"metrics.json.final.{split}.support"
        )
        if len(support) != config.num_classes or any(
            not isinstance(value, int) or value < 0 for value in support.values()
        ):
            raise ValueError(
                f"metrics.json.final.{split}.support deve conter "
                f"{config.num_classes} contagens não negativas"
            )

        has_confusion_matrix = "confusion_matrix" in split_metrics
        has_per_class = "per_class" in split_metrics
        if has_confusion_matrix != has_per_class:
            raise ValueError(
                f"metrics.json.final.{split}: confusion_matrix e per_class "
                "devem estar ambos presentes ou ambos ausentes"
            )
        if has_confusion_matrix and has_per_class:
            _validate_classification_diagnostics(
                split_metrics, config, split, total_support=sum(support.values())
            )


def _validate_classification_diagnostics(
    split_metrics: Mapping[str, Any],
    config: TrainConfig,
    split: str,
    total_support: int,
) -> None:
    prefix = f"metrics.json.final.{split}"
    num_classes = config.num_classes

    matrix = split_metrics.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != num_classes
        or any(
            not isinstance(row, list)
            or len(row) != num_classes
            or any(not isinstance(value, int) or value < 0 for value in row)
            for row in matrix
        )
    ):
        raise ValueError(
            f"{prefix}.confusion_matrix deve ser {num_classes}x{num_classes} "
            "de inteiros não negativos"
        )
    matrix_total = sum(sum(row) for row in matrix)
    if matrix_total != total_support:
        raise ValueError(
            f"{prefix}.confusion_matrix: soma total ({matrix_total}) diverge "
            f"do suporte total do split ({total_support})"
        )

    per_class = split_metrics.get("per_class")
    if not isinstance(per_class, Mapping) or len(per_class) != num_classes:
        raise ValueError(f"{prefix}.per_class deve conter {num_classes} entradas")

    seen_ids: set[int] = set()
    for name, entry in per_class.items():
        entry_prefix = f"{prefix}.per_class[{name!r}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{entry_prefix} deve ser um objeto")

        class_id = entry.get("id")
        if (
            not isinstance(class_id, int)
            or isinstance(class_id, bool)
            or not (0 <= class_id < num_classes)
        ):
            raise ValueError(f"{entry_prefix}.id deve ser um inteiro em [0, {num_classes})")
        if class_id in seen_ids:
            raise ValueError(f"{entry_prefix}.id repetido: {class_id}")
        seen_ids.add(class_id)

        int_fields = ("tp", "tn", "fp", "fn", "support")
        for field in int_fields:
            value = entry.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{entry_prefix}.{field} deve ser um inteiro não negativo")

        tp, tn, fp, fn, class_support = (entry[field] for field in int_fields)
        if tp + fn != class_support:
            raise ValueError(f"{entry_prefix}: tp + fn != support")
        if tn != total_support - tp - fp - fn:
            raise ValueError(f"{entry_prefix}: tn != N - tp - fp - fn")

        row_sum = sum(matrix[class_id])
        if row_sum != class_support:
            raise ValueError(
                f"{entry_prefix}: soma da linha {class_id} da confusion_matrix "
                f"({row_sum}) diverge de support ({class_support})"
            )

        matrix_tp = matrix[class_id][class_id]
        if tp != matrix_tp:
            raise ValueError(
                f"{entry_prefix}: tp ({tp}) diverge de confusion_matrix[{class_id}][{class_id}] "
                f"({matrix_tp})"
            )
        matrix_fn = row_sum - matrix_tp
        if fn != matrix_fn:
            raise ValueError(
                f"{entry_prefix}: fn ({fn}) diverge da confusion_matrix "
                f"(soma da linha {class_id} menos a diagonal = {matrix_fn})"
            )
        column_sum = sum(row[class_id] for row in matrix)
        matrix_fp = column_sum - matrix_tp
        if fp != matrix_fp:
            raise ValueError(
                f"{entry_prefix}: fp ({fp}) diverge da confusion_matrix "
                f"(soma da coluna {class_id} menos a diagonal = {matrix_fp})"
            )

        float_fields = ("precision", "recall", "f1")
        for field in float_fields:
            value = entry.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{entry_prefix}.{field} deve ser numérico")
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"{entry_prefix}.{field} deve ser finito e estar em [0,1]")

        precision_denom = tp + fp
        recall_denom = tp + fn
        expected_precision = tp / precision_denom if precision_denom > 0 else 0.0
        expected_recall = tp / recall_denom if recall_denom > 0 else 0.0
        if expected_precision + expected_recall == 0:
            expected_f1 = 0.0
        else:
            expected_f1 = (
                2 * expected_precision * expected_recall / (expected_precision + expected_recall)
            )
        for field, expected in (
            ("precision", expected_precision),
            ("recall", expected_recall),
            ("f1", expected_f1),
        ):
            if not math.isclose(entry[field], expected, abs_tol=1e-9, rel_tol=0):
                raise ValueError(
                    f"{entry_prefix}.{field} ({entry[field]}) diverge do valor "
                    f"recalculado ({expected})"
                )

    if seen_ids != set(range(num_classes)):
        raise ValueError(f"{prefix}.per_class: ids devem cobrir range(0, {num_classes})")


def load_compatible_checkpoint(path: Path, config: TrainConfig) -> TCNClassifier:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint não contém um state_dict")
        model = TCNClassifier(
            input_dim=config.input_dim,
            channels=config.channels,
            kernel_size=config.kernel_size,
            dilations=config.dilations,
            dropout=config.dropout,
            num_classes=config.num_classes,
        )
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(
            f"checkpoint.pt não é carregável/compatível com config.yaml: {exc}"
        ) from exc
    return model


def validate_training_run(
    run_dir: Path,
    expected_config: TrainConfig | None = None,
    fields_allowed_to_differ: frozenset[str] = frozenset(),
) -> TrainConfig:
    present = [
        name for name in REQUIRED_TRAINING_ARTIFACTS if (run_dir / name).is_file()
    ]
    if len(present) != len(REQUIRED_TRAINING_ARTIFACTS):
        missing = [
            name for name in REQUIRED_TRAINING_ARTIFACTS if name not in present
        ]
        raise RuntimeError(
            f"run parcial em {run_dir}: artefatos ausentes: {', '.join(missing)}"
        )

    try:
        config = load_config(run_dir / "config.yaml")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"config.yaml inválido em {run_dir}: {exc}") from exc
    if expected_config is not None:
        expected_dict = expected_config.to_dict()
        actual_dict = config.to_dict()
        disallowed_diffs = sorted(
            field
            for field in expected_dict
            if field not in fields_allowed_to_differ
            and actual_dict.get(field) != expected_dict[field]
        )
        if disallowed_diffs:
            raise RuntimeError(
                f"config.yaml em {run_dir} não corresponde à configuração "
                f"solicitada: campo(s) divergente(s): {', '.join(disallowed_diffs)}"
            )

    try:
        with (run_dir / "metrics.json").open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        validate_training_metrics(
            metrics,
            config,
            config_path=run_dir / "config.yaml",
            checkpoint_path=run_dir / "checkpoint.pt",
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"metrics.json inválido em {run_dir}: {exc}") from exc

    try:
        load_compatible_checkpoint(run_dir / "checkpoint.pt", config)
    except ValueError as exc:
        raise RuntimeError(f"checkpoint.pt inválido em {run_dir}: {exc}") from exc
    return config
