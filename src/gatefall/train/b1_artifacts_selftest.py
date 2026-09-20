"""Selftest sintético da validação de artefatos de treino do B1
(`b1_artifacts.py`). Não toca em dados reais: cobre sobretudo a rejeição de
runs incompatíveis, inclusive um checkpoint do B0 carregado sob strict=True."""

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from gatefall.hashing import sha256_file
from gatefall.train.b0_config import B0_FUSION_CONFIG
from gatefall.train.b0_model import B0FusionClassifier
from gatefall.train.b1_artifacts import (
    REQUIRED_B1_TRAINING_ARTIFACTS,
    load_compatible_b1_checkpoint,
    validate_b1_training_run,
)
from gatefall.train.b1_config import B1_ADAPTIVE_GATE_CONFIG, B1TrainConfig, save_config
from gatefall.train.b1_model import B1AdaptiveGateClassifier
from gatefall.train.metrics import RESTRICTED_CLASSES


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _build_b1_model(config: B1TrainConfig) -> B1AdaptiveGateClassifier:
    return B1AdaptiveGateClassifier(
        channels=config.channels,
        kernel_size=config.kernel_size,
        dilations=config.dilations,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )


def _write_metrics(run_dir: Path, config: B1TrainConfig) -> None:
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoint.pt"
    split = {
        "macro_f1_restricted": 0.0,
        "f1_by_class": {str(index): 0.0 for index in RESTRICTED_CLASSES},
        "support": {str(index): 0 for index in range(config.num_classes)},
    }
    metrics = {
        "run_name": config.run_name,
        "epochs_trained": config.epochs,
        "history": [
            {"epoch": epoch, "train_loss": 0.0, "val_macro_f1_restricted": 0.0}
            for epoch in range(1, config.epochs + 1)
        ],
        "final": {name: dict(split) for name in ("train", "val", "test")},
        "restricted_classes": RESTRICTED_CLASSES,
        "excluded_classes": [
            index for index in range(config.num_classes) if index not in RESTRICTED_CLASSES
        ],
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream)


def _write_valid_run(run_dir: Path, config: B1TrainConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml", force=True)
    torch.save(_build_b1_model(config).state_dict(), run_dir / "checkpoint.pt")
    _write_metrics(run_dir, config)


def check_valid_run_accepted() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        config = replace(B1_ADAPTIVE_GATE_CONFIG, seed=1, epochs=1)
        _write_valid_run(run_dir, config)
        accepted = True
        try:
            validate_b1_training_run(run_dir, expected_config=config)
        except RuntimeError:
            accepted = False
        return _check(
            "validate_b1_training_run aceita um run B1 completo e íntegro",
            accepted,
        )


def check_seed_and_param_count_only_difference_accepted() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        config = replace(
            B1_ADAPTIVE_GATE_CONFIG, seed=1, epochs=1, trainable_param_count=123
        )
        _write_valid_run(run_dir, config)

        expected = replace(config, seed=2, trainable_param_count=0)
        accepted = True
        try:
            validate_b1_training_run(
                run_dir,
                expected_config=expected,
                fields_allowed_to_differ=frozenset({"seed", "trainable_param_count"}),
            )
        except RuntimeError:
            accepted = False
        return _check(
            "validate_b1_training_run aceita run cuja config só diverge em "
            "'seed' e 'trainable_param_count'",
            accepted,
        )


def check_b0_checkpoint_rejected() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir(parents=True)
        config = replace(B1_ADAPTIVE_GATE_CONFIG, seed=1, epochs=1)
        b0_model = B0FusionClassifier(
            channels=B0_FUSION_CONFIG.channels,
            kernel_size=B0_FUSION_CONFIG.kernel_size,
            dilations=B0_FUSION_CONFIG.dilations,
            dropout=B0_FUSION_CONFIG.dropout,
            num_classes=B0_FUSION_CONFIG.num_classes,
        )
        checkpoint_path = run_dir / "checkpoint.pt"
        torch.save(b0_model.state_dict(), checkpoint_path)

        raised = False
        try:
            load_compatible_b1_checkpoint(checkpoint_path, config)
        except ValueError:
            raised = True
        return _check(
            "load_compatible_b1_checkpoint recusa um checkpoint do B0 (sem os "
            "pesos do gate) porque load_state_dict roda com strict=True",
            raised,
        )


def check_mismatched_architecture_rejected() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir(parents=True)
        config = replace(B1_ADAPTIVE_GATE_CONFIG, seed=1, epochs=1)
        checkpoint_path = run_dir / "checkpoint.pt"
        torch.save(_build_b1_model(config).state_dict(), checkpoint_path)

        wrong_channels = replace(config, channels=[c + 1 for c in config.channels])
        wrong_classes = replace(config, num_classes=config.num_classes + 1)

        channels_raised = False
        try:
            load_compatible_b1_checkpoint(checkpoint_path, wrong_channels)
        except ValueError:
            channels_raised = True

        classes_raised = False
        try:
            load_compatible_b1_checkpoint(checkpoint_path, wrong_classes)
        except ValueError:
            classes_raised = True

        return _check(
            "load_compatible_b1_checkpoint recusa checkpoint com channels ou "
            "num_classes divergentes da config",
            channels_raised and classes_raised,
        )


def check_arm_b0_config_rejected() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        config = replace(B1_ADAPTIVE_GATE_CONFIG, seed=1, epochs=1, arm="B0")
        _write_valid_run(run_dir, config)

        expected = replace(config, arm="B1")
        raised = False
        try:
            validate_b1_training_run(run_dir, expected_config=expected)
        except RuntimeError as exc:
            raised = "arm" in str(exc)
        return _check(
            "validate_b1_training_run recusa um run cujo config.yaml declara "
            "arm: B0",
            raised,
        )


def check_mismatched_quality_sha256_rejected() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        config = replace(
            B1_ADAPTIVE_GATE_CONFIG,
            seed=1,
            epochs=1,
            quality_features_path="data/features/le2i/quality",
            quality_features_sha256="deadbeef",
        )
        _write_valid_run(run_dir, config)

        expected = replace(config, quality_features_sha256="cafebabe")
        raised = False
        try:
            validate_b1_training_run(run_dir, expected_config=expected)
        except RuntimeError as exc:
            raised = "quality_features_sha256" in str(exc)
        return _check(
            "validate_b1_training_run recusa um run cujos sidecars de "
            "qualidade têm sha256 divergente do esperado",
            raised,
        )


def check_partial_run_rejected() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir(parents=True)
        config = replace(B1_ADAPTIVE_GATE_CONFIG, seed=1, epochs=1)
        save_config(config, run_dir / "config.yaml", force=True)
        # checkpoint.pt e metrics.json propositalmente ausentes.

        raised = False
        try:
            validate_b1_training_run(run_dir, expected_config=config)
        except RuntimeError:
            raised = True
        return _check(
            f"validate_b1_training_run recusa run parcial (artefatos ausentes "
            f"de {REQUIRED_B1_TRAINING_ARTIFACTS})",
            raised,
        )


def check_checkpoint_sha256_mismatch_rejected() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        config = replace(B1_ADAPTIVE_GATE_CONFIG, seed=1, epochs=1)
        _write_valid_run(run_dir, config)
        # Regrava o checkpoint com outros pesos sem atualizar metrics.json.
        torch.manual_seed(999)
        torch.save(_build_b1_model(config).state_dict(), run_dir / "checkpoint.pt")

        raised = False
        try:
            validate_b1_training_run(run_dir, expected_config=config)
        except RuntimeError as exc:
            raised = "checkpoint_sha256" in str(exc)
        return _check(
            "validate_b1_training_run recusa run cujo checkpoint_sha256 "
            "registrado em metrics.json não bate com o arquivo em disco",
            raised,
        )


def run_b1_artifacts_selftest() -> bool:
    checks = [
        check_valid_run_accepted(),
        check_seed_and_param_count_only_difference_accepted(),
        check_b0_checkpoint_rejected(),
        check_mismatched_architecture_rejected(),
        check_arm_b0_config_rejected(),
        check_mismatched_quality_sha256_rejected(),
        check_partial_run_rejected(),
        check_checkpoint_sha256_mismatch_rejected(),
    ]
    ok = all(checks)
    if not ok:
        print("\nb1 artifacts selftest FALHOU", file=sys.stderr)
    else:
        print("\nb1 artifacts selftest OK: todas as checagens passaram")
    return ok
