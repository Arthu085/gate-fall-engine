import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.multiseed_summary import EVENT_ARMS, RunConfig
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, save_alarm_protocol
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, save_config
from gatefall.train.baseline_b0.config import B0_FUSION_CONFIG, B0TrainConfig
from gatefall.train.baseline_b0.config import save_config as save_b0_config
from gatefall.train.baseline_b0.model import B0FusionClassifier
from gatefall.train.baseline_b1.config import B1_ADAPTIVE_GATE_CONFIG, B1TrainConfig
from gatefall.train.baseline_b1.config import save_config as save_b1_config
from gatefall.train.baseline_b1.model import B1AdaptiveGateClassifier
from gatefall.train.baseline_c0.config import C0_FUSION_CONFIG, C0TrainConfig
from gatefall.train.baseline_c0.config import save_config as save_c0_config
from gatefall.train.baseline_c0.model import C0FusionClassifier
from gatefall.train.baseline_c1.config import C1_ADAPTIVE_GATE_CONFIG, C1TrainConfig
from gatefall.train.baseline_c1.config import save_config as save_c1_config
from gatefall.train.baseline_c1.model import C1AdaptiveGateClassifier
from gatefall.train.shared.metrics import RESTRICTED_CLASSES, classification_summary, restricted_macro_f1
from gatefall.train.shared.tcn import TCNClassifier


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _build_event_split(
    n_fall_events: int,
    n_detected_events: int,
    sensitivity: float,
    fall_sensitivity: float,
    fall_or_fallen_sensitivity: float,
    false_alarms_per_hour: float,
    n_false_alarms: int,
    latency_mean: float | None,
    latency_median: float | None,
) -> dict:
    n_missed_events = n_fall_events - n_detected_events
    n_events_detected_in_fall = min(1, n_detected_events)
    n_events_detected_in_fall_or_fallen = min(1, n_detected_events)
    per_event = [latency_mean] * n_detected_events if n_detected_events > 0 else []
    return {
        "usable_windows": 100,
        "total_windows": 100,
        "labeled_windows": 80,
        "total_video_time_hours": 1.0,
        "labeled_time_hours": 0.8,
        "n_fall_events": n_fall_events,
        "n_detected_events": n_detected_events,
        "n_missed_events": n_missed_events,
        "sensitivity": sensitivity,
        "n_events_detected_in_fall": n_events_detected_in_fall,
        "n_events_detected_in_fall_or_fallen": n_events_detected_in_fall_or_fallen,
        "fall_sensitivity": fall_sensitivity,
        "fall_or_fallen_sensitivity": fall_or_fallen_sensitivity,
        "detected_events_alarm_within_fall_rate": 1.0 if n_detected_events > 0 else 0.0,
        "n_alarms_total": n_detected_events + n_false_alarms,
        "n_false_alarms": n_false_alarms,
        "n_pre_fall_false_alarms": 0,
        "false_alarms_per_hour": false_alarms_per_hour,
        "false_alarms_per_hour_labeled_time": false_alarms_per_hour,
        "window_binary_sensitivity": sensitivity,
        "window_binary_specificity": 0.9,
        "latency_seconds": {
            "per_event": per_event,
            "mean": latency_mean if per_event else None,
            "median": latency_median if per_event else None,
        },
    }


def _synthetic_classification_arrays(offset: int) -> tuple[np.ndarray, np.ndarray]:
    # 16 amostras cobrindo as 8 classes restritas com suporte 2 cada; as
    # classes 5 (lie_down) e 6 (lying) ficam sem suporte real, replicando o
    # cenário real do Le2i em stride 4.
    y_true = np.array(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 7, 7, 8, 8, 9, 9], dtype=np.int64
    )
    y_pred = y_true.copy()
    for i in range(offset):
        idx = (2 * i) % len(y_true)
        true_class = int(y_true[idx])
        position = RESTRICTED_CLASSES.index(true_class)
        wrong_class = RESTRICTED_CLASSES[(position + 1) % len(RESTRICTED_CLASSES)]
        y_pred[idx] = wrong_class
    return y_true, y_pred


def _synthetic_arm_config(arm: str, seed: int, epochs: int) -> RunConfig:
    if arm == "A":
        return replace(
            BASELINE_A_CONFIG,
            seed=seed,
            epochs=epochs,
            standardization_stats_path="synthetic",
            standardization_stats_sha256="synthetic",
        )
    common = dict(
        seed=seed,
        epochs=epochs,
        pose_standardization_stats_path="synthetic",
        pose_standardization_stats_sha256="synthetic",
        visual_standardization_stats_path="synthetic",
        visual_standardization_stats_sha256="synthetic",
    )
    if arm == "B0":
        return replace(B0_FUSION_CONFIG, **common)
    if arm == "B1":
        return replace(
            B1_ADAPTIVE_GATE_CONFIG,
            **common,
            quality_features_path="synthetic",
            quality_features_sha256="synthetic",
        )
    sam3 = dict(
        sam3_features_path="synthetic",
        sam3_features_sha256="synthetic",
        sam3_provenance={"synthetic": "synthetic"},
    )
    if arm == "C0":
        return replace(C0_FUSION_CONFIG, **common, **sam3)
    if arm == "C1":
        return replace(
            C1_ADAPTIVE_GATE_CONFIG,
            **common,
            **sam3,
            quality_features_path="synthetic",
            quality_features_sha256="synthetic",
            pose_features_sha256="synthetic",
        )
    raise ValueError(f"arma não suportada: {arm!r}")


def _write_synthetic_seed_run(
    run_dir: Path,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_event: dict,
    test_event: dict,
    epochs: int = 1,
    arm: str = "A",
) -> RunConfig:
    config = _synthetic_arm_config(arm, seed, epochs)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoint.pt"
    if isinstance(config, C1TrainConfig):
        save_c1_config(config, config_path, force=True)
        model = C1AdaptiveGateClassifier(
            config.channels, config.kernel_size, config.dilations,
            config.dropout, config.num_classes,
        )
    elif isinstance(config, B1TrainConfig):
        save_b1_config(config, config_path, force=True)
        model = B1AdaptiveGateClassifier(
            config.channels, config.kernel_size, config.dilations,
            config.dropout, config.num_classes,
        )
    elif isinstance(config, C0TrainConfig):
        save_c0_config(config, config_path, force=True)
        model = C0FusionClassifier(
            config.channels, config.kernel_size, config.dilations,
            config.dropout, config.num_classes,
        )
    elif isinstance(config, B0TrainConfig):
        save_b0_config(config, config_path, force=True)
        model = B0FusionClassifier(
            config.channels, config.kernel_size, config.dilations,
            config.dropout, config.num_classes,
        )
    else:
        save_config(config, config_path, force=True)
        model = TCNClassifier(
            config.input_dim, config.channels, config.kernel_size,
            config.dilations, config.dropout, config.num_classes,
        )
    torch.save(model.state_dict(), checkpoint_path)

    macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred, config.num_classes)
    summary = classification_summary(y_true, y_pred, LE2I_LABEL_NAMES, config.num_classes)

    split = {
        "macro_f1_restricted": macro_f1,
        "f1_by_class": {str(index): f1_by_class[index] for index in RESTRICTED_CLASSES},
        "support": {
            name: summary["per_class"][name]["support"] for name in LE2I_LABEL_NAMES
        },
        "confusion_matrix": summary["confusion_matrix"],
        "per_class": summary["per_class"],
    }
    metrics = {
        "run_name": config.run_name,
        "epochs_trained": config.epochs,
        "history": [
            {"epoch": epoch, "train_loss": 0.0, "val_macro_f1_restricted": macro_f1}
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
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream)

    if arm in EVENT_ARMS:
        alarm_protocol_path = run_dir / "alarm_protocol.yaml"
        save_alarm_protocol(BASELINE_A_ALARM_PROTOCOL, alarm_protocol_path, force=True)

        event_metrics_path = run_dir / "event_metrics.json"
        event_report = {
            "run_name": config.run_name,
            "checkpoint_path": str(checkpoint_path),
            "alarm_protocol_path": str(alarm_protocol_path),
            "splits": {"val": val_event, "test": test_event},
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_metrics_sha256": sha256_file(metrics_path),
            "alarm_protocol_sha256": sha256_file(alarm_protocol_path),
        }
        with event_metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(event_report, stream)

    return config


def _synthetic_pair(root: Path, arm: str) -> tuple[list[Path], RunConfig]:
    y_true, y_pred = _synthetic_classification_arrays(offset=0)
    event = _build_event_split(1, 1, 1.0, 1.0, 1.0, 0.0, 0, 1.0, 1.0)
    run_dirs = [root / "seed1", root / "seed2"]
    first = _write_synthetic_seed_run(
        run_dirs[0], 1, y_true, y_pred, event, event, arm=arm
    )
    _write_synthetic_seed_run(run_dirs[1], 2, y_true, y_pred, event, event, arm=arm)
    return run_dirs, replace(first, seed=42)


def _rewrite_synthetic_config(run_dir: Path, changes: dict) -> None:
    config_path = run_dir / "config.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config.update(changes)
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    metrics["config_sha256"] = sha256_file(config_path)
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream)
    event_path = run_dir / "event_metrics.json"
    if event_path.is_file():
        with event_path.open(encoding="utf-8") as stream:
            events = json.load(stream)
        events["training_metrics_sha256"] = sha256_file(metrics_path)
        with event_path.open("w", encoding="utf-8") as stream:
            json.dump(events, stream)
