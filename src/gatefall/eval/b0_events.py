"""Avaliação de eventos da arma B0 sobre o protocolo Le2i-CS."""

import argparse
import json
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE
from gatefall.data.fusion_dataset import FusionWindowDataset
from gatefall.data.windowing import build_window_index
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.dinov3.dataset_guard import ensure_dinov3_dataset_supported
from gatefall.dinov3.storage import dinov3_path, read_features
from gatefall.eval.alarm_protocol import (
    BASELINE_A_ALARM_PROTOCOL,
    load_alarm_protocol,
    save_alarm_protocol,
)
from gatefall.eval.event_artifacts import (
    EventEvaluationLock,
    _promote_event_outputs,
    _recover_event_publication,
    _require_event_lock,
    validate_event_metrics,
)
from gatefall.eval.events import extract_label_segments, split_event_report
from gatefall.features.dinov3_standardization import (
    Dinov3StandardizationStats,
    apply_standardization as apply_visual_standardization,
    load_stats as load_visual_stats,
    validate_stats_freshness as validate_visual_stats_freshness,
    validate_stats_layout as validate_visual_stats_layout,
)
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization as apply_pose_standardization,
    load_stats as load_pose_stats,
    validate_stats_layout as validate_pose_stats_layout,
)
from gatefall.features.standardize_dinov3 import DINOV3_STATS_PATH
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import default_run_dir_for_arm, validate_local_run_dir
from gatefall.train.b0_artifacts import (
    load_compatible_b0_checkpoint,
    validate_b0_training_run,
)
from gatefall.train.b0_config import B0_FUSION_CONFIG, B0TrainConfig
from gatefall.train.b0_model import B0FusionClassifier
from gatefall.train.b0_run import guard_not_arm_a_run_dir, resolve_b0_config

ARM_NAME = "b0_fusion"


class FusionWindowSource(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, int, tuple[str, int]]: ...


class FusionModel(Protocol):
    def __call__(
        self, pose: torch.Tensor, visual: torch.Tensor
    ) -> torch.Tensor: ...


@torch.no_grad()
def _predict_with_identity(
    model: FusionModel,
    source: FusionWindowSource,
    pose_stats: StandardizationStats,
    visual_stats: Dinov3StandardizationStats,
    device: str,
    batch_size: int,
) -> tuple[list[str], list[int], list[int], list[int]]:
    video_ids: list[str] = []
    k_ends: list[int] = []
    true_labels: list[int] = []
    pred_labels: list[int] = []

    batch_pose: list[np.ndarray] = []
    batch_visual: list[np.ndarray] = []
    batch_labels: list[int] = []
    batch_identity: list[tuple[str, int]] = []

    def flush() -> None:
        if not batch_pose:
            return
        pose = apply_pose_standardization(
            np.stack(batch_pose, axis=0), pose_stats
        )
        visual = apply_visual_standardization(
            np.stack(batch_visual, axis=0), visual_stats
        )
        logits = model(
            torch.from_numpy(pose).to(device),
            torch.from_numpy(visual).to(device),
        )
        predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        for (video_id, k_end), label, prediction in zip(
            batch_identity, batch_labels, predictions
        ):
            video_ids.append(video_id)
            k_ends.append(k_end)
            true_labels.append(label)
            pred_labels.append(int(prediction))
        batch_pose.clear()
        batch_visual.clear()
        batch_labels.clear()
        batch_identity.clear()

    for index in range(len(source)):
        pose, visual, label, identity = source[index]
        batch_pose.append(pose)
        batch_visual.append(visual)
        batch_labels.append(label)
        batch_identity.append(identity)
        if len(batch_pose) == batch_size:
            flush()
    flush()
    return video_ids, k_ends, true_labels, pred_labels


def _n_fall_segments_in_annotation(frames: pd.DataFrame, split: str) -> int:
    split_frames = cast(
        pd.DataFrame, frames[frames["split"] == split]
    ).sort_values(["video_id", "frame_index"])
    total_segments = 0
    for _video_id, group in split_frames.groupby("video_id", sort=False):
        total_segments += len(
            extract_label_segments(
                group["frame_index"].to_numpy(),
                group["label"].to_numpy(),
                BASELINE_A_ALARM_PROTOCOL.fall_label,
            )
        )
    return total_segments


def _load_run_assets(
    dataset_name: str, run_dir: Path
) -> tuple[
    DatasetAdapter,
    B0TrainConfig,
    StandardizationStats,
    Dinov3StandardizationStats,
    B0FusionClassifier,
]:
    adapter = get_dataset(dataset_name)
    ensure_dinov3_dataset_supported(adapter)
    pose_stats = load_pose_stats(adapter.pose_stats_path)
    validate_pose_stats_layout(pose_stats)
    visual_stats = load_visual_stats(DINOV3_STATS_PATH)
    validate_visual_stats_layout(visual_stats, dataset_name=dataset_name)
    validate_visual_stats_freshness(visual_stats, adapter.frames_path)
    expected_config = resolve_b0_config(
        B0_FUSION_CONFIG.seed,
        adapter.pose_stats_path,
        sha256_file(adapter.pose_stats_path),
        DINOV3_STATS_PATH,
        sha256_file(DINOV3_STATS_PATH),
    )
    try:
        config = validate_b0_training_run(
            run_dir,
            expected_config=expected_config,
            fields_allowed_to_differ=frozenset(
                {"seed", "trainable_param_count"}
            ),
        )
    except RuntimeError as exc:
        raise RuntimeError(f"run de treino B0 inválido em {run_dir}: {exc}") from exc
    if config.eval_stride != EVAL_STRIDE:
        raise ValueError(
            f"config.eval_stride ({config.eval_stride}) diverge de "
            f"EVAL_STRIDE ({EVAL_STRIDE})"
        )
    model = load_compatible_b0_checkpoint(run_dir / "checkpoint.pt", config)
    return adapter, config, pose_stats, visual_stats, model


def _run_evaluate_locked(
    force: bool,
    dataset_name: str,
    run_dir: Path,
    lock: EventEvaluationLock,
) -> None:
    _require_event_lock(run_dir, lock)
    checkpoint_path = run_dir / "checkpoint.pt"
    alarm_protocol_path = run_dir / "alarm_protocol.yaml"
    event_metrics_path = run_dir / "event_metrics.json"
    recovery = _recover_event_publication(run_dir, lock)
    if recovery is not None:
        print(f"recovery de avaliação concluído: {recovery}")

    adapter, config, pose_stats, visual_stats, model = _load_run_assets(
        dataset_name, run_dir
    )
    event_outputs = (alarm_protocol_path, event_metrics_path)
    present_outputs = [path for path in event_outputs if path.is_file()]
    previous_pair_valid = False
    if len(present_outputs) == len(event_outputs):
        try:
            protocol = load_alarm_protocol(alarm_protocol_path)
            if protocol != BASELINE_A_ALARM_PROTOCOL:
                raise ValueError("alarm_protocol.yaml incompatível com a arma B0")
            with event_metrics_path.open(encoding="utf-8") as stream:
                existing_report = json.load(stream)
            validate_event_metrics(
                existing_report,
                config,
                checkpoint_path,
                alarm_protocol_path,
                training_metrics_path=run_dir / "metrics.json",
                require_hashes=True,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            if not force:
                raise RuntimeError(
                    f"avaliação de eventos inconsistente ({exc}); "
                    "use --force para reconstruir"
                ) from exc
        else:
            previous_pair_valid = True
            if not force:
                print(f"skip {event_metrics_path} (avaliação completa e íntegra)")
                return
    elif present_outputs and not force:
        missing_outputs = [str(path) for path in event_outputs if not path.is_file()]
        raise RuntimeError(
            "avaliação de eventos parcial; artefatos ausentes: "
            + ", ".join(missing_outputs)
            + "; use --force para reconstruir"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    frames = adapter.load_frames()
    pose_loader: Callable[[str], np.ndarray] = lambda video_id: build_pose_features(
        video_id, pose_root=adapter.pose_root
    )[0]
    visual_loader: Callable[[str], np.ndarray] = lambda video_id: read_features(
        dinov3_path(video_id, dinov3_root=adapter.dinov3_root)
    ).astype("float32")

    splits: dict[str, dict] = {}
    for split in ("val", "test"):
        source = FusionWindowDataset(
            frames,
            split,
            EVAL_STRIDE,
            pose_loader,
            visual_loader,
            drop_ignored=False,
        )
        video_ids, k_ends, true_labels, pred_labels = _predict_with_identity(
            model,
            source,
            pose_stats,
            visual_stats,
            device,
            batch_size=config.batch_size,
        )
        split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
        usable_windows = len(source)
        total_windows = len(
            build_window_index(
                split_frames, stride=EVAL_STRIDE, drop_ignored=False
            )
        )
        if usable_windows != total_windows:
            raise RuntimeError(
                f"split={split!r}: usable_windows ({usable_windows}) != "
                f"total_windows ({total_windows}) apesar de drop_ignored=False"
            )
        labeled_windows = len(
            build_window_index(
                split_frames, stride=EVAL_STRIDE, drop_ignored=True
            )
        )
        split_report = split_event_report(
            video_ids,
            k_ends,
            true_labels,
            pred_labels,
            BASELINE_A_ALARM_PROTOCOL,
            usable_windows,
            total_windows,
            labeled_windows,
        )
        annotated_fall_segments = _n_fall_segments_in_annotation(frames, split)
        if split_report["n_fall_events"] != annotated_fall_segments:
            raise ValueError(
                f"split={split!r}: n_fall_events extraído das janelas usáveis "
                f"({split_report['n_fall_events']}) diverge da contagem de "
                "segmentos fall na anotação bruta "
                f"({annotated_fall_segments}) — possível janela IGNORE_LABEL "
                "descartada dentro de um run fall, dividindo um evento real em dois"
            )
        splits[split] = split_report

    report = {
        "run_name": config.run_name,
        "checkpoint_path": str(checkpoint_path),
        "alarm_protocol_path": str(alarm_protocol_path),
        "splits": splits,
    }
    token = uuid.uuid4().hex
    protocol_tmp = run_dir / f".alarm_protocol.pending-{token}.yaml"
    metrics_tmp = run_dir / f".event_metrics.pending-{token}.json"
    save_alarm_protocol(BASELINE_A_ALARM_PROTOCOL, protocol_tmp, force=True)
    report["checkpoint_sha256"] = sha256_file(checkpoint_path)
    report["training_metrics_sha256"] = sha256_file(run_dir / "metrics.json")
    report["alarm_protocol_sha256"] = sha256_file(protocol_tmp)
    with metrics_tmp.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    with metrics_tmp.open(encoding="utf-8") as stream:
        staged_report = json.load(stream)
    if load_alarm_protocol(protocol_tmp) != BASELINE_A_ALARM_PROTOCOL:
        raise RuntimeError("staging de alarm_protocol.yaml divergiu do protocolo")
    validate_event_metrics(
        staged_report,
        config,
        checkpoint_path,
        alarm_protocol_path,
        training_metrics_path=run_dir / "metrics.json",
        protocol_file_path=protocol_tmp,
        require_hashes=True,
    )
    _promote_event_outputs(
        protocol_tmp,
        metrics_tmp,
        alarm_protocol_path,
        event_metrics_path,
        lock,
        preserve_previous=previous_pair_valid,
    )
    print(
        f"{event_metrics_path}: métricas de eventos gravadas "
        f"(run_name={config.run_name})"
    )


def run_evaluate(
    force: bool, dataset_name: str = "le2i", run_dir: Path | None = None
) -> None:
    if dataset_name != "le2i":
        raise ValueError("avaliação de eventos B0 suporta somente le2i (CS)")
    if run_dir is None:
        run_dir = default_run_dir_for_arm(dataset_name, ARM_NAME)
    guard_not_arm_a_run_dir(run_dir, dataset_name)
    validate_local_run_dir(run_dir, dataset_name)
    with EventEvaluationLock(run_dir) as lock:
        _run_evaluate_locked(force, dataset_name, run_dir, lock)


def run_selftest() -> None:
    from gatefall.eval.b0_events_selftest import run_b0_events_selftest

    if not run_b0_events_selftest():
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Avalia eventos da arma B0 e grava event_metrics.json",
    )
    evaluate_parser.add_argument("--force", action="store_true")
    evaluate_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    evaluate_parser.add_argument("--run-dir", type=Path, default=None)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da avaliação B0")
    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(args.force, args.dataset, args.run_dir)
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
