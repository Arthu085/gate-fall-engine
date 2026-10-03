"""Avaliação de eventos da arma B1 sobre o protocolo Le2i-CS."""

import argparse
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
import yaml

from gatefall.config import EVAL_STRIDE
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.dinov3.consumer import validate_dinov3_feature_set
from gatefall.dinov3.dataset_guard import (
    DINOV3_SUPPORTED_DATASET_IDENTIFIERS,
    ensure_dinov3_dataset_supported,
)
from gatefall.dinov3.storage import dinov3_path, read_features
from gatefall.eval.shared.event_artifacts import EventEvaluationLock
from gatefall.eval.shared.orchestration import (
    EventEvaluation,
    Predictions,
    SplitEvaluator,
    run_event_evaluation,
)
from gatefall.features.dinov3_standardization import (
    Dinov3StandardizationStats,
    apply_standardization as apply_visual_standardization,
    load_stats as load_visual_stats,
    validate_stats_freshness as validate_visual_stats_freshness,
    validate_stats_layout as validate_visual_stats_layout,
)
from gatefall.features.quality_storage import (
    quality_path,
    quality_set_sha256,
    read_quality,
)
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization as apply_pose_standardization,
    load_stats as load_pose_stats,
    validate_stats_layout as validate_pose_stats_layout,
)
from gatefall.features.standardize_dinov3 import dinov3_stats_path
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import default_run_dir_for_arm, validate_local_run_dir
from gatefall.train.baseline_b1.artifacts import (
    load_compatible_b1_checkpoint,
    validate_b1_training_run,
)
from gatefall.train.baseline_b1.config import B1_ADAPTIVE_GATE_CONFIG, B1TrainConfig
from gatefall.train.baseline_b1.model import B1AdaptiveGateClassifier
from gatefall.train.baseline_b1.run import (
    guard_not_arm_a_run_dir,
    guard_not_arm_b0_run_dir,
    resolve_b1_config,
)

ARM_NAME = "b1_adaptive_gate"


class GatedFusionWindowSource(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[str, int]]: ...


class GatedFusionModel(Protocol):
    def __call__(
        self, pose: torch.Tensor, visual: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor: ...


@torch.no_grad()
def _predict_with_identity(
    model: GatedFusionModel,
    source: GatedFusionWindowSource,
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
    batch_quality: list[np.ndarray] = []
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
        # A qualidade NÃO é padronizada: q_pose (canal 0) e q_visual (canal 1)
        # entram no gate como proxies operacionais crus em [0, 1], exatamente
        # como no treino.
        quality = np.stack(batch_quality, axis=0)
        logits = model(
            torch.from_numpy(pose).to(device),
            torch.from_numpy(visual).to(device),
            torch.from_numpy(quality).to(device),
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
        batch_quality.clear()
        batch_labels.clear()
        batch_identity.clear()

    for index in range(len(source)):
        pose, visual, quality, label, identity = source[index]
        batch_pose.append(pose)
        batch_visual.append(visual)
        batch_quality.append(quality)
        batch_labels.append(label)
        batch_identity.append(identity)
        if len(batch_pose) == batch_size:
            flush()
    flush()
    return video_ids, k_ends, true_labels, pred_labels


def _quality_set_sha256(adapter: DatasetAdapter) -> str:
    from gatefall.features.quality_extract import validate_shared_quality_set

    validate_shared_quality_set(adapter)
    frames = adapter.load_frames()
    video_ids = [str(video_id) for video_id in frames["video_id"].unique()]
    return quality_set_sha256(video_ids, quality_root=adapter.quality_root)


def _load_run_assets(
    dataset_name: str, run_dir: Path
) -> tuple[
    DatasetAdapter,
    B1TrainConfig,
    StandardizationStats,
    Dinov3StandardizationStats,
    B1AdaptiveGateClassifier,
]:
    adapter = get_dataset(dataset_name)
    ensure_dinov3_dataset_supported(adapter)
    validate_dinov3_feature_set(adapter)
    pose_stats = load_pose_stats(adapter.pose_stats_path)
    validate_pose_stats_layout(pose_stats)
    visual_stats = load_visual_stats(dinov3_stats_path(dataset_name))
    validate_visual_stats_layout(visual_stats, dataset_name=dataset_name)
    validate_visual_stats_freshness(visual_stats, adapter.frames_path)
    expected_config = resolve_b1_config(
        B1_ADAPTIVE_GATE_CONFIG.seed,
        adapter.pose_stats_path,
        sha256_file(adapter.pose_stats_path),
        dinov3_stats_path(dataset_name),
        sha256_file(dinov3_stats_path(dataset_name)),
        adapter.quality_root,
        _quality_set_sha256(adapter),
    )
    try:
        config = validate_b1_training_run(
            run_dir,
            expected_config=expected_config,
            fields_allowed_to_differ=frozenset(
                {"seed", "trainable_param_count"}
            ),
        )
    except RuntimeError as exc:
        raise RuntimeError(f"run de treino B1 inválido em {run_dir}: {exc}") from exc
    if config.eval_stride != EVAL_STRIDE:
        raise ValueError(
            f"config.eval_stride ({config.eval_stride}) diverge de "
            f"EVAL_STRIDE ({EVAL_STRIDE})"
        )
    model = load_compatible_b1_checkpoint(run_dir / "checkpoint.pt", config)
    return adapter, config, pose_stats, visual_stats, model


def load_event_evaluation(dataset_name: str, run_dir: Path) -> EventEvaluation:
    adapter, config, pose_stats, visual_stats, model = _load_run_assets(
        dataset_name, run_dir
    )

    def prepare() -> tuple[pd.DataFrame, SplitEvaluator]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluation_model = model.to(device)
        evaluation_model.eval()
        frames = adapter.load_frames()
        pose_loader: Callable[[str], np.ndarray] = (
            lambda video_id: build_pose_features(
                video_id, pose_root=adapter.pose_root
            )[0]
        )
        visual_loader: Callable[[str], np.ndarray] = lambda video_id: read_features(
            dinov3_path(video_id, dinov3_root=adapter.dinov3_root)
        ).astype("float32")
        quality_loader: Callable[[str], np.ndarray] = lambda video_id: read_quality(
            quality_path(video_id, quality_root=adapter.quality_root)
        ).astype("float32")

        def evaluate_split(split: str) -> tuple[int, Predictions]:
            source = GatedFusionWindowDataset(
                frames,
                split,
                EVAL_STRIDE,
                pose_loader,
                visual_loader,
                quality_loader,
                drop_ignored=False,
            )
            predictions = _predict_with_identity(
                evaluation_model,
                source,
                pose_stats,
                visual_stats,
                device,
                batch_size=config.batch_size,
            )
            return len(source), predictions

        return frames, evaluate_split

    return EventEvaluation(config, prepare)


def _run_evaluate_locked(
    force: bool,
    dataset_name: str,
    run_dir: Path,
    lock: EventEvaluationLock,
) -> None:
    run_event_evaluation(
        force, "B1", run_dir, lock, lambda: load_event_evaluation(dataset_name, run_dir)
    )


def _guard_foreign_arm_run_dir(run_dir: Path) -> None:
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
    declared_run_name = data.get("run_name")
    foreign = declared_arm if declared_arm is not None else declared_run_name
    if (
        declared_arm is not None and declared_arm != B1_ADAPTIVE_GATE_CONFIG.arm
    ) or (declared_run_name is not None and declared_run_name != ARM_NAME):
        raise ValueError(
            f"run_dir {run_dir} pertence a outra arma ({foreign}); avaliação "
            "de eventos B1 recusa escrever nele"
        )


def run_evaluate(
    force: bool, dataset_name: str = "le2i", run_dir: Path | None = None
) -> None:
    if run_dir is None:
        run_dir = default_run_dir_for_arm(dataset_name, "B1")
    guard_not_arm_a_run_dir(run_dir, dataset_name)
    guard_not_arm_b0_run_dir(run_dir, dataset_name)
    validate_local_run_dir(run_dir, dataset_name)
    _guard_foreign_arm_run_dir(run_dir)
    with EventEvaluationLock(run_dir) as lock:
        _run_evaluate_locked(force, dataset_name, run_dir, lock)


def run_selftest() -> None:
    from gatefall.eval.baseline_b1.selftests.events import run_b1_events_selftest

    if not run_b1_events_selftest():
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Avalia eventos da arma B1 e grava event_metrics.json",
    )
    evaluate_parser.add_argument("--force", action="store_true")
    evaluate_parser.add_argument(
        "--dataset", default="le2i", choices=DINOV3_SUPPORTED_DATASET_IDENTIFIERS
    )
    evaluate_parser.add_argument("--run-dir", type=Path, default=None)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da avaliação B1")
    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(args.force, args.dataset, args.run_dir)
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
