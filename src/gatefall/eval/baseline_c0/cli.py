"""Avaliação de eventos da arma C0 sobre o protocolo Le2i-CS."""

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
from gatefall.data.fusion_dataset import FusionWindowDataset
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.eval.shared.event_artifacts import EventEvaluationLock
from gatefall.eval.shared.orchestration import (
    EventEvaluation,
    Predictions,
    SplitEvaluator,
    run_event_evaluation,
)
from gatefall.features.sam3_standardization import (
    Sam3StandardizationStats,
    apply_standardization as apply_visual_standardization,
)
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization as apply_pose_standardization,
)
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import default_run_dir_for_arm, validate_local_run_dir
from gatefall.sam3.dataset_guard import ensure_sam3_dataset_supported
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.sam3.features import load_v_t
from gatefall.train.baseline_c0.artifacts import (
    load_compatible_c0_checkpoint,
    validate_c0_training_run,
)
from gatefall.train.baseline_c0.config import C0_FUSION_CONFIG, C0TrainConfig
from gatefall.train.baseline_c0.model import C0FusionClassifier
from gatefall.train.baseline_c0.run import guard_not_comparison_run_dir
from gatefall.train.baseline_c0.run import resolve_c0_config_for_inputs as resolve_c0_config
from gatefall.train.shared.run_paths import repository_anchored_run_dir
from gatefall.train.shared.sam3_inputs import _validated_inputs


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
    visual_stats: Sam3StandardizationStats,
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


def _load_run_assets(
    dataset_name: str, run_dir: Path
) -> tuple[
    DatasetAdapter,
    C0TrainConfig,
    StandardizationStats,
    Sam3StandardizationStats,
    C0FusionClassifier,
    pd.DataFrame,
]:
    adapter = get_dataset(dataset_name)
    ensure_sam3_dataset_supported(adapter)
    inputs = _validated_inputs(adapter, dataset_name)
    expected_config = resolve_c0_config(C0_FUSION_CONFIG.seed, adapter, inputs)
    try:
        config = validate_c0_training_run(
            run_dir,
            expected_config=expected_config,
            fields_allowed_to_differ=frozenset(
                {"seed", "trainable_param_count"}
            ),
        )
    except RuntimeError as exc:
        raise RuntimeError(f"run de treino C0 inválido em {run_dir}: {exc}") from exc
    if config.eval_stride != EVAL_STRIDE:
        raise ValueError(
            f"config.eval_stride ({config.eval_stride}) diverge de "
            f"EVAL_STRIDE ({EVAL_STRIDE})"
        )
    model = load_compatible_c0_checkpoint(run_dir / "checkpoint.pt", config)
    return adapter, config, inputs.pose_stats, inputs.visual_stats, model, inputs.frames


def _run_evaluate_locked(
    force: bool,
    dataset_name: str,
    run_dir: Path,
    lock: EventEvaluationLock,
) -> None:
    def load_run() -> EventEvaluation:
        adapter, config, pose_stats, visual_stats, model, frames = _load_run_assets(
            dataset_name, run_dir
        )

        def prepare() -> tuple[pd.DataFrame, SplitEvaluator]:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            evaluation_model = model.to(device)
            evaluation_model.eval()
            pose_loader: Callable[[str], np.ndarray] = (
                lambda video_id: build_pose_features(
                    video_id, pose_root=adapter.pose_root
                )[0]
            )
            visual_loader: Callable[[str], np.ndarray] = lambda video_id: load_v_t(
                video_id, sam3_root=adapter.sam3_root
            )

            def evaluate_split(split: str) -> tuple[int, Predictions]:
                source = FusionWindowDataset(
                    frames,
                    split,
                    EVAL_STRIDE,
                    pose_loader,
                    visual_loader,
                    drop_ignored=False,
                    visual_dim=V_T_DIM,
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

    run_event_evaluation(force, "C0", run_dir, lock, load_run)


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
        declared_arm is not None and declared_arm != C0_FUSION_CONFIG.arm
    ) or (
        declared_run_name is not None
        and declared_run_name != C0_FUSION_CONFIG.run_name
    ):
        raise ValueError(
            f"run_dir {run_dir} pertence a outra arma ({foreign}); avaliação "
            "de eventos C0 recusa escrever nele"
        )


def run_evaluate(
    force: bool, dataset_name: str = "le2i", run_dir: Path | None = None
) -> None:
    if dataset_name != "le2i":
        raise ValueError("avaliação de eventos C0 suporta somente le2i (CS)")
    if run_dir is None:
        run_dir = default_run_dir_for_arm(dataset_name, "C0")
    guard_not_comparison_run_dir(run_dir, dataset_name)
    c1_run_dir = repository_anchored_run_dir(default_run_dir_for_arm(dataset_name, "C1"))
    resolved = run_dir.resolve()
    if (
        resolved == c1_run_dir
        or resolved in c1_run_dir.parents
        or c1_run_dir in resolved.parents
    ):
        raise ValueError(
            f"run_dir {run_dir} coincide com, contém ou está dentro do "
            "run_dir da arma C1"
        )
    validate_local_run_dir(run_dir, dataset_name)
    _guard_foreign_arm_run_dir(run_dir)
    with EventEvaluationLock(run_dir) as lock:
        _run_evaluate_locked(force, dataset_name, run_dir, lock)


def run_selftest() -> None:
    from gatefall.eval.baseline_c0.selftests.events import run_c0_events_selftest

    if not run_c0_events_selftest():
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Avalia eventos da arma C0 e grava event_metrics.json",
    )
    evaluate_parser.add_argument("--force", action="store_true")
    evaluate_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    evaluate_parser.add_argument("--run-dir", type=Path, default=None)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da avaliação C0")
    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(args.force, args.dataset, args.run_dir)
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
