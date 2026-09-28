"""Avaliação de eventos da arma C1 sobre o protocolo Le2i-CS."""

import argparse
import json
import sys
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import torch
import yaml

from gatefall.config import EVAL_STRIDE
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.data.windowing import build_window_index
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.eval.shared.alarm_protocol import (
    BASELINE_A_ALARM_PROTOCOL,
    load_alarm_protocol,
    save_alarm_protocol,
)
from gatefall.eval.shared.event_artifacts import (
    EventEvaluationLock,
    _promote_event_outputs,
    _recover_event_publication,
    _require_event_lock,
    validate_event_metrics,
)
from gatefall.eval.shared.events import extract_label_segments, split_event_report
from gatefall.features.sam3_standardization import (
    Sam3StandardizationStats,
    apply_standardization as apply_visual_standardization,
)
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization as apply_pose_standardization,
)
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features
from gatefall.pose.quality import compute_pose_quality
from gatefall.runs import default_run_dir_for_arm, validate_local_run_dir
from gatefall.sam3.dataset_guard import (
    SAM3_SUPPORTED_DATASET_IDENTIFIERS,
    ensure_sam3_dataset_supported,
)
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.sam3.features import load_v_t
from gatefall.sam3.quality import compute_sam3_quality
from gatefall.sam3.storage import read_sam_score, sam3_path
from gatefall.train.shared.sam3_inputs import _validated_inputs
from gatefall.train.baseline_c1.artifacts import (
    load_compatible_c1_checkpoint,
    validate_c1_training_run,
)
from gatefall.train.baseline_c1.config import C1_ADAPTIVE_GATE_CONFIG, C1TrainConfig
from gatefall.train.baseline_c1.run import resolve_c1_config_for_inputs as _resolve_config
from gatefall.train.baseline_c1.model import C1AdaptiveGateClassifier
from gatefall.train.baseline_c1.run import guard_not_comparison_run_dir

ARM_NAME = "c1_adaptive_gate"


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
    C1TrainConfig,
    StandardizationStats,
    Sam3StandardizationStats,
    C1AdaptiveGateClassifier,
    pd.DataFrame,
]:
    adapter = get_dataset(dataset_name)
    ensure_sam3_dataset_supported(adapter)
    inputs = _validated_inputs(adapter, dataset_name)
    expected_config = _resolve_config(C1_ADAPTIVE_GATE_CONFIG.seed, adapter, inputs)
    try:
        config = validate_c1_training_run(
            run_dir,
            expected_config=expected_config,
            fields_allowed_to_differ=frozenset({"seed", "trainable_param_count"}),
        )
    except RuntimeError as exc:
        raise RuntimeError(f"run de treino C1 inválido em {run_dir}: {exc}") from exc
    if config.eval_stride != EVAL_STRIDE:
        raise ValueError(
            f"config.eval_stride ({config.eval_stride}) diverge de "
            f"EVAL_STRIDE ({EVAL_STRIDE})"
        )
    model = load_compatible_c1_checkpoint(run_dir / "checkpoint.pt", config)
    return adapter, config, inputs.pose_stats, inputs.visual_stats, model, inputs.frames


def _quality_for_video(video_id: str, adapter: DatasetAdapter) -> np.ndarray:
    q_pose = compute_pose_quality(video_id, pose_root=adapter.pose_root).q_pose
    path = sam3_path(video_id, sam3_root=adapter.sam3_root)
    visual = load_v_t(video_id, sam3_root=adapter.sam3_root)
    q_sam3 = compute_sam3_quality(read_sam_score(path), visual[:, 0])
    if q_pose.shape != q_sam3.shape:
        raise ValueError(f"video_id={video_id!r}: qualidade pose e SAM 3 têm K diferentes")
    return np.column_stack((q_pose, q_sam3)).astype(np.float32)


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

    adapter, config, pose_stats, visual_stats, model, frames = _load_run_assets(
        dataset_name, run_dir
    )
    event_outputs = (alarm_protocol_path, event_metrics_path)
    present_outputs = [path for path in event_outputs if path.is_file()]
    previous_pair_valid = False
    if len(present_outputs) == len(event_outputs):
        try:
            protocol = load_alarm_protocol(alarm_protocol_path)
            if protocol != BASELINE_A_ALARM_PROTOCOL:
                raise ValueError("alarm_protocol.yaml incompatível com a arma C1")
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
    pose_loader: Callable[[str], np.ndarray] = lambda video_id: build_pose_features(
        video_id, pose_root=adapter.pose_root
    )[0]
    visual_loader: Callable[[str], np.ndarray] = lambda video_id: load_v_t(
        video_id, sam3_root=adapter.sam3_root
    )
    quality_loader: Callable[[str], np.ndarray] = lambda video_id: _quality_for_video(
        video_id, adapter
    )

    splits: dict[str, dict] = {}
    for split in ("val", "test"):
        source = GatedFusionWindowDataset(
            frames,
            split,
            EVAL_STRIDE,
            pose_loader,
            visual_loader,
            quality_loader,
            drop_ignored=False,
            visual_dim=V_T_DIM,
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
        declared_arm is not None and declared_arm != C1_ADAPTIVE_GATE_CONFIG.arm
    ) or (declared_run_name is not None and declared_run_name != ARM_NAME):
        raise ValueError(
            f"run_dir {run_dir} pertence a outra arma ({foreign}); avaliação "
            "de eventos C1 recusa escrever nele"
        )


def run_evaluate(
    force: bool, dataset_name: str = "le2i", run_dir: Path | None = None
) -> None:
    if dataset_name != "le2i":
        raise ValueError("avaliação de eventos C1 suporta somente le2i (CS)")
    if run_dir is None:
        run_dir = default_run_dir_for_arm(dataset_name, ARM_NAME)
    guard_not_comparison_run_dir(run_dir, dataset_name)
    validate_local_run_dir(run_dir, dataset_name)
    _guard_foreign_arm_run_dir(run_dir)
    with EventEvaluationLock(run_dir) as lock:
        _run_evaluate_locked(force, dataset_name, run_dir, lock)


def run_selftest() -> None:
    from gatefall.eval.baseline_c1.selftests.events import run_c1_events_selftest

    if not run_c1_events_selftest():
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Avalia eventos da arma C1 e grava event_metrics.json",
    )
    evaluate_parser.add_argument("--force", action="store_true")
    evaluate_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )
    evaluate_parser.add_argument("--run-dir", type=Path, default=None)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da avaliação C1")
    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(args.force, args.dataset, args.run_dir)
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
