"""Arma C0: fusão pose+SAM 3 V_t por concatenação simples, seguida de TCN dilatada rasa."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.fusion_dataset import FusionWindowDataset
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import (
    REFERENCE_RUN_ROOT,
    default_run_dir_for_arm,
    validate_local_run_dir,
)
from gatefall.sam3.dataset_guard import (
    SAM3_SUPPORTED_DATASET_IDENTIFIERS,
    ensure_sam3_dataset_supported,
)
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.sam3.features import load_v_t
from gatefall.train.shared.run_paths import repository_anchored_run_dir
from gatefall.train.baseline_c0.artifacts import load_compatible_c0_checkpoint, validate_c0_training_run
from gatefall.train.baseline_c0.config import C0_FUSION_CONFIG
from gatefall.train.baseline_c0.engine import _predict, _StandardizedFusionTorchDataset, run_c0_training
from gatefall.train.baseline_c0.run import (
    comparison_run_dirs,
    guard_not_comparison_run_dir,
    resolve_c0_config_for_inputs as _resolve_config,
)
from gatefall.train.shared.sam3_inputs import _validated_inputs
from gatefall.train.shared.classification_report import publish_classification_report

PROTECTED_ARTIFACT_NAMES = (
    "config.yaml",
    "metrics.json",
    "checkpoint.pt",
    "alarm_protocol.yaml",
    "event_metrics.json",
)


def _guard_run_dir(run_dir: Path, dataset_name: str) -> None:
    guard_not_comparison_run_dir(run_dir, dataset_name)
    validate_local_run_dir(run_dir, dataset_name)


def _protected_run_dirs(run_dir: Path, dataset_name: str) -> list[Path]:
    """Run dirs cujos artefatos o `report` do C0 nunca pode sobrescrever: o
    run pedido, o run canônico do C0 e os runs de comparação de A, B0 e B1."""
    return [
        run_dir.resolve(),
        repository_anchored_run_dir(default_run_dir_for_arm(dataset_name, "C0")),
        *comparison_run_dirs(dataset_name).values(),
    ]


def _guard_protected_output(
    run_dir: Path, output_path: Path, dataset_name: str
) -> None:
    resolved_output = output_path.resolve()
    for protected_run_dir in _protected_run_dirs(run_dir, dataset_name):
        for name in PROTECTED_ARTIFACT_NAMES:
            if resolved_output == (protected_run_dir / name).resolve():
                raise ValueError(
                    f"--output não pode apontar para o artefato protegido "
                    f"{name!r} em {protected_run_dir}"
                )
    if (
        resolved_output == REFERENCE_RUN_ROOT
        or REFERENCE_RUN_ROOT in resolved_output.parents
    ):
        raise ValueError(
            f"--output não pode apontar para dentro da referência histórica "
            f"somente leitura: {resolved_output}"
        )


def _split_sources(
    adapter: DatasetAdapter, frames: pd.DataFrame
) -> dict[str, FusionWindowDataset]:
    pose_loader = lambda video_id: build_pose_features(video_id, pose_root=adapter.pose_root)[0]
    visual_loader = lambda video_id: load_v_t(video_id, sam3_root=adapter.sam3_root)

    def source(split: str, stride: int) -> FusionWindowDataset:
        return FusionWindowDataset(
            frames, split, stride, pose_loader, visual_loader, visual_dim=V_T_DIM
        )

    return {
        "train": source("train", TRAIN_STRIDE),
        "val": source("val", EVAL_STRIDE),
        "test": source("test", EVAL_STRIDE),
    }


def run_train(
    force: bool,
    dataset_name: str = "le2i",
    run_dir: Path | None = None,
    seed: int = C0_FUSION_CONFIG.seed,
) -> None:
    if run_dir is None:
        run_dir = default_run_dir_for_arm(dataset_name, "C0")
    _guard_run_dir(run_dir, dataset_name)
    adapter = get_dataset(dataset_name)
    ensure_sam3_dataset_supported(adapter)

    inputs = _validated_inputs(adapter, dataset_name)
    config = _resolve_config(seed, adapter, inputs)
    sources = _split_sources(adapter, inputs.frames)

    run_c0_training(
        train_source=sources["train"],
        val_source=sources["val"],
        test_source=sources["test"],
        pose_stats=inputs.pose_stats,
        visual_stats=inputs.visual_stats,
        config=config,
        run_dir=run_dir,
        force=force,
        label_names=adapter.label_names,
    )


def run_report(
    dataset_name: str,
    run_dir: Path | None,
    output_path: Path,
    force: bool,
) -> bool:
    if run_dir is None:
        run_dir = default_run_dir_for_arm(dataset_name, "C0")
    _guard_run_dir(run_dir, dataset_name)
    _guard_protected_output(run_dir, output_path, dataset_name)
    if output_path.exists() and not force:
        raise RuntimeError(f"{output_path} já existe; use --force para sobrescrever")

    adapter = get_dataset(dataset_name)
    ensure_sam3_dataset_supported(adapter)
    inputs = _validated_inputs(adapter, dataset_name)

    expected_config = _resolve_config(C0_FUSION_CONFIG.seed, adapter, inputs)
    config = validate_c0_training_run(
        run_dir,
        expected_config=expected_config,
        fields_allowed_to_differ=frozenset({"seed", "trainable_param_count"}),
    )

    checkpoint_path = run_dir / "checkpoint.pt"
    model = load_compatible_c0_checkpoint(checkpoint_path, config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    split_sources = _split_sources(adapter, inputs.frames)

    def predictions():
        for split_name, source in split_sources.items():
            dataset = _StandardizedFusionTorchDataset(source, inputs.pose_stats, inputs.visual_stats)
            dataloader = DataLoader(
                dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
            )
            y_true, y_pred = _predict(model, dataloader, device)
            yield split_name, y_true, y_pred

    return publish_classification_report(
        predictions=predictions(),
        label_names=adapter.label_names,
        num_classes=config.num_classes,
        run_name=config.run_name,
        dataset_name=dataset_name,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        output_path=output_path,
    )


def run_selftest() -> None:
    from gatefall.data.selftests.fusion_dataset import run_fusion_dataset_selftest
    from gatefall.features.selftests.sam3_standardization import (
        run_sam3_standardization_selftest,
    )
    from gatefall.sam3.selftests.features import run_sam3_features_selftest
    from gatefall.train.baseline_c0.selftests.artifacts import run_c0_artifacts_selftest
    from gatefall.train.baseline_c0.selftests.config import run_c0_config_selftest
    from gatefall.train.baseline_c0.selftests.engine import run_c0_engine_selftest
    from gatefall.train.baseline_c0.selftests.cli import run_c0_fusion_selftest
    from gatefall.train.baseline_c0.selftests.model import run_c0_model_selftest

    results = [
        run_sam3_features_selftest(),
        run_sam3_standardization_selftest(),
        run_fusion_dataset_selftest(),
        run_c0_model_selftest(),
        run_c0_config_selftest(),
        run_c0_engine_selftest(),
        run_c0_artifacts_selftest(),
        run_c0_fusion_selftest(),
    ]
    if not all(results):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Treina a arma C0 (fusão pose+SAM 3 V_t) e grava config.yaml/metrics.json/checkpoint.pt",
    )
    train_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o run_dir já existente"
    )
    train_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )
    train_parser.add_argument("--run-dir", type=Path, default=None)
    train_parser.add_argument("--seed", type=int, default=C0_FUSION_CONFIG.seed)
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas da fusão pose+SAM 3"
    )

    report_parser = subparsers.add_parser(
        "report",
        help=(
            "Gera diagnóstico de classificação a partir de um run C0 já "
            "treinado, sem modificar nenhum artefato existente"
        ),
    )
    report_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o --output já existente"
    )
    report_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )
    report_parser.add_argument("--run-dir", type=Path, default=None)
    report_parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if args.command in ("train", "report") and args.run_dir is None:
        args.run_dir = default_run_dir_for_arm(args.dataset, "C0")
    if args.command == "train":
        run_train(
            force=args.force, dataset_name=args.dataset, run_dir=args.run_dir, seed=args.seed
        )
    elif args.command == "selftest":
        run_selftest()
    elif args.command == "report":
        output_path = args.output or (args.run_dir / "classification_report.json")
        ok = run_report(
            dataset_name=args.dataset,
            run_dir=args.run_dir,
            output_path=output_path,
            force=args.force,
        )
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
