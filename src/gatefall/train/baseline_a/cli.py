"""Arma A: TCN dilatada rasa treinada sobre o vetor de pose de 134 dimensões."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.features.standardization import load_stats, validate_stats_layout
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM, build_pose_features
from gatefall.runs import REFERENCE_RUN_ROOT, default_run_dir, validate_local_run_dir
from gatefall.train.baseline_a.artifacts import load_compatible_checkpoint, validate_training_run
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.baseline_a.engine import _StandardizedTorchDataset, _predict, run_training
from gatefall.train.shared.classification_report import publish_classification_report

PROTECTED_ARTIFACT_NAMES = (
    "config.yaml",
    "metrics.json",
    "checkpoint.pt",
    "alarm_protocol.yaml",
    "event_metrics.json",
)


def _resolve_config(seed: int, stats_path: Path, stats_sha256: str) -> TrainConfig:
    return replace(
        BASELINE_A_CONFIG,
        seed=seed,
        standardization_stats_path=str(stats_path),
        standardization_stats_sha256=stats_sha256,
    )


def run_train(
    force: bool,
    dataset_name: str = "le2i",
    run_dir: Path | None = None,
    seed: int = BASELINE_A_CONFIG.seed,
) -> None:
    if run_dir is None:
        run_dir = default_run_dir(dataset_name)
    validate_local_run_dir(run_dir, dataset_name)
    adapter = get_dataset(dataset_name)
    stats = load_stats(adapter.pose_stats_path)
    validate_stats_layout(stats)
    config = _resolve_config(
        seed, adapter.pose_stats_path, sha256_file(adapter.pose_stats_path)
    )
    frames = adapter.load_frames()
    loader = lambda video_id: build_pose_features(
        video_id, pose_root=adapter.pose_root
    )[0]

    train_source = PoseWindowDataset(frames, "train", TRAIN_STRIDE, loader)
    val_source = PoseWindowDataset(frames, "val", EVAL_STRIDE, loader)
    test_source = PoseWindowDataset(frames, "test", EVAL_STRIDE, loader)

    run_training(
        input_dim=POSE_FEATURE_DIM,
        train_source=train_source,
        val_source=val_source,
        test_source=test_source,
        stats=stats,
        config=config,
        run_dir=run_dir,
        force=force,
        label_names=adapter.label_names,
    )


def _guard_protected_output(run_dir: Path, output_path: Path) -> None:
    resolved_output = output_path.resolve()
    for name in PROTECTED_ARTIFACT_NAMES:
        if resolved_output == (run_dir / name).resolve():
            raise ValueError(
                f"--output não pode apontar para o artefato protegido {name!r} "
                f"em {run_dir}"
            )
    if (
        resolved_output == REFERENCE_RUN_ROOT
        or REFERENCE_RUN_ROOT in resolved_output.parents
    ):
        raise ValueError(
            f"--output não pode apontar para dentro da referência histórica "
            f"somente leitura: {resolved_output}"
        )


def run_report(
    dataset_name: str,
    run_dir: Path | None,
    output_path: Path,
    force: bool,
) -> bool:
    if run_dir is None:
        run_dir = default_run_dir(dataset_name)
    validate_local_run_dir(run_dir, dataset_name)
    _guard_protected_output(run_dir, output_path)
    if output_path.exists() and not force:
        raise RuntimeError(
            f"{output_path} já existe; use --force para sobrescrever"
        )

    adapter = get_dataset(dataset_name)
    stats = load_stats(adapter.pose_stats_path)
    validate_stats_layout(stats)
    expected_config = _resolve_config(
        BASELINE_A_CONFIG.seed, adapter.pose_stats_path, sha256_file(adapter.pose_stats_path)
    )
    config = validate_training_run(
        run_dir,
        expected_config=expected_config,
        fields_allowed_to_differ=frozenset({"seed"}),
    )

    checkpoint_path = run_dir / "checkpoint.pt"
    model = load_compatible_checkpoint(checkpoint_path, config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    frames = adapter.load_frames()
    loader = lambda video_id: build_pose_features(
        video_id, pose_root=adapter.pose_root
    )[0]

    split_sources = {
        "train": PoseWindowDataset(frames, "train", TRAIN_STRIDE, loader),
        "val": PoseWindowDataset(frames, "val", EVAL_STRIDE, loader),
        "test": PoseWindowDataset(frames, "test", EVAL_STRIDE, loader),
    }

    def predictions():
        for split_name, source in split_sources.items():
            dataset = _StandardizedTorchDataset(source, stats)
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
    from gatefall.train.baseline_a.selftests.artifacts import run_artifacts_selftest
    from gatefall.train.baseline_a.selftests.cli import run_baseline_a_selftest
    from gatefall.train.baseline_a.selftests.engine import run_engine_selftest
    from gatefall.train.shared.selftests.classification_report import run_classification_report_selftest
    from gatefall.train.shared.selftests.determinism import run_determinism_selftest
    from gatefall.train.shared.selftests.metrics import run_metrics_selftest
    from gatefall.train.shared.selftests.tcn import run_tcn_selftest

    tcn_ok = run_tcn_selftest()
    metrics_ok = run_metrics_selftest()
    report_ok = run_classification_report_selftest()
    determinism_ok = run_determinism_selftest()
    engine_ok = run_engine_selftest()
    baseline_a_ok = run_baseline_a_selftest()
    artifacts_ok = run_artifacts_selftest()
    if not (tcn_ok and metrics_ok and report_ok and determinism_ok and engine_ok and baseline_a_ok and artifacts_ok):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Treina a arma A (TCN) e grava config.yaml/metrics.json/checkpoint.pt"
    )
    train_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o run_dir já existente"
    )
    train_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)
    train_parser.add_argument("--run-dir", type=Path, default=None)
    train_parser.add_argument("--seed", type=int, default=BASELINE_A_CONFIG.seed)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da TCN e das métricas")

    report_parser = subparsers.add_parser(
        "report",
        help=(
            "Gera diagnóstico de classificação (matriz de confusão 10x10, "
            "métricas por classe e projeção binária fall/fallen) a partir de "
            "um run já treinado, sem modificar nenhum artefato existente"
        ),
    )
    report_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o --output já existente"
    )
    report_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)
    report_parser.add_argument("--run-dir", type=Path, default=None)
    report_parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if args.command in ("train", "report") and args.run_dir is None:
        args.run_dir = default_run_dir(args.dataset)
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
