"""Arma A: TCN dilatada rasa treinada sobre o vetor de pose de 134 dimensões."""

import argparse
import json
import math
import os
import sys
import uuid
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
from gatefall.runs import REFERENCE_RUN_ROOT, validate_local_run_dir
from gatefall.train.artifacts import load_compatible_checkpoint, validate_training_run
from gatefall.train.artifacts_selftest import run_artifacts_selftest
from gatefall.train.baseline_a_selftest import run_baseline_a_selftest
from gatefall.train.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.engine import _StandardizedTorchDataset, _predict, run_training
from gatefall.train.engine_selftest import run_engine_selftest
from gatefall.train.metrics import (
    BINARY_POSITIVE_LABELS,
    RESTRICTED_CLASSES,
    binary_projection_summary,
    class_support_table,
    classification_summary,
    macro_f1_policy_summary,
    restricted_macro_f1,
    support,
)
from gatefall.train.metrics_selftest import run_metrics_selftest
from gatefall.train.tcn_selftest import run_tcn_selftest

RUN_DIR = Path("runs/local/le2i/baseline_a")
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
    run_dir: Path = RUN_DIR,
    seed: int = BASELINE_A_CONFIG.seed,
) -> None:
    validate_local_run_dir(run_dir)
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


def _print_class_support_table(rows: list[dict]) -> None:
    columns = (
        "id",
        "label",
        "train_support",
        "val_support",
        "test_support",
        "included_in_macro_f1",
    )
    formatted_rows = [
        {column: str(row[column]) for column in columns} for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    for row in formatted_rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def _print_macro_f1_policy_summary(policy_summary: dict) -> None:
    restricted = policy_summary["restricted_classes"]
    excluded = policy_summary["excluded_classes"]
    with_support = policy_summary["classes_with_positive_train_support"]
    matches = policy_summary["matches_configured_restriction"]
    if matches:
        print(
            f"política de macro-F1: classes restritas {restricted}, "
            f"classes excluídas {excluded}, classes com suporte de treino "
            f"positivo {with_support} (conjuntos coincidem)"
        )
    else:
        print(
            f"ATENÇÃO: DESCASAMENTO na política de macro-F1: classes restritas "
            f"configuradas {restricted}, classes excluídas {excluded}, mas "
            f"classes com suporte de treino positivo {with_support} "
            f"(conjuntos NÃO coincidem)"
        )


def run_report(
    dataset_name: str,
    run_dir: Path,
    output_path: Path,
    force: bool,
) -> bool:
    validate_local_run_dir(run_dir)
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

    splits_report: dict[str, dict] = {}
    mismatches: list[dict] = []
    support_by_split: dict[str, dict[int, int]] = {}

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open(encoding="utf-8") as f:
        stored_metrics = json.load(f)
    stored_final = stored_metrics["final"]

    for split_name, source in split_sources.items():
        dataset = _StandardizedTorchDataset(source, stats)
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        y_true, y_pred = _predict(model, dataloader, device)

        summary = classification_summary(y_true, y_pred, adapter.label_names, config.num_classes)
        binary = binary_projection_summary(y_true, y_pred, BINARY_POSITIVE_LABELS)
        splits_report[split_name] = {
            "confusion_matrix": summary["confusion_matrix"],
            "per_class": summary["per_class"],
            "binary_fall_fallen": binary,
        }

        stored_split = stored_final[split_name]
        macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred, config.num_classes)
        stored_macro_f1 = stored_split["macro_f1_restricted"]
        if not math.isclose(stored_macro_f1, macro_f1, abs_tol=1e-12, rel_tol=0):
            mismatches.append(
                {
                    "split": split_name,
                    "field": "macro_f1_restricted",
                    "stored": stored_macro_f1,
                    "recomputed": macro_f1,
                }
            )
        stored_f1_by_class = stored_split["f1_by_class"]
        for c in RESTRICTED_CLASSES:
            stored_value = stored_f1_by_class[str(c)]
            recomputed_value = f1_by_class[c]
            if not math.isclose(stored_value, recomputed_value, abs_tol=1e-12, rel_tol=0):
                mismatches.append(
                    {
                        "split": split_name,
                        "field": f"f1_by_class[{c}]",
                        "stored": stored_value,
                        "recomputed": recomputed_value,
                    }
                )
        recomputed_support = support(y_true, config.num_classes)
        support_by_split[split_name] = recomputed_support
        stored_support = stored_split["support"]
        for c in range(config.num_classes):
            label_name = adapter.label_names[c]
            stored_count = stored_support[label_name]
            recomputed_count = recomputed_support[c]
            if stored_count != recomputed_count:
                mismatches.append(
                    {
                        "split": split_name,
                        "field": f"support[{label_name}]",
                        "stored": stored_count,
                        "recomputed": recomputed_count,
                    }
                )

    ok = len(mismatches) == 0

    support_table = class_support_table(
        adapter.label_names,
        support_by_split["train"],
        support_by_split["val"],
        support_by_split["test"],
    )
    policy_summary = macro_f1_policy_summary(support_by_split["train"])

    report = {
        "run_name": config.run_name,
        "dataset": dataset_name,
        "run_dir": str(run_dir),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": device,
        "splits": splits_report,
        "verification_against_metrics_json": {"ok": ok, "mismatches": mismatches},
        "class_support_table": support_table,
        "macro_f1_policy": policy_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)

    print(f"{output_path}: relatório de classificação gravado (run_name={config.run_name})")
    _print_class_support_table(support_table)
    _print_macro_f1_policy_summary(policy_summary)
    if not ok:
        print(
            f"verificação contra metrics.json falhou: {len(mismatches)} divergência(s)",
            file=sys.stderr,
        )
    return ok


def run_selftest() -> None:
    tcn_ok = run_tcn_selftest()
    metrics_ok = run_metrics_selftest()
    engine_ok = run_engine_selftest()
    baseline_a_ok = run_baseline_a_selftest()
    artifacts_ok = run_artifacts_selftest()
    if not (tcn_ok and metrics_ok and engine_ok and baseline_a_ok and artifacts_ok):
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
    train_parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
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
    report_parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    report_parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
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
