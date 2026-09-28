"""Arma C1: fusão adaptativa de pose e SAM 3 V_t com qualidade por quadro."""

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features
from gatefall.pose.quality import compute_pose_quality
from gatefall.runs import (
    LOCAL_RUN_ROOTS,
    REFERENCE_RUN_ROOT,
    REPOSITORY_ROOT,
    default_run_dir_for_arm,
    validate_local_run_dir,
)
from gatefall.sam3.dataset_guard import SAM3_SUPPORTED_DATASET_IDENTIFIERS, ensure_sam3_dataset_supported
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.sam3.features import load_v_t
from gatefall.sam3.quality import compute_sam3_quality
from gatefall.sam3.storage import read_sam_score, sam3_path
from gatefall.train.shared.gated_engine import _StandardizedGatedFusionTorchDataset, _predict
from gatefall.train.shared.run_paths import repository_anchored_run_dir
from gatefall.train.shared.sam3_inputs import _validated_inputs
from gatefall.train.baseline_c1.artifacts import load_compatible_c1_checkpoint, validate_c1_training_run
from gatefall.train.baseline_c1.config import C1_ADAPTIVE_GATE_CONFIG
from gatefall.train.baseline_c1.engine import run_c1_training
from gatefall.train.baseline_c1.run import ARM_NAME, comparison_run_dirs, guard_not_comparison_run_dir, resolve_c1_config_for_inputs as _resolve_config
from gatefall.train.shared.metrics import (
    BINARY_POSITIVE_LABELS,
    RESTRICTED_CLASSES,
    binary_projection_summary,
    class_support_table,
    classification_summary,
    macro_f1_policy_summary,
    restricted_macro_f1,
    support,
)

PROTECTED_ARTIFACT_NAMES = (
    "config.yaml", "metrics.json", "checkpoint.pt", "alarm_protocol.yaml", "event_metrics.json"
)


def _guard_run_dir(run_dir: Path, dataset_name: str) -> None:
    guard_not_comparison_run_dir(run_dir, dataset_name)
    validate_local_run_dir(run_dir, dataset_name)


def _protected_run_dirs(run_dir: Path, dataset_name: str) -> list[Path]:
    return [run_dir.resolve(),
            repository_anchored_run_dir(default_run_dir_for_arm(dataset_name, ARM_NAME)),
            *comparison_run_dirs(dataset_name).values()]


def _guard_protected_output(run_dir: Path, output_path: Path, dataset_name: str) -> None:
    resolved_output = output_path.resolve()
    for protocol, root in LOCAL_RUN_ROOTS.items():
        protected = (REPOSITORY_ROOT / root).resolve()
        if protocol != dataset_name and (
            resolved_output == protected or protected in resolved_output.parents
        ):
            raise ValueError(f"--output pertence ao protocolo {protocol!r}")
    for protected in comparison_run_dirs(dataset_name).values():
        if resolved_output == protected or protected in resolved_output.parents:
            raise ValueError(
                f"--output não pode apontar para dentro do run de comparação {protected}"
            )
    for protected in _protected_run_dirs(run_dir, dataset_name):
        for name in PROTECTED_ARTIFACT_NAMES:
            if resolved_output == (protected / name).resolve():
                raise ValueError(f"--output não pode apontar para o artefato protegido {name!r} em {protected}")
    if resolved_output == REFERENCE_RUN_ROOT or REFERENCE_RUN_ROOT in resolved_output.parents:
        raise ValueError(f"--output não pode apontar para dentro da referência histórica: {resolved_output}")


def _split_sources(adapter: DatasetAdapter, frames: pd.DataFrame) -> dict[str, GatedFusionWindowDataset]:
    pose_loader = lambda video_id: build_pose_features(video_id, pose_root=adapter.pose_root)[0]
    visual_loader = lambda video_id: load_v_t(video_id, sam3_root=adapter.sam3_root)

    def quality_loader(video_id: str) -> np.ndarray:
        q_pose = compute_pose_quality(video_id, pose_root=adapter.pose_root).q_pose
        path = sam3_path(video_id, sam3_root=adapter.sam3_root)
        visual = load_v_t(video_id, sam3_root=adapter.sam3_root)
        q_sam3 = compute_sam3_quality(read_sam_score(path), visual[:, 0])
        if q_pose.shape != q_sam3.shape:
            raise ValueError(f"video_id={video_id!r}: qualidade pose e SAM 3 têm K diferentes")
        return np.column_stack((q_pose, q_sam3)).astype(np.float32)

    return {
        "train": GatedFusionWindowDataset(frames, "train", TRAIN_STRIDE, pose_loader, visual_loader, quality_loader, visual_dim=V_T_DIM),
        "val": GatedFusionWindowDataset(frames, "val", EVAL_STRIDE, pose_loader, visual_loader, quality_loader, visual_dim=V_T_DIM),
        "test": GatedFusionWindowDataset(frames, "test", EVAL_STRIDE, pose_loader, visual_loader, quality_loader, visual_dim=V_T_DIM),
    }


def run_train(force: bool, dataset_name: str = "le2i", run_dir: Path | None = None, seed: int = C1_ADAPTIVE_GATE_CONFIG.seed) -> None:
    if run_dir is None:
        run_dir = default_run_dir_for_arm(dataset_name, ARM_NAME)
    _guard_run_dir(run_dir, dataset_name)
    adapter = get_dataset(dataset_name)
    ensure_sam3_dataset_supported(adapter)
    inputs = _validated_inputs(adapter, dataset_name)
    config = _resolve_config(seed, adapter, inputs)
    sources = _split_sources(adapter, inputs.frames)
    run_c1_training(sources["train"], sources["val"], sources["test"], inputs.pose_stats, inputs.visual_stats, config, run_dir, force, adapter.label_names)


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
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in formatted_rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def _print_macro_f1_policy_summary(policy_summary: dict) -> None:
    restricted = policy_summary["restricted_classes"]
    excluded = policy_summary["excluded_classes"]
    with_support = policy_summary["classes_with_positive_train_support"]
    if policy_summary["matches_configured_restriction"]:
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
    run_dir: Path | None,
    output_path: Path,
    force: bool,
) -> bool:
    if run_dir is None:
        run_dir = default_run_dir_for_arm(dataset_name, ARM_NAME)
    _guard_run_dir(run_dir, dataset_name)
    _guard_protected_output(run_dir, output_path, dataset_name)
    if output_path.exists() and not force:
        raise RuntimeError(f"{output_path} já existe; use --force para sobrescrever")

    adapter = get_dataset(dataset_name)
    ensure_sam3_dataset_supported(adapter)
    inputs = _validated_inputs(adapter, dataset_name)
    expected_config = _resolve_config(C1_ADAPTIVE_GATE_CONFIG.seed, adapter, inputs)
    config = validate_c1_training_run(
        run_dir,
        expected_config=expected_config,
        fields_allowed_to_differ=frozenset({"seed", "trainable_param_count"}),
    )

    checkpoint_path = run_dir / "checkpoint.pt"
    model = load_compatible_c1_checkpoint(checkpoint_path, config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    split_sources = _split_sources(adapter, inputs.frames)

    splits_report: dict[str, dict] = {}
    mismatches: list[dict] = []
    support_by_split: dict[str, dict[int, int]] = {}

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open(encoding="utf-8") as f:
        stored_metrics = json.load(f)
    stored_final = stored_metrics["final"]

    for split_name, source in split_sources.items():
        dataset = _StandardizedGatedFusionTorchDataset(source, inputs.pose_stats, inputs.visual_stats)
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
    from gatefall.train.baseline_c1.selftests.cli import run_c1_selftest

    if not run_c1_selftest():
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help=(
            "Treina a arma C1 (fusão adaptativa por gate) e grava "
            "config.yaml/metrics.json/checkpoint.pt"
        ),
    )
    train_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o run_dir já existente"
    )
    train_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )
    train_parser.add_argument("--run-dir", type=Path, default=None)
    train_parser.add_argument("--seed", type=int, default=C1_ADAPTIVE_GATE_CONFIG.seed)
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas da fusão adaptativa"
    )

    report_parser = subparsers.add_parser(
        "report",
        help=(
            "Gera diagnóstico de classificação a partir de um run C1 já "
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
        args.run_dir = default_run_dir_for_arm(args.dataset, ARM_NAME)
    if args.command == "train":
        run_train(
            force=args.force,
            dataset_name=args.dataset,
            run_dir=args.run_dir,
            seed=args.seed,
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
