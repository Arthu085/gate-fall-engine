"""Arma C0: fusão pose+SAM 3 V_t por concatenação simples, seguida de TCN dilatada rasa."""

import argparse
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.fusion_dataset import FusionWindowDataset
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.features.sam3_standardization import Sam3StandardizationStats
from gatefall.features.sam3_standardization import load_stats as load_visual_stats
from gatefall.features.sam3_standardization import (
    validate_stats_freshness as validate_visual_stats_freshness,
)
from gatefall.features.sam3_standardization import (
    validate_stats_layout as validate_visual_stats_layout,
)
from gatefall.features.standardization import StandardizationStats
from gatefall.features.standardization import load_stats as load_pose_stats
from gatefall.features.standardization import (
    validate_stats_layout as validate_pose_stats_layout,
)
from gatefall.features.standardize_sam3 import SAM3_STATS_PATH, split_by_video
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
from gatefall.sam3.features import collect_sam3_provenance, load_v_t, sam3_set_sha256
from gatefall.train.b1_run import repository_anchored_run_dir
from gatefall.train.c0_artifacts import load_compatible_c0_checkpoint, validate_c0_training_run
from gatefall.train.c0_config import C0_FUSION_CONFIG, C0TrainConfig
from gatefall.train.c0_engine import _predict, _StandardizedFusionTorchDataset, run_c0_training
from gatefall.train.c0_run import (
    ARM_NAME,
    comparison_run_dirs,
    guard_not_comparison_run_dir,
    resolve_c0_config,
)
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

PROTECTED_ARTIFACT_NAMES = (
    "config.yaml",
    "metrics.json",
    "checkpoint.pt",
    "alarm_protocol.yaml",
    "event_metrics.json",
)


@dataclass(frozen=True)
class _ValidatedInputs:
    frames: pd.DataFrame
    pose_stats: StandardizationStats
    visual_stats: Sam3StandardizationStats
    sam3_features_sha256: str
    sam3_provenance: dict[str, str]


def _resolve_config(
    seed: int,
    adapter: DatasetAdapter,
    inputs: _ValidatedInputs,
) -> C0TrainConfig:
    return resolve_c0_config(
        seed,
        adapter.pose_stats_path,
        sha256_file(adapter.pose_stats_path),
        SAM3_STATS_PATH,
        sha256_file(SAM3_STATS_PATH),
        adapter.sam3_root,
        inputs.sam3_features_sha256,
        inputs.sam3_provenance,
    )


def _guard_run_dir(run_dir: Path, dataset_name: str) -> None:
    guard_not_comparison_run_dir(run_dir, dataset_name)
    validate_local_run_dir(run_dir, dataset_name)


def _protected_run_dirs(run_dir: Path, dataset_name: str) -> list[Path]:
    """Run dirs cujos artefatos o `report` do C0 nunca pode sobrescrever: o
    run pedido, o run canônico do C0 e os runs de comparação de A, B0 e B1."""
    return [
        run_dir.resolve(),
        repository_anchored_run_dir(default_run_dir_for_arm(dataset_name, ARM_NAME)),
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


def _validated_inputs(adapter: DatasetAdapter, dataset_name: str) -> _ValidatedInputs:
    """Valida identidade, proveniência e frescor de todas as fontes antes de montar janelas."""
    frames = adapter.load_frames()
    splits = split_by_video(frames)
    sam3_provenance = collect_sam3_provenance(splits, sam3_root=adapter.sam3_root)
    sam3_features_sha256 = sam3_set_sha256(list(splits), sam3_root=adapter.sam3_root)

    pose_stats = load_pose_stats(adapter.pose_stats_path)
    validate_pose_stats_layout(pose_stats)
    visual_stats = load_visual_stats(SAM3_STATS_PATH)
    validate_visual_stats_layout(visual_stats, dataset_name=dataset_name)
    validate_visual_stats_freshness(visual_stats, adapter.frames_path, sam3_features_sha256)
    return _ValidatedInputs(
        frames=frames,
        pose_stats=pose_stats,
        visual_stats=visual_stats,
        sam3_features_sha256=sam3_features_sha256,
        sam3_provenance=sam3_provenance,
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
        run_dir = default_run_dir_for_arm(dataset_name, ARM_NAME)
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

    splits_report: dict[str, dict] = {}
    mismatches: list[dict] = []
    support_by_split: dict[str, dict[int, int]] = {}

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open(encoding="utf-8") as f:
        stored_metrics = json.load(f)
    stored_final = stored_metrics["final"]

    for split_name, source in split_sources.items():
        dataset = _StandardizedFusionTorchDataset(source, inputs.pose_stats, inputs.visual_stats)
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
    from gatefall.data.fusion_dataset_selftest import run_fusion_dataset_selftest
    from gatefall.features.sam3_standardization_selftest import (
        run_sam3_standardization_selftest,
    )
    from gatefall.sam3.features_selftest import run_sam3_features_selftest
    from gatefall.train.c0_artifacts_selftest import run_c0_artifacts_selftest
    from gatefall.train.c0_config_selftest import run_c0_config_selftest
    from gatefall.train.c0_engine_selftest import run_c0_engine_selftest
    from gatefall.train.c0_fusion_selftest import run_c0_fusion_selftest
    from gatefall.train.c0_model_selftest import run_c0_model_selftest

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
        args.run_dir = default_run_dir_for_arm(args.dataset, ARM_NAME)
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
