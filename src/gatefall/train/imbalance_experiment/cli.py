"""Orquestra as quatro receitas no braço A usando somente dobras de desenvolvimento."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.features.standardization import StandardizationStats
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM
from gatefall.runs import REPOSITORY_ROOT
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.baseline_a.engine import run_training
from gatefall.train.imbalance_experiment.report import _read_run, build_report, write_report
from gatefall.train.internal_folds import cli as cv_cli
from gatefall.train.internal_folds import folds as cv_folds
from gatefall.train.internal_folds import stats as cv_stats
from gatefall.train.internal_folds.folds import DEVELOPMENT_SPLIT, RETAINED_SPLIT, restricted_loader
from gatefall.train.shared.imbalance import RECIPES
from gatefall.train.subject_folds import cli as cs_cli
from gatefall.train.subject_folds import folds as cs_folds
from gatefall.train.subject_folds import stats as cs_stats

OUTPUT_ROOT = REPOSITORY_ROOT / "runs/local/imbalance_experiment"
SEEDS = (42, 43, 44, 45, 46)


def run_dir(root: Path, protocol: str, fold: str, recipe: str, seed: int) -> Path:
    if protocol not in ("cs", "cv") or recipe not in RECIPES or seed not in SEEDS:
        raise ValueError("protocolo, receita ou seed inválida")
    return root / protocol / fold / recipe / f"seed_{seed}"


def _output_root(root: Path) -> None:
    allowed = OUTPUT_ROOT.resolve()
    resolved = root.resolve()
    if resolved != allowed and not resolved.is_relative_to(allowed):
        raise ValueError(f"saída deve ficar em {OUTPUT_ROOT}")


def _sources(protocol: str, fold_root: Path, fold: str) -> tuple[PoseWindowDataset, PoseWindowDataset, TrainConfig, dict[str, object]]:
    if protocol == "cs":
        adapter = cs_cli.ADAPTER
        frames, _, input_hashes = cs_cli._official_train(adapter.frames_path, adapter.manifest_path)
        partitions = cs_folds.build_subject_fold_partitions(frames)
        document = cs_folds.read_folds_document(fold_root)
        pose = cs_cli.real_pose_inputs(frames, adapter.pose_root)
        cs_folds.validate_folds_document(document, partitions, input_hashes, pose.sha256)
        partition = next(partition for partition in partitions if partition.fold == fold)
        stats = cs_stats.load_fold_stats(
            partition.membership_record, cs_folds.stats_dir(fold_root, fold), input_hashes, pose.sha256
        )
        loader = pose.loader
        stats_path, stats_hash, standardization = stats.path, stats.sha256, stats.stats
        membership_hash = cast(str, document["membership_sha256"])
    else:
        adapter = cv_cli.ADAPTER
        frames, _, _, input_hashes = cv_cli._official_train(adapter.frames_path, adapter.manifest_path)
        partitions = cv_folds.build_fold_partitions(frames)
        document = cv_folds.read_folds_document(fold_root)
        cv_folds.validate_folds_document(document, partitions, input_hashes)
        partition = next(partition for partition in partitions if partition.fold == fold)
        stats = cv_stats.load_fold_stats(
            partition, cv_folds.stats_dir(fold_root, fold), adapter.frames_path, input_hashes
        )
        loader = cv_cli.feature_loaders().pose
        stats_path, stats_hash, standardization = stats.files.pose_path, stats.files.pose_sha256, stats.pose
        membership_hash = cast(str, document["membership_sha256"])
    development = PoseWindowDataset(
        partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE,
        restricted_loader(loader, partition.development_video_ids, f"{protocol}/{fold}/treino"),
    )
    retained = PoseWindowDataset(
        partition.frames, RETAINED_SPLIT, EVAL_STRIDE,
        restricted_loader(loader, partition.retained_video_ids, f"{protocol}/{fold}/retido"),
    )
    expected_train = cast(int, partition.membership_record["development_windows_train_stride"])
    expected_retained = cast(int, partition.membership_record["retained_windows_eval_stride"])
    if len(development) != expected_train or len(retained) != expected_retained:
        raise RuntimeError("janelas divergem da composição da dobra")
    config = replace(
        BASELINE_A_CONFIG, standardization_stats_path=str(stats_path),
        standardization_stats_sha256=stats_hash,
    )
    provenance = {
        "protocol": protocol,
        "fold": fold,
        "membership_sha256": membership_hash,
        "input_sha256": input_hashes,
        "standardization_stats_sha256": stats_hash,
        "development_videos": list(partition.development_video_ids),
        "retained_videos": list(partition.retained_video_ids),
        "checkpoint_policy": "last_epoch",
        "official_splits_read": ["train"],
    }
    if set(partition.development_video_ids) & set(partition.retained_video_ids):
        raise RuntimeError("vazamento de vídeos entre treino e retido")
    return development, retained, config, {"provenance": provenance, "stats": standardization}


def train(root: Path, fold_root: Path, protocol: str, fold: str, recipe: str, seed: int, force: bool) -> None:
    _output_root(root)
    if protocol not in ("cs", "cv") or recipe not in RECIPES or seed not in SEEDS:
        raise ValueError("protocolo, receita ou seed inválida")
    names = cs_folds.fold_names() if protocol == "cs" else cv_folds.fold_names()
    if fold not in names:
        raise ValueError(f"dobra desconhecida: {fold}")
    development, retained, base_config, context = _sources(protocol, fold_root, fold)
    config = replace(base_config, seed=seed, run_name=f"imbalance_a_{recipe}")
    destination = run_dir(root, protocol, fold, recipe, seed)
    provenance = cast(dict[str, object], context["provenance"])
    record = {**provenance, "arm": "A", "seed": seed, "recipe": vars(RECIPES[recipe]),
              "epochs": config.epochs}
    if recipe != "baseline":
        control_path = run_dir(root, protocol, fold, "baseline", seed)
        if not (control_path / "experiment.json").is_file():
            raise RuntimeError(f"controle pareado ausente: {control_path}")
        control_record = cast(dict[str, object], _read_run(control_path, protocol, fold, "baseline", seed)["provenance"])
        for key, value in record.items():
            if key != "recipe" and control_record.get(key) != value:
                raise RuntimeError(f"controle pareado divergente em {control_path}: {key}")
    sidecar = destination / "experiment.json"
    if sidecar.exists() and not force:
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        expected = {**record, "config_sha256": sha256_file(destination / "config.yaml"),
                    "checkpoint_sha256": sha256_file(destination / "checkpoint.pt")}
        if recorded != expected:
            raise RuntimeError(f"proveniência divergente em {sidecar}")
    elif destination.exists() and not force:
        raise RuntimeError(f"run sem proveniência em {destination}; use --force")
    metrics = run_training(
        input_dim=POSE_FEATURE_DIM, train_source=development, val_source=retained,
        test_source=retained, stats=cast(StandardizationStats, context["stats"]), config=config,
        run_dir=destination, force=force, label_names=LE2I_LABEL_NAMES,
        imbalance_recipe=RECIPES[recipe],
    )
    if metrics is not None:
        record["config_sha256"] = sha256_file(destination / "config.yaml")
        record["checkpoint_sha256"] = sha256_file(destination / "checkpoint.pt")
        temporary = sidecar.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(sidecar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    pilot = commands.add_parser("pilot")
    multiseed = commands.add_parser("multiseed")
    for command in (training, pilot, multiseed):
        command.add_argument("--protocol", choices=("cs", "cv"), required=True)
        command.add_argument("--fold-root", type=Path, required=True)
        command.add_argument("--force", action="store_true")
    training.add_argument("--fold", required=True)
    training.add_argument("--recipe", choices=tuple(RECIPES), required=True)
    training.add_argument("--seed", type=int, choices=SEEDS, required=True)
    report = commands.add_parser("report")
    report.add_argument("--protocol", choices=("cs", "cv", "both"), required=True)
    for command in (training, pilot, multiseed, report):
        command.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.command == "train":
        train(args.output_root, args.fold_root, args.protocol, args.fold, args.recipe, args.seed, args.force)
    elif args.command == "pilot":
        folds = cs_folds.fold_names() if args.protocol == "cs" else cv_folds.fold_names()
        for fold in folds:
            for recipe in RECIPES:
                train(args.output_root, args.fold_root, args.protocol, fold, recipe, 42, args.force)
        write_report(args.output_root, args.protocol)
    elif args.command == "multiseed":
        _output_root(args.output_root)
        report_data = build_report(args.output_root, args.protocol)
        decisions = cast(dict[str, dict[str, object]], report_data["pilot_decisions"])
        if any(decision.get("status") == "incomplete" for decision in decisions.values()):
            raise RuntimeError("piloto incompleto: execute todas as quatro receitas nas três dobras antes")
        selected = [name for name, decision in decisions.items() if decision.get("advance_to_multiseed") is True]
        if not selected:
            print("nenhuma receita passou pelo piloto; nenhuma seed adicional foi treinada")
            return
        folds = cs_folds.fold_names() if args.protocol == "cs" else cv_folds.fold_names()
        for fold in folds:
            for seed in SEEDS[1:]:
                for recipe in ("baseline", *selected):
                    train(args.output_root, args.fold_root, args.protocol, fold, recipe, seed, args.force)
        write_report(args.output_root, args.protocol)
    else:
        _output_root(args.output_root)
        write_report(args.output_root, args.protocol)


if __name__ == "__main__":
    main()
