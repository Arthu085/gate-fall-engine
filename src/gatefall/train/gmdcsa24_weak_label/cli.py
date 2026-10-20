"""Piloto T/S do GMDCSA24 com rótulos fracos (braço A): preflight, prepare, train e coverage.

Implementa a etapa de treino autorizada em
`docs/analysis/gmdcsa24-weak-label-protocol.md#autorizacao-do-passo-4` sem
mudar nenhuma decisão do protocolo: monta o treino aumentado de T ou S, ajusta
padronização e pesos da perda só sobre ele, confere os portões de pré-treino e
treina uma seed com a receita congelada do braço A. O treino avalia só
`train` e `val`; `test` não é lido nem avaliado antes da decisão GO/NO-GO.
Grava só em `data/processed/gmdcsa24_training/` e `runs/local/gmdcsa24_weak_label/`.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np

from gatefall.data.gmdcsa24.pose_extract import (
    DEFAULT_POSE_DIR,
    POSE_MANIFEST_FILE,
    read_json,
    report_validation,
    section,
    validate_pose_output,
)
from gatefall.data.gmdcsa24.source import read_checkout_state
from gatefall.data.gmdcsa24.weak_label import DEFAULT_OUTPUT_DIR, TREATMENTS, WINDOW_FILES
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.features.standardization import StandardizationStats, load_stats, save_stats
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_blocks
from gatefall.runs import REPOSITORY_ROOT
from gatefall.train.baseline_a.config import TrainConfig
from gatefall.train.baseline_a.engine import run_training
from gatefall.train.gmdcsa24_weak_label.assembly import (
    TreatmentSources,
    detection_coverage,
    gmdcsa24_windows,
    load_le2i_side,
    treatment_stats,
)
from gatefall.train.gmdcsa24_weak_label.gates import (
    DEFAULT_RUN_ROOT,
    DEFAULT_TRAINING_DIR,
    PINNED_TRAINING_GATE,
    PREPARED_MANIFEST_FILE,
    PREPARED_STATS_FILE,
    TREATMENT_RECORD_FILE,
    VALIDATION_SPLITS,
    ControlRun,
    TrainingGate,
    TreatmentInputs,
    assess_data,
    assess_training,
    location_problems,
    report,
)


def write_json(path: Path, data: dict[str, object]) -> None:
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def refuse(command: str, problems: list[str]) -> bool:
    print(f"\n{command} recusado:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    return False


def block_differences(
    control: StandardizationStats, treatment: StandardizationStats
) -> dict[str, dict[str, float]]:
    """Diferença absoluta média de média e desvio por bloco de features entre os JSONs."""
    differences: dict[str, dict[str, float]] = {}
    for name, start, end in feature_blocks():
        differences[name] = {
            field: float(
                np.abs(
                    np.asarray(getattr(treatment, field)[start:end])
                    - np.asarray(getattr(control, field)[start:end])
                ).mean()
            )
            for field in ("mean", "std")
        }
    return differences


def print_preparation(record: dict[str, object], differences: dict[str, dict[str, float]]) -> None:
    windows = cast(dict[str, object], record["windows"])
    print(
        f"\ntreino aumentado de {record['treatment']}: {windows['le2i_train']} Le2i-CS + "
        f"{windows['gmdcsa24']} GMDCSA24 = {windows['total']} janelas"
    )
    print("janelas por classe:", json.dumps(windows["class_counts"], ensure_ascii=False))
    weights = cast(list[float], record["class_weights"])
    print(
        "pesos da CrossEntropyLoss:",
        ", ".join(f"{name}={weight:.6g}" for name, weight in zip(LE2I_LABEL_NAMES, weights)),
    )
    print("\n|Δ| médio por bloco de features, JSON do tratamento − JSON do controle:")
    for name, values in differences.items():
        print(f"  {name}: média {values['mean']:.6g}, desvio {values['std']:.6g}")


def run_preflight(inputs: TreatmentInputs, seed: int, control: ControlRun) -> bool:
    assessment = assess_training(inputs, seed, control)
    return report(f"preflight do treino de {inputs.treatment}, seed {seed}", assessment.results)


def run_prepare(inputs: TreatmentInputs, force: bool) -> bool:
    assessment = assess_data(inputs)
    if not report(f"portões de dados de {inputs.treatment}", assessment.results):
        return False
    sources = cast(TreatmentSources, assessment.sources)
    record = cast(dict[str, object], assessment.record)
    manifest_path = inputs.prepared_dir / PREPARED_MANIFEST_FILE
    stats_path = inputs.prepared_dir / PREPARED_STATS_FILE
    if manifest_path.exists() and not force:
        return refuse("prepare", [f"{manifest_path} já existe; use --force para recriar"])
    stats = treatment_stats(sources.train, inputs.le2i.frames_path)
    inputs.prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    save_stats(stats, stats_path, force=True)
    head, _ = inputs.checkout
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **record,
        "code": {"commit": head, "dirty": False},
        "standardization": {
            "path": stats_path.as_posix(),
            "sha256": sha256_file(stats_path),
            "window_count": stats.window_count,
            "frames_hash": stats.frames_hash,
        },
    }
    write_json(manifest_path, manifest)
    print(f"\nmanifesto preparado gravado em {manifest_path}")
    print_preparation(record, block_differences(load_stats(inputs.le2i.stats_path), stats))
    return True


def treatment_record(
    inputs: TreatmentInputs, seed: int, control: ControlRun, checkpoint: str | None, run_dir: Path
) -> dict[str, object]:
    record = read_json(inputs.prepared_dir / PREPARED_MANIFEST_FILE)
    windows = section(record, "windows")
    return {
        "treatment": inputs.treatment,
        "seed": seed,
        "prepared_manifest_sha256": sha256_file(inputs.prepared_dir / PREPARED_MANIFEST_FILE),
        "standardization_stats_sha256": section(record, "standardization").get("sha256"),
        "train_index_sha256": windows.get("train_index_sha256"),
        "class_weights": record.get("class_weights"),
        "eval_index_sha256": {"val": section(record, "eval_index_sha256").get("val")},
        "evaluated_splits": list(VALIDATION_SPLITS),
        "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt"),
        "control": {
            "run_dir": control.run_dir.as_posix(),
            "checkpoint_sha256": checkpoint,
            "acceptance": (
                "retreinado nesta máquina" if control.retrained else "checkpoint_sha256 publicado"
            ),
        },
        "code": {"commit": inputs.checkout[0], "dirty": bool(inputs.checkout[1])},
    }


def run_train(
    inputs: TreatmentInputs, seed: int, control: ControlRun, run_dir: Path, force: bool
) -> bool:
    problems = location_problems(run_dir)
    if problems:
        return refuse("train", problems)
    assessment = assess_training(inputs, seed, control)
    if not report(f"preflight do treino de {inputs.treatment}, seed {seed}", assessment.results):
        return False
    sources = cast(TreatmentSources, assessment.sources)
    metrics = run_training(
        input_dim=POSE_FEATURE_DIM,
        train_source=sources.train,
        val_source=sources.val,
        test_source=None,
        stats=cast(StandardizationStats, assessment.stats),
        config=cast(TrainConfig, assessment.config),
        run_dir=run_dir,
        force=force,
        label_names=LE2I_LABEL_NAMES,
    )
    record_path = run_dir / TREATMENT_RECORD_FILE
    if metrics is not None or not record_path.exists():
        write_json(
            record_path,
            treatment_record(inputs, seed, control, assessment.control_checkpoint_sha256, run_dir),
        )
    print(f"\n{run_dir}: treino de {inputs.treatment}, seed {seed}; test não avaliado")
    return True


def run_coverage(adapter_dir: Path, pose_dir: Path, gate: TrainingGate) -> bool:
    le2i = load_le2i_side()
    validation = validate_pose_output(pose_dir, adapter_dir, le2i.reference, gate.extraction)
    if not report_validation(validation):
        return False
    records = cast(
        dict[str, dict[str, object]], section(read_json(pose_dir / POSE_MANIFEST_FILE), "videos")
    )
    windows = {treatment: gmdcsa24_windows(adapter_dir, treatment) for treatment in WINDOW_FILES}
    videos, summary = detection_coverage(records, windows, pose_dir)
    coverage = videos["coverage"].to_numpy(np.float64)
    total_k = int(videos["K"].to_numpy(np.int64).sum())
    total_found = int(videos["n_found"].to_numpy(np.int64).sum())
    print("\n=== cobertura da detecção de pose (descritiva; não exclui vídeo nem janela) ===")
    print(
        f"{len(videos)} vídeos, {total_k} quadros da grade, {total_found} com pessoa "
        f"({100.0 * total_found / total_k:.2f}%)"
    )
    print(
        f"cobertura por vídeo: mínimo {coverage.min():.3f}, mediana {np.median(coverage):.3f}, "
        f"{int((coverage == 1.0).sum())} vídeos com 100%"
    )
    for treatment, values in summary.items():
        print(f"{treatment}: {json.dumps(values, ensure_ascii=False)}")
    partial = videos.to_dict("records")[: int((coverage < 1.0).sum())]
    print(
        f"\nvídeos com algum quadro sem pessoa, da menor para a maior cobertura ({len(partial)}):"
    )
    for row in partial:
        print(
            f"  {row['video_id']}: {row['n_found']}/{row['K']} ({row['coverage']:.3f}); "
            f"janelas T {row['windows_T']}, S {row['windows_S']}"
        )
    return True


def build_inputs(args: argparse.Namespace) -> TreatmentInputs:
    return TreatmentInputs(
        treatment=args.treatment,
        adapter_dir=args.adapter_dir,
        pose_dir=args.pose_dir,
        training_dir=args.training_dir,
        le2i=load_le2i_side(),
        checkout=read_checkout_state(REPOSITORY_ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_inputs(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--treatment", required=True, choices=TREATMENTS)
        subparser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        subparser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
        subparser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)

    def add_control(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--seed", type=int, required=True)
        subparser.add_argument("--control-run-dir", type=Path, required=True)
        subparser.add_argument(
            "--control-retrained",
            action="store_true",
            help="Aceita controle retreinado nesta máquina se o checkpoint divergir do publicado",
        )

    preflight_parser = subparsers.add_parser(
        "preflight", help="Confere todos os portões de pré-treino de T ou S para uma seed"
    )
    add_inputs(preflight_parser)
    add_control(preflight_parser)
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Monta o treino aumentado e grava a padronização e os pesos só do treino",
    )
    add_inputs(prepare_parser)
    prepare_parser.add_argument("--force", action="store_true")
    train_parser = subparsers.add_parser(
        "train", help="Treina uma seed de T ou S com a receita congelada, avaliando só train e val"
    )
    add_inputs(train_parser)
    add_control(train_parser)
    train_parser.add_argument("--run-dir", type=Path, default=None)
    train_parser.add_argument(
        "--force", action="store_true", help="Refaz o run só depois de falha técnica registrada"
    )
    coverage_parser = subparsers.add_parser(
        "coverage", help="Relata a cobertura descritiva da detecção de pose dos 145 vídeos"
    )
    coverage_parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    coverage_parser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    subparsers.add_parser("selftest", help="Verifica montagem e portões com dados sintéticos")

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.train.gmdcsa24_weak_label.selftests.treatment import run_treatment_selftest

        run_treatment_selftest()
        return
    try:
        if args.command == "coverage":
            ok = run_coverage(args.adapter_dir, args.pose_dir, PINNED_TRAINING_GATE)
        else:
            inputs = build_inputs(args)
            if args.command == "prepare":
                ok = run_prepare(inputs, args.force)
            else:
                control = ControlRun(args.control_run_dir, args.control_retrained)
                if args.command == "preflight":
                    ok = run_preflight(inputs, args.seed, control)
                else:
                    run_dir = args.run_dir or (
                        DEFAULT_RUN_ROOT / args.treatment.lower() / f"seed_{args.seed}"
                    )
                    ok = run_train(inputs, args.seed, control, run_dir, args.force)
    except FileNotFoundError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        ok = False
    if not ok:
        print(f"\n{args.command} FALHOU", file=sys.stderr)
        sys.exit(1)
    print(f"\n{args.command} OK")


if __name__ == "__main__":
    main()
