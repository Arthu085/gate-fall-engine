"""Orquestração reproduzível dos pipelines experimentais."""

import argparse
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS
from gatefall.runs import default_run_dir, default_run_dir_for_arm


SUPPORTED_ARMS = ("A", "B0", "B1", "C0", "C1")
ARM_MODULES = {
    "B0": ("gatefall.train.baseline_b0", "gatefall.eval.baseline_b0"),
    "B1": ("gatefall.train.baseline_b1", "gatefall.eval.baseline_b1"),
    "C0": ("gatefall.train.baseline_c0", None),
    "C1": ("gatefall.train.baseline_c1", "gatefall.eval.baseline_c1"),
}


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]
    supports_force: bool = False


CommandRunner = Callable[[Sequence[str]], int]


def _module_step(
    name: str,
    module: str,
    command: str,
    *arguments: str,
    supports_force: bool = False,
) -> PipelineStep:
    return PipelineStep(
        name,
        (sys.executable, "-m", module, command, *arguments),
        supports_force,
    )


def build_pipeline(
    dataset: str = "le2i", arm: str = "A", force: bool = False
) -> list[PipelineStep]:
    if dataset not in ("le2i", "le2i-cv"):
        raise ValueError(f"dataset não suportado: {dataset!r}")
    if arm not in SUPPORTED_ARMS:
        raise ValueError(f"braço não suportado: {arm!r}")
    if dataset == "le2i-cv" and arm != "A":
        raise ValueError(f"braço {arm!r} não suporta --dataset {dataset!r}; use le2i")

    is_cv = dataset == "le2i-cv"
    run_dir = str(default_run_dir(dataset))
    fetch_labels_args = ("--protocol", "cv") if is_cv else ()
    steps = [
        PipelineStep(
            "Baixar anotações",
            (sys.executable, "scripts/fetch_labels.py", *fetch_labels_args),
            True,
        ),
        PipelineStep(
            "Verificar anotações",
            (sys.executable, "scripts/fetch_labels.py", "--verify", *fetch_labels_args),
        ),
        PipelineStep("Extrair arquivo Le2i", (sys.executable, "scripts/extract_le2i.py"), True),
        _module_step("Construir manifesto", "gatefall.data.ingest", "ingest", "--dataset", dataset, supports_force=True),
        _module_step("Verificar manifesto", "gatefall.data.ingest", "verify", "--dataset", dataset),
        _module_step("Auditar cobertura", "gatefall.data.coverage", "audit", "--dataset", dataset),
        _module_step("Validar grade temporal", "gatefall.data.timegrid", "selftest", "--dataset", dataset),
        _module_step("Construir grade temporal", "gatefall.data.timegrid", "build", "--dataset", dataset, supports_force=True),
        _module_step("Relatar grade temporal", "gatefall.data.timegrid", "report", "--dataset", dataset),
        _module_step("Validar janelamento", "gatefall.data.windows", "selftest", "--dataset", dataset),
        _module_step("Relatar janelamento", "gatefall.data.windows", "report", "--dataset", dataset),
        _module_step("Validar leitura de quadros", "gatefall.data.frames_io", "selftest", "--dataset", dataset),
        _module_step("Relatar leitura de quadros", "gatefall.data.frames_io", "report", "--dataset", dataset),
        _module_step("Extrair poses", "gatefall.pose.extract", "extract-all", "--dataset", dataset, supports_force=True),
        _module_step("Validar extração de poses", "gatefall.pose.extract", "report", "--dataset", dataset),
        _module_step("Validar cinemática", "gatefall.pose.kinematics", "selftest", "--dataset", dataset),
        _module_step("Relatar cinemática", "gatefall.pose.kinematics", "report", "--dataset", dataset),
        _module_step("Validar dataset de pose", "gatefall.data.pose_dataset", "selftest", "--dataset", dataset),
        _module_step("Relatar dataset de pose", "gatefall.data.pose_dataset", "report", "--dataset", dataset),
        _module_step("Validar padronização", "gatefall.features.standardize", "selftest"),
        _module_step("Construir padronização", "gatefall.features.standardize", "build", "--dataset", dataset, supports_force=True),
        _module_step("Relatar padronização", "gatefall.features.standardize", "report", "--dataset", dataset),
    ]
    if arm == "A":
        steps.extend(
            [
                _module_step("Validar TCN e métricas", "gatefall.train.baseline_a", "selftest"),
                _module_step("Treinar braço A", "gatefall.train.baseline_a", "train", "--dataset", dataset, "--run-dir", run_dir, supports_force=True),
                _module_step("Validar protocolo de eventos", "gatefall.eval.baseline_a", "selftest"),
                _module_step("Avaliar eventos", "gatefall.eval.baseline_a", "evaluate", "--dataset", dataset, "--run-dir", run_dir, supports_force=True),
            ]
        )
        if is_cv:
            steps.append(
                _module_step(
                    "Relatar generalização",
                    "gatefall.eval.analysis.generalization_report",
                    "report",
                    "--dataset",
                    dataset,
                    "--output",
                    f"{run_dir}/generalization_report.json",
                )
            )
    else:
        train_module, event_module = ARM_MODULES[arm]
        run_dir = str(default_run_dir_for_arm(dataset, arm))
        if arm in ("B0", "B1"):
            steps.extend([
                _module_step("Validar extração DINOv3", "gatefall.dinov3.extract", "selftest"),
                _module_step("Extrair features DINOv3", "gatefall.dinov3.extract", "extract-all", "--dataset", dataset, supports_force=True),
                _module_step("Relatar features DINOv3", "gatefall.dinov3.extract", "report", "--dataset", dataset),
                _module_step("Validar padronização DINOv3", "gatefall.features.standardize_dinov3", "selftest"),
                _module_step("Construir padronização DINOv3", "gatefall.features.standardize_dinov3", "build", "--dataset", dataset, supports_force=True),
                _module_step("Relatar padronização DINOv3", "gatefall.features.standardize_dinov3", "report", "--dataset", dataset),
            ])
            if arm == "B1":
                steps.extend([
                    _module_step("Validar features de qualidade", "gatefall.features.quality_extract", "selftest"),
                    _module_step("Extrair features de qualidade", "gatefall.features.quality_extract", "extract-all", "--dataset", dataset, supports_force=True),
                    _module_step("Relatar features de qualidade", "gatefall.features.quality_extract", "report", "--dataset", dataset),
                ])
        else:
            steps.extend([
                _module_step("Validar extração SAM 3", "gatefall.sam3.extract", "selftest"),
                _module_step("Extrair features SAM 3", "gatefall.sam3.extract", "extract-all", "--dataset", dataset, supports_force=True),
                _module_step("Relatar features SAM 3", "gatefall.sam3.extract", "report", "--dataset", dataset),
                _module_step("Validar padronização SAM 3", "gatefall.features.standardize_sam3", "selftest"),
                _module_step("Construir padronização SAM 3", "gatefall.features.standardize_sam3", "build", "--dataset", dataset, supports_force=True),
                _module_step("Relatar padronização SAM 3", "gatefall.features.standardize_sam3", "report", "--dataset", dataset),
            ])
            if arm == "C1":
                steps.append(_module_step("Validar proxy de qualidade SAM 3", "gatefall.sam3.quality", "selftest"))
        steps.extend([
            _module_step(f"Validar braço {arm}", train_module, "selftest"),
            _module_step(f"Treinar braço {arm}", train_module, "train", "--dataset", dataset, "--run-dir", run_dir, supports_force=True),
            _module_step(f"Relatar classificação do braço {arm}", train_module, "report", "--dataset", dataset, "--run-dir", run_dir, supports_force=True),
        ])
        if event_module is not None:
            steps.extend([
                _module_step("Validar protocolo de eventos", event_module, "selftest"),
                _module_step("Avaliar eventos", event_module, "evaluate", "--dataset", dataset, "--run-dir", run_dir, supports_force=True),
            ])
    if not force:
        return steps
    return [
        PipelineStep(step.name, (*step.command, "--force"), step.supports_force)
        if step.supports_force
        else step
        for step in steps
    ]


def _subprocess_runner(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode


def execute_pipeline(
    steps: Sequence[PipelineStep],
    runner: CommandRunner = _subprocess_runner,
    dry_run: bool = False,
) -> int:
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        rendered = shlex.join(step.command)
        print(f"[{index:02d}/{total:02d}] {step.name}")
        print(f"  {rendered}")
        if dry_run:
            continue
        exit_code = runner(step.command)
        if exit_code != 0:
            print(
                f"pipeline FALHOU no passo {index}/{total}: {step.name}\n"
                f"comando: {rendered}\n"
                f"exit code: {exit_code}\n"
                "os passos posteriores não foram executados",
                file=sys.stderr,
            )
            return exit_code
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Executa o pipeline completo")
    run_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)
    run_parser.add_argument("--arm", default="A", choices=SUPPORTED_ARMS)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas da orquestração"
    )

    args = parser.parse_args()
    if args.command == "run":
        try:
            steps = build_pipeline(args.dataset, args.arm, args.force)
        except ValueError as exc:
            parser.error(str(exc))
        raise SystemExit(execute_pipeline(steps, dry_run=args.dry_run))
    if args.command == "selftest":
        from gatefall.selftests.pipeline import run_pipeline_selftest

        run_pipeline_selftest()


if __name__ == "__main__":
    main()
