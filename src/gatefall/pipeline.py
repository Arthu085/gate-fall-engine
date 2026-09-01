"""Orquestração reproduzível do pipeline completo do braço A."""

import argparse
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass


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
    if dataset != "le2i":
        raise ValueError(f"dataset não suportado: {dataset!r}")
    if arm != "A":
        raise ValueError(f"braço não suportado: {arm!r}")

    run_dir = "runs/local/le2i/baseline_a"
    steps = [
        PipelineStep("Baixar anotações", (sys.executable, "scripts/fetch_labels.py"), True),
        PipelineStep("Verificar anotações", (sys.executable, "scripts/fetch_labels.py", "--verify")),
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
        _module_step("Validar TCN e métricas", "gatefall.train.baseline_a", "selftest"),
        _module_step("Treinar braço A", "gatefall.train.baseline_a", "train", "--dataset", dataset, "--run-dir", run_dir, supports_force=True),
        _module_step("Validar protocolo de eventos", "gatefall.eval.baseline_a_events", "selftest"),
        _module_step("Avaliar eventos", "gatefall.eval.baseline_a_events", "evaluate", "--dataset", dataset, "--run-dir", run_dir, supports_force=True),
    ]
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
    run_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    run_parser.add_argument("--arm", default="A", choices=("A",))
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas da orquestração"
    )

    args = parser.parse_args()
    if args.command == "run":
        steps = build_pipeline(args.dataset, args.arm, args.force)
        raise SystemExit(execute_pipeline(steps, dry_run=args.dry_run))
    if args.command == "selftest":
        from gatefall.pipeline_selftest import run_pipeline_selftest

        run_pipeline_selftest()


if __name__ == "__main__":
    main()
