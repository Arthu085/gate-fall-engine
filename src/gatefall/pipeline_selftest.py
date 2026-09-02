"""Selftest sintético da orquestração do pipeline do braço A."""

import contextlib
import io
import sys
from collections.abc import Sequence

from gatefall.pipeline import CommandRunner, PipelineStep, build_pipeline, execute_pipeline
from gatefall.features.standardize import build_cli_parser


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _execute_pipeline_silently(
    steps: Sequence[PipelineStep],
    runner: CommandRunner | None = None,
    dry_run: bool = False,
) -> int:
    kwargs = {} if runner is None else {"runner": runner}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return execute_pipeline(steps, dry_run=dry_run, **kwargs)


def _command_signature(step: PipelineStep) -> tuple[str, ...]:
    command = step.command
    if "-m" in command:
        module_index = command.index("-m") + 1
        signature = command[module_index : module_index + 2]
        return tuple(signature)
    return tuple(command[1:3])


def check_exact_command_order() -> bool:
    steps = build_pipeline(dataset="le2i", arm="A")
    actual = [_command_signature(step) for step in steps]
    expected = [
        ("scripts/fetch_labels.py",),
        ("scripts/fetch_labels.py", "--verify"),
        ("scripts/extract_le2i.py",),
        ("gatefall.data.ingest", "ingest"),
        ("gatefall.data.ingest", "verify"),
        ("gatefall.data.coverage", "audit"),
        ("gatefall.data.timegrid", "selftest"),
        ("gatefall.data.timegrid", "build"),
        ("gatefall.data.timegrid", "report"),
        ("gatefall.data.windows", "selftest"),
        ("gatefall.data.windows", "report"),
        ("gatefall.data.frames_io", "selftest"),
        ("gatefall.data.frames_io", "report"),
        ("gatefall.pose.extract", "extract-all"),
        ("gatefall.pose.extract", "report"),
        ("gatefall.pose.kinematics", "selftest"),
        ("gatefall.pose.kinematics", "report"),
        ("gatefall.data.pose_dataset", "selftest"),
        ("gatefall.data.pose_dataset", "report"),
        ("gatefall.features.standardize", "selftest"),
        ("gatefall.features.standardize", "build"),
        ("gatefall.features.standardize", "report"),
        ("gatefall.train.baseline_a", "selftest"),
        ("gatefall.train.baseline_a", "train"),
        ("gatefall.eval.baseline_a_events", "selftest"),
        ("gatefall.eval.baseline_a_events", "evaluate"),
    ]
    all_use_current_python = all(step.command[0] == sys.executable for step in steps)
    return _check(
        "plano: os 26 comandos estão na ordem exata e usam o Python atual",
        actual == expected and all_use_current_python,
    )


def check_dry_run_executes_no_child() -> bool:
    steps = build_pipeline(dataset="le2i", arm="A")
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> int:
        calls.append(tuple(command))
        return 0

    exit_code = _execute_pipeline_silently(steps, runner=runner, dry_run=True)
    return _check(
        "dry-run: imprime o plano sem executar processos filhos",
        exit_code == 0 and calls == [],
    )


def check_failure_stops_and_propagates_exit_code() -> bool:
    steps = build_pipeline(dataset="le2i", arm="A")
    failure_index = 9
    child_exit_code = 17
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> int:
        calls.append(tuple(command))
        if len(calls) - 1 == failure_index:
            return child_exit_code
        return 0

    exit_code = _execute_pipeline_silently(steps, runner=runner)
    expected_calls = [tuple(step.command) for step in steps[: failure_index + 1]]
    return _check(
        "falha: interrompe antes do passo seguinte e propaga o exit code do filho",
        exit_code == child_exit_code and calls == expected_calls,
    )


def check_success_reaches_final_step() -> bool:
    steps = build_pipeline(dataset="le2i", arm="A")
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> int:
        calls.append(tuple(command))
        return 0

    exit_code = _execute_pipeline_silently(steps, runner=runner)
    return _check(
        "sucesso: executa todos os passos e chega à avaliação de eventos",
        exit_code == 0
        and calls == [tuple(step.command) for step in steps]
        and _command_signature(steps[-1])
        == ("gatefall.eval.baseline_a_events", "evaluate"),
    )


def check_force_only_on_supported_producers() -> bool:
    normal_steps = build_pipeline(dataset="le2i", arm="A")
    forced_steps = build_pipeline(dataset="le2i", arm="A", force=True)
    forced_signatures = {
        _command_signature(step) for step in forced_steps if "--force" in step.command
    }
    expected_forced = {
        ("scripts/fetch_labels.py", "--force"),
        ("scripts/extract_le2i.py", "--force"),
        ("gatefall.data.ingest", "ingest"),
        ("gatefall.data.timegrid", "build"),
        ("gatefall.pose.extract", "extract-all"),
        ("gatefall.features.standardize", "build"),
        ("gatefall.train.baseline_a", "train"),
        ("gatefall.eval.baseline_a_events", "evaluate"),
    }
    unchanged_without_force = all("--force" not in step.command for step in normal_steps)
    force_count = sum(step.command.count("--force") for step in forced_steps)
    return _check(
        "force: aparece uma vez somente nos oito produtores suportados",
        unchanged_without_force
        and force_count == len(expected_forced)
        and forced_signatures == expected_forced,
    )


def check_output_is_always_local() -> bool:
    for force in (False, True):
        commands = [
            " ".join(step.command)
            for step in build_pipeline("le2i", "A", force=force)
        ]
        joined = "\n".join(commands)
        local_run = "runs/local/le2i/baseline_a"
        if "runs/reference" in joined or sum(local_run in command for command in commands) != 2:
            return _check(
                "saída: treino e avaliação usam sempre runs/local, nunca reference",
                False,
            )
    return _check(
        "saída: treino e avaliação usam sempre runs/local, nunca reference",
        True,
    )


def check_invalid_dataset_and_arm_rejected_before_child() -> bool:
    rejected = 0
    for dataset, arm in (("desconhecido", "A"), ("le2i", "D")):
        try:
            build_pipeline(dataset=dataset, arm=arm)
        except ValueError:
            rejected += 1
    return _check(
        "validação: dataset e braço inválidos falham antes de qualquer filho",
        rejected == 2,
    )


def check_standardize_cli_dataset_contract() -> bool:
    parser = build_cli_parser()
    report = parser.parse_args(["report", "--dataset", "le2i"])
    build = parser.parse_args(["build", "--dataset", "le2i"])
    selftest = parser.parse_args(["selftest"])
    return _check(
        "CLI standardize: report/build aceitam dataset e selftest preserva default",
        report.command == "report"
        and report.dataset == "le2i"
        and build.command == "build"
        and build.dataset == "le2i"
        and selftest.command == "selftest"
        and selftest.dataset == "le2i",
    )


def run_pipeline_selftest() -> None:
    checks = [
        check_exact_command_order(),
        check_dry_run_executes_no_child(),
        check_failure_stops_and_propagates_exit_code(),
        check_success_reaches_final_step(),
        check_force_only_on_supported_producers(),
        check_output_is_always_local(),
        check_invalid_dataset_and_arm_rejected_before_child(),
        check_standardize_cli_dataset_contract(),
    ]
    if not all(checks):
        print("\npipeline selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npipeline selftest OK: todas as checagens passaram")


if __name__ == "__main__":
    run_pipeline_selftest()
