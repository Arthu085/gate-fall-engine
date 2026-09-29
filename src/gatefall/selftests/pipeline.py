"""Selftest sintético da orquestração dos pipelines experimentais."""

import contextlib
import io
import sys
from collections.abc import Sequence

from gatefall.pipeline import CommandRunner, PipelineStep, build_pipeline, execute_pipeline
from gatefall.features.standardize import build_cli_parser
from gatefall.runs import default_run_dir_for_arm


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
        ("gatefall.eval.baseline_a", "selftest"),
        ("gatefall.eval.baseline_a", "evaluate"),
    ]
    expected_arguments = [
        (), (), (),
        ("--dataset", "le2i"), ("--dataset", "le2i"), ("--dataset", "le2i"),
        ("--dataset", "le2i"), ("--dataset", "le2i"), ("--dataset", "le2i"),
        ("--dataset", "le2i"), ("--dataset", "le2i"),
        ("--dataset", "le2i"), ("--dataset", "le2i"),
        ("--dataset", "le2i"), ("--dataset", "le2i"),
        ("--dataset", "le2i"), ("--dataset", "le2i"),
        ("--dataset", "le2i"), ("--dataset", "le2i"),
        (), ("--dataset", "le2i"), ("--dataset", "le2i"),
        (), ("--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_a"),
        (), ("--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_a"),
    ]
    expected_commands = [
        (sys.executable, *signature, *arguments)
        if signature[0].startswith("scripts/")
        else (sys.executable, "-m", *signature, *arguments)
        for signature, arguments in zip(expected, expected_arguments, strict=True)
    ]
    all_use_current_python = all(step.command[0] == sys.executable for step in steps)
    return _check(
        "plano: os 26 comandos A estão completos e na ordem exata",
        actual == expected
        and all_use_current_python
        and [step.command for step in steps] == expected_commands,
    )


def check_new_arm_command_order() -> bool:
    shared = build_pipeline("le2i", "A")[:22]
    expected_suffixes = {
        "B0": [
            ("gatefall.dinov3.extract", "selftest"),
            ("gatefall.dinov3.extract", "extract-all", "--dataset", "le2i"),
            ("gatefall.dinov3.extract", "report", "--dataset", "le2i"),
            ("gatefall.features.standardize_dinov3", "selftest"),
            ("gatefall.features.standardize_dinov3", "build", "--dataset", "le2i"),
            ("gatefall.features.standardize_dinov3", "report", "--dataset", "le2i"),
            ("gatefall.train.baseline_b0", "selftest"),
            ("gatefall.train.baseline_b0", "train", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_b0"),
            ("gatefall.train.baseline_b0", "report", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_b0"),
            ("gatefall.eval.baseline_b0", "selftest"),
            ("gatefall.eval.baseline_b0", "evaluate", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_b0"),
        ],
        "B1": [
            ("gatefall.dinov3.extract", "selftest"),
            ("gatefall.dinov3.extract", "extract-all", "--dataset", "le2i"),
            ("gatefall.dinov3.extract", "report", "--dataset", "le2i"),
            ("gatefall.features.standardize_dinov3", "selftest"),
            ("gatefall.features.standardize_dinov3", "build", "--dataset", "le2i"),
            ("gatefall.features.standardize_dinov3", "report", "--dataset", "le2i"),
            ("gatefall.features.quality_extract", "selftest"),
            ("gatefall.features.quality_extract", "extract-all", "--dataset", "le2i"),
            ("gatefall.features.quality_extract", "report", "--dataset", "le2i"),
            ("gatefall.train.baseline_b1", "selftest"),
            ("gatefall.train.baseline_b1", "train", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_b1"),
            ("gatefall.train.baseline_b1", "report", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_b1"),
            ("gatefall.eval.baseline_b1", "selftest"),
            ("gatefall.eval.baseline_b1", "evaluate", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_b1"),
        ],
        "C0": [
            ("gatefall.sam3.extract", "selftest"),
            ("gatefall.sam3.extract", "extract-all", "--dataset", "le2i"),
            ("gatefall.sam3.extract", "report", "--dataset", "le2i"),
            ("gatefall.features.standardize_sam3", "selftest"),
            ("gatefall.features.standardize_sam3", "build", "--dataset", "le2i"),
            ("gatefall.features.standardize_sam3", "report", "--dataset", "le2i"),
            ("gatefall.train.baseline_c0", "selftest"),
            ("gatefall.train.baseline_c0", "train", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_c0"),
            ("gatefall.train.baseline_c0", "report", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_c0"),
        ],
        "C1": [
            ("gatefall.sam3.extract", "selftest"),
            ("gatefall.sam3.extract", "extract-all", "--dataset", "le2i"),
            ("gatefall.sam3.extract", "report", "--dataset", "le2i"),
            ("gatefall.features.standardize_sam3", "selftest"),
            ("gatefall.features.standardize_sam3", "build", "--dataset", "le2i"),
            ("gatefall.features.standardize_sam3", "report", "--dataset", "le2i"),
            ("gatefall.sam3.quality", "selftest"),
            ("gatefall.train.baseline_c1", "selftest"),
            ("gatefall.train.baseline_c1", "train", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_c1"),
            ("gatefall.train.baseline_c1", "report", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_c1"),
            ("gatefall.eval.baseline_c1", "selftest"),
            ("gatefall.eval.baseline_c1", "evaluate", "--dataset", "le2i", "--run-dir", "runs/local/le2i/baseline_c1"),
        ],
    }
    for arm, suffix in expected_suffixes.items():
        steps = build_pipeline("le2i", arm)
        expected_commands = [(sys.executable, "-m", *command) for command in suffix]
        if steps[:22] != shared or [step.command for step in steps[22:]] != expected_commands:
            return _check("plano: B0, B1, C0 e C1 têm comandos na ordem exata", False)
    return _check("plano: B0, B1, C0 e C1 têm comandos na ordem exata", True)


def check_dry_run_executes_no_child() -> bool:
    for arm in ("A", "B0", "B1", "C0", "C1"):
        steps = build_pipeline(dataset="le2i", arm=arm)
        calls: list[tuple[str, ...]] = []

        def runner(command: Sequence[str]) -> int:
            calls.append(tuple(command))
            return 0

        with contextlib.redirect_stdout(io.StringIO()) as output:
            exit_code = execute_pipeline(steps, runner=runner, dry_run=True)
        if exit_code != 0 or calls or output.getvalue().count("  " + sys.executable) != len(steps):
            return _check("dry-run: todos os braços imprimem sem executar filhos", False)
    return _check("dry-run: todos os braços imprimem sem executar filhos", True)


def check_failure_stops_and_propagates_exit_code() -> bool:
    for arm in ("A", "B0", "B1", "C0", "C1"):
        steps = build_pipeline(dataset="le2i", arm=arm)
        failure_index = 23 if arm != "A" else 9
        child_exit_code = 17
        calls: list[tuple[str, ...]] = []

        def runner(command: Sequence[str]) -> int:
            calls.append(tuple(command))
            if len(calls) - 1 == failure_index:
                return child_exit_code
            return 0

        exit_code = _execute_pipeline_silently(steps, runner=runner)
        expected_calls = [tuple(step.command) for step in steps[: failure_index + 1]]
        if exit_code != child_exit_code or calls != expected_calls:
            return _check("falha: todos os braços param e propagam o exit code", False)
    return _check("falha: todos os braços param e propagam o exit code", True)


def check_success_reaches_final_step() -> bool:
    final_commands = {
        "A": ("gatefall.eval.baseline_a", "evaluate"),
        "B0": ("gatefall.eval.baseline_b0", "evaluate"),
        "B1": ("gatefall.eval.baseline_b1", "evaluate"),
        "C0": ("gatefall.train.baseline_c0", "report"),
        "C1": ("gatefall.eval.baseline_c1", "evaluate"),
    }
    for arm, final_command in final_commands.items():
        steps = build_pipeline(dataset="le2i", arm=arm)
        calls: list[tuple[str, ...]] = []

        def runner(command: Sequence[str]) -> int:
            calls.append(tuple(command))
            return 0

        exit_code = _execute_pipeline_silently(steps, runner=runner)
        if (
            exit_code != 0
            or calls != [step.command for step in steps]
            or _command_signature(steps[-1]) != final_command
        ):
            return _check("sucesso: todos os braços chegam à última etapa", False)
    return _check("sucesso: todos os braços chegam à última etapa", True)


def check_force_only_on_supported_producers() -> bool:
    shared_forced = {
        ("scripts/fetch_labels.py",),
        ("scripts/fetch_labels.py", "--protocol"),
        ("scripts/extract_le2i.py",),
        ("gatefall.data.ingest", "ingest"),
        ("gatefall.data.timegrid", "build"),
        ("gatefall.pose.extract", "extract-all"),
        ("gatefall.features.standardize", "build"),
    }
    expected_by_arm = {
        "A": {("gatefall.train.baseline_a", "train"), ("gatefall.eval.baseline_a", "evaluate")},
        "B0": {("gatefall.dinov3.extract", "extract-all"), ("gatefall.features.standardize_dinov3", "build"), ("gatefall.train.baseline_b0", "train"), ("gatefall.train.baseline_b0", "report"), ("gatefall.eval.baseline_b0", "evaluate")},
        "B1": {("gatefall.dinov3.extract", "extract-all"), ("gatefall.features.standardize_dinov3", "build"), ("gatefall.features.quality_extract", "extract-all"), ("gatefall.train.baseline_b1", "train"), ("gatefall.train.baseline_b1", "report"), ("gatefall.eval.baseline_b1", "evaluate")},
        "C0": {("gatefall.sam3.extract", "extract-all"), ("gatefall.features.standardize_sam3", "build"), ("gatefall.train.baseline_c0", "train"), ("gatefall.train.baseline_c0", "report")},
        "C1": {("gatefall.sam3.extract", "extract-all"), ("gatefall.features.standardize_sam3", "build"), ("gatefall.train.baseline_c1", "train"), ("gatefall.train.baseline_c1", "report"), ("gatefall.eval.baseline_c1", "evaluate")},
    }
    cases = [("le2i", arm) for arm in expected_by_arm] + [("le2i-cv", "A")]
    for dataset, arm in cases:
        arm_forced = expected_by_arm[arm]
        normal_steps = build_pipeline(dataset, arm)
        forced_steps = build_pipeline(dataset, arm, force=True)
        if len(normal_steps) != len(forced_steps):
            return _check("force: somente produtores compatíveis recebem --force", False)
        for normal, forced in zip(normal_steps, forced_steps, strict=True):
            should_force = _command_signature(normal) in shared_forced | arm_forced
            expected = (*normal.command, "--force") if should_force else normal.command
            if forced.command != expected or "--force" in normal.command:
                return _check("force: somente produtores compatíveis recebem --force", False)
    return _check("force: somente produtores compatíveis recebem --force", True)


def check_output_is_always_local() -> bool:
    expected_run_steps = {"A": 2, "B0": 3, "B1": 3, "C0": 2, "C1": 3}
    for arm, count in expected_run_steps.items():
        for force in (False, True):
            commands = [" ".join(step.command) for step in build_pipeline("le2i", arm, force=force)]
            local_run = str(default_run_dir_for_arm("le2i", arm))
            if "runs/reference" in "\n".join(commands) or sum(local_run in command for command in commands) != count:
                return _check("saída: cada braço usa seu run local, nunca reference", False)
    return _check(
        "saída: cada braço usa seu run local, nunca reference",
        True,
    )


def check_invalid_dataset_and_arm_rejected_before_child() -> bool:
    rejected = 0
    invalid = (("desconhecido", "A"), ("le2i", "D"), *(("le2i-cv", arm) for arm in ("B0", "B1", "C0", "C1")))
    for dataset, arm in invalid:
        try:
            build_pipeline(dataset=dataset, arm=arm)
        except ValueError as exc:
            if dataset != "le2i-cv" or "não suporta --dataset 'le2i-cv'" in str(exc):
                rejected += 1
    return _check(
        "validação: dataset, braço e combinações incompatíveis são recusados",
        rejected == len(invalid),
    )


def check_cv_step_list() -> bool:
    steps = build_pipeline(dataset="le2i-cv", arm="A")
    actual = [_command_signature(step) for step in steps]
    expected = [
        ("scripts/fetch_labels.py", "--protocol"),
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
        ("gatefall.eval.baseline_a", "selftest"),
        ("gatefall.eval.baseline_a", "evaluate"),
        ("gatefall.eval.analysis.generalization_report", "report"),
    ]
    run_dir_present = all(
        "runs/local/le2i_cv/baseline_a" in " ".join(step.command)
        for step in steps
        if _command_signature(step)
        in {
            ("gatefall.train.baseline_a", "train"),
            ("gatefall.eval.baseline_a", "evaluate"),
            ("gatefall.eval.analysis.generalization_report", "report"),
        }
    )
    no_reference = all("runs/reference" not in " ".join(step.command) for step in steps)
    expected_commands = []
    for index, step in enumerate(build_pipeline("le2i", "A")):
        if index < 2:
            expected_commands.append((*step.command, "--protocol", "cv"))
        else:
            expected_commands.append(
                tuple(
                    part.replace("runs/local/le2i/baseline_a", "runs/local/le2i_cv/baseline_a")
                    if part.startswith("runs/local/le2i/baseline_a")
                    else "le2i-cv" if part == "le2i" else part
                    for part in step.command
                )
            )
    expected_commands.append(
        (
            sys.executable, "-m", "gatefall.eval.analysis.generalization_report", "report",
            "--dataset", "le2i-cv", "--output",
            "runs/local/le2i_cv/baseline_a/generalization_report.json",
        )
    )
    return _check(
        "plano CV: 27 comandos completos preservam o protocolo e o run_dir isolado",
        actual == expected
        and [step.command for step in steps] == expected_commands
        and run_dir_present
        and no_reference,
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
        check_new_arm_command_order(),
        check_dry_run_executes_no_child(),
        check_failure_stops_and_propagates_exit_code(),
        check_success_reaches_final_step(),
        check_force_only_on_supported_producers(),
        check_output_is_always_local(),
        check_invalid_dataset_and_arm_rejected_before_child(),
        check_cv_step_list(),
        check_standardize_cli_dataset_contract(),
    ]
    if not all(checks):
        print("\npipeline selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\npipeline selftest OK: todas as checagens passaram")


if __name__ == "__main__":
    run_pipeline_selftest()
