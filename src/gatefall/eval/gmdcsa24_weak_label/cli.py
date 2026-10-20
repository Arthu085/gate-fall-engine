"""Eventos em `val` e decisão GO/NO-GO do piloto GMDCSA24 com rótulos fracos (braço A).

`evaluate` roda o protocolo de alarme congelado do braço A sobre a `val` do
Le2i-CS de um run canônico de T ou S e grava `val_event_metrics.json` nele;
`test` não é montado, lido, predito, avaliado nem publicado. `decide` lê as
métricas de classificação de `val` e os relatórios de eventos de `val` de T, S
e dos controles pareados, exige as cinco seeds do controle e todos os
portões, aplica o
[critério de decisão](docs/analysis/gmdcsa24-weak-label-protocol.md#criterio-de-decisao)
e registra a decisão uma única vez em
`runs/local/gmdcsa24_weak_label/decision.json`. As métricas descritivas de
`test`, calculadas uma vez depois da decisão, não fazem parte deste módulo.
"""

import argparse
import contextlib
import fcntl
import json
import os
import sys
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from gatefall.data.gmdcsa24.pose_extract import DEFAULT_POSE_DIR, read_json
from gatefall.data.gmdcsa24.source import read_checkout_state
from gatefall.data.gmdcsa24.weak_label import DEFAULT_OUTPUT_DIR, TREATMENTS
from gatefall.eval.gmdcsa24_weak_label.decision import (
    FALL_CLASS,
    INVALID,
    VALIDITY_CONDITIONS,
    PairedSeed,
    SeedMetrics,
    decide,
)
from gatefall.eval.gmdcsa24_weak_label.validation import (
    ALARM_PROTOCOL_FILE,
    CONTROL_EVENTS_CONDITION,
    RUN_CONDITION,
    VAL_EVENT_METRICS_FILE,
    CheckedRun,
    RunCheck,
    check_run,
    control_from_record,
    training_val,
    val_event_evaluation,
    val_report,
)
from gatefall.eval.shared.event_artifacts import EventEvaluationLock
from gatefall.eval.shared.orchestration import run_event_evaluation
from gatefall.hashing import sha256_file
from gatefall.runs import REPOSITORY_ROOT
from gatefall.train.gmdcsa24_weak_label.assembly import load_le2i_side
from gatefall.train.gmdcsa24_weak_label.cli import refuse
from gatefall.train.gmdcsa24_weak_label.gates import (
    CONDITIONS,
    DECISION_FILE,
    DEFAULT_RUN_ROOT,
    DEFAULT_TRAINING_DIR,
    TREATMENT_RECORD_FILE,
    ControlRun,
    TreatmentInputs,
    canonical_run_dir,
    decision_problems,
    execution_root_problems,
    location_problems,
    report,
)

DECISION_LOCK_FILE = ".decision.lock"
DECISION_CRITERION = "docs/analysis/gmdcsa24-weak-label-protocol.md#criterio-de-decisao"
# Campos do registro que mudam entre execuções idênticas e ficam fora da
# comparação com uma decisão já registrada.
DECISION_CONTEXT_FIELDS = ("code", "recorded_at")
INTEGRITY_CONDITIONS = (
    *(condition for condition in CONDITIONS if condition not in VALIDITY_CONDITIONS),
)
# Cada SHA-256 que a decisão registra por run: (campo, diretório, arquivo).
RUN_INPUT_FILES = (
    ("treatment_record_sha256", "run_dir", TREATMENT_RECORD_FILE),
    ("training_metrics_sha256", "run_dir", "metrics.json"),
    ("val_event_metrics_sha256", "run_dir", VAL_EVENT_METRICS_FILE),
    ("control_training_metrics_sha256", "control_run_dir", "metrics.json"),
    ("control_event_metrics_sha256", "control_run_dir", "event_metrics.json"),
)


@contextlib.contextmanager
def decision_lock(root: Path, exclusive: bool) -> Iterator[None]:
    """Lock entre processos da raiz dos runs: exclusivo para registrar a decisão,
    compartilhado para gravar uma avaliação; espera em vez de falhar."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / DECISION_LOCK_FILE).open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def complete_pair(run: CheckedRun) -> bool:
    """Par alarm_protocol.yaml/val_event_metrics.json presente e íntegro."""
    present = all(
        (run.run_dir / name).is_file() for name in (ALARM_PROTOCOL_FILE, VAL_EVENT_METRICS_FILE)
    )
    return present and not val_report(run)[1]


def run_val_evaluation(
    inputs: TreatmentInputs, seed: int, control: ControlRun, force: bool = False
) -> bool:
    """O `evaluate` do CLI: só o run canônico do tratamento e da seed, da raiz do
    repositório, depois dos portões de pré-treino e do run completo. Um par
    completo e íntegro é imutável, inclusive com `--force`; `--force` só
    recupera um par parcial ou inconsistente, e só antes da decisão registrada."""
    problems = execution_root_problems(inputs, seed, control)
    if problems:
        return refuse("evaluate", problems)
    check = check_run(inputs, seed, control)
    title = f"eventos em val de {inputs.treatment}, seed {seed}"
    if not report(title, check.results) or check.run is None:
        return False
    run = check.run
    with decision_lock(DEFAULT_RUN_ROOT, exclusive=False):
        complete = complete_pair(run)
        if complete and force:
            return refuse(
                "evaluate",
                [
                    f"{run.run_dir / VAL_EVENT_METRICS_FILE}: avaliação completa e íntegra é "
                    "imutável; --force só recupera um par parcial ou inconsistente"
                ],
            )
        if complete:
            print(f"\nskip {run.run_dir / VAL_EVENT_METRICS_FILE} (avaliação completa e íntegra)")
            return True
        problems = decision_problems()
        if problems:
            return refuse("evaluate", problems)
        evaluation = val_event_evaluation(inputs, run)
        try:
            with EventEvaluationLock(run.run_dir) as lock:
                run_event_evaluation(force, "A", run.run_dir, lock, lambda: evaluation)
        except (RuntimeError, ValueError) as exc:
            return refuse("evaluate", [str(exc)])
    _, problems = val_report(run)
    if problems:
        return refuse("evaluate", problems)
    print(f"\n{run.run_dir / VAL_EVENT_METRICS_FILE}: só val avaliada; test não avaliado")
    return True


def on_disk_problems(treatment: str, seeds: Sequence[int]) -> list[str]:
    """Um run canônico com `treatment.json` e `val_event_metrics.json` por seed do
    controle, e nenhum outro diretório de run do tratamento."""
    expected = [canonical_run_dir(treatment, seed) for seed in seeds]
    problems: list[str] = []
    for run_dir in expected:
        if not (run_dir / TREATMENT_RECORD_FILE).is_file():
            problems.append(f"{run_dir}: run ausente ou sem {TREATMENT_RECORD_FILE}")
        elif not (run_dir / VAL_EVENT_METRICS_FILE).is_file():
            problems.append(f"{run_dir}: sem {VAL_EVENT_METRICS_FILE}; rode evaluate")
    root = DEFAULT_RUN_ROOT / treatment.lower()
    if root.is_dir():
        problems.extend(
            f"{child}: fora das seeds do controle {list(seeds)}"
            for child in sorted(root.iterdir())
            if child not in expected
        )
    return problems


def recorded_controls(
    seeds: Sequence[int],
) -> tuple[dict[tuple[str, int], ControlRun], list[str]]:
    """Controle registrado em cada `treatment.json`, o mesmo para T e S em cada seed."""
    controls: dict[tuple[str, int], ControlRun] = {}
    identities: dict[tuple[str, int], object] = {}
    problems: list[str] = []
    for treatment in TREATMENTS:
        for seed in seeds:
            path = canonical_run_dir(treatment, seed) / TREATMENT_RECORD_FILE
            try:
                record = read_json(path)
            except (OSError, ValueError) as exc:
                problems.append(f"{path} ilegível ({exc})")
                continue
            control = control_from_record(record) if isinstance(record, dict) else None
            if control is None:
                problems.append(f"{path}: bloco control ilegível")
                continue
            controls[(treatment, seed)] = control
            block = cast(dict[str, object], record["control"])
            identities[(treatment, seed)] = (
                control.run_dir.resolve(),
                block.get("checkpoint_sha256"),
                block.get("acceptance"),
            )
    for seed in seeds:
        if ("T", seed) in identities and ("S", seed) in identities:
            if identities[("T", seed)] != identities[("S", seed)]:
                problems.append(
                    f"seed {seed}: T e S registram controles diferentes "
                    f"({identities[('T', seed)]} != {identities[('S', seed)]})"
                )
    return controls, problems


def seed_metrics(run_dir: Path, events: Mapping[str, object]) -> SeedMetrics:
    val = training_val(run_dir)
    f1_by_class = cast(dict[str, float], val["f1_by_class"])
    return SeedMetrics(
        macro_f1_restricted=float(cast(float, val["macro_f1_restricted"])),
        fall_f1=float(f1_by_class[FALL_CLASS]),
        n_missed_events=int(cast(int, events["n_missed_events"])),
        n_false_alarms=int(cast(int, events["n_false_alarms"])),
    )


def run_inputs(run: CheckedRun) -> dict[str, str]:
    """SHA-256 de tudo o que a decisão leu de um run e do controle pareado."""
    dirs = {"run_dir": run.run_dir, "control_run_dir": run.control_run_dir}
    return {
        **{key: path.as_posix() for key, path in dirs.items()},
        **{field: sha256_file(dirs[key] / name) for field, key, name in RUN_INPUT_FILES},
    }


def changed_inputs(record: Mapping[str, object]) -> list[str]:
    """Arquivos lidos pela decisão que mudaram ou sumiram desde a leitura."""
    inputs = cast(dict[str, dict[str, dict[str, str]]] | None, record.get("inputs"))
    problems: list[str] = []
    for by_seed in (inputs or {}).values():
        for entry in by_seed.values():
            for field, key, name in RUN_INPUT_FILES:
                path = Path(entry[key]) / name
                if not path.is_file() or sha256_file(path) != entry[field]:
                    problems.append(f"{path} mudou depois de lido pela decisão")
    return problems


def publish_once(path: Path, record: Mapping[str, object]) -> bool:
    """Publicação atômica que nunca substitui: o temporário completo vira o
    arquivo por link, que falha se ele já existe."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        return False
    finally:
        temporary.unlink()
    return True


def labeled(checks: Mapping[tuple[str, int], RunCheck], conditions: Sequence[str]) -> list[str]:
    return [
        f"{treatment}, seed {seed}, {condition}: {problem}"
        for (treatment, seed), check in checks.items()
        for condition in conditions
        for problem in check.results[condition]
    ]


def paired_evidence(
    checks: Mapping[tuple[str, int], RunCheck],
) -> tuple[dict[str, list[PairedSeed]], dict[str, dict[str, dict[str, str]]], list[str]]:
    pairs: dict[str, list[PairedSeed]] = {treatment: [] for treatment in TREATMENTS}
    inputs: dict[str, dict[str, dict[str, str]]] = {treatment: {} for treatment in TREATMENTS}
    problems: list[str] = []
    for (treatment, seed), check in checks.items():
        run = cast(CheckedRun, check.run)
        events, report_problems = val_report(run)
        if events is None:
            problems.extend(f"{treatment}, seed {seed}: {problem}" for problem in report_problems)
            continue
        pairs[treatment].append(
            PairedSeed(
                seed, seed_metrics(run.run_dir, events), seed_metrics(run.control_run_dir, run.control_val)
            )
        )
        inputs[treatment][str(seed)] = run_inputs(run)
    return pairs, inputs, problems


def print_decision(decision: Mapping[str, object]) -> None:
    print("\n=== decisão GO/NO-GO em val ===")
    if decision["decision"] == INVALID:
        failures = cast(dict[str, dict[str, list[str]]], decision["validity_failures"])
        for treatment, conditions in failures.items():
            for condition, problems in conditions.items():
                print(f"[FAIL] {treatment} {condition}")
                for problem in problems:
                    print(f"  {problem}")
    else:
        conditions = cast(dict[str, dict[str, dict[str, object]]], decision["conditions"])
        for treatment, results in conditions.items():
            for name, result in results.items():
                values = {key: value for key, value in result.items() if key != "passed"}
                status = "PASS" if result["passed"] else "FAIL"
                print(f"[{status}] {treatment} {name}: {json.dumps(values, ensure_ascii=False)}")
    print(f"\ndecisão: {decision['decision']}")
    print("test do Le2i-CS não avaliado; as métricas descritivas vêm depois da decisão registrada")


def comparable(record: Mapping[str, object]) -> object:
    return json.loads(
        json.dumps({key: value for key, value in record.items() if key not in DECISION_CONTEXT_FIELDS})
    )


def record_decision(path: Path, record: dict[str, object]) -> bool:
    """Grava a decisão uma única vez, sob lock exclusivo: recalculá-la igual é
    aceito, diferente é recusado, e uma decisão existente nunca é substituída."""
    problems = location_problems(path.parent)
    if problems:
        return refuse("decide", problems)
    differs = [
        f"{path}: a decisão registrada difere da recalculada; pelos critérios de "
        "parada 6 e 7 do protocolo ela não é substituída; inspecione manualmente"
    ]
    with decision_lock(path.parent, exclusive=True):
        if path.exists():
            try:
                existing = read_json(path)
            except (OSError, ValueError) as exc:
                return refuse("decide", [f"{path} ilegível ({exc}); inspecione manualmente"])
            if comparable(existing) == comparable(record):
                print(f"\nskip {path} (decisão já registrada e igual à recalculada)")
                return True
            return refuse("decide", differs)
        problems = changed_inputs(record)
        if problems:
            return refuse("decide", problems)
        if not publish_once(path, record):
            return refuse("decide", differs)
    print(f"\n{path}: decisão registrada")
    return True


def run_decide(inputs_by_treatment: Mapping[str, TreatmentInputs]) -> bool:
    inputs_t = inputs_by_treatment["T"]
    seeds = sorted(inputs_t.gate.control_checkpoint_sha256)
    problems = [problem for treatment in TREATMENTS for problem in on_disk_problems(treatment, seeds)]
    if problems:
        return refuse("decide", problems)
    controls, problems = recorded_controls(seeds)
    for (treatment, seed), control in controls.items():
        problems.extend(execution_root_problems(inputs_by_treatment[treatment], seed, control))
    if problems:
        return refuse("decide", problems)
    checks = {
        (treatment, seed): check_run(inputs_by_treatment[treatment], seed, controls[(treatment, seed)])
        for treatment in TREATMENTS
        for seed in seeds
    }
    problems = labeled(checks, INTEGRITY_CONDITIONS)
    if problems:
        return refuse("decide", problems)
    validity = {
        treatment: {
            condition: [
                f"seed {seed}: {problem}"
                for seed in seeds
                for problem in checks[(treatment, seed)].results[condition]
            ]
            for condition in VALIDITY_CONDITIONS
        }
        for treatment in TREATMENTS
    }
    inputs: dict[str, dict[str, dict[str, str]]] | None = None
    if any(problems for conditions in validity.values() for problems in conditions.values()):
        decision = decide(validity, None, seeds)
    else:
        problems = labeled(checks, (RUN_CONDITION, CONTROL_EVENTS_CONDITION))
        if problems:
            return refuse("decide", problems)
        pairs, inputs, problems = paired_evidence(checks)
        if problems:
            return refuse("decide", problems)
        try:
            decision = decide(validity, pairs, seeds)
        except ValueError as exc:
            return refuse("decide", [str(exc)])
    head, status = inputs_t.checkout
    record: dict[str, object] = {
        "criterion": DECISION_CRITERION,
        **decision,
        "evaluated_splits": ["val"],
        "inputs": inputs,
        "code": {"commit": head, "dirty": bool(status)},
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print_decision(decision)
    return record_decision(DEFAULT_RUN_ROOT / DECISION_FILE, record)


def build_inputs(args: argparse.Namespace, treatment: str) -> TreatmentInputs:
    return TreatmentInputs(
        treatment=treatment,
        adapter_dir=args.adapter_dir,
        pose_dir=args.pose_dir,
        training_dir=args.training_dir,
        le2i=load_le2i_side(),
        checkout=read_checkout_state(REPOSITORY_ROOT),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_inputs(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        subparser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
        subparser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Avalia eventos só em val no run canônico de T ou S e grava val_event_metrics.json",
    )
    evaluate_parser.add_argument("--treatment", required=True, choices=TREATMENTS)
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--control-run-dir", type=Path, required=True)
    evaluate_parser.add_argument(
        "--control-retrained",
        action="store_true",
        help="Aceita controle retreinado nesta máquina se o checkpoint divergir do publicado",
    )
    evaluate_parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstrói um par alarm_protocol.yaml/val_event_metrics.json parcial ou inconsistente",
    )
    add_inputs(evaluate_parser)
    decide_parser = subparsers.add_parser(
        "decide",
        help="Aplica o critério GO/NO-GO em val às cinco seeds de T e S e registra a decisão",
    )
    add_inputs(decide_parser)
    subparsers.add_parser(
        "selftest", help="Verifica a avaliação só em val e a decisão com dados sintéticos"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "selftest":
        from gatefall.eval.gmdcsa24_weak_label.selftests.val_decision import (
            run_val_decision_selftest,
        )

        run_val_decision_selftest()
        return
    try:
        if args.command == "evaluate":
            control = ControlRun(args.control_run_dir, args.control_retrained)
            ok = run_val_evaluation(build_inputs(args, args.treatment), args.seed, control, args.force)
        else:
            inputs = build_inputs(args, "T")
            ok = run_decide({"T": inputs, "S": replace(inputs, treatment="S")})
    except FileNotFoundError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        ok = False
    if not ok:
        print(f"\n{args.command} FALHOU", file=sys.stderr)
        sys.exit(1)
    print(f"\n{args.command} OK")


if __name__ == "__main__":
    main()
