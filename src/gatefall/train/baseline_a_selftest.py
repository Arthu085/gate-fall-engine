"""Selftest sintético do guard de --output do relatório (`baseline_a.py`). Não toca em dados reais."""

import sys
from pathlib import Path

from gatefall.runs import REFERENCE_RUN_ROOT


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_guard_rejects_output_under_reference_root() -> bool:
    from gatefall.train.baseline_a import _guard_protected_output

    run_dir = Path("runs/local/le2i/baseline_a")
    output_path = REFERENCE_RUN_ROOT / "le2i/baseline_a/classification_report.json"

    raised = False
    try:
        _guard_protected_output(run_dir, output_path)
    except ValueError:
        raised = True

    return _check(
        "_guard_protected_output recusa --output sob runs/reference/ mesmo "
        "quando o nome do arquivo não é um dos PROTECTED_ARTIFACT_NAMES",
        raised,
    )


def run_baseline_a_selftest() -> bool:
    checks = [
        check_guard_rejects_output_under_reference_root(),
    ]
    ok = all(checks)
    if not ok:
        print("\nbaseline_a selftest FALHOU", file=sys.stderr)
    else:
        print("\nbaseline_a selftest OK: todas as checagens passaram")
    return ok
