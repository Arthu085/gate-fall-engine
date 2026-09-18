"""Selftest sintético do guard de --output do relatório (`baseline_a.py`). Não toca em dados reais."""

import inspect
import sys
from dataclasses import replace
from pathlib import Path

from gatefall.runs import REFERENCE_RUN_ROOT
from gatefall.train.config import BASELINE_A_CONFIG


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


def check_resolve_config_propagates_seed() -> bool:
    from gatefall.train.baseline_a import _resolve_config

    stats_path = Path("src/gatefall/features/stats/pose_le2i_cs.json")
    stats_sha256 = "deadbeef"
    other_seed = BASELINE_A_CONFIG.seed + 1

    resolved = _resolve_config(other_seed, stats_path, stats_sha256)
    expected = replace(
        BASELINE_A_CONFIG,
        seed=other_seed,
        standardization_stats_path=str(stats_path),
        standardization_stats_sha256=stats_sha256,
    )
    only_expected_fields_differ = resolved == expected
    seed_propagated = resolved.seed == other_seed

    resolved_default = _resolve_config(BASELINE_A_CONFIG.seed, stats_path, stats_sha256)
    default_seed_preserved = resolved_default.seed == BASELINE_A_CONFIG.seed

    fields_that_differ = {
        field
        for field in resolved.to_dict()
        if resolved.to_dict()[field] != BASELINE_A_CONFIG.to_dict()[field]
    }
    only_allowed_fields_differ = fields_that_differ <= {
        "seed",
        "standardization_stats_path",
        "standardization_stats_sha256",
    }

    return _check(
        "_resolve_config propaga a seed e só diverge de BASELINE_A_CONFIG em "
        "{seed, standardization_stats_path, standardization_stats_sha256}",
        only_expected_fields_differ
        and seed_propagated
        and default_seed_preserved
        and only_allowed_fields_differ,
    )


def check_run_train_default_seed_preserved() -> bool:
    from gatefall.train.baseline_a import run_train

    default_seed = inspect.signature(run_train).parameters["seed"].default
    return _check(
        "run_train preserva BASELINE_A_CONFIG.seed como default do parâmetro seed",
        default_seed == BASELINE_A_CONFIG.seed,
    )


def run_baseline_a_selftest() -> bool:
    checks = [
        check_guard_rejects_output_under_reference_root(),
        check_resolve_config_propagates_seed(),
        check_run_train_default_seed_preserved(),
    ]
    ok = all(checks)
    if not ok:
        print("\nbaseline_a selftest FALHOU", file=sys.stderr)
    else:
        print("\nbaseline_a selftest OK: todas as checagens passaram")
    return ok
