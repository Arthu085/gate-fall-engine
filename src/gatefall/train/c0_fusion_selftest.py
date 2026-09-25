"""Selftest sintético dos guards de CLI da arma C0 (`c0_fusion.py`). Não toca em
dados reais. Espelha `b1_gate_selftest.py`, estendendo as guardas de run_dir e
de `--output` aos runs de comparação de A, B0 e B1."""

import inspect
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from gatefall.runs import REFERENCE_RUN_ROOT, default_run_dir_for_arm
from gatefall.train.b1_run import repository_anchored_run_dir
from gatefall.train.c0_config import C0_FUSION_CONFIG
from gatefall.train.c0_run import COMPARISON_ARM_NAMES, resolve_c0_config

_C0_RUN_DIR = Path("runs/local/le2i/c0_fusion")


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _raises_value_error(callback) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def _comparison_run_dirs() -> list[Path]:
    return [
        repository_anchored_run_dir(default_run_dir_for_arm("le2i", arm_name))
        for arm_name in COMPARISON_ARM_NAMES.values()
    ]


def check_guard_rejects_output_under_reference_root() -> bool:
    from gatefall.train.c0_fusion import _guard_protected_output

    output_path = REFERENCE_RUN_ROOT / "le2i/c0_fusion/classification_report.json"
    return _check(
        "_guard_protected_output recusa --output sob runs/reference/",
        _raises_value_error(
            lambda: _guard_protected_output(_C0_RUN_DIR, output_path, "le2i")
        ),
    )


def check_guard_rejects_protected_output_in_all_protected_run_dirs() -> bool:
    from gatefall.train.c0_fusion import (
        PROTECTED_ARTIFACT_NAMES,
        _guard_protected_output,
    )

    canonical_c0 = repository_anchored_run_dir(default_run_dir_for_arm("le2i", "c0_fusion"))
    non_canonical_c0 = canonical_c0.parent / "c0_fusion_seed7"
    protected = [_C0_RUN_DIR, canonical_c0, *_comparison_run_dirs()]
    all_raised = all(
        _raises_value_error(
            lambda run_dir=run_dir, name=name: _guard_protected_output(
                non_canonical_c0 if run_dir == canonical_c0 else _C0_RUN_DIR,
                run_dir / name,
                "le2i",
            )
        )
        for run_dir in protected
        for name in PROTECTED_ARTIFACT_NAMES
    )
    sibling_accepted = not _raises_value_error(
        lambda: _guard_protected_output(
            _C0_RUN_DIR, _C0_RUN_DIR / "classification_report.json", "le2i"
        )
    )
    return _check(
        "_guard_protected_output recusa --output apontando para artefato "
        "protegido do próprio run, do run canônico do C0 e dos runs de A, B0 e "
        "B1, e aceita classification_report.json no próprio run",
        all_raised and sibling_accepted,
    )


def check_guard_rejects_comparison_run_dirs() -> bool:
    from gatefall.train.c0_fusion import _guard_run_dir

    rejected = all(
        _raises_value_error(lambda candidate=candidate: _guard_run_dir(candidate, "le2i"))
        for run_dir in _comparison_run_dirs()
        for candidate in (run_dir, run_dir.parent, run_dir / "stray")
    )
    return _check(
        "_guard_run_dir recusa run_dir igual, ancestral ou descendente dos "
        "runs canônicos de A, B0 e B1 — impede que um --force do C0 destrua "
        "um run de comparação",
        rejected,
    )


def check_guards_are_anchored_at_repository_root() -> bool:
    from gatefall.train.c0_fusion import (
        PROTECTED_ARTIFACT_NAMES,
        _guard_protected_output,
        _guard_run_dir,
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(tempfile.gettempdir())
        rejects_run_dirs = all(
            _raises_value_error(lambda run_dir=run_dir: _guard_run_dir(run_dir, "le2i"))
            for run_dir in _comparison_run_dirs()
        )
        rejects_outputs = all(
            _raises_value_error(
                lambda run_dir=run_dir: _guard_protected_output(
                    _C0_RUN_DIR, run_dir / PROTECTED_ARTIFACT_NAMES[1], "le2i"
                )
            )
            for run_dir in _comparison_run_dirs()
        )
    finally:
        os.chdir(original_cwd)
    return _check(
        "as guardas continuam recusando os runs absolutos de A, B0 e B1 quando "
        "a CLI roda de outro cwd (ancoradas em REPOSITORY_ROOT)",
        rejects_run_dirs and rejects_outputs,
    )


def check_guard_accepts_c0_own_run_dir() -> bool:
    from gatefall.train.c0_fusion import _guard_run_dir

    return _check(
        "_guard_run_dir aceita o run_dir próprio da arma C0 "
        "(runs/local/le2i/c0_fusion), irmão dos runs de A, B0 e B1",
        not _raises_value_error(lambda: _guard_run_dir(_C0_RUN_DIR, "le2i")),
    )


def check_resolve_config_records_sources() -> bool:
    provenance = {"model_name": "facebook/sam3", "text_prompt": "person"}
    other_seed = C0_FUSION_CONFIG.seed + 1
    resolved = resolve_c0_config(
        other_seed,
        Path("src/gatefall/features/stats/pose_le2i_cs.json"),
        "deadbeef",
        Path("src/gatefall/features/stats/sam3_le2i_cs.json"),
        "cafebabe",
        Path("data/features/le2i/sam3"),
        "feedface",
        provenance,
    )
    provenance["model_name"] = "mutated-after-resolve"
    non_source_fields_match = replace(
        resolved,
        seed=C0_FUSION_CONFIG.seed,
        pose_standardization_stats_path="",
        pose_standardization_stats_sha256="",
        visual_standardization_stats_path="",
        visual_standardization_stats_sha256="",
        sam3_features_path="",
        sam3_features_sha256="",
        sam3_provenance={},
    ) == C0_FUSION_CONFIG
    return _check(
        "resolve_c0_config propaga a seed e registra estatísticas, digest e uma "
        "cópia da proveniência SAM 3, sem alterar o resto da receita",
        resolved.seed == other_seed
        and resolved.sam3_features_sha256 == "feedface"
        and resolved.sam3_features_path == "data/features/le2i/sam3"
        and resolved.sam3_provenance["model_name"] == "facebook/sam3"
        and non_source_fields_match,
    )


def check_run_train_default_seed_preserved() -> bool:
    from gatefall.train.c0_fusion import run_train

    default_seed = inspect.signature(run_train).parameters["seed"].default
    return _check(
        "run_train (C0) preserva C0_FUSION_CONFIG.seed como default do parâmetro seed",
        default_seed == C0_FUSION_CONFIG.seed,
    )


def check_report_allows_only_seed_and_param_count_to_differ() -> bool:
    from gatefall.train import c0_fusion

    source = inspect.getsource(c0_fusion.run_report)
    return _check(
        "run_report (C0) valida o run com fields_allowed_to_differ == "
        "{'seed', 'trainable_param_count'}",
        'frozenset({"seed", "trainable_param_count"})' in source,
    )


def run_c0_fusion_selftest() -> bool:
    checks = [
        check_guard_rejects_output_under_reference_root(),
        check_guard_rejects_protected_output_in_all_protected_run_dirs(),
        check_guard_rejects_comparison_run_dirs(),
        check_guards_are_anchored_at_repository_root(),
        check_guard_accepts_c0_own_run_dir(),
        check_resolve_config_records_sources(),
        check_run_train_default_seed_preserved(),
        check_report_allows_only_seed_and_param_count_to_differ(),
    ]
    ok = all(checks)
    if not ok:
        print("\nc0 fusion selftest FALHOU", file=sys.stderr)
    else:
        print("\nc0 fusion selftest OK: todas as checagens passaram")
    return ok
