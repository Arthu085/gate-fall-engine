"""Selftest sintético dos guards de CLI da arma B0 (`b0_fusion.py`). Não toca
em dados reais. Espelha `baseline_a_selftest.py` adaptado à arma B0."""

import inspect
import sys
from dataclasses import replace
from pathlib import Path

from gatefall.runs import REFERENCE_RUN_ROOT, default_run_dir
from gatefall.train.b0_config import B0_FUSION_CONFIG


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_guard_rejects_output_under_reference_root() -> bool:
    from gatefall.train.b0_fusion import _guard_protected_output

    run_dir = Path("runs/local/le2i/b0_fusion")
    output_path = REFERENCE_RUN_ROOT / "le2i/b0_fusion/classification_report.json"

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


def check_guard_rejects_protected_artifact_name() -> bool:
    from gatefall.train.b0_fusion import PROTECTED_ARTIFACT_NAMES, _guard_protected_output

    run_dir = Path("runs/local/le2i/b0_fusion")

    all_raised = True
    for name in PROTECTED_ARTIFACT_NAMES:
        raised = False
        try:
            _guard_protected_output(run_dir, run_dir / name)
        except ValueError:
            raised = True
        all_raised = all_raised and raised

    return _check(
        f"_guard_protected_output recusa --output apontando para cada um dos "
        f"artefatos protegidos {PROTECTED_ARTIFACT_NAMES}",
        all_raised,
    )


def check_guard_rejects_arm_a_run_dir() -> bool:
    from gatefall.train.b0_fusion import _guard_not_arm_a_run_dir

    arm_a_run_dir = default_run_dir("le2i")

    raised = False
    try:
        _guard_not_arm_a_run_dir(arm_a_run_dir, "le2i")
    except ValueError:
        raised = True

    return _check(
        "_guard_not_arm_a_run_dir recusa run_dir igual ao default_run_dir "
        "('le2i') da arma A, evitando sobrescrever o run de referência",
        raised,
    )


def check_guard_rejects_ancestor_of_arm_a_run_dir() -> bool:
    from gatefall.train.b0_fusion import _guard_not_arm_a_run_dir

    arm_a_run_dir = default_run_dir("le2i")
    dataset_root_run_dir = arm_a_run_dir.parent

    raised = False
    try:
        _guard_not_arm_a_run_dir(dataset_root_run_dir, "le2i")
    except ValueError:
        raised = True

    return _check(
        "_guard_not_arm_a_run_dir recusa run_dir que é ANCESTRAL do run_dir "
        "da arma A (ex.: a raiz do dataset), evitando que --force renomeie "
        "toda a árvore contendo o run de referência da arma A",
        raised,
    )


def check_guard_rejects_descendant_of_arm_a_run_dir() -> bool:
    from gatefall.train.b0_fusion import _guard_not_arm_a_run_dir

    arm_a_run_dir = default_run_dir("le2i")
    nested_run_dir = arm_a_run_dir / "stray"

    raised = False
    try:
        _guard_not_arm_a_run_dir(nested_run_dir, "le2i")
    except ValueError:
        raised = True

    return _check(
        "_guard_not_arm_a_run_dir recusa run_dir que é DESCENDENTE do "
        "run_dir da arma A, evitando aninhar a arma B0 dentro do run de "
        "referência da arma A",
        raised,
    )


def check_guard_accepts_b0_own_run_dir() -> bool:
    from gatefall.train.b0_fusion import _guard_not_arm_a_run_dir

    b0_run_dir = Path("runs/local/le2i/b0_fusion")

    accepted = True
    try:
        _guard_not_arm_a_run_dir(b0_run_dir, "le2i")
    except ValueError:
        accepted = False

    return _check(
        "_guard_not_arm_a_run_dir aceita o run_dir próprio da arma B0 "
        "(distinto do run_dir da arma A)",
        accepted,
    )


def check_resolve_config_propagates_seed() -> bool:
    from gatefall.train.b0_fusion import _resolve_config

    pose_stats_path = Path("src/gatefall/features/stats/pose_le2i_cs.json")
    pose_stats_sha256 = "deadbeef"
    visual_stats_path = Path("src/gatefall/features/stats/dinov3_le2i_cs.json")
    visual_stats_sha256 = "cafebabe"
    other_seed = B0_FUSION_CONFIG.seed + 1

    resolved = _resolve_config(
        other_seed, pose_stats_path, pose_stats_sha256, visual_stats_path, visual_stats_sha256
    )
    seed_propagated = resolved.seed == other_seed

    resolved_default = _resolve_config(
        B0_FUSION_CONFIG.seed,
        pose_stats_path,
        pose_stats_sha256,
        visual_stats_path,
        visual_stats_sha256,
    )
    default_seed_preserved = resolved_default.seed == B0_FUSION_CONFIG.seed

    non_seed_fields_match = replace(
        resolved,
        seed=B0_FUSION_CONFIG.seed,
        pose_standardization_stats_path=B0_FUSION_CONFIG.pose_standardization_stats_path,
        pose_standardization_stats_sha256=B0_FUSION_CONFIG.pose_standardization_stats_sha256,
        visual_standardization_stats_path=B0_FUSION_CONFIG.visual_standardization_stats_path,
        visual_standardization_stats_sha256=B0_FUSION_CONFIG.visual_standardization_stats_sha256,
    ) == B0_FUSION_CONFIG

    return _check(
        "_resolve_config propaga a seed via --seed para dentro da config "
        "persistida, sem alterar o resto da receita compartilhada",
        seed_propagated and default_seed_preserved and non_seed_fields_match,
    )


def check_run_train_default_seed_preserved() -> bool:
    from gatefall.train.b0_fusion import run_train

    default_seed = inspect.signature(run_train).parameters["seed"].default
    return _check(
        "run_train (B0) preserva B0_FUSION_CONFIG.seed como default do "
        "parâmetro seed",
        default_seed == B0_FUSION_CONFIG.seed,
    )


def run_b0_fusion_selftest() -> bool:
    checks = [
        check_guard_rejects_output_under_reference_root(),
        check_guard_rejects_protected_artifact_name(),
        check_guard_rejects_arm_a_run_dir(),
        check_guard_rejects_ancestor_of_arm_a_run_dir(),
        check_guard_rejects_descendant_of_arm_a_run_dir(),
        check_guard_accepts_b0_own_run_dir(),
        check_resolve_config_propagates_seed(),
        check_run_train_default_seed_preserved(),
    ]
    ok = all(checks)
    if not ok:
        print("\nb0 fusion selftest FALHOU", file=sys.stderr)
    else:
        print("\nb0 fusion selftest OK: todas as checagens passaram")
    return ok
