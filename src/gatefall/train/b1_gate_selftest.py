"""Selftest sintético dos guards de CLI da arma B1 (`b1_gate.py`). Não toca em
dados reais. Espelha `b0_fusion_selftest.py` e acrescenta a guarda que protege
o run_dir da arma B0."""

import inspect
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from gatefall.runs import (
    REFERENCE_RUN_ROOT,
    default_run_dir,
    default_run_dir_for_arm,
)
from gatefall.train.b1_config import B1_ADAPTIVE_GATE_CONFIG
from gatefall.train.b1_run import repository_anchored_run_dir

_B1_RUN_DIR = Path("runs/local/le2i/b1_adaptive_gate")


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


def check_guard_rejects_output_under_reference_root() -> bool:
    from gatefall.train.b1_gate import _guard_protected_output

    output_path = REFERENCE_RUN_ROOT / "le2i/b1_adaptive_gate/classification_report.json"
    return _check(
        "_guard_protected_output recusa --output sob runs/reference/ mesmo "
        "quando o nome do arquivo não é um dos PROTECTED_ARTIFACT_NAMES",
        _raises_value_error(
            lambda: _guard_protected_output(_B1_RUN_DIR, output_path, "le2i")
        ),
    )


def check_guard_rejects_protected_artifact_name() -> bool:
    from gatefall.train.b1_gate import PROTECTED_ARTIFACT_NAMES, _guard_protected_output

    all_raised = all(
        _raises_value_error(
            lambda name=name: _guard_protected_output(
                _B1_RUN_DIR, _B1_RUN_DIR / name, "le2i"
            )
        )
        for name in PROTECTED_ARTIFACT_NAMES
    )
    return _check(
        f"_guard_protected_output recusa --output apontando para cada um dos "
        f"artefatos protegidos {PROTECTED_ARTIFACT_NAMES}",
        all_raised,
    )


def check_guard_rejects_protected_output_in_arm_a_and_b0_run_dirs() -> bool:
    from gatefall.train.b1_gate import PROTECTED_ARTIFACT_NAMES, _guard_protected_output

    other_arm_run_dirs = (
        repository_anchored_run_dir(default_run_dir("le2i")),
        repository_anchored_run_dir(default_run_dir_for_arm("le2i", "b0_fusion")),
    )
    all_raised = all(
        _raises_value_error(
            lambda run_dir=run_dir, name=name: _guard_protected_output(
                _B1_RUN_DIR, run_dir / name, "le2i"
            )
        )
        for run_dir in other_arm_run_dirs
        for name in PROTECTED_ARTIFACT_NAMES
    )
    sibling_accepted = True
    try:
        _guard_protected_output(
            _B1_RUN_DIR,
            other_arm_run_dirs[0] / "classification_report.json",
            "le2i",
        )
    except ValueError:
        sibling_accepted = False
    return _check(
        "_guard_protected_output recusa --output apontando para um artefato "
        "protegido dentro do run_dir das armas A ou B0, mesmo com --run-dir do "
        "B1 — é essa guarda que impede o report do B1 de destruir o "
        "metrics.json do braço de comparação",
        all_raised and sibling_accepted,
    )


def check_guard_rejects_protected_output_in_canonical_b1_run_dir() -> bool:
    from gatefall.train.b1_gate import (
        ARM_NAME,
        PROTECTED_ARTIFACT_NAMES,
        _guard_protected_output,
    )

    canonical_b1_run_dir = repository_anchored_run_dir(
        default_run_dir_for_arm("le2i", ARM_NAME)
    )
    non_canonical_run_dir = canonical_b1_run_dir.parent / f"{ARM_NAME}_seed7"
    from_other_run_dir = all(
        _raises_value_error(
            lambda name=name: _guard_protected_output(
                non_canonical_run_dir, canonical_b1_run_dir / name, "le2i"
            )
        )
        for name in PROTECTED_ARTIFACT_NAMES
    )
    original_cwd = Path.cwd()
    try:
        os.chdir(tempfile.gettempdir())
        from_other_cwd = _raises_value_error(
            lambda: _guard_protected_output(
                non_canonical_run_dir,
                canonical_b1_run_dir / PROTECTED_ARTIFACT_NAMES[1],
                "le2i",
            )
        )
    finally:
        os.chdir(original_cwd)
    return _check(
        "_guard_protected_output recusa --output apontando para um artefato "
        "protegido do run canônico do B1 mesmo quando --run-dir é outro run do "
        "próprio B1 ou quando a CLI roda de outro cwd",
        from_other_run_dir and from_other_cwd,
    )


def check_guards_are_anchored_at_repository_root() -> bool:
    from gatefall.train.b1_gate import (
        PROTECTED_ARTIFACT_NAMES,
        _guard_not_arm_b0_run_dir,
        _guard_protected_output,
    )

    b0_run_dir = repository_anchored_run_dir(
        default_run_dir_for_arm("le2i", "b0_fusion")
    )
    original_cwd = Path.cwd()
    try:
        os.chdir(tempfile.gettempdir())
        rejects_run_dir = _raises_value_error(
            lambda: _guard_not_arm_b0_run_dir(b0_run_dir, "le2i")
        )
        rejects_output = _raises_value_error(
            lambda: _guard_protected_output(
                _B1_RUN_DIR, b0_run_dir / PROTECTED_ARTIFACT_NAMES[1], "le2i"
            )
        )
    finally:
        os.chdir(original_cwd)
    return _check(
        "as guardas continuam recusando o run_dir absoluto da arma B0 quando a "
        "CLI roda de outro cwd: o run_dir padrão é ancorado em "
        "REPOSITORY_ROOT, não no diretório corrente",
        rejects_run_dir and rejects_output,
    )


def check_guard_rejects_arm_a_run_dir() -> bool:
    from gatefall.train.b1_gate import _guard_not_arm_a_run_dir

    arm_a_run_dir = default_run_dir("le2i")
    same = _raises_value_error(
        lambda: _guard_not_arm_a_run_dir(arm_a_run_dir, "le2i")
    )
    ancestor = _raises_value_error(
        lambda: _guard_not_arm_a_run_dir(arm_a_run_dir.parent, "le2i")
    )
    descendant = _raises_value_error(
        lambda: _guard_not_arm_a_run_dir(arm_a_run_dir / "stray", "le2i")
    )
    return _check(
        "_guard_not_arm_a_run_dir recusa run_dir igual, ancestral ou "
        "descendente do run_dir do braço A",
        same and ancestor and descendant,
    )


def check_guard_rejects_arm_b0_run_dir() -> bool:
    from gatefall.train.b1_gate import _guard_not_arm_b0_run_dir

    b0_run_dir = default_run_dir_for_arm("le2i", "b0_fusion")
    same = _raises_value_error(
        lambda: _guard_not_arm_b0_run_dir(b0_run_dir, "le2i")
    )
    ancestor = _raises_value_error(
        lambda: _guard_not_arm_b0_run_dir(b0_run_dir.parent, "le2i")
    )
    descendant = _raises_value_error(
        lambda: _guard_not_arm_b0_run_dir(b0_run_dir / "stray", "le2i")
    )
    return _check(
        "_guard_not_arm_b0_run_dir recusa run_dir igual, ancestral ou "
        "descendente do run_dir da arma B0 — é essa guarda que impede um "
        "--force do B1 de destruir o run de comparação do B0",
        same and ancestor and descendant,
    )


def check_guard_accepts_b1_own_run_dir() -> bool:
    from gatefall.train.b1_gate import (
        _guard_not_arm_a_run_dir,
        _guard_not_arm_b0_run_dir,
    )

    accepted = True
    for guard in (_guard_not_arm_a_run_dir, _guard_not_arm_b0_run_dir):
        try:
            guard(default_run_dir_for_arm("le2i", "b1_adaptive_gate"), "le2i")
        except ValueError:
            accepted = False
    return _check(
        "as duas guardas aceitam o run_dir próprio da arma B1 "
        "(runs/local/le2i/b1_adaptive_gate), irmão dos runs de A e B0",
        accepted,
    )


def check_resolve_config_propagates_seed() -> bool:
    from gatefall.train.b1_gate import _resolve_config

    pose_stats_path = Path("src/gatefall/features/stats/pose_le2i_cs.json")
    visual_stats_path = Path("src/gatefall/features/stats/dinov3_le2i_cs.json")
    quality_root = Path("data/features/le2i/quality")
    other_seed = B1_ADAPTIVE_GATE_CONFIG.seed + 1

    resolved = _resolve_config(
        other_seed,
        pose_stats_path,
        "deadbeef",
        visual_stats_path,
        "cafebabe",
        quality_root,
        "feedface",
    )
    resolved_default = _resolve_config(
        B1_ADAPTIVE_GATE_CONFIG.seed,
        pose_stats_path,
        "deadbeef",
        visual_stats_path,
        "cafebabe",
        quality_root,
        "feedface",
    )

    non_seed_fields_match = replace(
        resolved,
        seed=B1_ADAPTIVE_GATE_CONFIG.seed,
        pose_standardization_stats_path=B1_ADAPTIVE_GATE_CONFIG.pose_standardization_stats_path,
        pose_standardization_stats_sha256=B1_ADAPTIVE_GATE_CONFIG.pose_standardization_stats_sha256,
        visual_standardization_stats_path=B1_ADAPTIVE_GATE_CONFIG.visual_standardization_stats_path,
        visual_standardization_stats_sha256=B1_ADAPTIVE_GATE_CONFIG.visual_standardization_stats_sha256,
        quality_features_path=B1_ADAPTIVE_GATE_CONFIG.quality_features_path,
        quality_features_sha256=B1_ADAPTIVE_GATE_CONFIG.quality_features_sha256,
    ) == B1_ADAPTIVE_GATE_CONFIG

    return _check(
        "_resolve_config propaga a seed via --seed e registra caminho e sha256 "
        "dos sidecars de qualidade, sem alterar o resto da receita "
        "compartilhada",
        resolved.seed == other_seed
        and resolved_default.seed == B1_ADAPTIVE_GATE_CONFIG.seed
        and resolved.quality_features_sha256 == "feedface"
        and resolved.quality_features_path == str(quality_root)
        and non_seed_fields_match,
    )


def check_run_train_default_seed_preserved() -> bool:
    from gatefall.train.b1_gate import run_train

    default_seed = inspect.signature(run_train).parameters["seed"].default
    return _check(
        "run_train (B1) preserva B1_ADAPTIVE_GATE_CONFIG.seed como default do "
        "parâmetro seed",
        default_seed == B1_ADAPTIVE_GATE_CONFIG.seed,
    )


def check_report_allows_only_seed_and_param_count_to_differ() -> bool:
    from gatefall.train import b1_gate

    source = inspect.getsource(b1_gate.run_report)
    return _check(
        "run_report (B1) valida o run com fields_allowed_to_differ == "
        "{'seed', 'trainable_param_count'}",
        'frozenset({"seed", "trainable_param_count"})' in source,
    )


def run_b1_gate_selftest() -> bool:
    checks = [
        check_guard_rejects_output_under_reference_root(),
        check_guard_rejects_protected_artifact_name(),
        check_guard_rejects_protected_output_in_arm_a_and_b0_run_dirs(),
        check_guard_rejects_protected_output_in_canonical_b1_run_dir(),
        check_guards_are_anchored_at_repository_root(),
        check_guard_rejects_arm_a_run_dir(),
        check_guard_rejects_arm_b0_run_dir(),
        check_guard_accepts_b1_own_run_dir(),
        check_resolve_config_propagates_seed(),
        check_run_train_default_seed_preserved(),
        check_report_allows_only_seed_and_param_count_to_differ(),
    ]
    ok = all(checks)
    if not ok:
        print("\nb1 gate selftest FALHOU", file=sys.stderr)
    else:
        print("\nb1 gate selftest OK: todas as checagens passaram")
    return ok
