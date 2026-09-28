"""Configuração e guardas de runs locais da arma B1."""

from dataclasses import replace
from pathlib import Path

from gatefall.runs import (
    default_run_dir,
    default_run_dir_for_arm,
)
from gatefall.train.baseline_b1.config import B1_ADAPTIVE_GATE_CONFIG, B1TrainConfig
from gatefall.train.shared.run_paths import repository_anchored_run_dir

B0_ARM_NAME = "b0_fusion"


def resolve_b1_config(
    seed: int,
    pose_stats_path: Path,
    pose_stats_sha256: str,
    visual_stats_path: Path,
    visual_stats_sha256: str,
    quality_root: Path,
    quality_sha256: str,
) -> B1TrainConfig:
    return replace(
        B1_ADAPTIVE_GATE_CONFIG,
        seed=seed,
        pose_standardization_stats_path=str(pose_stats_path),
        pose_standardization_stats_sha256=pose_stats_sha256,
        visual_standardization_stats_path=str(visual_stats_path),
        visual_standardization_stats_sha256=visual_stats_sha256,
        quality_features_path=str(quality_root),
        quality_features_sha256=quality_sha256,
    )


def guard_not_arm_a_run_dir(run_dir: Path, dataset_name: str) -> None:
    resolved_run_dir = run_dir.resolve()
    arm_a_run_dir = repository_anchored_run_dir(default_run_dir(dataset_name))
    is_same = resolved_run_dir == arm_a_run_dir
    is_ancestor = resolved_run_dir in arm_a_run_dir.parents
    is_descendant = arm_a_run_dir in resolved_run_dir.parents
    if is_same or is_ancestor or is_descendant:
        raise ValueError(
            f"run_dir {run_dir} coincide com, contém ou está dentro do "
            f"run_dir da arma A ({arm_a_run_dir}); a arma B1 precisa de um "
            "diretório próprio, irmão e nunca dentro e nunca contendo o run "
            "de referência da arma A"
        )


def guard_not_arm_b0_run_dir(run_dir: Path, dataset_name: str) -> None:
    resolved_run_dir = run_dir.resolve()
    arm_b0_run_dir = repository_anchored_run_dir(
        default_run_dir_for_arm(dataset_name, B0_ARM_NAME)
    )
    is_same = resolved_run_dir == arm_b0_run_dir
    is_ancestor = resolved_run_dir in arm_b0_run_dir.parents
    is_descendant = arm_b0_run_dir in resolved_run_dir.parents
    if is_same or is_ancestor or is_descendant:
        raise ValueError(
            f"run_dir {run_dir} coincide com, contém ou está dentro do "
            f"run_dir da arma B0 ({arm_b0_run_dir}); a arma B1 precisa de um "
            "diretório próprio, irmão e nunca dentro e nunca contendo o run "
            "de comparação da arma B0"
        )
