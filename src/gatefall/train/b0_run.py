"""Configuração e guardas compartilhadas para runs locais da arma B0."""

from dataclasses import replace
from pathlib import Path

from gatefall.runs import default_run_dir
from gatefall.train.b0_config import B0_FUSION_CONFIG, B0TrainConfig


def resolve_b0_config(
    seed: int,
    pose_stats_path: Path,
    pose_stats_sha256: str,
    visual_stats_path: Path,
    visual_stats_sha256: str,
) -> B0TrainConfig:
    return replace(
        B0_FUSION_CONFIG,
        seed=seed,
        pose_standardization_stats_path=str(pose_stats_path),
        pose_standardization_stats_sha256=pose_stats_sha256,
        visual_standardization_stats_path=str(visual_stats_path),
        visual_standardization_stats_sha256=visual_stats_sha256,
    )


def guard_not_arm_a_run_dir(run_dir: Path, dataset_name: str) -> None:
    resolved_run_dir = run_dir.resolve()
    arm_a_run_dir = default_run_dir(dataset_name).resolve()
    is_same = resolved_run_dir == arm_a_run_dir
    is_ancestor = resolved_run_dir in arm_a_run_dir.parents
    is_descendant = arm_a_run_dir in resolved_run_dir.parents
    if is_same or is_ancestor or is_descendant:
        raise ValueError(
            f"run_dir {run_dir} coincide com, contém ou está dentro do "
            f"run_dir da arma A ({arm_a_run_dir}); a arma B0 precisa de um "
            "diretório próprio, irmão e nunca dentro e nunca contendo o run "
            "de referência da arma A"
        )
