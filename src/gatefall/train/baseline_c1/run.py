"""Configuração e isolamento de runs locais da arma C1."""

from dataclasses import replace
from pathlib import Path

from gatefall.runs import default_run_dir_for_arm
from gatefall.train.baseline_b1.run import repository_anchored_run_dir
from gatefall.train.baseline_c0.run import comparison_run_dirs as c0_comparison_run_dirs
from gatefall.train.baseline_c1.config import C1_ADAPTIVE_GATE_CONFIG, C1TrainConfig

ARM_NAME = "c1_adaptive_gate"


def comparison_run_dirs(dataset_name: str) -> dict[str, Path]:
    return c0_comparison_run_dirs(dataset_name) | {
        "C0": repository_anchored_run_dir(default_run_dir_for_arm(dataset_name, "c0_fusion"))
    }


def guard_not_comparison_run_dir(run_dir: Path, dataset_name: str) -> None:
    resolved = run_dir.resolve()
    for arm, protected in comparison_run_dirs(dataset_name).items():
        if resolved == protected or resolved in protected.parents or protected in resolved.parents:
            raise ValueError(
                f"run_dir {run_dir} coincide com, contém ou está dentro do run_dir "
                f"da arma {arm} ({protected}); C1 exige um diretório próprio"
            )


def resolve_c1_config(
    seed: int,
    pose_stats_path: Path,
    pose_stats_sha256: str,
    visual_stats_path: Path,
    visual_stats_sha256: str,
    pose_features_sha256: str,
    sam3_root: Path,
    sam3_features_sha256: str,
    sam3_provenance: dict[str, str],
    quality_sha256: str,
) -> C1TrainConfig:
    return replace(
        C1_ADAPTIVE_GATE_CONFIG,
        seed=seed,
        pose_standardization_stats_path=str(pose_stats_path),
        pose_standardization_stats_sha256=pose_stats_sha256,
        visual_standardization_stats_path=str(visual_stats_path),
        visual_standardization_stats_sha256=visual_stats_sha256,
        quality_features_path="derived:pose+sam3",
        quality_features_sha256=quality_sha256,
        pose_features_sha256=pose_features_sha256,
        sam3_features_path=str(sam3_root),
        sam3_features_sha256=sam3_features_sha256,
        sam3_provenance=dict(sam3_provenance),
    )
