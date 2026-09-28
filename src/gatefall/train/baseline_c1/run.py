"""Configuração e isolamento de runs locais da arma C1."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pandas as pd

from gatefall.datasets import DatasetAdapter
from gatefall.features.standardize_sam3 import SAM3_STATS_PATH
from gatefall.hashing import sha256_file
from gatefall.pose.loading import pose_path
from gatefall.train.shared.sam3_inputs import _ValidatedInputs

from gatefall.runs import default_run_dir_for_arm
from gatefall.train.shared.run_paths import repository_anchored_run_dir
from gatefall.train.shared.run_paths import comparison_run_dirs as c0_comparison_run_dirs
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


def _source_hashes(adapter: DatasetAdapter, frames: pd.DataFrame, sam3_sha256: str) -> tuple[str, str]:
    digest = hashlib.sha256()
    for video_id in sorted(str(value) for value in frames["video_id"].unique()):
        path = pose_path(video_id, pose_root=adapter.pose_root)
        digest.update(f"{video_id} {sha256_file(path)}\n".encode("utf-8"))
    pose_sha256 = digest.hexdigest()
    quality = hashlib.sha256()
    quality.update(
        (
            f"{pose_sha256}\n{sam3_sha256}\n"
            f"{C1_ADAPTIVE_GATE_CONFIG.pose_quality_source}\n"
            f"{C1_ADAPTIVE_GATE_CONFIG.visual_quality_source}\n"
        ).encode("utf-8")
    )
    return pose_sha256, quality.hexdigest()


def resolve_c1_config_for_inputs(seed: int, adapter: DatasetAdapter, inputs: _ValidatedInputs) -> C1TrainConfig:
    pose_sha256, quality_sha256 = _source_hashes(adapter, inputs.frames, inputs.sam3_features_sha256)
    return resolve_c1_config(
        seed,
        adapter.pose_stats_path,
        sha256_file(adapter.pose_stats_path),
        SAM3_STATS_PATH,
        sha256_file(SAM3_STATS_PATH),
        pose_sha256,
        adapter.sam3_root,
        inputs.sam3_features_sha256,
        inputs.sam3_provenance,
        quality_sha256,
    )
