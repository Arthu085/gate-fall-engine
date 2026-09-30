"""Validação das fontes SAM 3 usadas por C0 e C1."""

from dataclasses import dataclass

import pandas as pd

from gatefall.datasets import DatasetAdapter
from gatefall.features.sam3_standardization import Sam3StandardizationStats
from gatefall.features.sam3_standardization import load_stats as load_visual_stats
from gatefall.features.sam3_standardization import validate_stats_freshness as validate_visual_stats_freshness
from gatefall.features.sam3_standardization import validate_stats_layout as validate_visual_stats_layout
from gatefall.features.standardization import StandardizationStats
from gatefall.features.standardization import load_stats as load_pose_stats
from gatefall.features.standardization import validate_stats_layout as validate_pose_stats_layout
from gatefall.features.standardize_sam3 import sam3_stats_path, split_by_video
from gatefall.features.shared_le2i import shared_source
from gatefall.sam3.features import collect_sam3_provenance, sam3_set_sha256, validate_shared_sam3_set


@dataclass(frozen=True)
class _ValidatedInputs:
    frames: pd.DataFrame
    pose_stats: StandardizationStats
    visual_stats: Sam3StandardizationStats
    sam3_features_sha256: str
    sam3_provenance: dict[str, str]


def _validated_inputs(adapter: DatasetAdapter, dataset_name: str) -> _ValidatedInputs:
    """Valida identidade, proveniência e frescor de todas as fontes antes de montar janelas."""
    frames = adapter.load_frames()
    validate_shared_sam3_set(adapter)
    source_frames = (
        shared_source(adapter).load_frames()
        if adapter.identifier == "le2i-cv" else frames
    )
    splits = split_by_video(source_frames)
    sam3_provenance = collect_sam3_provenance(splits, sam3_root=adapter.sam3_root)
    sam3_features_sha256 = sam3_set_sha256(list(splits), sam3_root=adapter.sam3_root)

    pose_stats = load_pose_stats(adapter.pose_stats_path)
    validate_pose_stats_layout(pose_stats)
    visual_stats = load_visual_stats(sam3_stats_path(dataset_name))
    validate_visual_stats_layout(visual_stats, dataset_name=dataset_name)
    validate_visual_stats_freshness(visual_stats, adapter.frames_path, sam3_features_sha256)
    return _ValidatedInputs(
        frames=frames,
        pose_stats=pose_stats,
        visual_stats=visual_stats,
        sam3_features_sha256=sam3_features_sha256,
        sam3_provenance=sam3_provenance,
    )
