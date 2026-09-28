"""Configuração e guardas compartilhadas para runs locais da arma C0."""

from dataclasses import replace
from pathlib import Path

from gatefall.train.baseline_c0.config import C0_FUSION_CONFIG, C0TrainConfig
from gatefall.datasets import DatasetAdapter
from gatefall.features.standardize_sam3 import SAM3_STATS_PATH
from gatefall.hashing import sha256_file
from gatefall.train.shared.sam3_inputs import _ValidatedInputs
from gatefall.train.shared.run_paths import comparison_run_dirs

ARM_NAME = "c0_fusion"

def resolve_c0_config(
    seed: int,
    pose_stats_path: Path,
    pose_stats_sha256: str,
    visual_stats_path: Path,
    visual_stats_sha256: str,
    sam3_root: Path,
    sam3_features_sha256: str,
    sam3_provenance: dict[str, str],
) -> C0TrainConfig:
    return replace(
        C0_FUSION_CONFIG,
        seed=seed,
        pose_standardization_stats_path=str(pose_stats_path),
        pose_standardization_stats_sha256=pose_stats_sha256,
        visual_standardization_stats_path=str(visual_stats_path),
        visual_standardization_stats_sha256=visual_stats_sha256,
        sam3_features_path=str(sam3_root),
        sam3_features_sha256=sam3_features_sha256,
        sam3_provenance=dict(sam3_provenance),
    )


def resolve_c0_config_for_inputs(
    seed: int,
    adapter: DatasetAdapter,
    inputs: _ValidatedInputs,
) -> C0TrainConfig:
    return resolve_c0_config(
        seed,
        adapter.pose_stats_path,
        sha256_file(adapter.pose_stats_path),
        SAM3_STATS_PATH,
        sha256_file(SAM3_STATS_PATH),
        adapter.sam3_root,
        inputs.sam3_features_sha256,
        inputs.sam3_provenance,
    )


def guard_not_comparison_run_dir(run_dir: Path, dataset_name: str) -> None:
    """Recusa run_dir igual, ancestral ou descendente dos runs canônicos de A, B0 e B1.

    Um `--force` do C0 faz backup-and-replace do run_dir inteiro; se ele
    coincidisse com, contivesse ou estivesse dentro de um run de comparação,
    destruiria esse run.
    """
    resolved_run_dir = run_dir.resolve()
    for arm, other_run_dir in comparison_run_dirs(dataset_name).items():
        if (
            resolved_run_dir == other_run_dir
            or resolved_run_dir in other_run_dir.parents
            or other_run_dir in resolved_run_dir.parents
        ):
            raise ValueError(
                f"run_dir {run_dir} coincide com, contém ou está dentro do "
                f"run_dir da arma {arm} ({other_run_dir}); a arma C0 precisa "
                "de um diretório próprio, irmão e nunca dentro e nunca "
                "contendo os runs de comparação"
            )
