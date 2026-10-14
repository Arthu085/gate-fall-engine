"""Estatísticas de padronização por dobra, ajustadas só no treino de desenvolvimento.

Cada dobra grava `pose.json` e `dinov3.json` no formato das estatísticas
oficiais, mais `provenance.json`, que liga os arquivos à dobra, aos vídeos de
desenvolvimento usados no ajuste e aos hashes das entradas. Os arquivos
oficiais `*_le2i_cv.json` nunca são lidos nem gravados aqui.
"""

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from gatefall.config import TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.eval.analysis.classification_diagnosis import display_path
from gatefall.features import dinov3_standardization, standardization
from gatefall.features.dinov3_standardization import Dinov3StandardizationStats
from gatefall.features.standardization import StandardizationStats
from gatefall.hashing import sha256_file
from gatefall.train.internal_folds.folds import (
    DATASET,
    DEVELOPMENT_SPLIT,
    FeatureLoader,
    FoldPartition,
    restricted_loader,
)

POSE_STATS_FILE = "pose.json"
DINOV3_STATS_FILE = "dinov3.json"
PROVENANCE_FILE = "provenance.json"


@dataclass(frozen=True)
class StatsFiles:
    pose_path: Path
    pose_sha256: str
    dinov3_path: Path
    dinov3_sha256: str


@dataclass(frozen=True)
class FoldStats:
    pose: StandardizationStats
    dinov3: Dinov3StandardizationStats
    files: StatsFiles
    window_count: int


def _development_source(partition: FoldPartition, loader: FeatureLoader, modality: str) -> PoseWindowDataset:
    return PoseWindowDataset(
        partition.frames,
        DEVELOPMENT_SPLIT,
        TRAIN_STRIDE,
        restricted_loader(loader, partition.development_video_ids, f"{partition.fold}/{modality}/estatísticas"),
    )


def expected_provenance(fold_record: dict[str, object], input_sha256: dict[str, str]) -> dict:
    return {
        "canonical": False,
        "dataset": DATASET,
        "fold": fold_record["fold"],
        "retained_environment": fold_record["retained_environment"],
        "development_environments": fold_record["development_environments"],
        "fit_split": DEVELOPMENT_SPLIT,
        "development_video_ids": fold_record["development_video_ids"],
        "stride": TRAIN_STRIDE,
        "window_count": fold_record["development_windows_train_stride"],
        "input_sha256": input_sha256,
    }


def _write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fit_fold_stats(
    partition: FoldPartition,
    directory: Path,
    frames_path: Path,
    input_sha256: dict[str, str],
    pose_loader: FeatureLoader,
    visual_loader: FeatureLoader,
    force: bool,
) -> FoldStats:
    targets = [directory / name for name in (POSE_STATS_FILE, DINOV3_STATS_FILE, PROVENANCE_FILE)]
    if any(path.exists() for path in targets) and not force:
        try:
            stats = load_fold_stats(partition, directory, frames_path, input_sha256)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(
                f"estatísticas da dobra em {directory} existem mas são inválidas ({exc}); use --force para refazer"
            ) from exc
        print(f"skip {directory} (estatísticas da dobra completas e íntegras)")
        return stats

    directory.mkdir(parents=True, exist_ok=True)
    (directory / PROVENANCE_FILE).unlink(missing_ok=True)
    pose_source = _development_source(partition, pose_loader, "pose")
    visual_source = _development_source(partition, visual_loader, "dinov3")
    provenance = expected_provenance(partition.membership_record, input_sha256)
    if len(pose_source) != provenance["window_count"]:
        raise RuntimeError(f"{partition.fold}: janelas de desenvolvimento divergem da composição da dobra")
    pose_stats = standardization.compute_train_stats(pose_source, frames_path, stride=TRAIN_STRIDE)
    visual_stats = dinov3_standardization.compute_train_stats(
        visual_source, frames_path, DATASET, stride=TRAIN_STRIDE
    )
    standardization.save_stats(pose_stats, directory / POSE_STATS_FILE, force=True)
    dinov3_standardization.save_stats(visual_stats, directory / DINOV3_STATS_FILE, force=True)
    provenance["files"] = {
        "pose": sha256_file(directory / POSE_STATS_FILE),
        "dinov3": sha256_file(directory / DINOV3_STATS_FILE),
    }
    _write_json_atomic(directory / PROVENANCE_FILE, provenance)
    return load_fold_stats(partition, directory, frames_path, input_sha256)


def _load_recorded_stats(
    directory: Path, fold_record: dict[str, object], input_sha256: dict[str, str]
) -> FoldStats:
    provenance_path = directory / PROVENANCE_FILE
    if not provenance_path.is_file():
        raise FileNotFoundError(f"{provenance_path} ausente; rode `prepare` antes")
    try:
        with provenance_path.open(encoding="utf-8") as stream:
            provenance = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{provenance_path} não é um JSON válido: {exc}") from exc
    if not isinstance(provenance, dict):
        raise ValueError(f"{provenance_path} deve ser um objeto")
    pose_path = directory / POSE_STATS_FILE
    dinov3_path = directory / DINOV3_STATS_FILE
    files = StatsFiles(
        pose_path=Path(display_path(pose_path)),
        pose_sha256=sha256_file(pose_path),
        dinov3_path=Path(display_path(dinov3_path)),
        dinov3_sha256=sha256_file(dinov3_path),
    )
    expected = {
        **expected_provenance(fold_record, input_sha256),
        "files": {"pose": files.pose_sha256, "dinov3": files.dinov3_sha256},
    }
    divergent = sorted(
        field for field in set(expected) | set(provenance) if provenance.get(field) != expected.get(field)
    )
    if divergent:
        raise ValueError(f"{provenance_path}: proveniência diverge da dobra registrada: {', '.join(divergent)}")

    window_count = expected["window_count"]
    pose = standardization.load_stats(pose_path)
    standardization.validate_stats_layout(pose)
    dinov3 = dinov3_standardization.load_stats(dinov3_path)
    dinov3_standardization.validate_stats_layout(dinov3, dataset_name=DATASET)
    if pose.frames_hash != input_sha256["frames"] or dinov3.frames_hash != input_sha256["frames"]:
        raise ValueError(f"{directory}: estatísticas ajustadas sobre frames diferentes dos registrados")
    if pose.window_count != window_count or dinov3.window_count != window_count:
        raise ValueError(f"{directory}: window_count das estatísticas diverge da proveniência")
    return FoldStats(pose=pose, dinov3=dinov3, files=files, window_count=window_count)


def load_recorded_stats_files(
    directory: Path, fold_record: dict[str, object], input_sha256: dict[str, str]
) -> StatsFiles:
    """Valida as estatísticas de uma dobra só contra os registros gravados, sem ler as entradas."""
    return _load_recorded_stats(directory, fold_record, input_sha256).files


def load_fold_stats(
    partition: FoldPartition, directory: Path, frames_path: Path, input_sha256: dict[str, str]
) -> FoldStats:
    stats = _load_recorded_stats(directory, partition.membership_record, input_sha256)
    dinov3_standardization.validate_stats_freshness(stats.dinov3, frames_path)
    return stats
