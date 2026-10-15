"""Estatísticas de padronização de pose por dobra, ajustadas só no treino de desenvolvimento.

Cada dobra grava `pose.json`, no formato das estatísticas oficiais, e
`provenance.json`, que liga o arquivo à dobra, aos sujeitos e vídeos de
desenvolvimento usados no ajuste e aos hashes das entradas. O arquivo oficial
`pose_le2i_cs.json` foi ajustado em todo o `train`, inclusive nos sujeitos
retidos: ele nunca é gravado aqui, e um `pose.json` idêntico a ele é recusado.
"""

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from gatefall.config import TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets.le2i import PROTOCOL_POSE_STATS_PATHS
from gatefall.eval.analysis.classification_diagnosis import display_path
from gatefall.features import standardization
from gatefall.features.standardization import StandardizationStats
from gatefall.hashing import sha256_file
from gatefall.runs import REPOSITORY_ROOT
from gatefall.train.internal_folds.folds import DEVELOPMENT_SPLIT, FeatureLoader, restricted_loader
from gatefall.train.subject_folds.folds import DATASET, PROTOCOL, SubjectFoldPartition

POSE_STATS_FILE = "pose.json"
PROVENANCE_FILE = "provenance.json"
OFFICIAL_POSE_STATS = REPOSITORY_ROOT / PROTOCOL_POSE_STATS_PATHS["cs"]


@dataclass(frozen=True)
class FoldPoseStats:
    stats: StandardizationStats
    path: Path
    sha256: str
    window_count: int


def expected_provenance(
    fold_record: dict[str, object], input_sha256: dict[str, str], pose_features_sha256: str
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "canonical": False,
        "dataset": DATASET,
        "fold": fold_record["fold"],
        "retained_subjects": fold_record["retained_subjects"],
        "development_subjects": fold_record["development_subjects"],
        "fit_split": DEVELOPMENT_SPLIT,
        "development_video_ids": fold_record["development_video_ids"],
        "stride": TRAIN_STRIDE,
        "window_count": fold_record["development_windows_train_stride"],
        "input_sha256": input_sha256,
        "pose_features_sha256": pose_features_sha256,
    }


def _write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fit_fold_stats(
    partition: SubjectFoldPartition,
    directory: Path,
    frames_path: Path,
    input_sha256: dict[str, str],
    pose_features_sha256: str,
    pose_loader: FeatureLoader,
    force: bool,
) -> FoldPoseStats:
    targets = [directory / POSE_STATS_FILE, directory / PROVENANCE_FILE]
    if any(path.exists() for path in targets) and not force:
        try:
            stats = load_fold_stats(partition.membership_record, directory, input_sha256, pose_features_sha256)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(
                f"estatísticas da dobra em {directory} existem mas são inválidas ({exc}); use --force para refazer"
            ) from exc
        print(f"skip {directory} (estatísticas da dobra completas e íntegras)")
        return stats

    directory.mkdir(parents=True, exist_ok=True)
    (directory / PROVENANCE_FILE).unlink(missing_ok=True)
    source = PoseWindowDataset(
        partition.frames,
        DEVELOPMENT_SPLIT,
        TRAIN_STRIDE,
        restricted_loader(pose_loader, partition.development_video_ids, f"{partition.fold}/pose/estatísticas"),
    )
    provenance = expected_provenance(partition.membership_record, input_sha256, pose_features_sha256)
    if len(source) != provenance["window_count"]:
        raise RuntimeError(f"{partition.fold}: janelas de desenvolvimento divergem da composição da dobra")
    stats = standardization.compute_train_stats(source, frames_path, stride=TRAIN_STRIDE)
    standardization.save_stats(stats, directory / POSE_STATS_FILE, force=True)
    provenance["pose_stats_sha256"] = sha256_file(directory / POSE_STATS_FILE)
    _write_json_atomic(directory / PROVENANCE_FILE, provenance)
    return load_fold_stats(partition.membership_record, directory, input_sha256, pose_features_sha256)


def load_fold_stats(
    fold_record: dict[str, object], directory: Path, input_sha256: dict[str, str], pose_features_sha256: str
) -> FoldPoseStats:
    """Valida as estatísticas de uma dobra contra o registro gravado da dobra e os hashes das entradas."""
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
    pose_sha256 = sha256_file(pose_path)
    if OFFICIAL_POSE_STATS.is_file() and pose_sha256 == sha256_file(OFFICIAL_POSE_STATS):
        raise ValueError(f"{pose_path} é a estatística oficial do Le2i-CS, ajustada também nos sujeitos retidos")
    expected = {
        **expected_provenance(fold_record, input_sha256, pose_features_sha256),
        "pose_stats_sha256": pose_sha256,
    }
    divergent = sorted(
        field for field in set(expected) | set(provenance) if provenance.get(field) != expected.get(field)
    )
    if divergent:
        raise ValueError(f"{provenance_path}: proveniência diverge da dobra registrada: {', '.join(divergent)}")

    window_count = expected["window_count"]
    stats = standardization.load_stats(pose_path)
    standardization.validate_stats_layout(stats)
    if stats.frames_hash != input_sha256["frames"]:
        raise ValueError(f"{directory}: estatísticas ajustadas sobre frames diferentes dos registrados")
    if stats.window_count != window_count:
        raise ValueError(f"{directory}: window_count das estatísticas diverge da proveniência")
    return FoldPoseStats(
        stats=stats, path=Path(display_path(pose_path)), sha256=pose_sha256, window_count=int(window_count)
    )
