"""Montagem do treino aumentado de T e S do piloto GMDCSA24 com rótulos fracos (braço A).

O treino de cada tratamento é o treino Le2i-CS do controle, sem mudança, mais
as janelas publicadas pelo adapter em `windows_t.parquet` ou
`windows_s.parquet`. As features são as do braço A (`build_pose_features`),
lidas da pose do próprio dataset de cada janela. A padronização e os pesos da
`CrossEntropyLoss` saem só desse treino aumentado; `val` do Le2i-CS é montada
como no controle, e `test` nunca é montada aqui.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, NUM_CLASSES, TRAIN_STRIDE
from gatefall.data.gmdcsa24.weak_label import (
    ENV,
    WINDOW_COLUMNS,
    WINDOW_FILES,
    Le2iReference,
    count_by,
    le2i_reference,
    read_table,
)
from gatefall.data.pose_dataset import PoseWindowDataset, WindowIndexDataset
from gatefall.data.windowing import build_window_index, window_frame_indices
from gatefall.datasets.le2i import LE2I_DATASET, LE2I_LABEL_NAMES, Le2iDatasetAdapter
from gatefall.features.standardization import StandardizationStats, compute_train_stats
from gatefall.hashing import sha256_dataframe
from gatefall.pose.kinematics import build_pose_features
from gatefall.pose.loading import load_person_found
from gatefall.train.baseline_a.engine import _class_weights, _collect_labels

FeatureLoader = Callable[[str, Path], np.ndarray]


def arm_a_features(video_id: str, pose_root: Path) -> np.ndarray:
    return build_pose_features(video_id, pose_root=pose_root)[0]


@dataclass(frozen=True)
class Le2iSide:
    """O Le2i-CS do controle, só leitura: grade, pose, padronização e referência de V2."""

    frames: pd.DataFrame
    frames_path: Path
    pose_root: Path
    stats_path: Path
    reference: Le2iReference


def load_le2i_side(adapter: Le2iDatasetAdapter = LE2I_DATASET) -> Le2iSide:
    frames = adapter.load_frames()
    return Le2iSide(
        frames=frames,
        frames_path=adapter.frames_path,
        pose_root=adapter.pose_root,
        stats_path=adapter.pose_stats_path,
        reference=le2i_reference(frames, adapter.load_manifest()),
    )


@dataclass(frozen=True)
class PoseRoots:
    le2i: Path
    gmdcsa24: Path

    def for_env(self, env: str) -> Path:
        return self.gmdcsa24 if env == ENV else self.le2i


@dataclass(frozen=True)
class TreatmentSources:
    le2i_train: pd.DataFrame
    gmdcsa24: pd.DataFrame
    train: WindowIndexDataset
    val: PoseWindowDataset


def le2i_windows(frames: pd.DataFrame, split: str) -> pd.DataFrame:
    """Índice de janelas de um split do Le2i-CS, como o controle o monta."""
    stride = TRAIN_STRIDE if split == "train" else EVAL_STRIDE
    split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
    return build_window_index(split_frames, stride=stride)


def gmdcsa24_windows(adapter_dir: Path, treatment: str) -> pd.DataFrame:
    return read_table(adapter_dir, WINDOW_FILES[treatment])


def training_index(le2i_train: pd.DataFrame, gmdcsa24: pd.DataFrame) -> pd.DataFrame:
    """Treino Le2i-CS do controle seguido das janelas do GMDCSA24 do tratamento."""
    parts = [cast(pd.DataFrame, frame[WINDOW_COLUMNS]) for frame in (le2i_train, gmdcsa24)]
    return pd.concat(parts, ignore_index=True)


def aligned_loader(
    windows: pd.DataFrame, roots: PoseRoots, load_features: FeatureLoader
) -> Callable[[str], np.ndarray]:
    """Features de cada vídeo lidas da pose do próprio dataset e conferidas contra o K do índice."""
    n_frames = {str(video): int(k) for video, k in zip(windows["video_id"], windows["n_frames"])}
    envs = {str(video): str(env) for video, env in zip(windows["video_id"], windows["env"])}

    def load(video_id: str) -> np.ndarray:
        features = load_features(video_id, roots.for_env(envs[video_id]))
        if features.shape[0] != n_frames[video_id]:
            raise ValueError(
                f"{video_id}: features com {features.shape[0]} quadros, o índice de "
                f"janelas registra K = {n_frames[video_id]}"
            )
        return features

    return load


def build_sources(
    le2i: Le2iSide,
    gmdcsa24: pd.DataFrame,
    roots: PoseRoots,
    load_features: FeatureLoader = arm_a_features,
) -> TreatmentSources:
    le2i_train = le2i_windows(le2i.frames, "train")
    train_windows = training_index(le2i_train, gmdcsa24)
    val = PoseWindowDataset(
        le2i.frames,
        "val",
        EVAL_STRIDE,
        aligned_loader(le2i_windows(le2i.frames, "val"), roots, load_features),
    )
    return TreatmentSources(
        le2i_train=le2i_train,
        gmdcsa24=gmdcsa24,
        train=WindowIndexDataset(
            train_windows, aligned_loader(train_windows, roots, load_features)
        ),
        val=val,
    )


def treatment_stats(train: WindowIndexDataset, frames_path: Path) -> StandardizationStats:
    """`compute_train_stats` sobre o treino aumentado; `frames_hash` passa a ser o
    SHA-256 do índice de treino combinado, já que o treino não vem de uma única grade."""
    stats = compute_train_stats(train, frames_path)
    return replace(stats, frames_hash=sha256_dataframe(train.windows))


def class_weights(train: WindowIndexDataset) -> list[float]:
    """Pesos de frequência inversa do engine do braço A, calculados só sobre o treino."""
    return [float(weight) for weight in _class_weights(_collect_labels(train), NUM_CLASSES)]


def class_counts(windows: pd.DataFrame) -> dict[str, int]:
    return count_by(windows, "label", list(range(NUM_CLASSES)), list(LE2I_LABEL_NAMES))


def window_detection_fraction(
    windows: pd.DataFrame, person_found: dict[str, np.ndarray]
) -> np.ndarray:
    """Fração dos quadros de cada janela com pessoa detectada."""
    fractions = np.empty(len(windows), dtype=np.float64)
    for position, row in enumerate(windows.to_dict("records")):
        found = person_found[str(row["video_id"])]
        indices = window_frame_indices(int(row["k_end"]), int(row["n_frames"]))
        fractions[position] = float(found[indices].mean())
    return fractions


def detection_coverage(
    pose_records: dict[str, dict[str, object]],
    windows: dict[str, pd.DataFrame],
    pose_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Cobertura descritiva da detecção de pose por vídeo e por janela de T e S.

    Não define limiar nem exclui vídeo ou janela: T e S continuam os publicados."""
    rows = []
    for video_id, record in sorted(pose_records.items()):
        k = int(cast(int, record["K"]))
        found = int(cast(int, record["n_found"]))
        rows.append(
            {
                "video_id": video_id,
                "K": k,
                "n_found": found,
                "coverage": found / k,
                **{
                    f"windows_{treatment}": int((frame["video_id"] == video_id).sum())
                    for treatment, frame in windows.items()
                },
            }
        )
    videos = pd.DataFrame(rows).sort_values(["coverage", "video_id"], ignore_index=True)
    person_found = {
        video_id: load_person_found(video_id, pose_root=pose_dir) for video_id in pose_records
    }
    summary: dict[str, dict[str, object]] = {}
    for treatment, frame in windows.items():
        fractions = window_detection_fraction(frame, person_found)
        summary[treatment] = {
            "windows": len(frame),
            "mean_detection_fraction": float(fractions.mean()) if len(frame) else None,
            "windows_full_detection": int((fractions == 1.0).sum()),
            "windows_partial_detection": int(((fractions > 0.0) & (fractions < 1.0)).sum()),
            "windows_without_detection": int((fractions == 0.0).sum()),
        }
    return videos, summary
