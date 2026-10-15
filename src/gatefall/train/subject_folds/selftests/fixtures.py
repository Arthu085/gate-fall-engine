import contextlib
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis import subject_fold_feasibility
from gatefall.pose.kinematics import POSE_FEATURE_DIM
from gatefall.train.subject_folds.cli import PoseInputs

W, F, FN, SD, ST, SU, SG, OT = 0, 1, 2, 3, 4, 7, 8, 9
N_FRAMES = 40
CYCLE = [W] * 5 + [F] * 5 + [FN] * 5 + [SD] * 5 + [ST] * 5 + [SU] * 5 + [SG] * 5 + [OT] * 5
# Os sujeitos de train são os mesmos IDs brutos do Le2i-CS ({0, 1, 3, 4, 6,
# 8}), então os grupos congelados {0}, {1} e {3, 4, 6, 8} valem sem mudança.
# Os sujeitos 0 e 1 aparecem em dois ambientes. Val (sujeito 5) e test
# (sujeitos 2 e 7) usam ambientes próprios: qualquer leitura deles aparece nos
# registros do loader ou em folds.json.
VIDEO_SPEC: tuple[tuple[str, str, str, int], ...] = (
    ("EnvA/s0a", "train", "EnvA", 0),
    ("EnvB/s0b", "train", "EnvB", 0),
    ("EnvA/s1a", "train", "EnvA", 1),
    ("EnvC/s1c", "train", "EnvC", 1),
    ("EnvB/s3", "train", "EnvB", 3),
    ("EnvC/s4", "train", "EnvC", 4),
    ("EnvA/s6", "train", "EnvA", 6),
    ("EnvB/s8", "train", "EnvB", 8),
    ("EnvVal/v5", "val", "EnvVal", 5),
    ("EnvTest/t2", "test", "EnvTest", 2),
    ("EnvTest/t7", "test", "EnvTest", 7),
)
TRAIN_VIDEOS = tuple(video_id for video_id, split, _env, _subject in VIDEO_SPEC if split == "train")
OFFICIAL_HELD_OUT_VIDEOS = tuple(video_id for video_id, split, _env, _subject in VIDEO_SPEC if split != "train")
SUBJECT_BY_VIDEO = {video_id: subject for video_id, _split, _env, subject in VIDEO_SPEC}
# Deslocamento grande e exclusivo do sujeito 0: uma estatística que misture o
# sujeito retido fica visivelmente longe da média do treino de desenvolvimento.
SUBJECT_OFFSET = {0: 500.0, 1: 2.0, 3: 0.0, 4: 1.0, 6: -1.0, 8: 3.0, 5: -900.0, 2: 900.0, 7: 900.0}
SYNTHETIC_POSE_SHA256 = "0" * 64


def _labels(index: int) -> list[int]:
    shift = index * 3
    return [CYCLE[(frame + shift) % len(CYCLE)] for frame in range(N_FRAMES)]


def frames_table() -> pd.DataFrame:
    rows = [
        {"video_id": video_id, "split": split, "env": env, "subject": subject, "frame_index": frame, "label": label}
        for index, (video_id, split, env, subject) in enumerate(VIDEO_SPEC)
        for frame, label in enumerate(_labels(index))
    ]
    return pd.DataFrame(rows)


def official_train_frames(frames: pd.DataFrame | None = None) -> pd.DataFrame:
    frames = frames_table() if frames is None else frames
    return cast(pd.DataFrame, frames[frames["split"] == "train"]).reset_index(drop=True)


def manifest_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": video_id, "split": split, "env": env, "cam": 1, "subject": subject,
             "width": 320, "height": 240, "fps": 25.0}
            for video_id, split, env, subject in VIDEO_SPEC
        ]
    )


def pose_features(video_id: str) -> np.ndarray:
    rng = np.random.default_rng(sum(ord(char) for char in video_id))
    offset = SUBJECT_OFFSET[SUBJECT_BY_VIDEO[video_id]]
    return (rng.normal(size=(N_FRAMES, POSE_FEATURE_DIM)) + offset).astype(np.float32)


class RecordingPose:
    def __init__(self, sha256: str = SYNTHETIC_POSE_SHA256) -> None:
        self.calls: list[str] = []
        self.sha256 = sha256

    def inputs(self) -> PoseInputs:
        def loader(video_id: str) -> np.ndarray:
            self.calls.append(video_id)
            return pose_features(video_id)

        return PoseInputs(loader=loader, sha256=self.sha256)

    def loaded_videos(self) -> set[str]:
        return set(self.calls)


def write_inputs(root: Path, frames: pd.DataFrame | None = None) -> tuple[Path, Path]:
    frames_path = root / "processed" / "frames.parquet"
    manifest_path = root / "processed" / "manifest.parquet"
    frames_path.parent.mkdir(parents=True)
    (frames_table() if frames is None else frames).to_parquet(frames_path)
    manifest_table().to_parquet(manifest_path)
    return frames_path, manifest_path


def synthetic_expected_train() -> dict[str, object]:
    train = official_train_frames()
    return {
        "environments": ("EnvA", "EnvB", "EnvC"),
        "n_videos": len(TRAIN_VIDEOS),
        "windows_total_eval_stride": len(build_window_index(train, stride=EVAL_STRIDE, drop_ignored=False)),
        "windows_labeled_eval_stride": len(build_window_index(train, stride=EVAL_STRIDE, drop_ignored=True)),
    }


@contextlib.contextmanager
def synthetic_workspace() -> Iterator[tuple[Path, Path, Path, Path]]:
    """Raiz das dobras, frames, manifesto e raiz de pose (inexistente: a pose vem de `RecordingPose`)."""
    with tempfile.TemporaryDirectory() as directory, patch.dict(
        subject_fold_feasibility.EXPECTED_CS_TRAIN, synthetic_expected_train()
    ):
        root = Path(directory)
        frames_path, manifest_path = write_inputs(root)
        yield root / "folds", frames_path, manifest_path, root / "pose"


def relabel(frames: pd.DataFrame, subjects: set[int], source: int, target: int) -> pd.DataFrame:
    changed = frames.copy()
    mask = changed["subject"].isin(list(subjects)) & (changed["label"] == source)
    changed.loc[mask, "label"] = target
    return changed
