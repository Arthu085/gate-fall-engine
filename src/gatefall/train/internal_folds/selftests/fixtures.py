import contextlib
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis import internal_fold_feasibility
from gatefall.features.dinov3_standardization import FEATURE_DIM as DINOV3_FEATURE_DIM
from gatefall.pose.kinematics import POSE_FEATURE_DIM
from gatefall.train.internal_folds.training import FeatureLoaders, QualityFeatures

W, F, FN, SD, ST, SU, SG, OT = 0, 1, 2, 3, 4, 7, 8, 9
N_FRAMES = 40
ENVIRONMENTS = ("EnvA", "EnvB", "EnvC")
# Cada ambiente de train tem dois vídeos com todas as classes restritas. Os
# vídeos de val e test oficiais usam ambientes próprios: qualquer leitura
# deles aparece nos registros dos loaders ou em folds.json.
CYCLE = [W] * 5 + [F] * 5 + [FN] * 5 + [SD] * 5 + [ST] * 5 + [SU] * 5 + [SG] * 5 + [OT] * 5
VIDEO_SPEC: tuple[tuple[str, str, str, int], ...] = (
    ("EnvA/a1", "train", "EnvA", 1),
    ("EnvA/a2", "train", "EnvA", 2),
    ("EnvB/b1", "train", "EnvB", 1),
    ("EnvB/b2", "train", "EnvB", 3),
    ("EnvC/c1", "train", "EnvC", 4),
    ("EnvC/c2", "train", "EnvC", 5),
    ("EnvVal/v1", "val", "EnvVal", 9),
    ("EnvTest/t1", "test", "EnvTest", 9),
)
TRAIN_VIDEOS = tuple(video_id for video_id, split, _env, _subject in VIDEO_SPEC if split == "train")
OFFICIAL_HELD_OUT_VIDEOS = tuple(video_id for video_id, split, _env, _subject in VIDEO_SPEC if split != "train")
# Deslocamento grande e exclusivo por ambiente: uma estatística que misture o
# ambiente retido fica visivelmente longe da média do treino de desenvolvimento.
ENVIRONMENT_OFFSET = {"EnvA": 0.0, "EnvB": 2.0, "EnvC": 500.0, "EnvVal": -900.0, "EnvTest": 900.0}


def _check(name: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return condition


def raises(error: type[Exception], action: Callable[[], object], fragment: str = "") -> bool:
    try:
        action()
    except error as exc:
        return fragment in str(exc)
    return False


def _labels(index: int) -> list[int]:
    shift = index * 3
    return [CYCLE[(frame + shift) % len(CYCLE)] for frame in range(N_FRAMES)]


def frames_table() -> pd.DataFrame:
    rows = [
        {
            "video_id": video_id,
            "split": split,
            "env": env,
            "subject": subject,
            "frame_index": frame,
            "label": label,
        }
        for index, (video_id, split, env, subject) in enumerate(VIDEO_SPEC)
        for frame, label in enumerate(_labels(index))
    ]
    return pd.DataFrame(rows)


def official_train_frames() -> pd.DataFrame:
    frames = frames_table()
    return cast(pd.DataFrame, frames[frames["split"] == "train"]).reset_index(drop=True)


def manifest_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": video_id, "split": split, "env": env, "cam": 1, "subject": subject,
             "width": 320, "height": 240, "fps": 25.0}
            for video_id, split, env, subject in VIDEO_SPEC
        ]
    )


def _environment(video_id: str) -> str:
    return video_id.partition("/")[0]


def _features(video_id: str, dim: int) -> np.ndarray:
    seed = sum(ord(char) for char in video_id) + dim
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(N_FRAMES, dim)) + ENVIRONMENT_OFFSET[_environment(video_id)]).astype(np.float32)


def pose_features(video_id: str) -> np.ndarray:
    return _features(video_id, POSE_FEATURE_DIM)


def visual_features(video_id: str) -> np.ndarray:
    return _features(video_id, DINOV3_FEATURE_DIM)


def quality_features(video_id: str) -> np.ndarray:
    rng = np.random.default_rng(sum(ord(char) for char in video_id))
    return rng.uniform(size=(N_FRAMES, 2)).astype(np.float32)


class RecordingLoaders:
    def __init__(self) -> None:
        self.calls: dict[str, list[str]] = {"pose": [], "visual": [], "quality": []}

    def loaders(self) -> FeatureLoaders:
        def record(name: str, loader: Callable[[str], np.ndarray]) -> Callable[[str], np.ndarray]:
            def wrapped(video_id: str) -> np.ndarray:
                self.calls[name].append(video_id)
                return loader(video_id)

            return wrapped

        return FeatureLoaders(
            pose=record("pose", pose_features),
            visual=record("visual", visual_features),
            quality=record("quality", quality_features),
        )

    def loaded_videos(self) -> set[str]:
        return {video_id for calls in self.calls.values() for video_id in calls}


QUALITY = QualityFeatures(root=Path("synthetic/quality"), sha256="0" * 64)


def write_inputs(root: Path) -> tuple[Path, Path]:
    frames_path = root / "processed" / "frames.parquet"
    manifest_path = root / "processed" / "manifest.parquet"
    frames_path.parent.mkdir(parents=True)
    frames_table().to_parquet(frames_path)
    manifest_table().to_parquet(manifest_path)
    return frames_path, manifest_path


def synthetic_expected_train() -> dict[str, object]:
    train = official_train_frames()
    return {
        "environments": ENVIRONMENTS,
        "n_videos": len(TRAIN_VIDEOS),
        "windows_total_eval_stride": len(build_window_index(train, stride=EVAL_STRIDE, drop_ignored=False)),
        "windows_labeled_eval_stride": len(build_window_index(train, stride=EVAL_STRIDE, drop_ignored=True)),
    }


@contextlib.contextmanager
def synthetic_workspace() -> Iterator[tuple[Path, Path, Path]]:
    with tempfile.TemporaryDirectory() as directory, patch.dict(
        internal_fold_feasibility.EXPECTED_OFFICIAL_TRAIN, synthetic_expected_train()
    ):
        root = Path(directory)
        frames_path, manifest_path = write_inputs(root)
        yield root / "folds", frames_path, manifest_path
