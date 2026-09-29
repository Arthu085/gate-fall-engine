"""Dados e utilitários sintéticos compartilhados pelos selftests SAM 3."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import gatefall.sam3.extract as sam3_extract
from gatefall.data.frames import read_frames
from gatefall.data.manifest import read_manifest
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.sam3.runtime import Sam3Instance


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


@dataclass(frozen=True)
class _SyntheticDatasetAdapter:
    raw_dir: Path
    manifest_path: Path
    frames_path: Path
    sam3_root: Path
    pose_root: Path = Path("pose")
    pose_stats_path: Path = Path("pose_stats.json")
    dinov3_root: Path = Path("dinov3")
    quality_root: Path = Path("quality")
    identifier: str = "le2i"
    label_names: tuple[str, ...] = LE2I_LABEL_NAMES

    def load_manifest(self) -> pd.DataFrame:
        return read_manifest(self.manifest_path)

    def load_frames(self) -> pd.DataFrame:
        return read_frames(self.frames_path)

    def video_paths(self) -> dict[str, Path]:
        manifest = self.load_manifest()
        return {
            str(video_id): self.resolve_video_path(str(relative_path))
            for video_id, relative_path in zip(
                manifest["video_id"], manifest["relative_path"]
            )
        }

    def resolve_video_path(self, relative_path: str) -> Path:
        return self.raw_dir / relative_path


class _ScriptedSam3Segmenter:
    """Segmentador fake dirigido por script: um quadro por chamada, na ordem."""

    def __init__(self, script: list[list[Sam3Instance]]) -> None:
        self._script = script
        self._call_count = 0

    def segment_frame(self, frame_rgb: np.ndarray, text_prompt: str) -> list[Sam3Instance]:
        instances = self._script[self._call_count]
        self._call_count += 1
        return instances


class _ManifestSam3Segmenter(_ScriptedSam3Segmenter):
    """Fake que também expõe `runtime_manifest`, como o segmentador real."""

    def __init__(
        self, script: list[list[Sam3Instance]], runtime_manifest: dict[str, object]
    ) -> None:
        super().__init__(script)
        self._runtime_manifest = runtime_manifest

    @property
    def runtime_manifest(self) -> dict[str, object]:
        return self._runtime_manifest


def _rect_mask(
    x_min: int, y_min: int, x_max: int, y_max: int, *, height: int, width: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[y_min : y_max + 1, x_min : x_max + 1] = True
    return mask


def _build_fixture(
    root: Path, *, video_id: str, k: int, width: int = 100, height: int = 50
) -> _SyntheticDatasetAdapter:
    env, _, video_name = video_id.partition("/")
    raw_dir = root / "raw"
    manifest_path = root / "manifest.parquet"
    frames_path = root / "frames.parquet"
    sam3_root = root / "sam3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "video_id": [video_id] * k,
            "frame_index": list(range(k)),
            "src_index": list(range(k)),
        }
    ).to_parquet(frames_path)
    pd.DataFrame(
        {
            "video_id": [video_id],
            "relative_path": [f"{env}/{video_name}.avi"],
            "env": [env],
            "split": ["train"],
            "subject": [1],
            "fps": [10.0],
            "width": [width],
            "height": [height],
        }
    ).to_parquet(manifest_path)

    return _SyntheticDatasetAdapter(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        frames_path=frames_path,
        sam3_root=sam3_root,
    )


def _install_fake_decode_frames(
    frames_by_src: dict[int, np.ndarray]
) -> Callable[[], None]:
    original = sam3_extract.decode_frames

    def fake_decode_frames(video_path: Path, src_indices: list[int]) -> list[np.ndarray]:
        return [frames_by_src[i] for i in src_indices]

    sam3_extract.decode_frames = fake_decode_frames

    def _restore() -> None:
        sam3_extract.decode_frames = original

    return _restore
