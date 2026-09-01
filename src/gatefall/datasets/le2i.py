"""Adapter mínimo do Le2i para as camadas genéricas do GateFall."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gatefall.data.frames import read_frames
from gatefall.data.manifest import read_manifest

LE2I_LABEL_NAMES = (
    "walk",
    "fall",
    "fallen",
    "sit_down",
    "sitting",
    "lie_down",
    "lying",
    "stand_up",
    "standing",
    "other",
)


@dataclass(frozen=True)
class Le2iDatasetAdapter:
    identifier: str = "le2i"
    raw_dir: Path = Path("data/raw/le2i")
    manifest_path: Path = Path("data/processed/le2i/manifest.parquet")
    frames_path: Path = Path("data/processed/le2i/frames.parquet")
    pose_root: Path = Path("data/features/le2i/pose")
    stats_path: Path = Path("src/gatefall/features/stats/pose_le2i_cs.json")
    label_names: tuple[str, ...] = LE2I_LABEL_NAMES
    feature_dim: int = 134

    def load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"manifesto não encontrado: {self.manifest_path}; rode "
                "`uv run python -m gatefall.data.ingest ingest` primeiro"
            )
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
        portable = Path(relative_path)
        if portable.is_absolute():
            raise ValueError(
                f"relative_path deve ser relativo, recebido: {relative_path!r}"
            )
        if ".." in portable.parts:
            raise ValueError(
                f"relative_path não pode conter '..': {relative_path!r}"
            )
        raw_root = self.raw_dir.resolve()
        resolved = (raw_root / portable).resolve()
        if resolved == raw_root or raw_root not in resolved.parents:
            raise ValueError(
                f"relative_path escapa de raw_dir: {relative_path!r}"
            )
        return resolved


LE2I_DATASET = Le2iDatasetAdapter()
