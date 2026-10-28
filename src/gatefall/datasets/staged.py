"""Adapters dos datasets encenados do estudo comparativo (UP-Fall e GMDCSA-24)."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.data.frames import read_frames
from gatefall.data.staged.build import load_sequence_index, load_staged_manifest, staged_paths
from gatefall.data.staged.sequences import ImageSequence, decode_sequence_frames
from gatefall.data.staged.specs import GMDCSA24_SPEC, UPFALL_SPEC, StagedDatasetSpec
from gatefall.datasets.le2i import LE2I_LABEL_NAMES


@dataclass(frozen=True)
class StagedVideoAdapter:
    spec: StagedDatasetSpec
    label_names: tuple[str, ...] = LE2I_LABEL_NAMES

    @property
    def identifier(self) -> str:
        return self.spec.identifier

    @property
    def raw_dir(self) -> Path:
        return self.spec.raw_dir

    @property
    def manifest_path(self) -> Path:
        return staged_paths(self.spec).manifest

    @property
    def frames_path(self) -> Path:
        return staged_paths(self.spec).frames

    @property
    def pose_root(self) -> Path:
        return self.spec.features_dir / "pose"

    @property
    def pose_stats_path(self) -> Path:
        # O estudo ajusta estatísticas por run, só no treino; este caminho
        # existe para o contrato e nunca é lido pelos pacotes do estudo.
        return self.spec.processed_dir / "stats" / "pose.json"

    @property
    def dinov3_root(self) -> Path:
        return self.spec.features_dir / "dinov3"

    @property
    def quality_root(self) -> Path:
        return self.spec.features_dir / "quality"

    @property
    def sam3_root(self) -> Path:
        return self.spec.features_dir / "sam3"

    def load_manifest(self) -> pd.DataFrame:
        return load_staged_manifest(self.spec)

    def load_frames(self) -> pd.DataFrame:
        if not self.frames_path.exists():
            raise FileNotFoundError(
                f"{self.frames_path}: ausente; rode `uv run python -m gatefall.data.staged "
                f"timegrid --dataset {self.identifier}`"
            )
        return read_frames(self.frames_path)

    def video_paths(self) -> dict[str, Path]:
        manifest = self.load_manifest()
        return {
            str(video_id): self.resolve_video_path(str(relative))
            for video_id, relative in zip(manifest["video_id"], manifest["relative_path"])
        }

    def resolve_video_path(self, relative_path: str) -> Path:
        portable = Path(relative_path)
        if portable.is_absolute() or ".." in portable.parts:
            raise ValueError(f"relative_path inválido: {relative_path!r}")
        root = self.raw_dir.resolve()
        resolved = (root / portable).resolve()
        if root not in resolved.parents:
            raise ValueError(f"relative_path escapa de raw_dir: {relative_path!r}")
        return resolved


@dataclass(frozen=True)
class StagedImageSequenceAdapter(StagedVideoAdapter):
    _sequences: dict[str, ImageSequence] = field(default_factory=dict, compare=False, repr=False)

    def decode_image_frames(self, video_id: str, src_indices: Sequence[int]) -> list[np.ndarray]:
        return decode_sequence_frames(self._sequence(video_id), src_indices)

    def _sequence(self, video_id: str) -> ImageSequence:
        if video_id not in self._sequences:
            index = load_sequence_index(self.spec)
            rows = cast(pd.DataFrame, index[index["video_id"] == video_id]).sort_values("src_index")
            if rows.empty:
                raise KeyError(f"{video_id}: ausente do índice de quadros")
            paths = self.video_paths()
            self._sequences[video_id] = ImageSequence(
                video_id=video_id,
                directory=paths[video_id],
                filenames=tuple(str(name) for name in rows["filename"]),
                timestamps_s=rows["timestamp_s"].to_numpy(dtype=np.float64),
                origin_iso=str(rows["origin_iso"].iloc[0]),
            )
        return self._sequences[video_id]


UPFALL_DATASET = StagedImageSequenceAdapter(UPFALL_SPEC)
GMDCSA24_DATASET = StagedVideoAdapter(GMDCSA24_SPEC)
