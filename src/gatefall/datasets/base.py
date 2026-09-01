"""Contrato mínimo entre datasets e as camadas genéricas do GateFall."""

from pathlib import Path
from typing import Protocol

import pandas as pd


class DatasetAdapter(Protocol):
    @property
    def identifier(self) -> str: ...

    @property
    def raw_dir(self) -> Path: ...

    @property
    def manifest_path(self) -> Path: ...

    @property
    def frames_path(self) -> Path: ...

    @property
    def pose_root(self) -> Path: ...

    @property
    def stats_path(self) -> Path: ...

    @property
    def label_names(self) -> tuple[str, ...]: ...

    @property
    def feature_dim(self) -> int: ...

    def load_manifest(self) -> pd.DataFrame: ...

    def load_frames(self) -> pd.DataFrame: ...

    def video_paths(self) -> dict[str, Path]: ...

    def resolve_video_path(self, relative_path: str) -> Path: ...
