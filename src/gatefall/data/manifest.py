"""Contrato e persistência do manifesto de vídeos."""

import os
from pathlib import Path
from typing import cast

import pandas as pd

MANIFEST_DTYPES: dict[str, str] = {
    "video_id": "string",
    "dataset": "string",
    "relative_path": "string",
    "absolute_path": "string",
    "env": "string",
    "subject": "int64",
    "cam": "int64",
    "split": "string",
    "fps": "float64",
    "fps_source": "string",
    "n_frames_header": "Int64",
    "n_frames_counted": "int64",
    "duration_s": "float64",
    "width": "int64",
    "height": "int64",
    "codec": "string",
    "sha256": "string",
    "pose_status": "string",
    "dino_status": "string",
    "sam_status": "string",
}


def apply_manifest_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    typed = cast(pd.DataFrame, dataframe.astype(MANIFEST_DTYPES))
    return cast(pd.DataFrame, typed[list(MANIFEST_DTYPES)])


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_manifest(dataframe: pd.DataFrame, path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"skip {path} (já existe, use --force para sobrescrever)")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_parquet(tmp_path)
    os.replace(tmp_path, path)
    print(f"{path}: {len(dataframe)} linhas, colunas={list(dataframe.columns)}")
