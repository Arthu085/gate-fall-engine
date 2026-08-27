"""Contrato e persistência da tabela de quadros da grade de reamostragem temporal."""

import os
from pathlib import Path
from typing import cast

import pandas as pd

FRAME_DTYPES: dict[str, str] = {
    "video_id": "string",
    "split": "string",
    "env": "string",
    "subject": "int64",
    "frame_index": "int32",
    "time_s": "float64",
    "src_index": "int32",
    "label": "int8",
    "gap_position": "string",
}


def apply_frame_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    typed = cast(pd.DataFrame, dataframe.astype(FRAME_DTYPES))
    return cast(pd.DataFrame, typed[list(FRAME_DTYPES)])


def read_frames(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_frames(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    dataframe.to_parquet(tmp_path)
    os.replace(tmp_path, path)
