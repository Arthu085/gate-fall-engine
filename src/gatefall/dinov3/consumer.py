"""Validação dos arquivos DINOv3 compartilhados para consumidores CV."""

from typing import cast

import h5py
import pandas as pd

from gatefall.datasets import DatasetAdapter
from gatefall.dinov3 import storage
from gatefall.dinov3.report import find_provenance_divergences
from gatefall.dinov3.storage import dinov3_path
from gatefall.features.dinov3_standardization import FEATURE_DIM
from gatefall.features.shared_le2i import shared_source


def validate_dinov3_feature_set(adapter: DatasetAdapter) -> None:
    source = shared_source(adapter)
    if adapter.identifier != "le2i-cv":
        return
    frames = source.load_frames()
    groups = cast(pd.DataFrame, frames.groupby("video_id").agg(
        k=("frame_index", "size"), env=("env", "first"),
        subject=("subject", "first"), split=("split", "first"),
    ))
    attrs_by_video: dict[str, dict[str, object]] = {}
    for video_id, row in groups.iterrows():
        video_id = str(video_id)
        row = cast(pd.Series, row)
        expected_k = int(cast(int, row["k"]))
        path = dinov3_path(video_id, dinov3_root=adapter.dinov3_root)
        if not path.exists():
            raise FileNotFoundError(f"le2i-cv: feature DINOv3 ausente: {path}")
        reasons = storage.validate_existing_file(
            path, expected_k=expected_k, feature_dim=FEATURE_DIM,
            expected_attrs={
                "video_id": video_id, "K": expected_k,
                "env": str(row["env"]), "subject": int(cast(int, row["subject"])),
                "split": str(row["split"]),
            },
        )
        if reasons:
            raise ValueError(f"le2i-cv: {path} inválido: {'; '.join(reasons)}")
        with h5py.File(path, "r") as h5_file:
            attrs_by_video[video_id] = {
                name: h5_file.attrs[name]
                for name in storage.PROVENANCE_ATTR_NAMES if name in h5_file.attrs
            }
    divergences = find_provenance_divergences(attrs_by_video)
    if divergences:
        raise ValueError("le2i-cv: proveniência DINOv3 inválida: " + "; ".join(divergences))
