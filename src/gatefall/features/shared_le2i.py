"""Validação da origem comum dos artefatos de vídeo nos protocolos Le2i."""

from typing import cast

import pandas as pd

from gatefall.datasets import DatasetAdapter, get_dataset


def shared_source(adapter: DatasetAdapter) -> DatasetAdapter:
    if adapter.identifier != "le2i-cv":
        return adapter
    source = get_dataset("le2i")
    for name, left, right in (
        ("manifest", source.load_manifest(), adapter.load_manifest()),
        ("timegrid", source.load_frames(), adapter.load_frames()),
    ):
        if set(left.columns) != set(right.columns):
            raise ValueError(f"le2i-cv: colunas de {name} divergem do le2i compartilhado")
        keys = ["video_id"] if name == "manifest" else ["video_id", "frame_index"]
        columns = [column for column in left.columns if column != "split"]
        expected = cast(pd.DataFrame, left[columns]).sort_values(by=keys).reset_index(drop=True)
        actual = cast(pd.DataFrame, right[columns]).sort_values(by=keys).reset_index(drop=True)
        if not expected.equals(actual):
            raise ValueError(
                f"le2i-cv: {name} não cobre ou não alinha com os artefatos "
                "compartilhados de le2i; confira vídeos e grade temporal"
            )
    return source
