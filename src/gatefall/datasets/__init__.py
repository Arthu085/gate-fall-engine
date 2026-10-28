"""Seleção dos adapters de dataset suportados."""

from gatefall.datasets.base import DatasetAdapter

SUPPORTED_DATASET_IDENTIFIERS = ("le2i", "le2i-cv")
# Datasets do estudo comparativo 2026: os CLIs históricos continuam restritos
# ao Le2i; extratores e pacotes do estudo aceitam também os encenados.
STAGED_DATASET_IDENTIFIERS = ("upfall", "gmdcsa24")
STUDY_DATASET_IDENTIFIERS = SUPPORTED_DATASET_IDENTIFIERS + STAGED_DATASET_IDENTIFIERS


def get_dataset(identifier: str = "le2i") -> DatasetAdapter:
    if identifier not in STUDY_DATASET_IDENTIFIERS:
        raise ValueError(
            f"dataset não suportado: {identifier!r}; opções disponíveis: "
            f"{', '.join(STUDY_DATASET_IDENTIFIERS)}"
        )
    if identifier in STAGED_DATASET_IDENTIFIERS:
        from gatefall.datasets.staged import GMDCSA24_DATASET, UPFALL_DATASET

        return UPFALL_DATASET if identifier == "upfall" else GMDCSA24_DATASET

    from gatefall.datasets.le2i import LE2I_DATASET, Le2iDatasetAdapter

    if identifier == "le2i":
        return LE2I_DATASET
    return Le2iDatasetAdapter(protocol="cv")


__all__ = [
    "DatasetAdapter",
    "STAGED_DATASET_IDENTIFIERS",
    "STUDY_DATASET_IDENTIFIERS",
    "SUPPORTED_DATASET_IDENTIFIERS",
    "get_dataset",
]
