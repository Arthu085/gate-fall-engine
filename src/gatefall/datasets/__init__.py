"""Seleção dos adapters de dataset suportados."""

from gatefall.datasets.base import DatasetAdapter

SUPPORTED_DATASET_IDENTIFIERS = ("le2i", "le2i-cv")


def get_dataset(identifier: str = "le2i") -> DatasetAdapter:
    if identifier not in SUPPORTED_DATASET_IDENTIFIERS:
        raise ValueError(
            f"dataset não suportado: {identifier!r}; opções disponíveis: "
            f"{', '.join(SUPPORTED_DATASET_IDENTIFIERS)}"
        )

    from gatefall.datasets.le2i import LE2I_DATASET, Le2iDatasetAdapter

    if identifier == "le2i":
        return LE2I_DATASET
    return Le2iDatasetAdapter(protocol="cv")


__all__ = ["DatasetAdapter", "SUPPORTED_DATASET_IDENTIFIERS", "get_dataset"]
