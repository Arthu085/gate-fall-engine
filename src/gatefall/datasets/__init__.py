"""Seleção dos adapters de dataset suportados."""

from gatefall.datasets.base import DatasetAdapter


def get_dataset(identifier: str = "le2i") -> DatasetAdapter:
    if identifier != "le2i":
        raise ValueError(
            f"dataset não suportado: {identifier!r}; opções disponíveis: le2i"
        )

    from gatefall.datasets.le2i import LE2I_DATASET

    return LE2I_DATASET


__all__ = ["DatasetAdapter", "get_dataset"]
