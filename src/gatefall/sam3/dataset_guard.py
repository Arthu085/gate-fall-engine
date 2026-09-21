"""Guarda de protocolo: as operações SAM 3 cobrem apenas o Le2i cs."""

from gatefall.datasets import DatasetAdapter

SAM3_SUPPORTED_DATASET_IDENTIFIERS: tuple[str, ...] = ("le2i",)


def ensure_sam3_dataset_supported(adapter: DatasetAdapter) -> None:
    if adapter.identifier not in SAM3_SUPPORTED_DATASET_IDENTIFIERS:
        raise ValueError(
            f"sam3 não suporta o dataset {adapter.identifier!r}: o protocolo "
            f"cv reutilizaria os .h5 de {adapter.sam3_root}, gravados com "
            "env/split/subject do manifesto cs; opções disponíveis: "
            f"{', '.join(SAM3_SUPPORTED_DATASET_IDENTIFIERS)}"
        )
