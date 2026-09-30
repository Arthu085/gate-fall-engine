"""Guardas de leitura e extração das features SAM 3 compartilhadas."""

from gatefall.datasets import DatasetAdapter

SAM3_SUPPORTED_DATASET_IDENTIFIERS: tuple[str, ...] = ("le2i", "le2i-cv")
SAM3_EXTRACTION_DATASET_IDENTIFIERS: tuple[str, ...] = ("le2i",)


def ensure_sam3_dataset_supported(adapter: DatasetAdapter) -> None:
    if adapter.identifier not in SAM3_SUPPORTED_DATASET_IDENTIFIERS:
        raise ValueError(
            f"sam3 não suporta o dataset {adapter.identifier!r}; opções disponíveis: "
            f"{', '.join(SAM3_SUPPORTED_DATASET_IDENTIFIERS)}"
        )


def ensure_sam3_extraction_supported(adapter: DatasetAdapter) -> None:
    if adapter.identifier not in SAM3_EXTRACTION_DATASET_IDENTIFIERS:
        raise ValueError(
            f"extração SAM 3 não suporta {adapter.identifier!r}: os .h5 em "
            f"{adapter.sam3_root} são compartilhados e só podem ser gravados "
            "pelo protocolo le2i"
        )
