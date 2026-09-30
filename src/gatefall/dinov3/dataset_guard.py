"""Guardas de leitura e extração das features DINOv3 compartilhadas."""

from gatefall.datasets import DatasetAdapter

DINOV3_SUPPORTED_DATASET_IDENTIFIERS: tuple[str, ...] = ("le2i", "le2i-cv")
DINOV3_EXTRACTION_DATASET_IDENTIFIERS: tuple[str, ...] = ("le2i",)


def ensure_dinov3_dataset_supported(adapter: DatasetAdapter) -> None:
    if adapter.identifier not in DINOV3_SUPPORTED_DATASET_IDENTIFIERS:
        raise ValueError(
            f"dinov3 não suporta o dataset {adapter.identifier!r}; opções disponíveis: "
            f"{', '.join(DINOV3_SUPPORTED_DATASET_IDENTIFIERS)}"
        )


def ensure_dinov3_extraction_supported(adapter: DatasetAdapter) -> None:
    if adapter.identifier not in DINOV3_EXTRACTION_DATASET_IDENTIFIERS:
        raise ValueError(
            f"extração DINOv3 não suporta {adapter.identifier!r}: os .h5 em "
            f"{adapter.dinov3_root} são compartilhados e só podem ser gravados "
            "pelo protocolo le2i"
        )
