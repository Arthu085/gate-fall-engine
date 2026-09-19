"""Guarda de protocolo: as operações DINOv3 cobrem apenas o Le2i cs."""

from gatefall.datasets import DatasetAdapter

DINOV3_SUPPORTED_DATASET_IDENTIFIERS: tuple[str, ...] = ("le2i",)


def ensure_dinov3_dataset_supported(adapter: DatasetAdapter) -> None:
    if adapter.identifier not in DINOV3_SUPPORTED_DATASET_IDENTIFIERS:
        raise ValueError(
            f"dinov3 não suporta o dataset {adapter.identifier!r}: o protocolo "
            f"cv reutilizaria os .h5 de {adapter.dinov3_root}, gravados com "
            "env/split/subject do manifesto cs; opções disponíveis: "
            f"{', '.join(DINOV3_SUPPORTED_DATASET_IDENTIFIERS)}"
        )
