"""Constrói e verifica `data/manifest.parquet` para o Le2i.

Casa os paths de vídeo anotados pelo OmniFall (train/val/test do config
`le2i-cs`) com os arquivos locais extraídos de `data/raw/le2i/`, sonda cada
vídeo via ffprobe e grava um manifesto único (`data/manifest.parquet`) com
metadados de vídeo, split e status de extração de features por branch
(pose/DINOv3/SAM), ainda pendentes de execução.

Uso:
    uv run python -m gatefall.data.ingest ingest [--force]
    uv run python -m gatefall.data.ingest verify
"""

import argparse

from gatefall.data.le2i.manifest import ingest_le2i_dataset
from gatefall.data.le2i.verification import verify_le2i_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Constrói data/manifest.parquet"
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve o manifesto já existente em vez de pulá-lo",
    )

    subparsers.add_parser(
        "verify", help="Verifica a integridade do manifesto já construído"
    )

    args = parser.parse_args()
    if args.command == "ingest":
        ingest_le2i_dataset(args.force)
    elif args.command == "verify":
        verify_le2i_manifest()


if __name__ == "__main__":
    main()
