"""Extrai o FallDataset.zip do Le2i preservando a estrutura original."""

import argparse
from pathlib import Path

from gatefall.data.le2i.archive import DEFAULT_ARCHIVE_PATH, extract_le2i_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
        help="Caminho do FallDataset.zip a extrair",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove e re-extrai diretórios já existentes",
    )
    args = parser.parse_args()
    extract_le2i_archive(args.zip, args.force)


if __name__ == "__main__":
    main()
