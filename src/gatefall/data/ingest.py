"""Constrói e verifica o manifesto processado de um dataset."""

import argparse

from gatefall.data.le2i.manifest import ingest_le2i_dataset
from gatefall.data.le2i.verification import verify_le2i_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Constrói data/processed/le2i/manifest.parquet"
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve o manifesto já existente em vez de pulá-lo",
    )
    ingest_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    verify_parser = subparsers.add_parser(
        "verify", help="Verifica a integridade do manifesto já construído"
    )
    verify_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    args = parser.parse_args()
    if args.command == "ingest":
        ingest_le2i_dataset(args.force)
    elif args.command == "verify":
        verify_le2i_manifest()


if __name__ == "__main__":
    main()
