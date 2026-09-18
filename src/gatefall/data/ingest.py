"""Constrói e verifica o manifesto processado de um dataset."""

import argparse
from typing import cast

from gatefall.data.le2i.manifest import ingest_le2i_dataset
from gatefall.data.le2i.verification import verify_le2i_manifest
from gatefall.data.le2i.verification_selftest import run_verification_selftest
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.datasets.le2i import Le2iDatasetAdapter


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
    ingest_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)

    verify_parser = subparsers.add_parser(
        "verify", help="Verifica a integridade do manifesto já construído"
    )
    verify_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)

    selftest_parser = subparsers.add_parser(
        "selftest", help="Verifica as estatísticas de duração de segmentos contra entradas sintéticas"
    )
    selftest_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)

    args = parser.parse_args()
    adapter = cast(Le2iDatasetAdapter, get_dataset(args.dataset))
    if args.command == "ingest":
        ingest_le2i_dataset(args.force, adapter=adapter)
    elif args.command == "verify":
        verify_le2i_manifest(adapter=adapter)
    elif args.command == "selftest":
        run_verification_selftest()


if __name__ == "__main__":
    main()
