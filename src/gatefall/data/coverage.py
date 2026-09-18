"""CLI de auditoria de cobertura das anotações sobre o manifesto Le2i."""

import argparse
from typing import cast

from gatefall.data.le2i.coverage import audit_le2i_coverage
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.datasets.le2i import Le2iDatasetAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audita a cobertura dos segmentos anotados sobre a duração dos vídeos",
    )
    audit_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)

    args = parser.parse_args()
    if args.command == "audit":
        audit_le2i_coverage(adapter=cast(Le2iDatasetAdapter, get_dataset(args.dataset)))


if __name__ == "__main__":
    main()
