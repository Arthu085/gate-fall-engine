"""Grade de reamostragem temporal e rótulo por quadro do Le2i."""

import argparse
from typing import cast

from gatefall.data.le2i.frames import build_le2i_timegrid
from gatefall.data.le2i.timeline import report_le2i_timegrid
from gatefall.data.resampling_selftest import run_resampling_selftest
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.datasets.le2i import Le2iDatasetAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report", help="Relata a grade de reamostragem temporal do Le2i"
    )
    report_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)
    selftest_parser = subparsers.add_parser(
        "selftest", help="Verifica a grade de reamostragem contra entradas sintéticas"
    )
    selftest_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)
    build_parser = subparsers.add_parser(
        "build",
        help="Grava a grade de reamostragem temporal do Le2i em "
        "data/processed/le2i/frames.parquet",
    )
    build_parser.add_argument("--force", action="store_true")
    build_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)

    args = parser.parse_args()
    adapter = cast(Le2iDatasetAdapter, get_dataset(args.dataset))
    if args.command == "report":
        report_le2i_timegrid(adapter=adapter)
    elif args.command == "selftest":
        run_resampling_selftest()
    elif args.command == "build":
        build_le2i_timegrid(force=args.force, adapter=adapter)


if __name__ == "__main__":
    main()
