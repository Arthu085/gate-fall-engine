"""Grade de reamostragem temporal e rótulo por quadro do Le2i."""

import argparse

from gatefall.data.le2i.frames import build_le2i_timegrid
from gatefall.data.le2i.timeline import report_le2i_timegrid
from gatefall.data.resampling_selftest import run_resampling_selftest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report", help="Relata a grade de reamostragem temporal do Le2i"
    )
    report_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    selftest_parser = subparsers.add_parser(
        "selftest", help="Verifica a grade de reamostragem contra entradas sintéticas"
    )
    selftest_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    build_parser = subparsers.add_parser(
        "build",
        help="Grava a grade de reamostragem temporal do Le2i em "
        "data/processed/le2i/frames.parquet",
    )
    build_parser.add_argument("--force", action="store_true")
    build_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    args = parser.parse_args()
    if args.command == "report":
        report_le2i_timegrid()
    elif args.command == "selftest":
        run_resampling_selftest()
    elif args.command == "build":
        build_le2i_timegrid(force=args.force)


if __name__ == "__main__":
    main()
