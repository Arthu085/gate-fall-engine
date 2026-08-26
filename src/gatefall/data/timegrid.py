"""Grade de reamostragem temporal e rótulo por quadro do Le2i."""

import argparse

from gatefall.data.le2i.timeline import report_le2i_timegrid
from gatefall.data.resampling_selftest import run_resampling_selftest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "report", help="Relata a grade de reamostragem temporal do Le2i"
    )
    subparsers.add_parser(
        "selftest", help="Verifica a grade de reamostragem contra entradas sintéticas"
    )

    args = parser.parse_args()
    if args.command == "report":
        report_le2i_timegrid()
    elif args.command == "selftest":
        run_resampling_selftest()


if __name__ == "__main__":
    main()
