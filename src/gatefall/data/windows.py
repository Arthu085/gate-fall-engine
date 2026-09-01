"""Janelamento deslizante sobre a grade de reamostragem temporal."""

import argparse

from gatefall.data.le2i.windows import report_le2i_windows
from gatefall.data.windowing_selftest import run_windowing_selftest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report", help="Relata as contagens reais de janelas do Le2i"
    )
    report_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    selftest_parser = subparsers.add_parser(
        "selftest", help="Verifica o janelamento contra entradas sintéticas"
    )
    selftest_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    args = parser.parse_args()
    if args.command == "report":
        report_le2i_windows()
    elif args.command == "selftest":
        run_windowing_selftest()


if __name__ == "__main__":
    main()
