"""Decodificação de quadros de vídeo do Le2i a partir da grade de reamostragem."""

import argparse

from gatefall.data.le2i.video_io import report_le2i_frame_decode
from gatefall.data.video_io_selftest import run_video_io_selftest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "report", help="Relata a decodificação de quadros de uma amostra do Le2i"
    )
    subparsers.add_parser(
        "selftest", help="Verifica a decodificação de vídeo contra entradas sintéticas"
    )

    args = parser.parse_args()
    if args.command == "report":
        report_le2i_frame_decode()
    elif args.command == "selftest":
        run_video_io_selftest()


if __name__ == "__main__":
    main()
