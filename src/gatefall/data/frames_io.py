"""Decodificação de quadros de vídeo do Le2i a partir da grade de reamostragem."""

import argparse
from typing import cast

from gatefall.data.le2i.video_io import report_le2i_frame_decode
from gatefall.data.video_io_selftest import run_video_io_selftest
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.datasets.le2i import Le2iDatasetAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report", help="Relata a decodificação de quadros de uma amostra do Le2i"
    )
    report_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)
    selftest_parser = subparsers.add_parser(
        "selftest", help="Verifica a decodificação de vídeo contra entradas sintéticas"
    )
    selftest_parser.add_argument("--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS)

    args = parser.parse_args()
    if args.command == "report":
        report_le2i_frame_decode(adapter=cast(Le2iDatasetAdapter, get_dataset(args.dataset)))
    elif args.command == "selftest":
        run_video_io_selftest()


if __name__ == "__main__":
    main()
