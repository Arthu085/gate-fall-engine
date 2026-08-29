"""Dataset de janelas de pose para treino/avaliação (`PoseWindowDataset`)."""

import argparse

from gatefall.data.le2i.pose_dataset import report_pose_dataset
from gatefall.data.le2i.pose_dataset_selftest import run_pose_dataset_selftest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "report", help="Relata as contagens reais de janelas do PoseWindowDataset"
    )
    subparsers.add_parser(
        "selftest", help="Verifica o PoseWindowDataset contra entradas sintéticas"
    )

    args = parser.parse_args()
    if args.command == "report":
        report_pose_dataset()
    elif args.command == "selftest":
        run_pose_dataset_selftest()


if __name__ == "__main__":
    main()
