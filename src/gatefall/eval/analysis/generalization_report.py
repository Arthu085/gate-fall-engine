"""CLI do relatório auditável de fatores de domínio por split (Le2i-CS/Le2i-CV)."""

import argparse
import json
import os
from pathlib import Path
from typing import cast

from gatefall.data.le2i.annotations import load_annotation_splits
from gatefall.data.le2i.generalization import build_generalization_report
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.datasets.le2i import Le2iDatasetAdapter


def run_report(dataset_name: str, output: Path) -> None:
    adapter = cast(Le2iDatasetAdapter, get_dataset(dataset_name))
    manifest = adapter.load_manifest()
    frames = adapter.load_frames()
    splits = load_annotation_splits(protocol=adapter.protocol)
    result = build_generalization_report(
        manifest,
        frames,
        splits,
        adapter.pose_root,
        identifier=adapter.identifier,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output)
    print(f"{output}: relatório de generalização gravado")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report",
        help="Gera o relatório de fatores de domínio a partir do dataset real",
    )
    report_parser.add_argument(
        "--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS
    )
    report_parser.add_argument("--output", type=Path, required=True)
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas do relatório de generalização"
    )

    args = parser.parse_args()
    if args.command == "report":
        run_report(args.dataset, args.output)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.generalization_report import run_generalization_selftest

        run_generalization_selftest()


if __name__ == "__main__":
    main()
