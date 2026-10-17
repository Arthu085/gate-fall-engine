"""Investigação não canônica de R3D-18 congelado, restrita ao train oficial."""

import argparse
import json
from pathlib import Path

from gatefall.train.internal_folds.folds import FOLD_RUN_ROOT as CV_FOLD_ROOT
from gatefall.train.subject_folds.folds import SUBJECT_FOLD_ROOT as CS_FOLD_ROOT
from gatefall.train.temporal_rgb.contract import OUTPUT_ROOT, validate_output, validate_persistent_checkout
from gatefall.train.temporal_rgb.data import load_context
from gatefall.train.temporal_rgb.extraction import extract, preflight, smoke
from gatefall.train.temporal_rgb.report import write_report
from gatefall.train.temporal_rgb.training import baseline_control, run_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("selftest")
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--download-weights", action="store_true")
    extraction = commands.add_parser("extract")
    plan = commands.add_parser("plan")
    pilot = commands.add_parser("pilot")
    report = commands.add_parser("report")
    controls = commands.add_parser("verify-controls")
    for command in (smoke_parser, extraction, plan, pilot, report):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        command.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    for command in (extraction, plan, pilot, report, controls):
        command.add_argument("--protocol", choices=("cs", "cv"), required=True)
        command.add_argument("--fold-root", type=Path)
    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.train.temporal_rgb.selftests import run_selftests

        run_selftests()
        return
    if args.command == "verify-controls":
        fold_root = args.fold_root or (CS_FOLD_ROOT if args.protocol == "cs" else CV_FOLD_ROOT)
        context = load_context(args.protocol, fold_root)
        verified = [baseline_control(context, partition, fold_root) for partition in context.partitions]
        print(json.dumps({"protocol": args.protocol, "canonical": False, "controls": verified}, indent=2))
        return
    root = validate_output(args.output_root)
    if args.command == "smoke":
        print(json.dumps(smoke(root, args.device, args.download_weights), indent=2))
        return
    validate_persistent_checkout()
    fold_root = args.fold_root or (CS_FOLD_ROOT if args.protocol == "cs" else CV_FOLD_ROOT)
    context = load_context(args.protocol, fold_root)
    if args.command == "plan":
        print(json.dumps(preflight(context, root, args.device), indent=2))
    elif args.command == "extract":
        extract(context, root, args.device)
    elif args.command == "pilot":
        for partition in context.partitions:
            for mode in ("dynamic", "static"):
                result = run_probe(context, partition, fold_root, root, mode, args.device)
                print(f"{partition.fold}/{mode}: {result['restricted_macro_f1']:.6f}", flush=True)
        report = write_report(context, fold_root, root, args.device)
        print(json.dumps(report["pilot_decision"], indent=2))
    else:
        report = write_report(context, fold_root, root, args.device)
        print(json.dumps(report["pilot_decision"], indent=2))
