"""Anotações, manifesto, grade temporal e auditoria dos datasets encenados (UP-Fall, GMDCSA-24)."""

import argparse
import sys
from pathlib import Path

from gatefall.data.staged.specs import STAGED_SPECS, get_staged_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("fetch-labels", "Baixa as anotações OmniFall pinadas do dataset"),
        ("verify-labels", "Confere hashes e revisão das anotações locais"),
        ("manifest", "Constrói manifesto, identidades e proveniência a partir de data/raw"),
        ("timegrid", "Constrói a grade de 10 fps com rótulo por timestep"),
        ("audit", "Relata cobertura, vazamento, duplicatas e sincronização"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--dataset", required=True, choices=tuple(STAGED_SPECS))
        if name in ("fetch-labels", "manifest", "timegrid"):
            command.add_argument("--force", action="store_true")
    stage = commands.add_parser(
        "stage-archives", help="Extrai os .zip de câmera do UP-Fall no layout canônico"
    )
    stage.add_argument("--archives", type=Path, required=True)
    commands.add_parser("selftest", help="Contratos sintéticos sem dados reais")
    args = parser.parse_args()

    if args.command == "selftest":
        from gatefall.data.staged.selftest import run_selftest

        run_selftest()
        return
    if args.command == "stage-archives":
        from gatefall.data.staged.sequences import stage_upfall_archive

        spec = get_staged_spec("upfall")
        archives = sorted(args.archives.glob("*.zip"))
        if not archives:
            parser.error(f"nenhum .zip em {args.archives}")
        for archive in archives:
            print(stage_upfall_archive(archive, spec.raw_dir))
        return

    spec = get_staged_spec(args.dataset)
    if args.command == "fetch-labels":
        from gatefall.data.staged.labels import fetch_staged_labels

        fetch_staged_labels(spec, args.force)
    elif args.command == "verify-labels":
        from gatefall.data.staged.labels import verify_staged_labels

        problems = verify_staged_labels(spec)
        for problem in problems:
            print(problem, file=sys.stderr)
        print("verify-labels " + ("FALHOU" if problems else "OK"))
        raise SystemExit(1 if problems else 0)
    elif args.command == "manifest":
        from gatefall.data.staged.build import build_manifest

        build_manifest(spec, args.force)
    elif args.command == "timegrid":
        from gatefall.data.staged.build import build_timegrid

        build_timegrid(spec, args.force)
    elif args.command == "audit":
        from gatefall.data.staged.audit import run_audit

        report = run_audit(spec)
        for problem in report["problems"]:
            print(problem, file=sys.stderr)
        print(f"audit {'OK' if report['passed'] else 'FALHOU'}: {spec.processed_dir / 'coverage.json'}")
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
