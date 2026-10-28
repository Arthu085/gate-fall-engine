"""CLI do estudo comparativo 2026; sem `run --execute`, nada é treinado, extraído ou avaliado."""

import argparse
import json
import sys
from pathlib import Path

from gatefall.study.plan import MATRICES, STAGES, build_plan, matrix_summary
from gatefall.study.protocol import FROZEN_PROTOCOL, protocol_sha256


def _matrices(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", choices=MATRICES, nargs="+", default=list(MATRICES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("protocol", help="Imprime o protocolo congelado e seu SHA-256")
    plan = commands.add_parser("plan", help="Contagem de passos por matriz e estado das saídas")
    _matrices(plan)
    plan.add_argument("--json", type=Path, help="Grava o plano completo neste arquivo (novo)")
    listing = commands.add_parser("commands", help="Comandos `uv run` exatos, agrupados por estágio")
    _matrices(listing)
    listing.add_argument("--stage", choices=STAGES)
    run = commands.add_parser("run", help="Executa o plano; padrão é --dry-run")
    _matrices(run)
    run.add_argument("--stage", choices=STAGES, nargs="+")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")
    run.add_argument("--allow-dirty", action="store_true")
    report = commands.add_parser("report", help="Agrega avaliações concluídas de uma matriz")
    report.add_argument("--matrix", choices=MATRICES, required=True)
    reuse = commands.add_parser("reuse-audit", help="Audita runs externos do experimento de movimento")
    reuse.add_argument("--source", type=Path, required=True,
                       help="Diretório equivalente a runs/local da máquina de origem")
    reuse.add_argument("--output", type=Path)
    features = commands.add_parser("features-report", help="Cobertura/alinhamento das features por vídeo")
    features.add_argument("--dataset", required=True, choices=("le2i", "le2i-cv", "upfall", "gmdcsa24"))
    profile = commands.add_parser("profile-extraction", help="Mede custo de extração em diretório temporário")
    profile.add_argument("--dataset", required=True, choices=("upfall", "gmdcsa24"))
    profile.add_argument("--videos", type=int, default=5)
    profile.add_argument("--execute", action="store_true")
    commands.add_parser("selftest", help="Contratos sintéticos em CPU")
    args = parser.parse_args()

    if args.command == "protocol":
        print(json.dumps({"sha256": protocol_sha256(), "protocol": FROZEN_PROTOCOL}, ensure_ascii=False, indent=2))
    elif args.command == "plan":
        steps = build_plan(tuple(args.matrix))
        from gatefall.study.registry import step_status

        status: dict[str, dict[str, int]] = {}
        for step in steps:
            bucket = status.setdefault(step.stage, {})
            state = step_status(step)
            bucket[state] = bucket.get(state, 0) + 1
        print(json.dumps({"protocol_sha256": protocol_sha256(), "counts": matrix_summary(steps),
                          "status": status}, ensure_ascii=False, indent=2))
        if args.json is not None:
            with args.json.open("x", encoding="utf-8") as handle:
                json.dump([{"key": s.key, "stage": s.stage, "command": list(s.command),
                            "outputs": [str(o) for o in s.outputs], "requires": list(s.requires),
                            "resources": s.resources, "matrices": sorted(s.matrices),
                            "identity_sha256": s.run.identity_sha256() if s.run else None} for s in steps],
                          handle, ensure_ascii=False, indent=2)
    elif args.command == "commands":
        from gatefall.study.registry import render_commands

        print(render_commands(build_plan(tuple(args.matrix)), args.stage))
    elif args.command == "run":
        from gatefall.study.registry import execute, render_commands

        full_plan = build_plan(tuple(args.matrix))
        steps = [step for step in full_plan if args.stage is None or step.stage in args.stage]
        if not args.execute:
            print(render_commands(steps))
            print(f"# dry-run: {len(steps)} passo(s); nada foi executado. Use --execute para rodar.")
            return
        print(f"log: {execute(steps, plan=full_plan, allow_dirty=args.allow_dirty)}")
    elif args.command == "report":
        from gatefall.study.report import build_report

        print(f"relatório: {build_report(build_plan(), args.matrix)}")
    elif args.command == "reuse-audit":
        from gatefall.study.reuse import audit_external_runs

        result = audit_external_runs(build_plan(("motion",)), args.source)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text + "\n")
        print(json.dumps({"counts": result["counts"], "planned": result["planned"]}, indent=2))
    elif args.command == "features-report":
        from gatefall.study.coverage import feature_coverage

        print(json.dumps(feature_coverage(args.dataset), ensure_ascii=False, indent=2))
    elif args.command == "profile-extraction":
        if not args.execute:
            print(f"# perfil não executado: rodaria pose/quality/dinov3/sam3 em {args.videos} vídeo(s) de "
                  f"teste de {args.dataset} em diretório temporário. Use --execute.")
            return
        from gatefall.study.profiling import profile_extraction

        print(f"perfil: {profile_extraction(args.dataset, args.videos)}")
    else:
        from gatefall.study.selftest import run_selftest

        run_selftest()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
