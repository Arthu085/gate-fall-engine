"""Auditoria opt-in, somente de metadados, do CMDFall no OmniFall: sem vídeo, pose nem treino."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd

from gatefall.config import TARGET_FPS, WINDOW_FRAMES
from gatefall.data.cmdfall.metadata import (
    BOUNDARY_TOLERANCE_S,
    annotate_segments,
    boundary_summary,
    cs_split_problems,
    grid_summary,
    inventory,
    label_summary,
    published_comparison,
    schema_problems,
    session_split_map,
    shared_units_across_splits,
    single_view,
    split_coverage,
    split_overview,
    taxonomy_problems,
    video_table,
    view_duplication,
    window_eligibility,
)
from gatefall.data.cmdfall.source import (
    CS_SPLIT_FILES,
    CV_SPLIT_FILES,
    DATASET_REPO_ID,
    DEFAULT_REPORT_PATH,
    DEFAULT_SOURCE_DIR,
    EXPECTED_SHA256,
    LABEL2ID_FILE,
    OMNIFALL_REVISION,
    SEGMENTS_FILE,
    fetch_sources,
    source_problems,
)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(cast(str, frame.reset_index().to_json(orient="records")))


def _print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(frame.to_string(float_format=lambda value: f"{value:.2f}"))


def _fail(title: str, problems: list[str]) -> bool:
    if not problems:
        return False
    print(f"\n=== {title}: {len(problems)} problema(s) ===", file=sys.stderr)
    for problem in problems:
        print(f"- {problem}", file=sys.stderr)
    return True


def build_report(source_dir: Path) -> tuple[dict[str, Any], list[str]]:
    raw = pd.read_csv(source_dir / SEGMENTS_FILE)
    label2id = pd.read_csv(source_dir / LABEL2ID_FILE)
    problems = schema_problems(raw) + taxonomy_problems(label2id)
    if problems:
        return {}, problems

    segments = annotate_segments(raw)
    videos = video_table(segments)
    sv = single_view(segments)
    duplication = view_duplication(segments)
    if duplication.divergent_sessions:
        problems.append(
            f"vistas divergentes em {len(duplication.divergent_sessions)} sessões "
            f"({duplication.divergent_sessions[:5]}): a contagem de uma vista por sessão não vale"
        )

    cs_splits = {name: pd.read_csv(source_dir / path) for name, path in CS_SPLIT_FILES.items()}
    cv_splits = {name: pd.read_csv(source_dir / path) for name, path in CV_SPLIT_FILES.items()}
    problems += [f"cs: {problem}" for problem in cs_split_problems(videos, cs_splits)]
    session_split = session_split_map(videos, cs_splits)

    report: dict[str, Any] = {
        "source": {
            "dataset_repo_id": DATASET_REPO_ID,
            "revision": OMNIFALL_REVISION,
            "sha256": EXPECTED_SHA256,
        },
        "contract": {
            "target_fps": TARGET_FPS,
            "window_frames": WINDOW_FRAMES,
            "boundary_tolerance_s": BOUNDARY_TOLERANCE_S,
        },
        "inventory": inventory(segments),
        "published_comparison": [
            {"field": name, "published": published, "observed": observed, "match": match}
            for name, published, observed, match in published_comparison(segments)
        ],
        "view_duplication": asdict(duplication),
        "labels": _records(label_summary(segments)),
        "boundaries": asdict(boundary_summary(sv)),
        "cs_overview": _records(split_overview(videos, cs_splits)),
        "cs_coverage": _records(split_coverage(sv, session_split)),
        "cs_grid": _records(grid_summary(sv, session_split)),
        "cs_windows": _records(window_eligibility(sv, session_split)),
        "cv_overview": _records(split_overview(videos, cv_splits)),
        "cv_shared_across_splits": shared_units_across_splits(videos, cv_splits),
    }
    return report, problems


def print_report(report: dict[str, Any]) -> None:
    print(f"fonte: {DATASET_REPO_ID}@{OMNIFALL_REVISION}")
    print("\n=== inventário (fatos do CSV) ===")
    for key, value in report["inventory"].items():
        print(f"{key}: {value}")
    _print_table(
        "CSV vs. estatísticas publicadas pelo OmniFall",
        pd.DataFrame(report["published_comparison"]).set_index("field"),
    )
    print("\n=== duplicação por vistas sincronizadas ===")
    for key, value in report["view_duplication"].items():
        print(f"{key}: {value}")
    _print_table(
        "rótulos (uma vista por sessão; mv_segments soma as vistas)",
        pd.DataFrame(report["labels"]).set_index("label"),
    )
    print("\n=== fronteiras entre segmentos consecutivos (uma vista por sessão) ===")
    for key, value in report["boundaries"].items():
        print(f"{key}: {value}")
    _print_table("splits cs", pd.DataFrame(report["cs_overview"]).set_index("split"))
    _print_table(
        "cobertura de classes por split cs (uma vista por sessão)",
        pd.DataFrame(report["cs_coverage"]).set_index(["split", "label"]),
    )
    _print_table(
        f"grade de {TARGET_FPS:g} fps sobre a extensão anotada",
        pd.DataFrame(report["cs_grid"]).set_index("split"),
    )
    _print_table(
        f"janelas causais potenciais de {WINDOW_FRAMES} quadros (uma vista por sessão)",
        pd.DataFrame(report["cs_windows"]).set_index(["stride", "split", "label"]),
    )
    _print_table("splits cv", pd.DataFrame(report["cv_overview"]).set_index("split"))
    print(f"\nunidades em mais de um split cv: {report['cv_shared_across_splits']}")


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    print(f"\n{path}: relatório não canônico gravado")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("fetch", "Baixa só os CSVs CMDFall da revisão fixada e confere o SHA-256"),
        ("report", "Confere o SHA-256 e relata os fatos verificáveis pelos CSVs"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
        if name == "report":
            subparser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    subparsers.add_parser("selftest", help="Verifica a auditoria com CSVs sintéticos")

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.data.cmdfall.selftests.audit import run_audit_selftest

        run_audit_selftest()
        return

    if args.command == "fetch":
        fetch_sources(args.source_dir)
    failed = _fail("integridade das fontes", source_problems(args.source_dir))
    if not failed and args.command == "report":
        report, problems = build_report(args.source_dir)
        if report:
            print_report(report)
            write_report(report, args.output)
        failed = _fail("metadados", problems)

    if failed:
        print(f"\n{args.command} FALHOU", file=sys.stderr)
        sys.exit(1)
    print(f"\n{args.command} OK")


if __name__ == "__main__":
    main()
