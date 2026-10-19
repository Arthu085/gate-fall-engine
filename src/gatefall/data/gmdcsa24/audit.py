"""Auditoria opt-in do GMDCSA24 v2.1 e das anotações OmniFall, sem ingestão nem features."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.data.gmdcsa24.annotations import (
    compare_identities,
    load_omnifall_segments,
    load_omnifall_splits,
    read_original_annotations,
    report_cross_check,
    report_omnifall,
    report_original,
    with_label_names,
)
from gatefall.data.gmdcsa24.probe import (
    probe_checkout,
    report_grid,
    report_probe,
    summarize_grid,
)
from gatefall.data.gmdcsa24.provenance import (
    DEFAULT_REVIEW_DIR,
    build_review_plan,
    check_auto_evidence,
    check_review_target,
    read_review_sheet,
    report_plan,
    report_status,
    source_digests,
    summarize_review,
    write_review_sheet,
)
from gatefall.data.gmdcsa24.review_render import render_review
from gatefall.data.gmdcsa24.source import (
    DEFAULT_ROOT,
    RELEASE_TAG,
    REPOSITORY_URL,
    build_inventory,
    report_checkout,
    report_inventory,
)


def require_root(root: Path) -> None:
    if root.is_dir():
        return
    print(
        f"erro: {root} não encontrado. Clone manualmente com\n"
        f"  git clone --depth 1 --branch {RELEASE_TAG} {REPOSITORY_URL} {root}\n"
        "e passe o caminho com --root.",
        file=sys.stderr,
    )
    sys.exit(1)


def run_verify(root: Path) -> bool:
    try:
        return report_checkout(root)
    except subprocess.CalledProcessError as exc:
        print(f"{root} não é um clone Git legível: {exc.stderr.strip()}")
        return False


def run_inventory(
    root: Path, omnifall: pd.DataFrame | None = None, splits: pd.DataFrame | None = None
) -> bool:
    inventory = build_inventory(root)
    duplicates = report_inventory(root, inventory)
    original = read_original_annotations(root, inventory.annotation_files)
    omnifall = load_omnifall_segments() if omnifall is None else omnifall
    splits = load_omnifall_splits() if splits is None else splits

    print("\n=== identidade dos vídeos: clone x CSV original x OmniFall ===")
    named_sets = {
        "clone": set(inventory.videos["video_id"]),
        "CSV original": set(original.videos["video_id"]),
        "OmniFall labels": set(omnifall["video_id"]),
        **{
            f"OmniFall {protocol}": set(group["video_id"])
            for protocol, group in splits.groupby("protocol")
        },
    }
    for name, members in named_sets.items():
        print(f"{name}: {len(members)} vídeos")
    problems = compare_identities(named_sets)
    repeated = original.videos[original.videos["video_id"].duplicated(keep=False)]
    problems.extend(f"{identifier}: linha repetida no CSV" for identifier in repeated["video_id"])
    for problem in problems:
        print(f"  {problem}")
    if not problems:
        print("OK: mesmos vídeos em todas as fontes")
    return not problems and not duplicates and not inventory.unrecognized


def run_annotations(
    root: Path, omnifall: pd.DataFrame | None = None, splits: pd.DataFrame | None = None
) -> bool:
    inventory = build_inventory(root)
    original = read_original_annotations(root, inventory.annotation_files)
    named = with_label_names(load_omnifall_segments() if omnifall is None else omnifall)
    report_original(original)
    report_omnifall(named, load_omnifall_splits() if splits is None else splits)
    report_cross_check(original, named, inventory.videos)
    return not original.videos.empty


def run_probe(root: Path, omnifall: pd.DataFrame | None = None) -> bool:
    inventory = build_inventory(root)
    original = read_original_annotations(root, inventory.annotation_files)
    probes = probe_checkout(root, inventory.videos)
    declared = cast(pd.Series, original.videos.set_index("video_id")["length_s"])
    ok = report_probe(probes, declared)
    if ok:
        omnifall = load_omnifall_segments() if omnifall is None else omnifall
        categories = cast(pd.Series, inventory.videos.set_index("video_id")["category"])
        report_grid(summarize_grid(probes, omnifall), categories)
    return ok


def run_review_plan(
    root: Path, review_dir: Path, force: bool, omnifall: pd.DataFrame | None = None
) -> bool:
    try:
        check_review_target(review_dir, force)
    except (FileExistsError, ValueError) as exc:
        print(f"erro: {exc}")
        return False
    if not run_verify(root):
        print("a amostra de revisão só vale para o clone pinado e limpo")
        return False
    inventory = build_inventory(root)
    original = read_original_annotations(root, inventory.annotation_files)
    named = with_label_names(load_omnifall_segments() if omnifall is None else omnifall)
    probes = probe_checkout(root, inventory.videos)
    if bool(probes["error"].notna().any()):
        print("há vídeos que não decodificam; rode `probe` para os detalhes")
        return False
    plan = build_review_plan(root, inventory.videos, named, original.segments, probes)
    report_plan(plan)
    try:
        sheet = write_review_sheet(plan, review_dir, force)
    except (FileExistsError, ValueError) as exc:
        print(f"erro: {exc}")
        return False
    print(f"\nplanilha gravada em {sheet}; todos os casos começam `pending`")
    return not plan.empty


def load_review_sheet(review_dir: Path) -> pd.DataFrame | None:
    try:
        return read_review_sheet(review_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"erro: {exc}; rode `review-plan` antes")
        return None


def run_review_render(root: Path, review_dir: Path, omnifall: pd.DataFrame | None = None) -> bool:
    sheet = load_review_sheet(review_dir)
    if sheet is None:
        return False
    named = with_label_names(load_omnifall_segments() if omnifall is None else omnifall)
    try:
        rendered = render_review(root, review_dir, sheet, build_inventory(root).videos, named)
    except ValueError as exc:
        print(f"erro: {exc}")
        return False
    print(f"\n=== contact sheets em {review_dir} (evidência automática) ===")
    for identifier, count in rendered:
        print(f"  {identifier}: {count} quadros")
    return bool(rendered) and all(count > 0 for _, count in rendered)


def run_review_status(root: Path, review_dir: Path, omnifall: pd.DataFrame | None = None) -> bool:
    sheet = load_review_sheet(review_dir)
    if sheet is None:
        return False
    named = with_label_names(load_omnifall_segments() if omnifall is None else omnifall)
    digests = source_digests(root, sheet, build_inventory(root).videos)
    status = summarize_review(sheet, check_auto_evidence(sheet, named, digests))
    report_status(status, sheet, review_dir)
    return status.approved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("verify", f"Confere se o clone está no commit da tag {RELEASE_TAG} e sem alterações"),
        ("inventory", "Relata vídeos, CSVs, duplicatas e identidade contra o OmniFall"),
        ("annotations", "Relata classes, gaps e sobreposições do CSV original e do OmniFall"),
        ("probe", "Decodifica os vídeos e mede grade de 10 fps e janela causal de 24 quadros"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    for name, help_text in (
        ("review-plan", "Sorteia os casos de revisão visual e grava a planilha `pending`"),
        ("review-render", "Gera contact sheets locais por PTS em torno de cada fronteira"),
        ("review-status", "Separa evidência automática de revisão humana e decide a aprovação"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        subparser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
        if name == "review-plan":
            subparser.add_argument(
                "--force", action="store_true", help="Recria a planilha, apagando a revisão"
            )
    subparsers.add_parser("selftest", help="Verifica a auditoria com um clone sintético")

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.data.gmdcsa24.selftests.audit import run_audit_selftest

        run_audit_selftest()
        return

    require_root(args.root)
    runners = {
        "verify": lambda: run_verify(args.root),
        "inventory": lambda: run_inventory(args.root),
        "annotations": lambda: run_annotations(args.root),
        "probe": lambda: run_probe(args.root),
        "review-plan": lambda: run_review_plan(args.root, args.review_dir, args.force),
        "review-render": lambda: run_review_render(args.root, args.review_dir),
        "review-status": lambda: run_review_status(args.root, args.review_dir),
    }
    if not runners[args.command]():
        print(f"\n{args.command} FALHOU", file=sys.stderr)
        sys.exit(1)
    print(f"\n{args.command} OK")


if __name__ == "__main__":
    main()
