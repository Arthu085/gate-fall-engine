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
    subparsers.add_parser("selftest", help="Verifica a auditoria com um clone sintético")

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.data.gmdcsa24.selftests.audit import run_audit_selftest

        run_audit_selftest()
        return

    require_root(args.root)
    runners = {
        "verify": run_verify,
        "inventory": run_inventory,
        "annotations": run_annotations,
        "probe": run_probe,
    }
    if not runners[args.command](args.root):
        print(f"\n{args.command} FALHOU", file=sys.stderr)
        sys.exit(1)
    print(f"\n{args.command} OK")


if __name__ == "__main__":
    main()
