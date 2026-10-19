"""Adapter isolado do GMDCSA24 para o piloto com rótulos fracos: build, validate e selftest.

Grava só em `data/processed/gmdcsa24/`. Não extrai features, não ajusta
padronização e não treina; essas etapas dependem da autorização separada do
passo 4 do protocolo.
"""

import argparse
import sys
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.data.gmdcsa24.annotations import (
    load_omnifall_segments,
    load_omnifall_splits,
    read_original_annotations,
)
from gatefall.data.gmdcsa24.audit import require_root, run_inventory, run_verify
from gatefall.data.gmdcsa24.provenance import DEFAULT_REVIEW_DIR, REVIEW_SHEET
from gatefall.data.gmdcsa24.source import DEFAULT_ROOT, build_inventory, read_checkout_state
from gatefall.data.gmdcsa24.weak_label import (
    DEFAULT_OUTPUT_DIR,
    MANIFEST_FILE,
    Le2iReference,
    build_tables,
    load_le2i_reference,
    optional_sha256,
    output_location_problems,
    probe_sources,
    protocol_problems,
    report_summary,
    report_validation,
    validate_outputs,
    write_outputs,
)
from gatefall.datasets.le2i import LE2I_DATASET
from gatefall.hashing import sha256_dataframe
from gatefall.runs import REPOSITORY_ROOT


def load_reference() -> Le2iReference | None:
    try:
        return load_le2i_reference(LE2I_DATASET)
    except FileNotFoundError as exc:
        print(f"erro: Le2i-CS canônico indisponível ({exc}); rode o pipeline do Le2i antes")
        return None


def omnifall_content_sha256(omnifall: pd.DataFrame) -> str:
    ordered = omnifall.sort_values(["video_id", "start", "end", "label"], kind="stable")
    return sha256_dataframe(ordered.reset_index(drop=True))


def run_build(root: Path, output_dir: Path, force: bool) -> bool:
    problems = protocol_problems() + output_location_problems(output_dir)
    if (output_dir / MANIFEST_FILE).exists() and not force:
        problems.append(f"{output_dir / MANIFEST_FILE} já existe; use --force para recriar")
    for problem in problems:
        print(f"erro: {problem}")
    le2i = load_reference()
    if problems or le2i is None:
        return False

    review_sheet = DEFAULT_REVIEW_DIR / REVIEW_SHEET
    review_before = optional_sha256(review_sheet)
    omnifall = load_omnifall_segments()
    if not run_verify(root) or not run_inventory(root, omnifall, load_omnifall_splits()):
        print("o adapter só roda sobre o clone pinado, limpo e com os 160 vídeos em comum")
        return False
    inventory = build_inventory(root)
    original = read_original_annotations(root, inventory.annotation_files)
    print("\nsondando, calculando SHA-256 e decodificando os vídeos (passthrough)...")
    sources = probe_sources(root, inventory.videos)
    failed = cast(pd.DataFrame, sources[sources["error"].map(lambda error: isinstance(error, str))])
    for identifier, error in zip(failed["video_id"], failed["error"]):
        print(f"  FALHA {identifier}: {error}")
    if not failed.empty:
        return False

    tables = build_tables(sources, omnifall, original.segments)
    code_commit, code_status = read_checkout_state(REPOSITORY_ROOT)
    provenance: dict[str, object] = {
        "source": {"root": root.as_posix()},
        "labels": {
            "segments": len(omnifall),
            "content_sha256": omnifall_content_sha256(omnifall),
        },
        "code": {"commit": code_commit, "dirty": bool(code_status)},
        "le2i_cs": {
            "frames_path": LE2I_DATASET.frames_path.as_posix(),
            "train_windows": le2i.train_windows,
            "eval_index_sha256": le2i.eval_index_sha256,
        },
        "review_sheet": {
            "path": review_sheet.as_posix(),
            "sha256_before": review_before,
            "sha256_after": optional_sha256(review_sheet),
        },
    }
    try:
        manifest_path = write_outputs(output_dir, tables, provenance, force)
    except (FileExistsError, ValueError) as exc:
        print(f"erro: {exc}")
        return False
    print(f"\nsaída gravada em {output_dir}; manifesto {manifest_path}")
    report_summary(tables.summary)
    return report_validation(validate_outputs(output_dir, le2i))


def run_validate(output_dir: Path) -> bool:
    le2i = load_reference()
    return report_validation(validate_outputs(output_dir, le2i))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser(
        "build", help="Monta T e S do clone pinado e do OmniFall pinado e valida a saída"
    )
    build_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--force", action="store_true", help="Recria uma saída existente")
    validate_parser = subparsers.add_parser(
        "validate", help="Confere W1, W2, V2, V5 e a integridade de uma saída gravada"
    )
    validate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    subparsers.add_parser("selftest", help="Verifica o adapter com vídeos e rótulos sintéticos")

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.data.gmdcsa24.selftests.adapter import run_adapter_selftest

        run_adapter_selftest()
        return
    if args.command == "build":
        require_root(args.root)
        ok = run_build(args.root, args.output_dir, args.force)
    else:
        ok = run_validate(args.output_dir)
    if not ok:
        print(f"\n{args.command} FALHOU", file=sys.stderr)
        sys.exit(1)
    print(f"\n{args.command} OK")


if __name__ == "__main__":
    main()
