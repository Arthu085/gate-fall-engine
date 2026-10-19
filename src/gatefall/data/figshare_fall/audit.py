"""Auditoria opt-in do VideoDataset.zip do Figshare 28596332, sem ingestão nem features."""

import argparse
import sys
import zipfile
from pathlib import Path

from gatefall.data.figshare_fall.inventory import (
    ARCHIVE_DOWNLOAD_URL,
    ARCHIVE_MD5,
    ARCHIVE_SIZE_BYTES,
    DEFAULT_ARCHIVE_PATH,
    FIGSHARE_DOI,
    compute_file_digests,
    read_inventory,
    report_actions,
    report_counts,
    report_duplicates,
    report_metadata,
    report_recording_anomalies,
    report_subjects_and_locations,
)
from gatefall.data.figshare_fall.probe import (
    probe_videos,
    report_probe,
    select_spread_sample,
)


def require_archive(path: Path) -> None:
    if path.exists():
        return
    print(
        f"erro: {path} não encontrado. Baixe manualmente {ARCHIVE_DOWNLOAD_URL} "
        f"(DOI {FIGSHARE_DOI}) e passe o caminho com --archive.",
        file=sys.stderr,
    )
    sys.exit(1)


def verify_archive(path: Path) -> bool:
    print(f"=== integridade de {path} ===")
    size, md5, sha256 = compute_file_digests(path)
    size_ok = size == ARCHIVE_SIZE_BYTES
    md5_ok = md5 == ARCHIVE_MD5
    print(f"tamanho: {size} (publicado {ARCHIVE_SIZE_BYTES}) {'OK' if size_ok else 'DIVERGENTE'}")
    print(f"md5: {md5} (publicado {ARCHIVE_MD5}) {'OK' if md5_ok else 'DIVERGENTE'}")
    print(f"sha256 local (registrar como evidência): {sha256}")
    return size_ok and md5_ok


def run_inventory(path: Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        inventory = read_inventory(archive)
        report_counts(inventory)
        report_recording_anomalies(inventory)
        report_duplicates(inventory)
        report_actions(inventory.recordings)
        report_subjects_and_locations(inventory.recordings)
        return report_metadata(archive, inventory.recordings)


def run_probe(path: Path, limit: int | None) -> bool:
    with zipfile.ZipFile(path) as archive:
        inventory = read_inventory(archive)
        members = select_spread_sample(list(inventory.videos["member"]), limit)
        return report_probe(probe_videos(archive, members))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("verify", "Confere tamanho e MD5 do zip contra o registro do Figshare"),
        ("inventory", "Relata contagens, sujeitos, locais, duplicatas e metadata.json"),
        ("probe", "Decodifica vídeos do zip e relata codec, FPS, duração e grade"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
        if name == "probe":
            subparser.add_argument(
                "--limit",
                type=int,
                default=None,
                help="Amostra determinística espalhada de N vídeos (padrão: todos)",
            )
    subparsers.add_parser("selftest", help="Verifica a auditoria com um zip sintético")

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.data.figshare_fall.selftests.audit import run_audit_selftest

        run_audit_selftest()
        return

    require_archive(args.archive)
    if args.command == "verify":
        ok = verify_archive(args.archive)
    elif args.command == "inventory":
        ok = run_inventory(args.archive)
    else:
        ok = run_probe(args.archive, args.limit)

    if not ok:
        print(f"\n{args.command} FALHOU", file=sys.stderr)
        sys.exit(1)
    print(f"\n{args.command} OK")


if __name__ == "__main__":
    main()
