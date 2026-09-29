"""Prepara e verifica um pacote portátil para extração Le2i em GPU remota."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PureWindowsPath

from gatefall.data.manifest import read_manifest
from gatefall.datasets.le2i import LE2I_DATASET, Le2iDatasetAdapter
from gatefall.hashing import sha256_file

MANIFEST_PATH = LE2I_DATASET.manifest_path
FRAMES_PATH = LE2I_DATASET.frames_path
RAW_PATH = LE2I_DATASET.raw_dir
RECEIPT_PATH = Path("bundle-sha256.json")


class BundleError(Exception):
    pass


def _canonical_files(root: Path) -> dict[str, Path]:
    files = {"manifest": root / MANIFEST_PATH, "frames": root / FRAMES_PATH}
    for path in files.values():
        if not path.is_file():
            raise BundleError(f"arquivo canônico ausente: {path}")
    return files


def _required_videos(root: Path) -> list[tuple[str, Path, str | None]]:
    manifest = read_manifest(root / MANIFEST_PATH)
    if manifest.empty or "relative_path" not in manifest:
        raise BundleError("manifesto vazio ou sem relative_path")
    adapter = Le2iDatasetAdapter(raw_dir=root / RAW_PATH)
    videos: dict[str, tuple[Path, str | None]] = {}
    for index, row in manifest.iterrows():
        relative_path = row["relative_path"]
        if not isinstance(relative_path, str) or not relative_path:
            raise BundleError(f"relative_path inválido na linha {index}: {relative_path!r}")
        if "\\" in relative_path or PureWindowsPath(relative_path).drive:
            raise BundleError(f"relative_path inválido na linha {index}: {relative_path!r}")
        try:
            path = adapter.resolve_video_path(relative_path)
        except ValueError as exc:
            raise BundleError(f"relative_path inválido na linha {index}: {exc}") from exc
        if not path.is_file():
            raise BundleError(f"vídeo referenciado ausente: {relative_path} ({path})")
        expected_hash = row.get("sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            expected_hash = None
        elif len(expected_hash) != 64 or any(
            char not in "0123456789abcdef" for char in expected_hash
        ):
            raise BundleError(f"sha256 inválido no manifesto: {relative_path}")
        if relative_path in videos and videos[relative_path] != (path, expected_hash):
            raise BundleError(f"entrada conflitante no manifesto: {relative_path}")
        videos[relative_path] = (path, expected_hash)
    return [(relative, *videos[relative]) for relative in sorted(videos)]


def _verify_videos(root: Path) -> int:
    videos = _required_videos(root)
    for relative_path, path, expected_hash in videos:
        if expected_hash is not None and sha256_file(path) != expected_hash:
            raise BundleError(f"sha256 divergente para vídeo: {relative_path}")
    return len(videos)


def verify_bundle(bundle: Path) -> int:
    root = bundle.resolve()
    receipt_path = root / RECEIPT_PATH
    if not receipt_path.is_file():
        raise BundleError(f"recibo de integridade ausente: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"recibo de integridade inválido: {receipt_path}") from exc
    if not isinstance(receipt, dict) or set(receipt) != {"manifest", "frames"}:
        raise BundleError(f"recibo de integridade inválido: {receipt_path}")
    for name, path in _canonical_files(root).items():
        if not isinstance(receipt[name], str) or sha256_file(path) != receipt[name]:
            raise BundleError(f"arquivo canônico alterado: {path}")
    return _verify_videos(root)


def prepare_bundle(source: Path, output: Path) -> int:
    source_root = source.resolve()
    target = output.absolute()
    if os.path.lexists(target):
        raise BundleError(f"destino já existe: {target}")
    files = _canonical_files(source_root)
    videos = _required_videos(source_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        receipt = {name: sha256_file(path) for name, path in files.items()}
        for name, source_path in files.items():
            relative = MANIFEST_PATH if name == "manifest" else FRAMES_PATH
            staged_path = stage / relative
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, staged_path)
        for relative_path, source_path, expected_hash in videos:
            staged_path = stage / RAW_PATH / relative_path
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, staged_path)
            source_hash = sha256_file(source_path)
            if expected_hash is not None and source_hash != expected_hash:
                raise BundleError(f"sha256 divergente para vídeo de origem: {relative_path}")
            if sha256_file(staged_path) != source_hash:
                raise BundleError(f"cópia divergente para vídeo: {relative_path}")
        (stage / RECEIPT_PATH).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        count = verify_bundle(stage)
        if os.path.lexists(target):
            raise BundleError(f"destino já existe: {target}")
        os.rename(stage, target)
        return count
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepara um pacote portátil Le2i CS")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--source-root", type=Path, default=Path("."))
    verify = commands.add_parser("verify", help="Verifica os arquivos do pacote")
    verify.add_argument("--bundle", type=Path, required=True)
    commands.add_parser("selftest", help="Executa casos sintéticos do pacote")
    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.data.le2i.selftests.bundle import run_bundle_selftest

        run_bundle_selftest()
        return
    try:
        if args.command == "prepare":
            count = prepare_bundle(args.source_root, args.output)
            print(f"pacote pronto: {args.output} ({count} vídeos)")
        else:
            count = verify_bundle(args.bundle)
            print(f"pacote íntegro: {args.bundle} ({count} vídeos)")
    except (BundleError, OSError) as exc:
        print(f"bundle FALHOU: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
