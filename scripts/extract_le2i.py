"""Extrai o FallDataset.zip do Le2i preservando a estrutura original."""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from _common import sha256_file

DEFAULT_ZIP = Path("data/raw/le2i/FallDataset.zip")


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _extract_readme(zf: zipfile.ZipFile, dest: Path, force: bool) -> None:
    readme_path = dest / "README.txt"
    if readme_path.exists() and not force:
        return
    if "README.txt" not in zf.namelist():
        return
    with zf.open("README.txt") as src, readme_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def _extract_nested(zf: zipfile.ZipFile, member_name: str, dest: Path, force: bool) -> None:
    with zf.open(member_name) as stream:
        nested = zipfile.ZipFile(stream)
        names = nested.namelist()
        if not names:
            print(
                f"aviso: {member_name} não contém nenhum arquivo, ignorando",
                file=sys.stderr,
            )
            return
        top_dir = names[0].split("/")[0]
        target = dest / top_dir
        if target.is_dir():
            if not force:
                print(f"{top_dir}: já extraído, pulando (use --force para sobrescrever)")
                return
            shutil.rmtree(target)
        total_size = sum(info.file_size for info in nested.infolist())
        nested.extractall(dest)
    print(f"extraído {top_dir}: {len(names)} arquivos, {_human_size(total_size)}")


def _print_tree(root: Path, max_depth: int = 2) -> None:
    def _walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for entry in sorted(path.iterdir()):
            print("  " * depth + entry.name)
            if entry.is_dir():
                _walk(entry, depth + 1)

    _walk(root, 0)


def extract(zip_path: Path, force: bool) -> None:
    if not zip_path.exists():
        print(f"erro: arquivo não encontrado: {zip_path}", file=sys.stderr)
        sys.exit(1)

    print(f"sha256({zip_path}) = {sha256_file(zip_path)}")

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        print(f"erro: {zip_path} não é um arquivo zip válido", file=sys.stderr)
        sys.exit(1)

    dest = zip_path.parent
    with zf:
        _extract_readme(zf, dest, force)
        for member_name in zf.namelist():
            if "/" in member_name or not member_name.lower().endswith(".zip"):
                continue
            _extract_nested(zf, member_name, dest, force)

    _print_tree(dest, max_depth=2)
    print(f"tamanho total em disco: {_human_size(_dir_size(dest))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=DEFAULT_ZIP,
        help="Caminho do FallDataset.zip a extrair",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove e re-extrai diretórios já existentes",
    )
    args = parser.parse_args()
    extract(args.zip, args.force)


if __name__ == "__main__":
    main()
