"""Extração da distribuição original do dataset Le2i."""

import shutil
import sys
import zipfile
from pathlib import Path

from gatefall.hashing import sha256_file

DEFAULT_ARCHIVE_PATH = Path("data/raw/le2i/FallDataset.zip")

EXPECTED_DIRECTORIES = (
    "Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02",
    "Lecture room", "Office",
)


def format_byte_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def calculate_directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def extract_dataset_readme(
    archive: zipfile.ZipFile, destination: Path, force: bool
) -> None:
    readme_path = destination / "README.txt"
    if readme_path.exists() and not force:
        return
    if "README.txt" not in archive.namelist():
        return
    with archive.open("README.txt") as source, readme_path.open("wb") as target:
        shutil.copyfileobj(source, target)


def extract_nested_archive(
    archive: zipfile.ZipFile,
    member_name: str,
    destination: Path,
    force: bool,
) -> None:
    with archive.open(member_name) as stream:
        nested_archive = zipfile.ZipFile(stream)
        member_names = nested_archive.namelist()
        if not member_names:
            print(
                f"aviso: {member_name} não contém nenhum arquivo, ignorando",
                file=sys.stderr,
            )
            return
        top_directory = member_names[0].split("/")[0]
        target = destination / top_directory
        if target.is_dir():
            if not force:
                print(
                    f"{top_directory}: já extraído, pulando "
                    "(use --force para sobrescrever)"
                )
                return
            shutil.rmtree(target)
        total_size = sum(info.file_size for info in nested_archive.infolist())
        nested_archive.extractall(destination)
    print(
        f"extraído {top_directory}: {len(member_names)} arquivos, "
        f"{format_byte_size(total_size)}"
    )


def print_directory_tree(root: Path, max_depth: int = 2) -> None:
    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for entry in sorted(path.iterdir()):
            print("  " * depth + entry.name)
            if entry.is_dir():
                walk(entry, depth + 1)

    walk(root, 0)


def extract_le2i_archive(archive_path: Path, force: bool) -> None:
    if not archive_path.exists():
        destination = archive_path.parent
        already_extracted = all(
            (destination / name).is_dir() for name in EXPECTED_DIRECTORIES
        )
        if not force and already_extracted:
            print(
                f"{archive_path} não encontrado, mas os {len(EXPECTED_DIRECTORIES)} "
                f"diretórios esperados já estão extraídos em {destination}, nada a fazer"
            )
            return
        print(f"erro: arquivo não encontrado: {archive_path}", file=sys.stderr)
        sys.exit(1)

    destination = archive_path.parent

    print(f"sha256({archive_path}) = {sha256_file(archive_path)}")

    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile:
        print(
            f"erro: {archive_path} não é um arquivo zip válido",
            file=sys.stderr,
        )
        sys.exit(1)

    with archive:
        extract_dataset_readme(archive, destination, force)
        for member_name in archive.namelist():
            if "/" in member_name or not member_name.lower().endswith(".zip"):
                continue
            extract_nested_archive(archive, member_name, destination, force)

    print_directory_tree(destination, max_depth=2)
    print(
        f"tamanho total em disco: "
        f"{format_byte_size(calculate_directory_size(destination))}"
    )
