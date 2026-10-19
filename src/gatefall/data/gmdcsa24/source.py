"""Fonte pinada e inventário do clone local do GMDCSA24 (Alam et al., 2024).

O dataset é distribuído como repositório Git, sem arquivo único com checksum
publicado no GitHub. A auditoria lê um clone local fixado na tag `v2.1` e
nunca grava em `data/`.
"""

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

REPOSITORY = "ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}.git"
RELEASE_TAG = "v2.1"
RELEASE_COMMIT = "5abac7693229900cf80f722e878fbb119211fc1c"
ZENODO_DOI = "10.5281/zenodo.13354453"
DEFAULT_ROOT = Path("data/raw/gmdcsa24/GMDCSA24")
OMNIFALL_DATASET_NAME = "GMDCSA24"

VIDEO_PATTERN = re.compile(
    r"^Subject (?P<subject>\d+)/(?P<category>ADL|Fall)/(?P<stem>[^/]+)\.mp4$"
)
ANNOTATION_PATTERN = re.compile(r"^Subject (?P<subject>\d+)/(?P<category>ADL|Fall)\.csv$")
REPOSITORY_FILES = frozenset({"README.md", "LICENSE"})

VIDEO_COLUMNS = ["video_id", "subject", "category", "file_name", "relative_path", "size"]


@dataclass(frozen=True)
class CheckoutInventory:
    videos: pd.DataFrame
    annotation_files: list[str]
    unrecognized: list[str]


def video_id(subject: int, category: str, stem: str) -> str:
    """Chave de vídeo no formato de `path` do OmniFall, ex. `Subject_1/ADL/01`."""
    return f"Subject_{subject}/{category}/{stem}"


def read_checkout_state(root: Path) -> tuple[str, list[str]]:
    """Devolve o commit de HEAD e as linhas de `git status --porcelain` do clone."""
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return head, status


def build_inventory(root: Path) -> CheckoutInventory:
    video_rows: list[dict[str, object]] = []
    annotation_files: list[str] = []
    unrecognized: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == ".git" or relative.startswith(".git/") or path.is_dir():
            continue
        video_match = VIDEO_PATTERN.match(relative)
        if video_match is not None:
            subject = int(video_match["subject"])
            video_rows.append(
                {
                    "video_id": video_id(
                        subject, video_match["category"], video_match["stem"]
                    ),
                    "subject": subject,
                    "category": video_match["category"],
                    "file_name": path.name,
                    "relative_path": relative,
                    "size": path.stat().st_size,
                }
            )
        elif ANNOTATION_PATTERN.match(relative):
            annotation_files.append(relative)
        elif relative not in REPOSITORY_FILES:
            unrecognized.append(relative)

    videos = pd.DataFrame(video_rows, columns=VIDEO_COLUMNS)
    return CheckoutInventory(
        videos=cast(pd.DataFrame, videos.sort_values("video_id", ignore_index=True)),
        annotation_files=annotation_files,
        unrecognized=unrecognized,
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_videos(root: Path, videos: pd.DataFrame) -> list[list[str]]:
    """Agrupa vídeos com SHA-256 idêntico.

    Tamanho diferente já prova conteúdo diferente, então só os vídeos que
    compartilham tamanho são lidos."""
    same_size = videos[videos.duplicated("size", keep=False)]
    digests = pd.Series(
        [sha256_of(root / path) for path in same_size["relative_path"]],
        index=same_size["video_id"],
        dtype=object,
    )
    groups = digests.groupby(digests).apply(lambda group: sorted(group.index))
    return sorted(
        cast(list[str], members) for members in groups if len(members) > 1
    )


def report_checkout(root: Path) -> bool:
    print(f"=== clone {root} ===")
    head, status = read_checkout_state(root)
    commit_ok = head == RELEASE_COMMIT
    print(
        f"HEAD: {head} (tag {RELEASE_TAG} = {RELEASE_COMMIT}) "
        f"{'OK' if commit_ok else 'DIVERGENTE'}"
    )
    print(f"arquivos modificados ou não rastreados: {len(status)}")
    for line in status:
        print(f"  {line}")
    return commit_ok and not status


def report_inventory(root: Path, inventory: CheckoutInventory) -> list[list[str]]:
    videos = inventory.videos
    print("\n=== inventário do clone ===")
    print(f"vídeos .mp4: {len(videos)} ({videos['size'].sum()} bytes)")
    print(pd.crosstab(videos["subject"], videos["category"], margins=True).to_string())
    print(f"arquivos de anotação: {len(inventory.annotation_files)}")
    for relative in inventory.annotation_files:
        print(f"  {relative}")
    print(f"arquivos fora do layout ({len(inventory.unrecognized)}):")
    for relative in inventory.unrecognized:
        print(f"  {relative}")

    print("\n=== duplicatas por SHA-256 ===")
    duplicates = find_duplicate_videos(root, videos)
    if not duplicates:
        print("nenhuma")
    for group in duplicates:
        print("conteúdo idêntico:")
        for member in group:
            print(f"  {member}")
    return duplicates
