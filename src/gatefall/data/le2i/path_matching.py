"""Casamento entre paths do OmniFall e vídeos extraídos do Le2i."""

import re
import sys
from pathlib import Path

import pandas as pd

VIDEO_EXTENSION = ".avi"
_VIDEO_NUMBER_PATTERN = re.compile(r"video\s*\((\d+)\)", re.IGNORECASE)


def normalize_environment_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def normalize_annotation_video_path(path: str) -> str:
    environment, _, video = path.partition("/")
    return f"{normalize_environment_name(environment)}/{video.strip().lower()}"


def normalize_extracted_video_path(relative_path: Path) -> str:
    parts = [part for part in relative_path.parts if part.lower() != "videos"]
    environment = parts[0]
    filename = parts[-1]
    match = _VIDEO_NUMBER_PATTERN.search(filename)
    if match is None:
        raise ValueError(
            f"não foi possível extrair o número do vídeo de {relative_path}"
        )
    video_number = int(match.group(1))
    return (
        f"{normalize_environment_name(environment)}/video_{video_number}"
    )


def discover_extracted_videos(raw_directory: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for path in sorted(raw_directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() != VIDEO_EXTENSION:
            continue
        relative_path = path.relative_to(raw_directory)
        normalized_path = normalize_extracted_video_path(relative_path)
        if normalized_path in videos and videos[normalized_path] != relative_path:
            print(
                f"erro: colisão de chave normalizada {normalized_path!r} entre "
                f"{videos[normalized_path]} e {relative_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        videos[normalized_path] = relative_path
    return videos


def index_annotation_paths(paths: pd.Series) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for path in paths.unique():
        normalized_path = normalize_annotation_video_path(path)
        if (
            normalized_path in annotations
            and annotations[normalized_path] != path
        ):
            print(
                f"erro: colisão de chave normalizada {normalized_path!r} entre "
                f"{annotations[normalized_path]!r} e {path!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        annotations[normalized_path] = path
    return annotations


def find_unmatched_video_keys(
    local_videos: dict[str, Path], annotation_paths: dict[str, str]
) -> tuple[list[str], list[str]]:
    local_keys = set(local_videos)
    annotation_keys = set(annotation_paths)
    return (
        sorted(local_keys - annotation_keys),
        sorted(annotation_keys - local_keys),
    )


def match_annotation_paths_to_videos(
    local_videos: dict[str, Path], annotation_paths: dict[str, str]
) -> dict[str, Path]:
    only_local, only_annotations = find_unmatched_video_keys(
        local_videos, annotation_paths
    )

    if only_local or only_annotations:
        print(
            "erro: bijeção entre vídeos locais e labels do OmniFall falhou.",
            file=sys.stderr,
        )
        if only_local:
            print(
                f"  presentes apenas localmente ({len(only_local)}):",
                file=sys.stderr,
            )
            for key in only_local:
                print(f"    {key}", file=sys.stderr)
        if only_annotations:
            print(
                f"  presentes apenas nas labels ({len(only_annotations)}):",
                file=sys.stderr,
            )
            for key in only_annotations:
                print(f"    {key}", file=sys.stderr)
        sys.exit(1)

    return {
        annotation_paths[key]: local_videos[key]
        for key in local_videos
    }
