"""Criação e verificação da proveniência de anotações do OmniFall."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast

from gatefall.hashing import sha256_file


class ProvenanceFile(TypedDict):
    filename: str
    sha256: str
    rows: int


class AnnotationProvenance(TypedDict):
    dataset_repo_id: str
    revision: str
    configs: list[str]
    fetched_at: str
    files: list[ProvenanceFile]


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file) - 1


def write_annotation_provenance(
    provenance_path: Path,
    files_directory: Path,
    filenames: list[str],
    dataset_repo_id: str,
    revision: str,
    configs: list[str],
) -> None:
    files = [
        ProvenanceFile(
            filename=filename,
            sha256=sha256_file(files_directory / filename),
            rows=count_csv_rows(files_directory / filename),
        )
        for filename in filenames
    ]
    provenance = AnnotationProvenance(
        dataset_repo_id=dataset_repo_id,
        revision=revision,
        configs=configs,
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        files=files,
    )

    tmp_path = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_path, provenance_path)
    print(f"{provenance_path}: proveniência gravada ({len(files)} arquivos)")


def verify_annotation_provenance(
    provenance_path: Path, files_directory: Path
) -> tuple[list[str], int]:
    provenance = cast(
        AnnotationProvenance, json.loads(provenance_path.read_text())
    )
    problems: list[str] = []

    for entry in provenance["files"]:
        path = files_directory / entry["filename"]
        if not path.exists():
            problems.append(f"{path}: arquivo ausente")
            continue

        actual_sha256 = sha256_file(path)
        if actual_sha256 != entry["sha256"]:
            problems.append(
                f"{path}: sha256 divergente "
                f"(esperado {entry['sha256']}, encontrado {actual_sha256})"
            )

    return problems, len(provenance["files"])
