"""Baixa anotações do Le2i a partir do dataset OmniFall (HuggingFace)."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "simplexsigil2/omnifall"
OUT_DIR = Path("data/labels/omnifall")
PROVENANCE_PATH = OUT_DIR / "PROVENANCE.json"
SPLITS = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
CONFIGS = ["le2i-cs", "labels"]

# Os nomes dos configs do dataset (`le2i-cs`, `labels`) já foram reestruturados
# uma vez upstream. Fixar a revisão garante que uma futura reestruturação não
# quebre ou altere silenciosamente este fetch.
OMNIFALL_REVISION = "68e5cee56a4bad38cca4aea791cac248f96e79a0"


def _write_csv(df, path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"skip {path} (já existe, use --force para sobrescrever)")
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    print(f"{path}: {len(df)} linhas, colunas={list(df.columns)}")


def fetch_splits(force: bool) -> None:
    dataset = load_dataset(DATASET_ID, "le2i-cs", revision=OMNIFALL_REVISION)
    for split, filename in SPLITS.items():
        df = dataset[split].to_pandas()
        _write_csv(df, OUT_DIR / filename, force)


def fetch_le2i_labels(force: bool) -> None:
    dataset = load_dataset(DATASET_ID, "labels", revision=OMNIFALL_REVISION)
    df = dataset["train"].to_pandas()
    df = df[df["dataset"] == "le2i"]
    _write_csv(df, OUT_DIR / "le2i.csv", force)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def write_provenance() -> None:
    filenames = [*SPLITS.values(), "le2i.csv"]
    files = []
    for filename in filenames:
        path = OUT_DIR / filename
        files.append(
            {
                "filename": filename,
                "sha256": _sha256(path),
                "rows": _row_count(path),
            }
        )

    provenance = {
        "dataset_repo_id": DATASET_ID,
        "revision": OMNIFALL_REVISION,
        "configs": CONFIGS,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }

    tmp_path = PROVENANCE_PATH.with_suffix(PROVENANCE_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_path, PROVENANCE_PATH)
    print(f"{PROVENANCE_PATH}: proveniência gravada ({len(files)} arquivos)")


def fetch(force: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fetch_splits(force)
    fetch_le2i_labels(force)
    write_provenance()


def verify() -> None:
    if not PROVENANCE_PATH.exists():
        print(
            f"erro: {PROVENANCE_PATH} não encontrado. "
            "Rode `uv run python scripts/fetch_labels.py` antes de verificar.",
            file=sys.stderr,
        )
        sys.exit(1)

    provenance = json.loads(PROVENANCE_PATH.read_text())
    problems = []

    for entry in provenance["files"]:
        path = OUT_DIR / entry["filename"]
        if not path.exists():
            problems.append(f"{path}: arquivo ausente")
            continue

        actual_sha256 = _sha256(path)
        if actual_sha256 != entry["sha256"]:
            problems.append(
                f"{path}: sha256 divergente "
                f"(esperado {entry['sha256']}, encontrado {actual_sha256})"
            )

    if problems:
        print("verificação falhou:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)

    print(f"verificação ok: {len(provenance['files'])} arquivos íntegros")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve arquivos já existentes em vez de pulá-los",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Não baixa nada; verifica a integridade dos arquivos já presentes"
        " contra PROVENANCE.json",
    )
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    fetch(args.force)


if __name__ == "__main__":
    main()
