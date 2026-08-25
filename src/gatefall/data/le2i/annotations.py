"""Configuração e preparação das anotações Le2i publicadas pelo OmniFall."""

import sys
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.data.omnifall.annotations import (
    annotation_split_dataframe,
    load_annotation_config,
    write_annotation_csv,
)
from gatefall.data.omnifall.provenance import (
    verify_annotation_provenance,
    write_annotation_provenance,
)

DATASET_ID = "simplexsigil2/omnifall"
LABELS_DIR = Path("data/labels/omnifall")
PROVENANCE_PATH = LABELS_DIR / "PROVENANCE.json"
SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
OMNIFALL_SPLITS = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
ANNOTATION_CONFIG_NAME = "le2i-cs"
LABEL_CONFIG_NAME = "labels"
OMNIFALL_CONFIGS = [ANNOTATION_CONFIG_NAME, LABEL_CONFIG_NAME]
OMNIFALL_REVISION = "68e5cee56a4bad38cca4aea791cac248f96e79a0"
LE2I_DATASET_NAME = "le2i"
LE2I_LABELS_FILENAME = "le2i.csv"


def fetch_annotations(force: bool) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    split_config = load_annotation_config(
        DATASET_ID, ANNOTATION_CONFIG_NAME, OMNIFALL_REVISION
    )
    for split, filename in OMNIFALL_SPLITS.items():
        dataframe = annotation_split_dataframe(split_config, split)
        write_annotation_csv(dataframe, LABELS_DIR / filename, force)

    labels_config = load_annotation_config(
        DATASET_ID, LABEL_CONFIG_NAME, OMNIFALL_REVISION
    )
    labels = annotation_split_dataframe(labels_config, "train")
    le2i_labels = cast(
        pd.DataFrame, labels[labels["dataset"] == LE2I_DATASET_NAME]
    )
    write_annotation_csv(
        le2i_labels, LABELS_DIR / LE2I_LABELS_FILENAME, force
    )

    write_annotation_provenance(
        provenance_path=PROVENANCE_PATH,
        files_directory=LABELS_DIR,
        filenames=[*OMNIFALL_SPLITS.values(), LE2I_LABELS_FILENAME],
        dataset_repo_id=DATASET_ID,
        revision=OMNIFALL_REVISION,
        configs=OMNIFALL_CONFIGS,
    )


def verify_annotations() -> None:
    if not PROVENANCE_PATH.exists():
        print(
            f"erro: {PROVENANCE_PATH} não encontrado. "
            "Rode `uv run python scripts/fetch_labels.py` antes de verificar.",
            file=sys.stderr,
        )
        sys.exit(1)

    problems, file_count = verify_annotation_provenance(
        PROVENANCE_PATH, LABELS_DIR
    )
    if problems:
        print("verificação falhou:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)

    print(f"verificação ok: {file_count} arquivos íntegros")


def load_annotation_splits(
    fetch_command: str = "`uv run python scripts/fetch_labels.py`"
) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split, filename in SPLIT_FILES.items():
        path = LABELS_DIR / filename
        if not path.exists():
            print(
                f"erro: {path} não encontrado. Rode "
                f"{fetch_command} antes.",
                file=sys.stderr,
            )
            sys.exit(1)
        splits[split] = pd.read_csv(path)
    return splits


def build_video_annotation_index(
    splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split, dataframe in splits.items():
        tagged = cast(pd.DataFrame, dataframe[["path", "subject", "cam"]].copy())
        tagged["split"] = split
        frames.append(tagged)
    pooled = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

    rows: list[dict[str, object]] = []
    for path, group in pooled.groupby("path", sort=False):
        subjects = group["subject"].unique()
        cameras = group["cam"].unique()
        splits_seen = group["split"].unique()
        if len(subjects) > 1 or len(cameras) > 1 or len(splits_seen) > 1:
            raise ValueError(
                f"invariante violado: {path} possui subject/cam/split "
                f"divergentes entre segmentos (subjects={subjects!r}, "
                f"cams={cameras!r}, splits={splits_seen!r})"
            )
        rows.append(
            {
                "path": path,
                "subject": int(subjects[0]),
                "cam": int(cameras[0]),
                "split": str(splits_seen[0]),
            }
        )
    return pd.DataFrame(rows, columns=["path", "subject", "cam", "split"])
