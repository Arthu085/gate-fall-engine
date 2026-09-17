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
PROVENANCE_FILENAME = "PROVENANCE.json"
PROVENANCE_PATH = LABELS_DIR / PROVENANCE_FILENAME
SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
OMNIFALL_SPLITS = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
ANNOTATION_CONFIG_NAME = "le2i-cs"
LABEL_CONFIG_NAME = "labels"
OMNIFALL_CONFIGS = [ANNOTATION_CONFIG_NAME, LABEL_CONFIG_NAME]
OMNIFALL_REVISION = "68e5cee56a4bad38cca4aea791cac248f96e79a0"
LE2I_DATASET_NAME = "le2i"
LE2I_LABELS_FILENAME = "le2i.csv"

# Nome do config OmniFall que carrega cada protocolo Le2i. A revisão pinada
# (OMNIFALL_REVISION) serve ambos: `parquet/le2i-cs/*` e `parquet/le2i-cv/*`.
PROTOCOL_ANNOTATION_CONFIG = {"cs": "le2i-cs", "cv": "le2i-cv"}
PROTOCOL_LABELS_DIR = {
    "cs": Path("data/labels/omnifall"),
    "cv": Path("data/labels/omnifall_cv"),
}


def _labels_dir(protocol: str) -> Path:
    if protocol not in PROTOCOL_LABELS_DIR:
        raise ValueError(
            f"protocol não suportado: {protocol!r}; opções disponíveis: "
            f"{tuple(PROTOCOL_LABELS_DIR)}"
        )
    return PROTOCOL_LABELS_DIR[protocol]


def fetch_annotations(force: bool, protocol: str = "cs") -> None:
    labels_dir = _labels_dir(protocol)
    annotation_config_name = PROTOCOL_ANNOTATION_CONFIG[protocol]
    labels_dir.mkdir(parents=True, exist_ok=True)

    split_config = load_annotation_config(
        DATASET_ID, annotation_config_name, OMNIFALL_REVISION
    )
    for split, filename in OMNIFALL_SPLITS.items():
        dataframe = annotation_split_dataframe(split_config, split)
        write_annotation_csv(dataframe, labels_dir / filename, force)

    labels_config = load_annotation_config(
        DATASET_ID, LABEL_CONFIG_NAME, OMNIFALL_REVISION
    )
    labels = annotation_split_dataframe(labels_config, "train")
    le2i_labels = cast(
        pd.DataFrame, labels[labels["dataset"] == LE2I_DATASET_NAME]
    )
    write_annotation_csv(
        le2i_labels, labels_dir / LE2I_LABELS_FILENAME, force
    )

    write_annotation_provenance(
        provenance_path=labels_dir / PROVENANCE_FILENAME,
        files_directory=labels_dir,
        filenames=[*OMNIFALL_SPLITS.values(), LE2I_LABELS_FILENAME],
        dataset_repo_id=DATASET_ID,
        revision=OMNIFALL_REVISION,
        configs=[annotation_config_name, LABEL_CONFIG_NAME],
    )


def verify_annotations(protocol: str = "cs") -> None:
    labels_dir = _labels_dir(protocol)
    provenance_path = labels_dir / PROVENANCE_FILENAME
    if not provenance_path.exists():
        print(
            f"erro: {provenance_path} não encontrado. "
            "Rode `uv run python scripts/fetch_labels.py` antes de verificar.",
            file=sys.stderr,
        )
        sys.exit(1)

    problems, file_count = verify_annotation_provenance(
        provenance_path, labels_dir
    )
    if problems:
        print("verificação falhou:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)

    print(f"verificação ok: {file_count} arquivos íntegros")


def load_annotation_splits(
    fetch_command: str = "`uv run python scripts/fetch_labels.py`",
    protocol: str = "cs",
) -> dict[str, pd.DataFrame]:
    labels_dir = _labels_dir(protocol)
    splits: dict[str, pd.DataFrame] = {}
    for split, filename in SPLIT_FILES.items():
        path = labels_dir / filename
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
