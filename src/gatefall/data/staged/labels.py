"""Anotações OmniFall pinadas dos datasets encenados e mapeamento para as dez classes GateFall."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, NUM_CLASSES
from gatefall.data.le2i.annotations import DATASET_ID, LABEL_CONFIG_NAME, OMNIFALL_REVISION
from gatefall.data.omnifall.annotations import (
    annotation_split_dataframe,
    load_annotation_config,
    write_annotation_csv,
)
from gatefall.data.omnifall.provenance import (
    verify_annotation_provenance,
    write_annotation_provenance,
)
from gatefall.data.staged.specs import StagedDatasetSpec

LABELS_FILENAME = "labels.csv"
PROVENANCE_FILENAME = "PROVENANCE.json"
REFERENCE_SPLIT_FILES = {
    "train": "omnifall_reference_train.csv",
    "validation": "omnifall_reference_val.csv",
    "test": "omnifall_reference_test.csv",
}
SEGMENT_COLUMNS = ("path", "label", "start", "end", "subject", "cam", "dataset")

# A taxonomia OmniFall tem 16 classes; os datasets encenados usam só 0–9, que
# coincidem em ordem e nome com as dez classes GateFall. As classes 10–15
# (kneel_down … jump) não existem no GateFall: em vez de fundi-las em `other`,
# viram IGNORE e são contadas como não suportadas no relatório de cobertura.
OMNIFALL_EXTENDED_LABELS = tuple(range(NUM_CLASSES, 16))


class UnverifiableLabelsError(ValueError):
    """Anotação ausente, malformada ou incompatível com a fonte temporal."""


def fetch_staged_labels(spec: StagedDatasetSpec, force: bool) -> None:
    spec.labels_dir.mkdir(parents=True, exist_ok=True)
    labels_config = load_annotation_config(DATASET_ID, LABEL_CONFIG_NAME, OMNIFALL_REVISION)
    labels = annotation_split_dataframe(labels_config, "train")
    subset = cast(pd.DataFrame, labels[labels["dataset"] == spec.omnifall_name])
    if subset.empty:
        raise UnverifiableLabelsError(
            f"OmniFall {OMNIFALL_REVISION}: nenhum segmento com dataset={spec.omnifall_name!r}"
        )
    write_annotation_csv(subset, spec.labels_dir / LABELS_FILENAME, force)
    filenames = [LABELS_FILENAME]
    configs = [LABEL_CONFIG_NAME]
    if spec.omnifall_reference_config is not None:
        reference = load_annotation_config(
            DATASET_ID, spec.omnifall_reference_config, OMNIFALL_REVISION
        )
        for split, filename in REFERENCE_SPLIT_FILES.items():
            write_annotation_csv(
                annotation_split_dataframe(reference, split), spec.labels_dir / filename, force
            )
            filenames.append(filename)
        configs.append(spec.omnifall_reference_config)
    write_annotation_provenance(
        provenance_path=spec.labels_dir / PROVENANCE_FILENAME,
        files_directory=spec.labels_dir,
        filenames=filenames,
        dataset_repo_id=DATASET_ID,
        revision=OMNIFALL_REVISION,
        configs=configs,
    )


def verify_staged_labels(spec: StagedDatasetSpec) -> list[str]:
    provenance_path = spec.labels_dir / PROVENANCE_FILENAME
    if not provenance_path.exists():
        return [f"{provenance_path}: ausente; rode `fetch-labels` primeiro"]
    problems, _ = verify_annotation_provenance(provenance_path, spec.labels_dir)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("revision") != OMNIFALL_REVISION:
        problems.append(
            f"{provenance_path}: revisão {provenance.get('revision')!r} diverge da pinada "
            f"{OMNIFALL_REVISION}"
        )
    return problems


def validate_segments(segments: pd.DataFrame, omnifall_name: str) -> pd.DataFrame:
    missing = [column for column in SEGMENT_COLUMNS if column not in segments.columns]
    if missing:
        raise UnverifiableLabelsError(f"colunas ausentes nas anotações: {missing}")
    if segments.empty:
        raise UnverifiableLabelsError("anotações vazias")
    datasets = set(segments["dataset"].astype(str))
    if datasets != {omnifall_name}:
        raise UnverifiableLabelsError(
            f"anotações misturam datasets {sorted(datasets)}; esperado {omnifall_name!r}"
        )
    start = segments["start"].to_numpy(dtype=np.float64)
    end = segments["end"].to_numpy(dtype=np.float64)
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        raise UnverifiableLabelsError("start/end não finitos nas anotações")
    if np.any(start < 0) or np.any(end <= start):
        bad = cast(pd.DataFrame, segments[(start < 0) | (end <= start)]).head(5)
        raise UnverifiableLabelsError(f"segmentos com intervalo inválido: {bad.to_dict(orient='records')}")
    labels = segments["label"].to_numpy()
    if labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels > 15):
        raise UnverifiableLabelsError("rótulos fora da taxonomia OmniFall 0–15")
    return cast(pd.DataFrame, segments[list(SEGMENT_COLUMNS)].copy())


def load_staged_segments(spec: StagedDatasetSpec) -> pd.DataFrame:
    path = spec.labels_dir / LABELS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path}: ausente; rode `fetch-labels --dataset {spec.identifier}`")
    return validate_segments(pd.read_csv(path), spec.omnifall_name)


def load_reference_splits(spec: StagedDatasetSpec) -> pd.DataFrame | None:
    frames = []
    for split, filename in REFERENCE_SPLIT_FILES.items():
        path = spec.labels_dir / filename
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        frame["reference_split"] = "val" if split == "validation" else split
        frames.append(frame)
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def map_to_gatefall(segments: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Devolve segmentos com rótulos GateFall e o número de segmentos não suportados.

    Segmentos 10–15 recebem IGNORE_LABEL explicitamente; nenhum rótulo é inventado.
    """
    mapped = segments.copy()
    unsupported = mapped["label"].isin(OMNIFALL_EXTENDED_LABELS)
    mapped.loc[unsupported, "label"] = IGNORE_LABEL
    return mapped, int(cast(int, unsupported.sum()))


def labels_path(spec: StagedDatasetSpec) -> Path:
    return spec.labels_dir / LABELS_FILENAME
