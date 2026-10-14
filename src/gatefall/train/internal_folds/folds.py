"""Dobras internas por ambiente do `train` oficial do Le2i-CV (Etapa 2C).

Protocolo exploratório e não canônico: cada dobra retém um ambiente do
`train` oficial e treina nos outros dois. Só linhas `split == train` entram em
memória; `val` e `test` oficiais nunca são lidos. Na tabela de quadros da
dobra, a coluna `split` é reescrita para `development_train` ou `retained`,
então os datasets de janela existentes recortam cada lado sem código novo de
janelamento.
"""

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis.classification_diagnosis import check_output_dir, display_path
from gatefall.eval.analysis.internal_fold_feasibility import (
    EXPECTED_OFFICIAL_TRAIN,
    OFFICIAL_TRAIN_SPLIT,
    build_folds,
)
from gatefall.runs import ARM_RUN_DIR_NAMES, LOCAL_RUN_ROOTS, REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.shared.metrics import RESTRICTED_CLASSES

DATASET = "le2i-cv"
PROTOCOL = "le2i_cv_internal_environment_folds"
FOLD_RUN_ROOT = Path("runs/local/le2i_cv_internal_folds")
FOLDS_FILE = "folds.json"
DEVELOPMENT_SPLIT = "development_train"
RETAINED_SPLIT = "retained"
ARMS: tuple[str, ...] = ("A", "B0", "B1")
SEED_SETS: dict[str, tuple[int, ...]] = {
    "pilot": (42,),
    "assessment": (42, 43, 44, 45, 46),
}
# O engine de treino exige `val` e `test`; nas dobras os dois avaliam o mesmo
# ambiente retido, e nenhum deles é o split oficial de mesmo nome.
EVALUATION_SPLIT_MAPPING = {"train": DEVELOPMENT_SPLIT, "val": RETAINED_SPLIT, "test": RETAINED_SPLIT}
CHECKPOINT_POLICY = "last_epoch"
REPRODUCTION_CRITERION = (
    "O deslocamento para other é considerado reproduzido se, em pelo menos dois dos três ambientes retidos, "
    "a menor fração predita como other entre as seeds de B0 (ou de B1) superar a maior fração entre as seeds "
    "de A. A fração é a soma da coluna other fora da linha other da matriz de confusão do ambiente retido, "
    "dividida pelas janelas rotuladas fora de other. Só se aplica ao conjunto de cinco seeds (42 a 46)."
)

FeatureLoader = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class FoldPartition:
    fold: str
    retained_environment: str
    development_environments: tuple[str, ...]
    development_video_ids: tuple[str, ...]
    retained_video_ids: tuple[str, ...]
    frames: pd.DataFrame

    def membership(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "retained_environment": self.retained_environment,
            "development_environments": list(self.development_environments),
            "development_video_ids": list(self.development_video_ids),
            "retained_video_ids": list(self.retained_video_ids),
            "development_windows_train_stride": len(
                build_window_index(self.side_frames(DEVELOPMENT_SPLIT), stride=TRAIN_STRIDE)
            ),
            "retained_windows_eval_stride": len(
                build_window_index(self.side_frames(RETAINED_SPLIT), stride=EVAL_STRIDE)
            ),
        }

    def side_frames(self, split: str) -> pd.DataFrame:
        return cast(pd.DataFrame, self.frames[self.frames["split"] == split])


def _video_ids(frames: pd.DataFrame, environments: Iterable[str]) -> tuple[str, ...]:
    selected = cast(pd.DataFrame, frames[frames["env"].isin(list(environments))])
    return tuple(sorted(str(video_id) for video_id in selected["video_id"].unique()))


def build_fold_partitions(official_train: pd.DataFrame) -> list[FoldPartition]:
    splits = sorted(set(official_train["split"].astype(str)))
    if splits != [OFFICIAL_TRAIN_SPLIT]:
        raise ValueError(f"as dobras só aceitam linhas do train oficial; recebido splits={splits}")
    environments = sorted(str(env) for env in official_train["env"].unique())
    partitions: list[FoldPartition] = []
    for fold in build_folds(environments):
        retained = cast(str, fold["retained_environment"])
        development = tuple(cast(list[str], fold["development_environments"]))
        frames = official_train.copy()
        frames["split"] = np.where(frames["env"] == retained, RETAINED_SPLIT, DEVELOPMENT_SPLIT)
        partition = FoldPartition(
            fold=cast(str, fold["fold"]),
            retained_environment=retained,
            development_environments=development,
            development_video_ids=_video_ids(official_train, development),
            retained_video_ids=_video_ids(official_train, [retained]),
            frames=frames,
        )
        validate_partition(partition, official_train)
        partitions.append(partition)
    return partitions


def validate_partition(partition: FoldPartition, official_train: pd.DataFrame) -> None:
    development = set(partition.development_video_ids)
    retained = set(partition.retained_video_ids)
    official = {str(video_id) for video_id in official_train["video_id"].unique()}
    problems: list[str] = []
    if development & retained:
        problems.append(f"vídeos nos dois lados: {sorted(development & retained)}")
    if development | retained != official:
        problems.append("os dois lados não cobrem exatamente os vídeos do train oficial")
    if not development or not retained:
        problems.append("um dos lados está vazio")
    if partition.retained_environment in partition.development_environments:
        problems.append("o ambiente retido aparece no treino de desenvolvimento")
    environments = sorted(str(env) for env in official_train["env"].unique())
    if sorted([partition.retained_environment, *partition.development_environments]) != environments:
        problems.append("os ambientes da dobra não reproduzem os ambientes do train oficial")
    for split, environments_on_side, videos in (
        (DEVELOPMENT_SPLIT, set(partition.development_environments), development),
        (RETAINED_SPLIT, {partition.retained_environment}, retained),
    ):
        side = partition.side_frames(split)
        if set(side["env"].astype(str)) != environments_on_side:
            problems.append(f"{split}: ambientes divergentes da definição da dobra")
        if {str(video_id) for video_id in side["video_id"].unique()} != videos:
            problems.append(f"{split}: vídeos divergentes da definição da dobra")
    if problems:
        raise ValueError(f"dobra {partition.fold} inválida: " + "; ".join(problems))


def restricted_loader(loader: FeatureLoader, allowed_video_ids: Iterable[str], context: str) -> FeatureLoader:
    allowed = frozenset(allowed_video_ids)

    def guarded(video_id: str) -> np.ndarray:
        if video_id not in allowed:
            raise RuntimeError(f"{context}: leitura de feature fora do lado permitido da dobra: {video_id!r}")
        return loader(video_id)

    return guarded


def fold_dir(root: Path, fold: str) -> Path:
    return root / fold


def stats_dir(root: Path, fold: str) -> Path:
    return fold_dir(root, fold) / "stats"


def fold_run_dir(root: Path, fold: str, arm: str, seed: int) -> Path:
    if arm not in ARMS:
        raise ValueError(f"arma não suportada nas dobras internas: {arm!r}; opções: {', '.join(ARMS)}")
    return fold_dir(root, fold) / "runs" / ARM_RUN_DIR_NAMES[arm] / f"seed_{seed}"


def summary_dir(root: Path) -> Path:
    return root / "summary"


def protected_paths(frames_path: Path, manifest_path: Path) -> list[Path]:
    return [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        *(REPOSITORY_ROOT / root for root in LOCAL_RUN_ROOTS.values()),
        frames_path.parent,
        manifest_path.parent,
    ]


def validate_output_root(root: Path, frames_path: Path, manifest_path: Path) -> None:
    check_output_dir(root, protected_paths(frames_path, manifest_path))


def fold_names() -> list[str]:
    environments = cast(tuple[str, ...], EXPECTED_OFFICIAL_TRAIN["environments"])
    return [cast(str, fold["fold"]) for fold in build_folds(list(environments))]


def run_plan(root: Path, seed_set: str) -> list[dict[str, object]]:
    seeds = SEED_SETS[seed_set]
    return [
        {"fold": fold, "arm": arm, "seed": seed, "run_dir": display_path(fold_run_dir(root, fold, arm, seed))}
        for fold in fold_names()
        for arm in ARMS
        for seed in seeds
    ]


def membership_sha256(partitions: list[FoldPartition]) -> str:
    payload = json.dumps([partition.membership() for partition in partitions], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def folds_document(
    partitions: list[FoldPartition],
    input_sha256: dict[str, str],
    quality_features_path: str,
    quality_features_sha256: str,
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "canonical": False,
        "dataset": DATASET,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "input_sha256": input_sha256,
        "window_contract": {"window_frames": WINDOW_FRAMES, "train_stride": TRAIN_STRIDE, "eval_stride": EVAL_STRIDE},
        "restricted_classes": list(RESTRICTED_CLASSES),
        "arms": list(ARMS),
        "seed_sets": {name: list(seeds) for name, seeds in SEED_SETS.items()},
        "evaluation_split_mapping": EVALUATION_SPLIT_MAPPING,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "reproduction_criterion": REPRODUCTION_CRITERION,
        "quality_features_path": quality_features_path,
        "quality_features_sha256": quality_features_sha256,
        "membership_sha256": membership_sha256(partitions),
        "folds": [partition.membership() for partition in partitions],
    }


def read_folds_document(root: Path) -> dict[str, object]:
    path = root / FOLDS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} não existe; rode `prepare` antes")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get("protocol") != PROTOCOL or data.get("canonical") is not False:
        raise ValueError(f"{path} não descreve o protocolo {PROTOCOL} não canônico")
    return data


def validate_folds_document(
    document: dict[str, object], partitions: list[FoldPartition], input_sha256: dict[str, str]
) -> None:
    if document.get("input_sha256") != input_sha256:
        raise ValueError(f"{FOLDS_FILE}: frames ou manifesto mudaram desde o `prepare`; rode `prepare --force`")
    if document.get("membership_sha256") != membership_sha256(partitions):
        raise ValueError(f"{FOLDS_FILE}: a composição das dobras diverge da partição atual; rode `prepare --force`")
