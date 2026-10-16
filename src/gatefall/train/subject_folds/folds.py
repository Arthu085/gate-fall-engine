"""Dobras por sujeito do `train` oficial do Le2i-CS (Etapa 4B).

Protocolo exploratório e não canônico, derivado do esquema
`pooled_minority_subjects` da Etapa 4A e congelado aqui: os grupos retidos
são os sujeitos brutos {0}, {1} e {3, 4, 6, 8}. Cada dobra retém um grupo e
treina nos demais sujeitos. Só linhas `split == train` entram em memória;
`val` e `test` oficiais nunca são lidos. Na tabela de quadros da dobra, a
coluna `split` é reescrita para `development_train` ou `retained`, então os
datasets de janela existentes recortam cada lado sem código novo de
janelamento.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.windowing import build_window_index
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import check_output_dir, display_path
from gatefall.eval.analysis.internal_fold_feasibility import OFFICIAL_TRAIN_SPLIT
from gatefall.hashing import sha256_file
from gatefall.pose.loading import pose_path
from gatefall.runs import ARM_RUN_DIR_NAMES, LOCAL_RUN_ROOTS, REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.internal_folds.folds import (
    CHECKPOINT_POLICY,
    DEVELOPMENT_SPLIT,
    EVALUATION_SPLIT_MAPPING,
    FOLD_RUN_ROOT as LE2I_CV_FOLD_RUN_ROOT,
    RETAINED_SPLIT,
    is_sha256,
    membership_sha256,
)
from gatefall.train.shared.metrics import RESTRICTED_CLASSES

DATASET = "le2i"
PROTOCOL = "le2i_cs_subject_folds"
STAGE = "4B"
SUBJECT_FOLD_ROOT = Path("runs/local/le2i/subject_folds")
FOLDS_FILE = "folds.json"
RETAINED_SUBJECT_GROUPS: tuple[tuple[int, ...], ...] = ((0,), (1,), (3, 4, 6, 8))
ARMS: tuple[str, ...] = ("A",)
SEEDS: tuple[int, ...] = (42,)
BASELINE_VARIANT = "baseline"
DETECTION_DROPOUT_VARIANT = "detection_dropout"
SUBJECT_BALANCED_VARIANT = "subject_balanced"
VARIANTS: tuple[str, ...] = (BASELINE_VARIANT, DETECTION_DROPOUT_VARIANT, SUBJECT_BALANCED_VARIANT)
SELECTION_POLICY = (
    "Nenhuma seleção de modelo: receita congelada do braço A, checkpoint da última época e macro-F1 restrita "
    "relatada por dobra. Não há escore agregado entre dobras; val e test oficiais e o alvo de 0,800 não entram "
    "em nenhuma decisão."
)
RESTRICTED_LABELS = tuple(LE2I_LABEL_NAMES[class_id] for class_id in RESTRICTED_CLASSES)


def fold_name(retained_subjects: Iterable[int]) -> str:
    return "retain_s" + "-".join(str(subject) for subject in retained_subjects)


def fold_names() -> list[str]:
    return [fold_name(group) for group in RETAINED_SUBJECT_GROUPS]


def frozen_subjects() -> list[int]:
    return sorted(subject for group in RETAINED_SUBJECT_GROUPS for subject in group)


def _label_support(windows: pd.DataFrame) -> dict[str, int]:
    return {name: int((windows["label"] == class_id).sum()) for class_id, name in enumerate(LE2I_LABEL_NAMES)}


def _restricted_spread(windows: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        LE2I_LABEL_NAMES[class_id]: int(cast(pd.Series, windows.loc[windows["label"] == class_id, column]).nunique())
        for class_id in RESTRICTED_CLASSES
    }


@dataclass(frozen=True)
class SubjectFoldPartition:
    fold: str
    retained_subjects: tuple[int, ...]
    development_subjects: tuple[int, ...]
    development_video_ids: tuple[str, ...]
    retained_video_ids: tuple[str, ...]
    frames: pd.DataFrame

    def side_frames(self, split: str) -> pd.DataFrame:
        return cast(pd.DataFrame, self.frames[self.frames["split"] == split])

    @cached_property
    def development_windows(self) -> pd.DataFrame:
        return build_window_index(self.side_frames(DEVELOPMENT_SPLIT), stride=TRAIN_STRIDE)

    @cached_property
    def retained_windows(self) -> pd.DataFrame:
        return build_window_index(self.side_frames(RETAINED_SPLIT), stride=EVAL_STRIDE)

    @cached_property
    def membership_record(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "retained_subjects": list(self.retained_subjects),
            "development_subjects": list(self.development_subjects),
            "development_video_ids": list(self.development_video_ids),
            "retained_video_ids": list(self.retained_video_ids),
            "development_windows_train_stride": len(self.development_windows),
            "retained_windows_eval_stride": len(self.retained_windows),
            "development_support_train_stride": _label_support(self.development_windows),
            "retained_support_eval_stride": _label_support(self.retained_windows),
            "retained_class_videos_eval_stride": _restricted_spread(self.retained_windows, "video_id"),
            "retained_class_subjects_eval_stride": _restricted_spread(self.retained_windows, "subject"),
        }


def _video_ids(frames: pd.DataFrame, subjects: Iterable[int]) -> tuple[str, ...]:
    selected = cast(pd.DataFrame, frames[frames["subject"].isin(list(subjects))])
    return tuple(sorted(str(video_id) for video_id in selected["video_id"].unique()))


def _validate_groups(official_train: pd.DataFrame, groups: tuple[tuple[int, ...], ...]) -> None:
    flattened = [subject for group in groups for subject in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"grupos retidos repetem sujeito: {groups}")
    present = sorted({int(subject) for subject in official_train["subject"]})
    if sorted(flattened) != present:
        raise ValueError(
            f"os grupos retidos congelados {sorted(flattened)} não cobrem exatamente os sujeitos do train "
            f"oficial {present}"
        )


def build_subject_fold_partitions(
    official_train: pd.DataFrame, groups: tuple[tuple[int, ...], ...] = RETAINED_SUBJECT_GROUPS
) -> list[SubjectFoldPartition]:
    splits = sorted(set(official_train["split"].astype(str)))
    if splits != [OFFICIAL_TRAIN_SPLIT]:
        raise ValueError(f"as dobras só aceitam linhas do train oficial; recebido splits={splits}")
    _validate_groups(official_train, groups)
    subjects = sorted({int(subject) for subject in official_train["subject"]})
    partitions: list[SubjectFoldPartition] = []
    for group in groups:
        retained = tuple(sorted(group))
        development = tuple(subject for subject in subjects if subject not in retained)
        frames = official_train.copy()
        frames["split"] = np.where(frames["subject"].isin(list(retained)), RETAINED_SPLIT, DEVELOPMENT_SPLIT)
        partition = SubjectFoldPartition(
            fold=fold_name(retained),
            retained_subjects=retained,
            development_subjects=development,
            development_video_ids=_video_ids(official_train, development),
            retained_video_ids=_video_ids(official_train, retained),
            frames=frames,
        )
        validate_partition(partition, official_train)
        partitions.append(partition)
    return partitions


def _missing_restricted(support: dict[str, int]) -> list[str]:
    return [label for label in RESTRICTED_LABELS if support[label] == 0]


def validate_partition(partition: SubjectFoldPartition, official_train: pd.DataFrame) -> None:
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
    if set(partition.development_subjects) & set(partition.retained_subjects):
        problems.append("sujeito retido aparece no treino de desenvolvimento")
    for split, subjects, videos in (
        (DEVELOPMENT_SPLIT, set(partition.development_subjects), development),
        (RETAINED_SPLIT, set(partition.retained_subjects), retained),
    ):
        side = partition.side_frames(split)
        if {int(subject) for subject in side["subject"]} != subjects:
            problems.append(f"{split}: sujeitos divergentes da definição da dobra")
        if {str(video_id) for video_id in side["video_id"].unique()} != videos:
            problems.append(f"{split}: vídeos divergentes da definição da dobra")
    if problems:
        raise ValueError(f"dobra {partition.fold} inválida: " + "; ".join(problems))
    record = partition.membership_record
    for side, field in (
        (DEVELOPMENT_SPLIT, "development_support_train_stride"),
        (RETAINED_SPLIT, "retained_support_eval_stride"),
    ):
        missing = _missing_restricted(cast(dict[str, int], record[field]))
        if missing:
            raise ValueError(
                f"dobra {partition.fold}: classes restritas ausentes em {side}: {missing}; a macro-F1 de oito "
                "classes não é comparável"
            )


def pose_features_sha256(video_ids: Iterable[str], pose_root: Path) -> str:
    """SHA-256 das linhas `<video_id> <sha256 do HDF5 de pose>` ordenadas por vídeo."""
    paths = {video_id: pose_path(video_id, pose_root=pose_root) for video_id in sorted(video_ids)}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"HDF5 de pose ausentes ({len(missing)}): {missing}")
    digest = hashlib.sha256()
    for video_id, path in paths.items():
        digest.update(f"{video_id} {sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def fold_dir(root: Path, fold: str) -> Path:
    return root / fold


def stats_dir(root: Path, fold: str) -> Path:
    return fold_dir(root, fold) / "stats"


def variant_run_name(arm: str, variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variante não suportada nas dobras por sujeito: {variant!r}; opções: {', '.join(VARIANTS)}")
    base = ARM_RUN_DIR_NAMES[arm]
    return base if variant == BASELINE_VARIANT else f"{base}_{variant}"


def fold_run_dir(root: Path, fold: str, arm: str, seed: int, variant: str = BASELINE_VARIANT) -> Path:
    if arm not in ARMS:
        raise ValueError(f"arma não suportada nas dobras por sujeito: {arm!r}; opções: {', '.join(ARMS)}")
    if seed not in SEEDS:
        raise ValueError(f"seed não suportada nas dobras por sujeito: {seed}; opções: {list(SEEDS)}")
    if fold not in fold_names():
        raise ValueError(f"dobra desconhecida: {fold!r}; opções: {', '.join(fold_names())}")
    return fold_dir(root, fold) / "runs" / variant_run_name(arm, variant) / f"seed_{seed}"


def summary_dir(root: Path) -> Path:
    return root / "summary"


def protected_paths(frames_path: Path, manifest_path: Path, pose_root: Path) -> list[Path]:
    return [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        REPOSITORY_ROOT / "docs",
        REPOSITORY_ROOT / LOCAL_RUN_ROOTS["le2i-cv"],
        REPOSITORY_ROOT / LE2I_CV_FOLD_RUN_ROOT,
        frames_path.parent,
        manifest_path.parent,
        pose_root,
    ]


def validate_output_root(root: Path, frames_path: Path, manifest_path: Path, pose_root: Path) -> None:
    """Recusa raízes protegidas e, dentro do repositório, tudo fora de `runs/local/le2i/subject_folds/`.

    A raiz própria fica dentro de `runs/local/le2i/`, dos runs canônicos do
    Le2i-CS; a exigência de prefixo impede que ela coincida com eles ou os
    contenha.
    """
    check_output_dir(root, protected_paths(frames_path, manifest_path, pose_root))
    resolved = root.resolve()
    own_root = (REPOSITORY_ROOT / SUBJECT_FOLD_ROOT).resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT) and not resolved.is_relative_to(own_root):
        raise ValueError(f"--output-root {root} dentro do repositório precisa ficar em {SUBJECT_FOLD_ROOT}")


def run_plan(root: Path, variant: str = BASELINE_VARIANT) -> list[dict[str, object]]:
    return [
        {
            "fold": fold,
            "arm": arm,
            "seed": seed,
            "variant": variant,
            "run_dir": display_path(fold_run_dir(root, fold, arm, seed, variant)),
        }
        for fold in fold_names()
        for arm in ARMS
        for seed in SEEDS
    ]


INPUT_ROLES = ("frames", "manifest")
FOLD_RECORD_FIELDS = (
    "fold",
    "retained_subjects",
    "development_subjects",
    "development_video_ids",
    "retained_video_ids",
    "development_windows_train_stride",
    "retained_windows_eval_stride",
    "development_support_train_stride",
    "retained_support_eval_stride",
    "retained_class_videos_eval_stride",
    "retained_class_subjects_eval_stride",
)


def protocol_contract() -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "canonical": False,
        "dataset": DATASET,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "grouping_unit": "raw_subject",
        "retained_subject_groups": [list(group) for group in RETAINED_SUBJECT_GROUPS],
        "window_contract": {"window_frames": WINDOW_FRAMES, "train_stride": TRAIN_STRIDE, "eval_stride": EVAL_STRIDE},
        "restricted_classes": list(RESTRICTED_CLASSES),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "evaluation_split_mapping": EVALUATION_SPLIT_MAPPING,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "selection_policy": SELECTION_POLICY,
    }


def folds_document(
    partitions: list[SubjectFoldPartition], input_sha256: dict[str, str], pose_sha256: str
) -> dict[str, object]:
    records = [partition.membership_record for partition in partitions]
    return {
        **protocol_contract(),
        "input_sha256": input_sha256,
        "pose_features_sha256": pose_sha256,
        "membership_sha256": membership_sha256(records),
        "folds": records,
    }


def _is_count(value: object, positive: bool = False) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and (value > 0 if positive else value >= 0)


def _video_list_problems(fold: str, field: str, value: object) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        return [f"{fold}: {field} deve ser uma lista não vazia de vídeos"]
    if value != sorted(set(value)):
        return [f"{fold}: {field} deve estar ordenada e sem repetição"]
    return []


def _support_problems(fold: str, field: str, value: object, total: object, labels: tuple[str, ...]) -> list[str]:
    if not isinstance(value, dict) or list(value) != list(labels) or not all(_is_count(v) for v in value.values()):
        return [f"{fold}: {field} deve ter contagens inteiras para {list(labels)}"]
    if total is not None and sum(value.values()) != total:
        return [f"{fold}: {field} não soma as janelas rotuladas do lado"]
    return []


def _fold_record_problems(record: object, group: tuple[int, ...]) -> list[str]:
    expected_fold = fold_name(group)
    if not isinstance(record, dict) or sorted(record) != sorted(FOLD_RECORD_FIELDS):
        return [f"{expected_fold}: campos da dobra divergem do contrato"]
    problems: list[str] = []
    if record["fold"] != expected_fold or record["retained_subjects"] != list(group):
        problems.append(f"{expected_fold}: nome ou sujeitos retidos divergem do grupo congelado")
    if record["development_subjects"] != [subject for subject in frozen_subjects() if subject not in group]:
        problems.append(f"{expected_fold}: sujeitos de desenvolvimento divergem dos demais sujeitos congelados")
    for field in ("development_video_ids", "retained_video_ids"):
        problems += _video_list_problems(expected_fold, field, record[field])
    for field in ("development_windows_train_stride", "retained_windows_eval_stride"):
        if not _is_count(record[field], positive=True):
            problems.append(f"{expected_fold}: {field} deve ser um inteiro positivo")
    for field, total in (
        ("development_support_train_stride", record["development_windows_train_stride"]),
        ("retained_support_eval_stride", record["retained_windows_eval_stride"]),
    ):
        support_problems = _support_problems(expected_fold, field, record[field], total, LE2I_LABEL_NAMES)
        problems += support_problems
        if not support_problems and _missing_restricted(record[field]):
            problems.append(f"{expected_fold}: {field} sem classe restrita {_missing_restricted(record[field])}")
    for field in ("retained_class_videos_eval_stride", "retained_class_subjects_eval_stride"):
        problems += _support_problems(expected_fold, field, record[field], None, RESTRICTED_LABELS)
    return problems


def _membership_problems(records: list[dict]) -> list[str]:
    retained_sets = [set(record["retained_video_ids"]) for record in records]
    universe = set().union(*retained_sets)
    problems: list[str] = []
    if sum(len(videos) for videos in retained_sets) != len(universe):
        problems.append("um vídeo aparece como retido em mais de uma dobra")
    for record, retained in zip(records, retained_sets):
        if set(record["development_video_ids"]) != universe - retained:
            problems.append(
                f"{record['fold']}: o treino de desenvolvimento não é exatamente o conjunto dos vídeos retidos "
                "nas outras dobras"
            )
    return problems


def validate_recorded_folds(document: object, source: str = FOLDS_FILE) -> dict[str, object]:
    """Reconcilia `folds.json` com o contrato congelado e com a própria composição gravada."""
    if not isinstance(document, dict):
        raise ValueError(f"{source} deve ser um objeto")
    contract = protocol_contract()
    expected_fields = {*contract, "input_sha256", "pose_features_sha256", "membership_sha256", "folds"}
    problems: list[str] = []
    if set(document) != expected_fields:
        problems.append(f"campos divergentes: {sorted(set(document) ^ expected_fields)}")
    problems += [f"{field} diverge do protocolo congelado" for field, value in contract.items()
                 if document.get(field) != value]
    input_sha256 = document.get("input_sha256")
    if not isinstance(input_sha256, dict) or sorted(input_sha256) != sorted(INPUT_ROLES) or not all(
        is_sha256(value) for value in input_sha256.values()
    ):
        problems.append(f"input_sha256 deve ter exatamente {list(INPUT_ROLES)} com SHA-256 válidos")
    if not is_sha256(document.get("pose_features_sha256")):
        problems.append("pose_features_sha256 inválido")
    records = document.get("folds")
    if not isinstance(records, list) or len(records) != len(RETAINED_SUBJECT_GROUPS):
        problems.append(f"folds deve listar as dobras {fold_names()}")
    else:
        record_problems = [
            problem
            for record, group in zip(records, RETAINED_SUBJECT_GROUPS)
            for problem in _fold_record_problems(record, group)
        ]
        problems += record_problems or _membership_problems(records)
        if document.get("membership_sha256") != membership_sha256(records):
            problems.append("membership_sha256 não corresponde à composição gravada")
    if problems:
        raise ValueError(f"{source} inconsistente: " + "; ".join(problems))
    return document


def read_folds_document(root: Path) -> dict[str, object]:
    path = root / FOLDS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} não existe; rode `prepare` antes")
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} não é um JSON válido: {exc}") from exc
    return validate_recorded_folds(data, str(path))


def validate_folds_document(
    document: dict[str, object],
    partitions: list[SubjectFoldPartition],
    input_sha256: dict[str, str],
    pose_sha256: str,
) -> None:
    if document.get("input_sha256") != input_sha256:
        raise ValueError(f"{FOLDS_FILE}: frames ou manifesto mudaram desde o `prepare`; rode `prepare --force`")
    if document.get("pose_features_sha256") != pose_sha256:
        raise ValueError(f"{FOLDS_FILE}: os HDF5 de pose mudaram desde o `prepare`; rode `prepare --force`")
    if document.get("folds") != [partition.membership_record for partition in partitions]:
        raise ValueError(f"{FOLDS_FILE}: a composição das dobras diverge da partição atual; rode `prepare --force`")
