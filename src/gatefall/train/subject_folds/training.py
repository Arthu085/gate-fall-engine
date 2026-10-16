"""Treino do braço A numa dobra por sujeito com a receita e o engine congelados.

O treino usa só o treino de desenvolvimento da dobra, em `TRAIN_STRIDE`; o
grupo de sujeitos retido é avaliado em `EVAL_STRIDE` e ocupa as posições
`val` e `test` do engine, que só registra a curva de `val` por época e guarda
o checkpoint da última época. Nenhum split oficial além de `train` é lido.

A variante `detection_dropout` (Etapa 4C) troca só a fonte de janelas do
treino de desenvolvimento; a variante `subject_balanced` (Etapa 5A) troca só
a amostragem das épocas de treino. Nas duas, o retido continua limpo e a
receita, a seed e a estatística são as do controle `baseline` da mesma dobra.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.pose.kinematics import POSE_FEATURE_DIM
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.baseline_a.engine import run_training
from gatefall.train.internal_folds.folds import (
    DEVELOPMENT_SPLIT,
    RETAINED_SPLIT,
    FeatureLoader,
    restricted_loader,
)
from gatefall.train.internal_folds.training import guard_existing_fold_run, publish_fold_run_document
from gatefall.train.subject_folds.detection_dropout import (
    STAGE as DETECTION_DROPOUT_STAGE,
    DetectionDropoutWindowDataset,
    RawPoseLoader,
    intervention_record,
)
from gatefall.train.subject_folds.folds import (
    ARMS,
    BASELINE_VARIANT,
    CHECKPOINT_POLICY,
    DATASET,
    DETECTION_DROPOUT_VARIANT,
    EVALUATION_SPLIT_MAPPING,
    PROTOCOL,
    SEEDS,
    STAGE,
    SUBJECT_BALANCED_VARIANT,
    SubjectFoldPartition,
    variant_run_name,
)
from gatefall.train.subject_folds.stats import OFFICIAL_POSE_STATS, FoldPoseStats
from gatefall.train.subject_folds.subject_balanced_sampling import (
    STAGE as SUBJECT_BALANCED_STAGE,
    SubjectBalancedSampler,
    draws_sha256,
    expand_runs,
    intervention_record as subject_balanced_intervention_record,
    sampling_record,
)


def resolve_fold_config(
    seed: int, stats_path: Path, stats_sha256: str, variant: str = BASELINE_VARIANT
) -> TrainConfig:
    if seed not in SEEDS:
        raise ValueError(f"seed não suportada nas dobras por sujeito: {seed}; opções: {list(SEEDS)}")
    if stats_path.resolve() == OFFICIAL_POSE_STATS.resolve():
        raise ValueError("as dobras por sujeito não podem usar a estatística oficial de pose do Le2i-CS")
    return replace(
        BASELINE_A_CONFIG,
        run_name=variant_run_name(BASELINE_A_CONFIG.arm, variant),
        seed=seed,
        standardization_stats_path=str(stats_path),
        standardization_stats_sha256=stats_sha256,
    )


def fold_run_document(
    fold_record: dict[str, object],
    arm: str,
    seed: int,
    stats_sha256: str,
    input_sha256: dict[str, str],
    pose_features_sha256: str,
    membership_sha256: str,
) -> dict[str, object]:
    if arm not in ARMS:
        raise ValueError(f"arma não suportada nas dobras por sujeito: {arm!r}; opções: {', '.join(ARMS)}")
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "canonical": False,
        "dataset": DATASET,
        "fold": fold_record["fold"],
        "arm": arm,
        "seed": seed,
        "retained_subjects": fold_record["retained_subjects"],
        "development_subjects": fold_record["development_subjects"],
        "evaluation_split_mapping": EVALUATION_SPLIT_MAPPING,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "input_sha256": input_sha256,
        "pose_features_sha256": pose_features_sha256,
        "membership_sha256": membership_sha256,
        "standardization_stats_sha256": {"pose": stats_sha256},
    }


def _control_record(control_document: dict[str, object], control_fold_run_sha256: str) -> dict[str, object]:
    seed = cast(int, control_document["seed"])
    return {
        "variant": BASELINE_VARIANT,
        "run": f"{variant_run_name(cast(str, control_document['arm']), BASELINE_VARIANT)}/seed_{seed}",
        "control_stage": STAGE,
        "fold_run_sha256": control_fold_run_sha256,
    }


def detection_dropout_run_document(
    control_document: dict[str, object], control_fold_run_sha256: str
) -> dict[str, object]:
    """Documento do run aumentado: o do controle, mais a intervenção e o hash do `fold_run.json` do controle."""
    return {
        **control_document,
        "stage": DETECTION_DROPOUT_STAGE,
        "variant": DETECTION_DROPOUT_VARIANT,
        "intervention": intervention_record(cast(int, control_document["seed"])),
        "control": _control_record(control_document, control_fold_run_sha256),
    }


def subject_balanced_run_document(
    control_document: dict[str, object], control_fold_run_sha256: str, sampling: dict[str, object]
) -> dict[str, object]:
    """Documento do run balanceado: o do controle, mais a intervenção, as contagens sorteadas e o hash do controle."""
    return {
        **control_document,
        "stage": SUBJECT_BALANCED_STAGE,
        "variant": SUBJECT_BALANCED_VARIANT,
        "intervention": subject_balanced_intervention_record(cast(int, control_document["seed"])),
        "sampling": sampling,
        "control": _control_record(control_document, control_fold_run_sha256),
    }


def development_window_targets(partition: SubjectFoldPartition) -> tuple[np.ndarray, np.ndarray]:
    """Sujeito e rótulo de cada janela do treino de desenvolvimento, na ordem de índice do dataset de treino."""
    windows = partition.development_windows
    return windows["subject"].to_numpy(dtype=np.int64), windows["label"].to_numpy(dtype=np.int64)


def subject_balanced_sampling_record(partition: SubjectFoldPartition, seed: int) -> dict[str, object]:
    subjects, labels = development_window_targets(partition)
    if sorted(set(subjects.tolist())) != list(partition.development_subjects):
        raise ValueError(f"{partition.fold}: sujeitos das janelas de treino divergem dos sujeitos de desenvolvimento")
    return sampling_record(subjects, labels, seed, BASELINE_A_CONFIG.epochs)


def validate_sampling_record(
    sampling: object, fold: dict[str, object], seed: int, source: Path
) -> dict[str, object]:
    """Recalcula o sorteio a partir do índice de janelas gravado e exige que as contagens gravadas coincidam.

    O índice (sujeito e rótulo de cada janela, por trechos) é reconciliado com
    a composição da dobra em `folds.json`; frames e features não são relidos.
    """
    if not isinstance(sampling, dict):
        raise RuntimeError(f"{source}: sampling ausente; use --force para retreinar o run")
    try:
        subjects = expand_runs(sampling.get("window_subject_runs"))
        labels = expand_runs(sampling.get("window_label_runs"))
    except ValueError as exc:
        raise RuntimeError(f"{source}: sampling inconsistente: {exc}") from exc
    support = {name: int((labels == class_id).sum()) for class_id, name in enumerate(LE2I_LABEL_NAMES)}
    problems: list[str] = []
    if len(subjects) != len(labels) or len(subjects) != fold["development_windows_train_stride"]:
        problems.append("o índice de janelas não tem as janelas de treino da dobra")
    if sorted(set(subjects.tolist())) != fold["development_subjects"]:
        problems.append("os sujeitos do índice divergem dos sujeitos de desenvolvimento")
    if support != fold["development_support_train_stride"]:
        problems.append("os rótulos do índice divergem de development_support_train_stride")
    if not problems and sampling != sampling_record(subjects, labels, seed, BASELINE_A_CONFIG.epochs):
        problems.append("contagens ou sorteios gravados divergem do sorteio recalculado")
    if problems:
        raise RuntimeError(f"{source}: sampling inconsistente: " + "; ".join(problems))
    return sampling


def _subject_balanced_sampler(
    partition: SubjectFoldPartition, seed: int, train_source: PoseWindowDataset
) -> SubjectBalancedSampler:
    subjects, labels = development_window_targets(partition)
    if len(subjects) != len(train_source) or any(
        train_source.label_at(index) != int(label) for index, label in enumerate(labels)
    ):
        raise RuntimeError(f"{partition.fold}: índice de janelas do sampler diverge do dataset de treino")
    return SubjectBalancedSampler(subjects, seed)


def _development_source(
    partition: SubjectFoldPartition,
    seed: int,
    loader: FeatureLoader,
    raw_pose_loader: RawPoseLoader | None,
    variant: str,
    context: str,
) -> PoseWindowDataset:
    allowed = partition.development_video_ids
    features = restricted_loader(loader, allowed, f"{context}/{DEVELOPMENT_SPLIT}")
    if variant in (BASELINE_VARIANT, SUBJECT_BALANCED_VARIANT):
        return PoseWindowDataset(partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE, features)
    if variant != DETECTION_DROPOUT_VARIANT:
        raise ValueError(f"variante não suportada nas dobras por sujeito: {variant!r}")
    if raw_pose_loader is None:
        raise ValueError(f"{context}: a variante {variant} exige a pose bruta dos HDF5")
    raw = restricted_loader(raw_pose_loader, allowed, f"{context}/{DEVELOPMENT_SPLIT}/pose bruta")
    return DetectionDropoutWindowDataset(partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE, features, raw, seed)


def train_fold_run(
    partition: SubjectFoldPartition,
    seed: int,
    stats: FoldPoseStats,
    loader: FeatureLoader,
    document: dict[str, object],
    run_dir: Path,
    force: bool,
    variant: str = BASELINE_VARIANT,
    raw_pose_loader: RawPoseLoader | None = None,
) -> None:
    config = resolve_fold_config(seed, stats.path, stats.sha256, variant)
    guard_existing_fold_run(run_dir, document, force)
    context = f"{partition.fold}/{document['arm']}"
    train_source = _development_source(partition, seed, loader, raw_pose_loader, variant, context)
    if len(train_source) != stats.window_count:
        raise RuntimeError(
            f"{partition.fold}: {len(train_source)} janelas de treino, mas as estatísticas foram ajustadas em "
            f"{stats.window_count}"
        )
    retained_source = PoseWindowDataset(
        partition.frames,
        RETAINED_SPLIT,
        EVAL_STRIDE,
        restricted_loader(loader, partition.retained_video_ids, f"{context}/{RETAINED_SPLIT}"),
    )
    if len(retained_source) != cast(int, partition.membership_record["retained_windows_eval_stride"]):
        raise RuntimeError(f"{partition.fold}: janelas do retido divergem da composição da dobra")
    sampler = (
        _subject_balanced_sampler(partition, seed, train_source) if variant == SUBJECT_BALANCED_VARIANT else None
    )
    metrics = run_training(
        input_dim=POSE_FEATURE_DIM,
        train_source=train_source,
        val_source=retained_source,
        test_source=retained_source,
        stats=stats.stats,
        config=config,
        run_dir=run_dir,
        force=force,
        label_names=LE2I_LABEL_NAMES,
        train_sampler=sampler,
    )
    if metrics is None:
        return
    if sampler is not None:
        recorded = cast(dict[str, object], document["sampling"])
        if len(sampler.draws) != config.epochs or draws_sha256(sampler.draws) != recorded["draws_sha256"]:
            raise RuntimeError(f"{run_dir}: os sorteios do treino divergem da amostragem registrada em fold_run.json")
    publish_fold_run_document(run_dir, document)
