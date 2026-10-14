"""Auditoria somente leitura das entradas e predições dos ambientes retidos das dobras internas (Etapa 2E).

Investiga o colapso de B0/B1 para `walk` no ambiente retido do piloto da
Etapa 2D sem retreinar nem alterar runs. Reusa a composição das dobras, a
proveniência das estatísticas por dobra, os loaders de feature e a validação
de runs do protocolo das dobras, e só lê linhas `split == train` do Le2i-CV.

Para cada vídeo do `train` oficial, confere a integridade de pose, DINOv3 e
qualidade. Para cada dobra, descreve as features padronizadas com as
estatísticas da própria dobra, separadas por lado, e refaz a inferência dos
runs no ambiente retido, reconciliando a matriz com `metrics.json`. Os
resumos são por vídeo, para não tratar janelas sobrepostas como observações
independentes. Defeitos técnicos verificados, deslocamentos descritivos e
hipóteses não testadas ficam separados.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, NUM_CLASSES
from gatefall.data.fusion_dataset import FusionWindowDataset
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import build_window_index
from gatefall.datasets.le2i import LE2I_LABEL_NAMES, Le2iDatasetAdapter
from gatefall.eval.analysis.classification_diagnosis import (
    check_output_dir,
    display_path,
    snapshot_inputs,
    validate_confusion_matrix,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.internal_fold_feasibility import OFFICIAL_TRAIN_SPLIT
from gatefall.eval.analysis.stratified_error_diagnosis import align_predictions, window_metadata
from gatefall.eval.baseline_a.cli import _predict_with_identity as predict_pose
from gatefall.eval.baseline_b0.cli import _predict_with_identity as predict_fusion
from gatefall.eval.baseline_b1.cli import _predict_with_identity as predict_gated_fusion
from gatefall.eval.shared.orchestration import Predictions
from gatefall.features.dinov3_standardization import FEATURE_DIM as DINOV3_FEATURE_DIM
from gatefall.features.dinov3_standardization import apply_standardization as standardize_visual
from gatefall.features.quality_storage import QUALITY_CHANNEL_NAMES, QUALITY_CHANNELS
from gatefall.features.standardization import KP_CONF_BLOCK_NAME
from gatefall.features.standardization import apply_standardization as standardize_pose
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_blocks
from gatefall.train.baseline_a.artifacts import load_compatible_checkpoint
from gatefall.train.baseline_a.config import TrainConfig
from gatefall.train.baseline_b0.artifacts import load_compatible_b0_checkpoint
from gatefall.train.baseline_b0.config import B0TrainConfig
from gatefall.train.baseline_b1.artifacts import load_compatible_b1_checkpoint
from gatefall.train.internal_folds.cli import _official_train, feature_loaders, validate_feature_files
from gatefall.train.internal_folds.folds import (
    ARMS,
    DATASET,
    DEVELOPMENT_SPLIT,
    FOLD_RUN_ROOT,
    FOLDS_FILE,
    PROTOCOL,
    RETAINED_SPLIT,
    SEED_SETS,
    FoldPartition,
    build_fold_partitions,
    fold_run_dir,
    protected_paths,
    read_folds_document,
    restricted_loader,
    stats_dir,
    validate_folds_document,
)
from gatefall.train.internal_folds.stats import (
    DINOV3_STATS_FILE,
    POSE_STATS_FILE,
    PROVENANCE_FILE,
    FoldStats,
    load_fold_stats,
)
from gatefall.train.internal_folds.summary import run_entry
from gatefall.train.internal_folds.training import (
    FOLD_RUN_FILE,
    RUN_ARTIFACTS,
    FeatureLoaders,
    FoldConfig,
    QualityFeatures,
    resolve_fold_config,
    validate_fold_run,
)

AUDIT_JSON_FILE = "internal_fold_retained_audit.json"
AUDIT_SUMMARY_FILE = "internal_fold_retained_audit.md"
SEED_SET = "pilot"
WALK_LABEL = "walk"
WALK_ID = LE2I_LABEL_NAMES.index(WALK_LABEL)
MODALITIES = ("pose", "dinov3")
SIDES = (DEVELOPMENT_SPLIT, RETAINED_SPLIT)
# Limiares só descritivos: contam valores e vídeos, não filtram nem
# selecionam nenhum resultado.
EXTREME_Z = 5.0
MOSTLY_WALK_FRACTION = 0.9

UNRESOLVED_HYPOTHESES = [
    "Mudança de entrada visual: B0/B1 podem recair em uma classe frequente do treino quando a entrada DINOv3 "
    "padronizada sai da faixa vista no treino de desenvolvimento. A auditoria só mostra se o deslocamento e o "
    "colapso coexistem; separar causa exigiria uma intervenção na inferência, como substituir a entrada visual, "
    "fora do escopo desta etapa.",
    "Variância visual estreita no desenvolvimento: quando os ambientes de desenvolvimento são visualmente "
    "próximos, o desvio-padrão por dimensão ajustado na dobra fica pequeno e amplia o z de um ambiente novo. É "
    "uma propriedade da padronização por dobra, não um defeito, e não se sabe se B0/B1 seriam robustos a ela com "
    "outra composição.",
    "Atalho de cena: B0/B1 quase memorizam o treino de desenvolvimento, e o DINOv3 pode codificar a cena junto "
    "com a postura. Nada aqui mede quanto da decisão vem da cena.",
    "Composição do retido e seed: o retido muda sujeitos, câmera e proporção de classes ao mesmo tempo, e o "
    "piloto tem uma seed por arma.",
]

LIMITATIONS = [
    "Exploratório e não canônico: só o train oficial do Le2i-CV é lido; val e test oficiais não entram, e nada "
    "aqui explica a falha observada no test oficial.",
    "As estatísticas por dobra são ajustadas no próprio treino de desenvolvimento, então os vídeos de "
    "desenvolvimento são uma referência dentro da amostra; a comparação justa entre ambientes retidos é entre "
    "dobras, cada uma com as próprias estatísticas.",
    "Os resumos são por vídeo (mediana, mínimo e máximo); vídeos têm durações diferentes, e frações por janela "
    "aparecem só como contexto, sobre janelas sobrepostas.",
    "As predições são de uma reinferência validada; a reconciliação com metrics.json é necessária, mas não prova "
    "que cada janela recebeu o mesmo rótulo da avaliação histórica, que nunca foi persistida por janela.",
    "Uma seed por arma (piloto). Os limiares de |z| e de fração walk só descrevem; não definem defeito nem "
    "selecionam nada.",
]


@dataclass(frozen=True)
class VideoFeatures:
    pose: np.ndarray
    visual: np.ndarray
    quality: np.ndarray


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _finite_mean(values: np.ndarray) -> float | None:
    finite = _finite(values)
    return float(finite.mean()) if finite.size else None


def _finite_rms(values: np.ndarray) -> float | None:
    finite = _finite(values)
    return float(np.sqrt(np.mean(np.square(finite)))) if finite.size else None


def _nonfinite(values: np.ndarray) -> int:
    return int(values.size - np.count_nonzero(np.isfinite(values)))


def spread(values: list[float | None]) -> dict[str, object]:
    present = [value for value in values if value is not None]
    if not present:
        return {"n_videos": 0, "median": None, "min": None, "max": None}
    return {
        "n_videos": len(present),
        "median": float(np.median(present)),
        "min": float(min(present)),
        "max": float(max(present)),
    }


def _kp_conf_slice() -> slice:
    start, end = next((start, end) for name, start, end in feature_blocks() if name == KP_CONF_BLOCK_NAME)
    return slice(start, end)


def video_integrity(video_id: str, env: str, subject: int, n_frames: int, features: VideoFeatures) -> dict:
    expected = {
        "pose": (features.pose, POSE_FEATURE_DIM),
        "dinov3": (features.visual, DINOV3_FEATURE_DIM),
        "quality": (features.quality, QUALITY_CHANNELS),
    }
    defects = [
        f"{video_id}: {name} com forma {tuple(array.shape)}, esperado ({n_frames}, {dim})"
        for name, (array, dim) in expected.items()
        if array.shape != (n_frames, dim)
    ]
    shapes_match = not defects
    defects += [
        f"{video_id}: {name} com {count} valor(es) não finito(s)"
        for name, (array, _dim) in expected.items()
        if (count := _nonfinite(array))
    ]
    record: dict[str, object] = {
        "video_id": video_id,
        "env": env,
        "subject": subject,
        "n_frames": n_frames,
        "nonfinite_values": {name: _nonfinite(array) for name, (array, _dim) in expected.items()},
    }
    if not shapes_match:
        record["defects"] = defects
        return record

    visual, quality = features.visual, features.quality
    zero_rows = int(np.count_nonzero(~np.any(visual != 0, axis=1)))
    temporal_std = _finite_mean(visual.std(axis=0)) if n_frames > 1 else None
    out_of_range = int(np.count_nonzero((quality < 0) | (quality > 1)))
    if zero_rows:
        defects.append(f"{video_id}: dinov3 com {zero_rows} quadro(s) inteiramente zerado(s)")
    if n_frames > 1 and bool(np.all(visual == visual[:1])):
        defects.append(f"{video_id}: dinov3 constante no tempo")
    if out_of_range:
        defects.append(f"{video_id}: qualidade com {out_of_range} valor(es) fora de [0, 1]")
    record.update(
        {
            "dinov3_zero_rows": zero_rows,
            "dinov3_temporal_std_mean": temporal_std,
            "dinov3_l2_norm_mean": _finite_mean(np.linalg.norm(visual, axis=1)),
            "pose_keypoint_confidence_mean": _finite_mean(features.pose[:, _kp_conf_slice()]),
            "quality_out_of_range": out_of_range,
            **{f"{name}_mean": _finite_mean(quality[:, index]) for index, name in enumerate(QUALITY_CHANNEL_NAMES)},
            "defects": defects,
        }
    )
    return record


INTEGRITY_DESCRIPTORS = (
    "pose_keypoint_confidence_mean",
    "dinov3_l2_norm_mean",
    "dinov3_temporal_std_mean",
    *(f"{name}_mean" for name in QUALITY_CHANNEL_NAMES),
)


def environment_integrity(videos: list[dict]) -> dict[str, dict]:
    environments: dict[str, dict] = {}
    for env in sorted({cast(str, video["env"]) for video in videos}):
        members = [video for video in videos if video["env"] == env]
        environments[env] = {
            "n_videos": len(members),
            "videos_with_defects": sum(1 for video in members if video["defects"]),
            **{name: spread([video.get(name) for video in members]) for name in INTEGRITY_DESCRIPTORS},
        }
    return environments


def label_composition(official_train: pd.DataFrame) -> dict[str, dict]:
    windows = build_window_index(official_train, stride=EVAL_STRIDE)
    composition: dict[str, dict] = {}
    for env, group in windows.groupby("env"):
        per_class = {}
        for label_id, name in enumerate(LE2I_LABEL_NAMES):
            members = cast(pd.DataFrame, group[group["label"] == label_id])
            if len(members):
                per_class[name] = {"windows": len(members), "videos": len(set(members["video_id"]))}
        composition[str(env)] = {
            "labeled_windows": len(group),
            "videos": len(set(group["video_id"])),
            "classes": per_class,
        }
    return composition


def standardized_shift(z: np.ndarray) -> dict[str, float | None]:
    return {
        "mean_z_rms": _finite_rms(z.mean(axis=0)),
        "abs_z_above_extreme_fraction": float(np.mean(np.abs(z) > EXTREME_Z)),
    }


def _std_summary(std: np.ndarray, standardized: np.ndarray) -> dict[str, object]:
    values = std[standardized]
    return {
        "standardized_dims": int(standardized.sum()),
        "std_p10": float(np.quantile(values, 0.1)),
        "std_median": float(np.median(values)),
        "std_p90": float(np.quantile(values, 0.9)),
    }


def fold_normalization(partition: FoldPartition, stats: FoldStats, features: dict[str, VideoFeatures]) -> dict:
    pose_mask = ~np.asarray(stats.pose.excluded_mask) & ~np.asarray(stats.pose.guarded_mask)
    visual_mask = ~np.asarray(stats.dinov3.guarded_mask)
    sides = {video_id: DEVELOPMENT_SPLIT for video_id in partition.development_video_ids}
    sides.update({video_id: RETAINED_SPLIT for video_id in partition.retained_video_ids})
    videos = []
    for video_id, side in sorted(sides.items()):
        video = features[video_id]
        videos.append(
            {
                "video_id": video_id,
                "side": side,
                "pose": standardized_shift(standardize_pose(video.pose, stats.pose)[:, pose_mask]),
                "dinov3": standardized_shift(standardize_visual(video.visual, stats.dinov3)[:, visual_mask]),
            }
        )
    summary = {
        side: {
            modality: {
                metric: spread([video[modality][metric] for video in videos if video["side"] == side])
                for metric in ("mean_z_rms", "abs_z_above_extreme_fraction")
            }
            for modality in MODALITIES
        }
        for side in SIDES
    }
    return {
        "stats": {
            "development_windows": stats.window_count,
            "pose": {
                **_std_summary(np.asarray(stats.pose.std), pose_mask),
                "excluded_dims": int(np.sum(stats.pose.excluded_mask)),
                "guarded_dims": stats.pose.guarded_count,
            },
            "dinov3": {
                **_std_summary(np.asarray(stats.dinov3.std), visual_mask),
                "guarded_dims": stats.dinov3.guarded_count,
            },
        },
        "sides": summary,
        "videos": videos,
    }


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float | None:
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not np.isfinite(norms) or norms == 0.0:
        return None
    return float(1.0 - float(first @ second) / norms)


def dinov3_geometry(features: dict[str, VideoFeatures], environments: dict[str, str]) -> dict:
    """Distâncias de cosseno entre médias DINOv3 brutas, sem estatística de nenhuma dobra.

    O centróide de um ambiente é a média das médias por vídeo, para que
    vídeos longos não pesem mais. A distância ao próprio ambiente deixa o
    vídeo fora do centróide.
    """
    video_means = {video_id: features[video_id].visual.mean(axis=0) for video_id in sorted(environments)}
    members = {
        env: [video_id for video_id in video_means if environments[video_id] == env]
        for env in sorted(set(environments.values()))
    }
    centroids = {env: np.mean([video_means[video_id] for video_id in ids], axis=0) for env, ids in members.items()}
    names = list(centroids)
    pairs = [
        {"first": first, "second": second, "cosine_distance": cosine_distance(centroids[first], centroids[second])}
        for index, first in enumerate(names)
        for second in names[index + 1:]
    ]
    per_environment = {}
    for env, ids in members.items():
        own: list[float | None] = []
        nearest_other: list[float | None] = []
        for video_id in ids:
            others = [video_means[other] for other in ids if other != video_id]
            own.append(cosine_distance(video_means[video_id], np.mean(others, axis=0)) if others else None)
            distances = [cosine_distance(video_means[video_id], centroids[other]) for other in names if other != env]
            present = [distance for distance in distances if distance is not None]
            nearest_other.append(min(present) if present else None)
        per_environment[env] = {
            "to_own_environment_leave_one_out": spread(own),
            "to_nearest_other_environment": spread(nearest_other),
        }
    return {"centroid_distances": pairs, "videos": per_environment}


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix.tolist()


def reconcile(aligned: pd.DataFrame, stored_split: dict, context: str) -> dict:
    labeled = cast(pd.DataFrame, aligned[aligned["label"] != IGNORE_LABEL])
    recomputed = confusion_matrix(labeled["label"].to_numpy(), labeled["pred_label"].to_numpy())
    stored = validate_confusion_matrix(stored_split["confusion_matrix"], context)
    differing = int(np.count_nonzero(np.asarray(recomputed) != np.asarray(stored)))
    return {
        "labeled_windows": len(labeled),
        "ignored_windows": len(aligned) - len(labeled),
        "matches_metrics_json": differing == 0,
        "differing_cells": differing,
    }


def video_predictions(aligned: pd.DataFrame) -> list[dict]:
    labeled = cast(pd.DataFrame, aligned[aligned["label"] != IGNORE_LABEL])
    videos = []
    for video_id, group in labeled.groupby("video_id"):
        predicted = np.bincount(group["pred_label"].to_numpy(), minlength=NUM_CLASSES)
        videos.append(
            {
                "video_id": str(video_id),
                "subject": int(group["subject"].iloc[0]),
                "labeled_windows": len(group),
                "true_walk_windows": int((group["label"] == WALK_ID).sum()),
                "predicted_walk_fraction": float(predicted[WALK_ID] / len(group)),
                "accuracy": float((group["label"] == group["pred_label"]).mean()),
                "predicted": {LE2I_LABEL_NAMES[index]: int(count) for index, count in enumerate(predicted) if count},
            }
        )
    return videos


def prediction_summary(videos: list[dict]) -> dict:
    windows = sum(video["labeled_windows"] for video in videos)
    predicted: dict[str, int] = {}
    for video in videos:
        for name, count in video["predicted"].items():
            predicted[name] = predicted.get(name, 0) + count
    fractions = [video["predicted_walk_fraction"] for video in videos]
    return {
        "labeled_windows": windows,
        "window_predicted_walk_fraction": predicted.get(WALK_LABEL, 0) / windows if windows else None,
        "predicted_class_windows": {name: predicted[name] for name in LE2I_LABEL_NAMES if name in predicted},
        "video_predicted_walk_fraction": spread(fractions),
        "videos_mostly_walk": sum(1 for fraction in fractions if fraction >= MOSTLY_WALK_FRACTION),
    }


def predict_retained(
    partition: FoldPartition,
    config: FoldConfig,
    run_dir: Path,
    stats: FoldStats,
    loaders: FeatureLoaders,
    device: str,
) -> Predictions:
    context = f"{partition.fold}/{config.arm}/auditoria"
    allowed = partition.retained_video_ids
    pose = restricted_loader(loaders.pose, allowed, f"{context}/pose")
    visual = restricted_loader(loaders.visual, allowed, f"{context}/dinov3")
    quality = restricted_loader(loaders.quality, allowed, f"{context}/qualidade")
    frames = partition.frames
    checkpoint = run_dir / "checkpoint.pt"
    if isinstance(config, TrainConfig):
        model = load_compatible_checkpoint(checkpoint, config).to(device).eval()
        source = PoseWindowDataset(frames, RETAINED_SPLIT, EVAL_STRIDE, pose, drop_ignored=False)
        return predict_pose(model, source, stats.pose, device, config.batch_size)
    if isinstance(config, B0TrainConfig):
        fusion = load_compatible_b0_checkpoint(checkpoint, config).to(device).eval()
        fusion_source = FusionWindowDataset(frames, RETAINED_SPLIT, EVAL_STRIDE, pose, visual, drop_ignored=False)
        return predict_fusion(fusion, fusion_source, stats.pose, stats.dinov3, device, config.batch_size)
    gated = load_compatible_b1_checkpoint(checkpoint, config).to(device).eval()
    gated_source = GatedFusionWindowDataset(
        frames, RETAINED_SPLIT, EVAL_STRIDE, pose, visual, quality, drop_ignored=False
    )
    return predict_gated_fusion(gated, gated_source, stats.pose, stats.dinov3, device, config.batch_size)


def run_audit_entry(
    root: Path,
    partition: FoldPartition,
    fold_record: dict,
    stats: FoldStats,
    quality: QualityFeatures,
    loaders: FeatureLoaders,
    arm: str,
    seed: int,
    device: str,
) -> dict:
    entry = run_entry(root, fold_record, stats.files, quality, arm, seed)
    run_dir = fold_run_dir(root, partition.fold, arm, seed)
    config = validate_fold_run(run_dir, resolve_fold_config(arm, seed, stats.files, quality))
    context = f"{partition.fold}/{arm}/seed_{seed}"
    predictions = predict_retained(partition, config, run_dir, stats, loaders, device)
    aligned = align_predictions(window_metadata(partition.frames, RETAINED_SPLIT, context), predictions, context)
    with (run_dir / "metrics.json").open(encoding="utf-8") as stream:
        stored = json.load(stream)["final"]["val"]
    videos = video_predictions(aligned)
    return {
        "arm": arm,
        "seed": seed,
        "run_dir": entry["run_dir"],
        "retained_macro_f1_restricted": entry["retained_macro_f1_restricted"],
        "reconciliation": reconcile(aligned, stored, context),
        "summary": prediction_summary(videos),
        "videos": videos,
    }


def _cached_loaders(features: dict[str, VideoFeatures]) -> FeatureLoaders:
    return FeatureLoaders(
        pose=lambda video_id: features[video_id].pose,
        visual=lambda video_id: features[video_id].visual,
        quality=lambda video_id: features[video_id].quality,
    )


def _fmt(value: object, digits: int = 3) -> str:
    return "n/d" if value is None else f"{cast(float, value):.{digits}f}"


def _spread_text(block: dict) -> str:
    return f"{_fmt(block['median'])} ({_fmt(block['min'])}–{_fmt(block['max'])}, {block['n_videos']} vídeos)"


def verified_defects(report: dict) -> list[str]:
    defects = [defect for video in report["integrity"]["videos"] for defect in video["defects"]]
    for fold in report["folds"]:
        alignment = fold["alignment"]
        if not alignment["matches_folds_json"]:
            defects.append(
                f"{fold['fold']}: {alignment['retained_labeled_windows']} janelas rotuladas no retido, "
                f"{alignment['recorded_retained_windows']} em {FOLDS_FILE}"
            )
        for run in fold["runs"]:
            reconciliation = run["reconciliation"]
            if not reconciliation["matches_metrics_json"]:
                defects.append(
                    f"{fold['fold']}/{run['arm']}: a reinferência diverge de final.val de metrics.json em "
                    f"{reconciliation['differing_cells']} célula(s) da matriz de confusão"
                )
    return defects


def descriptive_shifts(report: dict) -> list[str]:
    shifts = []
    for fold in report["folds"]:
        sides = fold["normalization"]["sides"]
        for modality in MODALITIES:
            retained = sides[RETAINED_SPLIT][modality]
            development = sides[DEVELOPMENT_SPLIT][modality]
            text = (
                f"`{fold['fold']}`, {modality}: RMS do z médio por vídeo no retido "
                f"{_spread_text(retained['mean_z_rms'])}, no desenvolvimento "
                f"{_spread_text(development['mean_z_rms'])}. Mediana por vídeo da fração com |z| > {EXTREME_Z:g}: "
                f"{_fmt(retained['abs_z_above_extreme_fraction']['median'], 4)} no retido e "
                f"{_fmt(development['abs_z_above_extreme_fraction']['median'], 4)} no desenvolvimento."
            )
            retained_min, development_max = retained["mean_z_rms"]["min"], development["mean_z_rms"]["max"]
            if retained_min is not None and development_max is not None and retained_min > development_max:
                text += " Todos os vídeos retidos ficam acima do maior valor do desenvolvimento."
            shifts.append(text)
        visual_std = fold["normalization"]["stats"]["dinov3"]
        shifts.append(
            f"`{fold['fold']}`: mediana do desvio-padrão DINOv3 ajustado no desenvolvimento "
            f"{_fmt(visual_std['std_median'], 4)} (p10 {_fmt(visual_std['std_p10'], 4)}, "
            f"p90 {_fmt(visual_std['std_p90'], 4)})."
        )
    for pair in report["dinov3_geometry"]["centroid_distances"]:
        shifts.append(
            f"Distância de cosseno entre centróides DINOv3 brutos de `{pair['first']}` e `{pair['second']}`: "
            f"{_fmt(pair['cosine_distance'], 4)}."
        )
    walk_shares = [
        f"`{env}` {_fmt(block['classes'].get(WALK_LABEL, {}).get('windows', 0) / block['labeled_windows'])}"
        for env, block in report["label_composition"].items()
    ]
    shifts.append(f"Fração `{WALK_LABEL}` das janelas rotuladas em stride 1: {', '.join(walk_shares)}.")
    return shifts


def prediction_observations(report: dict) -> list[str]:
    observations = []
    for fold in report["folds"]:
        for run in fold["runs"]:
            summary = run["summary"]
            per_video = summary["video_predicted_walk_fraction"]
            observations.append(
                f"`{fold['fold']}` / {run['arm']}: {_fmt(summary['window_predicted_walk_fraction'])} das janelas "
                f"rotuladas preditas como `{WALK_LABEL}`; por vídeo, {_spread_text(per_video)}; "
                f"{summary['videos_mostly_walk']} de {per_video['n_videos']} vídeos com fração ≥ "
                f"{MOSTLY_WALK_FRACTION:g}."
            )
    return observations


def build_report(
    root: Path,
    frames_path: Path,
    manifest_path: Path,
    loaders: FeatureLoaders | None = None,
    quality: QualityFeatures | None = None,
    device: str | None = None,
) -> tuple[dict, dict[str, str]]:
    frames, manifest, snapshot, input_sha256 = _official_train(frames_path, manifest_path)
    quality = quality or validate_feature_files(frames, manifest)
    loaders = loaders or feature_loaders()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    partitions = build_fold_partitions(frames)
    document = read_folds_document(root)
    validate_folds_document(document, partitions, input_sha256)
    if (document.get("quality_features_path"), document.get("quality_features_sha256")) != (
        str(quality.root),
        quality.sha256,
    ):
        raise ValueError(f"{FOLDS_FILE}: os sidecars de qualidade mudaram desde o `prepare`")

    metadata = cast(
        pd.DataFrame,
        frames.groupby("video_id").agg(env=("env", "first"), subject=("subject", "first"), K=("label", "size")),
    ).reset_index()
    rows = [
        (str(video_id), str(env), int(subject), int(n_frames))
        for video_id, env, subject, n_frames in zip(
            metadata["video_id"], metadata["env"], metadata["subject"], metadata["K"]
        )
    ]
    environments = {video_id: env for video_id, env, _subject, _n_frames in rows}
    features = {
        video_id: VideoFeatures(loaders.pose(video_id), loaders.visual(video_id), loaders.quality(video_id))
        for video_id in sorted(environments)
    }
    integrity = [
        video_integrity(video_id, env, subject, n_frames, features[video_id])
        for video_id, env, subject, n_frames in sorted(rows)
    ]
    cached = _cached_loaders(features)

    records = {cast(str, record["fold"]): record for record in cast(list[dict], document["folds"])}
    fold_blocks = []
    for partition in partitions:
        directory = stats_dir(root, partition.fold)
        stats = load_fold_stats(partition, directory, frames_path, input_sha256)
        snapshot.update(snapshot_inputs([directory / name for name in (POSE_STATS_FILE, DINOV3_STATS_FILE,
                                                                         PROVENANCE_FILE)]))
        retained_windows = len(build_window_index(partition.side_frames(RETAINED_SPLIT), stride=EVAL_STRIDE))
        runs = []
        for arm in ARMS:
            for seed in SEED_SETS[SEED_SET]:
                run_dir = fold_run_dir(root, partition.fold, arm, seed)
                snapshot.update(snapshot_inputs([run_dir / name for name in (*RUN_ARTIFACTS, FOLD_RUN_FILE)]))
                runs.append(
                    run_audit_entry(root, partition, records[partition.fold], stats, quality, cached, arm, seed,
                                    device)
                )
        fold_blocks.append(
            {
                "fold": partition.fold,
                "retained_environment": partition.retained_environment,
                "development_environments": list(partition.development_environments),
                "alignment": {
                    "retained_labeled_windows": retained_windows,
                    "recorded_retained_windows": records[partition.fold]["retained_windows_eval_stride"],
                    "matches_folds_json": retained_windows == records[partition.fold]["retained_windows_eval_stride"],
                },
                "normalization": fold_normalization(partition, stats, features),
                "runs": runs,
            }
        )
    snapshot.update(snapshot_inputs([root / FOLDS_FILE]))

    report: dict = {
        "analysis": "internal_fold_retained_audit",
        "canonical": False,
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "seed_set": SEED_SET,
        "seeds": list(SEED_SETS[SEED_SET]),
        "device": device,
        "descriptive_thresholds": {"extreme_abs_z": EXTREME_Z, "mostly_walk_fraction": MOSTLY_WALK_FRACTION},
        "integrity": {"videos": integrity, "environments": environment_integrity(integrity)},
        "label_composition": label_composition(frames),
        "dinov3_geometry": dinov3_geometry(features, environments),
        "folds": fold_blocks,
    }
    report["findings"] = {
        "verified_defects": verified_defects(report),
        "descriptive_shifts": descriptive_shifts(report),
        "prediction_observations": prediction_observations(report),
        "unresolved_hypotheses": UNRESOLVED_HYPOTHESES,
    }
    report["limitations"] = LIMITATIONS
    report["input_sha256"] = {display_path(Path(path)): digest for path, digest in sorted(snapshot.items())}
    return report, snapshot


def _bullets(items: list[str], empty: str) -> list[str]:
    return [f"- {item}" for item in items] or [f"- {empty}"]


def render_summary(report: dict) -> str:
    findings = report["findings"]
    lines = [
        "# Auditoria dos ambientes retidos das dobras internas (Etapa 2E)",
        "",
        "Exploratória, não canônica e somente leitura. Só o `train` oficial do Le2i-CV foi lido "
        f"(`official_splits_read: {report['official_splits_read']}`), com os runs da seed "
        f"{', '.join(str(seed) for seed in report['seeds'])}.",
        "",
        "## Defeitos técnicos verificados",
        "",
        *_bullets(findings["verified_defects"], "Nenhum defeito de integridade, alinhamento ou reconciliação."),
        "",
        "## Deslocamentos descritivos",
        "",
        *_bullets(findings["descriptive_shifts"], "Nenhum."),
        "",
        "## Predições por vídeo no ambiente retido",
        "",
        *_bullets(findings["prediction_observations"], "Nenhuma."),
        "",
        "| Dobra | Arma | Reconcilia com `metrics.json` | Janelas rotuladas | Fração `walk` (janelas) | "
        "Fração `walk` por vídeo: mediana (mín–máx) | Vídeos ≥ limiar |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for fold in report["folds"]:
        for run in fold["runs"]:
            summary = run["summary"]
            per_video = summary["video_predicted_walk_fraction"]
            lines.append(
                f"| `{fold['fold']}` | {run['arm']} | "
                f"{'sim' if run['reconciliation']['matches_metrics_json'] else 'não'} | "
                f"{summary['labeled_windows']} | {_fmt(summary['window_predicted_walk_fraction'])} | "
                f"{_fmt(per_video['median'])} ({_fmt(per_video['min'])}–{_fmt(per_video['max'])}) | "
                f"{summary['videos_mostly_walk']}/{per_video['n_videos']} |"
            )
    lines += [
        "",
        "## Padronização por dobra",
        "",
        "RMS do z médio por vídeo: mediana (mínimo–máximo) entre vídeos de cada lado.",
        "",
        "| Dobra | Lado | Vídeos | Pose | DINOv3 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for fold in report["folds"]:
        for side in SIDES:
            block = fold["normalization"]["sides"][side]
            pose, visual = block["pose"]["mean_z_rms"], block["dinov3"]["mean_z_rms"]
            lines.append(
                f"| `{fold['fold']}` | {side} | {pose['n_videos']} | "
                f"{_fmt(pose['median'])} ({_fmt(pose['min'])}–{_fmt(pose['max'])}) | "
                f"{_fmt(visual['median'])} ({_fmt(visual['min'])}–{_fmt(visual['max'])}) |"
            )
    lines += [
        "",
        "## Integridade por ambiente",
        "",
        "| Ambiente | Vídeos | Com defeito | Conf. keypoints | Norma DINOv3 | Desvio temporal DINOv3 | "
        + " | ".join(f"`{name}`" for name in QUALITY_CHANNEL_NAMES)
        + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in QUALITY_CHANNEL_NAMES) + " |",
    ]
    for env, block in report["integrity"]["environments"].items():
        medians = " | ".join(_fmt(block[name]["median"]) for name in INTEGRITY_DESCRIPTORS)
        lines.append(f"| `{env}` | {block['n_videos']} | {block['videos_with_defects']} | {medians} |")
    lines += [
        "",
        "Valores de integridade são medianas por vídeo.",
        "",
        "## Hipóteses não resolvidas",
        "",
        *_bullets(findings["unresolved_hypotheses"], "Nenhuma."),
        "",
        "## Limitações",
        "",
        *_bullets(report["limitations"], "Nenhuma."),
    ]
    return "\n".join(lines) + "\n"


def run_audit(
    root: Path,
    frames_path: Path,
    manifest_path: Path,
    output_dir: Path,
    force: bool = False,
    loaders: FeatureLoaders | None = None,
    quality: QualityFeatures | None = None,
    device: str | None = None,
) -> bool:
    check_output_dir(output_dir, [*protected_paths(frames_path, manifest_path), root])
    targets = [output_dir / AUDIT_JSON_FILE, output_dir / AUDIT_SUMMARY_FILE]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not force:
        print(f"skip {', '.join(existing)} (já existe, use --force para sobrescrever)")
        return False
    report, snapshot = build_report(root, frames_path, manifest_path, loaders, quality, device)
    written = write_text_outputs(
        output_dir,
        {
            targets[0]: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            targets[1]: render_summary(report),
        },
        force,
    )
    verify_inputs_unchanged(snapshot)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapter = Le2iDatasetAdapter(protocol="cv")
    audit_parser = subparsers.add_parser(
        "audit", help="Audita entradas, padronização e predições dos ambientes retidos das dobras internas"
    )
    audit_parser.add_argument("--fold-root", type=Path, default=FOLD_RUN_ROOT)
    audit_parser.add_argument("--frames", type=Path, default=adapter.frames_path)
    audit_parser.add_argument("--manifest", type=Path, default=adapter.manifest_path)
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da auditoria")

    args = parser.parse_args()
    if args.command == "audit":
        run_audit(args.fold_root, args.frames, args.manifest, args.output_dir, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.internal_fold_retained_audit import run_selftest

        if not run_selftest():
            sys.exit(1)


if __name__ == "__main__":
    main()
