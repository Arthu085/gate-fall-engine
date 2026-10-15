"""Contrafactual de inferência com a entrada DINOv3 neutralizada nas dobras internas (Etapa 2F).

Mede quão sensíveis são as predições congeladas de B0/B1 nos ambientes
retidos do piloto da Etapa 2D a uma única intervenção fixa: o tensor DINOv3
padronizado com as estatísticas da dobra é substituído por zeros, isto é,
pela média visual do treino de desenvolvimento da dobra em coordenadas
padronizadas. Pose, proxies de qualidade, pesos, checkpoints, rótulos e o
restante da inferência ficam iguais. A, só pose, entra sem intervenção como
referência.

Reusa a composição e a proveniência das dobras, os loaders de feature, a
validação de runs e os preditores dos avaliadores de eventos, e só lê linhas
`split == train` do Le2i-CV. As predições originais precisam reproduzir a
matriz de confusão e a Macro-F1 restrita de `final.val` de cada
`metrics.json` antes de qualquer contrafactual. O resultado é sensibilidade
observada à intervenção, não prova de causa nem melhoria validada.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, NUM_CLASSES
from gatefall.data.fusion_dataset import FusionWindowDataset
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES, Le2iDatasetAdapter
from gatefall.eval.analysis.classification_diagnosis import (
    check_output_dir,
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.internal_fold_feasibility import OFFICIAL_TRAIN_SPLIT
from gatefall.eval.analysis.internal_fold_retained_audit import (
    MOSTLY_WALK_FRACTION,
    SEED_SET,
    WALK_LABEL,
    confusion_matrix,
    predict_retained as predict_retained_pose,
    prediction_summary,
    reconcile,
    spread,
    video_predictions,
)
from gatefall.eval.analysis.stratified_error_diagnosis import align_predictions, window_metadata
from gatefall.eval.baseline_b0.cli import FusionModel
from gatefall.eval.baseline_b0.cli import _predict_with_identity as predict_fusion
from gatefall.eval.baseline_b1.cli import GatedFusionModel
from gatefall.eval.baseline_b1.cli import _predict_with_identity as predict_gated_fusion
from gatefall.eval.shared.orchestration import Predictions
from gatefall.train.baseline_a.config import TrainConfig
from gatefall.train.baseline_b0.artifacts import load_compatible_b0_checkpoint
from gatefall.train.baseline_b0.config import B0TrainConfig
from gatefall.train.baseline_b1.artifacts import load_compatible_b1_checkpoint
from gatefall.train.internal_folds.cli import _official_train, feature_loaders, validate_feature_files
from gatefall.train.internal_folds.folds import (
    ARMS,
    DATASET,
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
from gatefall.train.shared.metrics import restricted_macro_f1

REPORT_JSON_FILE = "visual_neutralization_counterfactual.json"
REPORT_SUMMARY_FILE = "visual_neutralization_counterfactual.md"
REPORT_VIDEOS_FILE = "visual_neutralization_counterfactual_videos.csv"
INTERVENTION = "dinov3_standardized_zero"
INTERVENED_ARMS = ("B0", "B1")
REFERENCE_ARM = "A"
MACRO_F1_TOLERANCE = 1e-12
MODALITIES = ("pose", "dinov3", "quality")
VIDEO_COLUMNS = [
    "fold",
    "retained_environment",
    "arm",
    "seed",
    "video_id",
    "subject",
    "labeled_windows",
    "original_predicted_walk_fraction",
    "counterfactual_predicted_walk_fraction",
    "original_accuracy",
    "counterfactual_accuracy",
    "changed_fraction",
]

INTERPRETATION_LIMITS = [
    "Sensibilidade observada não é causa. A intervenção mostra como as predições respondem quando a entrada "
    "visual vira a média do treino da dobra; ela não estabelece que o colapso vem de um atalho de cena nem de "
    "mudança de domínio. Essas leituras, e outras, preveem alguma sensibilidade à entrada visual.",
    "A neutralização cria uma entrada fora da distribuição de treino: um vetor visual constante, igual à média "
    "do desenvolvimento, em todos os passos da janela nunca ocorreu no treino. A resposta do modelo pode refletir "
    "essa entrada inédita, e não a remoção de informação de cena.",
    "Zero em coordenadas padronizadas é a média do treino de desenvolvimento da dobra, não uma cena neutra. Na "
    "dobra `retain_Lecture_room`, essa média vem só das duas salas de café.",
    "Em B1, o gate continua recebendo o `q_visual` original do vídeo, que pode não combinar com a entrada visual "
    "neutralizada.",
    "Não é uma melhoria validada: não há seleção, ajuste nem avaliação em val ou test oficiais, e uma Macro-F1 "
    "maior sob neutralização não torna a intervenção um remédio. A, só pose, já é a referência sem entrada visual.",
]

LIMITATIONS = [
    "Exploratório e não canônico: só o train oficial do Le2i-CV é lido; val e test oficiais não entram, e nada "
    "aqui explica a falha observada no test oficial.",
    "Uma intervenção fixa, definida antes da execução, e uma seed por arma (piloto, seed 42).",
    "Os resumos por vídeo usam mediana, mínimo e máximo; frações por janela são contexto, sobre janelas "
    "sobrepostas em stride 1.",
    "A reconciliação com metrics.json é necessária, mas não prova que cada janela recebeu o mesmo rótulo da "
    "avaliação histórica, que nunca foi persistida por janela.",
    "Cada dobra tem um ambiente retido; três ambientes não sustentam generalização sobre ambientes.",
]


class NeutralizedVisualFusion:
    """Entrega ao modelo B0 a mesma pose e um tensor visual padronizado de zeros."""

    def __init__(self, model: FusionModel) -> None:
        self.model = model

    def __call__(self, pose: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        return self.model(pose, torch.zeros_like(visual))


class NeutralizedVisualGatedFusion:
    """Entrega ao modelo B1 a mesma pose, a mesma qualidade e um tensor visual padronizado de zeros."""

    def __init__(self, model: GatedFusionModel) -> None:
        self.model = model

    def __call__(self, pose: torch.Tensor, visual: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        return self.model(pose, torch.zeros_like(visual), quality)


def predict_retained(
    partition: FoldPartition,
    config: FoldConfig,
    run_dir: Path,
    stats: FoldStats,
    loaders: FeatureLoaders,
    device: str,
    neutralize_visual: bool,
) -> Predictions:
    if isinstance(config, TrainConfig):
        if neutralize_visual:
            raise ValueError(f"{partition.fold}: a arma {REFERENCE_ARM} não tem entrada visual para neutralizar")
        return predict_retained_pose(partition, config, run_dir, stats, loaders, device)
    condition = "contrafactual" if neutralize_visual else "original"
    context = f"{partition.fold}/{config.arm}/{condition}"
    allowed = partition.retained_video_ids
    pose = restricted_loader(loaders.pose, allowed, f"{context}/pose")
    visual = restricted_loader(loaders.visual, allowed, f"{context}/dinov3")
    checkpoint = run_dir / "checkpoint.pt"
    if isinstance(config, B0TrainConfig):
        fusion = load_compatible_b0_checkpoint(checkpoint, config).to(device).eval()
        fusion_model: FusionModel = NeutralizedVisualFusion(fusion) if neutralize_visual else fusion
        fusion_source = FusionWindowDataset(
            partition.frames, RETAINED_SPLIT, EVAL_STRIDE, pose, visual, drop_ignored=False
        )
        return predict_fusion(fusion_model, fusion_source, stats.pose, stats.dinov3, device, config.batch_size)
    quality = restricted_loader(loaders.quality, allowed, f"{context}/qualidade")
    gated = load_compatible_b1_checkpoint(checkpoint, config).to(device).eval()
    gated_model: GatedFusionModel = NeutralizedVisualGatedFusion(gated) if neutralize_visual else gated
    gated_source = GatedFusionWindowDataset(
        partition.frames, RETAINED_SPLIT, EVAL_STRIDE, pose, visual, quality, drop_ignored=False
    )
    return predict_gated_fusion(gated_model, gated_source, stats.pose, stats.dinov3, device, config.batch_size)


def _labeled(aligned: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, aligned[aligned["label"] != IGNORE_LABEL])


def require_reconciled(aligned: pd.DataFrame, stored_split: dict, context: str) -> dict:
    reconciliation = reconcile(aligned, stored_split, context)
    labeled = _labeled(aligned)
    macro_f1, _f1_by_class = restricted_macro_f1(labeled["label"].to_numpy(), labeled["pred_label"].to_numpy())
    stored_macro_f1 = float(stored_split["macro_f1_restricted"])
    reconciliation["macro_f1_restricted"] = macro_f1
    reconciliation["macro_f1_matches_metrics_json"] = math.isclose(
        macro_f1, stored_macro_f1, rel_tol=0.0, abs_tol=MACRO_F1_TOLERANCE
    )
    if not (reconciliation["matches_metrics_json"] and reconciliation["macro_f1_matches_metrics_json"]):
        raise RuntimeError(
            f"{context}: as predições originais não reproduzem final.val de metrics.json "
            f"({reconciliation['differing_cells']} célula(s) divergentes; Macro-F1 restrita {macro_f1} contra "
            f"{stored_macro_f1}); o contrafactual não é calculado"
        )
    return reconciliation


def condition_metrics(aligned: pd.DataFrame) -> dict:
    labeled = _labeled(aligned)
    labels, predictions = labeled["label"].to_numpy(), labeled["pred_label"].to_numpy()
    matrix = np.asarray(confusion_matrix(labels, predictions))
    support = matrix.sum(axis=1)
    macro_f1, _f1_by_class = restricted_macro_f1(labels, predictions)
    return {
        **prediction_summary(video_predictions(aligned)),
        "macro_f1_restricted": macro_f1,
        "recall_by_class": {
            name: float(matrix[index, index] / support[index]) if support[index] else None
            for index, name in enumerate(LE2I_LABEL_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
    }


def paired_windows(original: pd.DataFrame, counterfactual: pd.DataFrame, context: str) -> pd.DataFrame:
    labeled = _labeled(original)[["video_id", "subject", "k_end", "label", "pred_label"]]
    paired = labeled.merge(
        counterfactual[["video_id", "k_end", "label", "pred_label"]],
        on=["video_id", "k_end"],
        how="left",
        suffixes=("", "_counterfactual"),
        validate="one_to_one",
    )
    counterfactual_predictions = paired["pred_label_counterfactual"].to_numpy()
    if pd.isna(counterfactual_predictions).any() or not np.array_equal(
        paired["label"].to_numpy(), paired["label_counterfactual"].to_numpy()
    ):
        raise ValueError(f"{context}: janelas rotuladas do contrafactual não pareiam com as originais")
    return pd.DataFrame(
        {
            "video_id": paired["video_id"].to_numpy(),
            "subject": paired["subject"].to_numpy(),
            "k_end": paired["k_end"].to_numpy(),
            "label": paired["label"].to_numpy(),
            "original": paired["pred_label"].to_numpy(),
            "counterfactual": counterfactual_predictions.astype(np.int64),
        }
    )


def paired_changes(paired: pd.DataFrame) -> dict:
    labels = paired["label"].to_numpy()
    original = paired["original"].to_numpy()
    counterfactual = paired["counterfactual"].to_numpy()
    changed = original != counterfactual
    original_correct = original == labels
    counterfactual_correct = counterfactual == labels
    transitions = Counter(zip(original[changed].tolist(), counterfactual[changed].tolist()))
    video_ids = paired["video_id"].to_numpy()
    video_fractions: list[float | None] = [
        float(changed[video_ids == video_id].mean()) for video_id in sorted(set(video_ids.tolist()))
    ]
    return {
        "labeled_windows": len(paired),
        "changed_windows": int(changed.sum()),
        "changed_fraction": float(changed.mean()) if len(paired) else None,
        "became_correct": int((~original_correct & counterfactual_correct).sum()),
        "became_incorrect": int((original_correct & ~counterfactual_correct).sum()),
        "video_changed_fraction": spread(video_fractions),
        "transitions": [
            {"from": LE2I_LABEL_NAMES[source], "to": LE2I_LABEL_NAMES[target], "windows": windows}
            for (source, target), windows in sorted(transitions.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def _predicted(predictions: np.ndarray) -> dict[str, int]:
    counts = np.bincount(predictions, minlength=NUM_CLASSES)
    return {LE2I_LABEL_NAMES[index]: int(count) for index, count in enumerate(counts) if count}


def paired_videos(paired: pd.DataFrame) -> list[dict]:
    walk_id = LE2I_LABEL_NAMES.index(WALK_LABEL)
    videos = []
    for video_id, group in paired.groupby("video_id"):
        labels = group["label"].to_numpy()
        original = group["original"].to_numpy()
        counterfactual = group["counterfactual"].to_numpy()
        videos.append(
            {
                "video_id": str(video_id),
                "subject": int(group["subject"].iloc[0]),
                "labeled_windows": len(group),
                "original_predicted_walk_fraction": float(np.mean(original == walk_id)),
                "counterfactual_predicted_walk_fraction": float(np.mean(counterfactual == walk_id)),
                "original_accuracy": float(np.mean(original == labels)),
                "counterfactual_accuracy": float(np.mean(counterfactual == labels)),
                "changed_fraction": float(np.mean(original != counterfactual)),
                "original_predicted": _predicted(original),
                "counterfactual_predicted": _predicted(counterfactual),
            }
        )
    return videos


def run_block(
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
    windows = window_metadata(partition.frames, RETAINED_SPLIT, context)
    with (run_dir / "metrics.json").open(encoding="utf-8") as stream:
        stored = json.load(stream)["final"]["val"]
    original = align_predictions(
        windows, predict_retained(partition, config, run_dir, stats, loaders, device, False), context
    )
    block: dict = {
        "arm": arm,
        "seed": seed,
        "run_dir": entry["run_dir"],
        "reconciliation": require_reconciled(original, stored, context),
        "original": condition_metrics(original),
    }
    if arm not in INTERVENED_ARMS:
        block["videos"] = video_predictions(original)
        return block
    counterfactual_context = f"{context}/contrafactual"
    counterfactual = align_predictions(
        windows, predict_retained(partition, config, run_dir, stats, loaders, device, True), counterfactual_context
    )
    paired = paired_windows(original, counterfactual, counterfactual_context)
    block["counterfactual"] = condition_metrics(counterfactual)
    block["paired"] = paired_changes(paired)
    block["videos"] = paired_videos(paired)
    return block


def feature_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256(f"{array.dtype.str}{array.shape}".encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_retained_features(
    partition: FoldPartition, loaders: FeatureLoaders
) -> tuple[FeatureLoaders, dict[str, dict[str, str]]]:
    """Carrega uma vez as features dos vídeos retidos da dobra e registra o SHA-256 de cada array.

    Os loaders devolvidos só conhecem os vídeos retidos da dobra; qualquer outro
    vídeo falha.
    """
    allowed = partition.retained_video_ids
    sources = {
        "pose": restricted_loader(loaders.pose, allowed, f"{partition.fold}/pose"),
        "dinov3": restricted_loader(loaders.visual, allowed, f"{partition.fold}/dinov3"),
        "quality": restricted_loader(loaders.quality, allowed, f"{partition.fold}/qualidade"),
    }
    arrays = {video_id: {name: sources[name](video_id) for name in MODALITIES} for video_id in sorted(allowed)}
    digests = {
        video_id: {name: feature_digest(array) for name, array in modalities.items()}
        for video_id, modalities in arrays.items()
    }
    cached = FeatureLoaders(
        pose=lambda video_id: arrays[video_id]["pose"],
        visual=lambda video_id: arrays[video_id]["dinov3"],
        quality=lambda video_id: arrays[video_id]["quality"],
    )
    return cached, digests


def verify_features_unchanged(loaders: FeatureLoaders, digests: dict[str, dict[str, str]]) -> None:
    sources = {"pose": loaders.pose, "dinov3": loaders.visual, "quality": loaders.quality}
    changed = [
        f"{video_id}:{name}"
        for video_id, modalities in sorted(digests.items())
        for name, digest in modalities.items()
        if feature_digest(sources[name](video_id)) != digest
    ]
    if changed:
        raise RuntimeError(f"features mudaram durante o contrafactual: {changed[:5]}")


def _fmt(value: object, digits: int = 3) -> str:
    return "n/d" if value is None else f"{cast(float, value):.{digits}f}"


def _spread_text(block: dict) -> str:
    return f"{_fmt(block['median'])} ({_fmt(block['min'])}–{_fmt(block['max'])})"


def reconciliation_findings(report: dict) -> list[str]:
    runs = [run for fold in report["folds"] for run in (fold["reference"], *fold["runs"])]
    return [
        f"As {len(runs)} reinferências originais reproduzem a matriz de confusão e a Macro-F1 restrita de "
        "final.val de cada metrics.json; o contrafactual só é calculado depois dessa reconciliação."
    ]


def sensitivity_observations(report: dict) -> list[str]:
    observations = []
    for fold in report["folds"]:
        reference = fold["reference"]["original"]
        observations.append(
            f"`{fold['fold']}` / {REFERENCE_ARM} (referência só pose, sem intervenção): "
            f"{_fmt(reference['window_predicted_walk_fraction'])} das janelas preditas como `{WALK_LABEL}`, "
            f"Macro-F1 restrita {_fmt(reference['macro_f1_restricted'])}."
        )
        for run in fold["runs"]:
            original, counterfactual, paired = run["original"], run["counterfactual"], run["paired"]
            n_videos = original["video_predicted_walk_fraction"]["n_videos"]
            observations.append(
                f"`{fold['fold']}` / {run['arm']}: fração `{WALK_LABEL}` por janela "
                f"{_fmt(original['window_predicted_walk_fraction'])} → "
                f"{_fmt(counterfactual['window_predicted_walk_fraction'])}; por vídeo, mediana "
                f"{_fmt(original['video_predicted_walk_fraction']['median'])} → "
                f"{_fmt(counterfactual['video_predicted_walk_fraction']['median'])}; vídeos com fração ≥ "
                f"{MOSTLY_WALK_FRACTION:g}: {original['videos_mostly_walk']}/{n_videos} → "
                f"{counterfactual['videos_mostly_walk']}/{n_videos}; Macro-F1 restrita "
                f"{_fmt(original['macro_f1_restricted'])} → {_fmt(counterfactual['macro_f1_restricted'])}; "
                f"{_fmt(paired['changed_fraction'])} das janelas mudam de predição ({paired['became_correct']} "
                f"passam a acertar, {paired['became_incorrect']} passam a errar)."
            )
    return observations


def build_report(
    root: Path,
    frames_path: Path,
    manifest_path: Path,
    loaders: FeatureLoaders | None = None,
    quality: QualityFeatures | None = None,
    device: str | None = None,
) -> tuple[dict, dict[str, str], dict[str, dict[str, str]]]:
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

    records = {cast(str, record["fold"]): record for record in cast(list[dict], document["folds"])}
    digests: dict[str, dict[str, str]] = {}
    fold_blocks = []
    for partition in partitions:
        directory = stats_dir(root, partition.fold)
        stats = load_fold_stats(partition, directory, frames_path, input_sha256)
        snapshot.update(snapshot_inputs([directory / name for name in (POSE_STATS_FILE, DINOV3_STATS_FILE,
                                                                         PROVENANCE_FILE)]))
        cached, fold_digests = load_retained_features(partition, loaders)
        digests.update(fold_digests)
        blocks = {}
        for arm in ARMS:
            for seed in SEED_SETS[SEED_SET]:
                run_dir = fold_run_dir(root, partition.fold, arm, seed)
                snapshot.update(snapshot_inputs([run_dir / name for name in (*RUN_ARTIFACTS, FOLD_RUN_FILE)]))
                blocks[arm] = run_block(root, partition, records[partition.fold], stats, quality, cached, arm,
                                        seed, device)
        fold_blocks.append(
            {
                "fold": partition.fold,
                "retained_environment": partition.retained_environment,
                "development_environments": list(partition.development_environments),
                "retained_videos": sorted(partition.retained_video_ids),
                "reference": blocks[REFERENCE_ARM],
                "runs": [blocks[arm] for arm in INTERVENED_ARMS],
            }
        )
    snapshot.update(snapshot_inputs([root / FOLDS_FILE]))

    report: dict = {
        "analysis": "visual_neutralization_counterfactual",
        "canonical": False,
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "seed_set": SEED_SET,
        "seeds": list(SEED_SETS[SEED_SET]),
        "device": device,
        "intervention": {
            "name": INTERVENTION,
            "arms": list(INTERVENED_ARMS),
            "reference_arm": REFERENCE_ARM,
            "standardized_dinov3_value": 0.0,
            "unchanged_inputs": ["pose", "quality"],
        },
        "descriptive_thresholds": {"mostly_walk_fraction": MOSTLY_WALK_FRACTION},
        "folds": fold_blocks,
    }
    report["findings"] = {
        "reconciliation": reconciliation_findings(report),
        "observed_sensitivity": sensitivity_observations(report),
        "interpretation_limits": INTERPRETATION_LIMITS,
    }
    report["limitations"] = LIMITATIONS
    report["input_sha256"] = {display_path(Path(path)): digest for path, digest in sorted(snapshot.items())}
    report["feature_sha256"] = {video_id: digests[video_id] for video_id in sorted(digests)}
    return report, snapshot, digests


def video_rows(report: dict) -> list[dict]:
    rows = []
    for fold in report["folds"]:
        for run in (fold["reference"], *fold["runs"]):
            for video in run["videos"]:
                intervened = "counterfactual" in run
                rows.append(
                    {
                        "fold": fold["fold"],
                        "retained_environment": fold["retained_environment"],
                        "arm": run["arm"],
                        "seed": run["seed"],
                        "video_id": video["video_id"],
                        "subject": video["subject"],
                        "labeled_windows": video["labeled_windows"],
                        "original_predicted_walk_fraction": (
                            video["original_predicted_walk_fraction"] if intervened
                            else video["predicted_walk_fraction"]
                        ),
                        "counterfactual_predicted_walk_fraction": (
                            video["counterfactual_predicted_walk_fraction"] if intervened else None
                        ),
                        "original_accuracy": video["original_accuracy"] if intervened else video["accuracy"],
                        "counterfactual_accuracy": video["counterfactual_accuracy"] if intervened else None,
                        "changed_fraction": video["changed_fraction"] if intervened else None,
                    }
                )
    return rows


def _bullets(items: list[str], empty: str) -> list[str]:
    return [f"- {item}" for item in items] or [f"- {empty}"]


def render_summary(report: dict) -> str:
    findings = report["findings"]
    lines = [
        "# Contrafactual de neutralização visual nas dobras internas (Etapa 2F)",
        "",
        "Exploratório, não canônico e somente leitura. Só o `train` oficial do Le2i-CV foi lido "
        f"(`official_splits_read: {report['official_splits_read']}`), com os runs da seed "
        f"{', '.join(str(seed) for seed in report['seeds'])}. Intervenção: o tensor DINOv3 padronizado de B0 e B1 "
        "vira zeros (média do treino de desenvolvimento da dobra); pose, qualidade, pesos e rótulos não mudam. "
        f"{REFERENCE_ARM} entra sem intervenção como referência.",
        "",
        "## Reconciliação das predições originais",
        "",
        *_bullets(findings["reconciliation"], "Nenhuma."),
        "",
        "## Sensibilidade observada",
        "",
        *_bullets(findings["observed_sensitivity"], "Nenhuma."),
        "",
        f"| Dobra | Arma | Fração `{WALK_LABEL}` (janelas) | Fração `{WALK_LABEL}` por vídeo: mediana (mín–máx) | "
        f"Vídeos ≥ {MOSTLY_WALK_FRACTION:g} | Macro-F1 restrita | Janelas alteradas |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for fold in report["folds"]:
        reference = fold["reference"]["original"]
        n_videos = reference["video_predicted_walk_fraction"]["n_videos"]
        lines.append(
            f"| `{fold['fold']}` | {REFERENCE_ARM} (referência) | "
            f"{_fmt(reference['window_predicted_walk_fraction'])} | "
            f"{_spread_text(reference['video_predicted_walk_fraction'])} | "
            f"{reference['videos_mostly_walk']}/{n_videos} | {_fmt(reference['macro_f1_restricted'])} | — |"
        )
        for run in fold["runs"]:
            original, counterfactual = run["original"], run["counterfactual"]
            lines.append(
                f"| `{fold['fold']}` | {run['arm']} | {_fmt(original['window_predicted_walk_fraction'])} → "
                f"{_fmt(counterfactual['window_predicted_walk_fraction'])} | "
                f"{_spread_text(original['video_predicted_walk_fraction'])} → "
                f"{_spread_text(counterfactual['video_predicted_walk_fraction'])} | "
                f"{original['videos_mostly_walk']}/{n_videos} → {counterfactual['videos_mostly_walk']}/{n_videos} | "
                f"{_fmt(original['macro_f1_restricted'])} → {_fmt(counterfactual['macro_f1_restricted'])} | "
                f"{_fmt(run['paired']['changed_fraction'])} |"
            )
    lines += [
        "",
        "## Revocação por classe no ambiente retido",
        "",
        "Original → contrafactual; `n/d` quando a classe não tem suporte no retido.",
        "",
        "| Dobra | Arma | " + " | ".join(f"`{name}`" for name in LE2I_LABEL_NAMES) + " |",
        "| --- | --- | " + " | ".join("---" for _ in LE2I_LABEL_NAMES) + " |",
    ]
    for fold in report["folds"]:
        reference = fold["reference"]["original"]["recall_by_class"]
        lines.append(
            f"| `{fold['fold']}` | {REFERENCE_ARM} (referência) | "
            + " | ".join(_fmt(reference[name]) for name in LE2I_LABEL_NAMES)
            + " |"
        )
        for run in fold["runs"]:
            original, counterfactual = run["original"]["recall_by_class"], run["counterfactual"]["recall_by_class"]
            cells = [
                "n/d" if original[name] is None else f"{_fmt(original[name])} → {_fmt(counterfactual[name])}"
                for name in LE2I_LABEL_NAMES
            ]
            lines.append(f"| `{fold['fold']}` | {run['arm']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Classes preditas",
        "",
        "Janelas rotuladas por classe predita, original → contrafactual.",
        "",
    ]
    for fold in report["folds"]:
        for run in fold["runs"]:
            original = run["original"]["predicted_class_windows"]
            counterfactual = run["counterfactual"]["predicted_class_windows"]
            names = [name for name in LE2I_LABEL_NAMES if name in original or name in counterfactual]
            counts = ", ".join(f"`{name}` {original.get(name, 0)} → {counterfactual.get(name, 0)}" for name in names)
            transitions = ", ".join(
                f"`{item['from']}` → `{item['to']}` {item['windows']}" for item in run["paired"]["transitions"][:5]
            )
            lines.append(
                f"- `{fold['fold']}` / {run['arm']}: {counts}. Transições mais frequentes: {transitions or 'nenhuma'}."
            )
    lines += [
        "",
        "## O que isto não estabelece",
        "",
        *_bullets(findings["interpretation_limits"], "Nenhum."),
        "",
        "## Limitações",
        "",
        *_bullets(report["limitations"], "Nenhuma."),
    ]
    return "\n".join(lines) + "\n"


def run_analysis(
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
    targets = [output_dir / REPORT_JSON_FILE, output_dir / REPORT_SUMMARY_FILE, output_dir / REPORT_VIDEOS_FILE]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not force:
        print(f"skip {', '.join(existing)} (já existe, use --force para sobrescrever)")
        return False
    loaders = loaders or feature_loaders()
    report, snapshot, digests = build_report(root, frames_path, manifest_path, loaders, quality, device)
    outputs = {
        targets[0]: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        targets[1]: render_summary(report),
        targets[2]: pd.DataFrame(video_rows(report), columns=VIDEO_COLUMNS).to_csv(index=False),
    }
    verify_features_unchanged(loaders, digests)
    verify_inputs_unchanged(snapshot)
    return write_text_outputs(output_dir, outputs, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapter = Le2iDatasetAdapter(protocol="cv")
    analyze_parser = subparsers.add_parser(
        "analyze", help="Compara predições originais e com DINOv3 neutralizado nos ambientes retidos das dobras"
    )
    analyze_parser.add_argument("--fold-root", type=Path, default=FOLD_RUN_ROOT)
    analyze_parser.add_argument("--frames", type=Path, default=adapter.frames_path)
    analyze_parser.add_argument("--manifest", type=Path, default=adapter.manifest_path)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do contrafactual")

    args = parser.parse_args()
    if args.command == "analyze":
        run_analysis(args.fold_root, args.frames, args.manifest, args.output_dir, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.visual_neutralization_counterfactual import run_selftest

        if not run_selftest():
            sys.exit(1)


if __name__ == "__main__":
    main()
