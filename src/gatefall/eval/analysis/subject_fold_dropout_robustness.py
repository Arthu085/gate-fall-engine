"""Robustez de inferência à perda simulada de detecção nas dobras por sujeito do Le2i-CS (Etapa 4D).

Diagnóstico somente leitura e não canônico. Reavalia os seis checkpoints
congelados do braço A, seed 42 — `baseline` (Etapa 4B) e
`detection_dropout` (Etapa 4C) de cada dobra — no grupo retido da própria
dobra, em duas condições: entrada limpa e uma única condição predeclarada de
perda de detecção. Nada é treinado, ajustado ou selecionado.

A condição degradada reusa as funções causais validadas da Etapa 4C: com
probabilidade `DROPOUT_PROBABILITY` por janela retida, `MASKED_FRAMES`
quadros reais consecutivos viram detecções ausentes numa cópia de
`PoseArrays`, e as 134 features são recalculadas pelo pipeline inalterado
sobre o prefixo causal do vídeo. A máscara de cada janela vem só da sua
identidade (`video_id`, `k_end`) e de uma seed de avaliação fixa, e é
materializada uma vez por dobra antes de qualquer modelo: os dois checkpoints
da dobra recebem exatamente os mesmos rótulos, janelas e entradas degradadas.

Antes de relatar qualquer número, o diagnóstico valida a partição oficial do
`train`, `folds.json`, os hashes da pose, a estatística de cada dobra, a
proveniência e o checkpoint dos dois runs, e exige que a avaliação limpa
reproduza exatamente a matriz de confusão, o suporte e a Macro-F1 restrita
gravados em `final.val`. As dobras nunca são agregadas e não há teste de
significância.
"""

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from gatefall.config import EVAL_STRIDE, NUM_CLASSES, WINDOW_FRAMES
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import window_frame_indices
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import (
    check_output_dir,
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.internal_fold_feasibility import OFFICIAL_TRAIN_SPLIT
from gatefall.features.standardization import StandardizationStats, apply_standardization
from gatefall.pose.kinematics import build_pose_features_from_arrays
from gatefall.pose.loading import PoseArrays
from gatefall.train.baseline_a.artifacts import REQUIRED_TRAINING_ARTIFACTS, load_compatible_checkpoint
from gatefall.train.internal_folds.folds import RETAINED_SPLIT, restricted_loader
from gatefall.train.internal_folds.training import FOLD_RUN_FILE
from gatefall.train.shared.determinism import configure_determinism
from gatefall.train.shared.metrics import RESTRICTED_CLASSES, confusion_matrix, restricted_macro_f1, support
from gatefall.train.shared.tcn import TCNClassifier
from gatefall.train.subject_folds.cli import (
    ADAPTER,
    PoseInputs,
    _official_train,
    missing_prerequisites,
    real_pose_inputs,
)
from gatefall.train.subject_folds.detection_dropout import (
    DROPOUT_PROBABILITY,
    MASKED_FRAMES,
    masked_window_features,
    valid_offsets,
)
from gatefall.train.subject_folds.folds import (
    BASELINE_VARIANT,
    DATASET,
    DETECTION_DROPOUT_VARIANT,
    FOLDS_FILE,
    PROTOCOL,
    RESTRICTED_LABELS,
    SUBJECT_FOLD_ROOT,
    SubjectFoldPartition,
    build_subject_fold_partitions,
    fold_dir,
    fold_names,
    fold_run_dir,
    protected_paths,
    read_folds_document,
    stats_dir,
    summary_dir,
    validate_folds_document,
    validate_output_root,
)
from gatefall.train.subject_folds.stats import POSE_STATS_FILE, PROVENANCE_FILE, load_fold_stats
from gatefall.train.subject_folds.summary import fold_limitations, run_entry
from gatefall.train.subject_folds.training import resolve_fold_config

STAGE = "4D"
ANALYSIS = "subject_fold_detection_dropout_robustness"
ARM = "A"
SEED = 42
MODEL_VARIANTS: tuple[str, ...] = (BASELINE_VARIANT, DETECTION_DROPOUT_VARIANT)
MODEL_STAGES = {BASELINE_VARIANT: "4B", DETECTION_DROPOUT_VARIANT: "4C"}
EVALUATION_STREAM_KEY = 0x4D
EVALUATION_SEED = 0
# Limiar só descritivo: abaixo dele a F1 do subconjunto afetado não é
# relatada; nada é filtrado nem selecionado por ele.
AFFECTED_MIN_CLASS_SUPPORT = 20
MACRO_F1_TOLERANCE = 1e-12
DIAGNOSTICS_DIR = "diagnostics"
OUTPUT_NAME = "detection_dropout_robustness"
REPORT_JSON_FILE = "subject_fold_dropout_robustness.json"
REPORT_SUMMARY_FILE = "subject_fold_dropout_robustness.md"
UNMASKED = -1

SIGN_CONVENTIONS = {
    "change": "degradado menos limpo; negativo indica queda com a perda de detecção",
    "degradation_loss": "limpo menos degradado; positivo indica perda de desempenho",
    "paired_loss_difference": (
        f"perda de {BASELINE_VARIANT} menos perda de {DETECTION_DROPOUT_VARIANT} na mesma dobra; positivo indica "
        f"que o modelo treinado com perda simulada de detecção perde menos"
    ),
}

LIMITATIONS = [
    "Exploratório, somente leitura e não canônico: só o train oficial do Le2i-CS é lido; val e test oficiais "
    "não entram, e nada aqui estima o desempenho no test oficial nem verifica o alvo de 0,800.",
    "Uma única condição degradada, predeclarada e sintética: lacunas de três quadros em 20% das janelas. Ela não "
    "reproduz a distribuição real das perdas de detecção, que se concentram em fall e em Home_01, e não mede "
    "robustez a lacunas mais longas, a detecções erradas ou a ruído de keypoints.",
    "O modelo detection_dropout foi treinado com a mesma família de perda simulada avaliada aqui, com outro "
    "fluxo aleatório. Uma perda menor nele mostra robustez à condição para a qual ele foi treinado, não "
    "robustez geral.",
    "Uma seed por dobra e uma única realização da máscara de avaliação: a variação entre seeds e entre "
    "máscaras não é estimada, e diferenças pequenas podem vir só delas. Não há teste de significância.",
    "Três dobras de tamanhos muito diferentes; os números não são agregados nem comparáveis entre dobras.",
    "Contagens por janela usam janelas sobrepostas (stride 1 no retido) e não são observações independentes; "
    "uma mesma lacuna de quadros pode aparecer em janelas vizinhas mascaradas de forma independente.",
    "O subconjunto afetado é descritivo: suas classes têm suporte desigual, e a F1 por classe só é relatada "
    f"com pelo menos {AFFECTED_MIN_CLASS_SUPPORT} janelas afetadas.",
    "Janelas que terminam antes do terceiro quadro do vídeo não têm três quadros reais distintos e ficam "
    "limpas, como na Etapa 4C.",
]


def condition_record() -> dict[str, object]:
    return {
        "name": "person_detection_dropout",
        "stage": STAGE,
        "canonical": False,
        "scope": "retained_evaluation",
        "probability": DROPOUT_PROBABILITY,
        "masked_frames": MASKED_FRAMES,
        "window_frames": WINDOW_FRAMES,
        "position": (
            "uniforme entre os deslocamentos da janela cujos quadros mascarados são quadros reais distintos do "
            "vídeo; janelas sem posição válida ficam limpas"
        ),
        "representation": (
            "cópia de PoseArrays com person_found falso e keypoints e bbox zerados nos quadros mascarados, antes da "
            "imputação, do dt efetivo e das 134 features (funções causais da Etapa 4C)"
        ),
        "causal_context": "features recalculadas sobre o prefixo do vídeo até o último quadro da janela",
        "random_stream": {
            "generator": "numpy.random.default_rng (PCG64)",
            "seed_sequence": ["EVALUATION_STREAM_KEY", "EVALUATION_SEED", "video_key", "k_end"],
            "video_key": "primeiros 8 bytes, big-endian, do SHA-256 UTF-8 de video_id",
            "evaluation_stream_key": EVALUATION_STREAM_KEY,
            "evaluation_seed": EVALUATION_SEED,
            "independent_of": ["modelo", "tamanho de batch", "ordem de iteração"],
        },
        "shared_between_models": "rótulos, janelas e entradas degradadas materializados uma vez por dobra",
    }


def video_key(video_id: str) -> int:
    return int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big")


def evaluation_offset(video_id: str, k_end: int) -> int | None:
    """Deslocamento da máscara de avaliação da janela, ou None quando ela fica limpa; depende só da identidade."""
    rng = np.random.default_rng([EVALUATION_STREAM_KEY, EVALUATION_SEED, video_key(video_id), k_end])
    if rng.random() >= DROPOUT_PROBABILITY:
        return None
    offsets = valid_offsets(k_end)
    if not offsets:
        return None
    return offsets[int(rng.integers(len(offsets)))]


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{contiguous.dtype.str}{contiguous.shape}".encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RetainedInputs:
    """Janelas retidas de uma dobra nas duas condições, compartilhadas pelos dois checkpoints."""

    video_ids: tuple[str, ...]
    subjects: np.ndarray
    k_ends: np.ndarray
    labels: np.ndarray
    offsets: np.ndarray
    clean: np.ndarray
    degraded: np.ndarray

    @property
    def masked(self) -> np.ndarray:
        return self.offsets != UNMASKED

    def digests(self) -> dict[str, str]:
        plan = "".join(
            f"{video_id} {k_end} {offset}\n" for video_id, k_end, offset in zip(self.video_ids, self.k_ends,
                                                                               self.offsets)
        )
        return {
            "mask_plan": hashlib.sha256(plan.encode("utf-8")).hexdigest(),
            "labels": array_sha256(self.labels),
            "clean_inputs": array_sha256(self.clean),
            "degraded_inputs": array_sha256(self.degraded),
        }


def _raw_pose_by_video(
    video_ids: tuple[str, ...],
    clean_loader: Callable[[str], np.ndarray],
    raw_loader: Callable[[str], PoseArrays],
    n_frames: dict[str, int],
    context: str,
) -> dict[str, PoseArrays]:
    poses: dict[str, PoseArrays] = {}
    for video_id in video_ids:
        pose = raw_loader(video_id)
        if pose.k != n_frames[video_id]:
            raise ValueError(f"{context}: {video_id}: pose bruta tem {pose.k} quadros, a grade temporal tem "
                             f"{n_frames[video_id]}")
        if not np.array_equal(build_pose_features_from_arrays(pose)[0], clean_loader(video_id)):
            raise ValueError(
                f"{context}: {video_id}: as features recalculadas da pose bruta divergem das features limpas; a "
                "condição degradada não seria comparável"
            )
        poses[video_id] = pose
    return poses


def build_retained_inputs(partition: SubjectFoldPartition, pose: PoseInputs) -> RetainedInputs:
    if pose.raw_loader is None:
        raise ValueError(f"{partition.fold}: a condição degradada exige a pose bruta dos HDF5")
    context = f"{partition.fold}/{RETAINED_SPLIT}"
    allowed = partition.retained_video_ids
    clean_loader = restricted_loader(pose.loader, allowed, context)
    raw_loader = restricted_loader(pose.raw_loader, allowed, f"{context}/pose bruta")
    source = PoseWindowDataset(partition.frames, RETAINED_SPLIT, EVAL_STRIDE, clean_loader)
    windows = partition.retained_windows
    if len(source) != len(windows):
        raise RuntimeError(f"{context}: janelas do retido divergem da composição da dobra")
    if not set(int(subject) for subject in windows["subject"]) <= set(partition.retained_subjects):
        raise RuntimeError(f"{context}: janela retida de sujeito fora do grupo retido")
    n_frames = {str(video_id): int(n) for video_id, n in zip(windows["video_id"], windows["n_frames"])}
    poses = _raw_pose_by_video(tuple(sorted(n_frames)), clean_loader, raw_loader, n_frames, context)

    count = len(source)
    clean = np.empty((count, WINDOW_FRAMES, source[0][0].shape[-1]), dtype=np.float32)
    degraded = np.empty_like(clean)
    labels = np.empty(count, dtype=np.int64)
    k_ends = np.empty(count, dtype=np.int64)
    offsets = np.full(count, UNMASKED, dtype=np.int64)
    video_ids: list[str] = []
    for index in range(count):
        window, label, (video_id, k_end) = source[index]
        clean[index] = window
        labels[index] = label
        k_ends[index] = k_end
        video_ids.append(video_id)
        offset = evaluation_offset(video_id, k_end)
        if offset is None:
            degraded[index] = window
            continue
        masked = masked_window_features(poses[video_id], k_end, offset)
        frames = window_frame_indices(k_end, n_frames[video_id])
        before_mask = frames < frames[offset]
        if not np.array_equal(masked[before_mask], window[before_mask]):
            raise RuntimeError(f"{context}: {video_id}@{k_end}: linhas anteriores à máscara divergem da janela limpa")
        degraded[index] = masked
        offsets[index] = offset
    identity = list(zip(video_ids, k_ends.tolist()))
    if identity != list(zip(windows["video_id"].astype(str), windows["k_end"].astype(int))):
        raise RuntimeError(f"{context}: ordem das janelas diverge do índice da dobra")
    return RetainedInputs(
        video_ids=tuple(video_ids),
        subjects=windows["subject"].to_numpy(dtype=np.int64),
        k_ends=k_ends,
        labels=labels,
        offsets=offsets,
        clean=clean,
        degraded=degraded,
    )


@torch.no_grad()
def predict(
    model: TCNClassifier, windows: np.ndarray, stats: StandardizationStats, device: str, batch_size: int
) -> np.ndarray:
    """Predições na mesma ordem e com o mesmo batch do `final.val` do engine."""
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        x = torch.from_numpy(apply_standardization(windows[start : start + batch_size], stats)).to(device)
        predictions.append(torch.argmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64)


def condition_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    macro_f1, f1_by_class = restricted_macro_f1(labels, predictions, NUM_CLASSES)
    counts = support(labels, NUM_CLASSES)
    return {
        "macro_f1_restricted": macro_f1,
        "per_class": {
            LE2I_LABEL_NAMES[class_id]: {"f1": f1_by_class[class_id], "support": counts[class_id]}
            for class_id in RESTRICTED_CLASSES
        },
        "support": {LE2I_LABEL_NAMES[class_id]: counts[class_id] for class_id in range(NUM_CLASSES)},
        "confusion_matrix": confusion_matrix(labels, predictions, NUM_CLASSES).tolist(),
    }


def require_clean_reproduction(clean: dict[str, object], stored: dict, context: str) -> dict[str, object]:
    stored_f1 = cast(dict[str, float], stored["f1_by_class"])
    recomputed_f1 = cast(dict[str, dict[str, float]], clean["per_class"])
    checks = {
        "confusion_matrix": clean["confusion_matrix"] == stored["confusion_matrix"],
        "support": clean["support"] == stored["support"],
        "macro_f1_restricted": math.isclose(
            cast(float, clean["macro_f1_restricted"]), float(stored["macro_f1_restricted"]), rel_tol=0.0,
            abs_tol=MACRO_F1_TOLERANCE,
        ),
        "f1_by_class": all(
            math.isclose(recomputed_f1[LE2I_LABEL_NAMES[class_id]]["f1"], float(stored_f1[str(class_id)]),
                         rel_tol=0.0, abs_tol=MACRO_F1_TOLERANCE)
            for class_id in RESTRICTED_CLASSES
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            f"{context}: a avaliação limpa não reproduz final.val de metrics.json ({', '.join(failed)}; Macro-F1 "
            f"restrita {clean['macro_f1_restricted']} contra {stored['macro_f1_restricted']}); nenhum resultado "
            "degradado é calculado"
        )
    return {
        "recorded_macro_f1_restricted": stored["macro_f1_restricted"],
        "recomputed_macro_f1_restricted": clean["macro_f1_restricted"],
        "matches": {name: True for name in checks},
    }


def _f1_change(clean: dict[str, object], degraded: dict[str, object]) -> dict[str, object]:
    clean_classes = cast(dict[str, dict[str, float]], clean["per_class"])
    degraded_classes = cast(dict[str, dict[str, float]], degraded["per_class"])
    return {
        "macro_f1_restricted": cast(float, degraded["macro_f1_restricted"]) - cast(float, clean["macro_f1_restricted"]),
        "per_class": {
            label: degraded_classes[label]["f1"] - clean_classes[label]["f1"] for label in RESTRICTED_LABELS
        },
    }


def affected_subset(
    labels: np.ndarray, clean: np.ndarray, degraded: np.ndarray, masked: np.ndarray
) -> dict[str, object]:
    """Descrição das janelas efetivamente mascaradas; F1 só para classes com suporte suficiente."""
    subset_labels, subset_clean, subset_degraded = labels[masked], clean[masked], degraded[masked]
    counts = support(subset_labels, NUM_CLASSES)
    _, clean_f1 = restricted_macro_f1(subset_labels, subset_clean, NUM_CLASSES)
    _, degraded_f1 = restricted_macro_f1(subset_labels, subset_degraded, NUM_CLASSES)
    per_class: dict[str, dict[str, object]] = {}
    for class_id in RESTRICTED_CLASSES:
        reportable = counts[class_id] >= AFFECTED_MIN_CLASS_SUPPORT
        per_class[LE2I_LABEL_NAMES[class_id]] = {
            "support": counts[class_id],
            "clean_f1": clean_f1[class_id] if reportable else None,
            "degraded_f1": degraded_f1[class_id] if reportable else None,
            "change_f1": degraded_f1[class_id] - clean_f1[class_id] if reportable else None,
        }
    clean_correct = subset_clean == subset_labels
    degraded_correct = subset_degraded == subset_labels
    windows = int(masked.sum())
    return {
        "windows": windows,
        "class_f1_min_support": AFFECTED_MIN_CLASS_SUPPORT,
        "clean_accuracy": float(clean_correct.mean()) if windows else None,
        "degraded_accuracy": float(degraded_correct.mean()) if windows else None,
        "changed_predictions": int((subset_clean != subset_degraded).sum()),
        "became_incorrect": int((clean_correct & ~degraded_correct).sum()),
        "became_correct": int((~clean_correct & degraded_correct).sum()),
        "per_class": per_class,
    }


def mask_plan(inputs: RetainedInputs) -> dict[str, object]:
    masked = inputs.masked
    counts = support(inputs.labels[masked], NUM_CLASSES)
    unchanged = sum(
        bool(np.array_equal(inputs.clean[index], inputs.degraded[index])) for index in np.flatnonzero(masked)
    )
    masked_videos = {video_id for video_id, flag in zip(inputs.video_ids, masked) if flag}
    return {
        "evaluation_windows": len(inputs.labels),
        "masked_windows": int(masked.sum()),
        "masked_fraction": float(masked.mean()),
        "masked_windows_with_unchanged_input": int(unchanged),
        "masked_videos": len(masked_videos),
        "retained_videos": len(set(inputs.video_ids)),
        "masked_support": {label: counts[LE2I_LABEL_NAMES.index(label)] for label in RESTRICTED_LABELS},
        "sha256": inputs.digests(),
    }


def _run_files(run_dir: Path) -> list[Path]:
    return [run_dir / name for name in (*REQUIRED_TRAINING_ARTIFACTS, FOLD_RUN_FILE)]


def model_block(
    root: Path,
    document: dict,
    fold: dict,
    variant: str,
    inputs: RetainedInputs,
    stats: StandardizationStats,
    stats_path: Path,
    stats_sha256: str,
    device: str,
) -> dict[str, object]:
    """Valida o run, reproduz a avaliação limpa gravada e só então avalia a condição degradada."""
    entry = run_entry(root, document, fold, ARM, SEED, variant)
    run_dir = fold_run_dir(root, fold["fold"], ARM, SEED, variant)
    context = f"{fold['fold']}/{variant}"
    with (run_dir / "metrics.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    if metrics.get("device") != device:
        raise RuntimeError(
            f"{context}: final.val foi gravado em {metrics.get('device')!r} e só há {device!r}; a reprodução exata "
            "da avaliação limpa não é garantida"
        )
    config = resolve_fold_config(SEED, stats_path, stats_sha256, variant)
    model = load_compatible_checkpoint(run_dir / "checkpoint.pt", config).to(device)
    clean_predictions = predict(model, inputs.clean, stats, device, config.batch_size)
    clean = condition_metrics(inputs.labels, clean_predictions)
    reproduction = require_clean_reproduction(clean, metrics["final"]["val"], context)
    degraded_predictions = predict(model, inputs.degraded, stats, device, config.batch_size)
    degraded = condition_metrics(inputs.labels, degraded_predictions)
    unmasked = ~inputs.masked
    return {
        "variant": variant,
        "stage": MODEL_STAGES[variant],
        "run_dir": entry["run_dir"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "standardization_stats_sha256": entry["standardization_stats_sha256"],
        "inputs_sha256": inputs.digests(),
        "clean_reproduction": reproduction,
        "clean": clean,
        "degraded": degraded,
        "change": _f1_change(clean, degraded),
        "affected": affected_subset(inputs.labels, clean_predictions, degraded_predictions, inputs.masked),
        "unmasked_prediction_changes": int((clean_predictions[unmasked] != degraded_predictions[unmasked]).sum()),
    }


def paired_loss_difference(models: dict[str, dict]) -> dict[str, object]:
    def loss(variant: str) -> dict[str, object]:
        change = cast(dict[str, object], models[variant]["change"])
        return {
            "macro_f1_restricted": -cast(float, change["macro_f1_restricted"]),
            "per_class": {label: -value for label, value in cast(dict[str, float], change["per_class"]).items()},
        }

    baseline, treated = loss(BASELINE_VARIANT), loss(DETECTION_DROPOUT_VARIANT)
    baseline_classes = cast(dict[str, float], baseline["per_class"])
    treated_classes = cast(dict[str, float], treated["per_class"])
    return {
        "degradation_loss": {BASELINE_VARIANT: baseline, DETECTION_DROPOUT_VARIANT: treated},
        "loss_difference": {
            "macro_f1_restricted": cast(float, baseline["macro_f1_restricted"])
            - cast(float, treated["macro_f1_restricted"]),
            "per_class": {label: baseline_classes[label] - treated_classes[label] for label in RESTRICTED_LABELS},
        },
    }


def diagnostics_output_dir(root: Path) -> Path:
    return root / DIAGNOSTICS_DIR / OUTPUT_NAME


def validate_diagnostics_output(root: Path, frames_path: Path, manifest_path: Path, pose_root: Path) -> Path:
    """Exige a raiz das dobras e recusa saída sobre `folds.json`, dobras, resumos, entradas e artefatos canônicos."""
    validate_output_root(root, frames_path, manifest_path, pose_root)
    output_dir = diagnostics_output_dir(root)
    check_output_dir(
        output_dir,
        [
            *protected_paths(frames_path, manifest_path, pose_root),
            root / FOLDS_FILE,
            summary_dir(root),
            *(fold_dir(root, fold) for fold in fold_names()),
        ],
    )
    return output_dir


def build_report(
    root: Path,
    frames_path: Path,
    manifest_path: Path,
    pose_root: Path,
    pose: PoseInputs | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    frames, snapshot, input_sha256 = _official_train(frames_path, manifest_path)
    pose = pose or real_pose_inputs(frames, pose_root)
    partitions = build_subject_fold_partitions(frames)
    document = read_folds_document(root)
    validate_folds_document(document, partitions, input_sha256, pose.sha256)
    snapshot.update(snapshot_inputs([root / FOLDS_FILE]))
    device = configure_determinism(SEED)
    records = {cast(str, record["fold"]): record for record in cast(list[dict], document["folds"])}

    fold_blocks = []
    for partition in partitions:
        fold = records[partition.fold]
        directory = stats_dir(root, partition.fold)
        stats = load_fold_stats(fold, directory, input_sha256, pose.sha256)
        snapshot.update(snapshot_inputs([directory / POSE_STATS_FILE, directory / PROVENANCE_FILE]))
        for variant in MODEL_VARIANTS:
            snapshot.update(snapshot_inputs(_run_files(fold_run_dir(root, partition.fold, ARM, SEED, variant))))
        inputs = build_retained_inputs(partition, pose)
        models = {
            variant: model_block(root, document, fold, variant, inputs, stats.stats, stats.path, stats.sha256, device)
            for variant in MODEL_VARIANTS
        }
        if len({json.dumps(block["inputs_sha256"], sort_keys=True) for block in models.values()}) != 1:
            raise RuntimeError(f"{partition.fold}: os dois checkpoints não receberam as mesmas entradas")
        fold_blocks.append(
            {
                "fold": partition.fold,
                "retained_subjects": fold["retained_subjects"],
                "retained_windows_eval_stride": fold["retained_windows_eval_stride"],
                "n_retained_videos": len(fold["retained_video_ids"]),
                "mask_plan": mask_plan(inputs),
                "models": models,
                "paired": paired_loss_difference(models),
                "limitations": fold_limitations(fold),
            }
        )

    report: dict[str, object] = {
        "analysis": ANALYSIS,
        "stage": STAGE,
        "canonical": False,
        "read_only": True,
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT],
        "arm": ARM,
        "seed": SEED,
        "models": list(MODEL_VARIANTS),
        "device": device,
        "condition": condition_record(),
        "sign_conventions": SIGN_CONVENTIONS,
        "input_sha256": document["input_sha256"],
        "pose_features_sha256": document["pose_features_sha256"],
        "membership_sha256": document["membership_sha256"],
        "pooled_score": None,
        "pooled_difference": None,
        "significance_test": None,
        "folds": fold_blocks,
        "limitations": LIMITATIONS,
        "artifact_sha256": {display_path(Path(path)): digest for path, digest in sorted(snapshot.items())},
    }
    return report, snapshot


def _fmt(value: object) -> str:
    return "n/d" if value is None else f"{cast(float, value):.3f}"


def _signed(value: object) -> str:
    return "n/d" if value is None else f"{cast(float, value):+.4f}"


def _render_fold(block: dict) -> list[str]:
    retained = ", ".join(str(subject) for subject in block["retained_subjects"])
    plan = block["mask_plan"]
    models = block["models"]
    baseline, treated = models[BASELINE_VARIANT], models[DETECTION_DROPOUT_VARIANT]
    difference = block["paired"]["loss_difference"]
    lines = [
        "",
        f"## Dobra `{block['fold']}` — retém os sujeitos {retained}",
        "",
        f"Retido: {block['n_retained_videos']} vídeos, {plan['evaluation_windows']} janelas rotuladas em stride 1. "
        f"Mascaradas: {plan['masked_windows']} janelas ({plan['masked_fraction']:.1%}) em {plan['masked_videos']} "
        f"vídeos; {plan['masked_windows_with_unchanged_input']} delas já não tinham detecção nos quadros "
        "mascarados e ficaram com a entrada inalterada.",
        "",
        "A avaliação limpa reproduziu a matriz de confusão, o suporte e a Macro-F1 restrita gravados nos dois runs.",
        "",
        "| Modelo | Limpo | Degradado | Mudança | Perda |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, model in models.items():
        clean = model["clean"]["macro_f1_restricted"]
        degraded = model["degraded"]["macro_f1_restricted"]
        lines.append(
            f"| `{variant}` ({model['stage']}) | {_fmt(clean)} | {_fmt(degraded)} | "
            f"{_signed(model['change']['macro_f1_restricted'])} | {_signed(clean - degraded)} |"
        )
    lines += [
        "",
        f"Diferença pareada das perdas (`{BASELINE_VARIANT}` − `{DETECTION_DROPOUT_VARIANT}`): "
        f"**{_signed(difference['macro_f1_restricted'])}** de Macro-F1 restrita.",
        "",
        "| Classe | Suporte | F1 limpo B | F1 degradado B | F1 limpo D | F1 degradado D | Δ perdas (B − D) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in RESTRICTED_LABELS:
        lines.append(
            f"| {label} | {baseline['clean']['per_class'][label]['support']} | "
            f"{_fmt(baseline['clean']['per_class'][label]['f1'])} | "
            f"{_fmt(baseline['degraded']['per_class'][label]['f1'])} | "
            f"{_fmt(treated['clean']['per_class'][label]['f1'])} | "
            f"{_fmt(treated['degraded']['per_class'][label]['f1'])} | "
            f"{_signed(difference['per_class'][label])} |"
        )
    lines += [
        "",
        "B = `baseline` (Etapa 4B); D = `detection_dropout` (Etapa 4C).",
        "",
        f"Subconjunto afetado (descritivo; F1 só com pelo menos {AFFECTED_MIN_CLASS_SUPPORT} janelas afetadas):",
        "",
        "| Modelo | Janelas | Acurácia limpa | Acurácia degradada | Predições mudadas | Viraram erro | Viraram acerto |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, model in models.items():
        affected = model["affected"]
        lines.append(
            f"| `{variant}` | {affected['windows']} | {_fmt(affected['clean_accuracy'])} | "
            f"{_fmt(affected['degraded_accuracy'])} | {affected['changed_predictions']} | "
            f"{affected['became_incorrect']} | {affected['became_correct']} |"
        )
    lines += [
        "",
        "| Classe | Janelas afetadas | Δ F1 B | Δ F1 D |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in RESTRICTED_LABELS:
        baseline_class = baseline["affected"]["per_class"][label]
        treated_class = treated["affected"]["per_class"][label]
        lines.append(
            f"| {label} | {baseline_class['support']} | {_signed(baseline_class['change_f1'])} | "
            f"{_signed(treated_class['change_f1'])} |"
        )
    lines += ["", "Limitações da dobra:", ""]
    lines += [f"- {text}" for text in block["limitations"]]
    return lines


def render_summary(report: dict) -> str:
    lines = [
        "# Dobras por sujeito do Le2i-CS — robustez de inferência à perda de detecção (Etapa 4D, não canônico)",
        "",
        "Diagnóstico somente leitura. Os checkpoints congelados do braço A, seed 42, `baseline` (Etapa 4B) e "
        "`detection_dropout` (Etapa 4C) são avaliados no grupo retido da própria dobra com entrada limpa e com uma "
        "única condição predeclarada: em 20% das janelas, três quadros reais consecutivos viram detecções "
        "ausentes na pose bruta antes do recálculo causal das 134 features. A máscara depende só da identidade da "
        "janela e de uma seed de avaliação fixa, e os dois modelos da dobra recebem as mesmas entradas. As dobras "
        "não são agregadas e não há teste de significância.",
        "",
        "Convenções: mudança = degradado − limpo; perda = limpo − degradado; diferença pareada = perda de "
        "`baseline` − perda de `detection_dropout` (positivo: o modelo da Etapa 4C perde menos).",
    ]
    for block in report["folds"]:
        lines += _render_fold(block)
    lines += ["", "## Limitações gerais", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def run_analysis(
    root: Path,
    frames_path: Path,
    manifest_path: Path,
    pose_root: Path,
    force: bool = False,
    pose: PoseInputs | None = None,
) -> bool:
    output_dir = validate_diagnostics_output(root, frames_path, manifest_path, pose_root)
    targets = [output_dir / REPORT_JSON_FILE, output_dir / REPORT_SUMMARY_FILE]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not force:
        print(f"skip {', '.join(existing)} (já existe, use --force para sobrescrever)")
        return False
    report, snapshot = build_report(root, frames_path, manifest_path, pose_root, pose)
    outputs = {
        targets[0]: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        targets[1]: render_summary(report),
    }
    verify_inputs_unchanged(snapshot)
    return write_text_outputs(output_dir, outputs, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser(
        "analyze", help="Avalia os checkpoints 4B e 4C de cada dobra com entrada limpa e com perda de detecção"
    )
    analyze_parser.add_argument("--fold-root", type=Path, default=SUBJECT_FOLD_ROOT)
    analyze_parser.add_argument("--frames", type=Path, default=ADAPTER.frames_path)
    analyze_parser.add_argument("--manifest", type=Path, default=ADAPTER.manifest_path)
    analyze_parser.add_argument("--pose-root", type=Path, default=ADAPTER.pose_root)
    analyze_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do diagnóstico de robustez")

    args = parser.parse_args()
    if args.command == "analyze":
        missing = missing_prerequisites(args.frames, args.manifest, args.pose_root, require_gpu=False)
        if missing:
            print(f"pré-requisitos ausentes: {missing}; nada foi gravado", file=sys.stderr)
            sys.exit(2)
        run_analysis(args.fold_root, args.frames, args.manifest, args.pose_root, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.subject_fold_dropout_robustness import run_selftest

        if not run_selftest():
            sys.exit(1)


if __name__ == "__main__":
    main()
