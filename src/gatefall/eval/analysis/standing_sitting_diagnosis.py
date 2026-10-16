"""Diagnóstico somente leitura dos erros compartilhados `standing→walk` e `sitting→stand_up` (Etapa 6A).

Diagnóstico exploratório e não canônico de Le2i-CS sobre evidência de
desenvolvimento: o `train` oficial, o `val` oficial e os grupos retidos das
dobras por sujeito da Etapa 4B, que são sujeitos do próprio `train`. Nenhuma
linha, feature, pose ou predição do `test` oficial é usada; o bloco
`final.test` de `metrics.json` só tem a presença conferida pelo validador
compartilhado dos runs. Nada é treinado, reextraído, ajustado, selecionado ou
promovido.

Fontes reaproveitadas:

- `val` oficial: o caminho validado do diagnóstico estratificado
  (`validate_run`, `analyze_split_with_table`) reexecuta cada checkpoint
  congelado só em `val`, alinha cada predição à janela exata de
  `build_window_index` e exige que a matriz recalculada reproduza
  `final.val` de `metrics.json`.
- dobras por sujeito: os validadores das Etapas 4B e 4D (`folds.json`,
  estatística da dobra, proveniência e checkpoint do controle do braço A,
  seed 42) e a reavaliação limpa do grupo retido, que precisa reproduzir
  exatamente `final.val` do run da dobra.
- `train` oficial: só rótulos por quadro e pose bruta, com as janelas em
  `TRAIN_STRIDE` conferidas contra `build_window_index` e contra o suporte de
  `final.train` de cada run. Predições por janela de `train` não existem e não
  são recalculadas; só a matriz agregada de `final.train` é relatada.

Cada janela recebe uma categoria exclusiva, nesta precedência: pose ausente
(menos de metade dos `WINDOW_FRAMES` índices com `person_found`), transição
(janela misturada, que cruza a fronteira inicial do trecho, ou quadro final a
menos de `TRANSITION_MARGIN_FRAMES` quadros do fim de um trecho seguido por
outro) e interior. O suporte limitado é propriedade da classe, não da janela.
As regras de decisão (`decision_criteria`) foram fixadas neste módulo antes
da execução sobre os dados reais. As associações relatadas não estabelecem
causa.
"""

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import build_window_index, window_end_indices
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import (
    EXPECTED_SEEDS,
    LOW_SUPPORT_FRACTION,
    PROTOCOL_NAMES,
    DiagnosisInput,
    _parse_inputs,
    check_output_dir,
    diagnose_split,
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.internal_fold_feasibility import OFFICIAL_TRAIN_SPLIT
from gatefall.eval.analysis.multiseed_summary import ARMS, _aggregate_stats
from gatefall.eval.analysis.other_walk_review_set import LOCAL_OUTPUT_ROOT
from gatefall.eval.analysis.other_walk_train_composition import VIDEO_END, window_pose_signals
from gatefall.eval.analysis.stratified_error_diagnosis import (
    analyze_split_with_table,
    load_evaluation,
    missing_prerequisites,
    run_input_files,
    validate_run,
    validate_runs,
)
from gatefall.eval.analysis.subject_fold_dropout_robustness import (
    _run_files,
    condition_metrics,
    predict,
    require_clean_reproduction,
)
from gatefall.eval.analysis.window_context_diagnosis import (
    CORRECT,
    ERROR,
    MIXED,
    OTHER_OUTCOME,
    _outcome_counts,
    _segments,
    _strata,
    check_conservation,
    label_name,
    video_context,
)
from gatefall.pose.loading import PoseArrays, load_pose, pose_path
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.baseline_a.artifacts import load_compatible_checkpoint
from gatefall.train.internal_folds.folds import RETAINED_SPLIT, restricted_loader
from gatefall.train.shared.determinism import configure_determinism
from gatefall.train.shared.metrics import RESTRICTED_CLASSES
from gatefall.train.subject_folds.cli import ADAPTER, PoseInputs, _official_train, real_pose_inputs
from gatefall.train.subject_folds.folds import (
    FOLDS_FILE,
    SUBJECT_FOLD_ROOT,
    SubjectFoldPartition,
    build_subject_fold_partitions,
    fold_run_dir,
    read_folds_document,
    stats_dir,
    validate_folds_document,
)
from gatefall.train.subject_folds.stats import POSE_STATS_FILE, PROVENANCE_FILE, load_fold_stats
from gatefall.train.subject_folds.summary import run_entry
from gatefall.train.subject_folds.training import resolve_fold_config

ANALYSIS = "standing_sitting_error_diagnosis"
STAGE = "6A"
SUPPORTED_DATASET = "le2i"
VAL_SPLIT = "val"
FOLD_ARM = "A"
FOLD_SEED = 42

REPORT_JSON_FILE = "standing_sitting_diagnosis.json"
REPORT_SUMMARY_FILE = "standing_sitting_diagnosis.md"
REPORT_WINDOWS_FILE = "standing_sitting_windows.csv"

OFFICIAL_TRAIN = "official_train"
OFFICIAL_VAL = "official_val"
SUBJECT_FOLD = "subject_fold"

FOCUS_PAIRS: tuple[tuple[str, str], ...] = (("standing", "walk"), ("sitting", "stand_up"))
FOCUS_TRUE_IDS: frozenset[int] = frozenset(LE2I_LABEL_NAMES.index(true) for true, _pred in FOCUS_PAIRS)
FOCUS_CLASS_IDS: frozenset[int] = frozenset(
    LE2I_LABEL_NAMES.index(label) for pair in FOCUS_PAIRS for label in pair
)

POSE_ABSENT = "pose_absent"
POSE_PRESENT = "pose_present"
TRANSITION = "transition"
INTERIOR = "interior"
CATEGORIES: tuple[str, ...] = (POSE_ABSENT, TRANSITION, INTERIOR)
RULE_STRATA: tuple[str, ...] = (POSE_ABSENT, POSE_PRESENT, TRANSITION, INTERIOR)

# Pose ausente: menos de metade dos índices da janela com person_found.
POSE_ABSENT_MAX_FRACTION = 0.5
# Quadros da grade de 10 fps (0,5 s) antes do fim de um trecho seguido por
# outro trecho; a fronteira inicial já é coberta pela janela misturada.
TRANSITION_MARGIN_FRAMES = 5

MIN_STRATUM_WINDOWS = 10
MIN_RATE_DIFFERENCE = 0.10
MIN_ERROR_SHARE = 0.5
MIN_MODEL_AGREEMENT = 0.8
MIN_UNIT_WINDOWS = 20
MIN_ELIGIBLE_UNITS = 3
MIN_UNIT_AGREEMENT = 2 / 3

CONSIDER_VERDICT = "consider_controlled_intervention"
INSUFFICIENT_VERDICT = "insufficient_data"
UNRESOLVED_VERDICT = "unresolved_limitation"
SUSTAINED = "sustained"
NOT_SUSTAINED = "not_sustained"
NOT_EVALUABLE = "not_evaluable"

CANDIDATE_INTERVENTION = {
    "name": "transition_margin_exclusion",
    "description": (
        "Só no treino, excluir das janelas de treino as da categoria transição (misturadas ou com o quadro final a "
        f"menos de {TRANSITION_MARGIN_FRAMES} quadros do fim do trecho), com a mesma regra em A, B0, B1, C0 e C1 e "
        "inferência, avaliação, janelamento e rótulos de avaliação inalterados."
    ),
    "status": "not_implemented",
}

PAIR_DIMENSIONS: tuple[str, ...] = (
    "category", "pose_presence", "subject", "environment", "video", "preceding_label", "following_label",
)
ATTRIBUTE_COLUMNS = [
    "k_end", "purity", "frames_since_segment_start", "frames_to_segment_end", "preceding_label",
    "following_label", "edge_padding", "detected_fraction", "mean_keypoint_conf", "category",
]
WINDOW_CSV_COLUMNS = [
    "evidence", "fold", "arm", "seed", "video_id", "env", "subject", "k_end", "true_label", "pred_label",
    *ATTRIBUTE_COLUMNS[1:],
]

PoseLoader = Callable[[str], PoseArrays]


def pair_key(true_label: str, predicted_label: str) -> str:
    return f"{true_label}->{predicted_label}"


def _arrow(key: str) -> str:
    return key.replace("->", "→")


def window_category(purity: str, following_label: str, frames_to_segment_end: int, detected_fraction: float) -> str:
    if detected_fraction < POSE_ABSENT_MAX_FRACTION:
        return POSE_ABSENT
    if purity == MIXED or (following_label != VIDEO_END and frames_to_segment_end < TRANSITION_MARGIN_FRAMES):
        return TRANSITION
    return INTERIOR


def window_attributes(labels: np.ndarray, pose: PoseArrays, k_ends: np.ndarray) -> pd.DataFrame:
    if pose.k != len(labels) or pose.person_found.shape != labels.shape:
        raise ValueError(f"pose com K={pose.k} e person_found {pose.person_found.shape}; a grade tem {len(labels)} "
                         "quadros")
    context = video_context(labels, pose.person_found.astype(bool))
    context = cast(pd.DataFrame, context[context["k_end"].isin(k_ends.tolist())]).reset_index(drop=True)
    last = len(labels) - 1
    following = [VIDEO_END if int(end) == last else label_name(int(labels[int(end) + 1]))
                 for end in context["segment_end"]]
    signals = [window_pose_signals(pose, int(k_end)) for k_end in context["k_end"]]
    table = context.assign(
        following_label=following,
        detected_fraction=[signal["detected_fraction"] for signal in signals],
        mean_keypoint_conf=[signal["mean_keypoint_conf"] for signal in signals],
    )
    table["category"] = [
        window_category(str(purity), str(following_label), int(to_end), float(fraction))
        for purity, following_label, to_end, fraction in zip(
            table["purity"], table["following_label"], table["frames_to_segment_end"], table["detected_fraction"]
        )
    ]
    return cast(pd.DataFrame, table[ATTRIBUTE_COLUMNS])


def labeled_windows(frames: pd.DataFrame, stride: int, label_ids: frozenset[int], load: PoseLoader) -> pd.DataFrame:
    """Atributos das janelas em `stride` cujo quadro final tem rótulo em `label_ids`, por vídeo."""
    tables = []
    for video_id, group in frames.groupby("video_id", sort=True):
        ordered = cast(pd.DataFrame, group.sort_values("frame_index"))
        labels = ordered["label"].to_numpy(dtype=np.int64)
        if not np.array_equal(ordered["frame_index"].to_numpy(), np.arange(len(labels))):
            raise ValueError(f"video_id={video_id!r}: frame_index não é contíguo 0..K-1")
        k_ends = window_end_indices(len(labels), stride)
        k_ends = k_ends[np.isin(labels[k_ends], sorted(label_ids))]
        if not len(k_ends):
            continue
        try:
            table = window_attributes(labels, load(str(video_id)), k_ends)
        except ValueError as exc:
            raise ValueError(f"video_id={video_id!r}: {exc}") from exc
        table.insert(0, "label", labels[table["k_end"].to_numpy(dtype=np.int64)])
        table.insert(0, "subject", int(ordered["subject"].iloc[0]))
        table.insert(0, "env", str(ordered["env"].iloc[0]))
        table.insert(0, "video_id", str(video_id))
        tables.append(table)
    if not tables:
        return pd.DataFrame(columns=["video_id", "env", "subject", "label", *ATTRIBUTE_COLUMNS])
    return pd.concat(tables, ignore_index=True)


def join_attributes(predictions: pd.DataFrame, attributes: pd.DataFrame, context: str) -> pd.DataFrame:
    """Une predições e atributos pela identidade exata `(video_id, k_end, label)`; qualquer sobra falha."""
    keys = ["video_id", "k_end", "label"]
    left = cast(pd.DataFrame, predictions[[*keys, "env", "subject", "pred_label"]]).astype({"k_end": np.int64,
                                                                                         "label": np.int64})
    right = cast(pd.DataFrame, attributes.drop(columns=["env", "subject"])).astype({"k_end": np.int64,
                                                                                   "label": np.int64})
    joined = left.merge(right, on=keys, how="outer", indicator=True)
    unmatched = cast(pd.DataFrame, joined[joined["_merge"] != "both"])
    if not unmatched.empty:
        sample = [(str(video_id), int(k_end)) for video_id, k_end in zip(unmatched["video_id"], unmatched["k_end"])]
        raise ValueError(f"{context}: atributos de janela não alinham com as predições em {len(unmatched)} "
                         f"janela(s): {sample[:5]}")
    ordered = cast(pd.DataFrame, joined.drop(columns="_merge").sort_values(["video_id", "k_end"]))
    return ordered.reset_index(drop=True)


def _with_dimensions(windows: pd.DataFrame) -> pd.DataFrame:
    return windows.assign(
        pose_presence=np.where(windows["category"] == POSE_ABSENT, POSE_ABSENT, POSE_PRESENT),
        subject=windows["subject"].astype(int),
        environment=windows["env"].astype(str),
        video=windows["video_id"].astype(str),
    )


def _column(table: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, table[name])


def _clean_values(values: pd.Series) -> list[float]:
    return [float(value) for value in values if value is not None and not math.isnan(float(value))]


def _median(values: pd.Series) -> float | None:
    clean = _clean_values(values)
    return float(statistics.median(clean)) if clean else None


def _mean(values: pd.Series) -> float | None:
    clean = _clean_values(values)
    return float(statistics.mean(clean)) if clean else None


def _share(part: float, total: float) -> float | None:
    return part / total if total > 0 else None


def pose_profile(group: pd.DataFrame) -> dict:
    n = len(group)
    return {
        "n": n,
        "median_detected_fraction": _median(_column(group, "detected_fraction")),
        "mean_keypoint_conf": _mean(_column(group, "mean_keypoint_conf")),
        **{f"{category}_share": _share(int((group["category"] == category).sum()), n) for category in CATEGORIES},
        "median_frames_to_segment_end": _median(_column(group, "frames_to_segment_end")),
    }


def unit_counts(subset: pd.DataFrame) -> dict[str, dict]:
    """Por sujeito bruto: janelas verdadeiras e erros do par em cada estrato das regras."""
    units: dict[str, dict] = {}
    for subject, group in subset.groupby("subject", sort=True):
        strata = {POSE_ABSENT: group["category"] == POSE_ABSENT, POSE_PRESENT: group["category"] != POSE_ABSENT,
                  TRANSITION: group["category"] == TRANSITION, INTERIOR: group["category"] == INTERIOR}
        is_error = group["outcome"] == ERROR
        units[f"subject_{int(cast(int, subject))}"] = {
            "n_true": len(group),
            "n_error": int(is_error.sum()),
            **{name: {"n_true": int(mask.sum()), "n_error": int((mask & is_error).sum())}
               for name, mask in strata.items()},
        }
    return units


def pair_breakdown(windows: pd.DataFrame, true_label: str, predicted_label: str, matrix: np.ndarray,
                   context: str) -> dict:
    true_id = LE2I_LABEL_NAMES.index(true_label)
    pred_id = LE2I_LABEL_NAMES.index(predicted_label)
    subset = _with_dimensions(cast(pd.DataFrame, windows[windows["label"] == true_id]))
    predicted = subset["pred_label"].to_numpy()
    subset = subset.assign(
        outcome=np.select([predicted == pred_id, predicted == true_id], [ERROR, CORRECT], OTHER_OUTCOME)
    )
    counts = _outcome_counts(subset)
    strata = {dimension: _strata(subset, dimension, counts) for dimension in PAIR_DIMENSIONS}
    description = f"{context} {pair_key(true_label, predicted_label)}"
    conservation = check_conservation(counts, strata, matrix, true_id, pred_id, description)
    units = unit_counts(subset)
    for field in ("n_true", "n_error"):
        if sum(unit[field] for unit in units.values()) != counts[field]:
            raise ValueError(f"{description}: soma de {field} por sujeito diverge de metrics.json")
    row = matrix[true_id]
    return {
        "counts": counts,
        "error_rate": _share(counts["n_error"], counts["n_true"]),
        "destinations": {LE2I_LABEL_NAMES[label]: int(row[label]) for label in range(len(row)) if row[label]},
        "profiles": {
            ERROR: pose_profile(cast(pd.DataFrame, subset[subset["outcome"] == ERROR])),
            CORRECT: pose_profile(cast(pd.DataFrame, subset[subset["outcome"] == CORRECT])),
        },
        "strata": strata,
        "units": units,
        "conservation": conservation,
    }


def matrix_fit(matrix: np.ndarray) -> dict[str, dict]:
    fit = {}
    for true_label, predicted_label in FOCUS_PAIRS:
        row = matrix[LE2I_LABEL_NAMES.index(true_label)]
        n_error = int(row[LE2I_LABEL_NAMES.index(predicted_label)])
        fit[pair_key(true_label, predicted_label)] = {"n_true": int(row.sum()), "n_error": n_error,
                                                      "error_rate": _share(n_error, int(row.sum()))}
    return fit


def support_by_name(split_block: dict) -> dict[str, int]:
    return {label: int(row["support"]) for label, row in split_block["per_class"].items()}


def _prediction_table(table: pd.DataFrame) -> pd.DataFrame:
    focus = cast(pd.DataFrame, table[table["label"].isin(sorted(FOCUS_TRUE_IDS))])
    return focus.reset_index(drop=True)


def official_val_run(item: DiagnosisInput, load: PoseLoader) -> tuple[dict, pd.DataFrame, dict[str, int]]:
    run = validate_run(item)
    seed = run.header["seed"]
    if not isinstance(seed, int):
        raise ValueError(f"{run.context}: config.yaml sem seed inteira")
    train_block = diagnose_split(run.final.get(OFFICIAL_TRAIN_SPLIT), LE2I_LABEL_NAMES, f"{run.context} [train]")
    frames, evaluate_split = load_evaluation(item.arm, item.dataset, item.path).prepare()
    context = f"{run.context} [{VAL_SPLIT}]"
    table, analysis = analyze_split_with_table(frames, evaluate_split, VAL_SPLIT, run.final[VAL_SPLIT], context)
    val_frames = cast(pd.DataFrame, frames[frames["split"] == VAL_SPLIT])
    attributes = labeled_windows(val_frames, EVAL_STRIDE, FOCUS_TRUE_IDS, load)
    windows = join_attributes(_prediction_table(table), attributes, context)
    matrix = np.asarray(run.final[VAL_SPLIT]["confusion_matrix"], dtype=np.int64)
    entry = {
        **run.header,
        "evidence": OFFICIAL_VAL,
        "reconciliation": analysis["reconciliation"],
        "val_support": support_by_name(analysis["overall"]),
        "pairs": {pair_key(true, pred): pair_breakdown(windows, true, pred, matrix, context)
                  for true, pred in FOCUS_PAIRS},
        "train_fit": matrix_fit(np.asarray(train_block["confusion_matrix"], dtype=np.int64)),
    }
    return entry, windows, support_by_name(train_block)


def retained_clean_windows(partition: SubjectFoldPartition, pose: PoseInputs) -> tuple[pd.DataFrame, np.ndarray]:
    """Janelas limpas do grupo retido, na ordem do índice da dobra, lidas só dos vídeos retidos."""
    context = f"{partition.fold}/{RETAINED_SPLIT}"
    loader = restricted_loader(pose.loader, partition.retained_video_ids, context)
    source = PoseWindowDataset(partition.frames, RETAINED_SPLIT, EVAL_STRIDE, loader)
    windows = partition.retained_windows
    if len(source) != len(windows):
        raise RuntimeError(f"{context}: janelas do retido divergem da composição da dobra")
    first = source[0][0]
    clean = np.empty((len(source), *first.shape), dtype=np.float32)
    identity: list[tuple[str, int, int]] = []
    for index in range(len(source)):
        window, label, (video_id, k_end) = source[index]
        clean[index] = window
        identity.append((video_id, int(k_end), int(label)))
    expected = list(zip(windows["video_id"].astype(str), windows["k_end"].astype(int), windows["label"].astype(int)))
    if identity != expected:
        raise RuntimeError(f"{context}: ordem das janelas diverge do índice da dobra")
    table = pd.DataFrame({
        "video_id": windows["video_id"].astype(str).to_numpy(),
        "env": windows["env"].astype(str).to_numpy(),
        "subject": windows["subject"].astype(np.int64).to_numpy(),
        "k_end": windows["k_end"].astype(np.int64).to_numpy(),
        "label": windows["label"].astype(np.int64).to_numpy(),
    })
    return table, clean


def subject_fold_block(
    root: Path, document: dict, fold: dict, partition: SubjectFoldPartition, pose: PoseInputs, load: PoseLoader,
    device: str,
) -> tuple[dict, pd.DataFrame]:
    context = f"{partition.fold}/{FOLD_ARM}/seed_{FOLD_SEED}"
    input_sha256 = cast(dict[str, str], document["input_sha256"])
    stats = load_fold_stats(fold, stats_dir(root, partition.fold), input_sha256, pose.sha256)
    entry = run_entry(root, document, fold, FOLD_ARM, FOLD_SEED)
    run_dir = fold_run_dir(root, partition.fold, FOLD_ARM, FOLD_SEED)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("device") != device:
        raise RuntimeError(f"{context}: final.val foi gravado em {metrics.get('device')!r} e só há {device!r}; a "
                           "reprodução exata das predições não é garantida")
    config = resolve_fold_config(FOLD_SEED, stats.path, stats.sha256)
    model = load_compatible_checkpoint(run_dir / "checkpoint.pt", config).to(device)
    table, clean = retained_clean_windows(partition, pose)
    predictions = predict(model, clean, stats.stats, device, config.batch_size)
    retained = metrics["final"]["val"]
    reproduction = require_clean_reproduction(
        condition_metrics(table["label"].to_numpy(dtype=np.int64), predictions), retained, context
    )
    table = table.assign(pred_label=predictions)
    retained_frames = partition.side_frames(RETAINED_SPLIT)
    retained_loader = restricted_loader(load, partition.retained_video_ids, f"{context}/pose bruta")
    attributes = labeled_windows(retained_frames, EVAL_STRIDE, FOCUS_TRUE_IDS, retained_loader)
    windows = join_attributes(_prediction_table(table), attributes, context)
    matrix = np.asarray(retained["confusion_matrix"], dtype=np.int64)
    block = {
        "evidence": SUBJECT_FOLD,
        "fold": partition.fold,
        "arm": FOLD_ARM,
        "seed": FOLD_SEED,
        "retained_subjects": list(partition.retained_subjects),
        "development_subjects": list(partition.development_subjects),
        "run_dir": entry["run_dir"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "standardization_stats_sha256": entry["standardization_stats_sha256"],
        "clean_reproduction": reproduction,
        "development_support_train_stride": fold["development_support_train_stride"],
        "retained_support_eval_stride": entry["retained_support"],
        "pairs": {pair_key(true, pred): pair_breakdown(windows, true, pred, matrix, context)
                  for true, pred in FOCUS_PAIRS},
        "train_fit": matrix_fit(np.asarray(metrics["final"]["train"]["confusion_matrix"], dtype=np.int64)),
    }
    return block, windows


def subject_fold_evidence(
    root: Path, frames: pd.DataFrame, input_sha256: dict[str, str], pose_inputs: Callable[[], PoseInputs],
    load: PoseLoader, snapshot: dict[str, str],
) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    folds_path = root / FOLDS_FILE
    if not folds_path.is_file():
        return [], [], [{"evidence": SUBJECT_FOLD, "source": display_path(root), "reason": "folds.json ausente",
                         "missing_files": [display_path(folds_path)]}]
    pose = pose_inputs()
    partitions = build_subject_fold_partitions(frames)
    document = read_folds_document(root)
    validate_folds_document(document, partitions, input_sha256, pose.sha256)
    snapshot.update(snapshot_inputs([folds_path]))
    device = configure_determinism(FOLD_SEED)
    records = {cast(str, record["fold"]): record for record in cast(list[dict], document["folds"])}
    blocks: list[dict] = []
    tables: list[pd.DataFrame] = []
    missing: list[dict] = []
    for partition in partitions:
        run_dir = fold_run_dir(root, partition.fold, FOLD_ARM, FOLD_SEED)
        directory = stats_dir(root, partition.fold)
        files = [*_run_files(run_dir), directory / POSE_STATS_FILE, directory / PROVENANCE_FILE]
        absent = [display_path(path) for path in files if not path.is_file()]
        if absent:
            missing.append({"evidence": SUBJECT_FOLD, "source": display_path(run_dir),
                            "reason": f"controle da dobra {partition.fold} incompleto", "missing_files": absent})
            continue
        snapshot.update(snapshot_inputs(files))
        block, windows = subject_fold_block(root, document, records[partition.fold], partition, pose, load, device)
        blocks.append(block)
        tables.append(windows.assign(evidence=SUBJECT_FOLD, fold=partition.fold, arm=FOLD_ARM, seed=FOLD_SEED))
    return blocks, tables, missing


def class_segments(train_frames: pd.DataFrame, label_id: int) -> dict:
    lengths: list[int] = []
    following: dict[str, int] = {}
    for _video_id, group in train_frames.groupby("video_id", sort=True):
        labels = cast(pd.DataFrame, group.sort_values("frame_index"))["label"].to_numpy(dtype=np.int64)
        starts, ends, _segment_of = _segments(labels)
        for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
            if int(labels[start]) != label_id:
                continue
            lengths.append(end - start + 1)
            name = VIDEO_END if index == len(starts) - 1 else label_name(int(labels[starts[index + 1]]))
            following[name] = following.get(name, 0) + 1
    return {
        "n_segments": len(lengths),
        "median_length_frames": float(statistics.median(lengths)) if lengths else None,
        "short_segments": sum(length < WINDOW_FRAMES for length in lengths),
        "following_label": dict(sorted(following.items(), key=lambda item: (-item[1], item[0]))),
    }


def _tally(values: pd.Series) -> dict[str, int]:
    counts = values.astype(str).value_counts()
    return {str(key): int(count) for key, count in sorted(counts.items(), key=lambda item: str(item[0]))}


def train_composition(train_frames: pd.DataFrame, load: PoseLoader,
                      recorded_supports: list[tuple[str, dict[str, int]]]) -> tuple[dict, pd.DataFrame]:
    windows = labeled_windows(train_frames, TRAIN_STRIDE, FOCUS_CLASS_IDS, load)
    index = build_window_index(train_frames, stride=TRAIN_STRIDE, drop_ignored=True)
    focus = cast(pd.DataFrame, index[index["label"].isin(sorted(FOCUS_CLASS_IDS))])
    expected = sorted(zip(focus["video_id"].astype(str), focus["k_end"].astype(int), focus["label"].astype(int)))
    observed = sorted(zip(windows["video_id"].astype(str), windows["k_end"].astype(int), windows["label"].astype(int)))
    if expected != observed:
        raise ValueError("janelas de train não alinham com build_window_index em TRAIN_STRIDE")
    support = {name: int((index["label"] == class_id).sum()) for class_id, name in enumerate(LE2I_LABEL_NAMES)}
    for source, recorded in recorded_supports:
        if recorded != support:
            raise ValueError(f"{source}: suporte de final.train diverge de build_window_index em TRAIN_STRIDE")
    n_labeled = len(index)
    classes = {}
    for class_id in sorted(FOCUS_CLASS_IDS):
        group = cast(pd.DataFrame, windows[windows["label"] == class_id])
        classes[LE2I_LABEL_NAMES[class_id]] = {
            "n_windows": len(group),
            "share_of_labeled": _share(len(group), n_labeled),
            "n_videos": len(set(_column(group, "video_id"))),
            "subjects": sorted({int(subject) for subject in _column(group, "subject")}),
            "by_subject": _tally(_column(group, "subject")),
            "by_environment": _tally(_column(group, "env")),
            "pose": pose_profile(group),
            "segments": class_segments(train_frames, class_id),
        }
    report = {"stride": TRAIN_STRIDE, "n_labeled_windows": n_labeled, "support": support, "classes": classes,
              "support_reconciled_with": [source for source, _support in recorded_supports]}
    return report, windows


def mean_unit(units: list[dict]) -> dict:
    """Média entre seeds das contagens de erro de uma unidade; o suporte é idêntico entre seeds."""
    first = units[0]
    for unit in units[1:]:
        if any(unit[name]["n_true"] != first[name]["n_true"] for name in RULE_STRATA):
            raise ValueError("suporte por estrato difere entre seeds da mesma unidade")
    return {
        "n_true": first["n_true"],
        "n_error": statistics.mean(unit["n_error"] for unit in units),
        **{name: {"n_true": first[name]["n_true"],
                  "n_error": statistics.mean(unit[name]["n_error"] for unit in units)}
           for name in RULE_STRATA},
    }


def at_least(value: float, threshold: float) -> bool:
    return value >= threshold or math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)


def model_condition(unit: dict, focus: str, reference: str) -> dict:
    focus_rate = _share(unit[focus]["n_error"], unit[focus]["n_true"])
    reference_rate = _share(unit[reference]["n_error"], unit[reference]["n_true"])
    error_share = _share(unit[focus]["n_error"], unit["n_error"])
    met = (
        focus_rate is not None and reference_rate is not None and error_share is not None
        and at_least(focus_rate - reference_rate, MIN_RATE_DIFFERENCE) and at_least(error_share, MIN_ERROR_SHARE)
    )
    return {"focus_rate": focus_rate, "reference_rate": reference_rate, "focus_error_share": error_share,
            "met": bool(met)}


def evaluate_rule(name: str, focus: str, reference: str, units: list[dict]) -> dict:
    """`units`: uma entrada por sujeito retido, com `models` = [(modelo, contagens)]."""
    rows = []
    for unit in units:
        first = unit["models"][0][1]
        eligible = first[focus]["n_true"] >= MIN_STRATUM_WINDOWS and first[reference]["n_true"] >= MIN_STRATUM_WINDOWS
        models = [{"model": model, **model_condition(counts, focus, reference)} for model, counts in unit["models"]]
        agreement = sum(model["met"] for model in models) / len(models)
        rows.append({"unit": unit["unit"], "evidence": unit["evidence"], "eligible": eligible,
                     "focus_windows": first[focus]["n_true"], "reference_windows": first[reference]["n_true"],
                     "model_agreement": agreement, "met": eligible and at_least(agreement, MIN_MODEL_AGREEMENT),
                     "models": models})
    eligible = [row for row in rows if row["eligible"]]
    agreeing = [row for row in eligible if row["met"]]
    if len(eligible) < MIN_ELIGIBLE_UNITS:
        status = NOT_EVALUABLE
    elif at_least(len(agreeing) / len(eligible), MIN_UNIT_AGREEMENT):
        status = SUSTAINED
    else:
        status = NOT_SUSTAINED
    return {"rule": name, "focus_stratum": focus, "reference_stratum": reference, "n_eligible_units": len(eligible),
            "n_agreeing_units": len(agreeing), "status": status, "units": rows}


def support_rule(train_class: dict, units: list[dict]) -> dict:
    unit_windows = {unit["unit"]: unit["models"][0][1]["n_true"] for unit in units}
    eligible = sorted(unit for unit, windows in unit_windows.items() if windows >= MIN_UNIT_WINDOWS)
    train_share = train_class["share_of_labeled"]
    limited_train = train_share is None or train_share < LOW_SUPPORT_FRACTION
    return {
        "rule": "R0_class_support",
        "train_windows": train_class["n_windows"],
        "train_share": train_share,
        "train_subjects": train_class["subjects"],
        "unit_windows": unit_windows,
        "eligible_units": eligible,
        "limited": limited_train or len(eligible) < MIN_ELIGIBLE_UNITS,
    }


def pair_units(key: str, val_entries: list[dict], fold_blocks: list[dict]) -> list[dict]:
    units: list[dict] = []
    by_arm: dict[str, list[dict]] = {}
    for entry in val_entries:
        by_arm.setdefault(entry["arm"], []).append(entry)
    val_units = sorted({unit for entry in val_entries for unit in entry["pairs"][key]["units"]})
    for unit in val_units:
        models = []
        for arm in ARMS:
            seeds = [entry["pairs"][key]["units"][unit] for entry in by_arm.get(arm, [])
                     if unit in entry["pairs"][key]["units"]]
            if seeds:
                models.append((arm, mean_unit(seeds)))
        units.append({"unit": f"{OFFICIAL_VAL}/{unit}", "evidence": OFFICIAL_VAL, "models": models})
    for block in fold_blocks:
        for unit, counts in sorted(block["pairs"][key]["units"].items()):
            units.append({"unit": f"{block['fold']}/{unit}", "evidence": SUBJECT_FOLD,
                          "models": [(f"{block['arm']}_seed{block['seed']}", counts)]})
    return units


def decision_criteria() -> dict:
    return {
        "category_precedence": list(CATEGORIES),
        "pose_absent_max_fraction": POSE_ABSENT_MAX_FRACTION,
        "transition_margin_frames": TRANSITION_MARGIN_FRAMES,
        "unit": "sujeito bruto retido: o sujeito do val oficial e cada sujeito dos grupos retidos das dobras",
        "min_stratum_windows": MIN_STRATUM_WINDOWS,
        "min_rate_difference": MIN_RATE_DIFFERENCE,
        "min_error_share": MIN_ERROR_SHARE,
        "min_model_agreement": MIN_MODEL_AGREEMENT,
        "min_unit_windows": MIN_UNIT_WINDOWS,
        "min_eligible_units": MIN_ELIGIBLE_UNITS,
        "min_unit_agreement": MIN_UNIT_AGREEMENT,
        "low_support_fraction": LOW_SUPPORT_FRACTION,
        "rules": {
            "R0_class_support": "suporte limitado se a classe verdadeira tem menos de low_support_fraction das "
                                "janelas rotuladas do train oficial em TRAIN_STRIDE ou menos de min_eligible_units "
                                "unidades com min_unit_windows janelas verdadeiras",
            "R1_transition": "taxa de erro do par em transição − taxa em interior ≥ min_rate_difference e "
                             "transição com ≥ min_error_share dos erros do par na unidade",
            "R2_pose_absence": "taxa de erro do par com pose ausente − taxa com pose presente ≥ "
                               "min_rate_difference e pose ausente com ≥ min_error_share dos erros do par na "
                               "unidade",
        },
        "rule_status": "regra elegível em uma unidade com min_stratum_windows janelas verdadeiras nos dois estratos; "
                       "no val oficial, a unidade atende se ≥ min_model_agreement dos braços atendem com a média "
                       "entre seeds; sustentada com ≥ min_eligible_units unidades elegíveis e ≥ min_unit_agreement "
                       "delas atendendo; sem unidades elegíveis suficientes, não avaliável",
        "pair_verdict": [
            f"R0 limitado → {INSUFFICIENT_VERDICT}",
            f"R1 sustentada → {CONSIDER_VERDICT} ({CANDIDATE_INTERVENTION['name']})",
            f"R1 não avaliável → {INSUFFICIENT_VERDICT}",
            f"caso contrário → {UNRESOLVED_VERDICT}; R2 é só informativa, porque a robustez à perda de detecção já "
            "foi testada e encerrada nas Etapas 4A–4D",
        ],
        "overall_verdict": f"{CONSIDER_VERDICT} se algum par a sustenta; {INSUFFICIENT_VERDICT} se todos os pares "
                           f"têm dados insuficientes; {UNRESOLVED_VERDICT} caso contrário",
        "candidate_intervention": CANDIDATE_INTERVENTION,
    }


def pair_verdict(support: dict, transition: dict) -> str:
    if support["limited"]:
        return INSUFFICIENT_VERDICT
    if transition["status"] == SUSTAINED:
        return CONSIDER_VERDICT
    if transition["status"] == NOT_EVALUABLE:
        return INSUFFICIENT_VERDICT
    return UNRESOLVED_VERDICT


def overall_verdict(verdicts: list[str]) -> str:
    if CONSIDER_VERDICT in verdicts:
        return CONSIDER_VERDICT
    if verdicts and all(verdict == INSUFFICIENT_VERDICT for verdict in verdicts):
        return INSUFFICIENT_VERDICT
    return UNRESOLVED_VERDICT


def decide(train: dict, val_entries: list[dict], fold_blocks: list[dict]) -> dict:
    pairs = {}
    for true_label, predicted_label in FOCUS_PAIRS:
        key = pair_key(true_label, predicted_label)
        units = pair_units(key, val_entries, fold_blocks)
        support = support_rule(train["classes"][true_label], units)
        transition = evaluate_rule("R1_transition", TRANSITION, INTERIOR, units)
        pose = evaluate_rule("R2_pose_absence", POSE_ABSENT, POSE_PRESENT, units)
        pairs[key] = {"support": support, "transition": transition, "pose_absence": pose,
                      "verdict": pair_verdict(support, transition)}
    verdict = overall_verdict([block["verdict"] for block in pairs.values()])
    return {
        "pairs": pairs,
        "verdict": verdict,
        "candidate_intervention": CANDIDATE_INTERVENTION if verdict == CONSIDER_VERDICT else None,
    }


def _stats(values: list[float | None]) -> dict:
    return _aggregate_stats([value for value in values if value is not None])


def _aggregate_strata(blocks: list[list[dict]]) -> list[dict]:
    if any([row["stratum"] for row in block] != [row["stratum"] for row in blocks[0]] for block in blocks):
        raise ValueError("estratos diferem entre seeds do mesmo braço")
    return [
        {"stratum": first["stratum"], "n_true": first["n_true"],
         "n_error": _stats([block[index]["n_error"] for block in blocks]),
         "error_rate": _stats([block[index]["error_rate"] for block in blocks])}
        for index, first in enumerate(blocks[0])
    ]


def aggregate_arm(entries: list[dict]) -> dict:
    pairs = {}
    for true_label, predicted_label in FOCUS_PAIRS:
        key = pair_key(true_label, predicted_label)
        blocks = [entry["pairs"][key] for entry in entries]
        destinations = sorted({label for block in blocks for label in block["destinations"]})
        pairs[key] = {
            "n_true": blocks[0]["counts"]["n_true"],
            "n_error": _stats([block["counts"]["n_error"] for block in blocks]),
            "error_rate": _stats([block["error_rate"] for block in blocks]),
            "destinations": {label: _stats([block["destinations"].get(label, 0) for block in blocks])
                             for label in destinations},
            "profiles": {outcome: {field: _stats([block["profiles"][outcome][field] for block in blocks])
                                   for field in blocks[0]["profiles"][outcome]}
                         for outcome in (ERROR, CORRECT)},
            "strata": {dimension: _aggregate_strata([block["strata"][dimension] for block in blocks])
                       for dimension in PAIR_DIMENSIONS},
            "train_fit_error_rate": _stats([entry["train_fit"][key]["error_rate"] for entry in entries]),
        }
    return {"seeds": [entry["seed"] for entry in entries], "n_seeds": len(entries), "pairs": pairs}


def coverage_gaps(entries: list[dict], provided_arms: set[str]) -> list[dict]:
    gaps = []
    for arm in ARMS:
        seeds = {entry["seed"] for entry in entries if entry["arm"] == arm}
        if arm not in provided_arms:
            gaps.append({"evidence": OFFICIAL_VAL, "source": arm, "reason": "nenhum run fornecido",
                         "missing_files": []})
            continue
        gaps.extend({"evidence": OFFICIAL_VAL, "source": f"{arm} seed {seed}", "reason": "seed esperada não analisada",
                     "missing_files": []} for seed in EXPECTED_SEEDS if seed not in seeds)
    return gaps


def structural_missing_evidence() -> list[dict]:
    return [
        {"evidence": OFFICIAL_TRAIN, "source": "metrics.json final.train",
         "reason": "predições por janela do train não foram persistidas e não são recalculadas; só a matriz "
                   "agregada de final.train é relatada", "missing_files": []},
        {"evidence": SUBJECT_FOLD, "source": "B0, B1, C0, C1",
         "reason": "as dobras por sujeito só têm o controle do braço A, seed 42 (Etapa 4B); não há predição "
                   "retida das demais armas", "missing_files": []},
        {"evidence": OFFICIAL_VAL, "source": "Le2i-CV",
         "reason": "fora do escopo: os dois pares focais são erros compartilhados do val de Le2i-CS",
         "missing_files": []},
    ]


def limitations() -> list[str]:
    return [
        "Análise exploratória e não canônica (Etapa 6A): nada substitui runs/reference nem seleciona run, seed, "
        "checkpoint, limiar ou hiperparâmetro. O test oficial não é lido e não entra em nenhuma regra.",
        "Associação não é causa: categorias de janela coincidem com posição no vídeo, ação, sujeito, ambiente e "
        "decisões de anotação. Uma taxa de erro maior em uma categoria não mostra que ela produz o erro.",
        f"Janelas com eval_stride {EVAL_STRIDE} se sobrepõem e não são observações independentes; a unidade das "
        "regras é o sujeito bruto retido, e a concordância entre unidades é descritiva, sem teste de significância.",
        "O val oficial tem um único sujeito; as dobras só têm o braço A com uma seed. Unidades do val e das dobras "
        "vêm de modelos diferentes e não são comparadas entre si em magnitude.",
        "Médias e desvios do val oficial são entre seeds e descrevem inicialização e otimização, não incerteza "
        "entre sujeitos.",
        "A pose descreve o artefato YOLO-Pose bruto: person_found e a confiança dos keypoints não medem a "
        "qualidade da pose imputada nem a entrada visual de B0, B1, C0 e C1.",
        "As predições vêm de reexecuções validadas. A reprodução exata das matrizes gravadas é necessária, mas não "
        "prova que cada janela recebeu historicamente o mesmo rótulo.",
        "Os limiares das regras são convenções fixadas antes da execução; outros limiares podem levar a outro "
        "veredito. O alvo de 0,800 não é estimado nem verificado aqui.",
    ]


def window_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows = table.assign(
        true_label=[label_name(int(label)) for label in table["label"]],
        pred_label=[label_name(int(label)) if label is not None and not pd.isna(label) else ""
                    for label in table["pred_label"]],
    )
    for column in WINDOW_CSV_COLUMNS:
        if column not in rows:
            rows[column] = ""
    return cast(pd.DataFrame, rows[WINDOW_CSV_COLUMNS])


def build_report(
    runs: list[DiagnosisInput],
    fold_root: Path,
    frames_path: Path,
    manifest_path: Path,
    pose_root: Path,
    load: PoseLoader,
    pose: PoseInputs | None = None,
) -> tuple[dict, pd.DataFrame, dict[str, str]]:
    train_frames, snapshot, input_sha256 = _official_train(frames_path, manifest_path)
    missing: list[dict] = []
    val_entries: list[dict] = []
    val_tables: list[pd.DataFrame] = []
    train_supports: list[tuple[str, dict[str, int]]] = []
    seen: set[tuple[str, int]] = set()
    for item in runs:
        absent = missing_prerequisites(item)
        if absent:
            missing.append({"evidence": OFFICIAL_VAL, "source": display_path(item.path),
                            "reason": "pré-requisitos ausentes", "missing_files": absent})
            continue
        snapshot.update(snapshot_inputs([path.resolve() for path in run_input_files(item)]))
        try:
            entry, windows, train_support = official_val_run(item, load)
        except FileNotFoundError as exc:
            missing.append({"evidence": OFFICIAL_VAL, "source": display_path(item.path),
                            "reason": f"arquivo ausente durante a inferência: {exc}", "missing_files": []})
            continue
        identity = (entry["arm"], entry["seed"])
        if identity in seen:
            raise ValueError(f"braço {entry['arm']} com seed {entry['seed']} fornecido mais de uma vez")
        seen.add(identity)
        val_entries.append(entry)
        val_tables.append(windows.assign(evidence=OFFICIAL_VAL, fold="", arm=entry["arm"], seed=entry["seed"]))
        train_supports.append((entry["source"], train_support))
    val_entries.sort(key=lambda entry: (ARMS.index(entry["arm"]), entry["seed"]))
    missing.extend(coverage_gaps(val_entries, {item.arm for item in runs}))

    fold_blocks, fold_tables, fold_missing = subject_fold_evidence(
        fold_root, train_frames, input_sha256, lambda: pose or real_pose_inputs(train_frames, pose_root), load,
        snapshot,
    )
    missing.extend(fold_missing)
    train, train_windows = train_composition(train_frames, load, train_supports)
    decision = decide(train, val_entries, fold_blocks)

    supports = {
        OFFICIAL_TRAIN: train["support"],
        OFFICIAL_VAL: val_entries[0]["val_support"] if val_entries else None,
        **{f"{block['fold']}/development_train": block["development_support_train_stride"] for block in fold_blocks},
        **{f"{block['fold']}/retained": block["retained_support_eval_stride"] for block in fold_blocks},
    }
    report = {
        "analysis": ANALYSIS,
        "stage": STAGE,
        "canonical": False,
        "read_only": True,
        "dataset": SUPPORTED_DATASET,
        "protocol": PROTOCOL_NAMES[SUPPORTED_DATASET],
        "official_splits_read": [OFFICIAL_TRAIN_SPLIT, VAL_SPLIT],
        "focus_pairs": [pair_key(true, pred) for true, pred in FOCUS_PAIRS],
        "label_names": list(LE2I_LABEL_NAMES),
        "restricted_labels": [LE2I_LABEL_NAMES[class_id] for class_id in RESTRICTED_CLASSES],
        "window_frames": WINDOW_FRAMES,
        "eval_stride": EVAL_STRIDE,
        "train_stride": TRAIN_STRIDE,
        "decision_criteria": decision_criteria(),
        "support": supports,
        "train_composition": train,
        "official_val": {
            "runs": val_entries,
            "arms": {arm: aggregate_arm([entry for entry in val_entries if entry["arm"] == arm])
                     for arm in ARMS if any(entry["arm"] == arm for entry in val_entries)},
        },
        "subject_folds": {"root": display_path(fold_root), "folds": fold_blocks},
        "decision": decision,
        "missing_evidence": [*missing, *structural_missing_evidence()],
        "limitations": limitations(),
    }
    tables = [table for table in (*val_tables, *fold_tables) if not table.empty]
    train_rows = train_windows.assign(evidence=OFFICIAL_TRAIN, fold="", arm="", seed="", pred_label=None)
    windows = pd.concat([window_rows(table) for table in (*tables, train_rows)], ignore_index=True)
    return report, windows, snapshot


def _pct(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/d" if value is None else f"{value:.{digits}f}"


def _mean_pct(stats: dict) -> str:
    if stats["mean"] is None:
        return "n/d"
    spread = "" if stats["std"] is None else f" ± {stats['std']:.0%}"
    return f"{stats['mean']:.0%}{spread}"


def _support_lines(report: dict) -> list[str]:
    columns = [(name, support) for name, support in report["support"].items() if support is not None]
    lines = ["", "## Suporte por classe", "",
             "Janelas rotuladas: train oficial e treino das dobras em stride 4; val oficial e retidos em stride 1.",
             "", "| Classe | " + " | ".join(f"`{name}`" for name, _support in columns) + " |",
             "| --- |" + " ---: |" * len(columns)]
    for label in report["restricted_labels"]:
        lines.append(f"| {label} | " + " | ".join(str(support[label]) for _name, support in columns) + " |")
    return lines


def _train_lines(train: dict) -> list[str]:
    lines = ["", "## Composição do train oficial (stride 4)", "",
             "| Classe | Janelas (fração) | Vídeos | Sujeitos | Pose ausente | Transição | Interior | Confiança média "
             "dos keypoints | Trechos (mediana, quadros) | Trecho seguinte mais comum |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for label, block in train["classes"].items():
        pose = block["pose"]
        segments = block["segments"]
        following = next(iter(segments["following_label"].items()), ("n/d", 0))
        lines.append(
            f"| {label} | {block['n_windows']} ({_pct(block['share_of_labeled'])}) | {block['n_videos']} | "
            f"{', '.join(map(str, block['subjects']))} | {_pct(pose['pose_absent_share'])} | "
            f"{_pct(pose['transition_share'])} | {_pct(pose['interior_share'])} | {_num(pose['mean_keypoint_conf'])} | "
            f"{segments['n_segments']} ({_num(segments['median_length_frames'], 0)}) | {following[0]} "
            f"({following[1]}) |"
        )
    lines += ["", "Janelas por sujeito e por ambiente:", ""]
    for label, block in train["classes"].items():
        by_subject = ", ".join(f"{subject}: {count}" for subject, count in block["by_subject"].items())
        by_env = ", ".join(f"{env}: {count}" for env, count in block["by_environment"].items())
        lines.append(f"- `{label}`: sujeitos {by_subject}; ambientes {by_env}.")
    return lines


def _stratum_rate(rows: list[dict], name: str) -> str:
    row = next((row for row in rows if row["stratum"] == name), None)
    if row is None:
        return "—"
    return f"{_mean_pct(row['error_rate'])} ({row['n_true']})"


def _val_lines(report: dict, key: str) -> list[str]:
    arms = report["official_val"]["arms"]
    if not arms:
        return ["", "Sem runs do val oficial."]
    lines = ["", "#### Val oficial (média ± desvio entre seeds)", "",
             "| Braço | Seeds | Erros / janelas | Taxa de erro | Pose ausente | Transição | Interior | Erro no train "
             "(`final.train`) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, block in arms.items():
        pair = block["pairs"][key]
        category = pair["strata"]["category"]
        lines.append(
            f"| {arm} | {block['n_seeds']} | {_num(pair['n_error']['mean'], 1)}/{pair['n_true']} | "
            f"{_mean_pct(pair['error_rate'])} | {_stratum_rate(category, POSE_ABSENT)} | "
            f"{_stratum_rate(category, TRANSITION)} | {_stratum_rate(category, INTERIOR)} | "
            f"{_mean_pct(pair['train_fit_error_rate'])} |"
        )
    environments = [row["stratum"] for row in next(iter(arms.values()))["pairs"][key]["strata"]["environment"]]
    lines += ["", "Taxa de erro por ambiente (janelas verdadeiras):", "",
              "| Ambiente | " + " | ".join(arms) + " |", "| --- |" + " ---: |" * len(arms)]
    for env in environments:
        lines.append(f"| {env} | " + " | ".join(_stratum_rate(block["pairs"][key]["strata"]["environment"], env)
                                                 for block in arms.values()) + " |")
    lines += ["", "Destino dos erros da classe verdadeira (média de janelas entre seeds):", ""]
    for arm, block in arms.items():
        destinations = ", ".join(f"{label} {stats['mean']:.1f}" for label, stats in
                                 sorted(block["pairs"][key]["destinations"].items(),
                                        key=lambda item: -(item[1]["mean"] or 0.0)))
        lines.append(f"- {arm}: {destinations}.")
    lines += ["", "Perfil de pose, erros / acertos (média entre seeds):", "",
              "| Braço | Fração de quadros com pose (mediana) | Confiança média dos keypoints | Quadros até o fim do "
              "trecho (mediana) |", "| --- | ---: | ---: | ---: |"]
    for arm, block in arms.items():
        error, correct = block["pairs"][key]["profiles"][ERROR], block["pairs"][key]["profiles"][CORRECT]
        lines.append(
            f"| {arm} | {_num(error['median_detected_fraction']['mean'])} / "
            f"{_num(correct['median_detected_fraction']['mean'])} | {_num(error['mean_keypoint_conf']['mean'])} / "
            f"{_num(correct['mean_keypoint_conf']['mean'])} | {_num(error['median_frames_to_segment_end']['mean'], 1)}"
            f" / {_num(correct['median_frames_to_segment_end']['mean'], 1)} |"
        )
    return lines


def _fold_lines(report: dict, key: str) -> list[str]:
    folds = report["subject_folds"]["folds"]
    if not folds:
        return ["", "Sem evidência das dobras por sujeito."]
    lines = ["", "#### Dobras por sujeito (braço A, seed 42, grupo retido)", "",
             "| Dobra / sujeito | Erros / janelas | Pose ausente | Transição | Interior | Erro no treino da dobra |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for block in folds:
        pair = block["pairs"][key]
        fit = block["train_fit"][key]
        for unit, counts in pair["units"].items():
            cells = [f"{_pct(_share(counts[name]['n_error'], counts[name]['n_true']))} ({counts[name]['n_true']})"
                     for name in CATEGORIES]
            lines.append(f"| {block['fold']} / {unit} | {counts['n_error']}/{counts['n_true']} | "
                         + " | ".join(cells) + f" | {_pct(fit['error_rate'])} ({fit['n_true']}) |")
    return lines


def _rule_lines(decision: dict, key: str) -> list[str]:
    block = decision["pairs"][key]
    support = block["support"]
    lines = ["", "#### Regras", "",
             f"- R0, suporte: {support['train_windows']} janelas no train ({_pct(support['train_share'])}), sujeitos "
             f"{', '.join(map(str, support['train_subjects']))}; unidades com {MIN_UNIT_WINDOWS}+ janelas: "
             f"{len(support['eligible_units'])} ({', '.join(support['eligible_units']) or 'nenhuma'}); "
             f"{'limitado' if support['limited'] else 'não limitado'}."]
    for name in ("transition", "pose_absence"):
        rule = block[name]
        lines.append(f"- {rule['rule']}: {rule['n_agreeing_units']} de {rule['n_eligible_units']} unidades elegíveis "
                     f"atendem; `{rule['status']}`.")
        for unit in rule["units"]:
            rates = "; ".join(
                f"{model['model']} {_pct(model['focus_rate'])} contra {_pct(model['reference_rate'])}, "
                f"{_pct(model['focus_error_share'])} dos erros" for model in unit["models"]
            )
            state = ("elegível, atende" if unit["met"] else "elegível, não atende") if unit["eligible"] \
                else "não elegível"
            lines.append(f"  - {unit['unit']} ({unit['focus_windows']}/{unit['reference_windows']} janelas; {state}): "
                         f"{rates}.")
    lines += ["", f"**Veredito do par: `{block['verdict']}`.**"]
    return lines


def render_summary(report: dict) -> str:
    decision = report["decision"]
    lines = [
        f"# Erros {_arrow(report['focus_pairs'][0])} e {_arrow(report['focus_pairs'][1])} (Etapa 6A, não canônico)",
        "",
        "Diagnóstico exploratório e somente leitura de Le2i-CS sobre train oficial, val oficial e grupos retidos "
        "das dobras por sujeito. O test oficial não é lido. Predições vêm de reexecuções que reproduzem "
        "exatamente as matrizes gravadas; categorias e regras descrevem associações, não causas.",
        "",
        f"**Veredito: `{decision['verdict']}`.**",
    ]
    if decision["candidate_intervention"]:
        lines += ["", f"Intervenção candidata, não implementada: `{CANDIDATE_INTERVENTION['name']}` — "
                      f"{CANDIDATE_INTERVENTION['description']}"]
    lines += _support_lines(report)
    lines += _train_lines(report["train_composition"])
    for key in report["focus_pairs"]:
        lines += ["", f"## {_arrow(key)}"]
        lines += _val_lines(report, key)
        lines += _fold_lines(report, key)
        lines += _rule_lines(decision, key)
    lines += ["", "## Evidência ausente", ""]
    lines += [f"- {item['evidence']} ({item['source']}): {item['reason']}"
              + (f" — {', '.join(item['missing_files'])}" if item["missing_files"] else "")
              for item in report["missing_evidence"]]
    lines += ["", "## Limitações", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def protected_paths(runs: list[DiagnosisInput], fold_root: Path, frames_path: Path, pose_root: Path) -> list[Path]:
    return [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        REPOSITORY_ROOT / "docs",
        frames_path.parent,
        pose_root,
        fold_root,
        *(item.path for item in runs),
    ]


def check_diagnosis_output_dir(output_dir: Path, protected: list[Path]) -> None:
    check_output_dir(output_dir, protected)
    resolved = output_dir.resolve()
    local_root = LOCAL_OUTPUT_ROOT.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT) and not (resolved == local_root or local_root in resolved.parents):
        raise ValueError(f"--output-dir {output_dir} dentro do repositório precisa ficar em {LOCAL_OUTPUT_ROOT}")


def pose_files(frames_path: Path, pose_root: Path) -> list[Path]:
    frames = pd.read_parquet(frames_path, columns=["video_id", "split"])
    video_ids = sorted({str(video_id) for video_id in
                        frames.loc[frames["split"].isin([OFFICIAL_TRAIN_SPLIT, VAL_SPLIT]), "video_id"]})
    return [pose_path(video_id, pose_root=pose_root) for video_id in video_ids]


def raw_pose_loader(pose_root: Path) -> PoseLoader:
    cache: dict[str, PoseArrays] = {}

    def load(video_id: str) -> PoseArrays:
        if video_id not in cache:
            cache[video_id] = load_pose(video_id, pose_root=pose_root)
        return cache[video_id]

    return load


def validate_diagnosis_runs(runs: list[DiagnosisInput]) -> None:
    validate_runs(runs)
    unsupported = sorted({item.dataset for item in runs if item.dataset != SUPPORTED_DATASET})
    if unsupported:
        raise ValueError(f"a Etapa 6A cobre só Le2i-CS ({SUPPORTED_DATASET!r}); recebido: {unsupported}")


def run_diagnosis(
    runs: list[DiagnosisInput],
    output_dir: Path,
    fold_root: Path = SUBJECT_FOLD_ROOT,
    frames_path: Path = ADAPTER.frames_path,
    manifest_path: Path = ADAPTER.manifest_path,
    pose_root: Path = ADAPTER.pose_root,
    force: bool = False,
    load: PoseLoader | None = None,
    pose: PoseInputs | None = None,
) -> bool:
    validate_diagnosis_runs(runs)
    check_diagnosis_output_dir(output_dir, protected_paths(runs, fold_root, frames_path, pose_root))
    targets = [output_dir / name for name in (REPORT_JSON_FILE, REPORT_SUMMARY_FILE, REPORT_WINDOWS_FILE)]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not force:
        print(f"skip {', '.join(existing)} (já existe, use --force para sobrescrever)")
        return False
    pose_snapshot: dict[str, str] = {}
    if load is None:
        files = pose_files(frames_path, pose_root)
        absent = [str(path) for path in files if not path.is_file()]
        if absent:
            raise FileNotFoundError(f"pose de train/val ausente: {absent[:5]}")
        pose_snapshot = snapshot_inputs([path.resolve() for path in files])
        load = raw_pose_loader(pose_root)
    report, windows, snapshot = build_report(runs, fold_root, frames_path, manifest_path, pose_root, load, pose)
    snapshot.update(pose_snapshot)
    report["input_sha256"] = {display_path(Path(path)): digest for path, digest in sorted(snapshot.items())}
    outputs = {
        targets[0]: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        targets[1]: render_summary(report),
        targets[2]: windows.to_csv(index=False),
    }
    verify_inputs_unchanged(snapshot)
    return write_text_outputs(output_dir, outputs, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Diagnostica standing→walk e sitting→stand_up no train, val e dobras por sujeito do Le2i-CS"
    )
    diagnose_parser.add_argument(
        "--run", nargs=3, action="append", metavar=("DATASET", "ARM", "RUN_DIR"), required=True,
        help="Repetível: run local de Le2i-CS com config.yaml, metrics.json e checkpoint.pt",
    )
    diagnose_parser.add_argument("--fold-root", type=Path, default=SUBJECT_FOLD_ROOT)
    diagnose_parser.add_argument("--output-dir", type=Path, required=True)
    diagnose_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do diagnóstico standing/sitting")

    args = parser.parse_args()
    if args.command == "diagnose":
        run_diagnosis(_parse_inputs(args.run), args.output_dir, fold_root=args.fold_root, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.standing_sitting_diagnosis import run_selftest

        if not run_selftest():
            sys.exit(1)


if __name__ == "__main__":
    main()
