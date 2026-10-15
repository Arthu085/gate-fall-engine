"""Composição temporal e sinais de pose de `other` e `walk` no treino de Le2i-CS (Etapa 3C).

Diagnóstico descritivo, somente leitura e não canônico. Usa só as linhas
`split == "train"` de `frames.parquet` e o HDF5 bruto de pose dos vídeos de
treino; não abre features, predições nem pose de `val` ou `test`, não
treina, não altera rótulos, janelamento, checkpoints ou decisões
experimentais.

Unidades: o trecho anotado (sequência contígua máxima de quadros da grade
com o mesmo rótulo, `IGNORE_LABEL` inclusive) e a janela de treino, isto é,
cada linha de `build_window_index` em `TRAIN_STRIDE` com rótulo `other` ou
`walk` no quadro final, como no treino dos braços. O contexto da janela
reaproveita `video_context` da Etapa 3A.

Os sinais de pose descrevem o artefato disponível, não a atividade humana:
detecção (`person_found`), confiança média dos keypoints detectados e o
deslocamento líquido do centro da bbox entre o primeiro índice da janela e o
quadro final, em alturas medianas da bbox. As regras de decisão são fixas
neste módulo (`decision_criteria`) e foram definidas antes da execução
sobre os dados reais.
"""

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, TARGET_FPS, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.windowing import build_window_index, window_frame_indices
from gatefall.datasets import get_dataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import (
    PROTOCOL_NAMES,
    check_output_dir,
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.other_walk_review_set import LOCAL_OUTPUT_ROOT
from gatefall.eval.analysis.window_context_diagnosis import (
    MIXED,
    PADDED,
    POSE_DETECTED,
    PURE,
    VIDEO_START,
    _segments,
    label_name,
    video_context,
)
from gatefall.pose.loading import PoseArrays, load_pose, pose_path
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT

SUPPORTED_DATASET = "le2i"
SPLIT = "train"
FOCUS_LABELS: tuple[str, ...] = ("other", "walk")
VIDEO_END = "video_end"

COMPOSITION_JSON_FILE = "other_walk_train_composition.json"
COMPOSITION_SUMMARY_FILE = "other_walk_train_composition.md"
COMPOSITION_WINDOWS_FILE = "other_walk_train_windows.csv"
COMPOSITION_SEGMENTS_FILE = "other_walk_train_segments.csv"

# Trecho com menos de WINDOW_FRAMES quadros não gera janela pura.
SHORT_SEGMENT_FRAMES = WINDOW_FRAMES
# Deslocamento em alturas medianas da bbox. O primeiro intervalo define o
# deslocamento baixo das regras de decisão.
DISPLACEMENT_BINS: tuple[tuple[float, float | None], ...] = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, None))
LOW_DISPLACEMENT_MAX = 0.25
UNDEFINED = "undefined"
QUANTILES: tuple[tuple[str, float], ...] = (
    ("min", 0.0), ("p10", 0.1), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p90", 0.9), ("max", 1.0),
)

MIN_POOLED_DIFFERENCE = 0.10
MIN_SUBJECT_WINDOWS = 10
MIN_ELIGIBLE_SUBJECTS = 3
MIN_SUBJECT_AGREEMENT = 2 / 3
SUPPORTED_VERDICT = "consider_controlled_intervention"
UNSUPPORTED_VERDICT = "no_model_modification"

SEGMENT_COLUMNS = [
    "video_id", "env", "subject", "label", "segment_start", "segment_end", "segment_length", "segment_seconds",
    "short", "preceding_label", "following_label", "n_train_windows", "n_pure_train_windows",
]
WINDOW_COLUMNS = [
    "video_id", "env", "subject", "label", "k_end", "purity", "n_final_label_frames", "segment_start",
    "segment_end", "segment_length", "frames_since_segment_start", "start_distance", "preceding_label",
    "n_replicated_frames", "edge_padding", "pose_final", "detected_fraction", "mean_keypoint_conf",
    "displacement", "displacement_bin",
]
CATEGORICAL_DIMENSIONS: tuple[str, ...] = (
    "purity", "start_distance", "preceding_label", "edge_padding", "pose_final", "displacement_bin",
)

PoseLoader = Callable[[str], PoseArrays]


def displacement_bin(value: float | None) -> str:
    if value is None or math.isnan(value):
        return UNDEFINED
    for low, high in DISPLACEMENT_BINS:
        if value >= low and (high is None or value < high):
            return f"{low:.2f}+" if high is None else f"{low:.2f}-{high:.2f}"
    raise ValueError(f"deslocamento negativo: {value}")


def bbox_displacement(bbox: np.ndarray, person_found: np.ndarray, first: int, last: int) -> float | None:
    """Distância entre os centros da bbox em `first` e `last`, em alturas medianas da bbox detectada no intervalo."""
    if last <= first or not (person_found[first] and person_found[last]):
        return None
    span = np.arange(first, last + 1)
    detected = span[person_found[span]]
    height = float(np.median(bbox[detected, 3] - bbox[detected, 1]))
    if not height > 0:
        return None
    centers = (bbox[[first, last], :2] + bbox[[first, last], 2:]) / 2.0
    return float(np.linalg.norm(centers[1] - centers[0]) / height)


def window_pose_signals(pose: PoseArrays, k_end: int) -> dict:
    indices = window_frame_indices(k_end, pose.k)
    found = pose.person_found[indices].astype(bool)
    confidence = np.clip(pose.keypoints[indices][found, :, 2], 0.0, 1.0)
    displacement = bbox_displacement(pose.bbox, pose.person_found.astype(bool), int(indices[0]), k_end)
    return {
        "detected_fraction": float(found.mean()),
        "mean_keypoint_conf": float(confidence.mean()) if confidence.size else None,
        "displacement": displacement,
        "displacement_bin": displacement_bin(displacement),
    }


def video_segments(labels: np.ndarray, train_k_ends: np.ndarray, context: pd.DataFrame) -> list[dict]:
    starts, ends, _segment_of = _segments(labels)
    pure_k_ends = set(context.loc[context["purity"] == PURE, "k_end"].astype(int))
    rows = []
    for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        label = int(labels[start])
        if label == IGNORE_LABEL or label_name(label) not in FOCUS_LABELS:
            continue
        inside = [int(k_end) for k_end in train_k_ends if start <= k_end <= end]
        length = end - start + 1
        rows.append({
            "label": label_name(label),
            "segment_start": start,
            "segment_end": end,
            "segment_length": length,
            "segment_seconds": length / TARGET_FPS,
            "short": length < SHORT_SEGMENT_FRAMES,
            "preceding_label": VIDEO_START if index == 0 else label_name(int(labels[starts[index - 1]])),
            "following_label": VIDEO_END if index == len(starts) - 1 else label_name(int(labels[starts[index + 1]])),
            "n_train_windows": len(inside),
            "n_pure_train_windows": sum(k_end in pure_k_ends for k_end in inside),
        })
    return rows


def video_tables(labels: np.ndarray, pose: PoseArrays) -> tuple[list[dict], list[dict]]:
    if pose.k != len(labels) or pose.person_found.shape != labels.shape or pose.bbox.shape[0] != len(labels):
        raise ValueError(f"pose com K={pose.k} e person_found {pose.person_found.shape}; a grade tem {len(labels)} quadros")
    person_found = pose.person_found.astype(bool)
    context = video_context(labels, person_found)
    train_k_ends = np.arange(0, len(labels), TRAIN_STRIDE, dtype=np.int64)
    context = cast(pd.DataFrame, context[context["k_end"].isin(train_k_ends.tolist())])
    windows = []
    for row in context.to_dict("records"):
        label = int(labels[int(row["k_end"])])
        if label == IGNORE_LABEL or label_name(label) not in FOCUS_LABELS:
            continue
        windows.append({**row, "label": label_name(label), **window_pose_signals(pose, int(row["k_end"]))})
    return video_segments(labels, train_k_ends, context), windows


def train_tables(frames: pd.DataFrame, load: PoseLoader) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = cast(pd.DataFrame, frames[frames["split"] == SPLIT])
    if train.empty:
        raise ValueError("frames.parquet não tem quadros de train")
    segments: list[dict] = []
    windows: list[dict] = []
    for video_id, group in train.groupby("video_id", sort=True):
        ordered = cast(pd.DataFrame, group.sort_values("frame_index"))
        labels = ordered["label"].to_numpy(dtype=np.int64)
        if not np.array_equal(ordered["frame_index"].to_numpy(), np.arange(len(labels))):
            raise ValueError(f"video_id={video_id!r}: frame_index não é contíguo 0..K-1")
        identity = {"video_id": str(video_id), "env": str(ordered["env"].iloc[0]),
                    "subject": int(ordered["subject"].iloc[0])}
        try:
            video_segments_rows, video_window_rows = video_tables(labels, load(str(video_id)))
        except ValueError as exc:
            raise ValueError(f"video_id={video_id!r}: {exc}") from exc
        segments.extend({**identity, **row} for row in video_segments_rows)
        windows.extend({**identity, **row} for row in video_window_rows)
    segment_table = pd.DataFrame(segments, columns=SEGMENT_COLUMNS)
    window_table = pd.DataFrame(windows, columns=WINDOW_COLUMNS)
    reconcile_windows(window_table, train)
    return segment_table.reset_index(drop=True), window_table.reset_index(drop=True)


def reconcile_windows(windows: pd.DataFrame, train: pd.DataFrame) -> None:
    focus_ids = [LE2I_LABEL_NAMES.index(label) for label in FOCUS_LABELS]
    index = build_window_index(train, stride=TRAIN_STRIDE, drop_ignored=True)
    index = cast(pd.DataFrame, index[index["label"].isin(focus_ids)])
    expected = sorted(zip(index["video_id"].astype(str), index["k_end"].astype(int),
                          (label_name(int(label)) for label in index["label"])))
    observed = sorted(zip(windows["video_id"].astype(str), windows["k_end"].astype(int), windows["label"].astype(str)))
    if expected != observed:
        raise ValueError(f"janelas não alinham com build_window_index em TRAIN_STRIDE: "
                         f"{len(observed)} calculadas, {len(expected)} esperadas")


def _share(part: int, total: int) -> float | None:
    return part / total if total > 0 else None


def _quantiles(values: pd.Series) -> dict:
    data = values.dropna().to_numpy(dtype=np.float64)
    if data.size == 0:
        return {name: None for name, _q in QUANTILES}
    return {name: float(np.quantile(data, q)) for name, q in QUANTILES}


def _mean(values: pd.Series) -> float | None:
    data = values.dropna().to_numpy(dtype=np.float64)
    return float(data.mean()) if data.size else None


def _column(table: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, table[name])


def _count(mask: pd.Series) -> int:
    return int(mask.to_numpy(dtype=bool).sum())


def _tally(values: pd.Series) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(values.astype(str).value_counts().items())}


def segment_summary(segments: pd.DataFrame) -> dict:
    lengths = _column(segments, "segment_length").astype(np.int64)
    short = cast(pd.DataFrame, segments[segments["short"].astype(bool)])
    n_frames = int(lengths.to_numpy().sum())
    return {
        "n_segments": len(segments),
        "n_frames": n_frames,
        "n_videos": _column(segments, "video_id").nunique(),
        "n_subjects": _column(segments, "subject").nunique(),
        "length_frames": _quantiles(lengths),
        "n_short": len(short),
        "short_share": _share(len(short), len(segments)),
        "short_frames_share": _share(int(_column(short, "segment_length").to_numpy().sum()), n_frames),
        "segments_with_pure_window": _count(_column(segments, "n_pure_train_windows") > 0),
        "preceding_label": _tally(_column(segments, "preceding_label")),
        "following_label": _tally(_column(segments, "following_label")),
    }


def window_summary(windows: pd.DataFrame) -> dict:
    defined = cast(pd.DataFrame, windows[windows["displacement_bin"] != UNDEFINED])
    return {
        "n_windows": len(windows),
        "n_videos": _column(windows, "video_id").nunique(),
        "n_subjects": _column(windows, "subject").nunique(),
        "counts": {dimension: _tally(_column(windows, dimension)) for dimension in CATEGORICAL_DIMENSIONS},
        "pure_share": _share(_count(_column(windows, "purity") == PURE), len(windows)),
        "edge_padded_share": _share(_count(_column(windows, "edge_padding") == PADDED), len(windows)),
        "pose_final_detected_share": _share(_count(_column(windows, "pose_final") == POSE_DETECTED), len(windows)),
        "mean_detected_fraction": _mean(_column(windows, "detected_fraction")),
        "mean_keypoint_conf": _mean(_column(windows, "mean_keypoint_conf")),
        "frames_since_segment_start": _quantiles(_column(windows, "frames_since_segment_start")),
        "displacement_defined": len(defined),
        "low_displacement_share": _share(_low_count(defined), len(defined)),
        "displacement": {
            "all": _quantiles(_column(windows, "displacement")),
            **{purity: _quantiles(cast(pd.Series, windows.loc[windows["purity"] == purity, "displacement"]))
               for purity in (PURE, MIXED)},
        },
        "displacement_bin_by_purity": {
            purity: _tally(cast(pd.Series, windows.loc[windows["purity"] == purity, "displacement_bin"]))
            for purity in (PURE, MIXED)
        },
    }


def _low_count(windows: pd.DataFrame) -> int:
    return _count(_column(windows, "displacement").astype(np.float64) < LOW_DISPLACEMENT_MAX)


def _pure_low_share(windows: pd.DataFrame) -> tuple[int, float | None]:
    pure = cast(pd.DataFrame, windows[(windows["purity"] == PURE) & (windows["displacement_bin"] != UNDEFINED)])
    return len(pure), _share(_low_count(pure), len(pure))


def _pure_share(windows: pd.DataFrame) -> tuple[int, float | None]:
    return len(windows), _share(_count(_column(windows, "purity") == PURE), len(windows))


def _pose_share(windows: pd.DataFrame) -> tuple[int, float | None]:
    return len(windows), _share(_count(_column(windows, "pose_final") == POSE_DETECTED), len(windows))


ShareFunction = Callable[[pd.DataFrame], tuple[int, float | None]]
# (chave, descrição, fração comparada, condição sobre Δ = other − walk).
RULES: tuple[tuple[str, str, ShareFunction, str], ...] = (
    ("R1_temporal_composition", "fração de janelas puras", _pure_share, "le_neg"),
    ("R2_low_displacement_pure", "fração de deslocamento baixo entre janelas puras com deslocamento definido",
     _pure_low_share, "lt_pos"),
    ("R3_pose_availability", "fração de janelas com pose no quadro final", _pose_share, "le_neg"),
)
CONDITION_TEXT = {
    "le_neg": f"Δ ≤ −{MIN_POOLED_DIFFERENCE:.2f}",
    "lt_pos": f"Δ < +{MIN_POOLED_DIFFERENCE:.2f}",
}


def _condition_met(condition: str, delta: float | None) -> bool:
    if delta is None:
        return False
    if condition == "le_neg":
        return delta <= -MIN_POOLED_DIFFERENCE + 1e-12
    return delta < MIN_POOLED_DIFFERENCE - 1e-12


def _delta(windows: pd.DataFrame, share: ShareFunction) -> tuple[dict, float | None]:
    values = {}
    for label in FOCUS_LABELS:
        n, value = share(cast(pd.DataFrame, windows[windows["label"] == label]))
        values[label] = {"n": n, "share": value}
    first, second = (values[label]["share"] for label in FOCUS_LABELS)
    return values, None if first is None or second is None else first - second


def evaluate_rule(windows: pd.DataFrame, key: str, description: str, share: ShareFunction, condition: str) -> dict:
    pooled, delta = _delta(windows, share)
    subjects = []
    for subject, group in windows.groupby("subject", sort=True):
        values, subject_delta = _delta(cast(pd.DataFrame, group), share)
        eligible = all(values[label]["n"] >= MIN_SUBJECT_WINDOWS for label in FOCUS_LABELS)
        subjects.append({"subject": int(cast(int, subject)), "values": values, "delta": subject_delta,
                         "eligible": eligible, "condition_met": eligible and _condition_met(condition, subject_delta)})
    n_eligible = sum(item["eligible"] for item in subjects)
    n_agree = sum(item["condition_met"] for item in subjects)
    pooled_met = _condition_met(condition, delta)
    supported = (pooled_met and n_eligible >= MIN_ELIGIBLE_SUBJECTS
                 and n_agree >= MIN_SUBJECT_AGREEMENT * n_eligible - 1e-12)
    return {
        "rule": key,
        "description": description,
        "condition": CONDITION_TEXT[condition],
        "pooled": pooled,
        "delta": delta,
        "pooled_condition_met": pooled_met,
        "subjects": subjects,
        "n_eligible_subjects": n_eligible,
        "n_agreeing_subjects": n_agree,
        "supported": supported,
    }


def decision_criteria() -> dict:
    return {
        "delta": f"{FOCUS_LABELS[0]} − {FOCUS_LABELS[1]}",
        "min_pooled_difference": MIN_POOLED_DIFFERENCE,
        "low_displacement_max_body_heights": LOW_DISPLACEMENT_MAX,
        "min_subject_windows_per_label": MIN_SUBJECT_WINDOWS,
        "min_eligible_subjects": MIN_ELIGIBLE_SUBJECTS,
        "min_subject_agreement": MIN_SUBJECT_AGREEMENT,
        "rules": [{"rule": key, "description": description, "condition": CONDITION_TEXT[condition]}
                  for key, description, _share_fn, condition in RULES],
        "supported_rule": (
            "a condição vale no agregado do treino, há pelo menos min_eligible_subjects sujeitos com "
            "min_subject_windows_per_label janelas de cada rótulo no denominador e a mesma condição vale em pelo "
            "menos min_subject_agreement desses sujeitos"
        ),
        "verdict": (
            f"{SUPPORTED_VERDICT} se ao menos uma regra for sustentada; caso contrário, {UNSUPPORTED_VERDICT}"
        ),
    }


def decide(rules: list[dict]) -> dict:
    supported = [rule["rule"] for rule in rules if rule["supported"]]
    return {"verdict": SUPPORTED_VERDICT if supported else UNSUPPORTED_VERDICT, "supported_rules": supported}


def group_table(segments: pd.DataFrame, windows: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for value in sorted({*segments[column].tolist(), *windows[column].tolist()}):
        entry: dict = {column: value}
        for label in FOCUS_LABELS:
            labeled = cast(pd.DataFrame, windows[(windows[column] == value) & (windows["label"] == label)])
            group_segments = cast(pd.DataFrame, segments[(segments[column] == value) & (segments["label"] == label)])
            n_pure_defined, low_share = _pure_low_share(labeled)
            entry[label] = {
                "n_segments": len(group_segments),
                "n_short_segments": _count(_column(group_segments, "short")),
                "n_windows": len(labeled),
                "pure_share": _pure_share(labeled)[1],
                "pose_final_detected_share": _pose_share(labeled)[1],
                "n_pure_displacement_defined": n_pure_defined,
                "pure_low_displacement_share": low_share,
            }
        rows.append(entry)
    return rows


def conservation(segments: pd.DataFrame, windows: pd.DataFrame, frames: pd.DataFrame,
                 summaries: dict, subjects: list[dict], environments: list[dict]) -> dict:
    train = cast(pd.DataFrame, frames[frames["split"] == SPLIT])
    checks = {}
    for label in FOCUS_LABELS:
        label_id = LE2I_LABEL_NAMES.index(label)
        n_frames = _count(_column(train, "label") == label_id)
        segment_block, window_block = summaries[label]["segments"], summaries[label]["windows"]
        labeled_segments = cast(pd.DataFrame, segments[segments["label"] == label])
        dimension_sums = {dimension: sum(counts.values()) for dimension, counts in window_block["counts"].items()}
        checks[label] = {
            "train_frames": n_frames,
            "segment_frames": segment_block["n_frames"],
            "train_windows": window_block["n_windows"],
            "segment_windows": int(_column(labeled_segments, "n_train_windows").to_numpy().sum()),
            "subject_windows": sum(entry[label]["n_windows"] for entry in subjects),
            "environment_windows": sum(entry[label]["n_windows"] for entry in environments),
            "dimension_sums": dimension_sums,
        }
        block = checks[label]
        consistent = (
            block["segment_frames"] == n_frames
            and block["segment_windows"] == block["train_windows"]
            and block["subject_windows"] == block["train_windows"]
            and block["environment_windows"] == block["train_windows"]
            and all(total == block["train_windows"] for total in dimension_sums.values())
        )
        if not consistent:
            raise ValueError(f"conservação violada para {label}: {block}")
    return {"ok": True, **checks}


def _limitations() -> list[str]:
    return [
        "Análise exploratória, descritiva e não canônica (Etapa 3C): usa só train de Le2i-CS, não treina, não "
        "altera rótulos, janelamento, features, checkpoints nem decisões experimentais e não seleciona modelo, "
        "seed, limiar ou checkpoint.",
        "Os erros da Etapa 3B motivaram as perguntas, mas não são alvo de otimização: nenhuma regra usa test, "
        "e uma regra sustentada não mostra que a causa dos erros em test foi encontrada.",
        f"Janelas de treino em TRAIN_STRIDE {TRAIN_STRIDE} se sobrepõem; janelas e trechos do mesmo vídeo ou "
        "sujeito não são observações independentes. A consistência por sujeito é descritiva, sem teste de "
        "significância.",
        "Os sinais de pose descrevem o artefato YOLO-Pose bruto, sem imputação: detecção não mede qualidade, e o "
        "deslocamento da bbox no plano da imagem depende da direção do movimento em relação à câmera, da "
        "oclusão e do enquadramento. Nenhum valor aqui identifica caminhada ou a atividade real.",
        "O deslocamento é indefinido quando falta pose no primeiro índice ou no quadro final da janela; em "
        "janelas com replicação de borda, ele cobre menos de WINDOW_FRAMES quadros.",
        "Os limiares das regras são convenções fixadas antes da execução, não valores estimados; outros "
        "limiares podem levar a outro veredito.",
        "A pose não é a entrada visual de B0, B1, C0 e C1; o diagnóstico não descreve o que cada braço usa para "
        "decidir.",
    ]


def build_report(frames: pd.DataFrame, load: PoseLoader) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    segments, windows = train_tables(frames, load)
    summaries = {
        label: {
            "segments": segment_summary(cast(pd.DataFrame, segments[segments["label"] == label])),
            "windows": window_summary(cast(pd.DataFrame, windows[windows["label"] == label])),
        }
        for label in FOCUS_LABELS
    }
    subjects = group_table(segments, windows, "subject")
    environments = group_table(segments, windows, "env")
    rules = [evaluate_rule(windows, key, description, share, condition)
             for key, description, share, condition in RULES]
    report = {
        "analysis": "other_walk_train_composition",
        "stage": "3C",
        "canonical": False,
        "read_only": True,
        "dataset": SUPPORTED_DATASET,
        "protocol": PROTOCOL_NAMES[SUPPORTED_DATASET],
        "split": SPLIT,
        "labels": list(FOCUS_LABELS),
        "train_stride": TRAIN_STRIDE,
        "window_frames": WINDOW_FRAMES,
        "target_fps": TARGET_FPS,
        "short_segment_frames": SHORT_SEGMENT_FRAMES,
        "displacement_bins": [displacement_bin(low) for low, _high in DISPLACEMENT_BINS],
        "criteria": decision_criteria(),
        "summaries": summaries,
        "subjects": subjects,
        "environments": environments,
        "rules": rules,
        "decision": decide(rules),
        "conservation": conservation(segments, windows, frames, summaries, subjects, environments),
        "limitations": _limitations(),
    }
    return report, segments, windows


def _pct(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/d" if value is None else f"{value:.{digits}f}"


def _signed(value: float | None) -> str:
    return "n/d" if value is None else f"{value * 100:+.0f} p.p."


def _label_columns(header: str) -> list[str]:
    return [f"| {header} | " + " | ".join(f"`{label}`" for label in FOCUS_LABELS) + " |",
            "| --- |" + " ---: |" * len(FOCUS_LABELS)]


def _row(name: str, values: list[str]) -> str:
    return f"| {name} | " + " | ".join(values) + " |"


def _count_rows(summaries: dict, dimension: str) -> list[str]:
    names = sorted({name for label in FOCUS_LABELS for name in summaries[label]["windows"]["counts"][dimension]})
    rows = []
    for name in names:
        cells = []
        for label in FOCUS_LABELS:
            block = summaries[label]["windows"]
            count = block["counts"][dimension].get(name, 0)
            cells.append(f"{count} ({_pct(_share(count, block['n_windows']))})")
        rows.append(_row(f"`{name}`", cells))
    return rows


def _group_lines(title: str, header: str, entries: list[dict], key: str) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        f"| {header} | " + " | ".join(
            f"`{label}` janelas (puras) | `{label}` pose final | `{label}` puras com deslocamento baixo"
            for label in FOCUS_LABELS) + " |",
        "| --- |" + " ---: |" * (3 * len(FOCUS_LABELS)),
    ]
    for entry in entries:
        cells = []
        for label in FOCUS_LABELS:
            block = entry[label]
            cells += [f"{block['n_windows']} ({_pct(block['pure_share'])})",
                      _pct(block["pose_final_detected_share"]),
                      f"{_pct(block['pure_low_displacement_share'])} de {block['n_pure_displacement_defined']}"]
        lines.append(_row(str(entry[key]), cells))
    return lines


def render_summary(report: dict) -> str:
    summaries = report["summaries"]
    segments = {label: summaries[label]["segments"] for label in FOCUS_LABELS}
    windows = {label: summaries[label]["windows"] for label in FOCUS_LABELS}
    lines = [
        "# Composição de `other` e `walk` no treino de Le2i-CS (Etapa 3C, não canônico)",
        "",
        f"Descritivo e somente leitura: só `{SPLIT}`, janelas de `build_window_index` em TRAIN_STRIDE "
        f"{TRAIN_STRIDE}, pose bruta sem imputação. Nada aqui identifica a atividade real.",
        "",
        "## Trechos anotados",
        "",
        *_label_columns("Medida"),
        _row("Trechos / quadros", [f"{segments[label]['n_segments']} / {segments[label]['n_frames']}"
                                   for label in FOCUS_LABELS]),
        _row("Vídeos / sujeitos", [f"{segments[label]['n_videos']} / {segments[label]['n_subjects']}"
                                   for label in FOCUS_LABELS]),
        _row("Duração mediana (p25–p75), quadros", [
            f"{_num(segments[label]['length_frames']['median'], 0)} "
            f"({_num(segments[label]['length_frames']['p25'], 0)}–{_num(segments[label]['length_frames']['p75'], 0)})"
            for label in FOCUS_LABELS]),
        _row(f"Trechos curtos (< {SHORT_SEGMENT_FRAMES} quadros)", [
            f"{segments[label]['n_short']} ({_pct(segments[label]['short_share'])}; "
            f"{_pct(segments[label]['short_frames_share'])} dos quadros)" for label in FOCUS_LABELS]),
        _row("Trechos com ao menos uma janela pura", [str(segments[label]["segments_with_pure_window"])
                                                      for label in FOCUS_LABELS]),
        "",
        "Rótulo do trecho anterior:",
        "",
        *_label_columns("Rótulo"),
    ]
    neighbors = sorted({name for label in FOCUS_LABELS for name in segments[label]["preceding_label"]})
    lines += [_row(f"`{name}`", [str(segments[label]["preceding_label"].get(name, 0)) for label in FOCUS_LABELS])
              for name in neighbors]
    lines += [
        "",
        "## Janelas de treino: composição temporal",
        "",
        *_label_columns("Medida"),
        _row("Janelas / vídeos / sujeitos", [
            f"{windows[label]['n_windows']} / {windows[label]['n_videos']} / {windows[label]['n_subjects']}"
            for label in FOCUS_LABELS]),
        _row("Janelas puras", [_pct(windows[label]["pure_share"]) for label in FOCUS_LABELS]),
        _row("Quadros desde o início do trecho, mediana", [
            _num(windows[label]["frames_since_segment_start"]["median"], 0) for label in FOCUS_LABELS]),
        _row("Replicação de borda", [_pct(windows[label]["edge_padded_share"]) for label in FOCUS_LABELS]),
        "",
        "Quadros desde o início do trecho:",
        "",
        *_label_columns("Intervalo"),
        *_count_rows(summaries, "start_distance"),
        "",
        "Rótulo do trecho anterior (por janela):",
        "",
        *_label_columns("Rótulo"),
        *_count_rows(summaries, "preceding_label"),
        "",
        "## Janelas de treino: sinais de pose disponíveis",
        "",
        *_label_columns("Medida"),
        _row("Pose no quadro final", [_pct(windows[label]["pose_final_detected_share"]) for label in FOCUS_LABELS]),
        _row("Fração média de quadros com pose", [_pct(windows[label]["mean_detected_fraction"])
                                                  for label in FOCUS_LABELS]),
        _row("Confiança média dos keypoints detectados", [_num(windows[label]["mean_keypoint_conf"])
                                                          for label in FOCUS_LABELS]),
        _row("Deslocamento definido", [str(windows[label]["displacement_defined"]) for label in FOCUS_LABELS]),
        _row(f"Deslocamento < {LOW_DISPLACEMENT_MAX:.2f} alturas", [_pct(windows[label]["low_displacement_share"])
                                                                    for label in FOCUS_LABELS]),
    ]
    for purity in ("all", PURE, MIXED):
        lines.append(_row(f"Deslocamento mediano (p25–p75), {purity}", [
            f"{_num(windows[label]['displacement'][purity]['median'])} "
            f"({_num(windows[label]['displacement'][purity]['p25'])}–"
            f"{_num(windows[label]['displacement'][purity]['p75'])})" for label in FOCUS_LABELS]))
    lines += [
        "",
        "Deslocamento da bbox em alturas, por pureza da janela:",
        "",
        "| Intervalo | " + " | ".join(f"`{label}` {purity}" for label in FOCUS_LABELS for purity in (PURE, MIXED))
        + " |",
        "| --- |" + " ---: |" * (2 * len(FOCUS_LABELS)),
    ]
    for name in [*report["displacement_bins"], UNDEFINED]:
        lines.append(_row(f"`{name}`", [str(windows[label]["displacement_bin_by_purity"][purity].get(name, 0))
                                        for label in FOCUS_LABELS for purity in (PURE, MIXED)]))
    lines += _group_lines("Por sujeito", "Sujeito", report["subjects"], "subject")
    lines += _group_lines("Por ambiente (descritivo, fora das regras)", "Ambiente", report["environments"], "env")
    criteria = report["criteria"]
    lines += [
        "",
        "## Regras de decisão (fixadas antes da execução)",
        "",
        f"Δ = {criteria['delta']}. Uma regra é sustentada quando a condição vale no agregado, há pelo menos "
        f"{MIN_ELIGIBLE_SUBJECTS} sujeitos com {MIN_SUBJECT_WINDOWS} ou mais janelas de cada rótulo no "
        f"denominador e a mesma condição vale em pelo menos {MIN_SUBJECT_AGREEMENT:.0%} deles.",
        "",
        "| Regra | Medida | Condição | `other` | `walk` | Δ | Sujeitos elegíveis | Sujeitos com a condição | "
        "Sustentada |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rule in report["rules"]:
        pooled = rule["pooled"]
        lines.append(
            f"| `{rule['rule']}` | {rule['description']} | {rule['condition']} | "
            f"{_pct(pooled['other']['share'])} de {pooled['other']['n']} | "
            f"{_pct(pooled['walk']['share'])} de {pooled['walk']['n']} | {_signed(rule['delta'])} | "
            f"{rule['n_eligible_subjects']} | {rule['n_agreeing_subjects']} | {'sim' if rule['supported'] else 'não'} |"
        )
    decision = report["decision"]
    lines += [
        "",
        f"Veredito: `{decision['verdict']}`"
        + (f" (regras sustentadas: {', '.join(decision['supported_rules'])})." if decision["supported_rules"]
           else "."),
        "",
        "## Limitações",
        "",
        *[f"- {text}" for text in report["limitations"]],
    ]
    return "\n".join(lines) + "\n"


def protected_paths(frames_path: Path, pose_root: Path) -> list[Path]:
    return [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        REPOSITORY_ROOT / "docs",
        frames_path.parent,
        pose_root,
    ]


def check_composition_output_dir(output_dir: Path, frames_path: Path, pose_root: Path) -> None:
    check_output_dir(output_dir, protected_paths(frames_path, pose_root))
    resolved = output_dir.resolve()
    local_root = LOCAL_OUTPUT_ROOT.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT) and not (resolved == local_root or local_root in resolved.parents):
        raise ValueError(f"--output-dir {output_dir} dentro do repositório precisa ficar em {LOCAL_OUTPUT_ROOT}")


def train_pose_files(frames: pd.DataFrame, pose_root: Path) -> list[Path]:
    video_ids = sorted({str(video_id) for video_id in frames.loc[frames["split"] == SPLIT, "video_id"]})
    return [pose_path(video_id, pose_root=pose_root) for video_id in video_ids]


def run_composition(frames_path: Path, pose_root: Path, output_dir: Path, force: bool = False) -> bool:
    check_composition_output_dir(output_dir, frames_path, pose_root)
    if not frames_path.is_file():
        raise FileNotFoundError(f"frames.parquet não encontrado: {frames_path}")
    frames_snapshot = snapshot_inputs([frames_path.resolve()])
    frames = pd.read_parquet(frames_path, columns=["video_id", "split", "env", "subject", "frame_index", "label"])
    pose_files = train_pose_files(frames, pose_root)
    missing = [str(path) for path in pose_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pose de train ausente: {missing[:5]}")
    snapshot = {**frames_snapshot, **snapshot_inputs(sorted(path.resolve() for path in pose_files))}
    report, segments, windows = build_report(frames, lambda video_id: load_pose(video_id, pose_root=pose_root))
    report["input_sha256"] = {display_path(Path(path)): digest for path, digest in snapshot.items()}
    targets = {
        output_dir / COMPOSITION_JSON_FILE: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        output_dir / COMPOSITION_SUMMARY_FILE: render_summary(report),
        output_dir / COMPOSITION_SEGMENTS_FILE: segments.to_csv(index=False),
        output_dir / COMPOSITION_WINDOWS_FILE: windows.to_csv(index=False),
    }
    verify_inputs_unchanged(snapshot)
    return write_text_outputs(output_dir, targets, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Descreve trechos e janelas other e walk do treino de Le2i-CS"
    )
    diagnose_parser.add_argument("--output-dir", type=Path, required=True)
    diagnose_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da composição de treino other/walk")

    args = parser.parse_args()
    if args.command == "diagnose":
        adapter = get_dataset(SUPPORTED_DATASET)
        run_composition(adapter.frames_path, adapter.pose_root, args.output_dir, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.other_walk_train_composition import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
