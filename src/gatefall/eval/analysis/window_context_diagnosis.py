"""Contexto de janela, somente leitura, dos erros `other→walk` compartilhados em Le2i-CS (Etapa 3A).

Reaproveita o caminho validado do diagnóstico estratificado
(`validate_run`, `analyze_split_with_table`): identidade do run, SHA-256 do
checkpoint, inferência pelo avaliador de eventos de cada braço, alinhamento
exato de `(video_id, k_end)` com `build_window_index` e reconciliação das
matrizes com `metrics.json`. Não retreina, não reextrai features, não altera
checkpoints e não publica `event_metrics.json`.

Para cada janela rotulada de `val` e `test`, o contexto é derivado só dos
rótulos por quadro de `frames.parquet` e de `person_found` do HDF5 de pose:
fração dos `WINDOW_FRAMES` índices de `window_frame_indices` (com os quadros
replicados pela borda) que carregam o rótulo final, rótulo do trecho
anterior, distância às fronteiras do trecho anotado, replicação de borda e
detecção de pose no quadro final. Um trecho é a sequência contígua máxima de
quadros com o mesmo rótulo, `IGNORE_LABEL` inclusive.

`other→walk` é comparado com os `other` corretamente classificados;
`standing→walk` e `sitting→stand_up` entram só como controles descritivos.
As contagens de cada par, somadas em qualquer estrato, precisam reproduzir a
célula correspondente de `confusion_matrix` em `metrics.json`. O resultado é
exploratório e não canônico, o split `test` já exposto é só descritivo e
nenhuma associação aqui estabelece causa.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, TARGET_FPS, WINDOW_FRAMES
from gatefall.data.windowing import window_end_indices, window_frame_indices
from gatefall.datasets import get_dataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import (
    EXPECTED_SEEDS,
    PROTOCOL_NAMES,
    DiagnosisInput,
    _parse_inputs,
    check_output_dir,
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.multiseed_summary import ARMS, _aggregate_stats
from gatefall.eval.analysis.stratified_error_diagnosis import (
    SPLITS,
    analyze_split_with_table,
    load_evaluation,
    missing_prerequisites,
    protected_paths,
    run_input_files,
    validate_run,
    validate_runs,
)
from gatefall.pose.loading import load_person_found, pose_path

WINDOW_CONTEXT_JSON_FILE = "window_context_diagnosis.json"
WINDOW_CONTEXT_CSV_FILE = "window_context_diagnosis.csv"
WINDOW_CONTEXT_WINDOWS_FILE = "window_context_windows.csv"
WINDOW_CONTEXT_SUMMARY_FILE = "window_context_diagnosis.md"

SUPPORTED_DATASET = "le2i"
PRIMARY = "primary"
CONTROL = "control"
FOCUS_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("other", "walk", PRIMARY),
    ("standing", "walk", CONTROL),
    ("sitting", "stand_up", CONTROL),
)

PURE = "pure"
MIXED = "mixed"
PADDED = "padded"
UNPADDED = "unpadded"
POSE_DETECTED = "detected"
POSE_MISSING = "missing"
VIDEO_START = "video_start"
IGNORED_NAME = "IGNORE_LABEL"

ERROR = "error"
CORRECT = "correct"
OTHER_OUTCOME = "other_outcome"
COUNT_FIELDS: tuple[str, ...] = ("n_true", "n_error", "n_correct", "n_other_outcome")
SHARE_FIELDS: tuple[str, ...] = ("error_rate", "error_share", "correct_share")
PROFILE_FIELDS: tuple[str, ...] = (
    "mixed_share",
    "mean_final_label_fraction",
    "edge_padded_share",
    "pose_missing_share",
    "median_frames_since_segment_start",
    "median_frames_to_segment_end",
)

# Distâncias em quadros da grade (TARGET_FPS). No último intervalo, a janela
# inteira fica do mesmo lado da fronteira.
DISTANCE_BINS: tuple[tuple[int, int | None], ...] = (
    (0, 5),
    (6, 11),
    (12, WINDOW_FRAMES - 2),
    (WINDOW_FRAMES - 1, None),
)
DIMENSIONS: tuple[str, ...] = (
    "purity",
    "start_distance",
    "end_distance",
    "preceding_label",
    "edge_padding",
    "pose_final",
    "environment",
    "video",
    "environment_purity",
    "video_purity",
)

CONTEXT_COLUMNS = [
    "k_end",
    "n_final_label_frames",
    "final_label_fraction",
    "purity",
    "n_ignored_frames",
    "segment_start",
    "segment_end",
    "segment_length",
    "frames_since_segment_start",
    "frames_to_segment_end",
    "start_distance",
    "end_distance",
    "preceding_label",
    "n_replicated_frames",
    "edge_padding",
    "pose_final",
]
WINDOW_COLUMNS = ["split", "video_id", "env", "subject", "true_label", *CONTEXT_COLUMNS]
CSV_COLUMNS = [
    "record",
    "arm",
    "seed",
    "source",
    "split",
    "pair",
    "dimension",
    "stratum",
    "metric",
    "value",
]

PersonFoundLoader = Callable[[str], np.ndarray]


def pair_key(true_label: str, predicted_label: str) -> str:
    return f"{true_label}->{predicted_label}"


def _arrow(key: str) -> str:
    return key.replace("->", "→")


def label_name(label: int) -> str:
    return IGNORED_NAME if label == IGNORE_LABEL else LE2I_LABEL_NAMES[label]


def distance_bin(frames: int) -> str:
    for low, high in DISTANCE_BINS:
        if frames >= low and (high is None or frames <= high):
            return f"{low:02d}+" if high is None else f"{low:02d}-{high:02d}"
    raise ValueError(f"distância negativa: {frames}")


def _segments(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], boundaries)).astype(np.int64)
    ends = np.concatenate((boundaries - 1, [len(labels) - 1])).astype(np.int64)
    segment_of = np.repeat(np.arange(len(starts)), ends - starts + 1)
    return starts, ends, segment_of


def video_context(labels: np.ndarray, person_found: np.ndarray) -> pd.DataFrame:
    n_frames = len(labels)
    starts, ends, segment_of = _segments(labels)
    rows = []
    for k_end in (int(value) for value in window_end_indices(n_frames, EVAL_STRIDE)):
        indices = window_frame_indices(k_end, n_frames)
        window_labels = labels[indices]
        final_label = int(labels[k_end])
        segment = int(segment_of[k_end])
        start, end = int(starts[segment]), int(ends[segment])
        n_final = int(np.sum(window_labels == final_label))
        n_replicated = WINDOW_FRAMES - int(np.unique(indices).size)
        rows.append(
            {
                "k_end": k_end,
                "n_final_label_frames": n_final,
                "final_label_fraction": n_final / WINDOW_FRAMES,
                "purity": PURE if n_final == WINDOW_FRAMES else MIXED,
                "n_ignored_frames": int(np.sum(window_labels == IGNORE_LABEL)),
                "segment_start": start,
                "segment_end": end,
                "segment_length": end - start + 1,
                "frames_since_segment_start": k_end - start,
                "frames_to_segment_end": end - k_end,
                "start_distance": distance_bin(k_end - start),
                "end_distance": distance_bin(end - k_end),
                "preceding_label": VIDEO_START if segment == 0 else label_name(int(labels[starts[segment - 1]])),
                "n_replicated_frames": n_replicated,
                "edge_padding": PADDED if n_replicated else UNPADDED,
                "pose_final": POSE_DETECTED if bool(person_found[k_end]) else POSE_MISSING,
            }
        )
    return pd.DataFrame(rows, columns=CONTEXT_COLUMNS)


def person_found_loader(pose_root: Path) -> PersonFoundLoader:
    cache: dict[str, np.ndarray] = {}

    def load(video_id: str) -> np.ndarray:
        if video_id not in cache:
            cache[video_id] = np.asarray(load_person_found(video_id, pose_root=pose_root), dtype=bool)
        return cache[video_id]

    return load


def split_context(frames: pd.DataFrame, split: str, person_found_for: PersonFoundLoader) -> pd.DataFrame:
    split_frames = cast(pd.DataFrame, frames[frames["split"] == split])
    tables = []
    for video_id, group in split_frames.groupby("video_id", sort=True):
        ordered = cast(pd.DataFrame, group.sort_values("frame_index"))
        labels = ordered["label"].to_numpy(dtype=np.int64)
        if not np.array_equal(ordered["frame_index"].to_numpy(), np.arange(len(labels))):
            raise ValueError(f"video_id={video_id!r}: frame_index não é contíguo 0..K-1")
        person_found = person_found_for(str(video_id))
        if person_found.shape != labels.shape:
            raise ValueError(
                f"video_id={video_id!r}: person_found tem forma {person_found.shape}, a grade tem {len(labels)} quadros"
            )
        context = video_context(labels, person_found)
        context.insert(0, "video_id", str(video_id))
        tables.append(context)
    if not tables:
        return pd.DataFrame(columns=["video_id", *CONTEXT_COLUMNS])
    return pd.concat(tables, ignore_index=True)


def join_context(table: pd.DataFrame, context: pd.DataFrame, description: str) -> pd.DataFrame:
    joined = table.merge(context, on=["video_id", "k_end"], how="outer", indicator=True)
    unmatched = cast(pd.DataFrame, joined[joined["_merge"] != "both"])
    if not unmatched.empty:
        sample = [(str(video_id), int(k_end)) for video_id, k_end in zip(unmatched["video_id"], unmatched["k_end"])]
        raise ValueError(
            f"{description}: contexto de janela não alinha com as predições em {len(unmatched)} janela(s): "
            f"{sample[:5]}"
        )
    ordered = cast(pd.DataFrame, joined.drop(columns="_merge").sort_values(["video_id", "k_end"]))
    return ordered.reset_index(drop=True)


def _with_dimensions(windows: pd.DataFrame) -> pd.DataFrame:
    return windows.assign(
        environment=windows["env"].astype(str),
        video=windows["video_id"].astype(str),
        environment_purity=windows["env"].astype(str) + "/" + windows["purity"],
        video_purity=windows["video_id"].astype(str) + "/" + windows["purity"],
    )


def _share(part: int, total: int) -> float | None:
    return part / total if total > 0 else None


def _outcome_counts(group: pd.DataFrame) -> dict[str, int]:
    outcomes = group["outcome"]
    return {
        "n_true": len(group),
        "n_error": int((outcomes == ERROR).sum()),
        "n_correct": int((outcomes == CORRECT).sum()),
        "n_other_outcome": int((outcomes == OTHER_OUTCOME).sum()),
    }


def _strata(subset: pd.DataFrame, dimension: str, totals: dict[str, int]) -> list[dict]:
    rows = []
    for stratum, group in subset.groupby(dimension, sort=True):
        counts = _outcome_counts(group)
        rows.append(
            {
                "stratum": str(stratum),
                **counts,
                "error_rate": _share(counts["n_error"], counts["n_true"]),
                "error_share": _share(counts["n_error"], totals["n_error"]),
                "correct_share": _share(counts["n_correct"], totals["n_correct"]),
            }
        )
    return rows


def _profile(group: pd.DataFrame) -> dict:
    if group.empty:
        return {"n": 0, **{field: None for field in PROFILE_FIELDS}}
    return {
        "n": len(group),
        "mixed_share": float((group["purity"] == MIXED).mean()),
        "mean_final_label_fraction": float(np.mean(group["final_label_fraction"].to_numpy(dtype=np.float64))),
        "edge_padded_share": float((group["edge_padding"] == PADDED).mean()),
        "pose_missing_share": float((group["pose_final"] == POSE_MISSING).mean()),
        "median_frames_since_segment_start": float(
            np.median(group["frames_since_segment_start"].to_numpy(dtype=np.float64))
        ),
        "median_frames_to_segment_end": float(np.median(group["frames_to_segment_end"].to_numpy(dtype=np.float64))),
    }


def check_conservation(
    counts: dict[str, int], strata: dict[str, list[dict]], matrix: np.ndarray, true_id: int, pred_id: int,
    description: str,
) -> dict:
    n_true = int(matrix[true_id].sum())
    n_error = int(matrix[true_id][pred_id])
    n_correct = int(matrix[true_id][true_id])
    expected = {"n_true": n_true, "n_error": n_error, "n_correct": n_correct,
                "n_other_outcome": n_true - n_error - n_correct}
    if counts != expected:
        raise ValueError(f"{description}: contagens do par {counts} divergem de metrics.json {expected}")
    for dimension, rows in strata.items():
        for field, value in expected.items():
            if sum(row[field] for row in rows) != value:
                raise ValueError(f"{description}: soma de {field} por {dimension} diverge de metrics.json")
    return {"ok": True, "dimensions_checked": list(strata), **expected}


def pair_analysis(
    labeled: pd.DataFrame, true_label: str, predicted_label: str, matrix: np.ndarray, description: str
) -> dict:
    true_id = LE2I_LABEL_NAMES.index(true_label)
    pred_id = LE2I_LABEL_NAMES.index(predicted_label)
    subset = cast(pd.DataFrame, labeled[labeled["label"] == true_id])
    predicted = subset["pred_label"].to_numpy()
    subset = subset.assign(
        outcome=np.select([predicted == pred_id, predicted == true_id], [ERROR, CORRECT], OTHER_OUTCOME)
    )
    counts = _outcome_counts(subset)
    strata = {dimension: _strata(subset, dimension, counts) for dimension in DIMENSIONS}
    context = f"{description} {pair_key(true_label, predicted_label)}"
    return {
        "true_label": true_label,
        "predicted_label": predicted_label,
        "counts": counts,
        "profiles": {
            ERROR: _profile(cast(pd.DataFrame, subset[subset["outcome"] == ERROR])),
            CORRECT: _profile(cast(pd.DataFrame, subset[subset["outcome"] == CORRECT])),
        },
        "strata": strata,
        "conservation": check_conservation(counts, strata, matrix, true_id, pred_id, context),
    }


def _frames_path(dataset: str) -> Path:
    return get_dataset(dataset).frames_path


def _pose_root(dataset: str) -> Path:
    return get_dataset(dataset).pose_root


def pose_input_files(dataset: str) -> list[Path]:
    frames_path = _frames_path(dataset)
    if not frames_path.is_file():
        return []
    frames = pd.read_parquet(frames_path, columns=["video_id", "split"])
    video_ids = sorted({str(video_id) for video_id in frames.loc[frames["split"].isin(SPLITS), "video_id"]})
    return [pose_path(video_id, pose_root=_pose_root(dataset)) for video_id in video_ids]


def analyze_run(item: DiagnosisInput, person_found_for: PersonFoundLoader) -> tuple[dict, dict[str, pd.DataFrame]]:
    run = validate_run(item)
    seed = run.header["seed"]
    if not isinstance(seed, int):
        raise ValueError(f"{run.context}: config.yaml sem seed inteira")
    frames, evaluate_split = load_evaluation(item.arm, item.dataset, item.path).prepare()
    splits: dict[str, dict] = {}
    labeled_tables: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        description = f"{run.context} [{split}]"
        table, analysis = analyze_split_with_table(frames, evaluate_split, split, run.final[split], description)
        windows = join_context(table, split_context(frames, split, person_found_for), description)
        labeled = _with_dimensions(cast(pd.DataFrame, windows[windows["label"] != IGNORE_LABEL]))
        matrix = np.asarray(run.final[split]["confusion_matrix"], dtype=np.int64)
        overall = analysis["overall"]
        splits[split] = {
            "n_windows": overall["n_windows"],
            "n_labeled_windows": overall["n_labeled_windows"],
            "n_ignored_windows": overall["n_ignored_windows"],
            "reconciliation": analysis["reconciliation"],
            "pairs": {
                pair_key(true_label, predicted_label): pair_analysis(
                    labeled, true_label, predicted_label, matrix, description
                )
                for true_label, predicted_label, _role in FOCUS_PAIRS
            },
        }
        labeled_tables[split] = labeled.reset_index(drop=True)
    return {**run.header, "splits": splits}, labeled_tables


def _prediction_column(arm: str, seed: int) -> str:
    return f"pred_{arm}_seed{seed}"


def window_table(labeled: pd.DataFrame, split: str) -> pd.DataFrame:
    table = labeled.assign(split=split, true_label=[label_name(int(label)) for label in labeled["label"]])
    return cast(pd.DataFrame, table[WINDOW_COLUMNS]).reset_index(drop=True)


def _strata_signature(entry: dict) -> dict:
    return {
        (split, key, dimension): [(row["stratum"], row["n_true"]) for row in rows]
        for split, block in entry["splits"].items()
        for key, pair in block["pairs"].items()
        for dimension, rows in pair["strata"].items()
    }


def _aggregate_strata(blocks: list[list[dict]]) -> list[dict]:
    aggregated = []
    for index, first in enumerate(blocks[0]):
        rows = [block[index] for block in blocks]
        aggregated.append(
            {
                "stratum": first["stratum"],
                "n_true": first["n_true"],
                **{field: _aggregate_stats([row[field] for row in rows])
                   for field in ("n_error", "n_correct", "n_other_outcome")},
                **{field: _aggregate_stats([row[field] for row in rows if row[field] is not None])
                   for field in SHARE_FIELDS},
            }
        )
    return aggregated


def aggregate_arm(entries: list[dict]) -> dict:
    splits: dict[str, dict] = {}
    for split in SPLITS:
        pairs: dict[str, dict] = {}
        for true_label, predicted_label, role in FOCUS_PAIRS:
            key = pair_key(true_label, predicted_label)
            blocks = [entry["splits"][split]["pairs"][key] for entry in entries]
            pairs[key] = {
                "role": role,
                "counts": {field: _aggregate_stats([block["counts"][field] for block in blocks])
                           for field in COUNT_FIELDS},
                "profiles": {
                    outcome: {
                        field: _aggregate_stats(
                            [block["profiles"][outcome][field] for block in blocks
                             if block["profiles"][outcome][field] is not None]
                        )
                        for field in PROFILE_FIELDS
                    }
                    for outcome in (ERROR, CORRECT)
                },
                "strata": {dimension: _aggregate_strata([block["strata"][dimension] for block in blocks])
                           for dimension in DIMENSIONS},
            }
        splits[split] = pairs
    return {"seeds": [entry["seed"] for entry in entries], "n_seeds": len(entries), "splits": splits}


def _stratum(rows: list[dict], name: str) -> dict | None:
    return next((row for row in rows if row["stratum"] == name), None)


def _require_stratum(rows: list[dict], name: str) -> dict:
    row = _stratum(rows, name)
    if row is None:
        raise ValueError(f"estrato ausente: {name}")
    return row


def cross_run_consistency(entries: list[dict]) -> dict:
    consistency: dict[str, dict] = {}
    for split in SPLITS:
        consistency[split] = {}
        for true_label, predicted_label, _role in FOCUS_PAIRS:
            key = pair_key(true_label, predicted_label)
            more_mixed = less_mixed = undefined_mixed = 0
            mixed_rate_higher = mixed_rate_lower = undefined_rate = 0
            for entry in entries:
                pair = entry["splits"][split]["pairs"][key]
                error_mixed = pair["profiles"][ERROR]["mixed_share"]
                correct_mixed = pair["profiles"][CORRECT]["mixed_share"]
                if error_mixed is None or correct_mixed is None or error_mixed == correct_mixed:
                    undefined_mixed += 1
                elif error_mixed > correct_mixed:
                    more_mixed += 1
                else:
                    less_mixed += 1
                mixed = _stratum(pair["strata"]["purity"], MIXED)
                pure = _stratum(pair["strata"]["purity"], PURE)
                if mixed is None or pure is None or mixed["error_rate"] == pure["error_rate"]:
                    undefined_rate += 1
                elif mixed["error_rate"] > pure["error_rate"]:
                    mixed_rate_higher += 1
                else:
                    mixed_rate_lower += 1
            consistency[split][key] = {
                "n_runs": len(entries),
                "error_mixed_share_above_correct": more_mixed,
                "error_mixed_share_below_correct": less_mixed,
                "mixed_share_tied_or_undefined": undefined_mixed,
                "mixed_error_rate_above_pure": mixed_rate_higher,
                "mixed_error_rate_below_pure": mixed_rate_lower,
                "error_rate_tied_or_undefined": undefined_rate,
            }
    return consistency


def _pct(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.0%}"


def _mean(stats: dict) -> float | None:
    return stats["mean"]


def _mean_pct(stats: dict) -> str:
    if stats["mean"] is None:
        return "n/d"
    spread = "" if stats["std"] is None else f" ± {stats['std']:.0%}"
    return f"{stats['mean']:.0%}{spread}"


def _arm_observation(split: str, key: str, arm: str, block: dict) -> str:
    counts = block["counts"]
    error = block["profiles"][ERROR]
    correct = block["profiles"][CORRECT]
    purity = {row["stratum"]: row for row in block["strata"]["purity"]}
    rates = "; ".join(
        f"taxa de erro {_pct(_mean(purity[name]['error_rate']))} em janelas "
        f"{'puras' if name == PURE else 'misturadas'} ({purity[name]['n_true']} janelas)"
        for name in (PURE, MIXED)
        if name in purity
    )
    return (
        f"{split}, {_arrow(key)}, {arm}: {_mean(counts['n_error']):.1f} erros em média de "
        f"{_mean(counts['n_true']):.0f} janelas verdadeiras; {_pct(_mean(error['mixed_share']))} dos erros e "
        f"{_pct(_mean(correct['mixed_share']))} dos acertos estão em janelas misturadas; {rates}; "
        f"{_pct(_mean(error['edge_padded_share']))} dos erros com replicação de borda e "
        f"{_pct(_mean(error['pose_missing_share']))} sem pose no quadro final."
    )


def build_observations(arms: dict, consistency: dict) -> list[dict]:
    observations = []
    for split in SPLITS:
        for true_label, predicted_label, role in FOCUS_PAIRS:
            key = pair_key(true_label, predicted_label)
            for arm, arm_block in arms.items():
                block = arm_block["splits"][split][key]
                if _mean(block["counts"]["n_true"]) is None:
                    continue
                observations.append({"split": split, "pair": key, "role": role, "arm": arm,
                                     "text": _arm_observation(split, key, arm, block)})
            summary = consistency.get(split, {}).get(key)
            if summary and summary["n_runs"]:
                observations.append({
                    "split": split, "pair": key, "role": role, "arm": None,
                    "text": (
                        f"{split}, {_arrow(key)}: a fração de erros em janelas misturadas supera a de acertos em "
                        f"{summary['error_mixed_share_above_correct']} de {summary['n_runs']} runs e fica abaixo "
                        f"em {summary['error_mixed_share_below_correct']}; a taxa de erro em janelas misturadas "
                        f"supera a de janelas puras em {summary['mixed_error_rate_above_pure']} runs e fica "
                        f"abaixo em {summary['mixed_error_rate_below_pure']}."
                    ),
                })
    return observations


def _limitations() -> list[str]:
    return [
        "Análise exploratória e não canônica (Etapa 3A): nada substitui runs/reference nem seleciona run, seed, "
        "checkpoint, limiar ou hiperparâmetro. O split test já foi exposto e é só descritivo; val e test são "
        "relatados separadamente e nunca somados.",
        "Associação não é causa: fronteiras de trecho coincidem com transições de ação, posição no vídeo, "
        "replicação de borda e decisões de anotação. Uma concentração de erros em janelas misturadas não "
        "autoriza mudar o janelamento congelado.",
        f"Janelas com eval_stride {EVAL_STRIDE} se sobrepõem: janelas vizinhas não são observações "
        "independentes, e contagens não equivalem a trechos, vídeos ou sujeitos.",
        "A unidade agregada é a seed de treino: médias e desvios são calculados sobre valores por seed e "
        "descrevem inicialização e otimização, não incerteza entre sujeitos. Janelas nunca são somadas "
        "entre seeds nem entre braços.",
        "Janela pura significa que os 24 índices de window_frame_indices, inclusive os replicados pela "
        "borda, têm o rótulo final; quadros IGNORE_LABEL dentro da janela a tornam misturada. Em janelas "
        "rotuladas, misturada equivale a cruzar a fronteira inicial do trecho, porque o trecho anterior "
        "tem por definição outro rótulo.",
        "O rótulo do trecho anterior descreve a anotação, não o conteúdo dos quadros; o trecho anterior "
        "pode estar fora da janela quando o trecho atual é longo.",
        "A detecção de pose vem de person_found do HDF5 bruto no quadro final; não mede qualidade da pose, "
        "pose imputada nem a entrada visual de B0, B1, C0 e C1.",
        "As predições vêm da reexecução validada sobre o checkpoint congelado. A reconciliação exata com "
        "metrics.json é necessária, mas não prova que cada janela recebeu historicamente o mesmo rótulo.",
        "standing→walk e sitting→stand_up são controles descritivos com suporte pequeno e concentrado em "
        "poucos vídeos; não servem de teste da hipótese sobre other→walk.",
    ]


def _coverage_gaps(entries_by_arm: dict[str, list[dict]], reported: set[str]) -> list[dict]:
    gaps = []
    for arm in ARMS:
        seeds = {entry["seed"] for entry in entries_by_arm[arm]}
        if not seeds and arm not in reported:
            gaps.append({"arm": arm, "seed": None, "source": None, "reason": "nenhum run fornecido",
                         "missing_files": []})
            continue
        gaps.extend(
            {"arm": arm, "seed": seed, "source": None, "reason": "seed esperada não analisada", "missing_files": []}
            for seed in EXPECTED_SEEDS
            if seed not in seeds
        )
    return gaps


def build_report(runs: list[DiagnosisInput]) -> tuple[dict, pd.DataFrame]:
    entries_by_arm: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    missing_inputs: list[dict] = []
    window_tables: dict[str, pd.DataFrame] = {}
    predictions: dict[str, dict[str, list[str]]] = {split: {} for split in SPLITS}
    pose_files = pose_input_files(SUPPORTED_DATASET)
    missing_pose = [str(path) for path in pose_files if not path.is_file()]
    person_found_for = person_found_loader(_pose_root(SUPPORTED_DATASET))
    for item in runs:
        missing = missing_prerequisites(item) + missing_pose
        if missing:
            missing_inputs.append({"arm": item.arm, "seed": None, "source": display_path(item.path),
                                   "reason": "pré-requisitos ausentes", "missing_files": missing})
            continue
        try:
            entry, labeled_tables = analyze_run(item, person_found_for)
        except FileNotFoundError as exc:
            missing_inputs.append({"arm": item.arm, "seed": None, "source": display_path(item.path),
                                   "reason": f"arquivo ausente durante a inferência: {exc}", "missing_files": []})
            continue
        column = _prediction_column(entry["arm"], entry["seed"])
        if any(column in predictions[split] for split in SPLITS):
            raise ValueError(f"braço {entry['arm']} com seed {entry['seed']} fornecido mais de uma vez")
        for split, labeled in labeled_tables.items():
            windows = window_table(labeled, split)
            if split not in window_tables:
                window_tables[split] = windows
            elif not window_tables[split].equals(windows):
                raise ValueError(f"{entry['source']} [{split}]: contexto de janela difere do de outro run")
            predictions[split][column] = [label_name(int(label)) for label in labeled["pred_label"]]
        entries_by_arm[item.arm].append(entry)
    for entries in entries_by_arm.values():
        entries.sort(key=lambda entry: entry["seed"])

    reported = {item["arm"] for item in missing_inputs}
    missing_inputs.extend(_coverage_gaps(entries_by_arm, reported))
    entries = [entry for arm in ARMS for entry in entries_by_arm[arm]]
    if entries:
        signature = _strata_signature(entries[0])
        for entry in entries[1:]:
            if _strata_signature(entry) != signature:
                raise ValueError(f"{entry['source']}: estratos ou suporte por estrato diferem entre runs")
    arms = {arm: aggregate_arm(arm_entries) for arm, arm_entries in entries_by_arm.items() if arm_entries}
    consistency = cross_run_consistency(entries)
    report = {
        "analysis": "window_context_diagnosis",
        "stage": "3A",
        "canonical": False,
        "read_only": True,
        "dataset": SUPPORTED_DATASET,
        "protocol": PROTOCOL_NAMES[SUPPORTED_DATASET],
        "label_names": list(LE2I_LABEL_NAMES),
        "ignore_label": IGNORE_LABEL,
        "eval_stride": EVAL_STRIDE,
        "window_frames": WINDOW_FRAMES,
        "target_fps": TARGET_FPS,
        "splits": list(SPLITS),
        "focus_pairs": [{"pair": pair_key(true_label, predicted_label), "true_label": true_label,
                         "predicted_label": predicted_label, "role": role}
                        for true_label, predicted_label, role in FOCUS_PAIRS],
        "dimensions": list(DIMENSIONS),
        "distance_bins_frames": [distance_bin(low) for low, _high in DISTANCE_BINS],
        "runs": entries,
        "arms": arms,
        "cross_run_consistency": consistency,
        "missing_inputs": missing_inputs,
        "observations": build_observations(arms, consistency),
        "limitations": _limitations(),
    }
    frames = []
    for split, windows in window_tables.items():
        frames.append(windows.assign(**predictions[split]))
    windows_csv = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=WINDOW_COLUMNS)
    return report, windows_csv


def _run_rows(entry: dict) -> list[dict]:
    rows: list[dict] = []
    base = {"record": "run", "arm": entry["arm"], "seed": entry["seed"], "source": entry["source"]}
    for split, block in entry["splits"].items():
        for key, pair in block["pairs"].items():
            location = {**base, "split": split, "pair": key}
            for outcome, profile in pair["profiles"].items():
                rows.extend({**location, "dimension": "profile", "stratum": outcome, "metric": field,
                             "value": profile[field]} for field in ("n", *PROFILE_FIELDS))
            for dimension, strata in pair["strata"].items():
                rows.extend({**location, "dimension": dimension, "stratum": row["stratum"], "metric": field,
                             "value": row[field]} for row in strata for field in (*COUNT_FIELDS, *SHARE_FIELDS))
    return rows


def _stats_rows(location: dict, dimension: str, stratum: str, field: str, stats: dict) -> list[dict]:
    return [{**location, "dimension": dimension, "stratum": stratum, "metric": f"{field}_{name}", "value": value}
            for name, value in stats.items()]


def _arm_rows(arm: str, block: dict) -> list[dict]:
    rows: list[dict] = []
    base = {"record": "arm_seed_aggregate", "arm": arm, "seed": None, "source": None}
    for split, pairs in block["splits"].items():
        for key, pair in pairs.items():
            location = {**base, "split": split, "pair": key}
            for field, stats in pair["counts"].items():
                rows.extend(_stats_rows(location, "counts", "all", field, stats))
            for outcome, profile in pair["profiles"].items():
                for field, stats in profile.items():
                    rows.extend(_stats_rows(location, "profile", outcome, field, stats))
            for dimension, strata in pair["strata"].items():
                for row in strata:
                    rows.append({**location, "dimension": dimension, "stratum": row["stratum"], "metric": "n_true",
                                 "value": row["n_true"]})
                    for field in ("n_error", "n_correct", "n_other_outcome", *SHARE_FIELDS):
                        rows.extend(_stats_rows(location, dimension, row["stratum"], field, row[field]))
    return rows


def csv_rows(report: dict) -> list[dict]:
    rows: list[dict] = []
    for entry in report["runs"]:
        rows.extend(_run_rows(entry))
    for arm, block in report["arms"].items():
        rows.extend(_arm_rows(arm, block))
    return rows


def _profile_table(arms: dict, split: str, key: str) -> list[str]:
    lines = [
        "",
        "| Braço | Seeds | Erros (média) | Erros em janelas misturadas | Acertos em janelas misturadas | "
        "Fração do rótulo final, erros / acertos | Erros com replicação de borda | Erros sem pose no quadro final | "
        "Quadros desde o início do trecho (mediana), erros / acertos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, block in arms.items():
        pair = block["splits"][split][key]
        error, correct = pair["profiles"][ERROR], pair["profiles"][CORRECT]
        n_error = _mean(pair["counts"]["n_error"])
        n_true = _mean(pair["counts"]["n_true"])

        def fraction(stats: dict) -> str:
            return "n/d" if stats["mean"] is None else f"{stats['mean']:.2f}"

        def frames(stats: dict) -> str:
            return "n/d" if stats["mean"] is None else f"{stats['mean']:.1f}"

        lines.append(
            f"| {arm} | {block['n_seeds']} | {n_error:.1f}/{n_true:.0f} | {_mean_pct(error['mixed_share'])} | "
            f"{_mean_pct(correct['mixed_share'])} | {fraction(error['mean_final_label_fraction'])} / "
            f"{fraction(correct['mean_final_label_fraction'])} | {_mean_pct(error['edge_padded_share'])} | "
            f"{_mean_pct(error['pose_missing_share'])} | {frames(error['median_frames_since_segment_start'])} / "
            f"{frames(correct['median_frames_since_segment_start'])} |"
        )
    return lines


def _strata_table(arms: dict, split: str, key: str, dimension: str, title: str, limit: int | None = None) -> list[str]:
    first = next(iter(arms.values()))["splits"][split][key]["strata"][dimension]
    if not first:
        return []
    names = [row["stratum"] for row in first]
    if limit is not None:
        totals = {
            name: sum(_mean(_require_stratum(block["splits"][split][key]["strata"][dimension], name)["n_error"])
                      or 0.0 for block in arms.values())
            for name in names
        }
        names = sorted(names, key=lambda name: (-totals[name], name))[:limit]
    lines = [
        "",
        f"{title} — taxa de erro média entre seeds (erros médios / janelas verdadeiras):",
        "",
        "| Estrato | " + " | ".join(arms) + " |",
        "| --- |" + " ---: |" * len(arms),
    ]
    for name in names:
        cells = []
        for block in arms.values():
            row = _require_stratum(block["splits"][split][key]["strata"][dimension], name)
            cells.append(f"{_pct(_mean(row['error_rate']))} ({_mean(row['n_error']):.1f}/{row['n_true']})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def render_summary(report: dict) -> str:
    lines = [
        f"# Contexto de janela dos erros {_arrow(pair_key('other', 'walk'))} (Etapa 3A, não canônico)",
        "",
        "Diagnóstico exploratório e somente leitura de Le2i-CS sobre predições reexecutadas a partir de "
        "checkpoints congelados, sem treino, ajuste ou seleção. Cada contagem por estrato soma a célula "
        "correspondente de confusion_matrix em metrics.json. Médias e desvios são entre seeds; janelas nunca "
        "são somadas entre seeds nem entre braços. val e test aparecem em seções separadas. Os controles "
        f"{_arrow(pair_key('standing', 'walk'))} e {_arrow(pair_key('sitting', 'stand_up'))} são só descritivos.",
    ]
    arms = report["arms"]
    if not arms:
        lines += ["", "Nenhum run disponível."]
    for split in report["splits"]:
        if not arms:
            break
        lines += ["", f"## {split}"]
        for item in report["focus_pairs"]:
            key = item["pair"]
            role = "par principal" if item["role"] == PRIMARY else "controle descritivo"
            lines += ["", f"### {_arrow(key)} ({role})"]
            lines += _profile_table(arms, split, key)
            lines += _strata_table(arms, split, key, "purity", "Pureza da janela")
            lines += _strata_table(arms, split, key, "start_distance", "Quadros desde o início do trecho")
            if item["role"] == PRIMARY:
                lines += _strata_table(arms, split, key, "end_distance", "Quadros até o fim do trecho")
                lines += _strata_table(arms, split, key, "preceding_label", "Rótulo do trecho anterior")
                lines += _strata_table(arms, split, key, "edge_padding", "Replicação de borda")
                lines += _strata_table(arms, split, key, "pose_final", "Pose no quadro final")
                lines += _strata_table(arms, split, key, "environment_purity", "Ambiente × pureza")
                lines += _strata_table(arms, split, key, "video", "Vídeos com mais erros", limit=8)
            lines += ["", "Observações:", ""]
            lines += [f"- {obs['text']}" for obs in report["observations"]
                      if obs["split"] == split and obs["pair"] == key] or ["- Nenhuma."]
    lines += ["", "## Entradas ausentes", ""]
    lines += [
        f"- {item['arm']}"
        + (f" seed {item['seed']}" if item["seed"] is not None else "")
        + (f" (`{item['source']}`)" if item["source"] else "")
        + f": {item['reason']}"
        + (f" — {', '.join(item['missing_files'])}" if item["missing_files"] else "")
        for item in report["missing_inputs"]
    ] or ["- Nenhuma."]
    lines += ["", "## Limitações", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def validate_window_context_runs(runs: list[DiagnosisInput]) -> None:
    validate_runs(runs)
    unsupported = sorted({item.dataset for item in runs if item.dataset != SUPPORTED_DATASET})
    if unsupported:
        raise ValueError(f"a Etapa 3A cobre só Le2i-CS ({SUPPORTED_DATASET!r}); recebido: {unsupported}")


def run_window_context_diagnosis(runs: list[DiagnosisInput], output_dir: Path, force: bool = False) -> bool:
    validate_window_context_runs(runs)
    check_output_dir(output_dir, protected_paths(runs))
    run_files = {path.resolve() for item in runs if not missing_prerequisites(item) for path in run_input_files(item)}
    pose_files = {path.resolve() for path in pose_input_files(SUPPORTED_DATASET) if path.is_file()}
    snapshot = snapshot_inputs(sorted(run_files | pose_files))
    report, windows = build_report(runs)
    report["input_sha256"] = {display_path(Path(path)): digest for path, digest in snapshot.items()}
    targets = {
        output_dir / WINDOW_CONTEXT_JSON_FILE: json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        output_dir / WINDOW_CONTEXT_CSV_FILE: pd.DataFrame(csv_rows(report), columns=CSV_COLUMNS).to_csv(index=False),
        output_dir / WINDOW_CONTEXT_WINDOWS_FILE: windows.to_csv(index=False),
        output_dir / WINDOW_CONTEXT_SUMMARY_FILE: render_summary(report),
    }
    verify_inputs_unchanged(snapshot)
    return write_text_outputs(output_dir, targets, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Atribui o contexto de janela dos erros other→walk em Le2i-CS"
    )
    diagnose_parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("DATASET", "ARM", "RUN_DIR"),
        required=True,
        help="Repetível: run local de Le2i-CS com config.yaml, metrics.json e checkpoint.pt",
    )
    diagnose_parser.add_argument("--output-dir", type=Path, required=True)
    diagnose_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do diagnóstico de contexto de janela")

    args = parser.parse_args()
    if args.command == "diagnose":
        run_window_context_diagnosis(_parse_inputs(args.run), args.output_dir, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.window_context_diagnosis import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
