"""Preparação, somente leitura, do conjunto de revisão visual de `standing→walk` e `sitting→stand_up` (Etapa 6B).

Lê as saídas por janela da [Etapa 6A] (`standing_sitting_windows.csv` e
`standing_sitting_diagnosis.json`) e o `frames.parquet` filtrado em
`train`/`val` e seleciona, de forma determinística, janelas de erro dos dois
pares e controles corretamente classificados pareados por evidência, classe
verdadeira, categoria de janela e sujeito, com preferência pelo mesmo vídeo.
O `val` oficial (25 runs) e os grupos retidos das dobras por sujeito (braço A,
seed 42) são selecionados, contabilizados e numerados separadamente. Não
refaz inferência, não avalia modelo e não altera anotações, janelamento,
features, partições, checkpoints nem a evidência da Etapa 6A.

Os quadros de cada janela selecionada são decodificados por
`gatefall.data.video_io.decode_frames` com o mapeamento `frame_index →
src_index` de `frames.parquet` e montados nas folhas de contato da Etapa 3B.
Só vídeos de `train`/`val` são resolvidos, hasheados ou decodificados. A
planilha e as folhas não mostram predições nem o papel (erro ou controle) de
cada item; a chave de desmascaramento fica em um diretório separado, com
permissão restrita ao dono. Nenhum conteúdo de vídeo é rotulado nem
interpretado automaticamente.
"""

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, TARGET_FPS, WINDOW_FRAMES
from gatefall.data.video_io import decode_frames
from gatefall.data.windowing import build_window_index
from gatefall.datasets import get_dataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import (
    EXPECTED_SEEDS,
    PROTOCOL_NAMES,
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.multiseed_summary import ARMS
from gatefall.eval.analysis.other_walk_review_set import (
    FRAME_COLUMNS,
    FRAMES_CSV_FILE,
    INSTRUCTIONS_FILE,
    KEY_CSV_FILE,
    KEY_DIR,
    PERSISTENCE_MIN_ARM_SHARE,
    SELECTION_JSON_FILE,
    SELECTION_SUMMARY_FILE,
    SHEETS_DIR,
    WORKSHEET_CSV_FILE,
    WORKSHEET_DIR,
    DecodeFrames,
    Record,
    ReviewSources,
    _csv_header,
    _existing_targets,
    _write_binary_outputs,
    check_review_output_dir,
    records,
    render_sheets,
    review_order_key,
    sheet_relative_path,
    source_files,
    video_files,
    verify_video_hashes,
    video_frame_table,
    window_mapping,
)
from gatefall.eval.analysis.other_walk_train_composition import VIDEO_END
from gatefall.eval.analysis.standing_sitting_diagnosis import ANALYSIS as SOURCE_ANALYSIS
from gatefall.eval.analysis.standing_sitting_diagnosis import (
    CATEGORIES,
    FOCUS_PAIRS,
    FOLD_ARM,
    FOLD_SEED,
    OFFICIAL_TRAIN,
    OFFICIAL_VAL,
    REPORT_JSON_FILE,
    REPORT_WINDOWS_FILE,
    SUBJECT_FOLD,
    WINDOW_CSV_COLUMNS,
    pair_key,
    window_category,
)
from gatefall.eval.analysis.standing_sitting_diagnosis import STAGE as SOURCE_STAGE
from gatefall.eval.analysis.window_context_diagnosis import MIXED, PURE, label_name, video_context
from gatefall.train.subject_folds.folds import RETAINED_SUBJECT_GROUPS, fold_name

ANALYSIS = "standing_sitting_review_set"
STAGE = "6B"
SUPPORTED_DATASET = "le2i"
DEVELOPMENT_SPLITS: tuple[str, ...] = ("train", "val")
EVIDENCES: tuple[str, ...] = (OFFICIAL_VAL, SUBJECT_FOLD)
EVIDENCE_SPLIT = {OFFICIAL_VAL: "val", SUBJECT_FOLD: "train"}
EVIDENCE_PREFIX = {OFFICIAL_VAL: "val", SUBJECT_FOLD: "fold"}
FOLD_SUBJECTS: dict[str, tuple[int, ...]] = {
    fold_name(group): tuple(sorted(group)) for group in RETAINED_SUBJECT_GROUPS
}
TRUE_LABELS: tuple[str, ...] = tuple(true for true, _predicted in FOCUS_PAIRS)
PREDICTED_FOR: dict[str, str] = dict(FOCUS_PAIRS)

ERROR = "error"
CONTROL = "correct_control"
ROLES: tuple[str, ...] = (ERROR, CONTROL)
# Erros selecionados por estrato (evidência, dobra, sujeito, classe verdadeira, categoria).
MAX_ERRORS_PER_STRATUM = 2
VIDEO_MATCH = "video"
SUBJECT_MATCH = "subject"
UNMATCHED = "unmatched"
MATCH_LEVELS: tuple[str, ...] = (VIDEO_MATCH, SUBJECT_MATCH)
NO_CANDIDATE = "no_candidate"
CANDIDATES_UNAVAILABLE = "overlap_or_segment_used"

KEY_DIR_MODE = 0o700
KEY_FILE_MODE = 0o600

MODEL_KEY: tuple[str, ...] = ("evidence", "fold", "arm", "seed")
WINDOW_ID: tuple[str, ...] = ("evidence", "fold", "video_id", "k_end")
SEGMENT_ID: tuple[str, ...] = ("evidence", "fold", "video_id", "segment_start", "segment_end")
STRATUM: tuple[str, ...] = ("evidence", "fold", "subject", "true_label", "category")
INT_COLUMNS: tuple[str, ...] = ("subject", "k_end", "seed", "frames_since_segment_start", "frames_to_segment_end")
ATTRIBUTES: tuple[str, ...] = (
    "env", "subject", "true_label", "purity", "frames_since_segment_start", "frames_to_segment_end",
    "preceding_label", "following_label", "edge_padding", "detected_fraction", "mean_keypoint_conf", "category",
)
CONTEXT_CHECKS: tuple[str, ...] = (
    "purity", "frames_since_segment_start", "frames_to_segment_end", "preceding_label", "edge_padding",
)
VOTE_FIELDS: tuple[str, ...] = ("error_votes", "correct_votes", "n_seeds")

REVIEW_FIELDS: dict[str, str] = {
    "reviewer": "texto livre",
    "visible_action": "texto livre: ação visível ao longo dos 24 quadros, sobretudo nos últimos",
    "apparent_motion": "none | in_place | stepping | posture_change | unclear",
    "confusable_reading": "no | plausible | unclear",
    "label_ambiguity": "none | boundary | label | unclear",
    "person_visibility": "full | partial | occluded | absent | unclear",
    "review_confidence": "high | medium | low | insufficient",
    "reviewer_notes": "texto livre",
}
WORKSHEET_COLUMNS = [
    "review_id",
    "split",
    "video_id",
    "env",
    "subject",
    "true_label",
    "k_end",
    "window_frame_start",
    "window_frame_end",
    "src_index_start",
    "src_index_end",
    "time_start_s",
    "time_end_s",
    "n_replicated_frames",
    "sheet",
    *REVIEW_FIELDS,
]
KEY_CONTEXT_FIELDS: tuple[str, ...] = (
    "segment_start", "segment_end", "purity", "frames_since_segment_start", "frames_to_segment_end",
    "preceding_label", "following_label", "edge_padding",
)


@dataclass(frozen=True)
class ReviewPlan:
    table: pd.DataFrame
    reviews: list[tuple[str, Record, pd.DataFrame]]
    exclusions: list[dict]
    matches: list[dict]
    source: dict
    missing_evidence: list[dict]
    development_videos: dict[str, str]
    frames_splits_read: list[str]

    @property
    def video_ids(self) -> list[str]:
        return sorted({str(item["video_id"]) for _review_id, item, _mapping in self.reviews})


def diagnosis_json_path(sources: ReviewSources) -> Path:
    return sources.windows_csv.with_name(REPORT_JSON_FILE)


def _sample(table: pd.DataFrame, columns: Sequence[str] = ("evidence", "fold", "video_id", "k_end")) -> list[tuple]:
    present = [column for column in columns if column in table.columns]
    return [tuple(row) for row in table[present].head(5).to_numpy().tolist()]


def validate_source_report(report: dict, path: Path) -> None:
    expected = {"analysis": SOURCE_ANALYSIS, "stage": SOURCE_STAGE, "dataset": SUPPORTED_DATASET,
                "protocol": PROTOCOL_NAMES[SUPPORTED_DATASET], "canonical": False}
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"{path}: {field}={report.get(field)!r}; esperado {value!r} da Etapa 6A")


def load_stage6a(sources: ReviewSources) -> tuple[pd.DataFrame, dict]:
    csv_path, json_path = sources.windows_csv, diagnosis_json_path(sources)
    for path in (csv_path, json_path):
        if not path.is_file():
            raise FileNotFoundError(f"saída da Etapa 6A não encontrada: {path}")
    header = _csv_header(csv_path)
    if header != WINDOW_CSV_COLUMNS:
        raise ValueError(f"{csv_path}: colunas {header}; esperadas {WINDOW_CSV_COLUMNS}")
    windows = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    validate_source_report(report, json_path)
    return windows, report


def _fail_rows(rows: pd.DataFrame, mask: object, path: Path, description: str) -> None:
    failing = cast(pd.DataFrame, rows[mask])
    if not failing.empty:
        raise ValueError(f"{path}: {len(failing)} linha(s) com {description}: {_sample(failing)}")


def parse_model_rows(windows: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Linhas por janela e modelo do `val` oficial e das dobras, com tipos e domínios conferidos."""
    known = {OFFICIAL_TRAIN, *EVIDENCES}
    _fail_rows(windows, ~windows["evidence"].isin(sorted(known)), path, "evidência desconhecida")
    train = cast(pd.DataFrame, windows[windows["evidence"] == OFFICIAL_TRAIN])
    _fail_rows(train, train["pred_label"] != "", path, "predição em official_train")
    rows = cast(pd.DataFrame, windows[windows["evidence"].isin(list(EVIDENCES))]).copy()
    for column in INT_COLUMNS:
        _fail_rows(rows, ~rows[column].str.fullmatch(r"0|[1-9]\d*"), path, f"{column} não inteiro")
        rows[column] = rows[column].astype(np.int64)
    is_val = rows["evidence"] == OFFICIAL_VAL
    is_fold = rows["evidence"] == SUBJECT_FOLD
    checks = {
        "classe verdadeira fora de standing/sitting": ~rows["true_label"].isin(list(TRUE_LABELS)),
        "predição fora dos rótulos do Le2i": ~rows["pred_label"].isin(list(LE2I_LABEL_NAMES)),
        "categoria desconhecida": ~rows["category"].isin(list(CATEGORIES)),
        "pureza desconhecida": ~rows["purity"].isin([PURE, MIXED]),
        "detected_fraction não numérica": cast(pd.Series, pd.to_numeric(rows["detected_fraction"],
                                                                         errors="coerce")).isna(),
        "val oficial com dobra, braço ou seed inesperados": is_val & (
            (rows["fold"] != "") | ~rows["arm"].isin(list(ARMS)) | ~rows["seed"].isin(list(EXPECTED_SEEDS))
        ),
        "dobra com nome, braço ou seed inesperados": is_fold & (
            ~rows["fold"].isin(list(FOLD_SUBJECTS)) | (rows["arm"] != FOLD_ARM) | (rows["seed"] != FOLD_SEED)
        ),
        "(modelo, video_id, k_end) repetido": rows.duplicated([*MODEL_KEY, "video_id", "k_end"], keep=False),
    }
    for description, mask in checks.items():
        _fail_rows(rows, mask, path, description)
    missing = [(arm, seed) for arm in ARMS for seed in EXPECTED_SEEDS
               if not ((rows["evidence"] == OFFICIAL_VAL) & (rows["arm"] == arm) & (rows["seed"] == seed)).any()]
    if missing:
        raise ValueError(f"{path}: val oficial sem os modelos (braço, seed) {missing}")
    distinct = rows.drop_duplicates([*WINDOW_ID, *ATTRIBUTES])
    _fail_rows(distinct, distinct.duplicated(list(WINDOW_ID), keep=False), path,
               "atributos de janela divergentes entre modelos")
    models = rows.drop_duplicates(list(MODEL_KEY))
    n_models = cast(pd.Series, models.groupby(["evidence", "fold"]).size()).reset_index(name="expected")
    per_window = cast(pd.Series, rows.groupby(list(WINDOW_ID)).size()).reset_index(name="observed")
    per_window = per_window.merge(n_models, on=["evidence", "fold"])
    _fail_rows(per_window, per_window["observed"] != per_window["expected"], path,
               "janela ausente em algum modelo da mesma evidência")
    return cast(pd.DataFrame, rows.sort_values([*MODEL_KEY, "video_id", "k_end"])).reset_index(drop=True)


def _pair_counts(group: pd.DataFrame, true_label: str) -> dict[str, int]:
    subset = cast(pd.DataFrame, group[group["true_label"] == true_label])
    n_error = int((subset["pred_label"] == PREDICTED_FOR[true_label]).sum())
    n_correct = int((subset["pred_label"] == true_label).sum())
    return {"n_true": len(subset), "n_error": n_error, "n_correct": n_correct,
            "n_other_outcome": len(subset) - n_error - n_correct}


def recorded_models(report: dict) -> dict[tuple[str, str, str, int], dict]:
    models = {(OFFICIAL_VAL, "", str(run["arm"]), int(run["seed"])): run for run in report["official_val"]["runs"]}
    models.update({(SUBJECT_FOLD, str(block["fold"]), str(block["arm"]), int(block["seed"])): block
                   for block in report["subject_folds"]["folds"]})
    return models


def reconcile_counts(rows: pd.DataFrame, report: dict, path: Path) -> None:
    """Contagens por modelo e par no CSV precisam reproduzir as do JSON da Etapa 6A."""
    recorded = recorded_models(report)
    observed = {(str(evidence), str(fold), str(arm), int(seed))
                for evidence, fold, arm, seed in rows[list(MODEL_KEY)].drop_duplicates().to_numpy().tolist()}
    if observed != set(recorded):
        raise ValueError(f"{path}: modelos do CSV e do JSON divergem: só no CSV {sorted(observed - set(recorded))}, "
                         f"só no JSON {sorted(set(recorded) - observed)}")
    for key, group in rows.groupby(list(MODEL_KEY), sort=True):
        evidence, fold, arm, seed = cast(tuple, key)
        model = recorded[(str(evidence), str(fold), str(arm), int(seed))]
        for true_label in TRUE_LABELS:
            pair = pair_key(true_label, PREDICTED_FOR[true_label])
            counts = _pair_counts(cast(pd.DataFrame, group), true_label)
            expected = {field: int(model["pairs"][pair]["counts"][field]) for field in counts}
            if counts != expected:
                raise ValueError(f"{path}: {evidence} {fold} {arm} seed {seed} {pair}: contagens do CSV {counts} "
                                 f"divergem do JSON {expected}")


def source_record(report: dict, sources: ReviewSources) -> dict:
    models = []
    for (evidence, fold, arm, seed), model in sorted(recorded_models(report).items()):
        models.append({
            "evidence": evidence, "fold": fold, "arm": arm, "seed": seed,
            "source": model.get("source", model.get("run_dir")),
            "checkpoint_sha256": model.get("checkpoint_sha256"),
            **({"retained_subjects": model.get("retained_subjects")} if evidence == SUBJECT_FOLD else {}),
        })
    return {
        "analysis": SOURCE_ANALYSIS,
        "stage": SOURCE_STAGE,
        "windows_csv": display_path(sources.windows_csv),
        "report_json": display_path(diagnosis_json_path(sources)),
        "verdict": report.get("decision", {}).get("verdict"),
        "models": models,
        "missing_evidence": report.get("missing_evidence", []),
    }


def load_development_frames(frames_path: Path) -> pd.DataFrame:
    """Lê só linhas de `train`/`val`; linhas de `test` não são materializadas."""
    if not frames_path.is_file():
        raise FileNotFoundError(f"frames.parquet não encontrado: {frames_path}")
    frames = pd.read_parquet(frames_path, filters=[("split", "in", list(DEVELOPMENT_SPLITS))])
    unexpected = sorted(set(frames["split"].astype(str)) - set(DEVELOPMENT_SPLITS))
    if unexpected:
        raise ValueError(f"{frames_path}: leitura filtrada devolveu splits fora de train/val: {unexpected}")
    return frames


def window_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por janela e evidência, com votos por braço e o papel de erro ou controle."""
    rows = rows.assign(predicted_label=[PREDICTED_FOR[str(label)] for label in rows["true_label"]])
    rows = rows.assign(is_error=rows["pred_label"] == rows["predicted_label"],
                       is_correct=rows["pred_label"] == rows["true_label"])
    votes = rows.groupby([*WINDOW_ID, "arm"], sort=True).agg(
        error_votes=("is_error", "sum"), correct_votes=("is_correct", "sum"), n_seeds=("seed", "size"),
    ).reset_index()
    votes = votes.assign(error_share=votes["error_votes"] / votes["n_seeds"],
                         correct_share=votes["correct_votes"] / votes["n_seeds"])
    per_window = votes.groupby(list(WINDOW_ID), sort=True).agg(
        n_models=("n_seeds", "sum"), error_votes=("error_votes", "sum"), correct_votes=("correct_votes", "sum"),
        min_model_error_share=("error_share", "min"), min_model_correct_share=("correct_share", "min"),
    ).reset_index()
    per_arm = votes.pivot(index=list(WINDOW_ID), columns="arm", values=list(VOTE_FIELDS))
    per_arm.columns = [f"{field}_{arm}" for field, arm in per_arm.columns]
    attributes = rows.drop_duplicates(list(WINDOW_ID))[[*WINDOW_ID, *ATTRIBUTES, "predicted_label"]]
    table = attributes.merge(per_window, on=list(WINDOW_ID)).merge(per_arm.reset_index(), on=list(WINDOW_ID))
    table = table.assign(
        error_share=table["error_votes"] / table["n_models"],
        correct_share=table["correct_votes"] / table["n_models"],
        segment_start=table["k_end"] - table["frames_since_segment_start"],
        segment_end=table["k_end"] + table["frames_to_segment_end"],
        order_key=[hashlib.sha256(f"{evidence}/{fold}/{video_id}/{k_end}".encode()).hexdigest()
                   for evidence, fold, video_id, k_end in table[list(WINDOW_ID)].to_numpy().tolist()],
    )
    table["role"] = np.select(
        [table["min_model_error_share"] >= PERSISTENCE_MIN_ARM_SHARE,
         table["min_model_correct_share"] >= PERSISTENCE_MIN_ARM_SHARE],
        [ERROR, CONTROL],
        "",
    )
    return cast(pd.DataFrame, table.sort_values(list(WINDOW_ID))).reset_index(drop=True)


def expected_windows(frames: pd.DataFrame, evidence: str, fold: str) -> set[tuple[str, int]]:
    side = cast(pd.DataFrame, frames[frames["split"] == EVIDENCE_SPLIT[evidence]])
    if evidence == SUBJECT_FOLD:
        side = cast(pd.DataFrame, side[side["subject"].isin(list(FOLD_SUBJECTS[fold]))])
    if side.empty:
        return set()
    index = build_window_index(side, EVAL_STRIDE, drop_ignored=True)
    true_ids = [LE2I_LABEL_NAMES.index(label) for label in TRUE_LABELS]
    focus = cast(pd.DataFrame, index[index["label"].isin(true_ids)])
    return set(zip(focus["video_id"].astype(str), focus["k_end"].astype(int)))


def reconcile_windows(table: pd.DataFrame, frames: pd.DataFrame, path: Path) -> None:
    outside = sorted(set(table["video_id"].astype(str)) - set(frames["video_id"].astype(str)))
    if outside:
        raise ValueError(f"{path}: vídeo(s) fora de train/val em frames.parquet: {outside[:5]}")
    for key, group in table.groupby(["evidence", "fold"], sort=True):
        evidence, fold = cast(tuple, key)
        observed = set(zip(group["video_id"].astype(str), group["k_end"].astype(int)))
        expected = expected_windows(frames, str(evidence), str(fold))
        if observed != expected:
            raise ValueError(f"{path}: janelas de {evidence} {fold} divergem de frames.parquet: "
                             f"só no CSV {sorted(observed - expected)[:5]}, ausentes do CSV "
                             f"{sorted(expected - observed)[:5]}")


def _following_label(labels: np.ndarray, segment_end: int) -> str:
    return VIDEO_END if segment_end == len(labels) - 1 else label_name(int(labels[segment_end + 1]))


def validate_context(table: pd.DataFrame, frames: pd.DataFrame, path: Path) -> None:
    """Contexto de trecho, rótulo final e categoria de cada janela recalculados a partir de `frames.parquet`."""
    for key, group in table.groupby(["evidence", "fold", "video_id"], sort=True):
        evidence, fold, video_id = cast(tuple, key)
        video = video_frame_table(frames, str(video_id))
        first = records(video.iloc[:1])[0]
        where = f"{path}: {evidence} {fold} {video_id}"
        if str(first["split"]) != EVIDENCE_SPLIT[str(evidence)]:
            raise ValueError(f"{where}: split {first['split']!r} em frames.parquet não corresponde à evidência")
        labels = video["label"].to_numpy(dtype=np.int64)
        context = video_context(labels, np.ones(len(labels), dtype=bool)).set_index("k_end")
        for row in records(group):
            k_end = int(row["k_end"])
            at = f"{where} k_end={k_end}"
            if str(row["env"]) != str(first["env"]) or int(row["subject"]) != int(first["subject"]):
                raise ValueError(f"{at}: ambiente ou sujeito divergem de frames.parquet")
            if int(labels[k_end]) != LE2I_LABEL_NAMES.index(str(row["true_label"])):
                raise ValueError(f"{at}: rótulo do quadro final em frames.parquet diverge de {row['true_label']}")
            expected = context.loc[k_end]
            diverging = [field for field in CONTEXT_CHECKS if str(expected[field]) != str(row[field])]
            if _following_label(labels, int(row["segment_end"])) != str(row["following_label"]):
                diverging.append("following_label")
            if diverging:
                raise ValueError(f"{at}: {diverging} divergem de frames.parquet")
            category = window_category(str(row["purity"]), str(row["following_label"]),
                                       int(row["frames_to_segment_end"]), float(row["detected_fraction"]))
            if category != str(row["category"]):
                raise ValueError(f"{at}: categoria {row['category']!r} não segue a regra da Etapa 6A ({category!r})")


def _share(row: Record, role: str) -> float:
    return float(row["min_model_error_share" if role == ERROR else "min_model_correct_share"])


def candidate_rank(row: Record, role: str) -> tuple[float, str]:
    return -_share(row, role), str(row["order_key"])


def _overlapping(row: Record, selected: list[Record]) -> bool:
    return any(
        item["evidence"] == row["evidence"]
        and item["video_id"] == row["video_id"]
        and abs(int(item["k_end"]) - int(row["k_end"])) < WINDOW_FRAMES
        for item in selected
    )


def _window_id(row: Record) -> tuple:
    return tuple(row[field] for field in WINDOW_ID)


def _segment_id(row: Record) -> tuple:
    return tuple(row[field] for field in SEGMENT_ID)


def _segment_stratum(row: Record) -> tuple:
    return (*_segment_id(row), row["category"])


def stratum_order(key: Sequence) -> tuple:
    evidence, fold, subject, true_label, category = key
    return (EVIDENCES.index(str(evidence)), str(fold), int(subject), TRUE_LABELS.index(str(true_label)),
            CATEGORIES.index(str(category)))


def _stratum(row: Record) -> dict:
    return {field: row[field] for field in STRATUM}


def _segment_entry(row: Record, role: str, reason: str) -> dict:
    return {**{field: row[field] for field in SEGMENT_ID}, **_stratum(row), "role": role, "reason": reason}


def select_errors(table: pd.DataFrame) -> tuple[list[Record], list[dict]]:
    selected: list[Record] = []
    exclusions: list[dict] = []
    pool = cast(pd.DataFrame, table[table["role"] == ERROR])
    strata = sorted(((key, group) for key, group in pool.groupby(list(STRATUM), sort=False)),
                    key=lambda item: stratum_order(cast(tuple, item[0])))
    for _key, group in strata:
        segments = [sorted(records(segment), key=lambda row: candidate_rank(row, ERROR))
                    for _segment, segment in group.groupby(list(SEGMENT_ID), sort=True)]
        segments.sort(key=lambda rows: candidate_rank(rows[0], ERROR))
        n_selected = 0
        for rows in segments:
            if n_selected >= MAX_ERRORS_PER_STRATUM:
                exclusions.append(_segment_entry(rows[0], ERROR, "cap"))
                continue
            choice = next((row for row in rows if not _overlapping(row, selected)), None)
            if choice is None:
                exclusions.append(_segment_entry(rows[0], ERROR, "overlap"))
                continue
            selected.append(choice)
            n_selected += 1
    return selected, exclusions


def match_level(error: Record, candidate: Record) -> str:
    return VIDEO_MATCH if candidate["video_id"] == error["video_id"] else SUBJECT_MATCH


def select_controls(table: pd.DataFrame, errors: list[Record]) -> tuple[list[Record], list[dict]]:
    """Um controle por erro e por trecho e estrato, da mesma evidência, dobra, sujeito, classe e categoria."""
    pool = records(cast(pd.DataFrame, table[table["role"] == CONTROL]))
    selected = list(errors)
    controls: list[Record] = []
    matches: list[dict] = []
    used_segments: set[tuple] = set()
    for error in errors:
        candidates = [row for row in pool if all(row[field] == error[field] for field in STRATUM)]
        available = [row for row in candidates
                     if _segment_stratum(row) not in used_segments and not _overlapping(row, selected)]
        match = {"error": _window_id(error), "control": None, "level": UNMATCHED, "same_segment": False,
                 "reason": None, "n_candidates": len(candidates)}
        if not available:
            match["reason"] = CANDIDATES_UNAVAILABLE if candidates else NO_CANDIDATE
            matches.append(match)
            continue
        choice = min(available, key=lambda row: (MATCH_LEVELS.index(match_level(error, row)),
                                                 *candidate_rank(row, CONTROL)))
        selected.append(choice)
        controls.append(choice)
        used_segments.add(_segment_stratum(choice))
        match.update(control=_window_id(choice), level=match_level(error, choice),
                     same_segment=_segment_id(choice) == _segment_id(error))
        matches.append(match)
    return controls, matches


def assign_review_ids(items: list[Record]) -> list[tuple[str, Record]]:
    assigned: list[tuple[str, Record]] = []
    for evidence in EVIDENCES:
        prefix = EVIDENCE_PREFIX[evidence]
        ordered = sorted((item for item in items if item["evidence"] == evidence),
                         key=lambda item: review_order_key(prefix, str(item["video_id"]), int(item["k_end"])))
        assigned.extend((f"{prefix}-{index:02d}", item) for index, item in enumerate(ordered, start=1))
    return assigned


def validate_mapping(item: Record, mapping: pd.DataFrame) -> None:
    where = f"{item['evidence']} {item['video_id']} k_end={item['k_end']}"
    labels = mapping["label"].astype(str).tolist()
    if labels[-1] != item["true_label"]:
        raise ValueError(f"{where}: quadro final do mapeamento não tem o rótulo {item['true_label']}")
    if (labels.count(str(item["true_label"])) == WINDOW_FRAMES) != (item["purity"] == PURE):
        raise ValueError(f"{where}: pureza do mapeamento diverge do CSV")
    if bool(mapping["replicated"].any()) != (item["edge_padding"] != "unpadded"):
        raise ValueError(f"{where}: replicação de borda do mapeamento diverge do CSV")


def missing_folds(table: pd.DataFrame) -> list[dict]:
    present = set(table.loc[table["evidence"] == SUBJECT_FOLD, "fold"].astype(str))
    return [{"evidence": SUBJECT_FOLD, "source": fold,
             "reason": "dobra sem janelas na saída da Etapa 6A; nada selecionado nela"}
            for fold in FOLD_SUBJECTS if fold not in present]


def plan_review_set(sources: ReviewSources) -> ReviewPlan:
    windows, report = load_stage6a(sources)
    rows = parse_model_rows(windows, sources.windows_csv)
    reconcile_counts(rows, report, sources.windows_csv)
    frames = load_development_frames(sources.frames_path)
    table = window_table(rows)
    reconcile_windows(table, frames, sources.windows_csv)
    validate_context(table, frames, sources.windows_csv)
    errors, exclusions = select_errors(table)
    controls, matches = select_controls(table, errors)
    error_ids = {_window_id(error) for error in errors}
    reviews: list[tuple[str, Record, pd.DataFrame]] = []
    for review_id, item in assign_review_ids([*errors, *controls]):
        item = {**item, "role": ERROR if _window_id(item) in error_ids else CONTROL}
        mapping = window_mapping(video_frame_table(frames, str(item["video_id"])), int(item["k_end"]))
        validate_mapping(item, mapping)
        reviews.append((review_id, item, mapping))
    development_videos = {str(video_id): str(split)
                          for video_id, split in sorted(set(zip(frames["video_id"], frames["split"])))}
    return ReviewPlan(
        table=table, reviews=reviews, exclusions=exclusions, matches=matches,
        source=source_record(report, sources), missing_evidence=missing_folds(table),
        development_videos=development_videos, frames_splits_read=sorted(set(frames["split"].astype(str))),
    )


def sheet_header(review_id: str, item: Record, mapping: pd.DataFrame) -> tuple[str, ...]:
    return (
        f"{review_id} | {item['video_id']} | {item['env']} | true_label: {item['true_label']}",
        f"janela k={int(mapping['frame_index'].iloc[0])}..{int(item['k_end'])} | "
        f"src={int(mapping['src_index'].iloc[0])}..{int(mapping['src_index'].iloc[-1])} | "
        f"{float(mapping['time_s'].iloc[0]):.1f}s..{float(mapping['time_s'].iloc[-1]):.1f}s | "
        f"{WINDOW_FRAMES} quadros a {TARGET_FPS} fps",
    )


def guarded_decoder(decode_fn: DecodeFrames, allowed: dict[Path, str], decoded: dict[str, int]) -> DecodeFrames:
    def guarded(path: Path, src_indices: Sequence[int]) -> list[np.ndarray]:
        split = allowed.get(path.resolve())
        if split is None:
            raise RuntimeError(f"decodificação recusada fora de train/val: {path}")
        decoded[split] = decoded.get(split, 0) + 1
        return decode_fn(path, src_indices)

    return guarded


def _is_blank(value: object) -> bool:
    return value is None or value == "" or (isinstance(value, float) and np.isnan(value))


def _optional_float(value: object) -> float | None:
    return None if _is_blank(value) else float(cast(float, value))


def _optional_int(value: object) -> int | None:
    return None if _is_blank(value) else int(cast(float, value))


def key_row(review_id: str, item: Record, mapping: pd.DataFrame, match: dict, review_ids: dict[tuple, str],
            sheet: str | None) -> dict:
    partner = match["control"] if item["role"] == ERROR else match["error"]
    row = {
        "review_id": review_id,
        "evidence": item["evidence"],
        "split": EVIDENCE_SPLIT[str(item["evidence"])],
        "fold": item["fold"],
        "role": item["role"],
        "pair": pair_key(str(item["true_label"]), str(item["predicted_label"])),
        "video_id": item["video_id"],
        "env": item["env"],
        "subject": int(item["subject"]),
        "true_label": item["true_label"],
        "predicted_label": item["predicted_label"],
        "category": item["category"],
        "k_end": int(item["k_end"]),
        "window_frame_start": int(mapping["frame_index"].iloc[0]),
        "window_frame_end": int(mapping["frame_index"].iloc[-1]),
        "src_index_start": int(mapping["src_index"].iloc[0]),
        "src_index_end": int(mapping["src_index"].iloc[-1]),
        "time_start_s": float(mapping["time_s"].iloc[0]),
        "time_end_s": float(mapping["time_s"].iloc[-1]),
        "n_replicated_frames": int(np.sum(mapping["replicated"].to_numpy())),
        **{field: item[field] for field in KEY_CONTEXT_FIELDS},
        "detected_fraction": float(item["detected_fraction"]),
        "mean_keypoint_conf": _optional_float(item["mean_keypoint_conf"]),
        "n_models": int(item["n_models"]),
        "error_votes": int(item["error_votes"]),
        "correct_votes": int(item["correct_votes"]),
        "error_share": float(item["error_share"]),
        "correct_share": float(item["correct_share"]),
        "min_model_error_share": float(item["min_model_error_share"]),
        "min_model_correct_share": float(item["min_model_correct_share"]),
        **{f"{field}_{arm}": _optional_int(item.get(f"{field}_{arm}")) for arm in ARMS for field in VOTE_FIELDS},
        "matched_review_id": None if partner is None else review_ids[partner],
        "match_level": match["level"],
        "match_same_segment": match["same_segment"],
        "unmatched_reason": match["reason"],
        "sheet": sheet,
    }
    return row


def worksheet_row(key: dict) -> dict:
    return {
        **{column: key[column] for column in WORKSHEET_COLUMNS if column in key},
        "sheet": key["sheet"] or "",
        **{field: "" for field in REVIEW_FIELDS},
    }


def _segments(table: pd.DataFrame) -> int:
    return int(table.groupby(list(SEGMENT_ID)).ngroups) if len(table) else 0


def coverage(table: pd.DataFrame, keys: list[dict], exclusions: list[dict]) -> dict[str, list[dict]]:
    """Por evidência e estrato, sem somar janelas entre seeds, dobras ou evidências."""
    report: dict[str, list[dict]] = {}
    for evidence in EVIDENCES:
        entries = []
        in_evidence = cast(pd.DataFrame, table[table["evidence"] == evidence])
        strata = sorted(((key, group) for key, group in in_evidence.groupby(list(STRATUM), sort=False)),
                        key=lambda item: stratum_order(cast(tuple, item[0])))
        for key, group in strata:
            stratum = dict(zip(STRATUM, cast(tuple, key)))
            chosen = [row for row in keys if all(str(row[field]) == str(stratum[field]) for field in STRATUM)]
            skipped = [row for row in exclusions if all(str(row[field]) == str(stratum[field]) for field in STRATUM)]
            errors = cast(pd.DataFrame, group[group["role"] == ERROR])
            controls = cast(pd.DataFrame, group[group["role"] == CONTROL])
            selected_errors = [row for row in chosen if row["role"] == ERROR]
            entries.append({
                **{field: stratum[field] for field in STRATUM[1:]},
                "n_windows": len(group),
                "n_segments": _segments(group),
                ERROR: {"n_windows": len(errors), "n_segments": _segments(errors), "n_selected": len(selected_errors),
                        "n_excluded_cap": sum(row["reason"] == "cap" for row in skipped),
                        "n_excluded_overlap": sum(row["reason"] == "overlap" for row in skipped)},
                CONTROL: {"n_windows": len(controls), "n_segments": _segments(controls),
                          "n_selected": sum(row["role"] == CONTROL for row in chosen)},
                "match_levels": {level: sum(row["match_level"] == level for row in selected_errors)
                                 for level in (*MATCH_LEVELS, UNMATCHED)},
            })
        report[evidence] = entries
    return report


def criteria() -> dict:
    return {
        "dataset": SUPPORTED_DATASET,
        "pairs": [pair_key(true, predicted) for true, predicted in FOCUS_PAIRS],
        "evidence": {
            OFFICIAL_VAL: "val oficial, 25 runs (A, B0, B1, C0, C1 × seeds 42–46) da Etapa 6A",
            SUBJECT_FOLD: f"grupos retidos das dobras por sujeito, braço {FOLD_ARM}, seed {FOLD_SEED} (um modelo)",
        },
        "persistence_min_arm_share": PERSISTENCE_MIN_ARM_SHARE,
        "role_rule": "erro: em todo braço disponível, pelo menos persistence_min_arm_share das seeds preveem a classe "
                     "confundível do par; controle: o mesmo com a classe verdadeira. Nas dobras há um único modelo",
        "stratum": list(STRATUM),
        "max_errors_per_stratum": MAX_ERRORS_PER_STRATUM,
        "error_rank": ["maior fração mínima entre braços do papel",
                       "sha256 de '<evidence>/<fold>/<video_id>/<k_end>'"],
        "one_item_per_segment_stratum_and_role": True,
        "control_match": {
            "required": list(STRATUM),
            "preference": list(MATCH_LEVELS),
            "one_control_per_segment_and_stratum": True,
            "unmatched": "erro mantido e registrado sem controle; nenhum requisito é relaxado",
        },
        "min_k_end_gap_same_video": WINDOW_FRAMES,
        "selection_order": "erros de todos os estratos antes dos controles; estratos por evidência, dobra, sujeito, "
                           "classe e categoria",
        "review_id_order": "sha256 de '<prefixo>/<video_id>/<k_end>' dentro de cada evidência; independe do papel",
        "window_frames": WINDOW_FRAMES,
        "eval_stride": EVAL_STRIDE,
        "target_fps": TARGET_FPS,
    }


def _limitations() -> list[str]:
    return [
        "Preparação exploratória e não canônica (Etapa 6B): não refaz inferência, não avalia modelo e não altera "
        "anotações, features, partições, checkpoints, treino nem a evidência da Etapa 6A.",
        "As predições são as reexecuções validadas da Etapa 6A e herdam suas limitações. O val oficial tem um único "
        "sujeito; as dobras têm um único modelo (braço A, seed 42), então a persistência entre seeds e braços só "
        "existe no val. As duas evidências não são somadas nem comparadas em magnitude.",
        "Só train e val são lidos em frames.parquet, e só vídeos desses splits são resolvidos, hasheados ou "
        "decodificados. O manifesto é lido inteiro como metadado e logo filtrado.",
        "Um item por trecho, estrato e papel, e janelas do mesmo vídeo a pelo menos WINDOW_FRAMES quadros umas das "
        "outras. Itens do mesmo vídeo ou sujeito ainda não são observações independentes; o conjunto descreve "
        "casos e não estima taxas.",
        "O limite por estrato e a ordem por hash tornam a seleção reprodutível, não representativa.",
        "O mascaramento é parcial: ambiente, vídeo, sujeito e rótulos anotados aparecem na planilha e nas folhas, "
        "e quem conhece a Etapa 6A pode reconhecer trechos ou inferir a categoria de janela.",
        "Nada aqui descreve o conteúdo dos vídeos; achados visuais só existem depois do preenchimento humano da "
        "planilha e não são interpretados por esta etapa.",
    ]


def render_summary(report: dict) -> str:
    lines = [
        "# Conjunto de revisão de standing→walk e sitting→stand_up (Etapa 6B, não canônico)",
        "",
        "Chave de desmascaramento: não abrir antes de concluir a revisão visual cega.",
        "",
        f"Persistência no val: fração mínima entre braços ≥ {PERSISTENCE_MIN_ARM_SHARE:.0%} das seeds. Dobras: um "
        f"modelo. No máximo {MAX_ERRORS_PER_STRATUM} erros por estrato, cada um com um controle da mesma "
        "evidência, sujeito, classe e categoria. Contagens são de janelas distintas por evidência; não há totais "
        "entre seeds, dobras ou evidências.",
    ]
    for evidence, entries in report["coverage"].items():
        lines += [
            "",
            f"## {evidence}",
            "",
            "| Dobra | Sujeito | Classe | Categoria | Janelas | Erro: janelas / trechos | "
            "Controle: janelas / trechos | Selecionados erro / controle | Pareamento vídeo / sujeito / sem par | "
            "Excluídos (limite / sobreposição) |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        if not entries:
            lines.append("| — | — | — | — | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 / 0 | 0 / 0 |")
        for entry in entries:
            error, control, levels = entry[ERROR], entry[CONTROL], entry["match_levels"]
            lines.append(
                f"| {entry['fold'] or '—'} | {entry['subject']} | {entry['true_label']} | {entry['category']} | "
                f"{entry['n_windows']} | {error['n_windows']} / {error['n_segments']} | {control['n_windows']} / "
                f"{control['n_segments']} | {error['n_selected']} / {control['n_selected']} | "
                f"{levels[VIDEO_MATCH]} / {levels[SUBJECT_MATCH]} / {levels[UNMATCHED]} | "
                f"{error['n_excluded_cap']} / {error['n_excluded_overlap']} |"
            )
    lines += ["", "## Erros sem controle pareado", ""]
    lines += [f"- {item['review_id']}: {item['unmatched_reason']}" for item in report["items"]
              if item["role"] == ERROR and item["match_level"] == UNMATCHED] or ["- Nenhum."]
    lines += ["", "## Entradas e evidências ausentes", ""]
    lines += [f"- {item['video_id']}: {item['reason']} ({', '.join(item['review_ids'])})"
              for item in report["missing_inputs"]]
    lines += [f"- {item['evidence']} {item['source']}: {item['reason']}" for item in report["missing_evidence"]]
    if not report["missing_inputs"] and not report["missing_evidence"]:
        lines.append("- Nenhuma.")
    lines += ["", "## Limitações", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def render_instructions() -> str:
    lines = [
        "# Revisão visual cega de janelas standing e sitting (Etapa 6B)",
        "",
        "1. Abra cada folha em `sheets/` na ordem dos `review_id`. Cada folha mostra os 24 quadros da janela, da "
        "esquerda para a direita e de cima para baixo, com o índice da grade (`k`), o quadro de origem (`src`), o "
        "tempo e o rótulo anotado; `(borda)` marca quadros replicados no início do vídeo. O rótulo verdadeiro é o "
        "do último quadro.",
        "2. Julgue só pelo que as imagens mostram. Não consulte predições, a chave, outras folhas do mesmo vídeo "
        "nem documentos das Etapas 6A e 6B durante a revisão.",
        "3. Preencha só as colunas abaixo de `review_worksheet.csv`. Prefira `unclear` ou `insufficient` a um "
        "palpite e deixe em branco o que não puder ser julgado.",
        "4. Não abra o diretório `key/` antes de terminar todas as linhas.",
        "",
        "| Campo | Valores |",
        "| --- | --- |",
    ]
    lines += [f"| `{field}` | {values} |" for field, values in REVIEW_FIELDS.items()]
    lines += [
        "",
        "- `apparent_motion`: `none`, pessoa parada; `in_place`, gestos ou ajustes sem deslocamento; `stepping`, "
        "passos ou deslocamento; `posture_change`, sentar, levantar ou abaixar.",
        "- `confusable_reading`: se a ação nos últimos quadros poderia ser lida como a classe vizinha do rótulo "
        "(`walk` para `standing`; `stand_up` para `sitting`).",
        "- `label_ambiguity`: `boundary`, quando a ação muda perto de uma fronteira anotada; `label`, quando o "
        "rótulo do último quadro parece não descrever a ação visível.",
        "- `person_visibility`: visibilidade da pessoa anotada nos últimos quadros.",
        "- `review_confidence`: `insufficient` quando os 24 quadros não permitem julgar a ação.",
    ]
    return "\n".join(lines) + "\n"


def _match_by_window(matches: list[dict]) -> dict[tuple, dict]:
    by_window: dict[tuple, dict] = {}
    for match in matches:
        by_window[match["error"]] = match
        if match["control"] is not None:
            by_window[match["control"]] = match
    return by_window


def build_review_set(
    plan: ReviewPlan, sources: ReviewSources, decode_fn: DecodeFrames
) -> tuple[dict, dict[Path, str], dict[str, bytes]]:
    development_paths = {video_id: path for video_id, path in sources.video_paths.items()
                         if video_id in plan.development_videos}
    allowed = {path.resolve(): plan.development_videos[video_id] for video_id, path in development_paths.items()}
    decoded: dict[str, int] = {}
    sheets, missing = render_sheets(plan.reviews, development_paths, guarded_decoder(decode_fn, allowed, decoded),
                                    header=sheet_header)
    review_ids = {_window_id(item): review_id for review_id, item, _mapping in plan.reviews}
    by_window = _match_by_window(plan.matches)
    keys = []
    frame_rows = []
    for review_id, item, mapping in plan.reviews:
        sheet = sheet_relative_path(review_id) if review_id in sheets else None
        keys.append(key_row(review_id, item, mapping, by_window[_window_id(item)], review_ids, sheet))
        frame_rows.append(mapping.assign(review_id=review_id)[FRAME_COLUMNS])
    keys.sort(key=lambda key: (EVIDENCES.index(key["evidence"]), key["review_id"]))
    frames_csv = pd.concat(frame_rows, ignore_index=True) if frame_rows else pd.DataFrame(columns=FRAME_COLUMNS)
    frames_csv = cast(pd.DataFrame, frames_csv.sort_values(["review_id", "position"])).reset_index(drop=True)
    matches = [{**match, "error": review_ids[match["error"]],
                "control": None if match["control"] is None else review_ids[match["control"]]}
               for match in plan.matches]

    report = {
        "analysis": ANALYSIS,
        "stage": STAGE,
        "canonical": False,
        "read_only": True,
        "dataset": SUPPORTED_DATASET,
        "protocol": PROTOCOL_NAMES[SUPPORTED_DATASET],
        "source": plan.source,
        "split_access": {
            "frames_parquet_rows_read": plan.frames_splits_read,
            "selection": [EVIDENCE_SPLIT[evidence] for evidence in EVIDENCES],
            "videos_decoded_by_split": dict(sorted(decoded.items())),
            "note": "frames.parquet é lido com filtro de split; só vídeos de train/val são resolvidos, hasheados ou "
                    "decodificados; test não é lido nem selecionado.",
        },
        "criteria": criteria(),
        "review_fields": REVIEW_FIELDS,
        "layout": {
            "blinded": [f"{WORKSHEET_DIR}/{WORKSHEET_CSV_FILE}", f"{WORKSHEET_DIR}/{FRAMES_CSV_FILE}",
                        f"{WORKSHEET_DIR}/{SHEETS_DIR}/", f"{WORKSHEET_DIR}/{INSTRUCTIONS_FILE}"],
            "unblinding_key": [f"{KEY_DIR}/{SELECTION_JSON_FILE}", f"{KEY_DIR}/{KEY_CSV_FILE}",
                               f"{KEY_DIR}/{SELECTION_SUMMARY_FILE}"],
        },
        "coverage": coverage(plan.table, keys, plan.exclusions),
        "items": keys,
        "matches": matches,
        "exclusions": plan.exclusions,
        "missing_inputs": missing,
        "missing_evidence": plan.missing_evidence,
        "limitations": _limitations(),
    }
    worksheet = pd.DataFrame([worksheet_row(key) for key in keys], columns=WORKSHEET_COLUMNS)
    texts = {
        Path(WORKSHEET_DIR) / WORKSHEET_CSV_FILE: worksheet.to_csv(index=False),
        Path(WORKSHEET_DIR) / FRAMES_CSV_FILE: frames_csv.to_csv(index=False),
        Path(WORKSHEET_DIR) / INSTRUCTIONS_FILE: render_instructions(),
        Path(KEY_DIR) / KEY_CSV_FILE: pd.DataFrame(keys).to_csv(index=False),
    }
    return report, texts, sheets


def protect_key(key_dir: Path) -> None:
    os.chmod(key_dir, KEY_DIR_MODE)
    for path in key_dir.iterdir():
        if path.is_file():
            os.chmod(path, KEY_FILE_MODE)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_review_set(
    sources: ReviewSources, output_dir: Path, force: bool = False, decode_fn: DecodeFrames = decode_frames
) -> bool:
    check_review_output_dir(output_dir, sources)
    snapshot = snapshot_inputs(source_files(sources))
    plan = plan_review_set(sources)
    snapshot.update(snapshot_inputs(video_files(sources, plan.video_ids)))
    verify_video_hashes(snapshot, sources, plan.video_ids)
    report, texts, sheets = build_review_set(plan, sources, decode_fn)
    images = {Path(WORKSHEET_DIR) / sheet_relative_path(review_id): data for review_id, data in sheets.items()}
    report["worksheet_sha256"] = {
        **{str(name): _sha256(text.encode("utf-8")) for name, text in texts.items() if name.parts[0] == WORKSHEET_DIR},
        **{str(name): _sha256(data) for name, data in images.items()},
    }
    report["worksheet_sha256"] = dict(sorted(report["worksheet_sha256"].items()))
    report["input_sha256"] = {display_path(Path(path)): digest for path, digest in sorted(snapshot.items())}
    texts[Path(KEY_DIR) / SELECTION_JSON_FILE] = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    texts[Path(KEY_DIR) / SELECTION_SUMMARY_FILE] = render_summary(report)
    existing = _existing_targets(output_dir, [*texts, *images])
    if existing and not force:
        print(f"skip {', '.join(existing)} (já existe, use --force para sobrescrever)")
        return False
    verify_inputs_unchanged(snapshot)
    for directory in (KEY_DIR, WORKSHEET_DIR):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    _write_binary_outputs({output_dir / name: data for name, data in images.items()})
    for directory in (KEY_DIR, WORKSHEET_DIR):
        in_directory = {output_dir / name: text for name, text in texts.items() if name.parts[0] == directory}
        write_text_outputs(output_dir / directory, in_directory, force=True)
    protect_key(output_dir / KEY_DIR)
    return True


def le2i_sources(diagnosis_dir: Path) -> ReviewSources:
    adapter = get_dataset(SUPPORTED_DATASET)
    development = pd.read_parquet(adapter.frames_path, columns=["video_id", "split"],
                                  filters=[("split", "in", list(DEVELOPMENT_SPLITS))])
    development_ids = set(development["video_id"].astype(str))
    manifest = adapter.load_manifest()
    manifest = cast(pd.DataFrame, manifest[manifest["video_id"].astype(str).isin(sorted(development_ids))])
    return ReviewSources(
        windows_csv=diagnosis_dir / REPORT_WINDOWS_FILE,
        frames_path=adapter.frames_path,
        video_paths={str(video_id): adapter.resolve_video_path(str(relative_path))
                     for video_id, relative_path in zip(manifest["video_id"], manifest["relative_path"])},
        video_sha256={str(video_id): str(digest) for video_id, digest in zip(manifest["video_id"], manifest["sha256"])},
        extra_inputs=(diagnosis_dir / REPORT_JSON_FILE, adapter.manifest_path),
        protected=(adapter.raw_dir,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Seleciona e renderiza o conjunto de revisão cega de standing→walk e sitting→stand_up"
    )
    prepare_parser.add_argument("--diagnosis-dir", type=Path, required=True,
                                help="diretório de saída da Etapa 6A (standing_sitting_windows.csv e .json)")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do conjunto de revisão standing/sitting")

    args = parser.parse_args()
    if args.command == "prepare":
        run_review_set(le2i_sources(args.diagnosis_dir), args.output_dir, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.standing_sitting_review_set import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
