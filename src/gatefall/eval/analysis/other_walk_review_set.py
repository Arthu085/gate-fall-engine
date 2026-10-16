"""Preparação, somente leitura, do conjunto de revisão qualitativa dos erros `other→walk` (Etapa 3B).

Lê `window_context_windows.csv` da [Etapa 3A] e `frames.parquet` e seleciona,
de forma determinística, trechos `other` de Le2i-CS em que o erro
`other→walk` persiste entre braços e seeds, mais controles `other`
corretamente classificados dos mesmos ambientes. Não refaz inferência, não
avalia modelo e não altera anotações, janelamento, features ou checkpoints.

Os quadros de cada janela selecionada são decodificados por
`gatefall.data.video_io.decode_frames` com o mapeamento `frame_index →
src_index` de `frames.parquet` e montados em folhas de contato locais. A
planilha de revisão e as folhas não mostram predições nem o papel (erro ou
controle) de cada item; a chave de desmascaramento fica em um diretório
separado. Nenhum conteúdo de vídeo é rotulado automaticamente.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from PIL import Image

from gatefall.config import EVAL_STRIDE, TARGET_FPS, WINDOW_FRAMES
from gatefall.data.video_io import decode_frames
from gatefall.data.windowing import build_window_index, window_frame_indices
from gatefall.datasets import get_dataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import (
    EXPECTED_SEEDS,
    check_output_dir,
    display_path,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_text_outputs,
)
from gatefall.eval.analysis.multiseed_summary import ARMS
from gatefall.eval.analysis.qualitative import _with_caption_strip
from gatefall.eval.analysis.stratified_error_diagnosis import SPLITS
from gatefall.eval.analysis.window_context_diagnosis import PURE, WINDOW_COLUMNS, label_name
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT

SUPPORTED_DATASET = "le2i"
TRUE_LABEL = "other"
PREDICTED_LABEL = "walk"
ERROR = "persistent_error"
CONTROL = "persistent_correct"
ROLES: tuple[str, ...] = (ERROR, CONTROL)

# Persistência: em todo braço, pelo menos esta fração das seeds prevê `walk`
# (erro) ou `other` (controle). Com cinco seeds, são 4 de 5 em cada braço.
PERSISTENCE_MIN_ARM_SHARE = 0.8
FOCUS_ENVIRONMENTS: tuple[str, ...] = ("Lecture_room", "Coffee_room_02")
MAX_SEGMENTS_PER_GROUP = 8
SHEET_COLUMNS = 6

KEY_DIR = "key"
WORKSHEET_DIR = "worksheet"
SHEETS_DIR = "sheets"
SELECTION_JSON_FILE = "review_selection.json"
SELECTION_SUMMARY_FILE = "review_selection.md"
KEY_CSV_FILE = "review_key.csv"
WORKSHEET_CSV_FILE = "review_worksheet.csv"
FRAMES_CSV_FILE = "review_frames.csv"
INSTRUCTIONS_FILE = "instructions.md"
LOCAL_OUTPUT_ROOT = REPOSITORY_ROOT / "runs" / "local"

PREDICTION_COLUMN = re.compile(r"^pred_(?P<arm>[A-Z][A-Z0-9]*)_seed(?P<seed>\d+)$")
SEGMENT_KEY: tuple[str, ...] = ("split", "video_id", "segment_start", "segment_end")
WINDOW_KEY: tuple[str, ...] = ("split", "video_id", "k_end")
WINDOW_IDENTITY: tuple[str, ...] = ("env", "subject", "true_label")

REVIEW_FIELDS: dict[str, str] = {
    "reviewer": "texto livre",
    "visible_action": "texto livre: ação visível ao longo dos 24 quadros",
    "apparent_walking": "yes | no | partial | unclear",
    "annotation_ambiguity": "none | boundary | label | unclear",
    "person_visibility": "full | partial | occluded | absent | unclear",
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
FRAME_COLUMNS = [
    "review_id",
    "position",
    "frame_index",
    "src_index",
    "time_s",
    "label",
    "replicated",
]
CONTEXT_FIELDS = [
    "segment_start",
    "segment_end",
    "segment_length",
    "preceding_label",
    "purity",
    "n_final_label_frames",
    "frames_since_segment_start",
    "frames_to_segment_end",
    "start_distance",
    "end_distance",
    "n_replicated_frames",
    "edge_padding",
    "pose_final",
]

DecodeFrames = Callable[[Path, Sequence[int]], list[np.ndarray]]
Record = dict[str, Any]
SheetHeader = Callable[[str, Record, pd.DataFrame], tuple[str, ...]]


@dataclass(frozen=True)
class ReviewSources:
    windows_csv: Path
    frames_path: Path
    video_paths: Mapping[str, Path]
    video_sha256: Mapping[str, str]
    extra_inputs: tuple[Path, ...] = ()
    protected: tuple[Path, ...] = ()


def prediction_columns(columns: Sequence[str]) -> dict[str, list[str]]:
    duplicated = sorted({column for column in columns if list(columns).count(column) > 1})
    if duplicated:
        raise ValueError(f"window_context_windows.csv com colunas repetidas: {duplicated}")
    seen: dict[tuple[str, int], str] = {}
    for column in columns:
        if not column.startswith("pred_"):
            continue
        match = PREDICTION_COLUMN.match(column)
        if match is None or match["arm"] not in ARMS:
            raise ValueError(f"coluna de predição inesperada: {column!r}")
        arm, seed = match["arm"], int(match["seed"])
        if (arm, seed) in seen:
            raise ValueError(f"braço {arm} com seed {seed} repetido: {seen[(arm, seed)]!r} e {column!r}")
        if match["seed"] != str(seed):
            raise ValueError(f"coluna de predição com seed fora da forma canônica: {column!r}")
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"coluna de predição com seed inesperada: {column!r}; esperadas {list(EXPECTED_SEEDS)}")
        seen[(arm, seed)] = column
    missing = [f"pred_{arm}_seed{seed}" for arm in ARMS for seed in EXPECTED_SEEDS if (arm, seed) not in seen]
    if missing:
        raise ValueError(f"window_context_windows.csv sem as colunas de predição {missing}")
    return {arm: [seen[(arm, seed)] for seed in EXPECTED_SEEDS] for arm in ARMS}


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle), [])


def validate_predictions(windows: pd.DataFrame, columns: dict[str, list[str]], path: Path) -> None:
    for column in (column for arm in ARMS for column in columns[arm]):
        invalid = sorted(set(windows[column].astype(str)) - set(LE2I_LABEL_NAMES))
        if invalid:
            raise ValueError(f"{path}: {column} com rótulos inválidos {invalid}; válidos {list(LE2I_LABEL_NAMES)}")


def load_windows(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"window_context_windows.csv da Etapa 3A não encontrado: {path}")
    columns = prediction_columns(_csv_header(path))
    windows = pd.read_csv(path, keep_default_na=False)
    absent = [column for column in WINDOW_COLUMNS if column not in windows.columns]
    if absent:
        raise ValueError(f"{path}: colunas ausentes {absent}")
    unknown_splits = sorted(set(windows["split"].astype(str)) - set(SPLITS))
    if unknown_splits:
        raise ValueError(f"{path}: splits inesperados {unknown_splits}")
    if windows.duplicated(list(WINDOW_KEY)).any():
        raise ValueError(f"{path}: (split, video_id, k_end) repetido")
    validate_predictions(windows, columns, path)
    return windows, columns


def expected_windows(frames: pd.DataFrame) -> pd.DataFrame:
    in_splits = cast(pd.DataFrame, frames[frames["split"].isin(SPLITS)])
    index = build_window_index(in_splits, EVAL_STRIDE, drop_ignored=True)
    return pd.DataFrame({
        "split": index["split"].astype(str),
        "video_id": index["video_id"].astype(str),
        "k_end": index["k_end"].astype(np.int64),
        "env": index["env"].astype(str),
        "subject": index["subject"].astype(np.int64),
        "true_label": [label_name(int(label)) for label in index["label"]],
    })


def _sample(table: pd.DataFrame) -> list[tuple[str, str, int]]:
    rows = cast(pd.DataFrame, table.sort_values(list(WINDOW_KEY))).head(5)
    return [(str(split), str(video_id), int(k_end)) for split, video_id, k_end in rows[list(WINDOW_KEY)].to_numpy()]


def reconcile_windows(windows: pd.DataFrame, frames: pd.DataFrame, path: Path) -> None:
    observed = pd.DataFrame({
        "split": windows["split"].astype(str),
        "video_id": windows["video_id"].astype(str),
        "k_end": windows["k_end"].astype(np.int64),
        **{field: windows[field].astype(str) for field in ("env", "true_label")},
        "subject": windows["subject"].astype(np.int64),
    })
    joined = observed.merge(expected_windows(frames), on=list(WINDOW_KEY), how="outer",
                            suffixes=("", "_frames"), indicator=True)
    for side, description in (("right_only", "ausentes do CSV"), ("left_only", "sem janela rotulada em frames.parquet")):
        unmatched = cast(pd.DataFrame, joined[joined["_merge"] == side])
        if not unmatched.empty:
            raise ValueError(f"{path}: {len(unmatched)} janela(s) rotulada(s) de val/test {description}: "
                             f"{_sample(unmatched)}")
    for field in WINDOW_IDENTITY:
        diverging = cast(pd.DataFrame, joined[joined[field] != joined[f"{field}_frames"]])
        if not diverging.empty:
            raise ValueError(f"{path}: {field} diverge de frames.parquet em {len(diverging)} janela(s): "
                             f"{_sample(diverging)}")


def persistence_table(windows: pd.DataFrame, columns: dict[str, list[str]]) -> pd.DataFrame:
    other = cast(pd.DataFrame, windows[windows["true_label"] == TRUE_LABEL]).copy()
    walk_shares = []
    other_shares = []
    for arm in ARMS:
        predictions = other[columns[arm]]
        walk_votes = (predictions == PREDICTED_LABEL).sum(axis=1).astype(int)
        other_votes = (predictions == TRUE_LABEL).sum(axis=1).astype(int)
        other[f"walk_votes_{arm}"] = walk_votes
        other[f"other_votes_{arm}"] = other_votes
        other[f"n_seeds_{arm}"] = len(columns[arm])
        walk_shares.append(walk_votes / len(columns[arm]))
        other_shares.append(other_votes / len(columns[arm]))
    all_columns = [column for arm in ARMS for column in columns[arm]]
    other["n_runs"] = len(all_columns)
    other["walk_share"] = (other[all_columns] == PREDICTED_LABEL).sum(axis=1) / len(all_columns)
    other["other_share"] = (other[all_columns] == TRUE_LABEL).sum(axis=1) / len(all_columns)
    other["min_arm_walk_share"] = np.min(np.stack(walk_shares, axis=1), axis=1) if len(other) else []
    other["min_arm_other_share"] = np.min(np.stack(other_shares, axis=1), axis=1) if len(other) else []
    other["role"] = np.select(
        [
            other["min_arm_walk_share"] >= PERSISTENCE_MIN_ARM_SHARE,
            other["min_arm_other_share"] >= PERSISTENCE_MIN_ARM_SHARE,
        ],
        [ERROR, CONTROL],
        "",
    )
    other = other.drop(columns=all_columns)
    return cast(pd.DataFrame, other.sort_values(["split", "video_id", "k_end"])).reset_index(drop=True)


def records(table: pd.DataFrame) -> list[Record]:
    return [{str(key): _scalar(value) for key, value in row.items()} for row in table.to_dict("records")]


def _role_share(row: Record, role: str) -> float:
    return float(row["min_arm_walk_share" if role == ERROR else "min_arm_other_share"])


def candidate_rank(row: Record, role: str) -> tuple:
    boundary_distance = min(int(row["frames_since_segment_start"]), int(row["frames_to_segment_end"]))
    return (row["purity"] != PURE, -_role_share(row, role), -boundary_distance, int(row["k_end"]))


def _overlapping(row: Record, selected: list[Record]) -> bool:
    return any(
        item["split"] == row["split"]
        and item["video_id"] == row["video_id"]
        and abs(int(item["k_end"]) - int(row["k_end"])) < WINDOW_FRAMES
        for item in selected
    )


def _segment_candidates(table: pd.DataFrame, split: str, env: str, role: str) -> list[list[Record]]:
    pool = cast(pd.DataFrame, table[(table["split"] == split) & (table["env"] == env) & (table["role"] == role)])
    segments = []
    for _key, group in pool.groupby(list(SEGMENT_KEY), sort=True):
        rows = records(group)
        segments.append(sorted(rows, key=lambda row: candidate_rank(row, role)))
    return sorted(
        segments,
        key=lambda rows: (*candidate_rank(rows[0], role), str(rows[0]["video_id"]), int(rows[0]["segment_start"])),
    )


def select_items(table: pd.DataFrame) -> tuple[list[Record], list[dict]]:
    selected: list[Record] = []
    exclusions: list[dict] = []
    for split in SPLITS:
        for env in FOCUS_ENVIRONMENTS:
            for role in ROLES:
                if role == CONTROL and not any(
                    item["split"] == split and item["env"] == env and item["role"] == ERROR for item in selected
                ):
                    continue
                n_selected = 0
                for rows in _segment_candidates(table, split, env, role):
                    segment = {field: rows[0][field] for field in SEGMENT_KEY}
                    if n_selected >= MAX_SEGMENTS_PER_GROUP:
                        exclusions.append({**_plain(segment), "env": env, "role": role, "reason": "cap"})
                        continue
                    choice = next((row for row in rows if not _overlapping(row, selected)), None)
                    if choice is None:
                        exclusions.append({**_plain(segment), "env": env, "role": role, "reason": "overlap"})
                        continue
                    selected.append(choice)
                    n_selected += 1
    return selected, exclusions


def _scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _plain(record: Mapping) -> dict:
    return {key: _scalar(value) for key, value in record.items()}


def review_order_key(split: str, video_id: str, k_end: int) -> str:
    return hashlib.sha256(f"{split}/{video_id}/{k_end}".encode()).hexdigest()


def assign_review_ids(items: list[Record]) -> list[tuple[str, Record]]:
    assigned: list[tuple[str, Record]] = []
    for split in SPLITS:
        in_split = [item for item in items if item["split"] == split]
        ordered = sorted(in_split, key=lambda item: review_order_key(split, str(item["video_id"]), int(item["k_end"])))
        assigned.extend((f"{split}-{index:02d}", item) for index, item in enumerate(ordered, start=1))
    return assigned


def video_frame_table(frames: pd.DataFrame, video_id: str) -> pd.DataFrame:
    video = cast(pd.DataFrame, frames[frames["video_id"] == video_id])
    if video.empty:
        raise ValueError(f"video_id={video_id!r} ausente de frames.parquet")
    ordered = cast(pd.DataFrame, video.sort_values("frame_index")).reset_index(drop=True)
    if not np.array_equal(ordered["frame_index"].to_numpy(), np.arange(len(ordered))):
        raise ValueError(f"video_id={video_id!r}: frame_index não é contíguo 0..K-1")
    return ordered


def window_mapping(video: pd.DataFrame, k_end: int) -> pd.DataFrame:
    n_frames = len(video)
    if not 0 <= k_end < n_frames:
        raise ValueError(f"k_end={k_end} fora da grade de {n_frames} quadros")
    indices = window_frame_indices(k_end, n_frames)
    raw = np.arange(k_end - WINDOW_FRAMES + 1, k_end + 1)
    rows = video.iloc[indices]
    return pd.DataFrame(
        {
            "position": np.arange(WINDOW_FRAMES),
            "frame_index": indices.astype(int),
            "src_index": rows["src_index"].to_numpy(dtype=np.int64),
            "time_s": rows["time_s"].to_numpy(dtype=np.float64),
            "label": [label_name(int(label)) for label in rows["label"]],
            "replicated": raw < 0,
        }
    )


def validate_alignment(item: Record, video: pd.DataFrame, mapping: pd.DataFrame) -> None:
    context = f"{item['split']}/{item['video_id']} k_end={item['k_end']}"
    first = records(video.iloc[:1])[0]
    for field in ("split", "env"):
        if str(first[field]) != str(item[field]):
            raise ValueError(f"{context}: {field} {first[field]!r} de frames.parquet diverge do CSV {item[field]!r}")
    if int(first["subject"]) != int(item["subject"]):
        raise ValueError(f"{context}: sujeito de frames.parquet diverge do CSV")
    labels = video["label"].to_numpy(dtype=np.int64)
    k_end = int(item["k_end"])
    true_id = LE2I_LABEL_NAMES.index(TRUE_LABEL)
    if labels[k_end] != true_id:
        raise ValueError(f"{context}: rótulo do quadro final em frames.parquet não é {TRUE_LABEL}")
    if int(np.sum(mapping["label"].to_numpy() == TRUE_LABEL)) != int(item["n_final_label_frames"]):
        raise ValueError(f"{context}: quadros com o rótulo final divergem do CSV")
    if int(np.sum(mapping["replicated"].to_numpy())) != int(item["n_replicated_frames"]):
        raise ValueError(f"{context}: quadros replicados divergem do CSV")
    start, end = int(item["segment_start"]), int(item["segment_end"])
    inside = labels[start : end + 1]
    if (
        not start <= k_end <= end
        or not np.all(inside == true_id)
        or (start > 0 and labels[start - 1] == true_id)
        or (end < len(labels) - 1 and labels[end + 1] == true_id)
    ):
        raise ValueError(f"{context}: trecho [{start}, {end}] do CSV não corresponde a frames.parquet")


def tile_lines(row: Record) -> tuple[str, ...]:
    replicated = " (borda)" if bool(row["replicated"]) else ""
    return (
        f"#{int(row['position']):02d} k={int(row['frame_index'])} src={int(row['src_index'])}",
        f"{float(row['time_s']):.1f}s {row['label']}{replicated}",
    )


def contact_sheet(frames_rgb: Sequence[np.ndarray], mapping: pd.DataFrame, header: tuple[str, ...]) -> np.ndarray:
    tiles = [_with_caption_strip(frame, tile_lines(row)) for frame, row in zip(frames_rgb, records(mapping))]
    cell_height = max(tile.shape[0] for tile in tiles)
    cell_width = max(tile.shape[1] for tile in tiles)
    n_rows = -(-len(tiles) // SHEET_COLUMNS)
    grid = np.zeros((n_rows * cell_height, SHEET_COLUMNS * cell_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        top, left = tile_origin(index, cell_height, cell_width)
        grid[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
    return _with_caption_strip(grid, header)


def tile_origin(index: int, cell_height: int, cell_width: int) -> tuple[int, int]:
    return (index // SHEET_COLUMNS) * cell_height, (index % SHEET_COLUMNS) * cell_width


def sheet_header(review_id: str, item: Record, mapping: pd.DataFrame) -> tuple[str, ...]:
    return (
        f"{review_id} | {item['video_id']} | {item['env']} | true_label: {TRUE_LABEL}",
        f"janela k={int(mapping['frame_index'].iloc[0])}..{int(item['k_end'])} | "
        f"src={int(mapping['src_index'].iloc[0])}..{int(mapping['src_index'].iloc[-1])} | "
        f"{float(mapping['time_s'].iloc[0]):.1f}s..{float(mapping['time_s'].iloc[-1]):.1f}s | "
        f"{WINDOW_FRAMES} quadros a {TARGET_FPS} fps",
    )


def png_bytes(image_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def sheet_relative_path(review_id: str) -> str:
    return f"{SHEETS_DIR}/{review_id}.png"


def render_sheets(
    reviews: list[tuple[str, Record, pd.DataFrame]],
    video_paths: Mapping[str, Path],
    decode_fn: DecodeFrames,
    header: SheetHeader = sheet_header,
) -> tuple[dict[str, bytes], list[dict]]:
    sheets: dict[str, bytes] = {}
    missing: list[dict] = []
    by_video: dict[str, list[tuple[str, Record, pd.DataFrame]]] = {}
    for review in reviews:
        by_video.setdefault(str(review[1]["video_id"]), []).append(review)
    for video_id in sorted(by_video):
        path = video_paths.get(video_id)
        if path is None or not path.is_file():
            missing.append({
                "video_id": video_id,
                "path": None if path is None else display_path(path),
                "reason": "vídeo bruto ausente; folha de contato não gerada",
                "review_ids": [review_id for review_id, _item, _mapping in by_video[video_id]],
            })
            continue
        src_indices = sorted({int(src) for _id, _item, mapping in by_video[video_id] for src in mapping["src_index"]})
        decoded = decode_fn(path, src_indices)
        if len(decoded) != len(src_indices):
            raise ValueError(f"{video_id}: decode_frames devolveu {len(decoded)} quadros para {len(src_indices)} índices")
        frame_by_src = dict(zip(src_indices, decoded))
        for review_id, item, mapping in by_video[video_id]:
            frames_rgb = [frame_by_src[int(src)] for src in mapping["src_index"]]
            sheets[review_id] = png_bytes(contact_sheet(frames_rgb, mapping, header(review_id, item, mapping)))
    return sheets, missing


def _segment_counts(table: pd.DataFrame) -> pd.DataFrame:
    counts = table.groupby(list(SEGMENT_KEY), sort=True)["role"].agg(
        segment_error_windows=lambda roles: int((roles == ERROR).sum()),
        segment_control_windows=lambda roles: int((roles == CONTROL).sum()),
        segment_other_windows="size",
    )
    return counts.reset_index()


def key_row(review_id: str, item: Record, mapping: pd.DataFrame, segment: Record, sheet: str | None) -> dict:
    row = {
        "review_id": review_id,
        "split": item["split"],
        "role": item["role"],
        "video_id": item["video_id"],
        "env": item["env"],
        "subject": int(item["subject"]),
        "true_label": item["true_label"],
        "k_end": int(item["k_end"]),
        "window_frame_start": int(mapping["frame_index"].iloc[0]),
        "window_frame_end": int(mapping["frame_index"].iloc[-1]),
        "src_index_start": int(mapping["src_index"].iloc[0]),
        "src_index_end": int(mapping["src_index"].iloc[-1]),
        "time_start_s": float(mapping["time_s"].iloc[0]),
        "time_end_s": float(mapping["time_s"].iloc[-1]),
        **{field: _scalar(item[field]) for field in CONTEXT_FIELDS},
        "n_runs": int(item["n_runs"]),
        "walk_share": float(item["walk_share"]),
        "other_share": float(item["other_share"]),
        "min_arm_walk_share": float(item["min_arm_walk_share"]),
        "min_arm_other_share": float(item["min_arm_other_share"]),
    }
    for arm in ARMS:
        row[f"walk_votes_{arm}"] = int(item[f"walk_votes_{arm}"])
        row[f"other_votes_{arm}"] = int(item[f"other_votes_{arm}"])
        row[f"n_seeds_{arm}"] = int(item[f"n_seeds_{arm}"])
    for field in ("segment_other_windows", "segment_error_windows", "segment_control_windows"):
        row[field] = int(segment[field])
    row["sheet"] = sheet
    return row


def worksheet_row(key: dict) -> dict:
    return {
        **{column: key[column] for column in WORKSHEET_COLUMNS if column in key},
        "sheet": key["sheet"] or "",
        **{field: "" for field in REVIEW_FIELDS},
    }


def coverage(table: pd.DataFrame, keys: list[dict], exclusions: list[dict]) -> dict:
    report: dict[str, dict] = {}
    for split in SPLITS:
        report[split] = {}
        in_split = cast(pd.DataFrame, table[table["split"] == split])
        for env in sorted(set(in_split["env"].astype(str))):
            group = cast(pd.DataFrame, in_split[in_split["env"] == env])
            entry: dict = {
                "focus": env in FOCUS_ENVIRONMENTS,
                "n_other_windows": len(group),
                "n_other_segments": int(group.groupby(list(SEGMENT_KEY)).ngroups),
            }
            for role in ROLES:
                in_role = cast(pd.DataFrame, group[group["role"] == role])
                chosen = [key for key in keys if key["split"] == split and key["env"] == env and key["role"] == role]
                skipped = [item for item in exclusions
                           if item["split"] == split and item["env"] == env and item["role"] == role]
                entry[role] = {
                    "n_windows": len(in_role),
                    "n_segments": int(in_role.groupby(list(SEGMENT_KEY)).ngroups) if len(in_role) else 0,
                    "n_selected": len(chosen),
                    "n_excluded_cap": sum(item["reason"] == "cap" for item in skipped),
                    "n_excluded_overlap": sum(item["reason"] == "overlap" for item in skipped),
                    "selected_purity": _tally(key["purity"] for key in chosen),
                    "selected_start_distance": _tally(key["start_distance"] for key in chosen),
                }
            report[split][env] = entry
    return report


def _tally(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _limitations() -> list[str]:
    return [
        "Preparação exploratória e não canônica (Etapa 3B): não refaz inferência, não avalia modelo e não altera "
        "anotações, janelamento, features, checkpoints, treino, métricas nem decisões experimentais.",
        "As predições são as da Etapa 3A (reexecução validada sobre checkpoints congelados); a seleção herda "
        "as limitações daquela etapa.",
        "O split test já foi exposto pelos diagnósticos anteriores; val e test são contabilizados separadamente "
        "e nada aqui seleciona modelo, seed, limiar ou protocolo.",
        "A persistência é contada entre seeds e braços que compartilham anotação, janela e partição; ela não "
        "mede incerteza entre sujeitos nem independência entre braços.",
        "Um item por trecho anotado e por papel; janelas do mesmo vídeo a menos de WINDOW_FRAMES quadros não são "
        "selecionadas juntas. Trechos do mesmo vídeo ou sujeito ainda não são observações independentes.",
        "O mascaramento é parcial: ambiente, vídeo e rótulos anotados aparecem na planilha e nas folhas.",
        "Nada aqui descreve o conteúdo dos vídeos; achados visuais só existem depois do preenchimento manual "
        "da planilha.",
    ]


def criteria() -> dict:
    return {
        "dataset": SUPPORTED_DATASET,
        "true_label": TRUE_LABEL,
        "predicted_label": PREDICTED_LABEL,
        "persistence_min_arm_share": PERSISTENCE_MIN_ARM_SHARE,
        "focus_environments": list(FOCUS_ENVIRONMENTS),
        "max_segments_per_group": MAX_SEGMENTS_PER_GROUP,
        "segment_key": list(SEGMENT_KEY),
        "candidate_rank": [
            "janelas puras primeiro",
            "maior fração mínima entre braços do papel",
            "maior distância à fronteira mais próxima do trecho",
            "menor k_end",
        ],
        "controls_only_in_split_environments_with_selected_errors": True,
        "min_k_end_gap_same_video": WINDOW_FRAMES,
        "review_id_order": "sha256 de '<split>/<video_id>/<k_end>' dentro de cada split",
        "window_frames": WINDOW_FRAMES,
        "eval_stride": EVAL_STRIDE,
        "target_fps": TARGET_FPS,
    }


def render_summary(report: dict) -> str:
    lines = [
        f"# Conjunto de revisão dos erros {TRUE_LABEL}→{PREDICTED_LABEL} (Etapa 3B, não canônico)",
        "",
        "Chave de desmascaramento: não abrir antes de concluir a revisão visual cega.",
        "",
        f"Persistência: fração mínima entre braços ≥ {PERSISTENCE_MIN_ARM_SHARE:.0%} das seeds. Ambientes "
        f"priorizados: {', '.join(FOCUS_ENVIRONMENTS)}. No máximo {MAX_SEGMENTS_PER_GROUP} trechos por "
        "(split, ambiente, papel).",
    ]
    for split, envs in report["coverage"].items():
        lines += [
            "",
            f"## {split}",
            "",
            "| Ambiente | Prioritário | Janelas other | Trechos other | Janelas erro / trechos | "
            "Janelas controle / trechos | Selecionados erro / controle | Excluídos (limite / sobreposição) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        if not envs:
            lines.append("| — | — | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |")
        for env, entry in envs.items():
            error, control = entry[ERROR], entry[CONTROL]
            lines.append(
                f"| {env} | {'sim' if entry['focus'] else 'não'} | {entry['n_other_windows']} | "
                f"{entry['n_other_segments']} | {error['n_windows']} / {error['n_segments']} | "
                f"{control['n_windows']} / {control['n_segments']} | {error['n_selected']} / "
                f"{control['n_selected']} | {error['n_excluded_cap'] + control['n_excluded_cap']} / "
                f"{error['n_excluded_overlap'] + control['n_excluded_overlap']} |"
            )
    lines += ["", "## Entradas ausentes", ""]
    lines += [f"- {item['video_id']}: {item['reason']} ({', '.join(item['review_ids'])})"
              for item in report["missing_inputs"]] or ["- Nenhuma."]
    lines += ["", "## Limitações", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def render_instructions() -> str:
    lines = [
        f"# Revisão visual cega dos trechos {TRUE_LABEL} (Etapa 3B)",
        "",
        "1. Abra cada folha em `sheets/` na ordem dos `review_id`. Cada folha mostra os 24 quadros da janela, da "
        "esquerda para a direita e de cima para baixo, com o índice da grade (`k`), o quadro de origem (`src`), o "
        "tempo e o rótulo anotado; `(borda)` marca quadros replicados no início do vídeo.",
        "2. Preencha só as colunas abaixo de `review_worksheet.csv`; deixe em branco o que não puder ser julgado.",
        "3. Não abra o diretório `key/` antes de terminar todas as linhas.",
        "",
        "| Campo | Valores |",
        "| --- | --- |",
    ]
    lines += [f"| `{field}` | {values} |" for field, values in REVIEW_FIELDS.items()]
    lines += [
        "",
        "`annotation_ambiguity`: `boundary` quando a ação muda perto da fronteira anotada; `label` quando o "
        f"rótulo {TRUE_LABEL} parece não descrever a ação visível.",
    ]
    return "\n".join(lines) + "\n"


def protected_paths(sources: ReviewSources) -> list[Path]:
    return [
        REFERENCE_RUN_ROOT,
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "data",
        REPOSITORY_ROOT / "docs",
        sources.frames_path.parent,
        sources.windows_csv.parent,
        *sources.protected,
    ]


def check_review_output_dir(output_dir: Path, sources: ReviewSources) -> None:
    check_output_dir(output_dir, protected_paths(sources))
    resolved = output_dir.resolve()
    local_root = LOCAL_OUTPUT_ROOT.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT) and not (resolved == local_root or local_root in resolved.parents):
        raise ValueError(f"--output-dir {output_dir} dentro do repositório precisa ficar em {LOCAL_OUTPUT_ROOT}")


@dataclass(frozen=True)
class ReviewPlan:
    columns: dict[str, list[str]]
    table: pd.DataFrame
    reviews: list[tuple[str, Record, pd.DataFrame]]
    exclusions: list[dict]

    @property
    def video_ids(self) -> list[str]:
        return sorted({str(item["video_id"]) for _review_id, item, _mapping in self.reviews})


def plan_review_set(sources: ReviewSources) -> ReviewPlan:
    windows, columns = load_windows(sources.windows_csv)
    if not sources.frames_path.is_file():
        raise FileNotFoundError(f"frames.parquet não encontrado: {sources.frames_path}")
    frames = pd.read_parquet(sources.frames_path)
    reconcile_windows(windows, frames, sources.windows_csv)
    table = persistence_table(windows, columns)
    items, exclusions = select_items(table)
    reviews: list[tuple[str, Record, pd.DataFrame]] = []
    for review_id, item in assign_review_ids(items):
        video = video_frame_table(frames, str(item["video_id"]))
        mapping = window_mapping(video, int(item["k_end"]))
        validate_alignment(item, video, mapping)
        reviews.append((review_id, item, mapping))
    return ReviewPlan(columns=columns, table=table, reviews=reviews, exclusions=exclusions)


def build_review_set(
    plan: ReviewPlan, sources: ReviewSources, decode_fn: DecodeFrames
) -> tuple[dict, dict[Path, str], dict[str, bytes]]:
    sheets, missing = render_sheets(plan.reviews, sources.video_paths, decode_fn)
    segments = {tuple(row[field] for field in SEGMENT_KEY): row for row in records(_segment_counts(plan.table))}
    keys = []
    frame_rows = []
    for review_id, item, mapping in plan.reviews:
        segment = segments[tuple(item[field] for field in SEGMENT_KEY)]
        sheet = sheet_relative_path(review_id) if review_id in sheets else None
        keys.append(key_row(review_id, item, mapping, segment, sheet))
        frame_rows.append(mapping.assign(review_id=review_id)[FRAME_COLUMNS])
    keys.sort(key=lambda key: key["review_id"])
    frames_csv = pd.concat(frame_rows, ignore_index=True) if frame_rows else pd.DataFrame(columns=FRAME_COLUMNS)
    frames_csv = cast(pd.DataFrame, frames_csv.sort_values(["review_id", "position"])).reset_index(drop=True)

    report = {
        "analysis": "other_walk_review_set",
        "stage": "3B",
        "canonical": False,
        "read_only": True,
        "criteria": criteria(),
        "prediction_columns": plan.columns,
        "review_fields": REVIEW_FIELDS,
        "layout": {
            "blinded": [f"{WORKSHEET_DIR}/{WORKSHEET_CSV_FILE}", f"{WORKSHEET_DIR}/{FRAMES_CSV_FILE}",
                        f"{WORKSHEET_DIR}/{SHEETS_DIR}/", f"{WORKSHEET_DIR}/{INSTRUCTIONS_FILE}"],
            "unblinding_key": [f"{KEY_DIR}/{SELECTION_JSON_FILE}", f"{KEY_DIR}/{KEY_CSV_FILE}",
                               f"{KEY_DIR}/{SELECTION_SUMMARY_FILE}"],
        },
        "coverage": coverage(plan.table, keys, plan.exclusions),
        "items": keys,
        "exclusions": plan.exclusions,
        "missing_inputs": missing,
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


def source_files(sources: ReviewSources) -> list[Path]:
    candidates = [sources.windows_csv, sources.frames_path, *sources.extra_inputs]
    return sorted({path.resolve() for path in candidates if path.is_file()})


def video_files(sources: ReviewSources, video_ids: Sequence[str]) -> list[Path]:
    paths = [sources.video_paths[video_id] for video_id in video_ids if video_id in sources.video_paths]
    return sorted({path.resolve() for path in paths if path.is_file()})


def verify_video_hashes(snapshot: dict[str, str], sources: ReviewSources, video_ids: Sequence[str]) -> None:
    for video_id in video_ids:
        path = sources.video_paths.get(video_id)
        digest = None if path is None else snapshot.get(str(path.resolve()))
        expected = sources.video_sha256.get(video_id)
        if digest is not None and expected is not None and digest != expected:
            raise ValueError(f"{video_id}: SHA-256 do vídeo bruto diverge do manifesto")


def _existing_targets(output_dir: Path, names: Sequence[Path]) -> list[str]:
    return [str(output_dir / name) for name in names if (output_dir / name).exists()]


def _write_binary_outputs(targets: dict[Path, bytes]) -> None:
    token = uuid.uuid4().hex
    staged = {path: path.with_name(f".{path.name}.tmp-{token}") for path in targets}
    try:
        for path, data in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            staged[path].write_bytes(data)
        for path, temporary in staged.items():
            os.replace(temporary, path)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    for path in targets:
        print(f"{path}: gravado")


def run_review_set(
    sources: ReviewSources, output_dir: Path, force: bool = False, decode_fn: DecodeFrames = decode_frames
) -> bool:
    check_review_output_dir(output_dir, sources)
    snapshot = snapshot_inputs(source_files(sources))
    plan = plan_review_set(sources)
    snapshot.update(snapshot_inputs(video_files(sources, plan.video_ids)))
    verify_video_hashes(snapshot, sources, plan.video_ids)
    report, texts, sheets = build_review_set(plan, sources, decode_fn)
    report["input_sha256"] = {display_path(Path(path)): digest for path, digest in snapshot.items()}
    texts[Path(KEY_DIR) / SELECTION_JSON_FILE] = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    texts[Path(KEY_DIR) / SELECTION_SUMMARY_FILE] = render_summary(report)
    images = {Path(WORKSHEET_DIR) / sheet_relative_path(review_id): data for review_id, data in sheets.items()}
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
    return True


def le2i_sources(windows_csv: Path) -> ReviewSources:
    adapter = get_dataset(SUPPORTED_DATASET)
    manifest = adapter.load_manifest()
    return ReviewSources(
        windows_csv=windows_csv,
        frames_path=adapter.frames_path,
        video_paths=adapter.video_paths(),
        video_sha256={str(video_id): str(digest) for video_id, digest in zip(manifest["video_id"], manifest["sha256"])},
        extra_inputs=(adapter.manifest_path,),
        protected=(adapter.raw_dir,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Seleciona e renderiza o conjunto de revisão dos erros other→walk persistentes"
    )
    prepare_parser.add_argument("--windows-csv", type=Path, required=True,
                                help="window_context_windows.csv da Etapa 3A")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do conjunto de revisão")

    args = parser.parse_args()
    if args.command == "prepare":
        run_review_set(le2i_sources(args.windows_csv), args.output_dir, force=args.force)
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.other_walk_review_set import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
