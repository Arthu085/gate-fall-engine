"""Adapter isolado do GMDCSA24 para o piloto exploratório com rótulos fracos (braço A).

Implementa `docs/analysis/gmdcsa24-weak-label-protocol.md`, aprovado em
2026-10-19 só para o adapter: rótulos OmniFall pinados aceitos sem
verificação humana, os 15 vídeos com anomalia documentada excluídos inteiros,
o tratamento T e a análise de sensibilidade S. As saídas ficam só em
`data/processed/gmdcsa24/`, fora dos artefatos do Le2i. Nada aqui extrai
features, ajusta padronização, monta treino com features ou treina.

As expectativas publicadas no protocolo (`EXPECTED_EXCLUSIONS`,
`PUBLISHED_AGGREGATES`) são fixas: uma divergência reprova a validação e
exige investigação, nunca um ajuste das expectativas ou das regras.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, TARGET_FPS, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.frames import apply_frame_schema, read_frames
from gatefall.data.gmdcsa24.annotations import with_label_names
from gatefall.data.gmdcsa24.decoding import count_decoded_frames
from gatefall.data.gmdcsa24.probe import FIRST_UNPADDED_INDEX, probe_checkout
from gatefall.data.gmdcsa24.provenance import (
    DEFAULT_REVIEW_DIR,
    anomaly_reasons,
    ensure_ignored_output,
    video_boundaries,
)
from gatefall.data.gmdcsa24.source import sha256_of
from gatefall.data.intervals import sweep_gaps_and_overlap, tag_gap_positions
from gatefall.data.resampling import build_time_grid, labels_for_grid
from gatefall.data.windowing import build_window_index
from gatefall.datasets.le2i import LE2I_LABEL_NAMES, PROTOCOL_PROCESSED_DIRS, Le2iDatasetAdapter
from gatefall.hashing import sha256_dataframe
from gatefall.runs import REPOSITORY_ROOT

PROTOCOL_DOC = Path("docs/analysis/gmdcsa24-weak-label-protocol.md")
PROTOCOL_APPROVAL_MARKER = "Status: APROVADO em 2026-10-19"
DEFAULT_OUTPUT_DIR = Path("data/processed/gmdcsa24")
FORBIDDEN_OUTPUT_DIRS = (
    *PROTOCOL_PROCESSED_DIRS.values(),
    Path("data/features"),
    Path("data/labels"),
    Path("runs"),
    Path("src/gatefall/features/stats"),
    DEFAULT_REVIEW_DIR,
)

# S remove a janela cujo `k_end / TARGET_FPS` fica a 0,5 s ou menos de uma
# fronteira; a folga só absorve o arredondamento de ponto flutuante.
BOUNDARY_MARGIN_S = 0.5
DISTANCE_EPSILON_S = 1e-9
DROPPED_CLASSES = ("lie_down", "lying")
DROPPED_LABELS = tuple(LE2I_LABEL_NAMES.index(name) for name in DROPPED_CLASSES)
FALL_LABEL = LE2I_LABEL_NAMES.index("fall")
SPLIT = "train"
ENV = "GMDCSA24"
TREATMENTS = ("T", "S")
DECODER = "ffmpeg -map 0:v:0 -fps_mode passthrough -f rawvideo -pix_fmt rgb24"

MANIFEST_FILE = "manifest.json"
VIDEOS_FILE = "videos.parquet"
FRAMES_FILE = "frames.parquet"
CANDIDATES_FILE = "window_candidates.parquet"
WINDOW_FILES = {"T": "windows_t.parquet", "S": "windows_s.parquet"}
TABLE_FILES = (VIDEOS_FILE, FRAMES_FILE, CANDIDATES_FILE, *WINDOW_FILES.values())

VIDEO_DTYPES: dict[str, str] = {
    "video_id": "string",
    "subject": "int64",
    "category": "string",
    "relative_path": "string",
    "size": "int64",
    "source_sha256": "string",
    "codec": "string",
    "width": "int64",
    "height": "int64",
    "fps": "float64",
    "fps_source": "string",
    "n_frames_header": "Int64",
    "n_frames_counted": "int64",
    "n_frames_decoded": "Int64",
    "grid_frames": "int64",
    "max_pts_drift_s": "float64",
    "excluded": "bool",
    "exclusion_reasons": "string",
    "n_boundaries": "int64",
}
WINDOW_COLUMNS = ["video_id", "split", "env", "subject", "k_end", "label", "n_frames"]
CANDIDATE_FLAGS = ["edge_padded", "dropped_class", "excluded_video", "near_boundary", "in_T", "in_S"]

# Tabela "Os 15 vídeos excluídos" do protocolo, no texto que `anomaly_reasons` produz.
EXPECTED_EXCLUSIONS: dict[str, list[str]] = {
    "Subject_1/ADL/04": ["desvio de PTS de 0.108 s"],
    "Subject_1/ADL/05": ["lacuna de 1.00 s"],
    "Subject_1/Fall/03": ["início do `fall` -1.70 s do autor"],
    "Subject_1/Fall/04": ["início do `fall` -1.58 s do autor"],
    "Subject_1/Fall/09": ["desvio de PTS de 0.101 s"],
    "Subject_1/Fall/12": ["desvio de PTS de 0.131 s"],
    "Subject_1/Fall/15": ["início do `fall` -1.54 s do autor"],
    "Subject_2/ADL/21": ["lacuna de 1.26 s"],
    "Subject_2/Fall/03": ["vídeo Fall sem `fall` no OmniFall"],
    "Subject_2/Fall/15": ["início do `fall` -1.56 s do autor"],
    "Subject_3/ADL/09": ["sobreposição de 4.30 s"],
    "Subject_3/Fall/08": ["desvio de PTS de 0.104 s"],
    "Subject_3/Fall/09": ["sobreposição de 0.10 s"],
    "Subject_4/ADL/02": ["lacuna de 0.56 s"],
    "Subject_4/Fall/15": ["`fall` sem intervalo Falling no CSV original"],
}

# Seções "Contagem esperada" e "Receita de contagem e congelamento (W2)" do protocolo.
PUBLISHED_AGGREGATES: dict[str, object] = {
    "excluded_videos": sorted(EXPECTED_EXCLUSIONS),
    "funnel": {
        "labeled": 3231,
        "edge_padded": 945,
        "dropped_class": 142,
        "excluded_video": 171,
        "T": 1973,
        "near_boundary": 411,
        "S": 1562,
    },
    "class_counts": {
        "T": {
            "walk": 113,
            "fall": 225,
            "fallen": 440,
            "sit_down": 66,
            "sitting": 388,
            "lie_down": 0,
            "lying": 0,
            "stand_up": 46,
            "standing": 112,
            "other": 583,
        },
        "S": {
            "walk": 68,
            "fall": 121,
            "fallen": 364,
            "sit_down": 37,
            "sitting": 353,
            "lie_down": 0,
            "lying": 0,
            "stand_up": 25,
            "standing": 80,
            "other": 514,
        },
    },
    "videos": {
        "source": 160,
        "excluded": 15,
        "included": 145,
        "with_window_T": 137,
        "with_fall_T": 58,
        "with_fall_S": 45,
    },
    "subjects": {
        "T": {"S1": 208, "S2": 759, "S3": 568, "S4": 438},
        "T_fall": {"S1": 25, "S2": 101, "S3": 72, "S4": 27},
    },
}

# Treino do controle Le2i-CS em TRAIN_STRIDE, publicado no protocolo verificado.
LE2I_CS_TRAIN_WINDOWS = 5219
LE2I_EVAL_SPLITS = ("val", "test")


@dataclass(frozen=True)
class AdapterTables:
    videos: pd.DataFrame
    frames: pd.DataFrame
    candidates: pd.DataFrame
    windows: dict[str, pd.DataFrame]
    summary: dict[str, object]


@dataclass(frozen=True)
class Le2iReference:
    """O que o adapter precisa saber do Le2i-CS para provar a separação (V2)."""

    eval_video_ids: frozenset[str]
    source_sha256: frozenset[str]
    train_windows: int
    eval_index_sha256: dict[str, str]


def output_location_problems(output_dir: Path) -> list[str]:
    """Recusa saída dentro, ou acima, dos artefatos do Le2i, das features, dos
    rótulos, dos runs, das estatísticas de padronização e da revisão."""
    resolved = output_dir.resolve()
    problems: list[str] = []
    for forbidden in FORBIDDEN_OUTPUT_DIRS:
        target = forbidden.resolve()
        if resolved == target or target in resolved.parents or resolved in target.parents:
            problems.append(f"{output_dir} se sobrepõe a {forbidden}")
    try:
        ensure_ignored_output(output_dir)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def probe_sources(root: Path, inventory: pd.DataFrame) -> pd.DataFrame:
    """Sondagem, SHA-256 e contagem do decodificador do adapter para cada vídeo."""
    probes = probe_checkout(root, inventory)
    rows: list[dict[str, object]] = []
    for info, probe in zip(inventory.to_dict("records"), probes.to_dict("records")):
        path = root / str(info["relative_path"])
        decoded = (
            None
            if isinstance(probe["error"], str)
            else count_decoded_frames(path, int(probe["width"]), int(probe["height"]))
        )
        rows.append({**info, **probe, "source_sha256": sha256_of(path), "n_frames_decoded": decoded})
    return pd.DataFrame(rows)


def video_frames(
    identifier: str, subject: int, n_frames: int, fps: float, segments: pd.DataFrame
) -> pd.DataFrame:
    """Grade de 10 fps de um vídeo, no esquema de `frames.parquet` do Le2i.

    `src_index` indexa os quadros na ordem do decodificador passthrough."""
    times, src_indices = build_time_grid(n_frames, fps, TARGET_FPS)
    labels = labels_for_grid(segments, times, IGNORE_LABEL).labels
    duration_s = n_frames / fps
    intervals = list(zip(segments["start"].astype(float), segments["end"].astype(float)))
    _, _, gaps = sweep_gaps_and_overlap(intervals, duration_s)
    gap_position = np.full(len(times), None, dtype=object)
    for gap_start, gap_end, position in tag_gap_positions(gaps, duration_s):
        lo = int(np.searchsorted(times, gap_start, side="left"))
        hi = int(np.searchsorted(times, gap_end, side="left"))
        gap_position[lo:hi] = position
    return apply_frame_schema(
        pd.DataFrame(
            {
                "video_id": identifier,
                "split": SPLIT,
                "env": ENV,
                "subject": subject,
                "frame_index": np.arange(len(times)),
                "time_s": times,
                "src_index": src_indices,
                "label": labels,
                "gap_position": gap_position,
            }
        )
    )


def near_boundary(k_end: np.ndarray, boundaries_s: list[float]) -> np.ndarray:
    if not boundaries_s:
        return np.zeros(len(k_end), dtype=bool)
    distance = np.abs(k_end[:, None] / TARGET_FPS - np.asarray(boundaries_s)[None, :])
    return (distance <= BOUNDARY_MARGIN_S + DISTANCE_EPSILON_S).any(axis=1)


def build_candidates(
    frames: pd.DataFrame, boundaries: dict[str, list[float]], excluded: frozenset[str]
) -> pd.DataFrame:
    """Janelas rotuladas em TRAIN_STRIDE de todos os vídeos, com os filtros de T e S.

    T: vídeo não excluído, `k_end >= WINDOW_FRAMES - 1` e rótulo fora de
    `lie_down`/`lying`. S: T sem as janelas perto de fronteira."""
    candidates = build_window_index(frames, stride=TRAIN_STRIDE, drop_ignored=True)
    k_end = candidates["k_end"].to_numpy(np.int64)
    near = np.zeros(len(candidates), dtype=bool)
    for identifier, positions in candidates.groupby("video_id").indices.items():
        near[positions] = near_boundary(k_end[positions], boundaries.get(str(identifier), []))
    edge = k_end < FIRST_UNPADDED_INDEX
    dropped = candidates["label"].isin(DROPPED_LABELS).to_numpy()
    excluded_video = candidates["video_id"].isin(sorted(excluded)).to_numpy()
    in_t = ~edge & ~dropped & ~excluded_video
    return candidates.assign(
        edge_padded=edge,
        dropped_class=dropped,
        excluded_video=excluded_video,
        near_boundary=near,
        in_T=in_t,
        in_S=in_t & ~near,
    )


def treatment_windows(candidates: pd.DataFrame, treatment: str) -> pd.DataFrame:
    selected = cast(pd.DataFrame, candidates[candidates[f"in_{treatment}"]])
    return cast(pd.DataFrame, selected[WINDOW_COLUMNS]).reset_index(drop=True)


def count_by(frame: pd.DataFrame, column: str, keys: list[int], names: list[str]) -> dict[str, int]:
    values = frame[column].to_numpy(np.int64)
    return {name: int((values == key).sum()) for key, name in zip(keys, names)}


def video_count(frame: pd.DataFrame) -> int:
    return len(set(frame["video_id"].astype(str)))


def summarize(videos: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, object]:
    """Agregados comparáveis com o protocolo, recalculados das tabelas do adapter."""
    flags = {name: candidates[name].to_numpy(bool) for name in CANDIDATE_FLAGS}
    edge, dropped = flags["edge_padded"], flags["dropped_class"]
    windows = {treatment: treatment_windows(candidates, treatment) for treatment in TREATMENTS}
    falls = {
        treatment: cast(pd.DataFrame, frame[frame["label"] == FALL_LABEL])
        for treatment, frame in windows.items()
    }
    excluded = sorted(str(identifier) for identifier in videos.loc[videos["excluded"], "video_id"])
    subjects = sorted(int(subject) for subject in videos["subject"].unique())
    subject_names = [f"S{subject}" for subject in subjects]
    labels = list(range(len(LE2I_LABEL_NAMES)))
    return {
        "excluded_videos": excluded,
        "funnel": {
            "labeled": len(candidates),
            "edge_padded": int(edge.sum()),
            "dropped_class": int((~edge & dropped).sum()),
            "excluded_video": int((~edge & ~dropped & flags["excluded_video"]).sum()),
            "T": len(windows["T"]),
            "near_boundary": int((flags["in_T"] & flags["near_boundary"]).sum()),
            "S": len(windows["S"]),
        },
        "class_counts": {
            treatment: count_by(frame, "label", labels, list(LE2I_LABEL_NAMES))
            for treatment, frame in windows.items()
        },
        "videos": {
            "source": len(videos),
            "excluded": len(excluded),
            "included": len(videos) - len(excluded),
            "with_window_T": video_count(windows["T"]),
            "with_fall_T": video_count(falls["T"]),
            "with_fall_S": video_count(falls["S"]),
        },
        "subjects": {
            "T": count_by(windows["T"], "subject", subjects, subject_names),
            "T_fall": count_by(falls["T"], "subject", subjects, subject_names),
        },
    }


def build_tables(
    sources: pd.DataFrame, omnifall: pd.DataFrame, original_segments: pd.DataFrame
) -> AdapterTables:
    """Monta vídeos, grade, janelas candidatas e T/S a partir da sondagem.

    `omnifall` usa os ids inteiros do config `labels`; `sources` vem de
    `probe_sources`, sem vídeo com erro de decodificação."""
    named = with_label_names(omnifall)
    reasons = anomaly_reasons(sources, named, original_segments, sources)
    excluded = frozenset(reasons)
    segments_by_video = {str(key): group for key, group in omnifall.groupby("video_id")}
    named_by_video = {str(key): group for key, group in named.groupby("video_id")}
    empty = cast(pd.DataFrame, omnifall.iloc[0:0])
    boundaries = {
        identifier: [boundary.time_s for boundary in video_boundaries(group)]
        for identifier, group in named_by_video.items()
    }
    frames = pd.concat(
        [
            video_frames(
                str(row["video_id"]),
                int(row["subject"]),
                int(row["n_frames_counted"]),
                float(row["fps"]),
                segments_by_video.get(str(row["video_id"]), empty),
            )
            for row in sources.to_dict("records")
        ],
        ignore_index=True,
    )
    candidates = build_candidates(frames, boundaries, excluded)
    videos = sources.assign(
        excluded=sources["video_id"].isin(sorted(excluded)),
        exclusion_reasons=["; ".join(reasons.get(str(key), [])) for key in sources["video_id"]],
        n_boundaries=[len(boundaries.get(str(key), [])) for key in sources["video_id"]],
    )
    videos = cast(pd.DataFrame, videos[list(VIDEO_DTYPES)].astype(VIDEO_DTYPES))
    videos = videos.sort_values("video_id", ignore_index=True)
    included_frames = cast(pd.DataFrame, frames[~frames["video_id"].isin(sorted(excluded))])
    return AdapterTables(
        videos=videos,
        frames=included_frames.reset_index(drop=True),
        candidates=candidates,
        windows={treatment: treatment_windows(candidates, treatment) for treatment in TREATMENTS},
        summary=summarize(videos, candidates),
    )


def le2i_reference(frames: pd.DataFrame, manifest: pd.DataFrame) -> Le2iReference:
    def split_frames(split: str) -> pd.DataFrame:
        return cast(pd.DataFrame, frames[frames["split"] == split])

    eval_frames = cast(pd.DataFrame, frames[frames["split"].isin(LE2I_EVAL_SPLITS)])
    return Le2iReference(
        eval_video_ids=frozenset(str(identifier) for identifier in eval_frames["video_id"]),
        source_sha256=frozenset(str(digest) for digest in manifest["sha256"]),
        train_windows=len(build_window_index(split_frames("train"), stride=TRAIN_STRIDE)),
        eval_index_sha256={
            split: sha256_dataframe(build_window_index(split_frames(split), stride=EVAL_STRIDE))
            for split in LE2I_EVAL_SPLITS
        },
    )


def load_le2i_reference(adapter: Le2iDatasetAdapter) -> Le2iReference:
    """Lê o Le2i-CS canônico sem alterá-lo; falha se a grade ou o manifesto não existirem."""
    if not adapter.frames_path.exists():
        raise FileNotFoundError(f"{adapter.frames_path} não encontrado")
    return le2i_reference(adapter.load_frames(), adapter.load_manifest())


def optional_sha256(path: Path) -> str | None:
    return sha256_of(path) if path.is_file() else None


def protocol_problems() -> list[str]:
    path = REPOSITORY_ROOT / PROTOCOL_DOC
    if not path.is_file():
        return [f"{PROTOCOL_DOC} não encontrado"]
    if PROTOCOL_APPROVAL_MARKER not in path.read_text(encoding="utf-8"):
        return [f"{PROTOCOL_DOC} sem a marca de aprovação {PROTOCOL_APPROVAL_MARKER!r}"]
    return []


def write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    dataframe.to_parquet(tmp_path)
    os.replace(tmp_path, path)


def table_entry(dataframe: pd.DataFrame) -> dict[str, object]:
    return {"rows": len(dataframe), "content_sha256": sha256_dataframe(dataframe)}


def write_outputs(
    output_dir: Path, tables: AdapterTables, provenance: dict[str, object], force: bool
) -> Path:
    """Grava as tabelas e, por último, o manifesto; sem o manifesto a saída não valida."""
    problems = output_location_problems(output_dir)
    if problems:
        raise ValueError("; ".join(problems))
    manifest_path = output_dir / MANIFEST_FILE
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} já existe; use --force para recriar a saída")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    named_tables = {
        VIDEOS_FILE: tables.videos,
        FRAMES_FILE: tables.frames,
        CANDIDATES_FILE: tables.candidates,
        **{WINDOW_FILES[treatment]: frame for treatment, frame in tables.windows.items()},
    }
    entries: dict[str, dict[str, object]] = {}
    for name, frame in named_tables.items():
        write_parquet(frame, output_dir / name)
        entries[name] = table_entry(read_table(output_dir, name))
    excluded = cast(pd.DataFrame, tables.videos[tables.videos["excluded"]])
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **provenance,
        "parameters": {
            "target_fps": TARGET_FPS,
            "window_frames": WINDOW_FRAMES,
            "train_stride": TRAIN_STRIDE,
            "first_unpadded_index": FIRST_UNPADDED_INDEX,
            "boundary_margin_s": BOUNDARY_MARGIN_S,
            "dropped_classes": list(DROPPED_CLASSES),
            "split": SPLIT,
            "decoder": DECODER,
        },
        "excluded_videos": [
            {
                "video_id": str(row["video_id"]),
                "reasons": str(row["exclusion_reasons"]).split("; "),
                "source_sha256": str(row["source_sha256"]),
            }
            for row in excluded.to_dict("records")
        ],
        "summary": tables.summary,
        "files": entries,
    }
    tmp_path = manifest_path.with_suffix(f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, manifest_path)
    return manifest_path


def compare_expected(expected: object, actual: object, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        problems: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            label = f"{path}{key}"
            if key not in actual:
                problems.append(f"{label}: ausente na saída (esperado {expected[key]!r})")
            elif key not in expected:
                problems.append(f"{label}: inesperado na saída ({actual[key]!r})")
            else:
                problems.extend(compare_expected(expected[key], actual[key], f"{label}."))
        return problems
    if expected != actual:
        return [f"{path.rstrip('.')}: esperado {expected!r}, obtido {actual!r}"]
    return []


def read_table(output_dir: Path, name: str) -> pd.DataFrame:
    return read_frames(output_dir / name)


def integrity_problems(
    output_dir: Path, manifest: dict[str, object], tables: dict[str, pd.DataFrame]
) -> list[str]:
    """Hash e linhas de cada tabela, coerência entre tabelas e com o manifesto."""
    problems: list[str] = []
    recorded = cast(dict[str, dict[str, object]], manifest.get("files", {}))
    for name, frame in tables.items():
        problems.extend(compare_expected(recorded.get(name), table_entry(frame), f"files.{name}."))
    videos = tables[VIDEOS_FILE]
    candidates = tables[CANDIDATES_FILE]
    for treatment, name in WINDOW_FILES.items():
        if not tables[name].equals(treatment_windows(candidates, treatment)):
            problems.append(f"{name} difere das janelas `in_{treatment}` de {CANDIDATES_FILE}")
    problems.extend(
        compare_expected(manifest.get("summary"), summarize(videos, candidates), "summary.")
    )
    frame_counts = {
        str(identifier): int(cast(int, count))
        for identifier, count in tables[FRAMES_FILE].groupby("video_id").size().items()
    }
    included = cast(pd.DataFrame, videos[~videos["excluded"]])
    grid = {str(row["video_id"]): int(row["grid_frames"]) for row in included.to_dict("records")}
    if frame_counts != grid:
        problems.append(f"{FRAMES_FILE} não tem exatamente a grade de cada vídeo incluído")
    for name in WINDOW_FILES.values():
        n_frames = [frame_counts.get(str(identifier)) for identifier in tables[name]["video_id"]]
        if n_frames != [int(value) for value in tables[name]["n_frames"]]:
            problems.append(f"{name}: n_frames diverge da grade de {FRAMES_FILE}")
    listed = [
        str(entry["video_id"])
        for entry in cast(list[dict[str, object]], manifest.get("excluded_videos", []))
    ]
    if listed != sorted(str(identifier) for identifier in videos.loc[videos["excluded"], "video_id"]):
        problems.append(f"lista de exclusão do manifesto difere de {VIDEOS_FILE}")
    return problems


def w1_problems(
    manifest: dict[str, object], videos: pd.DataFrame, expected: dict[str, list[str]]
) -> list[str]:
    problems = protocol_problems()
    excluded = cast(pd.DataFrame, videos[videos["excluded"]])
    actual = {
        str(row["video_id"]): str(row["exclusion_reasons"]).split("; ")
        for row in excluded.to_dict("records")
    }
    problems.extend(compare_expected(expected, actual, "exclusões."))
    review = cast(dict[str, object], manifest.get("review_sheet", {}))
    if review.get("sha256_before") != review.get("sha256_after"):
        problems.append(f"{review.get('path')} mudou durante a execução do adapter")
    return problems


def v2_problems(
    output_dir: Path,
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
    le2i: Le2iReference | None,
) -> list[str]:
    problems = output_location_problems(output_dir)
    videos = tables[VIDEOS_FILE]
    included = set(videos.loc[~videos["excluded"], "video_id"].astype(str))
    for name in (FRAMES_FILE, CANDIDATES_FILE, *WINDOW_FILES.values()):
        splits = set(tables[name]["split"].astype(str))
        if splits - {SPLIT}:
            problems.append(f"{name}: splits {sorted(splits)} fora de {SPLIT!r}")
    for name in (FRAMES_FILE, *WINDOW_FILES.values()):
        outside = set(tables[name]["video_id"].astype(str)) - included
        if outside:
            problems.append(f"{name}: vídeos excluídos ou desconhecidos {sorted(outside)}")
    if le2i is None:
        return [*problems, "referência Le2i-CS indisponível; a separação não foi verificada"]
    shared_ids = set(videos["video_id"].astype(str)) & le2i.eval_video_ids
    if shared_ids:
        problems.append(f"video_id do GMDCSA24 em val/test do Le2i-CS: {sorted(shared_ids)}")
    shared_sources = set(videos["source_sha256"].astype(str)) & le2i.source_sha256
    if shared_sources:
        problems.append(f"{len(shared_sources)} vídeos do GMDCSA24 com o mesmo SHA-256 de um vídeo Le2i")
    if le2i.train_windows != LE2I_CS_TRAIN_WINDOWS:
        problems.append(
            f"treino Le2i-CS com {le2i.train_windows} janelas, o protocolo publica "
            f"{LE2I_CS_TRAIN_WINDOWS}"
        )
    recorded = cast(dict[str, object], manifest.get("le2i_cs", {})).get("eval_index_sha256")
    problems.extend(compare_expected(recorded, le2i.eval_index_sha256, "le2i_cs.eval_index_sha256."))
    return problems


def v5_problems(videos: pd.DataFrame) -> list[str]:
    return [
        f"{row['video_id']}: decodificador do adapter entrega {row['n_frames_decoded']} "
        f"quadros, ffprobe -count_frames conta {row['n_frames_counted']}"
        for row in videos.to_dict("records")
        if pd.isna(row["n_frames_decoded"]) or row["n_frames_decoded"] != row["n_frames_counted"]
    ]


def validate_outputs(
    output_dir: Path,
    le2i: Le2iReference | None,
    expected_exclusions: dict[str, list[str]] = EXPECTED_EXCLUSIONS,
    expected_aggregates: dict[str, object] = PUBLISHED_AGGREGATES,
) -> dict[str, list[str]]:
    """Problemas por condição; a saída só vale com todas as listas vazias.

    Tudo é recalculado das tabelas gravadas: o resumo do manifesto só é
    conferido contra elas, nunca usado como fonte."""
    manifest_path = output_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        return {"integridade": [f"{manifest_path} não encontrado"]}
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
    missing = [name for name in TABLE_FILES if not (output_dir / name).is_file()]
    if missing:
        return {"integridade": [f"tabelas ausentes em {output_dir}: {missing}"]}
    tables = {name: read_table(output_dir, name) for name in TABLE_FILES}
    videos = tables[VIDEOS_FILE]
    return {
        "integridade": integrity_problems(output_dir, manifest, tables),
        "W1": w1_problems(manifest, videos, expected_exclusions),
        "W2": compare_expected(
            expected_aggregates, summarize(videos, tables[CANDIDATES_FILE]), "agregados."
        ),
        "V2": v2_problems(output_dir, manifest, tables, le2i),
        "V5": v5_problems(videos),
    }


def report_validation(results: dict[str, list[str]]) -> bool:
    print("\n=== validação do adapter (W1, W2, V2, V5) ===")
    for condition, problems in results.items():
        print(f"[{'FAIL' if problems else 'PASS'}] {condition}")
        for problem in problems:
            print(f"  {problem}")
    return all(not problems for problems in results.values())


def report_summary(summary: dict[str, object]) -> None:
    print("\n=== agregados do adapter ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
