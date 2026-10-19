"""Sondagem dos MP4 do clone local e encaixe na grade e na janela causal do GateFall.

Lê os vídeos no lugar, sem copiar nem gravar nada em `data/`.
"""

import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, TARGET_FPS, WINDOW_FRAMES
from gatefall.data.gmdcsa24.annotations import OMNIFALL_FALL
from gatefall.data.resampling import build_time_grid, labels_for_grid
from gatefall.data.video_metadata import probe_video, resolve_frame_rate

# Primeiro índice da grade cuja janela causal de WINDOW_FRAMES quadros não
# depende da replicação de borda do quadro 0.
FIRST_UNPADDED_INDEX = WINDOW_FRAMES - 1

PROBE_COLUMNS = [
    "video_id",
    "codec",
    "width",
    "height",
    "fps",
    "fps_source",
    "r_frame_rate",
    "n_frames_header",
    "n_frames_counted",
    "duration_s",
    "grid_frames",
    "max_pts_drift_s",
    "error",
]
GRID_COLUMNS = [
    "video_id",
    "grid_frames",
    "unpadded_frames",
    "ignored_frames",
    "overlap_resolved_frames",
    "skipped_segments",
    "fall_frames",
    "unpadded_fall_frames",
    "fall_onset_s",
    "first_fall_index",
    "label_end_minus_duration_s",
]


def read_frame_times(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return np.array(
        [float(line.strip().rstrip(",")) for line in result.stdout.splitlines() if line.strip()]
    )


def max_pts_drift(frame_times: np.ndarray, fps: float) -> float:
    """Maior |PTS − índice/fps|: o erro de tratar o vídeo como CFR, como faz `build_time_grid`."""
    relative = frame_times - frame_times[0]
    return float(np.abs(relative - np.arange(len(relative)) / fps).max())


def probe_file(path: Path, identifier: str) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in PROBE_COLUMNS}
    row["video_id"] = identifier
    try:
        metadata = probe_video(path)
        fps, fps_source = resolve_frame_rate(
            metadata["r_frame_rate"], metadata["avg_frame_rate"]
        )
        times, _ = build_time_grid(metadata["n_frames_counted"], fps, TARGET_FPS)
        row.update(
            codec=metadata["codec"],
            width=metadata["width"],
            height=metadata["height"],
            fps=fps,
            fps_source=fps_source,
            r_frame_rate=metadata["r_frame_rate"],
            n_frames_header=metadata["n_frames_header"],
            n_frames_counted=metadata["n_frames_counted"],
            duration_s=metadata["n_frames_counted"] / fps,
            grid_frames=len(times),
            max_pts_drift_s=max_pts_drift(read_frame_times(path), fps),
        )
    except (subprocess.CalledProcessError, ValueError, OSError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def probe_checkout(root: Path, videos: pd.DataFrame) -> pd.DataFrame:
    rows = [
        probe_file(root / relative, identifier)
        for identifier, relative in zip(videos["video_id"], videos["relative_path"])
    ]
    return pd.DataFrame(rows, columns=PROBE_COLUMNS)


def grid_row(
    identifier: str, grid_frames: int, duration_s: float, segments: pd.DataFrame
) -> dict[str, object]:
    times = np.arange(grid_frames, dtype=np.float64) / TARGET_FPS
    grid = labels_for_grid(segments, times, IGNORE_LABEL)
    fall = grid.labels == OMNIFALL_FALL
    unpadded = np.arange(grid_frames) >= FIRST_UNPADDED_INDEX
    fall_starts = cast(pd.Series, segments.loc[segments["label"] == OMNIFALL_FALL, "start"])
    return {
        "video_id": identifier,
        "grid_frames": grid_frames,
        "unpadded_frames": int(unpadded.sum()),
        "ignored_frames": int((grid.labels == IGNORE_LABEL).sum()),
        "overlap_resolved_frames": grid.n_overlap_resolved,
        "skipped_segments": grid.n_segments_skipped,
        "fall_frames": int(fall.sum()),
        "unpadded_fall_frames": int((fall & unpadded).sum()),
        "fall_onset_s": float(fall_starts.min()) if not fall_starts.empty else None,
        "first_fall_index": int(np.argmax(fall)) if fall.any() else None,
        "label_end_minus_duration_s": (
            float(cast(float, segments["end"].max())) - duration_s
            if not segments.empty
            else None
        ),
    }


def summarize_grid(probes: pd.DataFrame, omnifall: pd.DataFrame) -> pd.DataFrame:
    """Aplica os segmentos OmniFall (ids inteiros) à grade de 10 fps de cada vídeo."""
    decoded = probes[probes["error"].isna()]
    by_video = {key: group for key, group in omnifall.groupby("video_id")}
    empty = cast(pd.DataFrame, omnifall.iloc[0:0])
    rows = [
        grid_row(str(identifier), int(frames), float(duration), by_video.get(identifier, empty))
        for identifier, frames, duration in zip(
            decoded["video_id"], decoded["grid_frames"], decoded["duration_s"]
        )
    ]
    return pd.DataFrame(rows, columns=GRID_COLUMNS)


def report_probe(probes: pd.DataFrame, declared_lengths: pd.Series) -> bool:
    print("\n=== sondagem de vídeo (ffprobe com contagem por decodificação) ===")
    print(f"vídeos sondados: {len(probes)}")
    if probes.empty:
        print("nenhum vídeo selecionado: a sondagem não verificou nada")
        return False
    failed = probes[probes["error"].notna()]
    for identifier, error in zip(failed["video_id"], failed["error"]):
        print(f"  FALHA {identifier}: {error}")
    decoded = cast(pd.DataFrame, probes[probes["error"].isna()])
    if decoded.empty:
        print("nenhum vídeo decodificado")
        return False

    print("\ncodec x resolução por sujeito:")
    subject = decoded["video_id"].str.extract(r"^Subject_(\d+)/")[0].astype(int).rename("subject")
    print(decoded.groupby([subject, "codec", "width", "height"]).size().to_string())
    print("\nr_frame_rate nominal:")
    print(decoded["r_frame_rate"].value_counts().to_string())
    print("fps resolvido (avg_frame_rate):")
    print(decoded["fps"].astype(float).describe(percentiles=[0.05, 0.5, 0.95]).to_string())
    print(f"\nduração (s) = quadros decodificados / fps; grade a {TARGET_FPS:g} fps:")
    durations = decoded[["duration_s", "n_frames_counted", "grid_frames"]].astype(float)
    print(durations.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
    header_frames = cast(pd.Series, decoded["n_frames_header"])
    mismatch = decoded[header_frames.notna() & (header_frames != decoded["n_frames_counted"])]
    print(f"nb_frames do cabeçalho != quadros decodificados: {len(mismatch)}")
    drift = decoded["max_pts_drift_s"].astype(float)
    print("maior |PTS − índice/fps| por vídeo (s), erro da hipótese de taxa constante:")
    print(drift.describe(percentiles=[0.5, 0.95]).to_string())
    print(f"vídeos com desvio > 1/{TARGET_FPS:g} s: {(drift > 1 / TARGET_FPS).sum()}")

    declared = declared_lengths.reindex(decoded["video_id"]).to_numpy(float)
    delta = decoded["duration_s"].to_numpy(float) - declared
    print("duração medida − Length (seconds) do CSV:")
    print(pd.Series(delta).describe(percentiles=[0.05, 0.5, 0.95]).to_string())
    for identifier, value, length in zip(decoded["video_id"], delta, declared):
        if abs(value) > 1.5:
            print(f"  {identifier}: Length={length:g} s, medida {length + value:.2f} s")

    short = decoded[decoded["grid_frames"] < WINDOW_FRAMES]
    print(
        f"\nvídeos com grade >= WINDOW_FRAMES={WINDOW_FRAMES} (ao menos uma janela "
        f"causal sem replicação de borda): {len(decoded) - len(short)} de {len(decoded)} "
        f"({(len(decoded) - len(short)) / len(decoded):.1%})"
    )
    for identifier in short["video_id"]:
        print(f"  curto: {identifier}")
    return failed.empty


def report_grid(grid: pd.DataFrame, categories: pd.Series) -> None:
    print(f"\n=== segmentos OmniFall na grade de {TARGET_FPS:g} fps (labels_for_grid) ===")
    total = int(cast(int, grid["grid_frames"].sum()))
    print(f"quadros da grade: {total}")
    print(
        f"quadros com janela causal sem replicação de borda: {grid['unpadded_frames'].sum()} "
        f"({grid['unpadded_frames'].sum() / total:.1%})"
    )
    print(f"quadros IGNORE_LABEL (não cobertos): {grid['ignored_frames'].sum()}")
    print(f"vídeos com quadros não cobertos: {(grid['ignored_frames'] > 0).sum()}")
    for row in cast(pd.DataFrame, grid[grid["ignored_frames"] > 1]).to_dict("records"):
        print(f"  {row['video_id']}: {row['ignored_frames']} de {row['grid_frames']}")
    print(
        f"quadros com sobreposição resolvida pelo menor start: "
        f"{grid['overlap_resolved_frames'].sum()} em "
        f"{(grid['overlap_resolved_frames'] > 0).sum()} vídeos"
    )
    print(f"segmentos sem quadro na grade: {grid['skipped_segments'].sum()}")
    end_delta = grid["label_end_minus_duration_s"].astype(float)
    print("último fim OmniFall − duração medida (s):")
    print(end_delta.describe(percentiles=[0.05, 0.5, 0.95]).round(3).to_string())

    fall_frames = int(cast(int, grid["fall_frames"].sum()))
    unpadded_fall = int(cast(int, grid["unpadded_fall_frames"].sum()))
    print(
        f"\nquadros `fall`: {fall_frames}; com janela causal sem replicação de borda: "
        f"{unpadded_fall} ({unpadded_fall / max(fall_frames, 1):.1%})"
    )
    first_fall = grid["first_fall_index"].dropna().astype(int)
    complete = first_fall >= FIRST_UNPADDED_INDEX
    print(
        f"vídeos com quadro `fall`: {len(first_fall)}; primeiro quadro `fall` com índice "
        f">= {FIRST_UNPADDED_INDEX} (janela no início da queda sem replicação de borda): "
        f"{complete.sum()} ({complete.mean():.1%})"
    )
    onsets = grid["fall_onset_s"].dropna().astype(float).round(3)
    print("início do primeiro `fall` (s):")
    print(onsets.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())

    print("\nquadros da grade por categoria e sujeito:")
    subject = grid["video_id"].str.extract(r"^Subject_(\d+)/")[0].astype(int)
    category = categories.reindex(grid["video_id"]).to_numpy()
    per_group = (
        grid.assign(subject=subject, category=category)
        .groupby(["subject", "category"])[["grid_frames", "fall_frames"]]
        .sum()
    )
    print(cast(pd.DataFrame, per_group).to_string())
