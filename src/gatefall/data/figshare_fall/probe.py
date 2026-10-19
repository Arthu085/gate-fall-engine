"""Sondagem opt-in de decodificação, FPS e grade temporal dos vídeos do zip.

Cada membro é extraído para um diretório temporário descartado ao fim da
sondagem; nada é gravado em `data/`.
"""

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import TARGET_FPS, WINDOW_FRAMES
from gatefall.data.resampling import build_time_grid
from gatefall.data.video_metadata import probe_video, resolve_frame_rate

PROBE_COLUMNS = [
    "member",
    "codec",
    "width",
    "height",
    "fps",
    "fps_source",
    "n_frames_header",
    "n_frames_counted",
    "duration_s",
    "grid_frames",
    "unpadded_windows",
    "error",
]


def select_spread_sample(members: list[str], limit: int | None) -> list[str]:
    """Amostra determinística e espalhada sobre a lista ordenada de membros."""
    if limit is not None and limit <= 0:
        raise ValueError(f"limit deve ser positivo, recebido: {limit}")
    ordered = sorted(members)
    if limit is None or limit >= len(ordered):
        return ordered
    positions = np.unique(np.round(np.linspace(0, len(ordered) - 1, limit)).astype(int))
    return [ordered[position] for position in positions]


def count_unpadded_windows(grid_frames: int, window_frames: int = WINDOW_FRAMES) -> int:
    """Janelas em EVAL_STRIDE=1 que não dependem da replicação de borda do início."""
    return max(0, grid_frames - window_frames + 1)


def probe_member(archive: zipfile.ZipFile, member: str, directory: Path) -> dict[str, object]:
    target = directory / "video.mp4"
    row: dict[str, object] = {column: None for column in PROBE_COLUMNS}
    row["member"] = member
    try:
        with archive.open(member) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        metadata = probe_video(target)
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
            n_frames_header=metadata["n_frames_header"],
            n_frames_counted=metadata["n_frames_counted"],
            duration_s=metadata["duration_s"],
            grid_frames=len(times),
            unpadded_windows=count_unpadded_windows(len(times)),
        )
    except (subprocess.CalledProcessError, ValueError, zipfile.BadZipFile, OSError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        target.unlink(missing_ok=True)
    return row


def probe_videos(archive: zipfile.ZipFile, members: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for index, member in enumerate(members, start=1):
            rows.append(probe_member(archive, member, Path(tmp_dir)))
            if index % 100 == 0:
                print(f"  {index}/{len(members)} vídeos sondados", flush=True)
    return pd.DataFrame(rows, columns=PROBE_COLUMNS)


def report_probe(probes: pd.DataFrame) -> bool:
    print("\n=== sondagem de vídeo (ffprobe com contagem por decodificação) ===")
    print(f"vídeos sondados: {len(probes)}")
    if probes.empty:
        print("nenhum vídeo selecionado: a sondagem não verificou nada")
        return False
    failed = probes[probes["error"].notna()]
    for member, error in zip(failed["member"], failed["error"]):
        print(f"  FALHA {member}: {error}")
    decoded = probes[probes["error"].isna()]
    if decoded.empty:
        print("nenhum vídeo decodificado")
        return False

    print("\ncodec x resolução:")
    print(decoded.groupby(["codec", "width", "height"]).size().to_string())
    print("\nfps (fonte):")
    print(decoded.groupby(["fps_source", "fps"]).size().to_string())
    print("\nduração (s), quadros contados e quadros da grade a "
          f"{TARGET_FPS:g} fps:")
    durations = cast(
        pd.DataFrame, decoded[["duration_s", "n_frames_counted", "grid_frames"]]
    ).astype(float)
    print(durations.describe(percentiles=[0.05, 0.5, 0.95]).to_string())
    header_frames = cast(pd.Series, decoded["n_frames_header"])
    header_mismatch = decoded[
        header_frames.notna() & (header_frames != decoded["n_frames_counted"])
    ]
    print(f"\nnb_frames do cabeçalho != quadros decodificados: {len(header_mismatch)}")
    short = decoded[decoded["grid_frames"] < WINDOW_FRAMES]
    print(
        f"vídeos com grade menor que WINDOW_FRAMES={WINDOW_FRAMES}: "
        f"{len(short)} de {len(decoded)} (toda janela desses vídeos depende "
        "de replicação de borda)"
    )
    return failed.empty
