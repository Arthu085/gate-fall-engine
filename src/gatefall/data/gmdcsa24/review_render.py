"""Contact sheets locais em torno das fronteiras da planilha de revisão de proveniência.

Os quadros vêm do MP4 na ordem de decodificação, sem duplicar nem descartar
quadros (`-fps_mode passthrough`), e cada um é rotulado pelo seu PTS real. A
imagem é evidência automática: mostra o vídeo, não confirma a anotação.
Contact sheets contêm rostos e nunca saem de um diretório ignorado pelo Git.
"""

import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from gatefall.data.gmdcsa24.probe import read_frame_times
from gatefall.data.gmdcsa24.provenance import (
    contact_sheet_path,
    empty_segments,
    ensure_ignored_output,
    labels_at,
)
from gatefall.data.video_metadata import read_ffprobe_header

REVIEW_CONTEXT_S = 0.5
TILE_WIDTH = 240
TILE_COLUMNS = 8
FONT_SIZE = 13
LINE_HEIGHT = FONT_SIZE + 4
TILE_CAPTION_LINES = 2
BORDER = 4
COLOR_BACKGROUND = (255, 255, 255)
COLOR_TEXT = (0, 0, 0)
COLOR_PTS_FRAME = (214, 39, 40)
COLOR_CFR_FRAME = (31, 119, 180)
COLOR_BOTH = (148, 103, 189)


def frames_in_window(frame_times: np.ndarray, time_s: float) -> np.ndarray:
    relative = frame_times - frame_times[0]
    return np.flatnonzero(np.abs(relative - time_s) <= REVIEW_CONTEXT_S + 1e-9)


def decode_scaled_frames(
    path: Path, indices: list[int], width: int, height: int
) -> dict[int, np.ndarray]:
    """Decodifica só a faixa de vídeo, sem áudio, até o maior índice pedido."""
    wanted = set(indices)
    frame_size = width * height * 3
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-fps_mode",
            "passthrough",
            "-vf",
            f"scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    decoded: dict[int, np.ndarray] = {}
    try:
        for index in range(max(indices) + 1):
            chunk = process.stdout.read(frame_size)
            if len(chunk) < frame_size:
                raise EOFError(f"{path}: vídeo esgotado no quadro {index}")
            if index in wanted:
                decoded[index] = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()
    return decoded


def tile_size(path: Path) -> tuple[int, int]:
    streams = cast(list[dict[str, object]], read_ffprobe_header(path)["streams"])
    width, height = int(str(streams[0]["width"])), int(str(streams[0]["height"]))
    tile_height = round(TILE_WIDTH * height / width / 2) * 2
    return TILE_WIDTH, tile_height


# A fonte padrão do PIL não tem acentos nem setas; o texto da imagem fica em ASCII.
def header_lines(case: dict[str, str]) -> list[str]:
    return [
        "EVIDENCIA AUTOMATICA - nao confirma a anotacao; o julgamento vai em review.csv",
        f"{case['case_id']}",
        f"{case['video_id']}  sha256 {case['source_sha256'][:16]}  fps {case['fps']}",
        f"{case['boundary']} anotado em {float(case['annotated_s']):.3f} s (OmniFall): "
        f"{case['label_before']} -> {case['label_after']}",
        f"vermelho: primeiro quadro com PTS >= anotado (#{case['pts_frame_index']}); "
        f"azul: quadro da grade CFR (#{case['cfr_frame_index']}, "
        f"{float(case['cfr_offset_s']):+.3f} s); roxo: ambos",
    ]


def render_contact_sheet(
    path: Path, case: dict[str, str], segments: pd.DataFrame, output: Path
) -> int:
    """Grava a contact sheet de um caso e devolve quantos quadros ela mostra."""
    frame_times = read_frame_times(path)
    relative = frame_times - frame_times[0]
    time_s = float(case["annotated_s"])
    indices = [int(index) for index in frames_in_window(frame_times, time_s)]
    width, height = tile_size(path)
    frames = decode_scaled_frames(path, indices, width, height)
    font = ImageFont.load_default(size=FONT_SIZE)

    lines = header_lines(case)
    rows = -(-len(indices) // TILE_COLUMNS)
    tile_w = width + 2 * BORDER
    tile_h = height + 2 * BORDER + TILE_CAPTION_LINES * LINE_HEIGHT
    top = len(lines) * LINE_HEIGHT + BORDER
    canvas = Image.new("RGB", (TILE_COLUMNS * tile_w, top + rows * tile_h), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for line_index, line in enumerate(lines):
        draw.text((BORDER, BORDER + line_index * LINE_HEIGHT), line, fill=COLOR_TEXT, font=font)

    pts_index = int(case["pts_frame_index"])
    cfr_index = int(case["cfr_frame_index"])
    for position, index in enumerate(indices):
        x = (position % TILE_COLUMNS) * tile_w
        y = top + (position // TILE_COLUMNS) * tile_h
        color = {
            (True, True): COLOR_BOTH,
            (True, False): COLOR_PTS_FRAME,
            (False, True): COLOR_CFR_FRAME,
        }.get((index == pts_index, index == cfr_index))
        if color is not None:
            draw.rectangle((x, y, x + tile_w - 1, y + height + 2 * BORDER - 1), fill=color)
        canvas.paste(Image.fromarray(frames[index]), (x + BORDER, y + BORDER))
        pts = float(relative[index])
        caption = [
            f"#{index} {pts:.3f} s ({pts - time_s:+.3f})",
            labels_at(segments, pts),
        ]
        for line_index, line in enumerate(caption):
            draw.text(
                (x + BORDER, y + height + 2 * BORDER + line_index * LINE_HEIGHT),
                line,
                fill=COLOR_TEXT,
                font=font,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    canvas.save(temporary)
    temporary.replace(output)
    return len(indices)


def render_review(
    root: Path,
    review_dir: Path,
    sheet: pd.DataFrame,
    videos: pd.DataFrame,
    omnifall: pd.DataFrame,
) -> list[tuple[str, int]]:
    """Gera uma contact sheet por caso. `omnifall` usa nomes de classe."""
    ensure_ignored_output(review_dir)
    paths = cast(pd.Series, videos.set_index("video_id")["relative_path"])
    by_video = {str(key): group for key, group in omnifall.groupby("video_id")}
    rendered: list[tuple[str, int]] = []
    for case in cast(list[dict[str, str]], sheet.to_dict("records")):
        output = contact_sheet_path(review_dir, case["case_id"])
        segments = by_video.get(case["video_id"], empty_segments(omnifall))
        count = render_contact_sheet(root / str(paths[case["video_id"]]), case, segments, output)
        rendered.append((case["case_id"], count))
    return rendered
