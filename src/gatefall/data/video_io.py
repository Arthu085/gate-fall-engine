"""Decodificação de vídeo agnóstica de dataset, sem seek.

Decodifica via pipe do `ffmpeg` do sistema (já uma dependência do pipeline —
`ffprobe`, do mesmo pacote, é exigido em `video_metadata.py`), em vez de
`cv2.VideoCapture`: o backend FFmpeg embutido no wheel do OpenCV trava
(SIGSEGV) ao ler AVIs `rawvideo` do Le2i cujo stream de áudio MP3 tem cabeçalho
corrompido — o `ffmpeg` do sistema decodifica o stream de vídeo normalmente e
apenas registra o erro de áudio, sem travar.

Sempre varre sequencialmente a partir do quadro 0 lendo o pipe, nunca via
seek — `ffmpeg -ss` antes de um AVI `rawvideo` ainda depende do mesmo
demuxer, e a varredura sequencial de rawvideo já é rápida o bastante.
"""

import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np


def _probe_video_dimensions(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if not streams:
        raise OSError(f"nenhum stream de vídeo encontrado em {video_path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def _open_rawvideo_pipe(video_path: Path) -> subprocess.Popen[bytes]:
    # stderr descartado: quando paramos de ler antes do ffmpeg esgotar o
    # vídeo (índice máximo já coberto), o pipe fecha e o ffmpeg reporta
    # "Broken pipe" no stream de saída — ruído esperado da própria
    # terminação antecipada, não um erro de decodificação.
    return subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def decode_frames(
    video_path: Path, src_indices: Sequence[int]
) -> list[np.ndarray]:
    if not src_indices:
        return []

    wanted = set(src_indices)
    max_wanted = max(src_indices)
    width, height = _probe_video_dimensions(video_path)
    frame_size = width * height * 3

    process = _open_rawvideo_pipe(video_path)
    assert process.stdout is not None
    try:
        decoded: dict[int, np.ndarray] = {}
        running_index = 0
        while running_index <= max_wanted:
            chunk = process.stdout.read(frame_size)
            if len(chunk) < frame_size:
                raise EOFError(
                    f"{video_path}: vídeo esgotado no quadro {running_index}, "
                    f"antes de alcançar o índice máximo solicitado {max_wanted}"
                )
            if running_index in wanted:
                decoded[running_index] = np.frombuffer(
                    chunk, dtype=np.uint8
                ).reshape(height, width, 3).copy()
            running_index += 1
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()

    return [decoded[index] for index in src_indices]


def probe_frame_count(video_path: Path) -> int:
    width, height = _probe_video_dimensions(video_path)
    frame_size = width * height * 3

    process = _open_rawvideo_pipe(video_path)
    assert process.stdout is not None
    try:
        count = 0
        while True:
            chunk = process.stdout.read(frame_size)
            if len(chunk) < frame_size:
                break
            count += 1
    finally:
        process.stdout.close()
        process.wait()

    return count
