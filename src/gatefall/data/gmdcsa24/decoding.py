"""Decodificação dos MP4 do GMDCSA24 sem duplicar nem descartar quadros.

Os MP4 do GMDCSA24 têm taxa variável. Sem `-fps_mode passthrough`, o pipe do
`ffmpeg` (como em `gatefall.data.video_io`, feito para os AVI de taxa
constante do Le2i) duplica quadros até a taxa nominal: em `Subject_1/Fall/12`
ele entrega 206 quadros para um vídeo de 200. Com passthrough, o quadro `i`
do pipe é o `i`-ésimo quadro decodificado, o mesmo que `ffprobe
-count_frames` conta, e o `src_index` da grade aponta para ele.
"""

import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from gatefall.data.gmdcsa24.review_render import decode_scaled_frames


def open_passthrough_pipe(path: Path) -> subprocess.Popen[bytes]:
    # stderr descartado: com passthrough, o muxer rawvideo avisa de DTS não
    # crescente nos quadros de PTS irregular, sem descartar nenhum quadro.
    return subprocess.Popen(
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
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def count_decoded_frames(path: Path, width: int, height: int) -> int:
    """Quadros entregues pelo decodificador do adapter, lendo o vídeo inteiro.

    Falha se o `ffmpeg` terminar com erro: uma contagem parcial passaria por
    contagem real na checagem V5."""
    frame_size = width * height * 3
    process = open_passthrough_pipe(path)
    assert process.stdout is not None
    count = 0
    try:
        while len(process.stdout.read(frame_size)) == frame_size:
            count += 1
    finally:
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        raise OSError(f"{path}: ffmpeg terminou com código {return_code} ao contar quadros")
    return count


def decode_frames(
    path: Path, src_indices: Sequence[int], width: int, height: int
) -> list[np.ndarray]:
    """Quadros RGB nativos nos índices `src_indices` da grade, na ordem pedida."""
    if not src_indices:
        return []
    decoded = decode_scaled_frames(path, list(src_indices), width, height)
    return [decoded[index] for index in src_indices]


class GridFrameReader:
    """Lê o vídeo inteiro pelo pipe passthrough e entrega só os quadros da grade.

    `src_indices` precisa ser estritamente crescente. Depois de esgotada a
    iteração, `decoded_frames` é o total de quadros que o decodificador
    entregou, para conferir contra a contagem do adapter: um pipe que
    duplicasse ou descartasse quadros mudaria esse total."""

    def __init__(self, path: Path, src_indices: Sequence[int], width: int, height: int) -> None:
        if any(later <= earlier for earlier, later in zip(src_indices, src_indices[1:])):
            raise ValueError(f"{path}: src_index da grade não é estritamente crescente")
        self.path = path
        self.src_indices = list(src_indices)
        self.width = width
        self.height = height
        self.decoded_frames: int | None = None

    def __iter__(self) -> Iterator[np.ndarray]:
        frame_size = self.width * self.height * 3
        process = open_passthrough_pipe(self.path)
        assert process.stdout is not None
        decoded = 0
        position = 0
        try:
            while True:
                chunk = process.stdout.read(frame_size)
                if len(chunk) < frame_size:
                    break
                if position < len(self.src_indices) and decoded == self.src_indices[position]:
                    position += 1
                    yield np.frombuffer(chunk, dtype=np.uint8).reshape(
                        self.height, self.width, 3
                    ).copy()
                decoded += 1
        finally:
            process.stdout.close()
            return_code = process.wait()
        if return_code != 0:
            raise OSError(f"{self.path}: ffmpeg terminou com código {return_code}")
        self.decoded_frames = decoded
