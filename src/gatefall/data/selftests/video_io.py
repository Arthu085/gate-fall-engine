"""Selftest sintético da decodificação de vídeo (`video_io.py`).

Não toca no dataset real — todo o vídeo usado aqui é gerado em memória, num
diretório temporário, para travar o comportamento de `decode_frames` e
`probe_frame_count` contra futuras mudanças.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from gatefall.data.video_io import decode_frames, probe_frame_count

_FRAME_SIZE = (16, 16)
_FRAME_COLORS_RGB: list[tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _write_synthetic_video(directory: Path) -> Path:
    path = directory / "synthetic.avi"
    width, height = _FRAME_SIZE

    raw_frames = b"".join(
        _expected_rgb(index).tobytes()
        for index in range(len(_FRAME_COLORS_RGB))
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "10",
            "-i",
            "-",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            str(path),
        ],
        input=raw_frames,
        check=True,
    )
    return path


def _expected_rgb(color_index: int) -> np.ndarray:
    width, height = _FRAME_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = _FRAME_COLORS_RGB[color_index]
    return frame


def check_probe_frame_count(video_path: Path) -> bool:
    count = probe_frame_count(video_path)
    ok = count == len(_FRAME_COLORS_RGB)
    return _check(
        f"probe_frame_count retorna a contagem exata gravada ({len(_FRAME_COLORS_RGB)})",
        ok,
    )


def check_decode_frames_increasing_indices(video_path: Path) -> bool:
    indices = [0, 1, 2, 3, 4]
    frames = decode_frames(video_path, indices)
    ok = len(frames) == len(indices)
    for position, index in enumerate(indices):
        ok = ok and bool(np.array_equal(frames[position], _expected_rgb(index)))
    return _check(
        "decode_frames com índices crescentes retorna os quadros na ordem "
        "gravada, em RGB",
        ok,
    )


def check_decode_frames_duplicate_indices(video_path: Path) -> bool:
    indices = [0, 0, 2]
    frames = decode_frames(video_path, indices)
    ok = len(frames) == 3
    ok = ok and bool(np.array_equal(frames[0], _expected_rgb(0)))
    ok = ok and bool(np.array_equal(frames[1], _expected_rgb(0)))
    ok = ok and bool(np.array_equal(frames[2], _expected_rgb(2)))
    return _check(
        "decode_frames com índices duplicados [0, 0, 2] retorna 3 quadros, "
        "os dois primeiros iguais ao conteúdo do quadro 0",
        ok,
    )


def check_decode_frames_non_monotonic_indices(video_path: Path) -> bool:
    indices = [2, 0, 1]
    frames = decode_frames(video_path, indices)
    ok = len(frames) == len(indices)
    for position, index in enumerate(indices):
        ok = ok and bool(np.array_equal(frames[position], _expected_rgb(index)))
    return _check(
        "decode_frames com índices não monotônicos [2, 0, 1] devolve cada "
        "quadro na ordem solicitada, casando com o conteúdo do índice pedido",
        ok,
    )


def check_decode_frames_out_of_bounds_raises(video_path: Path) -> bool:
    out_of_bounds = len(_FRAME_COLORS_RGB)
    raised = False
    try:
        decode_frames(video_path, [out_of_bounds])
    except EOFError:
        raised = True
    return _check(
        "decode_frames levanta EOFError para um índice >= contagem de "
        "quadros gravados",
        raised,
    )


def check_decode_frames_empty_returns_empty() -> bool:
    ok = decode_frames(Path("/nao/existe.avi"), []) == []
    return _check(
        "decode_frames([]) retorna [] sem abrir o pipe do ffmpeg",
        ok,
    )


def run_video_io_selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = _write_synthetic_video(Path(tmp_dir))
        checks = [
            check_probe_frame_count(video_path),
            check_decode_frames_increasing_indices(video_path),
            check_decode_frames_duplicate_indices(video_path),
            check_decode_frames_non_monotonic_indices(video_path),
            check_decode_frames_out_of_bounds_raises(video_path),
            check_decode_frames_empty_returns_empty(),
        ]
    if not all(checks):
        print("\nselftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: todos os casos passaram")
