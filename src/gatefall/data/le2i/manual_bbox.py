"""Leitura das caixas manuais da distribuição original do Le2i."""

import math
import re
from dataclasses import dataclass
from pathlib import Path

from gatefall.data.le2i.path_matching import normalize_environment_name

_VIDEO_FILE = re.compile(r"video \((\d+)\)\.txt", re.IGNORECASE)
_INTEGER = re.compile(r"[+-]?\d+")


@dataclass(frozen=True)
class ManualBox:
    frame: int
    auxiliary: int
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class ManualAnnotations:
    boxes: tuple[ManualBox, ...]
    metadata_lines: int
    zero_rows: int
    negative_rows: int


def discover_annotation_files(raw_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not raw_root.is_dir():
        raise FileNotFoundError(f"diretório Le2i não encontrado: {raw_root}")
    for path in sorted(raw_root.glob("*/Annotation_files/*.txt")):
        match = _VIDEO_FILE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"nome de anotação manual inválido: {path}")
        video_id = f"{normalize_environment_name(path.parent.parent.name)}/video_{int(match.group(1))}"
        if video_id in files:
            raise ValueError(
                f"anotações manuais duplicadas para {video_id}: {files[video_id]}, {path}"
            )
        files[video_id] = path
    return files


def parse_annotation_file(path: Path, *, n_frames: int) -> ManualAnnotations:
    boxes: list[ManualBox] = []
    metadata_lines = 0
    zero_rows = 0
    negative_rows = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        if _INTEGER.fullmatch(stripped):
            metadata_lines += 1
            continue
        fields = [field.strip() for field in stripped.split(",")]
        if len(fields) != 6 or any(
            _INTEGER.fullmatch(field) is None for field in fields
        ):
            raise ValueError(f"estrutura inválida em {path}:{line_number}: {line!r}")
        values = [int(field) for field in fields]
        frame, auxiliary, x1, y1, x2, y2 = values
        if frame < 1 or frame > n_frames:
            raise ValueError(
                f"quadro manual fora de 1..{n_frames} em {path}:{line_number}: {frame}"
            )
        coordinates = (x1, y1, x2, y2)
        if any(value < 0 for value in coordinates):
            negative_rows += 1
        elif coordinates == (0, 0, 0, 0):
            zero_rows += 1
        elif x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"caixa manual positiva impossível em {path}:{line_number}: {coordinates}"
            )
        else:
            try:
                finite = all(math.isfinite(float(value)) for value in coordinates)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(
                    f"coordenada manual não finita em {path}:{line_number}: {coordinates}"
                )
            boxes.append(ManualBox(frame, auxiliary, coordinates))
    return ManualAnnotations(tuple(boxes), metadata_lines, zero_rows, negative_rows)
