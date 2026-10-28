"""Sequências ordenadas de imagens RGB com carimbo de tempo no nome (UP-Fall)."""

import hashlib
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from gatefall.hashing import sha256_file

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
# Nome de quadro do UP-Fall: `2018-07-04T12_04_17.738369.png` (hora com `_`).
TIMESTAMP_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"
    r"(?:\.(?P<fraction>\d{1,6}))?"
)
ARCHIVE_PATTERN = re.compile(
    r"Subject(?P<subject>\d+)Activity(?P<activity>\d+)Trial(?P<trial>\d+)Camera(?P<camera>\d+)\.zip"
)


class SequenceError(ValueError):
    """Sequência de imagens ausente, sem carimbo de tempo verificável ou fora de ordem."""


@dataclass(frozen=True)
class ImageSequence:
    video_id: str
    directory: Path
    filenames: tuple[str, ...]
    timestamps_s: np.ndarray
    origin_iso: str

    @property
    def n_frames(self) -> int:
        return len(self.filenames)


def parse_frame_timestamp(stem: str) -> datetime:
    match = TIMESTAMP_PATTERN.fullmatch(stem)
    if match is None:
        raise SequenceError(f"nome de quadro sem carimbo de tempo verificável: {stem!r}")
    fraction = (match["fraction"] or "0").ljust(6, "0")
    return datetime.fromisoformat(
        f"{match['date']}T{match['hour']}:{match['minute']}:{match['second']}.{fraction}"
    )


def read_image_sequence(directory: Path, video_id: str) -> ImageSequence:
    if not directory.is_dir():
        raise SequenceError(f"{video_id}: diretório de quadros ausente: {directory}")
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) < 2:
        raise SequenceError(f"{video_id}: menos de dois quadros em {directory}")
    stamped = sorted((parse_frame_timestamp(path.stem), path.name) for path in images)
    instants = [instant for instant, _ in stamped]
    if len(set(instants)) != len(instants):
        raise SequenceError(f"{video_id}: carimbos de tempo duplicados em {directory}")
    origin = instants[0]
    timestamps = np.array(
        [(instant - origin).total_seconds() for instant in instants], dtype=np.float64
    )
    return ImageSequence(
        video_id=video_id,
        directory=directory,
        filenames=tuple(name for _, name in stamped),
        timestamps_s=timestamps,
        origin_iso=origin.isoformat(),
    )


def source_fps(sequence: ImageSequence) -> float:
    span = float(sequence.timestamps_s[-1])
    if span <= 0:
        raise SequenceError(f"{sequence.video_id}: duração nula entre o primeiro e o último quadro")
    return (sequence.n_frames - 1) / span


def sequence_duration_s(sequence: ImageSequence) -> float:
    # O último quadro dura um intervalo típico, como em n_frames / fps nos vídeos.
    return float(sequence.timestamps_s[-1] + np.median(np.diff(sequence.timestamps_s)))


def image_size(sequence: ImageSequence) -> tuple[int, int]:
    with Image.open(sequence.directory / sequence.filenames[0]) as image:
        return image.size


def decode_sequence_frames(sequence: ImageSequence, src_indices: Sequence[int]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for index in src_indices:
        if not 0 <= index < sequence.n_frames:
            raise SequenceError(
                f"{sequence.video_id}: src_index {index} fora de [0, {sequence.n_frames})"
            )
        with Image.open(sequence.directory / sequence.filenames[index]) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
    return frames


def sequence_sha256(sequence: ImageSequence) -> str:
    digest = hashlib.sha256()
    for name in sequence.filenames:
        digest.update(name.encode())
        digest.update(sha256_file(sequence.directory / name).encode())
    return digest.hexdigest()


def stage_upfall_archive(archive: Path, raw_dir: Path) -> Path:
    """Extrai `SubjectSActivityATrialTCameraC.zip` no layout canônico, sem sobrescrever."""
    match = ARCHIVE_PATTERN.fullmatch(archive.name)
    if match is None:
        raise SequenceError(f"nome de arquivo UP-Fall fora do padrão: {archive.name}")
    s, a, t, c = (match[name] for name in ("subject", "activity", "trial", "camera"))
    target = raw_dir / f"Subject{s}" / f"Activity{a}" / f"Trial{t}" / f"Subject{s}Activity{a}Trial{t}Camera{c}"
    if target.exists():
        raise FileExistsError(f"destino já existe; não sobrescrevo: {target}")
    with zipfile.ZipFile(archive) as handle:
        members = [info for info in handle.infolist() if not info.is_dir()]
        names = [Path(info.filename).name for info in members]
        if len(set(names)) != len(names):
            raise SequenceError(f"{archive}: nomes de quadro repetidos em subpastas")
        if not any(name.lower().endswith(IMAGE_SUFFIXES) for name in names):
            raise SequenceError(f"{archive}: nenhum quadro de imagem")
        staging = target.with_name(target.name + ".partial")
        staging.mkdir(parents=True, exist_ok=False)
        for info, name in zip(members, names):
            if not name.lower().endswith(IMAGE_SUFFIXES):
                continue
            # Só o nome-base é usado: caminhos do zip nunca escapam do destino.
            (staging / name).write_bytes(handle.read(info))
        staging.rename(target)
    return target
