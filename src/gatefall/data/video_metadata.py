"""Sondagem de metadados de vídeo com ffprobe."""

import json
import subprocess
from pathlib import Path
from typing import TypedDict


class VideoMetadata(TypedDict):
    r_frame_rate: str
    avg_frame_rate: str
    n_frames_header: int | None
    n_frames_counted: int
    duration_s: float
    width: int
    height: int
    codec: str


def read_ffprobe_header(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,codec_name:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def read_ffprobe_frame_count(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def count_video_frames(path: Path) -> int:
    output = read_ffprobe_frame_count(path)
    try:
        return int(output)
    except ValueError as exc:
        raise ValueError(
            f"ffprobe não retornou um inteiro parseável para {path}: {output!r}"
        ) from exc


def resolve_frame_rate(
    r_frame_rate: str, avg_frame_rate: str
) -> tuple[float, str]:
    """Resolve o FPS preferindo a média do stream e usando a taxa nominal como fallback."""

    def parse_fraction(value: str) -> float | None:
        numerator_text, _, denominator_text = value.partition("/")
        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text) if denominator_text else 1.0
        except ValueError:
            return None
        if denominator == 0:
            return None
        return numerator / denominator

    average = parse_fraction(avg_frame_rate)
    if average is not None and average > 0:
        return average, "avg_frame_rate"

    nominal = parse_fraction(r_frame_rate)
    if nominal is None:
        raise ValueError(
            f"não foi possível calcular fps de r_frame_rate={r_frame_rate!r} "
            f"nem avg_frame_rate={avg_frame_rate!r}"
        )
    return nominal, "r_frame_rate"


def probe_video(path: Path) -> VideoMetadata:
    header = read_ffprobe_header(path)

    raw_streams = header.get("streams")
    stream: dict[str, object] = {}
    if isinstance(raw_streams, list) and raw_streams and isinstance(raw_streams[0], dict):
        stream = raw_streams[0]

    raw_format = header.get("format")
    video_format: dict[str, object] = (
        raw_format if isinstance(raw_format, dict) else {}
    )

    r_frame_rate = str(stream.get("r_frame_rate", "0/0"))
    avg_frame_rate = str(stream.get("avg_frame_rate", "0/0"))

    try:
        n_frames_header = int(str(stream.get("nb_frames")))
    except (TypeError, ValueError):
        n_frames_header = None

    return VideoMetadata(
        r_frame_rate=r_frame_rate,
        avg_frame_rate=avg_frame_rate,
        n_frames_header=n_frames_header,
        n_frames_counted=count_video_frames(path),
        duration_s=float(str(video_format.get("duration", "nan"))),
        width=int(str(stream.get("width", 0))),
        height=int(str(stream.get("height", 0))),
        codec=str(stream.get("codec_name", "")),
    )
