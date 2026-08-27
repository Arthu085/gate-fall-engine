"""Decodificação de vídeo do Le2i: resolução de path, amostra de relatório e dump manual."""

import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.data.frames import read_frames
from gatefall.data.le2i.frames import FRAMES_PATH
from gatefall.data.le2i.verification import load_le2i_manifest
from gatefall.data.video_io import decode_frames, probe_frame_count


def load_le2i_video_paths() -> dict[str, Path]:
    manifest = load_le2i_manifest()
    return {
        str(video_id): Path(str(absolute_path))
        for video_id, absolute_path in zip(
            manifest["video_id"], manifest["absolute_path"]
        )
    }


def select_le2i_report_sample(
    manifest: pd.DataFrame, videos_per_env: int = 2
) -> list[str]:
    longest_id = str(
        manifest.loc[manifest["n_frames_counted"].idxmax(), "video_id"]
    )
    shortest_id = str(
        manifest.loc[manifest["n_frames_counted"].idxmin(), "video_id"]
    )

    home_candidates = cast(
        pd.DataFrame,
        manifest[
            manifest["env"].isin(["Home_01", "Home_02"])
            & (manifest["width"] == 320)
            & (manifest["height"] == 180)
        ],
    ).sort_values("video_id")
    if home_candidates.empty:
        print(
            "erro: nenhum vídeo em Home_01/Home_02 com resolução 320x180 "
            "encontrado no manifesto",
            file=sys.stderr,
        )
        sys.exit(1)
    home_320_id = str(home_candidates.iloc[0]["video_id"])

    forced_by_id = {longest_id, shortest_id, home_320_id}

    selected: list[str] = []
    for environment in sorted(cast(pd.Series, manifest["env"]).unique()):
        env_rows = cast(
            pd.DataFrame, manifest[manifest["env"] == environment]
        ).sort_values("video_id")
        env_video_ids = list(env_rows["video_id"].astype(str))

        env_forced = [
            video_id for video_id in env_video_ids if video_id in forced_by_id
        ]
        picked = list(env_forced)
        for video_id in env_video_ids:
            if len(picked) >= videos_per_env:
                break
            if video_id not in picked:
                picked.append(video_id)
        selected.extend(picked[:videos_per_env])

    n_envs = cast(pd.Series, manifest["env"]).nunique()
    assert len(selected) == videos_per_env * n_envs
    assert len(set(selected)) == len(selected)
    return selected


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_decoded_frame_count(video_id: str, n_decoded: int, n_requested: int) -> bool:
    return _check(
        f"{video_id}: decode_frames retorna a mesma contagem solicitada "
        f"({n_requested})",
        n_decoded == n_requested,
    )


def check_decoded_resolution(
    video_id: str, decoded_shape: tuple[int, int], manifest_shape: tuple[int, int]
) -> bool:
    return _check(
        f"{video_id}: resolução do quadro decodificado {decoded_shape} bate "
        f"com o manifesto {manifest_shape}",
        decoded_shape == manifest_shape,
    )


def check_max_src_index_in_bounds(
    video_id: str, max_src_index: int, probed_count: int
) -> bool:
    return _check(
        f"{video_id}: max(src_index)={max_src_index} < contagem decodificada "
        f"({probed_count})",
        max_src_index < probed_count,
    )


def check_all_src_indices_decode(video_id: str, decoded_ok: bool) -> bool:
    return _check(
        f"{video_id}: decode_frames decodifica todos os src_index do vídeo "
        "sem levantar exceção",
        decoded_ok,
    )


def report_le2i_frame_decode() -> None:
    manifest = load_le2i_manifest()
    if not FRAMES_PATH.exists():
        print(
            f"\nframes report FALHOU: {FRAMES_PATH} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)
    frames = read_frames(FRAMES_PATH)

    sample_video_ids = select_le2i_report_sample(manifest)
    video_paths = load_le2i_video_paths()
    manifest_by_video_id = manifest.set_index("video_id")

    checks: list[bool] = []
    for video_id in sample_video_ids:
        video_path = video_paths[video_id]
        video_frames = cast(
            pd.DataFrame, frames[frames["video_id"] == video_id]
        )
        src_indices = [int(index) for index in video_frames["src_index"]]
        max_src_index = max(src_indices)

        probed_count = probe_frame_count(video_path)
        checks.append(
            check_max_src_index_in_bounds(video_id, max_src_index, probed_count)
        )

        first_frame = decode_frames(video_path, [0])[0]
        decoded_shape = (int(first_frame.shape[0]), int(first_frame.shape[1]))
        manifest_row = manifest_by_video_id.loc[video_id]
        manifest_shape = (
            int(cast(int, manifest_row["height"])),
            int(cast(int, manifest_row["width"])),
        )
        checks.append(
            check_decoded_resolution(video_id, decoded_shape, manifest_shape)
        )

        try:
            decoded_frames = decode_frames(video_path, src_indices)
            decoded_ok = True
        except (OSError, EOFError):
            decoded_frames = []
            decoded_ok = False
        checks.append(check_all_src_indices_decode(video_id, decoded_ok))
        checks.append(
            check_decoded_frame_count(video_id, len(decoded_frames), len(src_indices))
        )

    if not all(checks):
        print("\nframes report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nframes report OK: todas as checagens passaram")


def dump_le2i_frame_pngs(
    video_id: str,
    frame_index_start: int,
    frame_index_end: int,
    output_dir: Path = Path("data/scratch"),
) -> list[Path]:
    frames = read_frames(FRAMES_PATH)
    video_frames = cast(
        pd.DataFrame,
        frames[
            (frames["video_id"] == video_id)
            & (frames["frame_index"] >= frame_index_start)
            & (frames["frame_index"] < frame_index_end)
        ],
    ).sort_values("frame_index")

    video_path = load_le2i_video_paths()[video_id]
    frame_indices = [int(index) for index in video_frames["frame_index"]]
    src_indices = [int(index) for index in video_frames["src_index"]]
    decoded = decode_frames(video_path, src_indices)

    video_output_dir = output_dir / video_id.replace("/", "_")
    video_output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for frame_index, src_index, frame_rgb in zip(
        frame_indices, src_indices, decoded
    ):
        path = (
            video_output_dir
            / f"frame_{frame_index:05d}_src_{src_index:05d}.png"
        )
        _write_png(frame_rgb, path)
        written_paths.append(path)

    return written_paths


def _write_png(frame_rgb: np.ndarray, path: Path) -> None:
    height, width = frame_rgb.shape[0], frame_rgb.shape[1]
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
            "-i",
            "-",
            str(path),
        ],
        input=frame_rgb.tobytes(),
        check=True,
    )
