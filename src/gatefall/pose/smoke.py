"""Smoke test de pose (YOLO-Pose + ByteTrack) sobre um único vídeo do Le2i."""

import argparse
import sys
from collections import Counter
from typing import cast

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from gatefall.data.frames import read_frames
from gatefall.data.le2i.frames import FRAMES_PATH
from gatefall.data.le2i.video_io import load_le2i_video_paths
from gatefall.data.video_io import decode_frames

DEFAULT_VIDEO_ID = "coffee_room_01/video_1"
DEFAULT_MODEL = "yolo26n-pose.pt"


def _select_src_indices(video_id: str) -> list[int]:
    if not FRAMES_PATH.exists():
        print(
            f"\npose smoke test FALHOU: {FRAMES_PATH} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = read_frames(FRAMES_PATH)
    video_frames = cast(
        pd.DataFrame, frames[frames["video_id"] == video_id]
    ).sort_values("frame_index")
    if video_frames.empty:
        print(
            f"\npose smoke test FALHOU: video_id '{video_id}' não encontrado "
            f"em {FRAMES_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    return [int(x) for x in video_frames["src_index"]]


def _to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    # decode_frames devolve RGB (rgb24 via ffmpeg); Ultralytics espera BGR
    # ao receber um ndarray bruto como fonte, daí a inversão de canais.
    return frame_rgb[:, :, ::-1]


def run_pose_smoke_test(video_id: str, model_name: str) -> None:
    src_indices = _select_src_indices(video_id)
    k_expected = len(src_indices)

    video_paths = load_le2i_video_paths()
    if video_id not in video_paths:
        print(
            f"\npose smoke test FALHOU: video_id '{video_id}' não encontrado "
            "no manifesto do Le2i",
            file=sys.stderr,
        )
        sys.exit(1)

    frames_rgb = decode_frames(video_paths[video_id], src_indices)
    assert len(frames_rgb) == k_expected

    model = YOLO(model_name)

    detections_per_frame: list[int] = []
    track_frame_counts: Counter[int] = Counter()
    per_frame_cache: list[tuple[list[int], np.ndarray | None]] = []

    for frame_rgb in frames_rgb:
        frame_bgr = _to_bgr(frame_rgb)
        results = model.track(
            frame_bgr, persist=True, tracker="bytetrack.yaml", verbose=False
        )
        result = results[0]

        n_det = len(result.boxes) if result.boxes is not None else 0
        detections_per_frame.append(n_det)

        ids = result.boxes.id if result.boxes is not None else None
        frame_track_ids = [int(t) for t in ids.tolist()] if ids is not None else []
        for track_id in frame_track_ids:
            track_frame_counts[track_id] += 1

        kp_conf = (
            cast(torch.Tensor, result.keypoints.conf).cpu().numpy()
            if (result.keypoints is not None and result.keypoints.conf is not None)
            else None
        )
        per_frame_cache.append((frame_track_ids, kp_conf))

    print(f"\nquadros processados: {len(frames_rgb)} (esperado do grid: {k_expected})")

    print(f"\ntracks distintas: {len(track_frame_counts)}")
    for track_id in sorted(track_frame_counts, key=lambda t: (-track_frame_counts[t], t)):
        count = track_frame_counts[track_id]
        print(
            f"  track {track_id}: {count} quadros "
            f"({100.0 * count / k_expected:.1f}%)"
        )

    histogram: Counter[int | str] = Counter()
    for n_det in detections_per_frame:
        bucket: int | str = n_det if n_det < 3 else "3+"
        histogram[bucket] += 1
    print("\nhistograma de detecções por quadro:")
    for bucket in (0, 1, 2, "3+"):
        print(f"  {bucket}: {histogram.get(bucket, 0)}")

    zero_fraction = histogram.get(0, 0) / k_expected
    print(f"\nfração de quadros sem detecção: {zero_fraction:.3f}")

    longest_zero_run = 0
    current_run = 0
    for n_det in detections_per_frame:
        if n_det == 0:
            current_run += 1
            longest_zero_run = max(longest_zero_run, current_run)
        else:
            current_run = 0
    print(f"maior sequência consecutiva de quadros sem detecção: {longest_zero_run}")

    most_persistent = (
        min(track_frame_counts, key=lambda t: (-track_frame_counts[t], t))
        if track_frame_counts
        else None
    )

    print("\nconfiança de keypoints da track mais persistente:")
    if most_persistent is None:
        print("  nenhuma track detectada")
        return

    values: list[float] = []
    for frame_track_ids, kp_conf in per_frame_cache:
        if kp_conf is None or most_persistent not in frame_track_ids:
            continue
        values.extend(kp_conf[frame_track_ids.index(most_persistent)].tolist())

    if not values:
        print("  nenhuma track detectada")
        return

    print(f"  track {most_persistent}: média={np.mean(values):.3f}, p10={np.percentile(values, 10):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report",
        help="Roda YOLO-Pose + ByteTrack sobre um vídeo do Le2i e imprime estatísticas",
    )
    report_parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    report_parser.add_argument("--model", default=DEFAULT_MODEL)

    args = parser.parse_args()
    if args.command == "report":
        run_pose_smoke_test(args.video_id, args.model)


if __name__ == "__main__":
    main()
