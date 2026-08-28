"""Extração de pose (YOLO-Pose + ByteTrack) de vídeos do Le2i para HDF5."""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

from gatefall.config import TARGET_FPS
from gatefall.data.frames import read_frames
from gatefall.data.le2i.frames import FRAMES_PATH
from gatefall.data.le2i.verification import load_le2i_manifest
from gatefall.data.le2i.video_io import load_le2i_video_paths
from gatefall.data.video_io import decode_frames
from gatefall.pose.selection import select_person_index
from gatefall.pose.smoke import DEFAULT_MODEL

POSE_ROOT = Path("data/features/le2i/pose")
TRACKER_NAME = "bytetrack.yaml"
WEIGHTS_DIR = Path("data/scratch/weights")

N_KEYPOINTS = 17


class PoseExtractSkipped(Exception):
    """.h5 do vídeo já existe e --force não foi passado."""


class PoseExtractError(Exception):
    """Falha ao processar um vídeo (dados ausentes, video_id não encontrado, verificação pós-escrita divergente)."""


@dataclass(frozen=True)
class PoseExtractResult:
    video_id: str
    env: str
    split: str
    k: int
    n_found: int


def _resolve_model_path(model_name: str) -> str:
    if Path(model_name).parent != Path("."):
        return model_name
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(WEIGHTS_DIR / model_name)


def _to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    # decode_frames devolve RGB (rgb24 via ffmpeg); Ultralytics espera BGR
    # ao receber um ndarray bruto como fonte, daí a inversão de canais.
    return frame_rgb[:, :, ::-1]


def _output_path(video_id: str) -> Path:
    env, _, video_name = video_id.partition("/")
    return POSE_ROOT / env / f"{video_name}.h5"


def _select_src_indices(video_id: str) -> list[int]:
    if not FRAMES_PATH.exists():
        raise PoseExtractError(
            f"\npose extract FALHOU: {FRAMES_PATH} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro"
        )

    frames = read_frames(FRAMES_PATH)
    video_frames = cast(
        pd.DataFrame, frames[frames["video_id"] == video_id]
    ).sort_values("frame_index")
    if video_frames.empty:
        raise PoseExtractError(
            f"\npose extract FALHOU: video_id '{video_id}' não encontrado "
            f"em {FRAMES_PATH}"
        )

    return [int(x) for x in video_frames["src_index"]]


def _reset_tracker_state(model: YOLO) -> None:
    # model.track(..., persist=True) mantém o mesmo BYTETracker durante todo o
    # ciclo de vida de model.predictor (ultralytics.trackers.track.on_predict_start:
    # com persist=True e predictor.trackers já existente, a reinicialização é
    # pulada). Como o mesmo objeto YOLO é reaproveitado entre vídeos, os track_ids
    # vazariam entre eles sem um reset explícito. BYTETracker.reset() (ultralytics
    # 8.4.131, trackers/byte_tracker.py) limpa as faixas ativas/perdidas/removidas
    # e zera o contador global de IDs via BaseTrack.reset_id().
    trackers = getattr(getattr(model, "predictor", None), "trackers", None)
    if not trackers:
        return
    for tracker in trackers:
        tracker.reset()


def run_pose_extract(
    video_id: str, model_name: str, force: bool, *, model: YOLO | None = None
) -> PoseExtractResult:
    output_path = _output_path(video_id)
    if output_path.exists() and not force:
        raise PoseExtractSkipped(
            f"skip {output_path} (já existe, use --force para sobrescrever)"
        )

    src_indices = _select_src_indices(video_id)
    k = len(src_indices)

    manifest = load_le2i_manifest()
    manifest_row = cast(
        pd.DataFrame, manifest[manifest["video_id"] == video_id]
    )
    if manifest_row.empty:
        raise PoseExtractError(
            f"\npose extract FALHOU: video_id '{video_id}' não encontrado "
            "no manifesto do Le2i"
        )
    manifest_row = manifest_row.iloc[0]

    video_paths = load_le2i_video_paths()
    if video_id not in video_paths:
        raise PoseExtractError(
            f"\npose extract FALHOU: video_id '{video_id}' não encontrado "
            "no manifesto do Le2i"
        )

    frames_rgb = decode_frames(video_paths[video_id], src_indices)
    assert len(frames_rgb) == k

    keypoints = np.zeros((k, N_KEYPOINTS, 3), dtype=np.float32)
    bbox = np.zeros((k, 4), dtype=np.float32)
    person_found = np.zeros((k,), dtype=np.bool_)
    n_detections = np.zeros((k,), dtype=np.int16)
    track_id = np.full((k,), -1, dtype=np.int32)

    if model is None:
        model = YOLO(_resolve_model_path(model_name))

    for frame_index, frame_rgb in enumerate(frames_rgb):
        frame_bgr = _to_bgr(frame_rgb)
        results = model.track(
            frame_bgr, persist=True, tracker=TRACKER_NAME, verbose=False
        )
        result = results[0]

        n_det = len(result.boxes) if result.boxes is not None else 0
        n_detections[frame_index] = n_det

        box_conf = (
            cast(torch.Tensor, result.boxes.conf).cpu().numpy()
            if (result.boxes is not None and result.boxes.conf is not None)
            else None
        )
        selected_idx = select_person_index(n_det, box_conf)

        if selected_idx is None:
            continue

        assert result.boxes is not None and result.keypoints is not None
        person_found[frame_index] = True

        box_xyxy = cast(torch.Tensor, result.boxes.xyxy).cpu().numpy()
        bbox[frame_index] = box_xyxy[selected_idx]

        kp_xy = cast(torch.Tensor, result.keypoints.xy).cpu().numpy()
        keypoints[frame_index, :, :2] = kp_xy[selected_idx]
        if result.keypoints.conf is not None:
            kp_conf = cast(torch.Tensor, result.keypoints.conf).cpu().numpy()
            keypoints[frame_index, :, 2] = kp_conf[selected_idx]

        ids = result.boxes.id
        if ids is not None:
            frame_track_ids = [int(t) for t in ids.tolist()]
            track_id[frame_index] = frame_track_ids[selected_idx]

    attrs: dict[str, object] = {
        "video_id": video_id,
        "env": str(manifest_row["env"]),
        "split": str(manifest_row["split"]),
        "subject": int(manifest_row["subject"]),
        "K": k,
        "fps": float(manifest_row["fps"]),
        "width": int(manifest_row["width"]),
        "height": int(manifest_row["height"]),
        "model_name": model_name,
        "ultralytics_version": ultralytics.__version__,
        "tracker_name": TRACKER_NAME,
        "target_fps": TARGET_FPS,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.{os.getpid()}.tmp")
    with h5py.File(tmp_path, "w") as h5_file:
        h5_file.create_dataset("keypoints", data=keypoints)
        h5_file.create_dataset("bbox", data=bbox)
        h5_file.create_dataset("person_found", data=person_found)
        h5_file.create_dataset("n_detections", data=n_detections)
        h5_file.create_dataset("track_id", data=track_id)
        for key, value in attrs.items():
            h5_file.attrs[key] = value
    os.replace(tmp_path, output_path)

    _verify_written_file(
        output_path,
        keypoints=keypoints,
        bbox=bbox,
        person_found=person_found,
        n_detections=n_detections,
        track_id=track_id,
        attrs=attrs,
    )

    n_found = int(person_found.sum())
    print(
        f"\n{video_id}: K={k}, quadros com pessoa encontrada={n_found} "
        f"({100.0 * n_found / k:.1f}%)"
    )

    return PoseExtractResult(
        video_id=video_id,
        env=str(manifest_row["env"]),
        split=str(manifest_row["split"]),
        k=k,
        n_found=n_found,
    )


def _verify_written_file(
    path: Path,
    *,
    keypoints: np.ndarray,
    bbox: np.ndarray,
    person_found: np.ndarray,
    n_detections: np.ndarray,
    track_id: np.ndarray,
    attrs: dict[str, object],
) -> None:
    expected_datasets = {
        "keypoints": keypoints,
        "bbox": bbox,
        "person_found": person_found,
        "n_detections": n_detections,
        "track_id": track_id,
    }
    with h5py.File(path, "r") as h5_file:
        for name, expected in expected_datasets.items():
            actual_dataset = cast(h5py.Dataset, h5_file[name])
            if actual_dataset.shape != expected.shape or actual_dataset.dtype != expected.dtype:
                raise PoseExtractError(
                    f"\npose extract FALHOU: dataset '{name}' relido de {path} "
                    f"diverge do esperado (shape/dtype)"
                )
            if not np.array_equal(actual_dataset[()], expected):
                raise PoseExtractError(
                    f"\npose extract FALHOU: dataset '{name}' relido de {path} "
                    "diverge dos valores gravados"
                )

        for key, expected_value in attrs.items():
            if key not in h5_file.attrs:
                raise PoseExtractError(
                    f"\npose extract FALHOU: atributo '{key}' ausente em {path}"
                )
            actual_value = h5_file.attrs[key]
            if actual_value != expected_value:
                raise PoseExtractError(
                    f"\npose extract FALHOU: atributo '{key}' relido de {path} "
                    f"diverge (esperado {expected_value!r}, encontrado "
                    f"{actual_value!r})"
                )


def _read_existing_stats(video_id: str) -> PoseExtractResult:
    output_path = _output_path(video_id)
    with h5py.File(output_path, "r") as h5_file:
        person_found = cast(h5py.Dataset, h5_file["person_found"])
        return PoseExtractResult(
            video_id=video_id,
            env=str(h5_file.attrs["env"]),
            split=str(h5_file.attrs["split"]),
            k=int(cast(int, h5_file.attrs["K"])),
            n_found=int(person_found[()].sum()),
        )


def _run_extract_cli(video_id: str, model_name: str, force: bool) -> None:
    try:
        run_pose_extract(video_id, model_name, force)
    except PoseExtractSkipped as exc:
        print(str(exc))
        sys.exit(0)
    except PoseExtractError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


def _print_summary(
    per_video: pd.DataFrame,
    processed: int,
    skipped: int,
    failures: list[tuple[str, str]],
    stats: list[PoseExtractResult],
) -> None:
    total_grid_frames = int(cast(pd.Series, per_video["k"]).sum())
    total_found = sum(r.n_found for r in stats)

    print("\nResumo pose extract-all")
    print(f"vídeos processados: {processed}")
    print(f"vídeos pulados (já existiam): {skipped}")
    print(f"vídeos com falha: {len(failures)}")
    print(f"total de quadros da grade: {total_grid_frames}")
    if total_grid_frames > 0:
        print(
            f"total de quadros com pessoa encontrada: {total_found} "
            f"({100.0 * total_found / total_grid_frames:.1f}%)"
        )

    found_by_video = {r.video_id: r.n_found for r in stats}

    print("\nPor ambiente (env):")
    for env, group in per_video.groupby("env"):
        n_videos = len(group)
        n_frames = int(cast(pd.Series, group["k"]).sum())
        n_found = sum(found_by_video.get(str(vid), 0) for vid in group.index)
        pct = 100.0 * n_found / n_frames if n_frames > 0 else 0.0
        print(
            f"  {env}: {n_videos} vídeos, {n_frames} quadros, "
            f"{n_found} com pessoa encontrada ({pct:.1f}%)"
        )

    print("\nPor split:")
    for split, group in per_video.groupby("split"):
        n_videos = len(group)
        n_frames = int(cast(pd.Series, group["k"]).sum())
        n_found = sum(found_by_video.get(str(vid), 0) for vid in group.index)
        pct = 100.0 * n_found / n_frames if n_frames > 0 else 0.0
        print(
            f"  {split}: {n_videos} vídeos, {n_frames} quadros, "
            f"{n_found} com pessoa encontrada ({pct:.1f}%)"
        )

    if failures:
        print("\npose extract-all: falhas por vídeo:", file=sys.stderr)
        for video_id, message in failures:
            print(f"  {video_id}: {message}", file=sys.stderr)


def run_pose_extract_all(model_name: str, force: bool) -> None:
    if not FRAMES_PATH.exists():
        print(
            f"\npose extract-all FALHOU: {FRAMES_PATH} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = read_frames(FRAMES_PATH)
    per_video = cast(
        pd.DataFrame,
        frames.groupby("video_id").agg(
            env=("env", "first"), split=("split", "first"), k=("frame_index", "size")
        ),
    )

    model = YOLO(_resolve_model_path(model_name))

    processed = 0
    skipped = 0
    failures: list[tuple[str, str]] = []
    stats: list[PoseExtractResult] = []

    for video_id in per_video.index:
        video_id = str(video_id)
        _reset_tracker_state(model)
        try:
            result = run_pose_extract(video_id, model_name, force, model=model)
        except PoseExtractSkipped:
            skipped += 1
            existing = _read_existing_stats(video_id)
            stats.append(existing)
            pct = (
                100.0 * existing.n_found / existing.k if existing.k > 0 else 0.0
            )
            print(
                f"skip {video_id}: K={existing.k}, quadros com pessoa "
                f"encontrada={existing.n_found} ({pct:.1f}%) (já existe)"
            )
        except Exception as exc:
            failures.append((video_id, str(exc)))
            print(f"\npose extract-all FALHOU em {video_id}: {exc}", file=sys.stderr)
            continue
        else:
            processed += 1
            stats.append(result)

    _print_summary(per_video, processed, skipped, failures, stats)

    if failures:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Roda YOLO-Pose + ByteTrack sobre um vídeo do Le2i e grava um .h5",
    )
    extract_parser.add_argument("--video-id", required=True)
    extract_parser.add_argument("--model", default=DEFAULT_MODEL)
    extract_parser.add_argument("--force", action="store_true")

    extract_all_parser = subparsers.add_parser(
        "extract-all",
        help="Roda YOLO-Pose + ByteTrack sobre todos os vídeos do Le2i listados em frames.parquet",
    )
    extract_all_parser.add_argument("--model", default=DEFAULT_MODEL)
    extract_all_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "extract":
        _run_extract_cli(args.video_id, args.model, args.force)
    elif args.command == "extract-all":
        run_pose_extract_all(args.model, args.force)


if __name__ == "__main__":
    main()
