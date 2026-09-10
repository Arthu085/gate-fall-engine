"""Confere que decode_frames retorna o quadro correto para o src_index solicitado."""

import sys
from typing import cast

import numpy as np
import torch

from gatefall.data.video_io import decode_frames
from gatefall.datasets import DatasetAdapter
from gatefall.dinov3 import storage
from gatefall.dinov3.backbone import (
    configure_deterministic_inference,
    ensure_backbone_paths_exist,
    load_backbone,
    resolve_repo_dir,
    resolve_weights_path,
)
from gatefall.dinov3.extract import Dinov3ExtractError, select_src_indices
from gatefall.dinov3.features import Dinov3Backbone, compute_features
from gatefall.dinov3.preprocessing import preprocess_frames
from gatefall.dinov3.storage import dinov3_path

FIXED_SAMPLE_VIDEO_IDS: tuple[str, ...] = (
    "coffee_room_01/video_1",  # fps ~25.000
    "home_01/video_1",         # fps ~24.000384
)

ULP_TOLERANCE_MULTIPLE = 2.0


def grid_positions(k: int) -> list[int]:
    if k <= 0:
        return []
    seen: list[int] = []
    for p in (0, k // 2, k - 1):
        if p not in seen:
            seen.append(p)
    return seen


def max_abs_diff_within_tolerance(
    fresh: np.ndarray, stored: np.ndarray, *, ulp_multiple: float = ULP_TOLERANCE_MULTIPLE
) -> tuple[bool, float, float]:
    diff = np.abs(fresh.astype(np.float64) - stored.astype(np.float64))
    max_abs_diff = float(np.max(diff))
    reference_magnitude = float(max(np.max(np.abs(fresh)), np.max(np.abs(stored))))
    ulp = float(np.spacing(np.float16(reference_magnitude))) if reference_magnitude > 0 else 0.0
    tolerance = ulp_multiple * ulp
    return max_abs_diff <= tolerance, max_abs_diff, tolerance


def check_discriminative_match(
    fresh: np.ndarray,
    stored_at_position: np.ndarray,
    *,
    stored_prev: np.ndarray | None,
    stored_next: np.ndarray | None,
) -> tuple[bool, dict[str, float]]:
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))

    distances = {"self": dist(fresh, stored_at_position)}
    ok = True
    if stored_prev is not None:
        distances["prev"] = dist(fresh, stored_prev)
        ok = ok and distances["self"] < distances["prev"]
    if stored_next is not None:
        distances["next"] = dist(fresh, stored_next)
        ok = ok and distances["self"] < distances["next"]
    return ok, distances


def run_dinov3_verify_frame_alignment(
    *,
    adapter: DatasetAdapter,
    repo_dir_value: str | None,
    weights_path_value: str | None,
    video_ids: tuple[str, ...] = FIXED_SAMPLE_VIDEO_IDS,
) -> None:
    repo_dir = resolve_repo_dir(repo_dir_value)
    weights_path = resolve_weights_path(weights_path_value)
    ensure_backbone_paths_exist(repo_dir, weights_path)
    configure_deterministic_inference()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_backbone(repo_dir, weights_path, device)
    video_paths = adapter.video_paths()

    failures: list[str] = []
    for video_id in video_ids:
        try:
            src_indices = select_src_indices(video_id, adapter=adapter)
        except Dinov3ExtractError as exc:
            failures.append(str(exc))
            continue
        k = len(src_indices)
        stored = storage.read_features(dinov3_path(video_id, dinov3_root=adapter.dinov3_root))
        if stored.shape[0] != k:
            failures.append(
                f"{video_id}: K armazenado ({stored.shape[0]}) != frames.parquet ({k})"
            )
            continue

        for position in grid_positions(k):
            src_index = src_indices[position]
            frames_rgb = decode_frames(video_paths[video_id], [src_index])
            batch = preprocess_frames(frames_rgb).to(device)
            fresh = compute_features(cast(Dinov3Backbone, backbone), batch)[0]
            stored_row = stored[position]

            within_tol, max_abs_diff, tolerance = max_abs_diff_within_tolerance(
                fresh, stored_row
            )
            stored_prev = stored[position - 1] if position > 0 else None
            stored_next = stored[position + 1] if position < k - 1 else None
            discriminative_ok, distances = check_discriminative_match(
                fresh, stored_row, stored_prev=stored_prev, stored_next=stored_next
            )

            print(
                f"{video_id} k={position} (src_index={src_index}): "
                f"max_abs_diff={max_abs_diff:.6f} (tolerância={tolerance:.6f}), "
                f"distances={distances}"
            )
            if not within_tol:
                failures.append(
                    f"{video_id} k={position}: max_abs_diff {max_abs_diff:.6f} "
                    f"maior que a tolerância {tolerance:.6f}"
                )
            if not discriminative_ok:
                failures.append(
                    f"{video_id} k={position}: vetor recomputado não é "
                    "estritamente mais próximo da linha correta que das vizinhas"
                )

    if failures:
        print("\ndinov3 verify-frame-alignment FALHOU:", file=sys.stderr)
        for message in failures:
            print(f"  {message}", file=sys.stderr)
        sys.exit(1)
    print(
        "\ndinov3 verify-frame-alignment OK: todas as posições amostradas "
        "batem e são discriminativas"
    )
