import tempfile
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
import torch

from gatefall.datasets import DatasetAdapter
from gatefall.dinov3 import storage
from gatefall.dinov3.backbone import FEATURE_DIM
from gatefall.dinov3.features import Dinov3Backbone, compute_features
from gatefall.dinov3.frame_alignment import (
    check_discriminative_match,
    grid_positions,
    max_abs_diff_within_tolerance,
    run_dinov3_verify_frame_alignment,
)
from gatefall.dinov3.preprocessing import preprocess_frames
from gatefall.dinov3.selftests.fixtures import _check, _SyntheticDatasetAdapter
from gatefall.dinov3.storage import dinov3_path, write_dinov3_features_atomic


def _check_grid_positions() -> bool:
    ok = (
        grid_positions(0) == []
        and grid_positions(1) == [0]
        and grid_positions(2) == [0, 1]
        and grid_positions(5) == [0, 2, 4]
        and grid_positions(62) == [0, 31, 61]
    )
    return _check(
        "grid_positions: início, meio e fim, sem duplicatas para K pequeno", ok
    )


def _check_max_abs_diff_within_tolerance() -> bool:
    stored = np.full((1, FEATURE_DIM), 5.86, dtype=np.float16)
    close = stored.copy()
    close[0, 0] = np.nextafter(np.float16(5.86), np.float16(0))
    close_ok, _, _ = max_abs_diff_within_tolerance(close, stored)

    far = stored.copy().astype(np.float32)
    far[0, 0] = 1.0
    far_ok, _, _ = max_abs_diff_within_tolerance(far, stored)

    ok = close_ok and not far_ok
    return _check(
        "max_abs_diff_within_tolerance: diferença de arredondamento float16 "
        "passa, diferença grande falha", ok
    )


def _check_check_discriminative_match() -> bool:
    stored_at_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    stored_prev = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    stored_next = np.array([-10.0, -20.0, -30.0], dtype=np.float32)

    match_ok, _, match_inconclusive = check_discriminative_match(
        stored_at_position,
        stored_at_position,
        stored_prev=stored_prev,
        stored_next=stored_next,
    )

    shifted_ok, _, _ = check_discriminative_match(
        stored_next,
        stored_at_position,
        stored_prev=stored_prev,
        stored_next=stored_next,
    )

    tie_ok, _, tie_inconclusive = check_discriminative_match(
        stored_at_position,
        stored_at_position,
        stored_prev=stored_prev,
        stored_next=stored_at_position.copy(),
    )

    ok = (
        match_ok
        and not match_inconclusive
        and not shifted_ok
        and tie_ok
        and tie_inconclusive == ["next"]
    )
    return _check(
        "check_discriminative_match: vetor recomputado igual ao esperado "
        "passa; simulação de deslocamento de um quadro falha; empate exato "
        "com uma vizinha é inconclusivo, não falha", ok
    )


class _IndexAwareFakeBackbone:
    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean_pixel = x.mean(dim=(1, 2, 3))
        batch_size = x.shape[0]
        cls_token = mean_pixel.view(batch_size, 1).expand(batch_size, 768).clone()
        patch_tokens = (
            mean_pixel.view(batch_size, 1, 1).expand(batch_size, 4, 768).clone()
        )
        return {"x_norm_clstoken": cls_token, "x_norm_patchtokens": patch_tokens}


def _frame_for_src_index(src_index: int) -> np.ndarray:
    value = min(255, (src_index + 1) * 10)
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _build_frame_alignment_fixture(
    root: Path,
    *,
    video_id: str,
    k: int,
    include_manifest_row: bool = True,
    include_h5: bool = True,
    h5_row_count: int | None = None,
    stored_row_src_index: Callable[[int], int] | None = None,
) -> _SyntheticDatasetAdapter:
    env, _, video_name = video_id.partition("/")
    raw_dir = root / "raw"
    manifest_path = root / "manifest.parquet"
    frames_path = root / "frames.parquet"
    dinov3_root = root / "dinov3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "video_id": [video_id] * k,
            "frame_index": list(range(k)),
            "src_index": list(range(k)),
        }
    ).to_parquet(frames_path)

    if include_manifest_row:
        pd.DataFrame(
            {
                "video_id": [video_id],
                "relative_path": [f"{env}/{video_name}.avi"],
            }
        ).to_parquet(manifest_path)
    else:
        pd.DataFrame({"video_id": [], "relative_path": []}).to_parquet(manifest_path)

    if include_h5:
        row_count = h5_row_count if h5_row_count is not None else k
        resolve_src_index = stored_row_src_index or (lambda position: position)
        backbone = _IndexAwareFakeBackbone()
        stored = np.zeros((row_count, FEATURE_DIM), dtype=np.float16)
        for position in range(row_count):
            frame = _frame_for_src_index(resolve_src_index(position))
            batch = preprocess_frames([frame])
            stored[position] = compute_features(backbone, batch)[0]
        write_dinov3_features_atomic(
            dinov3_path(video_id, dinov3_root=dinov3_root), stored, {"K": row_count}
        )

    return _SyntheticDatasetAdapter(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        frames_path=frames_path,
        dinov3_root=dinov3_root,
    )


def _run_frame_alignment_and_get_exit_code(
    *,
    adapter: DatasetAdapter,
    video_ids: tuple[str, ...],
    backbone: Dinov3Backbone,
    decode_single_frame: Callable[[Path, int], np.ndarray],
) -> int | None:
    try:
        run_dinov3_verify_frame_alignment(
            adapter=adapter,
            repo_dir_value=None,
            weights_path_value=None,
            video_ids=video_ids,
            backbone=backbone,
            decode_single_frame=decode_single_frame,
        )
    except SystemExit as exc:
        return cast("int | None", exc.code)
    return None


def _check_run_dinov3_verify_frame_alignment_happy_path() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index),
        )
    ok = exit_code in (None, 0)
    return _check(
        "run_dinov3_verify_frame_alignment: caminho feliz sintético não "
        "reporta falha", ok
    )


def _check_run_dinov3_verify_frame_alignment_shifted() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index + 1),
        )
    ok = exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: quadro decodificado deslocado "
        "de uma posição é reportado como falha", ok
    )


class _FinelySpacedFakeBackbone:
    """Backbone sintético cujo valor de saída difere entre posições vizinhas
    por apenas alguns ULPs de float16 na magnitude de `base`, em vez de um
    passo cheio de intensidade de pixel — simula um deslocamento de quadro
    sutil o bastante para passar na checagem de tolerância isoladamente."""

    def __init__(
        self, reference_means: list[float], *, base: float, fine_step: float
    ) -> None:
        self._reference_means = reference_means
        self._base = base
        self._fine_step = fine_step

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean_pixel = x.mean(dim=(1, 2, 3))
        batch_size = x.shape[0]
        outputs = [
            self._base
            + self._fine_step
            * min(
                range(len(self._reference_means)),
                key=lambda i: abs(self._reference_means[i] - value),
            )
            for value in mean_pixel.tolist()
        ]
        output_tensor = torch.tensor(outputs, dtype=torch.float32)
        cls_token = output_tensor.view(batch_size, 1).expand(batch_size, 768).clone()
        patch_tokens = (
            output_tensor.view(batch_size, 1, 1).expand(batch_size, 4, 768).clone()
        )
        return {"x_norm_clstoken": cls_token, "x_norm_patchtokens": patch_tokens}


def _build_fine_grid_frame_alignment_fixture(
    root: Path, *, video_id: str, k: int, backbone: Dinov3Backbone
) -> _SyntheticDatasetAdapter:
    env, _, video_name = video_id.partition("/")
    raw_dir = root / "raw"
    manifest_path = root / "manifest.parquet"
    frames_path = root / "frames.parquet"
    dinov3_root = root / "dinov3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "video_id": [video_id] * k,
            "frame_index": list(range(k)),
            "src_index": list(range(k)),
        }
    ).to_parquet(frames_path)
    pd.DataFrame(
        {"video_id": [video_id], "relative_path": [f"{env}/{video_name}.avi"]}
    ).to_parquet(manifest_path)

    stored = np.zeros((k, FEATURE_DIM), dtype=np.float16)
    for position in range(k):
        frame = _frame_for_src_index(position)
        batch = preprocess_frames([frame])
        stored[position] = compute_features(backbone, batch)[0]
    write_dinov3_features_atomic(
        dinov3_path(video_id, dinov3_root=dinov3_root), stored, {"K": k}
    )

    return _SyntheticDatasetAdapter(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        frames_path=frames_path,
        dinov3_root=dinov3_root,
    )


def _check_run_dinov3_verify_frame_alignment_shifted_within_tolerance() -> bool:
    k = 3
    base = float(np.float16(5.0))
    fine_step = float(np.spacing(np.float16(5.0)))
    reference_means = [
        float(preprocess_frames([_frame_for_src_index(position)]).mean().item())
        for position in range(k)
    ]
    backbone = _FinelySpacedFakeBackbone(reference_means, base=base, fine_step=fine_step)

    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_fine_grid_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=k, backbone=backbone
        )
        stored = storage.read_features(
            dinov3_path(video_id, dinov3_root=adapter.dinov3_root)
        )

        shifted_batch = preprocess_frames([_frame_for_src_index(2)])
        shifted_fresh = compute_features(backbone, shifted_batch)[0]
        tolerance_alone_passes, _, _ = max_abs_diff_within_tolerance(
            shifted_fresh, stored[1]
        )

        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=backbone,
            decode_single_frame=lambda path, index: _frame_for_src_index(index + 1),
        )

    ok = tolerance_alone_passes and exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: deslocamento de um quadro sutil "
        "(poucos ULPs de float16) passa isoladamente na checagem de "
        "tolerância, mas ainda é pego pela checagem discriminativa por "
        "ficar mais próximo da linha vizinha do que da posição correta", ok
    )


def _check_run_dinov3_verify_frame_alignment_k_mismatch() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5, h5_row_count=4
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index),
        )
    ok = exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: K do .h5 divergente de "
        "frames.parquet é reportado como falha, sem exceção não tratada", ok
    )


def _check_run_dinov3_verify_frame_alignment_missing_manifest_row() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5, include_manifest_row=False
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index),
        )
    ok = exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: video_id ausente do manifesto é "
        "acumulado como falha, sem KeyError não tratado", ok
    )


def _check_run_dinov3_verify_frame_alignment_missing_h5() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5, include_h5=False
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index),
        )
    ok = exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: .h5 ausente é acumulado como "
        "falha, sem exceção não tratada de leitura", ok
    )
