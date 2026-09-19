"""Selftest sintético do `FusionWindowDataset` (`fusion_dataset.py`).

Não toca em dados reais — features de pose e visuais vêm de loaders
sintéticos, para travar o alinhamento por índice de quadro entre as duas
fontes e a validação de layout (K e feature_dim) de cada uma.
"""

import sys
from typing import Callable, cast

import numpy as np
import pandas as pd

from gatefall.config import WINDOW_FRAMES
from gatefall.data.fusion_dataset import FusionWindowDataset
from gatefall.data.windowing import build_window_index, window_frame_indices

_POSE_DIM = 134
_VISUAL_DIM = 1536
_N_FRAMES_A = 30
_N_FRAMES_B = 32


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _make_frames(n_frames_by_video: dict[str, int]) -> pd.DataFrame:
    tables = []
    for i, (video_id, n_frames) in enumerate(n_frames_by_video.items()):
        tables.append(
            pd.DataFrame(
                {
                    "video_id": video_id,
                    "split": "train",
                    "env": "coffee_room",
                    "subject": i + 1,
                    "frame_index": np.arange(n_frames, dtype=np.int32),
                    "label": np.zeros(n_frames, dtype=np.int8),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def _synthetic_pose_loader(n_frames_by_video: dict[str, int]) -> Callable[[str], np.ndarray]:
    def loader(video_id: str) -> np.ndarray:
        n_frames = n_frames_by_video[video_id]
        base = (np.arange(n_frames, dtype=np.float32) + 1).reshape(n_frames, 1)
        return np.tile(base, (1, _POSE_DIM))

    return loader


def _synthetic_visual_loader(n_frames_by_video: dict[str, int]) -> Callable[[str], np.ndarray]:
    def loader(video_id: str) -> np.ndarray:
        n_frames = n_frames_by_video[video_id]
        # Marca cada linha com um valor 1000x maior que a pose sintética
        # correspondente, para distinguir facilmente as duas fontes na
        # mesma janela e ainda assim verificar que os índices de quadro
        # batem entre elas.
        base = ((np.arange(n_frames, dtype=np.float32) + 1) * 1000.0).reshape(n_frames, 1)
        return np.tile(base, (1, _VISUAL_DIM))

    return loader


def check_pose_and_visual_windows_share_frame_indices() -> bool:
    n_frames_by_video = {"video_a": _N_FRAMES_A, "video_b": _N_FRAMES_B}
    frames = _make_frames(n_frames_by_video)
    pose_loader = _synthetic_pose_loader(n_frames_by_video)
    visual_loader = _synthetic_visual_loader(n_frames_by_video)
    dataset = FusionWindowDataset(
        frames, split="train", stride=1, pose_loader=pose_loader, visual_loader=visual_loader
    )

    k_end = _N_FRAMES_A - 1
    windows = build_window_index(
        cast(pd.DataFrame, frames[frames["split"] == "train"]), stride=1, drop_ignored=True
    )
    match = windows[(windows["video_id"] == "video_a") & (windows["k_end"] == k_end)]
    index = int(cast(int, match.index[0]))

    x_pose, x_visual, _label, (video_id, returned_k_end) = dataset[index]
    expected_indices = window_frame_indices(k_end, _N_FRAMES_A)
    expected_pose_rows = (expected_indices.astype(np.float32) + 1)
    expected_visual_rows = expected_pose_rows * 1000.0

    ok = (
        x_pose.shape == (WINDOW_FRAMES, _POSE_DIM)
        and x_visual.shape == (WINDOW_FRAMES, _VISUAL_DIM)
        and bool(np.array_equal(x_pose[:, 0], expected_pose_rows))
        and bool(np.array_equal(x_visual[:, 0], expected_visual_rows))
        and video_id == "video_a"
        and returned_k_end == k_end
    )
    return _check(
        "janela de pose [24,134] e janela visual [24,1536] vêm dos mesmos "
        "índices de quadro (window_frame_indices) para o mesmo item",
        ok,
    )


def check_mismatched_video_k_raises_naming_video() -> bool:
    n_frames_by_video = {"video_a": _N_FRAMES_A, "video_b": _N_FRAMES_B}
    frames = _make_frames(n_frames_by_video)
    pose_loader = _synthetic_pose_loader(n_frames_by_video)

    def mismatched_visual_loader(video_id: str) -> np.ndarray:
        # video_a tem K de pose = 30 mas K visual = 29: descasamento.
        k = n_frames_by_video[video_id] - 1 if video_id == "video_a" else n_frames_by_video[video_id]
        return np.zeros((k, _VISUAL_DIM), dtype=np.float32)

    raised = False
    message = ""
    try:
        FusionWindowDataset(
            frames,
            split="train",
            stride=1,
            pose_loader=pose_loader,
            visual_loader=mismatched_visual_loader,
        )
    except ValueError as exc:
        raised = True
        message = str(exc)

    ok = raised and "video_a" in message
    return _check(
        "vídeo cujo array visual tem K diferente do array de pose levanta "
        "ValueError nomeando o video_id",
        ok,
    )


def check_wrong_visual_feature_dim_raises() -> bool:
    n_frames_by_video = {"video_a": _N_FRAMES_A}
    frames = _make_frames(n_frames_by_video)
    pose_loader = _synthetic_pose_loader(n_frames_by_video)

    def wrong_dim_visual_loader(video_id: str) -> np.ndarray:
        return np.zeros((n_frames_by_video[video_id], _VISUAL_DIM - 1), dtype=np.float32)

    raised = False
    try:
        FusionWindowDataset(
            frames,
            split="train",
            stride=1,
            pose_loader=pose_loader,
            visual_loader=wrong_dim_visual_loader,
        )
    except ValueError:
        raised = True
    return _check(
        "array visual com feature_dim != 1536 levanta ValueError",
        raised,
    )


def check_wrong_pose_feature_dim_raises() -> bool:
    n_frames_by_video = {"video_a": _N_FRAMES_A}
    frames = _make_frames(n_frames_by_video)
    visual_loader = _synthetic_visual_loader(n_frames_by_video)

    def wrong_dim_pose_loader(video_id: str) -> np.ndarray:
        return np.zeros((n_frames_by_video[video_id], _POSE_DIM - 1), dtype=np.float32)

    raised = False
    try:
        FusionWindowDataset(
            frames,
            split="train",
            stride=1,
            pose_loader=wrong_dim_pose_loader,
            visual_loader=visual_loader,
        )
    except ValueError:
        raised = True
    return _check(
        "array de pose com feature_dim != 134 levanta ValueError",
        raised,
    )


def run_fusion_dataset_selftest() -> bool:
    checks = [
        check_pose_and_visual_windows_share_frame_indices(),
        check_mismatched_video_k_raises_naming_video(),
        check_wrong_visual_feature_dim_raises(),
        check_wrong_pose_feature_dim_raises(),
    ]
    ok = all(checks)
    if not ok:
        print("\nfusion dataset selftest FALHOU", file=sys.stderr)
    else:
        print("\nfusion dataset selftest OK: todas as checagens passaram")
    return ok
