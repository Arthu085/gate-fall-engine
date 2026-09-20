"""Selftest sintético do `GatedFusionWindowDataset` (`gated_fusion_dataset.py`).

Não toca em dados reais: rampas sintéticas por quadro nas três fontes travam o
alinhamento temporal (os mesmos índices de quadro cortam pose, visual e
qualidade), a causalidade (nenhum quadro depois de `k_end` entra na janela), a
replicação de borda no início do vídeo e a ordem de canal da qualidade.
"""

import sys
from typing import Callable, cast

import numpy as np
import pandas as pd

from gatefall.config import WINDOW_FRAMES
from gatefall.data.gated_fusion_dataset import GatedFusionWindowDataset
from gatefall.data.windowing import build_window_index, window_frame_indices
from gatefall.features.quality_storage import QUALITY_CHANNELS

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


def _pose_loader(n_frames_by_video: dict[str, int]) -> Callable[[str], np.ndarray]:
    def loader(video_id: str) -> np.ndarray:
        n_frames = n_frames_by_video[video_id]
        base = (np.arange(n_frames, dtype=np.float32) + 1).reshape(n_frames, 1)
        return np.tile(base, (1, _POSE_DIM))

    return loader


def _visual_loader(n_frames_by_video: dict[str, int]) -> Callable[[str], np.ndarray]:
    def loader(video_id: str) -> np.ndarray:
        n_frames = n_frames_by_video[video_id]
        base = ((np.arange(n_frames, dtype=np.float32) + 1) * 1000.0).reshape(
            n_frames, 1
        )
        return np.tile(base, (1, _VISUAL_DIM))

    return loader


def _quality_loader(n_frames_by_video: dict[str, int]) -> Callable[[str], np.ndarray]:
    def loader(video_id: str) -> np.ndarray:
        n_frames = n_frames_by_video[video_id]
        # Rampa crescente em q_pose e decrescente em q_visual, para que uma
        # troca de canais seja detectável.
        q_pose = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
        q_visual = 1.0 - q_pose
        return np.stack([q_pose, q_visual], axis=1)

    return loader


def _dataset(n_frames_by_video: dict[str, int]) -> GatedFusionWindowDataset:
    return GatedFusionWindowDataset(
        _make_frames(n_frames_by_video),
        split="train",
        stride=1,
        pose_loader=_pose_loader(n_frames_by_video),
        visual_loader=_visual_loader(n_frames_by_video),
        quality_loader=_quality_loader(n_frames_by_video),
    )


def _index_of(n_frames_by_video: dict[str, int], video_id: str, k_end: int) -> int:
    frames = _make_frames(n_frames_by_video)
    windows = build_window_index(
        cast(pd.DataFrame, frames[frames["split"] == "train"]),
        stride=1,
        drop_ignored=True,
    )
    match = windows[(windows["video_id"] == video_id) & (windows["k_end"] == k_end)]
    return int(cast(int, match.index[0]))


def check_interior_window_shares_frame_indices() -> bool:
    n_frames_by_video = {"video_a": _N_FRAMES_A, "video_b": _N_FRAMES_B}
    dataset = _dataset(n_frames_by_video)
    k_end = _N_FRAMES_A - 1
    index = _index_of(n_frames_by_video, "video_a", k_end)

    pose, visual, quality, _label, (video_id, returned_k_end) = dataset[index]
    expected_indices = window_frame_indices(k_end, _N_FRAMES_A)
    expected_pose_rows = expected_indices.astype(np.float32) + 1
    expected_q_pose = np.linspace(0.0, 1.0, _N_FRAMES_A, dtype=np.float32)[
        expected_indices
    ]

    ok = (
        pose.shape == (WINDOW_FRAMES, _POSE_DIM)
        and visual.shape == (WINDOW_FRAMES, _VISUAL_DIM)
        and quality.shape == (WINDOW_FRAMES, QUALITY_CHANNELS)
        and quality.dtype == np.float32
        and bool(np.array_equal(pose[:, 0], expected_pose_rows))
        and bool(np.array_equal(visual[:, 0], expected_pose_rows * 1000.0))
        and bool(np.array_equal(quality[:, 0], expected_q_pose))
        and bool(np.allclose(quality[:, 1], 1.0 - expected_q_pose))
        and video_id == "video_a"
        and returned_k_end == k_end
    )
    return _check(
        "janela interior: pose [24,134], visual [24,1536] e qualidade [24,2] "
        "vêm dos mesmos índices de quadro, com canal 0 = q_pose e canal 1 = "
        "q_visual",
        ok,
    )


def check_edge_window_replicates_first_frame_and_stays_causal() -> bool:
    n_frames_by_video = {"video_a": _N_FRAMES_A}
    dataset = _dataset(n_frames_by_video)
    k_end = 5
    index = _index_of(n_frames_by_video, "video_a", k_end)

    pose, _visual, quality, _label, _diag = dataset[index]
    expected_indices = window_frame_indices(k_end, _N_FRAMES_A)
    q_pose_full = np.linspace(0.0, 1.0, _N_FRAMES_A, dtype=np.float32)

    observed_frames = np.rint(pose[:, 0]).astype(np.int64) - 1
    causal = bool(np.all(observed_frames <= k_end))
    replicated = bool(np.all(observed_frames[: WINDOW_FRAMES - k_end - 1] == 0))
    quality_aligned = bool(np.array_equal(quality[:, 0], q_pose_full[expected_indices]))

    return _check(
        f"janela de borda (k_end={k_end} < {WINDOW_FRAMES - 1}): replica o "
        "primeiro quadro, nenhum quadro posterior a k_end entra na janela "
        "(causalidade) e a qualidade segue exatamente os mesmos índices",
        causal and replicated and quality_aligned,
    )


def check_quality_window_is_deterministic() -> bool:
    n_frames_by_video = {"video_a": _N_FRAMES_A}
    index = _index_of(n_frames_by_video, "video_a", _N_FRAMES_A - 1)
    first = _dataset(n_frames_by_video)[index][2]
    second = _dataset(n_frames_by_video)[index][2]
    return _check(
        "duas construções do dataset devolvem a mesma janela de qualidade bit "
        "a bit para o mesmo item",
        bool(np.array_equal(first, second)),
    )


def check_mismatched_quality_k_raises_naming_video_and_source() -> bool:
    n_frames_by_video = {"video_a": _N_FRAMES_A}
    frames = _make_frames(n_frames_by_video)

    def short_quality_loader(video_id: str) -> np.ndarray:
        return np.zeros((_N_FRAMES_A - 1, QUALITY_CHANNELS), dtype=np.float32)

    raised = False
    message = ""
    try:
        GatedFusionWindowDataset(
            frames,
            split="train",
            stride=1,
            pose_loader=_pose_loader(n_frames_by_video),
            visual_loader=_visual_loader(n_frames_by_video),
            quality_loader=short_quality_loader,
        )
    except ValueError as exc:
        raised = True
        message = str(exc)

    ok = raised and "video_a" in message and "qualidade" in message
    return _check(
        "qualidade com K divergente do n_frames da tabela de frames levanta "
        "ValueError nomeando o video_id e a fonte",
        ok,
    )


def check_wrong_quality_channel_count_raises() -> bool:
    n_frames_by_video = {"video_a": _N_FRAMES_A}
    frames = _make_frames(n_frames_by_video)

    def wide_quality_loader(video_id: str) -> np.ndarray:
        return np.zeros((_N_FRAMES_A, QUALITY_CHANNELS + 1), dtype=np.float32)

    raised = False
    try:
        GatedFusionWindowDataset(
            frames,
            split="train",
            stride=1,
            pose_loader=_pose_loader(n_frames_by_video),
            visual_loader=_visual_loader(n_frames_by_video),
            quality_loader=wide_quality_loader,
        )
    except ValueError:
        raised = True
    return _check(
        f"qualidade com número de canais != {QUALITY_CHANNELS} levanta ValueError",
        raised,
    )


def run_gated_fusion_dataset_selftest() -> bool:
    checks = [
        check_interior_window_shares_frame_indices(),
        check_edge_window_replicates_first_frame_and_stays_causal(),
        check_quality_window_is_deterministic(),
        check_mismatched_quality_k_raises_naming_video_and_source(),
        check_wrong_quality_channel_count_raises(),
    ]
    ok = all(checks)
    if not ok:
        print("\ngated fusion dataset selftest FALHOU", file=sys.stderr)
    else:
        print("\ngated fusion dataset selftest OK: todas as checagens passaram")
    return ok
