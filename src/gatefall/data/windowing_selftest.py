"""Selftest sintético do janelamento (`windowing.py`).

Não toca no dataset real — todas as entradas são sintéticas, para travar o
comportamento do padding por replicação de borda, das fronteiras de janela e
da construção do índice de janelas contra futuras mudanças.
"""

import sys
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.data.windowing import (
    build_window_index,
    window_end_indices,
    window_frame_indices,
)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_window_all_padding() -> bool:
    indices = window_frame_indices(k_end=0, n_frames=100)
    ok = indices.shape[0] == 24 and bool(np.all(indices == 0))
    return _check("k_end=0, n_frames=100 -> 24 índices, todos 0", ok)


def check_window_partial_padding() -> bool:
    indices = window_frame_indices(k_end=10, n_frames=100)
    expected = np.concatenate(
        [np.zeros(14, dtype=np.int64), np.arange(1, 11, dtype=np.int64)]
    )
    ok = bool(np.array_equal(indices, expected))
    return _check("k_end=10, n_frames=100 -> quatorze 0s seguidos de 1..10", ok)


def check_window_no_padding() -> bool:
    indices = window_frame_indices(k_end=23, n_frames=100)
    expected = np.arange(24, dtype=np.int64)
    ok = bool(np.array_equal(indices, expected))
    return _check("k_end=23, n_frames=100 -> exatamente 0..23, sem repetição", ok)


def check_window_single_frame_video() -> bool:
    indices = window_frame_indices(k_end=0, n_frames=1)
    ok = indices.shape[0] == 24 and bool(np.all(indices == 0))
    return _check("n_frames=1 -> a única janela é 24 cópias do índice 0", ok)


def check_window_short_video_stride1() -> bool:
    n_frames = 5
    ends = window_end_indices(n_frames, stride=1)
    ok = ends.shape[0] == 5
    for k_end in ends:
        indices = window_frame_indices(int(k_end), n_frames)
        ok = ok and indices.shape[0] == 24 and int(indices[-1]) == int(k_end)
        ok = ok and bool(np.all(indices <= n_frames - 1))
    return _check(
        "n_frames=5, stride=1 -> 5 janelas, todas clipadas, cada uma "
        "terminando no próprio índice",
        ok,
    )


def check_end_indices_stride4_long() -> bool:
    ends = window_end_indices(100, stride=4)
    expected = np.arange(0, 100, 4, dtype=np.int64)
    ok = ends.shape[0] == 25 and bool(np.array_equal(ends, expected))
    ok = ok and int(ends[0]) == 0 and int(ends[-1]) == 96
    return _check("window_end_indices(100, 4) -> 25 fins, 0,4,...,96", ok)


def check_end_indices_stride4_remainder() -> bool:
    ends = window_end_indices(53, stride=4)
    ok = ends.shape[0] == 14 and int(ends[-1]) == 52
    return _check("window_end_indices(53, 4) -> 14 fins, o último é 52", ok)


def check_end_indices_stride1_length() -> bool:
    ok = True
    for n_frames in (1, 2, 7, 24, 100):
        ends = window_end_indices(n_frames, stride=1)
        ok = ok and ends.shape[0] == n_frames
    return _check(
        "window_end_indices(K, 1) tem comprimento K, para vários valores de K",
        ok,
    )


def _make_two_video_frames() -> pd.DataFrame:
    video_a = pd.DataFrame(
        {
            "video_id": "video_a",
            "split": "train",
            "env": "coffee_room",
            "subject": 1,
            "frame_index": np.arange(10, dtype=np.int32),
            "label": np.array(
                [0, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=np.int8
            ),
        }
    )
    video_b = pd.DataFrame(
        {
            "video_id": "video_b",
            "split": "test",
            "env": "office",
            "subject": 2,
            "frame_index": np.arange(17, dtype=np.int32),
            "label": np.zeros(17, dtype=np.int8),
        }
    )
    return pd.concat([video_a, video_b], ignore_index=True)


def check_build_window_index_no_cross_video() -> bool:
    frames = _make_two_video_frames()
    windows = build_window_index(frames, stride=1, drop_ignored=False)

    ok = True
    window_video_ids = windows["video_id"].to_numpy()
    window_k_ends = windows["k_end"].to_numpy()
    window_labels = windows["label"].to_numpy()
    frame_video_ids = frames["video_id"].to_numpy()
    frame_indices = frames["frame_index"].to_numpy()
    frame_labels = frames["label"].to_numpy()
    for video_id, k_end, label in zip(
        window_video_ids, window_k_ends, window_labels
    ):
        match = (frame_video_ids == video_id) & (frame_indices == k_end)
        expected_label = int(frame_labels[match][0])
        ok = ok and int(label) == expected_label

    ok = ok and bool(set(windows["video_id"].unique()) == {"video_a", "video_b"})
    return _check(
        "tabela sintética com dois vídeos: nenhuma janela mistura video_id, "
        "e o label de cada janela é o label do quadro (video_id, k_end)",
        ok,
    )


def _make_frames_with_ignore() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": "video_a",
            "split": "train",
            "env": "coffee_room",
            "subject": 1,
            "frame_index": np.arange(8, dtype=np.int32),
            "label": np.array(
                [IGNORE_LABEL, 0, 0, IGNORE_LABEL, 1, 1, IGNORE_LABEL, 0],
                dtype=np.int8,
            ),
        }
    )


def check_drop_ignored_removes_only_ignored_ends() -> bool:
    frames = _make_frames_with_ignore()
    stride = 1

    kept_all = build_window_index(frames, stride=stride, drop_ignored=False)
    dropped = build_window_index(frames, stride=stride, drop_ignored=True)

    ignored_ends = set(
        int(k) for k in kept_all[kept_all["label"] == IGNORE_LABEL]["k_end"]
    )
    remaining_ends = set(int(k) for k in dropped["k_end"])
    all_ends = set(int(k) for k in kept_all["k_end"])

    ok = remaining_ends == (all_ends - ignored_ends)
    ok = ok and bool((dropped["label"] != IGNORE_LABEL).all())
    ok = ok and len(kept_all) == len(all_ends)
    return _check(
        "drop_ignored=True remove exatamente as janelas cujo quadro final é "
        "IGNORE_LABEL e nenhuma outra; drop_ignored=False mantém todas",
        ok,
    )


def check_window_count_matches_ceil() -> bool:
    frames = _make_two_video_frames()
    ok = True
    for stride in (1, 4):
        windows = build_window_index(frames, stride=stride, drop_ignored=False)
        for video_id, n_frames in (("video_a", 10), ("video_b", 17)):
            count = int((windows["video_id"] == video_id).sum())
            expected = -(-n_frames // stride)
            ok = ok and count == expected
    return _check(
        "contagem de janelas por vídeo é ceil(K / stride) antes de descartar "
        "(drop_ignored=False)",
        ok,
    )


def run_windowing_selftest() -> None:
    checks = [
        check_window_all_padding(),
        check_window_partial_padding(),
        check_window_no_padding(),
        check_window_single_frame_video(),
        check_window_short_video_stride1(),
        check_end_indices_stride4_long(),
        check_end_indices_stride4_remainder(),
        check_end_indices_stride1_length(),
        check_build_window_index_no_cross_video(),
        check_drop_ignored_removes_only_ignored_ends(),
        check_window_count_matches_ceil(),
    ]
    if not all(checks):
        print("\nselftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: todos os casos passaram")
