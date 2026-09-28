"""Selftest sintético da coleta de rótulos no treino."""

import sys

import numpy as np

from gatefall.train.baseline_b0.engine import _collect_labels as collect_b0_labels
from gatefall.train.shared.gated_engine import _collect_labels as collect_b1_labels
from gatefall.train.baseline_c0.engine import _collect_labels as collect_c0_labels
from gatefall.train.baseline_a.engine import _collect_labels as collect_pose_labels


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_legacy_sources_keep_item_fallback() -> bool:
    labels = (2, 0, 1)
    window = np.zeros((1, 1), dtype=np.float32)

    class PoseSource:
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(self, index: int) -> tuple[np.ndarray, int, object]:
            return window, labels[index], index

    class FusionSource:
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(
            self, index: int
        ) -> tuple[np.ndarray, np.ndarray, int, object]:
            return window, window, labels[index], index

    class GatedSource:
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(
            self, index: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, object]:
            return window, window, window, labels[index], index

    expected = np.array(labels, dtype=np.int64)
    fusion = FusionSource()
    return _check(
        "fontes sintéticas sem label_at mantêm a coleta por __getitem__",
        all(
            np.array_equal(observed, expected)
            for observed in (
                collect_pose_labels(PoseSource()),
                collect_b0_labels(fusion),
                collect_b1_labels(GatedSource()),
                collect_c0_labels(fusion),
            )
        ),
    )


def run_engine_selftest() -> bool:
    checks = [
        check_legacy_sources_keep_item_fallback(),
    ]
    ok = all(checks)
    if not ok:
        print("\nengine selftest FALHOU", file=sys.stderr)
    else:
        print("\nengine selftest OK: todas as checagens passaram")
    return ok
