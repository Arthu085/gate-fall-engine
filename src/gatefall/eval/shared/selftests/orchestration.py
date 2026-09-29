"""Checagens sintéticas da orquestração compartilhada de eventos."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.eval.shared.event_artifacts import EventEvaluationLock, validate_event_metrics
from gatefall.eval.shared.orchestration import (
    EventEvaluation,
    Predictions,
    SplitEvaluator,
    run_event_evaluation,
)


class _Config:
    run_name = "synthetic"


def _frames() -> pd.DataFrame:
    tables = []
    for split in ("val", "test"):
        tables.append(
            pd.DataFrame(
                {
                    "video_id": [f"{split}-video"] * 5,
                    "split": [split] * 5,
                    "env": ["room"] * 5,
                    "subject": [split] * 5,
                    "frame_index": list(range(5)),
                    "label": [IGNORE_LABEL, 0, 1, 1, 0],
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def _evaluation(labels: list[int] | None = None, usable_windows: int = 5) -> EventEvaluation:
    frames = _frames()

    def prepare() -> tuple[pd.DataFrame, SplitEvaluator]:
        def evaluate_split(split: str) -> tuple[int, Predictions]:
            return usable_windows, (
                [f"{split}-video"] * 5,
                list(range(5)),
                labels if labels is not None else [IGNORE_LABEL, 0, 1, 1, 0],
                [0] * 5,
            )

        return frames, evaluate_split

    return EventEvaluation(_Config(), prepare)


def _run(run_dir: Path, force: bool, evaluation: EventEvaluation) -> None:
    with EventEvaluationLock(run_dir) as lock:
        run_event_evaluation(force, "A", run_dir, lock, lambda: evaluation)


def _raises(callback, expected: type[Exception]) -> bool:
    try:
        callback()
    except expected:
        return True
    return False


def run_orchestration_selftest() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        evaluation = _evaluation()
        _run(run_dir, False, evaluation)
        protocol_path = run_dir / "alarm_protocol.yaml"
        metrics_path = run_dir / "event_metrics.json"
        initial_bytes = metrics_path.read_bytes()
        report = json.loads(initial_bytes)
        fields_ordered = list(report) == [
            "run_name",
            "checkpoint_path",
            "alarm_protocol_path",
            "splits",
            "checkpoint_sha256",
            "training_metrics_sha256",
            "alarm_protocol_sha256",
        ]
        validate_event_metrics(
            report,
            _Config(),
            run_dir / "checkpoint.pt",
            protocol_path,
            training_metrics_path=run_dir / "metrics.json",
            require_hashes=True,
        )
        counts_valid = all(
            report["splits"][split]["total_windows"] == 5
            and report["splits"][split]["usable_windows"] == 5
            and report["splits"][split]["labeled_windows"] == 4
            and report["splits"][split]["n_fall_events"] == 1
            for split in ("val", "test")
        )
        _run(run_dir, False, evaluation)
        skipped = metrics_path.read_bytes() == initial_bytes
        protocol_path.write_text("invalid: true", encoding="utf-8")
        rejects_corrupt = _raises(lambda: _run(run_dir, False, evaluation), RuntimeError)
        _run(run_dir, True, evaluation)
        metrics_path.unlink()
        rejects_partial = _raises(lambda: _run(run_dir, False, evaluation), RuntimeError)
        _run(run_dir, True, evaluation)
        published_bytes = metrics_path.read_bytes()
        rejects_window_gap = _raises(
            lambda: _run(run_dir, True, _evaluation(usable_windows=4)), RuntimeError
        )
        rejects_fall_mismatch = _raises(
            lambda: _run(run_dir, True, _evaluation([IGNORE_LABEL, 0, 1, 0, 1])),
            ValueError,
        )
        with patch(
            "gatefall.eval.shared.orchestration.validate_event_metrics",
            side_effect=ValueError("staging inválido"),
        ):
            rejects_staging = _raises(lambda: _run(run_dir, True, evaluation), ValueError)
        preserves_pair = metrics_path.read_bytes() == published_bytes
        no_transaction = not (run_dir / ".event-evaluation-transaction.json").exists()
    ok = all(
        (
            counts_valid,
            fields_ordered,
            skipped,
            rejects_corrupt,
            rejects_partial,
            rejects_window_gap,
            rejects_fall_mismatch,
            rejects_staging,
            preserves_pair,
            no_transaction,
        )
    )
    print(f"[{'PASS' if ok else 'FAIL'}] orquestração de eventos: par, janelas, fall, hashes e staging")
    return ok
