import json
import tempfile
from dataclasses import replace
from pathlib import Path

from gatefall.datasets import get_dataset
from gatefall.eval.analysis.multiseed_summary import ARMS, MIN_SEEDS, _resolve_shared_expected, _summarize
from gatefall.eval.analysis.selftests.fixtures import (
    _build_event_split,
    _check,
    _rewrite_synthetic_config,
    _synthetic_classification_arrays,
    _synthetic_pair,
    _write_synthetic_seed_run,
)
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG


def _selftest_missing_diagnostics_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_ok = root / "run_ok"
        run_missing_diagnostics = root / "run_missing_diagnostics"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run_ok, seed=1, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )
        _write_synthetic_seed_run(
            run_missing_diagnostics, seed=2, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )

        metrics_path = run_missing_diagnostics / "metrics.json"
        with metrics_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        del metrics["final"]["val"]["confusion_matrix"]
        del metrics["final"]["val"]["per_class"]
        with metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream)

        event_metrics_path = run_missing_diagnostics / "event_metrics.json"
        with event_metrics_path.open(encoding="utf-8") as stream:
            event_metrics = json.load(stream)
        event_metrics["training_metrics_sha256"] = sha256_file(metrics_path)
        with event_metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(event_metrics, stream)

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run_ok, run_missing_diagnostics], shared_expected, adapter)
        except RuntimeError as exc:
            raised = str(run_missing_diagnostics) in str(exc) and "val" in str(exc)
        return _check(
            "final.<split> sem confusion_matrix/per_class levanta RuntimeError "
            "nomeando o run_dir e o split",
            raised,
        )


def _selftest_duplicate_seed_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "run_a"
        run2 = root / "run_b"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run1, seed=5, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )
        _write_synthetic_seed_run(
            run2, seed=5, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run1, run2], shared_expected, adapter)
        except RuntimeError as exc:
            raised = "seed" in str(exc) and str(run1) in str(exc) and str(run2) in str(exc)
        return _check(
            "seed duplicada entre run_dirs levanta RuntimeError nomeando "
            "os dois run_dirs envolvidos",
            raised,
        )


def _selftest_non_seed_config_divergence_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "run_a"
        run2 = root / "run_b"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event, epochs=1
        )
        _write_synthetic_seed_run(
            run2, seed=2, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event, epochs=2
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run1, run2], shared_expected, adapter)
        except RuntimeError as exc:
            raised = str(run2) in str(exc)
        return _check(
            "divergência de configuração fora do campo seed (epochs) "
            "levanta RuntimeError nomeando o run_dir divergente",
            raised,
        )


def _selftest_malformed_run_dir_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_ok = root / "run_ok"
        run_missing_event_metrics = root / "run_missing_event_metrics"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run_ok, seed=1, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )
        _write_synthetic_seed_run(
            run_missing_event_metrics,
            seed=2,
            y_true=y_true,
            y_pred=y_pred,
            val_event=event,
            test_event=event,
        )
        (run_missing_event_metrics / "event_metrics.json").unlink()

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run_ok, run_missing_event_metrics], shared_expected, adapter)
        except RuntimeError as exc:
            raised = str(run_missing_event_metrics) in str(exc)
        return _check(
            "run_dir malformado (event_metrics.json ausente) levanta "
            "RuntimeError nomeando o run_dir",
            raised,
        )


def _selftest_fewer_than_min_seeds_raises() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "run_a"
        y_true, y_pred = _synthetic_classification_arrays(offset=0)
        event = _build_event_split(
            n_fall_events=1,
            n_detected_events=1,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true, y_pred=y_pred, val_event=event, test_event=event
        )
        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        raised = False
        try:
            _summarize([run1], shared_expected, adapter)
        except ValueError:
            raised = True
        return _check(
            f"menos de MIN_SEEDS={MIN_SEEDS} run_dirs levanta ValueError",
            raised,
        )


def _selftest_arm_and_config_guards() -> bool:
    adapter = get_dataset("le2i")
    cv_rejected = True
    for arm in ARMS[1:]:
        try:
            _resolve_shared_expected("le2i-cv", arm)
        except ValueError:
            continue
        cv_rejected = False
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs, expected = _synthetic_pair(root / "b0", "B0")
        duplicate_path = False
        try:
            _summarize([runs[0], runs[0]], expected, adapter)
        except ValueError as exc:
            duplicate_path = "duplicado" in str(exc)
        _rewrite_synthetic_config(runs[1], {"seed": 1})
        duplicate_seed = False
        try:
            _summarize(runs, expected, adapter)
        except RuntimeError as exc:
            duplicate_seed = "seed duplicada" in str(exc)
        _rewrite_synthetic_config(runs[1], {"seed": 2, "lr": 0.5})
        mismatch = False
        try:
            _summarize(runs, expected, adapter)
        except RuntimeError as exc:
            mismatch = "lr" in str(exc)
        _rewrite_synthetic_config(runs[1], {"lr": expected.lr, "trainable_param_count": 123})
        allowed_report, _ = _summarize(runs, expected, adapter)
        audit_allowed = allowed_report["n_seeds"] == 2

        c0_runs, _ = _synthetic_pair(root / "c0", "C0")
        mixed = False
        try:
            _summarize([runs[0], c0_runs[0]], expected, adapter)
        except RuntimeError as exc:
            mixed = str(c0_runs[0]) in str(exc)
        return _check(
            "paths/seeds duplicados, arma mista e configuração científica divergente rejeitados; campo de auditoria permitido",
            duplicate_path and duplicate_seed and mismatch and mixed and audit_allowed and cv_rejected,
        )
