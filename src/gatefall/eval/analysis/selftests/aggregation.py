import statistics
import tempfile
from dataclasses import replace
from pathlib import Path

from gatefall.datasets import get_dataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.multiseed_summary import (
    BINARY_FIELDS,
    CLASSIFICATION_SPLITS,
    EVENT_SCALAR_FIELDS,
    _aggregate_stats,
    _summarize,
)
from gatefall.eval.analysis.selftests.fixtures import (
    _build_event_split,
    _check,
    _synthetic_classification_arrays,
    _write_synthetic_seed_run,
)
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG
from gatefall.train.shared.metrics import (
    BINARY_POSITIVE_LABELS,
    binary_projection_from_confusion_matrix,
    classification_summary,
    restricted_macro_f1,
)


def _selftest_aggregate_stats_known_array() -> bool:
    stats = _aggregate_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    ok = (
        stats["n"] == 5
        and stats["mean"] == 3.0
        and stats["std"] is not None
        and abs(stats["std"] - statistics.stdev([1.0, 2.0, 3.0, 4.0, 5.0])) < 1e-12
        and stats["min"] == 1.0
        and stats["max"] == 5.0
    )
    single = _aggregate_stats([7.0])
    single_ok = single["n"] == 1 and single["mean"] == 7.0 and single["std"] is None
    empty = _aggregate_stats([])
    empty_ok = (
        empty["n"] == 0
        and empty["mean"] is None
        and empty["std"] is None
        and empty["min"] is None
        and empty["max"] is None
    )
    return _check(
        "_aggregate_stats reproduz n/mean/std/min/max sobre um array "
        "conhecido, com std=None para n<2 e todos os campos None para n=0",
        ok and single_ok and empty_ok,
    )


def _selftest_two_valid_seed_runs_aggregate() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "seed1"
        run2 = root / "seed2"

        y_true_1, y_pred_1 = _synthetic_classification_arrays(offset=1)
        y_true_2, y_pred_2 = _synthetic_classification_arrays(offset=3)
        expected_macro_f1_1, _ = restricted_macro_f1(y_true_1, y_pred_1)
        expected_macro_f1_2, _ = restricted_macro_f1(y_true_2, y_pred_2)
        expected_summary_1 = classification_summary(y_true_1, y_pred_1, LE2I_LABEL_NAMES)
        expected_binary_1 = binary_projection_from_confusion_matrix(
            expected_summary_1["confusion_matrix"], BINARY_POSITIVE_LABELS
        )

        val_event_1 = _build_event_split(
            n_fall_events=2,
            n_detected_events=1,
            sensitivity=0.5,
            fall_sensitivity=0.5,
            fall_or_fallen_sensitivity=0.5,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        test_event_1 = val_event_1
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true_1, y_pred=y_pred_1, val_event=val_event_1, test_event=test_event_1
        )

        val_event_2 = _build_event_split(
            n_fall_events=2,
            n_detected_events=2,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=2.0,
            n_false_alarms=2,
            latency_mean=3.0,
            latency_median=3.0,
        )
        test_event_2 = val_event_2
        _write_synthetic_seed_run(
            run2, seed=2, y_true=y_true_2, y_pred=y_pred_2, val_event=val_event_2, test_event=test_event_2
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        report, csv_rows = _summarize([run1, run2], shared_expected, adapter)

        expected_macro_f1_values = [expected_macro_f1_1, expected_macro_f1_2]
        expected_stats = _aggregate_stats(expected_macro_f1_values)

        macro_f1_val = report["aggregate"]["classification"]["val"]["macro_f1_restricted"]
        sensitivity_val = report["aggregate"]["events"]["val"]["sensitivity"]
        false_alarms_val = report["aggregate"]["events"]["val"]["false_alarms_per_hour"]
        latency_mean_val = report["aggregate"]["events"]["val"]["latency_seconds_mean"]

        ok = (
            report["n_seeds"] == 2
            and len(report["seeds"]) == 2
            and macro_f1_val["n"] == expected_stats["n"]
            and abs(macro_f1_val["mean"] - expected_stats["mean"]) < 1e-12
            and abs(macro_f1_val["std"] - expected_stats["std"]) < 1e-12
            and macro_f1_val["min"] == expected_stats["min"]
            and macro_f1_val["max"] == expected_stats["max"]
            and sensitivity_val["n"] == 2
            and abs(sensitivity_val["mean"] - 0.75) < 1e-12
            and false_alarms_val["n"] == 2
            and abs(false_alarms_val["mean"] - 1.0) < 1e-12
            and latency_mean_val["n"] == 2
            and abs(latency_mean_val["mean"] - 2.0) < 1e-12
            and len(csv_rows) > 0
        )
        return _check(
            "duas seeds válidas produzem n/mean/std/min/max corretos, "
            "agregados sobre macro_f1_restricted e métricas de evento",
            ok,
        )


def _selftest_seed_blocks_round_trip_and_new_aggregates_exist() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "seed1"
        run2 = root / "seed2"

        y_true_1, y_pred_1 = _synthetic_classification_arrays(offset=1)
        y_true_2, y_pred_2 = _synthetic_classification_arrays(offset=3)
        expected_summary_1 = classification_summary(y_true_1, y_pred_1, LE2I_LABEL_NAMES)

        val_event_1 = _build_event_split(
            n_fall_events=2,
            n_detected_events=1,
            sensitivity=0.5,
            fall_sensitivity=0.5,
            fall_or_fallen_sensitivity=0.5,
            false_alarms_per_hour=0.0,
            n_false_alarms=0,
            latency_mean=1.0,
            latency_median=1.0,
        )
        test_event_1 = val_event_1
        config1 = _write_synthetic_seed_run(
            run1, seed=1, y_true=y_true_1, y_pred=y_pred_1, val_event=val_event_1, test_event=test_event_1
        )

        val_event_2 = _build_event_split(
            n_fall_events=2,
            n_detected_events=2,
            sensitivity=1.0,
            fall_sensitivity=1.0,
            fall_or_fallen_sensitivity=1.0,
            false_alarms_per_hour=2.0,
            n_false_alarms=2,
            latency_mean=3.0,
            latency_median=3.0,
        )
        test_event_2 = val_event_2
        _write_synthetic_seed_run(
            run2, seed=2, y_true=y_true_2, y_pred=y_pred_2, val_event=val_event_2, test_event=test_event_2
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        report, _csv_rows = _summarize([run1, run2], shared_expected, adapter)

        seed1_report = next(r for r in report["seeds"] if r["seed"] == 1)
        round_trip_ok = (
            seed1_report["classification"]["train"]["confusion_matrix"]
            == expected_summary_1["confusion_matrix"]
            and seed1_report["classification"]["train"]["per_class"]
            == expected_summary_1["per_class"]
            and seed1_report["events"]["val"]["latency_seconds"]["per_event"]
            == val_event_1["latency_seconds"]["per_event"]
        )

        aggregate = report["aggregate"]
        new_leaves_ok = (
            "per_class" in aggregate
            and all(
                label in aggregate["per_class"]["val"] for label in LE2I_LABEL_NAMES
            )
            and "binary_fall_fallen" in aggregate
            and all(field in aggregate["binary_fall_fallen"]["val"] for field in BINARY_FIELDS)
            and all(field in aggregate["events"]["val"] for field in EVENT_SCALAR_FIELDS)
        )

        return _check(
            "blocos por seed (confusion_matrix, per_class, latência per_event) "
            "chegam verbatim no relatório e as novas folhas de agregado "
            "(per_class, binary_fall_fallen, campos de evento ampliados) existem",
            round_trip_ok and new_leaves_ok,
        )


def _selftest_binary_fall_fallen_matches_independent_recomputation() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "seed1"
        run2 = root / "seed2"

        y_true_1, y_pred_1 = _synthetic_classification_arrays(offset=1)
        y_true_2, y_pred_2 = _synthetic_classification_arrays(offset=3)
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
            run1, seed=1, y_true=y_true_1, y_pred=y_pred_1, val_event=event, test_event=event
        )
        _write_synthetic_seed_run(
            run2, seed=2, y_true=y_true_2, y_pred=y_pred_2, val_event=event, test_event=event
        )

        shared_expected = replace(config1, seed=BASELINE_A_CONFIG.seed)
        report, _csv_rows = _summarize([run1, run2], shared_expected, adapter)

        ok = True
        for seed_report in report["seeds"]:
            for split in CLASSIFICATION_SPLITS:
                matrix = seed_report["classification"][split]["confusion_matrix"]
                expected = binary_projection_from_confusion_matrix(matrix, BINARY_POSITIVE_LABELS)
                ok = ok and seed_report["binary_fall_fallen"][split] == expected

        return _check(
            "binary_fall_fallen[split] armazenado é idêntico a uma "
            "recomputação independente a partir da confusion_matrix armazenada",
            ok,
        )
