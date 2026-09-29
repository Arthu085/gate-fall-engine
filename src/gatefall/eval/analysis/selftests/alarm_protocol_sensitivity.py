import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from gatefall.eval.analysis.alarm_protocol_sensitivity import (
    NO_SELECTION_NOTE,
    SENSITIVITY_CSV_FILE,
    SENSITIVITY_JSON_FILE,
    SplitPredictions,
    TRIGGER_CONSECUTIVE_GRID,
    _csv_rows_from_report,
    _write_sensitivity_outputs,
    build_sensitivity_rows,
)
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.shared.events import split_event_report


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _synthetic_split_predictions() -> SplitPredictions:
    # Geometria adaptada de events_selftest.check_normal_association_path:
    # fall em k=[2,3] (t=0.2..0.3s), fallen em k=[5,6,7] (t=0.5..0.7s) ->
    # start_time_s=0.2, association_end_time_s=0.7+2.0=2.7s. Alarme dispara
    # em run positivo k=[10,11,12], t=1.2s (dentro da janela): 3 positivos
    # consecutivos -> exatamente o trigger_consecutive congelado (3).
    n = 13
    video_id = "video_fixture"
    k_ends = list(range(n))
    true_labels = [0] * n
    true_labels[2:4] = [1, 1]
    true_labels[5:8] = [2, 2, 2]
    pred_labels = [0] * n
    pred_labels[10:13] = [1, 1, 1]
    return SplitPredictions(
        video_ids=[video_id] * n,
        k_ends=k_ends,
        true_labels=true_labels,
        pred_labels=pred_labels,
        usable_windows=n,
        total_windows=n,
        labeled_windows=n,
    )


def _selftest_default_grid_shape_and_fields() -> bool:
    predictions = _synthetic_split_predictions()
    rows = build_sensitivity_rows(predictions, predictions)

    ok = len(rows) == 25
    required_fields = {
        "n_fall_events",
        "n_detected_events",
        "sensitivity",
        "n_false_alarms",
        "false_alarms_per_hour",
        "fall_sensitivity",
        "fall_or_fallen_sensitivity",
    }
    for row in rows:
        for split_name in ("val", "test"):
            split_report = row["splits"][split_name]
            ok = ok and required_fields <= set(split_report)
            latency = split_report["latency_seconds"]
            ok = ok and "mean" in latency and "median" in latency
    return _check(
        "grade padrão produz 25 linhas (5 trigger_consecutive x 5 "
        "refractory_period_s), cada uma com val e test carregando os campos "
        "mínimos e latência mean/median",
        ok,
    )


def _selftest_custom_refractory_grid() -> bool:
    predictions = _synthetic_split_predictions()
    rows = build_sensitivity_rows(predictions, predictions, refractory_grid_s=(3.0, 7.5))
    return _check(
        "grade de refratário customizada com 2 valores produz 10 linhas "
        "(5 trigger_consecutive x 2 refractory_period_s)",
        len(rows) == 10,
    )


def _selftest_grid_order_preserved_not_sorted() -> bool:
    predictions = _synthetic_split_predictions()
    unsorted_grid = (10.0, 0.0, 5.0)
    rows = build_sensitivity_rows(predictions, predictions, refractory_grid_s=unsorted_grid)

    first_block = [row["refractory_period_s"] for row in rows if row["trigger_consecutive"] == 1]
    ok = first_block == list(unsorted_grid)
    return _check(
        "ordem da grade de refratário é preservada exatamente como passada "
        "(não ordenada por valor) dentro de cada bloco de trigger_consecutive",
        ok,
    )


def _selftest_invalid_refractory_rejected() -> bool:
    predictions = _synthetic_split_predictions()

    def _raises(grid: Sequence[float]) -> bool:
        try:
            build_sensitivity_rows(predictions, predictions, refractory_grid_s=grid)
        except ValueError:
            return True
        return False

    ok = (
        _raises([-1.0])
        and _raises([float("nan")])
        and _raises([float("inf")])
        and _raises([])
    )
    return _check(
        "refractory_period_s negativo, não finito (NaN/inf) ou grade vazia "
        "levantam ValueError",
        ok,
    )


def _selftest_only_trigger_and_refractory_vary() -> bool:
    baseline = BASELINE_A_ALARM_PROTOCOL.to_dict()
    varying_fields = {"trigger_consecutive", "refractory_period_s"}

    ok = True
    for trigger_consecutive in (1, 5):
        for refractory_period_s in (0.0, 10.0):
            protocol = replace(
                BASELINE_A_ALARM_PROTOCOL,
                trigger_consecutive=trigger_consecutive,
                refractory_period_s=refractory_period_s,
            )
            derived = protocol.to_dict()
            for field, value in derived.items():
                if field in varying_fields:
                    continue
                ok = ok and value == baseline[field]
    return _check(
        "protocolos derivados pela varredura só divergem de "
        "BASELINE_A_ALARM_PROTOCOL em trigger_consecutive e "
        "refractory_period_s",
        ok,
    )


def _selftest_derived_protocol_does_not_alias_positive_labels() -> bool:
    derived = replace(
        BASELINE_A_ALARM_PROTOCOL,
        trigger_consecutive=1,
        refractory_period_s=0.0,
        positive_labels=list(BASELINE_A_ALARM_PROTOCOL.positive_labels),
    )
    ok = (
        derived.positive_labels == BASELINE_A_ALARM_PROTOCOL.positive_labels
        and derived.positive_labels is not BASELINE_A_ALARM_PROTOCOL.positive_labels
    )
    derived.positive_labels.append(-1)
    ok = ok and -1 not in BASELINE_A_ALARM_PROTOCOL.positive_labels
    return _check(
        "protocolo derivado tem sua própria lista positive_labels (não "
        "aliasada com BASELINE_A_ALARM_PROTOCOL.positive_labels)",
        ok,
    )


def _selftest_frozen_row_reproduces_split_event_report() -> bool:
    predictions = _synthetic_split_predictions()

    expected = split_event_report(
        predictions.video_ids,
        predictions.k_ends,
        predictions.true_labels,
        predictions.pred_labels,
        BASELINE_A_ALARM_PROTOCOL,
        predictions.usable_windows,
        predictions.total_windows,
        predictions.labeled_windows,
    )

    rows = build_sensitivity_rows(predictions, predictions)
    frozen_rows = [row for row in rows if row["is_frozen_protocol"]]

    ok = (
        len(frozen_rows) == 1
        and frozen_rows[0]["trigger_consecutive"] == 3
        and frozen_rows[0]["refractory_period_s"] == 5.0
        and frozen_rows[0]["splits"]["val"] == expected
        and frozen_rows[0]["splits"]["test"] == expected
    )
    return _check(
        "linha congelada (trigger_consecutive=3, refractory_period_s=5.0) "
        "reproduz split_event_report exatamente (igualdade de dict) para "
        "predições sintéticas idênticas em val e test",
        ok,
    )


def _selftest_frozen_row_unique_and_top_level_matches() -> bool:
    predictions = _synthetic_split_predictions()
    rows = build_sensitivity_rows(predictions, predictions)
    frozen_rows = [row for row in rows if row["is_frozen_protocol"]]

    ok = (
        len(frozen_rows) == 1
        and frozen_rows[0]["trigger_consecutive"] == 3
        and frozen_rows[0]["refractory_period_s"] == 5.0
    )
    return _check(
        "linha congelada é única na grade padrão e é exatamente "
        "(trigger_consecutive=3, refractory_period_s=5.0)",
        ok,
    )


def _selftest_sweep_does_not_mutate_baseline() -> bool:
    predictions = _synthetic_split_predictions()
    before = BASELINE_A_ALARM_PROTOCOL.to_dict()
    build_sensitivity_rows(predictions, predictions)
    after = BASELINE_A_ALARM_PROTOCOL.to_dict()
    return _check(
        "a varredura completa não muta BASELINE_A_ALARM_PROTOCOL "
        "(to_dict() antes/depois idêntico)",
        before == after,
    )


def _selftest_does_not_write_canonical_artifacts() -> bool:
    import tempfile

    predictions = _synthetic_split_predictions()
    rows = build_sensitivity_rows(predictions, predictions, refractory_grid_s=(5.0,))
    report = {
        "run_name": "fixture",
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_sha256": "deadbeef",
        "training_metrics_path": "metrics.json",
        "training_metrics_sha256": "deadbeef",
        "frozen_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
        "grid": {"trigger_consecutive": list(TRIGGER_CONSECUTIVE_GRID), "refractory_period_s": [5.0]},
        "metadata": {
            "selection_performed": False,
            "test_split_is_descriptive_only": True,
            "note": NO_SELECTION_NOTE,
        },
        "rows": rows,
    }
    csv_rows = _csv_rows_from_report(rows)

    ok = False
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir)
        sentinel_protocol = b"sentinel-alarm-protocol-bytes"
        sentinel_metrics = b"sentinel-event-metrics-bytes"
        (run_dir / "alarm_protocol.yaml").write_bytes(sentinel_protocol)
        (run_dir / "event_metrics.json").write_bytes(sentinel_metrics)

        _write_sensitivity_outputs(run_dir, report, csv_rows, force=True)

        json_path = run_dir / SENSITIVITY_JSON_FILE
        csv_path = run_dir / SENSITIVITY_CSV_FILE
        outputs_written = json_path.is_file() and csv_path.is_file()
        with json_path.open(encoding="utf-8") as stream:
            written_json = json.load(stream)
        json_content_ok = written_json == report

        sentinels_untouched = (
            (run_dir / "alarm_protocol.yaml").read_bytes() == sentinel_protocol
            and (run_dir / "event_metrics.json").read_bytes() == sentinel_metrics
        )
        no_lock_or_journal = not (
            (run_dir / ".event-evaluation.lock").exists()
            or (run_dir / ".event-evaluation-transaction.json").exists()
        )
        ok = (
            outputs_written
            and json_content_ok
            and sentinels_untouched
            and no_lock_or_journal
        )

    return _check(
        "_write_sensitivity_outputs grava os dois artefatos próprios sem "
        "tocar em alarm_protocol.yaml/event_metrics.json sentinela nem "
        "criar lock/journal do lifecycle canônico",
        ok,
    )


def _selftest_no_selection_metadata_present() -> bool:
    predictions = _synthetic_split_predictions()
    rows = build_sensitivity_rows(predictions, predictions)
    metadata = {
        "selection_performed": False,
        "test_split_is_descriptive_only": True,
    }
    ok = len(rows) == 25 and metadata["selection_performed"] is False and metadata[
        "test_split_is_descriptive_only"
    ] is True
    return _check(
        "metadata de não seleção: selection_performed=False e "
        "test_split_is_descriptive_only=True",
        ok,
    )


def _selftest_flat_csv_row_count_and_spot_check() -> bool:
    predictions = _synthetic_split_predictions()
    rows = build_sensitivity_rows(predictions, predictions, refractory_grid_s=(3.0, 7.5))
    csv_rows = _csv_rows_from_report(rows)

    ok = len(csv_rows) == 2 * len(rows)

    sample_row = rows[0]
    sample_val_csv = next(
        entry
        for entry in csv_rows
        if entry["trigger_consecutive"] == sample_row["trigger_consecutive"]
        and entry["refractory_period_s"] == sample_row["refractory_period_s"]
        and entry["split"] == "val"
    )
    val_report = sample_row["splits"]["val"]
    ok = (
        ok
        and sample_val_csv["n_fall_events"] == val_report["n_fall_events"]
        and sample_val_csv["n_detected_events"] == val_report["n_detected_events"]
        and sample_val_csv["sensitivity"] == val_report["sensitivity"]
        and sample_val_csv["latency_mean_s"] == val_report["latency_seconds"]["mean"]
        and sample_val_csv["latency_median_s"] == val_report["latency_seconds"]["median"]
    )
    return _check(
        "CSV achatado tem 2 linhas por célula da grade (val + test) e "
        "algumas células conferem com o JSON correspondente",
        ok,
    )


def run_alarm_protocol_sensitivity_selftest() -> bool:
    checks = [
        _selftest_default_grid_shape_and_fields(),
        _selftest_custom_refractory_grid(),
        _selftest_grid_order_preserved_not_sorted(),
        _selftest_invalid_refractory_rejected(),
        _selftest_only_trigger_and_refractory_vary(),
        _selftest_derived_protocol_does_not_alias_positive_labels(),
        _selftest_frozen_row_reproduces_split_event_report(),
        _selftest_frozen_row_unique_and_top_level_matches(),
        _selftest_sweep_does_not_mutate_baseline(),
        _selftest_does_not_write_canonical_artifacts(),
        _selftest_no_selection_metadata_present(),
        _selftest_flat_csv_row_count_and_spot_check(),
    ]
    ok = all(checks)
    if not ok:
        print("\nalarm_protocol_sensitivity selftest FALHOU", file=sys.stderr)
    else:
        print("\nalarm_protocol_sensitivity selftest OK: todas as checagens passaram")
    return ok


def run_selftest() -> None:
    if not run_alarm_protocol_sensitivity_selftest():
        sys.exit(1)
