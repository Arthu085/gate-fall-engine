import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gatefall.datasets import get_dataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis import multiseed_summary as summary_module
from gatefall.eval.analysis.multiseed_summary import (
    BINARY_FIELDS,
    CLASSIFICATION_SPLITS,
    CSV_COLUMNS,
    EVENT_SCALAR_FIELDS,
    EVENT_SPLITS,
    MULTISEED_SUMMARY_CSV_FILE,
    MULTISEED_SUMMARY_JSON_FILE,
    PER_CLASS_METRIC_FIELDS,
    _summarize,
    _write_multiseed_summary_outputs,
    run_summarize,
)
from gatefall.eval.analysis.selftests.fixtures import (
    _build_event_split,
    _check,
    _synthetic_classification_arrays,
    _synthetic_pair,
    _write_synthetic_seed_run,
)
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG
from gatefall.train.shared.metrics import RESTRICTED_CLASSES


def _selftest_csv_row_inventory_matches_schema() -> bool:
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
        _report, csv_rows = _summarize([run1, run2], shared_expected, adapter)

        n_restricted = len(RESTRICTED_CLASSES)
        n_labels = len(LE2I_LABEL_NAMES)
        expected_n_rows = (
            len(CLASSIFICATION_SPLITS) * (1 + n_restricted)
            + len(CLASSIFICATION_SPLITS) * n_labels * len(PER_CLASS_METRIC_FIELDS)
            + len(CLASSIFICATION_SPLITS) * len(BINARY_FIELDS)
            + len(EVENT_SPLITS) * len(EVENT_SCALAR_FIELDS)
        )
        count_ok = len(csv_rows) == expected_n_rows

        rows_by_key = {
            (row["split"], row["metric_group"], row["entity"], row["metric"]): row
            for row in csv_rows
        }
        expected_present = [
            ("val", "classification", "", "macro_f1_restricted"),
            ("val", "classification", "fall", "f1_by_class"),
            ("val", "per_class", "fall", "precision"),
            ("val", "binary", "", "tp"),
            ("val", "events", "", "sensitivity"),
            ("val", "events", "", "latency_seconds_mean"),
        ]
        presence_ok = all(key in rows_by_key for key in expected_present)
        n_ok = all(rows_by_key[key]["n"] == 2 for key in expected_present)

        return _check(
            "inventário de linhas do CSV bate com a regra do esquema "
            "(contagem derivada da regra, independente do número de seeds) "
            "e tuplas (split, metric_group, entity, metric) esperadas estão "
            "presentes com n==2",
            count_ok and presence_ok and n_ok,
        )


def _selftest_writer_honors_force() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        report = {"n_seeds": 1, "seeds": [], "aggregate": {}, "arm": "A", "config_fingerprint_sha256": "x"}
        csv_rows: list[dict] = []

        wrote_first = _write_multiseed_summary_outputs(output_dir, report, csv_rows, force=False)
        json_path = output_dir / MULTISEED_SUMMARY_JSON_FILE
        original_text = json_path.read_text(encoding="utf-8")

        changed_report = dict(report)
        changed_report["n_seeds"] = 999
        wrote_second_without_force = _write_multiseed_summary_outputs(
            output_dir, changed_report, csv_rows, force=False
        )
        unchanged = json_path.read_text(encoding="utf-8") == original_text

        wrote_third_with_force = _write_multiseed_summary_outputs(
            output_dir, changed_report, csv_rows, force=True
        )
        overwritten = "999" in json_path.read_text(encoding="utf-8")

        canonical_artifacts = (
            "config.yaml",
            "metrics.json",
            "checkpoint.pt",
            "alarm_protocol.yaml",
            "event_metrics.json",
        )
        no_canonical_artifacts = not any(
            (output_dir / name).exists() for name in canonical_artifacts
        )

        ok = (
            wrote_first
            and not wrote_second_without_force
            and unchanged
            and wrote_third_with_force
            and overwritten
            and no_canonical_artifacts
        )
        return _check(
            "writer honra --force: sem --force não sobrescreve nem sinaliza "
            "sucesso; com --force sobrescreve; nunca escreve artefatos "
            "canônicos de treino/avaliação",
            ok,
        )


def _selftest_arm_a_output_compatibility() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs, expected = _synthetic_pair(root / "a", "A")
        default_out = root / "default"
        explicit_out = root / "explicit"
        with patch.object(summary_module, "_resolve_shared_expected", return_value=expected):
            run_summarize("le2i", runs, default_out, False)
            run_summarize("le2i", runs, explicit_out, False, arm="A")
        with (default_out / MULTISEED_SUMMARY_JSON_FILE).open(encoding="utf-8") as stream:
            report = json.load(stream)
        rows = pd.read_csv(default_out / MULTISEED_SUMMARY_CSV_FILE)
        valid = (
            list(report) == ["arm", "config_fingerprint_sha256", "n_seeds", "seeds", "aggregate"]
            and list(report["aggregate"]) == ["classification", "per_class", "binary_fall_fallen", "events"]
            and len(rows) == 340
            and list(rows) == CSV_COLUMNS
            and all(
                (default_out / name).read_bytes() == (explicit_out / name).read_bytes()
                for name in (MULTISEED_SUMMARY_JSON_FILE, MULTISEED_SUMMARY_CSV_FILE)
            )
        )
        return _check("A mantém esquema e bytes JSON/CSV do caminho padrão", valid)
