"""Caracterização sintética do relatório de classificação compartilhado."""

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gatefall.hashing import sha256_file
from gatefall.train.shared.classification_report import publish_classification_report
from gatefall.train.shared.metrics import restricted_macro_f1, support


def _check(name: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    return condition


def run_classification_report_selftest() -> bool:
    label_names = ("ação", "fall", "fallen", "3", "4", "5", "6", "7", "8", "9")
    predictions = {
        "train": (
            np.array([0, 1, 2, 3, 4, 7, 8, 9]),
            np.array([0, 1, 2, 3, 4, 7, 8, 9]),
        ),
        "val": (np.array([1, 2]), np.array([0, 2])),
        "test": (np.array([0, 7]), np.array([0, 0])),
    }
    stored_final = {}
    for split_name, (y_true, y_pred) in predictions.items():
        macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred)
        split_support = support(y_true)
        stored_final[split_name] = {
            "macro_f1_restricted": macro_f1,
            "f1_by_class": {str(c): value for c, value in f1_by_class.items()},
            "support": {label_names[c]: count for c, count in split_support.items()},
        }
    stored_final["train"]["macro_f1_restricted"] += 5e-13
    stored_final["train"]["f1_by_class"]["0"] += 5e-13

    with tempfile.TemporaryDirectory() as temporary_dir:
        run_dir = Path(temporary_dir) / "run"
        run_dir.mkdir()
        checkpoint_path = run_dir / "checkpoint.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps({"final": stored_final}), encoding="utf-8")
        output_path = Path(temporary_dir) / "reports" / "classification_report.json"

        def publish() -> tuple[bool, str, str]:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                ok = publish_classification_report(
                    predictions=(
                        (split_name, y_true, y_pred)
                        for split_name, (y_true, y_pred) in predictions.items()
                    ),
                    label_names=label_names,
                    num_classes=10,
                    run_name="baseline_a",
                    dataset_name="le2i",
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    device="cpu",
                    output_path=output_path,
                )
            return ok, stdout.getvalue(), stderr.getvalue()

        with patch(
            "gatefall.train.shared.classification_report.os.replace", wraps=os.replace
        ) as replace:
            ok, stdout, stderr = publish()
        temporary_path, replaced_path = replace.call_args.args
        report = json.loads(output_path.read_text(encoding="utf-8"))
        shape_ok = (
            list(report) == [
                "run_name", "dataset", "run_dir", "checkpoint_sha256", "device",
                "splits", "verification_against_metrics_json", "class_support_table",
                "macro_f1_policy",
            ]
            and list(report["splits"]) == ["train", "val", "test"]
            and list(report["splits"]["train"]) == [
                "confusion_matrix", "per_class", "binary_fall_fallen"
            ]
            and report["checkpoint_sha256"] == sha256_file(checkpoint_path)
            and report["class_support_table"][0]["label"] == "ação"
            and report["class_support_table"][0]["train_support"] == 1
            and report["macro_f1_policy"]["matches_configured_restriction"]
        )
        clean_ok = (
            ok
            and report["verification_against_metrics_json"] == {"ok": True, "mismatches": []}
            and output_path.read_text(encoding="utf-8")
            == json.dumps(report, indent=2, ensure_ascii=False)
            and stdout.startswith(
                f"{output_path}: relatório de classificação gravado (run_name=baseline_a)\n"
            )
            and "política de macro-F1: classes restritas" in stdout
            and stderr == ""
        )
        atomic_ok = (
            replace.call_count == 1
            and Path(temporary_path).parent == output_path.parent
            and Path(temporary_path).name.startswith(f".{output_path.name}.tmp-")
            and replaced_path == output_path
            and not Path(temporary_path).exists()
        )

        stored_final["val"]["macro_f1_restricted"] += 1e-6
        stored_final["test"]["f1_by_class"]["0"] += 1e-6
        stored_final["test"]["support"]["ação"] += 1
        metrics_path.write_text(json.dumps({"final": stored_final}), encoding="utf-8")
        mismatch_ok, mismatch_stdout, mismatch_stderr = publish()
        mismatch_report = json.loads(output_path.read_text(encoding="utf-8"))
        mismatches = mismatch_report["verification_against_metrics_json"]["mismatches"]
        mismatch_result_ok = (
            not mismatch_ok
            and mismatch_report["verification_against_metrics_json"]["ok"] is False
            and [(entry["split"], entry["field"]) for entry in mismatches] == [
                ("val", "macro_f1_restricted"),
                ("test", "f1_by_class[0]"),
                ("test", "support[ação]"),
            ]
            and all(list(entry) == ["split", "field", "stored", "recomputed"] for entry in mismatches)
            and "política de macro-F1: classes restritas" in mismatch_stdout
            and mismatch_stderr == "verificação contra metrics.json falhou: 3 divergência(s)\n"
        )

    checks = [
        _check("relatório preserva esquema, ordem e suporte", shape_ok),
        _check("relatório preserva JSON, stdout e tolerância de 1e-12", clean_ok),
        _check("relatório publica via arquivo temporário e os.replace", atomic_ok),
        _check("relatório preserva divergências, ordem, stderr e sobrescrita", mismatch_result_ok),
    ]
    if not all(checks):
        print("\nclassification_report selftest FALHOU", file=sys.stderr)
    return all(checks)
