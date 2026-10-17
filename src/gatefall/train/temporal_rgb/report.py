"""Diferenças pareadas e critério de progressão registrado antes do piloto CS."""

from pathlib import Path
from statistics import mean
from typing import Any

from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.train.temporal_rgb.contract import MODES, PILOT_CRITERIA, SEED, contract, guard_destination
from gatefall.train.temporal_rgb.data import Context
from gatefall.train.temporal_rgb.storage import write_json
from gatefall.train.temporal_rgb.training import run_probe


def paired(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if any(control["per_class"][name]["support"] != candidate["per_class"][name]["support"] for name in LE2I_LABEL_NAMES):
        raise ValueError("suportes diferentes não podem ser pareados")
    return {
        "restricted_macro_f1": candidate["restricted_macro_f1"] - control["restricted_macro_f1"],
        "per_class": {
            name: {key: candidate["per_class"][name][key] - control["per_class"][name][key]
                   for key in ("precision", "recall", "f1", "fp", "fn")}
            for name in LE2I_LABEL_NAMES
        },
        "other_walk_fp": sum(candidate["per_class"][name]["fp"] - control["per_class"][name]["fp"]
                             for name in ("other", "walk")),
    }


def pilot_decision(rows: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    if len(rows) != 3 or len({row["fold"] for row in rows}) != 3:
        return {"status": "incomplete", "advance": False}
    vs_a = [row["dynamic_minus_a"] for row in rows]
    vs_static = [row["dynamic_minus_static"] for row in rows]
    criteria = PILOT_CRITERIA
    checks = {
        "mean_gain_vs_a": mean(delta["restricted_macro_f1"] for delta in vs_a) >= criteria["mean_delta_vs_a_min"],
        "positive_folds_vs_a": sum(delta["restricted_macro_f1"] > 0 for delta in vs_a) >= criteria["positive_folds_vs_a_min"],
        "temporal_gain": mean(delta["restricted_macro_f1"] for delta in vs_static) >= criteria["mean_delta_vs_static_min"],
        "positive_folds_vs_static": sum(delta["restricted_macro_f1"] > 0 for delta in vs_static) >= criteria["positive_folds_vs_static_min"],
        "fall_fallen_recall": all(delta["per_class"][name]["recall"] >= criteria["fall_fallen_recall_delta_min_each_fold"]
                                 for delta in vs_a for name in ("fall", "fallen")),
        "other_walk_fp": all(delta["other_walk_fp"] <= criteria["other_walk_fp_delta_max_each_fold"] for delta in vs_a),
    }
    return {
        "status": "complete", "checks": checks,
        "mean_dynamic_minus_a": mean(delta["restricted_macro_f1"] for delta in vs_a),
        "mean_dynamic_minus_static": mean(delta["restricted_macro_f1"] for delta in vs_static),
        "window_criteria_met": all(checks.values()),
        "advance": protocol == "cs" and all(checks.values()),
        "promotion": False,
    }


def build_report(context: Context, fold_root: Path, root: Path, device: str) -> dict[str, Any]:
    rows = []
    missing = []
    for partition in context.partitions:
        runs = {}
        for mode in MODES:
            path = root / context.protocol / "runs" / partition.fold / mode / f"seed_{SEED}"
            if not path.exists():
                missing.append({"fold": partition.fold, "mode": mode, "seed": SEED})
            else:
                runs[mode] = run_probe(context, partition, fold_root, root, mode, device, verify_only=True)
        if len(runs) != len(MODES):
            continue
        left, right = runs["dynamic"]["provenance"], runs["static"]["provenance"]
        if {key: value for key, value in left.items() if key != "mode"} != {key: value for key, value in right.items() if key != "mode"}:
            raise ValueError("dinâmico e estático não são pareados")
        baseline = left["baseline_a"]
        rows.append({
            "fold": partition.fold, "seed": SEED, "baseline_a": baseline, **runs,
            "dynamic_minus_a": paired(baseline, runs["dynamic"]),
            "static_minus_a": paired(baseline, runs["static"]),
            "dynamic_minus_static": paired(runs["static"], runs["dynamic"]),
        })
    return {
        "protocol": context.protocol, "contract": contract(), "runs": rows, "missing": missing,
        "pilot_decision": pilot_decision(rows, context.protocol),
        "event_metrics": {"fall_event_sensitivity": None, "false_alarms_per_hour": None, "latency_seconds": None,
                          "status": "unavailable_without_validated_fold_event_evaluator"},
        "observed_at_or_above_0_800": [
            {"fold": row["fold"], "seed": SEED, "mode": mode, "restricted_macro_f1": row[mode]["restricted_macro_f1"]}
            for row in rows for mode in MODES if row[mode]["restricted_macro_f1"] >= 0.8
        ],
        "candidate_promoted": False, "independent_validation": False,
        "limitations": [
            "Só três grupos retidos e uma seed; janelas sobrepostas não são independentes.",
            "O test oficial já foi exposto por estudos anteriores e não é lido nesta investigação.",
            "Revocação de fall/fallen por janela não estabelece sensibilidade de eventos.",
            "A sonda RGB e a TCN A têm representações e arquiteturas diferentes; a comparação não isola só a modalidade.",
            "A taxa temporal dos clipes Le2i difere dos 15 fps da avaliação Kinetics descrita pelo fornecedor.",
        ],
    }


def write_report(context: Context, fold_root: Path, root: Path, device: str) -> dict[str, Any]:
    report = build_report(context, fold_root, root, device)
    path = root / context.protocol / "comparison.json"
    guard_destination(path)
    write_json(path, report)
    return report
