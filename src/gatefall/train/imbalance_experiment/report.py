"""Relatório pareado e legível por máquina das receitas exploratórias."""

import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import cast

from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.artifacts import validate_training_run
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, load_config
from gatefall.train.shared.imbalance import RECIPES
from gatefall.train.subject_folds.folds import fold_names as cs_fold_names
from gatefall.train.internal_folds.folds import fold_names as cv_fold_names

SEEDS = (42, 43, 44, 45, 46)
FOCUS = ("standing", "other", "walk", "fall", "fallen")


def _read_run(path: Path, protocol: str, fold: str, recipe: str, seed: int) -> dict[str, object]:
    validate_training_run(path)
    sidecar = json.loads((path / "experiment.json").read_text(encoding="utf-8"))
    if (sidecar["protocol"], sidecar["fold"], sidecar["recipe"]["name"], sidecar["seed"]) != (
        protocol, fold, recipe, seed
    ):
        raise RuntimeError(f"proveniência incompatível: {path}")
    if sidecar["recipe"] != vars(RECIPES[recipe]):
        raise RuntimeError(f"parâmetros da receita incompatíveis: {path}")
    config = load_config(path / "config.yaml")
    expected_config = replace(
        BASELINE_A_CONFIG, seed=seed, run_name=f"imbalance_a_{recipe}",
        standardization_stats_path=config.standardization_stats_path,
        standardization_stats_sha256=sidecar["standardization_stats_sha256"],
    )
    if config != expected_config or sidecar["arm"] != "A" or sidecar["epochs"] != config.epochs:
        raise RuntimeError(f"receita de treino incompatível: {path}")
    if sidecar["config_sha256"] != sha256_file(path / "config.yaml") or sidecar[
        "checkpoint_sha256"
    ] != sha256_file(path / "checkpoint.pt"):
        raise RuntimeError(f"hash de artefato incompatível: {path}")
    if sidecar["checkpoint_policy"] != "last_epoch" or sidecar["official_splits_read"] != ["train"]:
        raise RuntimeError(f"contrato de desenvolvimento incompatível: {path}")
    if set(sidecar["development_videos"]) & set(sidecar["retained_videos"]):
        raise RuntimeError(f"vazamento de vídeos: {path}")
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if metrics["epochs_trained"] != sidecar["epochs"] or metrics["final"]["val"] != metrics["final"]["test"]:
        raise RuntimeError(f"épocas ou split retido incompatíveis: {path}")
    retained = metrics["final"]["val"]
    return {
        "path": str(path), "provenance": sidecar,
        "restricted_macro_f1": retained["macro_f1_restricted"],
        "per_class": retained["per_class"],
        "confusion_matrix": retained["confusion_matrix"],
        "event_metrics": {
            "fall_event_sensitivity": None,
            "false_alarms_per_hour": None,
            "latency_seconds": None,
            "status": "protocolo de eventos ainda não validado nas dobras retidas",
        },
    }


def _paired(control: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    left = cast(dict[str, object], control["provenance"])
    right = cast(dict[str, object], candidate["provenance"])
    for key in ("protocol", "fold", "membership_sha256", "input_sha256", "standardization_stats_sha256",
                "development_videos", "retained_videos", "seed", "epochs", "checkpoint_policy"):
        if left[key] != right[key]:
            raise RuntimeError(f"comparação não pareada: {key}")
    control_classes = cast(dict[str, dict[str, float]], control["per_class"])
    candidate_classes = cast(dict[str, dict[str, float]], candidate["per_class"])
    if any(control_classes[name]["support"] != candidate_classes[name]["support"] for name in LE2I_LABEL_NAMES):
        raise RuntimeError("suporte do retido mudou entre controle e candidato")
    return {
        "restricted_macro_f1": cast(float, candidate["restricted_macro_f1"]) - cast(float, control["restricted_macro_f1"]),
        "per_class": {
            name: {metric: candidate_classes[name][metric] - control_classes[name][metric]
                   for metric in ("precision", "recall", "f1", "fp", "fn")}
            for name in LE2I_LABEL_NAMES
        },
        "other_walk_fp": sum(candidate_classes[name]["fp"] - control_classes[name]["fp"]
                             for name in ("other", "walk")),
    }


def build_report(root: Path, protocol: str) -> dict[str, object]:
    if protocol not in ("cs", "cv"):
        raise ValueError("protocolo inválido")
    folds = cs_fold_names() if protocol == "cs" else cv_fold_names()
    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for fold in folds:
        for seed in SEEDS:
            control_path = root / protocol / fold / "baseline" / f"seed_{seed}"
            control = _read_run(control_path, protocol, fold, "baseline", seed) if control_path.is_dir() else None
            for recipe in RECIPES:
                path = root / protocol / fold / recipe / f"seed_{seed}"
                if not path.is_dir():
                    missing.append({"fold": fold, "seed": seed, "recipe": recipe})
                    continue
                row = _read_run(path, protocol, fold, recipe, seed)
                row.update({"fold": fold, "seed": seed, "recipe": recipe})
                if recipe != "baseline":
                    if control is None:
                        raise RuntimeError(f"controle pareado ausente: {control_path}")
                    row["paired_delta"] = _paired(control, row)
                rows.append(row)
    pilot: dict[str, object] = {}
    for recipe in RECIPES:
        if recipe == "baseline":
            continue
        pilot_rows = [row for row in rows if row["seed"] == 42 and row["recipe"] == recipe]
        if len(pilot_rows) != len(folds):
            pilot[recipe] = {"status": "incomplete"}
            continue
        deltas = [cast(dict[str, object], row["paired_delta"]) for row in pilot_rows]
        macro = [cast(float, delta["restricted_macro_f1"]) for delta in deltas]
        safe_falls = all(
            all(cast(dict[str, dict[str, float]], delta["per_class"])[name]["recall"] >= -0.05
                for name in ("fall", "fallen"))
            for delta in deltas
        )
        pilot[recipe] = {
            "mean_paired_delta": sum(macro) / len(macro),
            "positive_folds": sum(value > 0 for value in macro),
            "fall_recall_guard": safe_falls,
            "advance_to_multiseed": sum(value > 0 for value in macro) >= 2
            and sum(macro) / len(macro) >= 0.02 and safe_falls,
        }
    advanced = {
        name for name, decision in pilot.items()
        if cast(dict[str, object], decision).get("advance_to_multiseed") is True
    }
    for entry in missing:
        if entry["seed"] == 42:
            entry["reason"] = "pilot_incomplete"
        elif entry["recipe"] in advanced or (entry["recipe"] == "baseline" and advanced):
            entry["reason"] = "required_multiseed_missing"
        else:
            entry["reason"] = "stopped_by_pilot"
    multiseed: dict[str, object] = {}
    for recipe in RECIPES:
        if recipe == "baseline":
            continue
        recipe_rows = [row for row in rows if row["recipe"] == recipe]
        if len(recipe_rows) != len(folds) * len(SEEDS):
            multiseed[recipe] = {"status": "incomplete"}
            continue
        per_fold: dict[str, object] = {}
        for fold in folds:
            deltas = [cast(dict[str, object], row["paired_delta"]) for row in recipe_rows if row["fold"] == fold]
            classes = [cast(dict[str, dict[str, float]], delta["per_class"]) for delta in deltas]
            per_fold[fold] = {
                "median_restricted_macro_f1_delta": median(cast(float, delta["restricted_macro_f1"]) for delta in deltas),
                "mean_fall_recall_delta": mean(values["fall"]["recall"] for values in classes),
                "mean_fallen_recall_delta": mean(values["fallen"]["recall"] for values in classes),
                "mean_standing_recall_delta": mean(values["standing"]["recall"] for values in classes),
            }
        fold_values = cast(dict[str, dict[str, float]], per_fold)
        multiseed[recipe] = {
            "median_restricted_macro_f1_delta": median(
                cast(float, cast(dict[str, object], row["paired_delta"])["restricted_macro_f1"])
                for row in recipe_rows
            ),
            "positive_fold_medians": sum(
                values["median_restricted_macro_f1_delta"] > 0 for values in fold_values.values()
            ),
            "other_walk_fp_delta": sum(
                cast(int, cast(dict[str, object], row["paired_delta"])["other_walk_fp"])
                for row in recipe_rows
            ),
            "per_fold": per_fold,
            "window_criteria_met": (
                median(cast(float, cast(dict[str, object], row["paired_delta"])["restricted_macro_f1"])
                       for row in recipe_rows) >= 0.02
                and sum(values["median_restricted_macro_f1_delta"] > 0 for values in fold_values.values()) >= 2
                and all(values["mean_fall_recall_delta"] >= -0.05
                        and values["mean_fallen_recall_delta"] >= -0.05
                        and values["mean_standing_recall_delta"] >= -0.05
                        for values in fold_values.values())
                and sum(cast(int, cast(dict[str, object], row["paired_delta"])["other_walk_fp"])
                        for row in recipe_rows) < 0
            ),
            "promotion_status": "blocked_without_validated_fold_event_metrics",
        }
    observed = [
        {"protocol": protocol, "fold": row["fold"], "arm": "A", "seed": row["seed"],
         "recipe": row["recipe"], "restricted_macro_f1": row["restricted_macro_f1"]}
        for row in rows if cast(float, row["restricted_macro_f1"]) >= 0.8
    ]
    return {
        "protocol": protocol, "canonical": False, "selection_split": "official_train_only",
        "restricted_classes": [0, 1, 2, 3, 4, 7, 8, 9],
        "focus_classes": list(FOCUS), "folds": folds, "seeds": list(SEEDS),
        "recipes": list(RECIPES), "runs": rows, "missing": missing,
        "pilot_decisions": pilot, "multiseed_decisions": multiseed,
        "observed_at_or_above_0_800": observed, "candidate_promoted": False,
        "independent_validation": False,
        "limitations": [
            "Janelas sobrepostas não são observações independentes.",
            "Só três grupos de sujeitos ou ambientes são independentes.",
            "O test oficial já foi exposto em análises anteriores.",
            "A avaliação de eventos ainda não foi validada para as dobras retidas.",
        ],
    }


def write_report(root: Path, protocol: str) -> None:
    if protocol == "both":
        reports = {name: build_report(root, name) for name in ("cs", "cv")}
        destination = root / "comparison.json"
        _write_json(destination, {"canonical": False, "protocols": reports})
        print(destination)
        return
    report = build_report(root, protocol)
    destination = root / protocol / "comparison.json"
    _write_json(destination, report)
    print(destination)
    for recipe, decision in cast(dict[str, dict[str, object]], report["pilot_decisions"]).items():
        print(f"{recipe}: {decision}")


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
