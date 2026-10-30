"""Comparações descritivas pareadas, sem selecionar parâmetros pelo teste."""

import csv
import json
import statistics
from pathlib import Path

from gatefall.experiments.motion import RECIPES
from gatefall.experiments.physics.data import write_json
from gatefall.hashing import sha256_file
from gatefall.runs import validate_local_run_dir

METRICS = ("macro_f1", "sensitivity", "false_alarms_per_hour", "latency_mean_s", "detected_events")


def read_run(path: Path) -> tuple[dict, dict, list]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    training = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    report = json.loads((path / "evaluation-test.json").read_text(encoding="utf-8"))
    if config.get("motion", {}).get("recipe") not in RECIPES:
        raise ValueError("resumo exige receitas motion explícitas")
    if (report["split"] != "test" or report.get("selection_performed") is not False
            or report["manifest_sha256"] != config["manifest_sha256"]):
        raise ValueError("avaliação não corresponde ao protocolo")
    for filename, field in (("config.json", "config_sha256"), ("checkpoint.pt", "checkpoint_sha256")):
        if not (sha256_file(path / filename) == training[field] == report[field]):
            raise ValueError("hash do run/avaliação divergente")
    events = report["events"]
    row = {"seed": config["seed"], "recipe": config["motion"]["recipe"],
           "macro_f1": report["classification"]["macro_f1_restricted"],
           "sensitivity": events["sensitivity"], "false_alarms_per_hour": events["false_alarms_per_hour"],
           "latency_mean_s": events["latency_seconds"]["mean"], "detected_events": events["n_detected_events"],
           "fall_events": events["n_fall_events"], "false_alarms": events["n_false_alarms"]}
    support = [(p["video_id"], p["k_end"], p["label"]) for p in report["predictions"]]
    environment = {key: training[key] for key in ("device", "torch_version", "numpy_version")}
    return config, row, [support, report["alarm_protocol"], report["latency_precision"],
                         report["code_provenance"]["python_sources_sha256"], environment]


def summarize(runs: list[Path], output: Path) -> dict:
    output = output.resolve()
    validate_local_run_dir(output)
    if output.exists():
        raise FileExistsError(output)
    grouped: dict[int, dict] = {}
    signature = None
    common_support = None
    rows = []
    for path in runs:
        config, row, support = read_run(path)
        seed, recipe = row["seed"], row["recipe"]
        pair = grouped.setdefault(seed, {})
        if recipe in pair:
            raise ValueError("receita/semente duplicada")
        pair[recipe] = (row, config["motion"]["initial_state_sha256"])
        normalized = json.loads(json.dumps(config))
        normalized.pop("seed")
        normalized.pop("code_provenance")
        normalized["motion"].pop("recipe")
        normalized["motion"].pop("initial_state_sha256")
        # A implementação deve ser a mesma; somente estado do checkout pode variar.
        normalized["source_hashes"] = config["code_provenance"]["python_sources_sha256"]
        if signature is None:
            signature, common_support = normalized, support
        elif signature != normalized or common_support != support:
            raise ValueError("runs não pareáveis: configuração, código, fontes ou suporte divergem")
        rows.append(row)
    if not grouped:
        raise ValueError("nenhum run")
    deltas = []
    for seed, pair in sorted(grouped.items()):
        if set(pair) != set(RECIPES) or len({p[1] for p in pair.values()}) != 1:
            raise ValueError("cada semente exige três receitas com inicialização idêntica")
        for left, right in (("motion", "baseline"), ("motion_physics", "baseline"), ("motion_physics", "motion")):
            delta = {"seed": seed, "comparison": left + " - " + right}
            for metric in METRICS:
                a, b = pair[left][0][metric], pair[right][0][metric]
                delta[metric] = None if a is None or b is None else a - b
            deltas.append(delta)

    def aggregate(items: list[dict], key: str) -> dict:
        result = {}
        for label in sorted({row[key] for row in items}):
            result[label] = {}
            for metric in METRICS:
                values = [row[metric] for row in items if row[key] == label and row[metric] is not None]
                result[label][metric] = {
                    "mean": statistics.mean(values) if values else None,
                    "sd": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}
        return result

    result = {"seeds": sorted(grouped), "selection_performed": False, "rows": rows, "paired_deltas": deltas,
              "by_recipe": aggregate(rows, "recipe"), "by_comparison": aggregate(deltas, "comparison"),
              "interpretation": "Descritivo: DP entre sementes não é IC populacional. "
              "Avaliar sensibilidade, FA/h, F1 e latência; erro auxiliar menor não prova ganho.",
              "configuration": signature}
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    for filename, items in (("runs.csv", rows), ("paired-deltas.csv", deltas)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    return result
