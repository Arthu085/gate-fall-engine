"""Agregação verificada das avaliações do plano em tabelas, figuras e manifesto."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import TARGET_FPS
from gatefall.data.staged.specs import STAGED_SPECS
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.shared.events import detect_alarms_for_video
from gatefall.hashing import sha256_file
from gatefall.provenance import git_state, utc_now
from gatefall.study import STUDY_DIR_NAME, STUDY_ID
from gatefall.study.figures import scatter, seed_dot_plot, timeline
from gatefall.study.methods import GATE_ABLATION_FUSIONS, MOTION_CONTRASTS, MOTION_METHODS, PRIMARY_CONTRASTS
from gatefall.study.metrics import cluster_statistics, evaluation_metrics
from gatefall.study.plan import Step
from gatefall.study.profiling import extraction_seconds_per_frame, latest_profile
from gatefall.study.protocol import SEEDS, canonical_json, protocol_sha256
from gatefall.study.registry import verify_run_provenance
from gatefall.study.statistics import cluster_bootstrap, metric_functions, paired_seed_deltas, seed_summary

REPORT_DIR = Path("runs/local") / STUDY_DIR_NAME / "reports"
LABEL_NAMES = tuple(LE2I_LABEL_NAMES)
SUMMARY_METRICS = ("macro_f1_supported", "macro_f1_restricted8", "event_sensitivity", "sensitivity_during_fall",
                   "sensitivity_only_after_fall", "false_alarms_per_hour", "latency_detected_median_s",
                   "latency_median_all_events_s", "missed_events")
BOOTSTRAP_METRICS = ("macro_f1_supported", "event_sensitivity", "sensitivity_during_fall", "false_alarms_per_hour")
TIMELINE_VIDEOS = 3


class IncompatibleAggregation(ValueError):
    pass


@dataclass(frozen=True)
class EvaluatedRun:
    step: Step
    dataset: str
    label: str
    seed: int
    report: dict
    metrics: dict
    evaluation_sha256: str
    provenance_sha256: str
    origin: str = "executed"
    equivalence_sha256: str | None = None


CLUSTER_UNITS = ("subject", "recording")
PRIMARY_CLUSTER_UNIT = "subject"
ClusterMaps = dict[str, dict[str, str]]


def subject_by_video(dataset: str, frames: pd.DataFrame) -> dict[str, str]:
    """Sujeito de cada vídeo a partir da grade do dataset; recusa vídeo com mais de um sujeito."""
    pairs = cast(pd.DataFrame, frames[["video_id", "subject"]]).drop_duplicates()
    counts = pairs.groupby("video_id")["subject"].nunique().to_dict()
    ambiguous = sorted(str(video) for video, count in counts.items() if count > 1)
    if ambiguous:
        raise IncompatibleAggregation(f"{dataset}: vídeo(s) com mais de um sujeito: {ambiguous[:5]}")
    mapping = {str(video): f"subject-{int(subject)}" for video, subject in zip(pairs["video_id"], pairs["subject"])}
    spec = STAGED_SPECS.get(dataset)
    if spec is not None:
        divergent = [video for video, subject in mapping.items()
                     if subject != f"subject-{spec.parse_video_id(video)['subject']}"]
        if divergent:
            raise IncompatibleAggregation(f"{dataset}: sujeito da grade diverge do identificador: {divergent[:5]}")
    return mapping


def recording_of(dataset: str, video_id: str) -> str:
    # Vistas sincronizadas do UP-Fall formam uma gravação; nos demais, gravação = vídeo.
    spec = STAGED_SPECS.get(dataset)
    return spec.recording_id(video_id) if spec is not None else video_id


def cluster_maps(dataset: str, evaluated: set[str], frames: pd.DataFrame) -> ClusterMaps:
    """Mapas vídeo→cluster verificados contra os vídeos de teste da grade e as predições."""
    test_videos = {str(video) for video in frames.loc[frames["split"] == "test", "video_id"].unique()}
    if test_videos != evaluated:
        raise IncompatibleAggregation(
            f"{dataset}: vídeos avaliados divergem do teste da grade "
            f"(faltam {sorted(test_videos - evaluated)[:3]}, sobram {sorted(evaluated - test_videos)[:3]})")
    subjects = subject_by_video(dataset, frames)
    return {"subject": {video: subjects[video] for video in sorted(evaluated)},
            "recording": {video: recording_of(dataset, video) for video in sorted(evaluated)}}


def load_frames(dataset: str) -> pd.DataFrame:
    from gatefall.datasets import get_dataset

    return get_dataset(dataset).load_frames()


def run_label(step: Step) -> str:
    assert step.run is not None
    return step.run.method.name if step.run.recipe is None else f"{step.run.method.name}/{step.run.recipe}"


def flat_metrics(metrics: dict) -> dict[str, float | None]:
    events = metrics["events"]
    return {
        "macro_f1_supported": metrics["macro_f1_supported"],
        "macro_f1_restricted8": metrics["macro_f1_restricted8"],
        "event_sensitivity": events["sensitivity"],
        "sensitivity_during_fall": events["sensitivity_during_fall"],
        "sensitivity_only_after_fall": events["sensitivity_only_after_fall"],
        "false_alarms_per_hour": events["false_alarms"]["per_hour"],
        "latency_detected_median_s": events["latency_detected_s"]["median"],
        "latency_median_all_events_s": events["latency_median_all_events_s"],
        "missed_events": float(events["n_missed"]),
    }


def truth_sha256(report: dict) -> str:
    rows = sorted((row["video_id"], row["k_end"], row["label"]) for row in report["predictions"])
    return hashlib.sha256(canonical_json(rows)).hexdigest()


def collect(steps: list[Step], matrix: str) -> list[EvaluatedRun]:
    members = [step for step in steps if matrix in step.matrices and step.stage in ("evaluate", "transfer")]
    if not members:
        raise IncompatibleAggregation(f"matriz {matrix} sem avaliações planejadas")
    missing = [step.key for step in members if not step.outputs[0].exists()]
    if missing:
        raise IncompatibleAggregation(f"{len(missing)} avaliação(ões) ausente(s), ex.: {missing[:3]}; "
                                      "relatório parcial recusado")
    runs = []
    for step in members:
        assert step.run is not None
        if step.verify is None:
            raise IncompatibleAggregation(f"{step.key}: avaliação sem verificador de integridade")
        verdict = step.verify()
        if verdict.status != "done":
            raise IncompatibleAggregation(f"{step.key}: avaliação não confere ({verdict.status}): "
                                          + "; ".join(verdict.reasons))
        provenance = verify_run_provenance(step.run)
        if provenance["protocol_sha256"] != protocol_sha256():
            raise IncompatibleAggregation(f"{step.key}: protocolo congelado divergente")
        path = step.outputs[0]
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["checkpoint_sha256"] != provenance["artifacts"]["checkpoint.pt"]:
            raise IncompatibleAggregation(f"{step.key}: avaliação de outro checkpoint")
        if report.get("selection_performed") is not False:
            raise IncompatibleAggregation(f"{step.key}: avaliação com seleção no teste")
        if step.stage == "transfer" and report.get("adaptation") != "none":
            raise IncompatibleAggregation(f"{step.key}: transferência com adaptação")
        dataset = report.get("target_dataset", step.run.dataset)
        runs.append(EvaluatedRun(step, dataset, run_label(step), step.run.seed, report,
                                 evaluation_metrics(report, LABEL_NAMES), sha256_file(path),
                                 sha256_file(step.run.run_dir / "study-provenance.json"), provenance["origin"],
                                 (provenance.get("equivalence") or {}).get("sha256")))
    check_compatible(runs)
    return runs


def check_compatible(runs: list[EvaluatedRun]) -> None:
    by_dataset: dict[str, set[str]] = {}
    seeds: dict[tuple[str, str], set[int]] = {}
    for run in runs:
        by_dataset.setdefault(run.dataset, set()).add(truth_sha256(run.report))
        seeds.setdefault((run.dataset, run.label), set()).add(run.seed)
    divergent = sorted(dataset for dataset, hashes in by_dataset.items() if len(hashes) != 1)
    if divergent:
        raise IncompatibleAggregation(f"rótulos/janelas de teste divergem entre métodos em {divergent}")
    incomplete = sorted(f"{d}:{label}" for (d, label), found in seeds.items() if found != set(SEEDS))
    if incomplete:
        raise IncompatibleAggregation(f"sementes incompletas: {incomplete}")


def contrasts_for(matrix: str, labels: set[str]) -> list[tuple[str, str]]:
    if matrix == "ablations":
        pairs = [(f"{arm}-{fusion}", f"{arm}-constant") for arm in ("B1", "C1")
                 for fusion in GATE_ABLATION_FUSIONS if fusion != "constant"]
    elif matrix == "motion":
        pairs = [(f"{method.name}/{treatment}", f"{method.name}/{reference}")
                 for method in MOTION_METHODS for treatment, reference in MOTION_CONTRASTS]
    else:
        pairs = list(PRIMARY_CONTRASTS)
    return [pair for pair in pairs if set(pair) <= labels]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def aggregate(runs: list[EvaluatedRun], matrix: str, maps: dict[str, ClusterMaps]) -> dict:
    groups: dict[tuple[str, str], dict[int, EvaluatedRun]] = {}
    for run in runs:
        groups.setdefault((run.dataset, run.label), {})[run.seed] = run
    summary, per_class, contrasts = [], [], []
    for (dataset, label), by_seed in sorted(groups.items()):
        classes = tuple(next(iter(by_seed.values())).metrics["supported_classes"])
        row: dict = {"dataset": dataset, "method": label, "n_seeds": len(by_seed)}
        for name in SUMMARY_METRICS:
            stats = seed_summary({seed: flat_metrics(run.metrics)[name] for seed, run in by_seed.items()})
            row[f"{name}_mean"] = stats["mean"]; row[f"{name}_std"] = stats["std"]
        functions = metric_functions(classes)
        for unit in CLUSTER_UNITS:
            clustered = {seed: cluster_statistics(run.report["predictions"], maps[dataset][unit].__getitem__)
                         for seed, run in by_seed.items()}
            # Sujeito é a unidade primária; gravação é análise secundária (sufixo _recording).
            suffix = "" if unit == PRIMARY_CLUSTER_UNIT else f"_{unit}"
            for name in BOOTSTRAP_METRICS:
                interval = cluster_bootstrap(clustered, None, functions[name])
                row[f"{name}{suffix}_ci_low"] = interval["ci_low"]
                row[f"{name}{suffix}_ci_high"] = interval["ci_high"]
            row[f"n_{unit}s"] = len(next(iter(clustered.values())))
        summary.append(row)
        for index, name in enumerate(LABEL_NAMES):
            values = {seed: run.metrics["per_class"][index]["f1"] for seed, run in by_seed.items()}
            stats = seed_summary(values)
            per_class.append({"dataset": dataset, "method": label, "class": name,
                              "support": next(iter(by_seed.values())).metrics["per_class"][index]["support"],
                              "f1_mean": stats["mean"], "f1_std": stats["std"]})
    datasets = sorted({run.dataset for run in runs})
    for dataset in datasets:
        labels = {label for d, label in groups if d == dataset}
        for treatment, reference in contrasts_for(matrix, labels):
            left, right = groups[(dataset, treatment)], groups[(dataset, reference)]
            classes = tuple(next(iter(left.values())).metrics["supported_classes"])
            functions = metric_functions(classes)
            for unit in CLUSTER_UNITS:
                cluster_of = maps[dataset][unit].__getitem__
                left_stats = {s: cluster_statistics(r.report["predictions"], cluster_of) for s, r in left.items()}
                right_stats = {s: cluster_statistics(r.report["predictions"], cluster_of) for s, r in right.items()}
                for name in BOOTSTRAP_METRICS:
                    deltas = paired_seed_deltas({s: flat_metrics(r.metrics)[name] for s, r in left.items()},
                                                {s: flat_metrics(r.metrics)[name] for s, r in right.items()})
                    interval = cluster_bootstrap(left_stats, right_stats, functions[name])
                    contrasts.append({"dataset": dataset, "treatment": treatment, "reference": reference,
                                      "metric": name, "cluster_unit": unit,
                                      "cluster_role": "primary" if unit == PRIMARY_CLUSTER_UNIT else "secondary",
                                      "delta_mean": deltas["mean"], "delta_std": deltas["std"],
                                      "seeds_positive": deltas["n_positive"], "seeds_negative": deltas["n_negative"],
                                      "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
                                      "n_clusters": interval["n_clusters"]})
    return {"summary": summary, "per_class": per_class, "contrasts": contrasts, "groups": groups}


def cost_rows(runs: list[EvaluatedRun]) -> list[dict]:
    rows = []
    profiles: dict[str, dict | None] = {}
    for run in runs:
        assert run.step.run is not None
        if run.dataset not in profiles:
            profiles[run.dataset] = latest_profile(run.dataset)
        timing = run.report.get("timing") or {}
        extraction = extraction_seconds_per_frame(profiles[run.dataset], run.step.run.method.visual)
        classification = timing.get("seconds_per_window")
        training = json.loads((run.step.run.run_dir / "metrics.json").read_text(encoding="utf-8")).get("timing", {})
        rows.append({"dataset": run.dataset, "method": run.label, "seed": run.seed,
                     "extraction_s_per_frame": extraction, "classification_s_per_window": classification,
                     "end_to_end_s_per_frame": None if extraction is None or classification is None
                     else extraction + classification,
                     "train_seconds": training.get("train_seconds"),
                     "train_peak_memory_bytes": training.get("peak_memory_bytes")})
    return rows


def _timelines(runs: list[EvaluatedRun], output: Path) -> list[str]:
    seed = min(SEEDS)
    chosen = [run for run in runs if run.seed == seed]
    if not chosen:
        return []
    written = []
    by_dataset: dict[str, list[EvaluatedRun]] = {}
    for run in chosen:
        by_dataset.setdefault(run.dataset, []).append(run)
    for dataset, members in sorted(by_dataset.items()):
        reference = members[0].report["predictions"]
        with_falls = sorted({row["video_id"] for row in reference if row["label"] == 1})[:TIMELINE_VIDEOS]
        for index, video_id in enumerate(with_falls):
            rows = []
            truth: list[int] = []
            for run in sorted(members, key=lambda item: item.label):
                video = sorted((row for row in run.report["predictions"] if row["video_id"] == video_id),
                               key=lambda row: row["k_end"])
                truth = [row["label"] for row in video]
                ends = np.array([row["k_end"] for row in video], dtype=np.int64)
                predicted = np.array([row["prediction"] for row in video], dtype=np.int64)
                alarms = detect_alarms_for_video(video_id, ends, predicted, BASELINE_A_ALARM_PROTOCOL)
                rows.append((run.label, predicted.tolist(), [alarm.trigger_k for alarm in alarms]))
            name = f"timeline-{dataset}-{index + 1}.svg"
            (output / name).write_text(timeline(f"{dataset}: {video_id} (semente {seed})", LABEL_NAMES,
                                                truth, rows, TARGET_FPS), encoding="utf-8")
            written.append(name)
    return written


def write_markdown(path: Path, matrix: str, aggregated: dict, costs: list[dict]) -> None:
    lines = [f"# Estudo comparativo GateFall 2026 — matriz `{matrix}`", "",
             "Média ± desvio entre sementes. IC 95% por bootstrap de sujeitos (primário) e de gravações "
             "(secundário); com poucos sujeitos de teste o IC por sujeito é grosseiro.", "",
             "| dataset | método | macro-F1 (suporte) | IC sujeito | IC gravação | sens. evento | durante a queda | "
             "só após | FA/h | latência mediana (detectados) | perdidos | sujeitos / gravações |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in aggregated["summary"]:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {_fmt(row['macro_f1_supported_mean'])} ± "
            f"{_fmt(row['macro_f1_supported_std'])} | [{_fmt(row['macro_f1_supported_ci_low'])}, "
            f"{_fmt(row['macro_f1_supported_ci_high'])}] | [{_fmt(row['macro_f1_supported_recording_ci_low'])}, "
            f"{_fmt(row['macro_f1_supported_recording_ci_high'])}] | {_fmt(row['event_sensitivity_mean'])} | "
            f"{_fmt(row['sensitivity_during_fall_mean'])} | {_fmt(row['sensitivity_only_after_fall_mean'])} | "
            f"{_fmt(row['false_alarms_per_hour_mean'], 2)} | {_fmt(row['latency_detected_median_s_mean'], 2)} | "
            f"{_fmt(row['missed_events_mean'], 1)} | {row['n_subjects']} / {row['n_recordings']} |")
    if aggregated["contrasts"]:
        lines += ["", "## Contrastes pareados (tratamento − referência)", "",
                  "| dataset | tratamento | referência | métrica | Δ média ± dp | sementes +/− | IC 95% | cluster |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in aggregated["contrasts"]:
            lines.append(f"| {row['dataset']} | {row['treatment']} | {row['reference']} | {row['metric']} | "
                         f"{_fmt(row['delta_mean'])} ± {_fmt(row['delta_std'])} | "
                         f"{row['seeds_positive']}/{row['seeds_negative']} | "
                         f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] | {row['cluster_unit']} "
                         f"({row['cluster_role']}, {row['n_clusters']}) |")
    measured = [row for row in costs if row["end_to_end_s_per_frame"] is not None]
    lines += ["", "## Custo", "", f"{len(measured)}/{len(costs)} runs com perfil de extração medido; "
              "detalhes em `cost.csv`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_cluster_maps(runs: list[EvaluatedRun], frames_by_dataset: dict[str, pd.DataFrame] | None) -> dict:
    maps = {}
    for dataset in sorted({run.dataset for run in runs}):
        evaluated = {row["video_id"] for row in next(run for run in runs if run.dataset == dataset).report["predictions"]}
        frames = (frames_by_dataset or {}).get(dataset)
        maps[dataset] = cluster_maps(dataset, evaluated, load_frames(dataset) if frames is None else frames)
    return maps


def build_report(steps: list[Step], matrix: str, output_root: Path = REPORT_DIR,
                 frames_by_dataset: dict[str, pd.DataFrame] | None = None) -> Path:
    runs = collect(steps, matrix)
    maps = resolve_cluster_maps(runs, frames_by_dataset)
    output = output_root / f"{matrix}-{utc_now().replace(':', '')}"
    output.mkdir(parents=True, exist_ok=False)
    aggregated = aggregate(runs, matrix, maps)
    _write_csv(output / "runs.csv", [{"dataset": run.dataset, "method": run.label, "seed": run.seed,
                                      "origin": run.origin, "equivalence_sha256": run.equivalence_sha256,
                                      **flat_metrics(run.metrics)} for run in runs])
    _write_csv(output / "summary.csv", aggregated["summary"])
    _write_csv(output / "per_class.csv", aggregated["per_class"])
    if aggregated["contrasts"]:
        _write_csv(output / "contrasts.csv", aggregated["contrasts"])
    costs = cost_rows(runs)
    _write_csv(output / "cost.csv", costs)
    write_markdown(output / "summary.md", matrix, aggregated, costs)
    figures = []
    for dataset in sorted({run.dataset for run in runs}):
        rows = [(label, [flat_metrics(run.metrics)["macro_f1_supported"] or 0.0 for run in by_seed.values()])
                for (d, label), by_seed in sorted(aggregated["groups"].items()) if d == dataset]
        (output / f"macro-f1-{dataset}.svg").write_text(
            seed_dot_plot(f"{dataset}: macro-F1 (classes com suporte), por semente", rows), encoding="utf-8")
        points = [(run.label, flat_metrics(run.metrics)["false_alarms_per_hour"] or 0.0,
                   flat_metrics(run.metrics)["event_sensitivity"] or 0.0) for run in runs if run.dataset == dataset]
        (output / f"sensitivity-fa-{dataset}.svg").write_text(
            scatter(f"{dataset}: sensibilidade por evento × FA/h", "falsos alarmes por hora",
                    "sensibilidade por evento", points), encoding="utf-8")
        figures += [f"macro-f1-{dataset}.svg", f"sensitivity-fa-{dataset}.svg"]
    figures += _timelines(runs, output)
    manifest = {"study": STUDY_ID, "matrix": matrix, "created_utc": utc_now(), "code": git_state(),
                "protocol_sha256": protocol_sha256(), "figures": figures,
                "inputs": [{"key": run.step.key, "evaluation_sha256": run.evaluation_sha256,
                            "provenance_sha256": run.provenance_sha256, "origin": run.origin,
                            "equivalence_sha256": run.equivalence_sha256} for run in runs]}
    with (output / "report-manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output
