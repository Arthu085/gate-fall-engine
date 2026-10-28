"""Diferenças pareadas por semente e bootstrap por cluster (sujeito ou gravação)."""

import math
from collections.abc import Callable

import numpy as np

from gatefall.study.metrics import macro_f1
from gatefall.study.protocol import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL

ClusterStats = dict[str, dict]
Metric = Callable[[dict], float | None]


def seed_summary(values: dict[int, float | None]) -> dict:
    finite = [value for value in values.values() if value is not None and math.isfinite(value)]
    return {"n_seeds": len(values), "n_defined": len(finite),
            "mean": float(np.mean(finite)) if finite else None,
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "per_seed": {str(seed): value for seed, value in sorted(values.items())}}


def paired_seed_deltas(treatment: dict[int, float | None], reference: dict[int, float | None]) -> dict:
    if set(treatment) != set(reference):
        raise ValueError("diferença pareada exige o mesmo conjunto de sementes nos dois métodos")
    deltas: dict[int, float | None] = {}
    for seed in sorted(treatment):
        left, right = treatment[seed], reference[seed]
        deltas[seed] = None if left is None or right is None else left - right
    summary = seed_summary(deltas)
    defined = [value for value in deltas.values() if value is not None]
    summary.update(n_positive=sum(value > 0 for value in defined), n_negative=sum(value < 0 for value in defined))
    return summary


def pooled(stats: ClusterStats, clusters: list[str]) -> dict:
    total = {"confusion": 0, "events": 0, "detected": 0, "during": 0, "false_alarms": 0, "hours": 0.0}
    for cluster in clusters:
        for name in total:
            total[name] = total[name] + stats[cluster][name]
    return total


def metric_functions(classes: tuple[int, ...]) -> dict[str, Metric]:
    return {
        "macro_f1_supported": lambda total: macro_f1(np.asarray(total["confusion"]), classes),
        "event_sensitivity": lambda total: total["detected"] / total["events"] if total["events"] else None,
        "sensitivity_during_fall": lambda total: total["during"] / total["events"] if total["events"] else None,
        "false_alarms_per_hour": lambda total: total["false_alarms"] / total["hours"] if total["hours"] else None,
    }


def _seed_mean(stats_by_seed: dict[int, ClusterStats], clusters: list[str], metric: Metric) -> float | None:
    values = [metric(pooled(stats, clusters)) for _, stats in sorted(stats_by_seed.items())]
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if len(defined) == len(values) else None


def cluster_bootstrap(treatment: dict[int, ClusterStats], reference: dict[int, ClusterStats] | None,
                      metric: Metric, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                      level: float = CONFIDENCE_LEVEL) -> dict:
    """IC percentil da média entre sementes; o mesmo sorteio de clusters vale para os dois métodos e todas as sementes."""
    if reference is not None and set(treatment) != set(reference):
        raise ValueError("bootstrap pareado exige as mesmas sementes")
    cluster_sets = {tuple(sorted(stats)) for stats in treatment.values()}
    if reference is not None:
        cluster_sets |= {tuple(sorted(stats)) for stats in reference.values()}
    if len(cluster_sets) != 1:
        raise ValueError("bootstrap exige o mesmo conjunto de clusters em todos os runs")
    clusters = list(next(iter(cluster_sets)))

    def estimate(sample: list[str]) -> float | None:
        value = _seed_mean(treatment, sample, metric)
        if reference is None or value is None:
            return value
        base = _seed_mean(reference, sample, metric)
        return None if base is None else value - base

    point = estimate(clusters)
    rng = np.random.default_rng(seed)
    draws = []
    undefined = 0
    for _ in range(resamples):
        sample = [clusters[index] for index in rng.integers(0, len(clusters), len(clusters))]
        value = estimate(sample)
        if value is None:
            undefined += 1
        else:
            draws.append(value)
    alpha = (1 - level) / 2
    low, high = (np.quantile(draws, [alpha, 1 - alpha]).tolist() if draws else (None, None))
    return {"estimate": point, "ci_low": low, "ci_high": high, "level": level, "resamples": resamples,
            "undefined_resamples": undefined, "n_clusters": len(clusters), "seed": seed}
