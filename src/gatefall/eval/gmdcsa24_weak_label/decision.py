"""Decisão GO/NO-GO do piloto GMDCSA24 com rótulos fracos, só sobre `val`.

Aplica sem mudança o
[critério de decisão](docs/analysis/gmdcsa24-weak-label-protocol.md#criterio-de-decisao):
qualquer falha de W1, W2 ou V2–V5 em T ou S torna o piloto INVÁLIDO; com elas
cumpridas, T contra o controle decide pelas condições G1–G4 do protocolo
verificado (GO exploratório ou NO-GO), e S, com G1–G4 calculadas e relatadas,
só rebaixa um GO de T a GO frágil quando a média de Δ `macro_f1_restricted`
de S é <= 0. A comparação é pareada por seed: Δ = tratamento − controle.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import cast

from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL

GO = "GO exploratório"
GO_FRAGILE = "GO frágil"
NO_GO = "NO-GO"
INVALID = "INVÁLIDO"

TREATMENTS = ("T", "S")
VALIDITY_CONDITIONS = ("W1", "W2", "V2", "V3", "V4", "V5")

# Limites de G1–G4 da tabela "Critério de GO/NO-GO" do protocolo verificado
# (docs/analysis/gmdcsa24-pilot-protocol.md), mantidos pelo protocolo com
# rótulos fracos.
G1_MIN_MEAN_DELTA = 0.015
G1_MIN_POSITIVE_SEEDS = 4
G4_MIN_MEAN_DELTA = 0.0
S_FRAGILE_MAX_MEAN_DELTA = 0.0

# F1 da classe `fall` no `f1_by_class` do metrics.json (índice como texto).
FALL_CLASS = str(BASELINE_A_ALARM_PROTOCOL.fall_label)


@dataclass(frozen=True)
class SeedMetrics:
    """As quatro medidas de `val` de um run que entram em G1–G4."""

    macro_f1_restricted: float
    fall_f1: float
    n_missed_events: int
    n_false_alarms: int


@dataclass(frozen=True)
class PairedSeed:
    seed: int
    treatment: SeedMetrics
    control: SeedMetrics


def coverage_problems(seeds: Sequence[int], expected: Sequence[int], label: str) -> list[str]:
    """Exatamente as seeds do controle, cada uma uma vez."""
    counts = Counter(seeds)
    problems = [f"{label}: seed {seed} repetida" for seed, n in sorted(counts.items()) if n > 1]
    problems.extend(f"{label}: seed {seed} ausente" for seed in sorted(set(expected) - set(counts)))
    problems.extend(
        f"{label}: seed {seed} fora das seeds do controle {sorted(expected)}"
        for seed in sorted(set(counts) - set(expected))
    )
    return problems


def control_problems(pairs: Mapping[str, Sequence[PairedSeed]]) -> list[str]:
    """T e S comparados com o mesmo controle em cada seed."""
    controls = {treatment: {pair.seed: pair.control for pair in pairs[treatment]} for treatment in TREATMENTS}
    return [
        f"seed {seed}: controle de T difere do controle de S"
        for seed in sorted(set(controls["T"]) & set(controls["S"]))
        if controls["T"][seed] != controls["S"][seed]
    ]


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def go_conditions(pairs: Sequence[PairedSeed]) -> dict[str, dict[str, object]]:
    ordered = sorted(pairs, key=lambda pair: pair.seed)
    macro = {
        pair.seed: pair.treatment.macro_f1_restricted - pair.control.macro_f1_restricted
        for pair in ordered
    }
    fall = {pair.seed: pair.treatment.fall_f1 - pair.control.fall_f1 for pair in ordered}
    macro_mean = mean(list(macro.values()))
    positive = sum(delta > 0 for delta in macro.values())
    fall_mean = mean(list(fall.values()))
    sums = {
        field: {
            "treatment": sum(getattr(pair.treatment, field) for pair in ordered),
            "control": sum(getattr(pair.control, field) for pair in ordered),
        }
        for field in ("n_missed_events", "n_false_alarms")
    }
    return {
        "G1": {
            "passed": macro_mean >= G1_MIN_MEAN_DELTA and positive >= G1_MIN_POSITIVE_SEEDS,
            "mean_delta_macro_f1_restricted": macro_mean,
            "positive_seeds": positive,
            "delta_by_seed": {str(seed): delta for seed, delta in macro.items()},
        },
        "G2": {
            "passed": sums["n_missed_events"]["treatment"] <= sums["n_missed_events"]["control"],
            **{f"{side}_n_missed_events": value for side, value in sums["n_missed_events"].items()},
        },
        "G3": {
            "passed": sums["n_false_alarms"]["treatment"] <= sums["n_false_alarms"]["control"],
            **{f"{side}_n_false_alarms": value for side, value in sums["n_false_alarms"].items()},
        },
        "G4": {
            "passed": fall_mean >= G4_MIN_MEAN_DELTA,
            "mean_delta_fall_f1": fall_mean,
            "delta_by_seed": {str(seed): delta for seed, delta in fall.items()},
        },
    }


def decide(
    validity: Mapping[str, Mapping[str, Sequence[str]]],
    pairs: Mapping[str, Sequence[PairedSeed]] | None,
    seeds: Sequence[int],
) -> dict[str, object]:
    """Resultado do critério de decisão. Cobertura incompleta de seeds ou
    controle inconsistente entre T e S levantam ValueError: não há decisão."""
    failed = {
        treatment: {condition: list(problems) for condition, problems in conditions.items() if problems}
        for treatment, conditions in validity.items()
    }
    if set(validity) != set(TREATMENTS) or any(
        set(conditions) != set(VALIDITY_CONDITIONS) for conditions in validity.values()
    ):
        raise ValueError(f"validade deve trazer {list(VALIDITY_CONDITIONS)} para T e S")
    base: dict[str, object] = {"split": "val", "seeds": sorted(seeds)}
    if any(failed.values()):
        return {**base, "decision": INVALID, "validity_failures": failed}
    if pairs is None or set(pairs) != set(TREATMENTS):
        raise ValueError("decisão válida exige as seeds pareadas de T e S")
    problems = [
        problem
        for treatment in TREATMENTS
        for problem in coverage_problems([pair.seed for pair in pairs[treatment]], seeds, treatment)
    ]
    problems.extend(control_problems(pairs))
    if problems:
        raise ValueError("; ".join(problems))
    conditions = {treatment: go_conditions(pairs[treatment]) for treatment in TREATMENTS}
    t_go = all(condition["passed"] for condition in conditions["T"].values())
    s_mean = cast(float, conditions["S"]["G1"]["mean_delta_macro_f1_restricted"])
    if not t_go:
        decision = NO_GO
    elif s_mean <= S_FRAGILE_MAX_MEAN_DELTA:
        decision = GO_FRAGILE
    else:
        decision = GO
    per_seed = {
        treatment: {
            str(pair.seed): {"treatment": asdict(pair.treatment), "control": asdict(pair.control)}
            for pair in sorted(pairs[treatment], key=lambda pair: pair.seed)
        }
        for treatment in TREATMENTS
    }
    return {**base, "decision": decision, "conditions": conditions, "per_seed": per_seed}
