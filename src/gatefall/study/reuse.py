"""Auditoria de runs externos do experimento de movimento antes de qualquer reuso."""

import json
from dataclasses import replace
from pathlib import Path

from gatefall.experiments.motion import SCHEMA as MOTION_SCHEMA
from gatefall.experiments.physics import FEATURE_SCHEMA, SCHEMA_VERSION
from gatefall.hashing import sha256_file
from gatefall.study.integrity import config_mismatches, input_problems
from gatefall.study.plan import Step

REQUIRED_FILES = ("config.json", "metrics.json", "stats.json", "checkpoint.pt", "last.pt")


def audit_candidate(step: Step, candidate: Path) -> tuple[list[str], dict | None]:
    """Motivos de incompatibilidade com o passo planejado (vazio = reutilizável) e o atestado usado."""
    run = step.run
    assert run is not None and run.recipe is not None
    missing = [name for name in REQUIRED_FILES if not (candidate / name).is_file()]
    if missing:
        return [f"arquivos ausentes: {missing}"], None
    config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    reasons = [f"campo divergente: {name}" for name in config_mismatches(run, config)]
    if config.get("schema_version") != SCHEMA_VERSION or config.get("feature_schema") != FEATURE_SCHEMA:
        reasons.append("schema de features divergente")
    if (config.get("motion") or {}).get("schema") != MOTION_SCHEMA:
        reasons.append("schema motion divergente")
    if config.get("dataset") != run.dataset or config.get("visual_kind") != run.method.visual:
        reasons.append("dataset ou fonte visual divergente")
    inputs, attested = input_problems(replace(run, run_dir=candidate), config)
    reasons += inputs
    for name, field in (("config.json", "config_sha256"), ("stats.json", "stats_sha256"),
                        ("checkpoint.pt", "checkpoint_sha256"), ("last.pt", "last_checkpoint_sha256")):
        if sha256_file(candidate / name) != metrics.get(field):
            reasons.append(f"hash de {name} não confere com metrics.json")
    return reasons, attested


def audit_external_runs(steps: list[Step], source: Path) -> dict:
    """Procura, em `source`, runs com o layout importado (<raiz>/motion-<braço>/seed-<n>/<receita>)."""
    planned = [step for step in steps if step.stage == "train" and step.run is not None
               and step.run.recipe is not None]
    entries = []
    for step in planned:
        assert step.run is not None
        relative = step.run.run_dir.relative_to(step.run.run_dir.parents[3])
        candidate = source / relative
        reasons, attested = audit_candidate(step, candidate) if candidate.exists() else ([], None)
        status = "absent" if not candidate.exists() else "incompatible" if reasons else "compatible"
        historical = sorted(path.name for path in candidate.glob("evaluation-*.json"))
        entries.append({"key": step.key, "candidate": str(candidate), "target": str(step.run.run_dir),
                        "status": status, "reasons": reasons,
                        "inputs": None if status != "compatible" else "attested" if attested else "current",
                        "equivalence_sha256": attested["sha256"] if attested and not reasons else None,
                        "historical_evaluations": historical})
    counts = {status: sum(entry["status"] == status for entry in entries)
              for status in ("compatible", "incompatible", "absent")}
    counts["compatible_attested"] = sum(entry["inputs"] == "attested" for entry in entries)
    return {"source": str(source), "planned": len(planned), "counts": counts, "entries": entries,
            "policy": "reuso só de runs compatíveis, copiando para o destino planejado apenas "
                      f"{', '.join(REQUIRED_FILES)}; o executor revalida hashes e grava "
                      "study-provenance.json com origin=reused-verified. Avaliações históricas não são "
                      "reusadas em runs atestados: a avaliação é refeita sobre o pacote atual."}
