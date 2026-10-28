"""Auditoria de runs externos do experimento de movimento antes de qualquer reuso."""

import json
from pathlib import Path

from gatefall.experiments.motion import SCHEMA as MOTION_SCHEMA
from gatefall.experiments.physics import FEATURE_SCHEMA, SCHEMA_VERSION
from gatefall.hashing import sha256_file
from gatefall.study.plan import Step
from gatefall.study.registry import config_mismatches

REQUIRED_FILES = ("config.json", "metrics.json", "stats.json", "checkpoint.pt", "last.pt")


def audit_candidate(step: Step, candidate: Path) -> list[str]:
    """Motivos de incompatibilidade de um run externo com o passo planejado (vazio = reutilizável)."""
    run = step.run
    assert run is not None and run.recipe is not None
    missing = [name for name in REQUIRED_FILES if not (candidate / name).is_file()]
    if missing:
        return [f"arquivos ausentes: {missing}"]
    config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    reasons = [f"campo divergente: {name}" for name in config_mismatches(run, config)]
    if config.get("schema_version") != SCHEMA_VERSION or config.get("feature_schema") != FEATURE_SCHEMA:
        reasons.append("schema de features divergente")
    if (config.get("motion") or {}).get("schema") != MOTION_SCHEMA:
        reasons.append("schema motion divergente")
    if config.get("dataset") != run.dataset or config.get("visual_kind") != run.method.visual:
        reasons.append("dataset ou fonte visual divergente")
    manifest = run.bundle / "manifest.json"
    if not manifest.is_file():
        reasons.append(f"pacote local ausente: {run.bundle}")
    elif config.get("manifest_sha256") != sha256_file(manifest):
        reasons.append("pacote diferente do pacote local do estudo (manifest_sha256)")
    targets = run.targets / "motion-manifest.json" if run.targets is not None else None
    if targets is None or not targets.is_file():
        reasons.append(f"alvos locais ausentes: {run.targets}")
    elif (config.get("motion") or {}).get("targets_sha256") != sha256_file(targets):
        reasons.append("alvos de movimento diferentes dos locais (targets_sha256)")
    for name, field in (("config.json", "config_sha256"), ("stats.json", "stats_sha256"),
                        ("checkpoint.pt", "checkpoint_sha256"), ("last.pt", "last_checkpoint_sha256")):
        if sha256_file(candidate / name) != metrics.get(field):
            reasons.append(f"hash de {name} não confere com metrics.json")
    return reasons


def audit_external_runs(steps: list[Step], source: Path) -> dict:
    """Procura, em `source`, runs com o layout importado (<raiz>/motion-<braço>/seed-<n>/<receita>)."""
    planned = [step for step in steps if step.stage == "train" and step.run is not None
               and step.run.recipe is not None]
    entries = []
    for step in planned:
        assert step.run is not None
        relative = step.run.run_dir.relative_to(step.run.run_dir.parents[3])
        candidate = source / relative
        reasons = audit_candidate(step, candidate) if candidate.exists() else []
        status = "absent" if not candidate.exists() else "incompatible" if reasons else "compatible"
        entries.append({"key": step.key, "candidate": str(candidate), "target": str(step.run.run_dir),
                        "status": status, "reasons": reasons})
    counts = {status: sum(entry["status"] == status for entry in entries)
              for status in ("compatible", "incompatible", "absent")}
    return {"source": str(source), "planned": len(planned), "counts": counts, "entries": entries,
            "policy": "reuso só de runs compatíveis copiados para o destino planejado; "
                      "o executor revalida hashes e grava study-provenance.json com origin=reused-verified"}
