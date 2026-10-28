"""Proveniência por run, ledger de execução e executor explícito do plano."""

import hashlib
import json
import os
import platform
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from gatefall.hashing import sha256_file
from gatefall.provenance import git_state, utc_now
from gatefall.study import STUDY_DIR_NAME, STUDY_ID, STUDY_SCHEMA
from gatefall.study.plan import STAGES, RunSpec, Step
from gatefall.study.protocol import canonical_json, protocol_sha256

RUN_PROVENANCE = "study-provenance.json"
EXECUTION_DIR = Path("runs/local") / STUDY_DIR_NAME / "execution"
WEIGHTS_CACHE = Path("runs/local") / STUDY_DIR_NAME / "weights-sha256.json"
CONFIG_IDENTITY_FIELDS = ("arm", "fusion", "seed", "epochs", "batch_size", "checkpoint_policy",
                          "augment", "order_weight")

Runner = Callable[[Sequence[str]], int]


class StudyExecutionError(RuntimeError):
    pass


def weights_paths() -> dict[str, Path]:
    from gatefall.dinov3.backbone import resolve_weights_path
    from gatefall.pose.extract import WEIGHTS_DIR
    from gatefall.pose.smoke import DEFAULT_MODEL
    from gatefall.sam3.runtime import resolve_checkpoint_path

    return {"pose": WEIGHTS_DIR / DEFAULT_MODEL, "dinov3": resolve_weights_path(None),
            "sam3": resolve_checkpoint_path(None)}


def weights_fingerprint(paths: dict[str, Path] | None = None, cache: Path = WEIGHTS_CACHE) -> dict:
    """SHA-256 dos pesos congelados, em cache por (caminho, tamanho, mtime) para não reler GBs."""
    try:
        known = json.loads(cache.read_text(encoding="utf-8"))
    except FileNotFoundError:
        known = {}
    result: dict[str, dict | None] = {}
    for name, path in (paths or weights_paths()).items():
        if not path.is_file():
            result[name] = None
            continue
        stat = path.stat()
        key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        if key not in known:
            known[key] = sha256_file(path)
        result[name] = {"path": str(path), "bytes": stat.st_size, "sha256": known[key]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def environment() -> dict:
    import torch

    devices = ([torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
               if torch.cuda.is_available() else [])
    return {"host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda_devices": devices, "cpu_count": os.cpu_count()}


def split_identity_sha256(manifest: dict) -> str:
    pairs = sorted((entry["video_id"], entry["split"]) for entry in manifest["videos"])
    return hashlib.sha256(canonical_json(pairs)).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def config_mismatches(run: RunSpec, config: dict) -> list[str]:
    identity = run.identity()
    problems = [name for name in CONFIG_IDENTITY_FIELDS if config.get(name) != identity[name]]
    motion = config.get("motion")
    if (motion or {}).get("recipe") != run.recipe:
        problems.append("motion_recipe")
    return problems


def run_provenance(run: RunSpec, origin: str, timing: dict | None, weights: dict) -> dict:
    config = _read_json(run.run_dir / "config.json")
    metrics = _read_json(run.run_dir / "metrics.json")
    manifest = _read_json(run.bundle / "manifest.json")
    mismatches = config_mismatches(run, config)
    if mismatches or config.get("manifest_sha256") != sha256_file(run.bundle / "manifest.json"):
        raise StudyExecutionError(f"{run.run_dir}: run não corresponde à identidade planejada {mismatches}")
    artifacts = {name: metrics[field] for name, field in (
        ("config.json", "config_sha256"), ("stats.json", "stats_sha256"),
        ("checkpoint.pt", "checkpoint_sha256"), ("last.pt", "last_checkpoint_sha256"))}
    altered = [name for name, digest in artifacts.items() if sha256_file(run.run_dir / name) != digest]
    if altered:
        raise StudyExecutionError(f"{run.run_dir}: artefatos não conferem com metrics.json: {altered}")
    return {
        "study": STUDY_ID, "schema": STUDY_SCHEMA, "created_utc": utc_now(), "origin": origin,
        "identity": run.identity(), "identity_sha256": run.identity_sha256(),
        "protocol_sha256": protocol_sha256(), "code": git_state(),
        "bundle": {"path": str(run.bundle), "manifest_sha256": config["manifest_sha256"],
                   "dataset": manifest["dataset"], "split_identity_sha256": split_identity_sha256(manifest),
                   "feature_schema": manifest["feature_schema"], "sources": {
                       name: value for name, value in manifest["sources"].items() if name != "files"}},
        "artifacts": artifacts,
        "selected_epoch": metrics.get("selected_epoch"),
        "training_timing": metrics.get("timing"), "wall_timing": timing,
        "weights": weights, "environment": environment(),
    }


def write_run_provenance(run: RunSpec, origin: str, timing: dict | None, weights: dict) -> Path:
    path = run.run_dir / RUN_PROVENANCE
    with path.open("x", encoding="utf-8") as handle:
        json.dump(run_provenance(run, origin, timing, weights), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def verify_run_provenance(run: RunSpec) -> dict:
    path = run.run_dir / RUN_PROVENANCE
    record = _read_json(path)
    if record.get("identity_sha256") != run.identity_sha256():
        raise StudyExecutionError(f"{path}: identidade de run divergente do plano")
    for name, digest in record["artifacts"].items():
        if sha256_file(run.run_dir / name) != digest:
            raise StudyExecutionError(f"{path}: artefato alterado após o registro: {name}")
    return record


def step_status(step: Step) -> str:
    if not step.outputs:
        return "always"
    return "done" if all(output.exists() for output in step.outputs) else "pending"


def render_commands(steps: list[Step], stage: str | None = None) -> str:
    lines: list[str] = []
    for current in STAGES:
        if stage not in (None, current):
            continue
        members = [step for step in steps if step.stage == current]
        if not members:
            continue
        lines.append(f"# ==== estágio {current}: {len(members)} passo(s)")
        for step in members:
            lines.append(f"# [{step.key}] recursos: {step.resources}")
            if step.requires:
                lines.append(f"#   depende de: {', '.join(step.requires)}")
            if step.outputs:
                lines.append(f"#   reuso: pula se existir {', '.join(map(str, step.outputs))}")
            lines.append(shlex.join(step.command) if step.command
                         else f"# pré-requisito manual: {', '.join(map(str, step.outputs))}")
        lines.append("")
    return "\n".join(lines)


def _subprocess_runner(command: Sequence[str]) -> int:
    return subprocess.run(list(command), check=False).returncode


def execute(steps: list[Step], *, log_dir: Path = EXECUTION_DIR, runner: Runner = _subprocess_runner,
            allow_dirty: bool = False, weights: dict | None = None) -> Path:
    """Executa passos pendentes em ordem; reusa só saídas verificadas e para na primeira falha."""
    code = git_state()
    if code["git_dirty"] and not allow_dirty:
        raise StudyExecutionError("árvore Git com alterações; registre-as ou passe --allow-dirty")
    weights = weights_fingerprint() if weights is None else weights
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"execution-{utc_now().replace(':', '')}-{uuid.uuid4().hex[:8]}.jsonl"
    with log_path.open("x", encoding="utf-8") as log:
        def record(entry: dict) -> None:
            log.write(json.dumps(entry, ensure_ascii=False) + "\n"); log.flush()

        record({"event": "start", "study": STUDY_ID, "protocol_sha256": protocol_sha256(), "code": code,
                "weights": weights, "environment": environment(), "utc": utc_now(),
                "steps": [step.key for step in steps]})
        for step in steps:
            status = step_status(step)
            if step.manual:
                if status != "done":
                    record({"event": "missing", "key": step.key})
                    raise StudyExecutionError(f"pré-requisito ausente para {step.key}: {step.outputs}")
                record({"event": "present", "key": step.key}); continue
            if status == "done":
                if step.stage == "train" and step.run is not None:
                    if (step.run.run_dir / RUN_PROVENANCE).exists():
                        verify_run_provenance(step.run)
                    else:
                        write_run_provenance(step.run, "reused-verified", None, weights)
                record({"event": "reused", "key": step.key}); continue
            started = time.perf_counter(); started_utc = utc_now()
            returncode = runner(step.command)
            timing = {"started_utc": started_utc, "seconds": time.perf_counter() - started}
            record({"event": "finished", "key": step.key, "command": list(step.command),
                    "returncode": returncode, **timing})
            if returncode != 0:
                raise StudyExecutionError(f"{step.key} falhou com código {returncode}; log: {log_path}")
            if step.outputs and step_status(step) != "done":
                raise StudyExecutionError(f"{step.key} terminou sem as saídas declaradas")
            if step.stage == "train" and step.run is not None:
                write_run_provenance(step.run, "executed", timing, weights)
        record({"event": "complete", "utc": utc_now()})
    return log_path
