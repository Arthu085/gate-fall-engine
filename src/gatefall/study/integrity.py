"""Integridade, proveniência e identidade das dependências antes de reusar um passo do plano.

`done` só quando as saídas conferem; `pending` quando nada foi produzido ou falta trabalho
retomável; `invalid` quando há saída existente que não confere e nunca será sobrescrita.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gatefall.data.staged.build import manifest_problems, staged_paths, timegrid_problems
from gatefall.data.staged.labels import PROVENANCE_FILENAME, verify_staged_labels
from gatefall.data.staged.sequences import SequenceError, archive_stage_state
from gatefall.data.staged.specs import StagedDatasetSpec
from gatefall.hashing import sha256_file

if TYPE_CHECKING:
    from gatefall.study.plan import RunSpec

CONFIG_IDENTITY_FIELDS = ("arm", "fusion", "seed", "epochs", "batch_size", "checkpoint_policy",
                          "augment", "order_weight")
RUN_ARTIFACTS = (("config.json", "config_sha256"), ("stats.json", "stats_sha256"),
                 ("checkpoint.pt", "checkpoint_sha256"), ("last.pt", "last_checkpoint_sha256"))


@dataclass(frozen=True)
class Verdict:
    status: str
    reasons: tuple[str, ...] = ()


Verifier = Callable[[], Verdict]


def _from_problems(problems: list[str]) -> Verdict:
    return Verdict("invalid", tuple(problems)) if problems else Verdict("done")


def _guarded(verify: Verifier) -> Verdict:
    try:
        return verify()
    except (OSError, ValueError, KeyError, TypeError) as error:
        return Verdict("invalid", (f"{type(error).__name__}: {error}",))


def _produced(outputs: tuple[Path, ...], problems: Callable[[], list[str]]) -> Verdict:
    if not any(output.exists() for output in outputs):
        return Verdict("pending")
    return _guarded(lambda: _from_problems(problems()))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mismatches(path: Path, recorded: dict, expected: dict[str, object]) -> list[str]:
    return [f"{path}: {key} registrado {recorded.get(key)!r} diverge do atual {value!r}"
            for key, value in expected.items() if recorded.get(key) != value]


def config_mismatches(run: RunSpec, config: dict) -> list[str]:
    identity = run.identity()
    problems = [name for name in CONFIG_IDENTITY_FIELDS if config.get(name) != identity[name]]
    motion = config.get("motion")
    if (motion or {}).get("recipe") != run.recipe:
        problems.append("motion_recipe")
    return problems


def upfall_staging(archives_dir: Path, raw_dir: Path) -> Verdict:
    """Confere cada .zip de câmera contra seu destino; `raw_dir` existir não basta."""
    archives = sorted(archives_dir.glob("*.zip")) if archives_dir.is_dir() else []
    if not archives:
        if raw_dir.exists():
            return Verdict("done", (f"{archives_dir}: sem .zip para conferir; o manifesto confere as "
                                    "fontes anotadas",))
        return Verdict("pending", (f"{archives_dir}: nenhum .zip baixado",))
    unstaged: list[str] = []
    problems: list[str] = []
    for archive in archives:
        try:
            state, found = archive_stage_state(archive, raw_dir)
        except (SequenceError, zipfile.BadZipFile, OSError) as error:
            state, found = "invalid", [f"{archive}: {error}"]
        problems += found
        if state == "unstaged":
            unstaged.append(archive.name)
    if problems:
        return Verdict("invalid", tuple(problems))
    if unstaged:
        return Verdict("pending", (f"{len(unstaged)} arquivo(s) ainda não preparado(s), ex. {unstaged[:3]}",))
    return Verdict("done")


def staged_labels(spec: StagedDatasetSpec) -> Verdict:
    return _produced((spec.labels_dir / PROVENANCE_FILENAME,), lambda: verify_staged_labels(spec))


def staged_manifest(spec: StagedDatasetSpec) -> Verdict:
    paths = staged_paths(spec)
    return _produced((paths.manifest, paths.identities, paths.provenance), lambda: manifest_problems(spec))


def staged_timegrid(spec: StagedDatasetSpec) -> Verdict:
    paths = staged_paths(spec)
    return _produced((paths.frames, paths.alignment), lambda: timegrid_problems(spec))


def staged_audit(spec: StagedDatasetSpec) -> Verdict:
    """Auditoria desatualizada é refeita (`pending`); auditoria reprovada sobre a grade atual é `invalid`."""
    paths = staged_paths(spec)
    if not paths.coverage.exists():
        return Verdict("pending")

    def verify() -> Verdict:
        problems = timegrid_problems(spec)
        if problems:
            return Verdict("invalid", tuple(problems))
        report = _read_json(paths.coverage)
        stale = _mismatches(paths.coverage, report, {"frames_sha256": sha256_file(paths.frames),
                                                     "alignment_sha256": sha256_file(paths.alignment)})
        if stale:
            return Verdict("pending", tuple(stale))
        if report.get("passed") is not True:
            return Verdict("invalid", (f"{paths.coverage}: auditoria reprovada: {report.get('problems')}",))
        return Verdict("done")

    return _guarded(verify)


def bundle_problems(root: Path, frames_path: Path | None) -> list[str]:
    from gatefall.experiments.physics import FEATURE_SCHEMA, SCHEMA_VERSION

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return [f"{manifest_path}: ausente em pacote existente"]
    manifest = _read_json(manifest_path)
    problems = _mismatches(manifest_path, manifest, {"schema_version": SCHEMA_VERSION,
                                                     "feature_schema": FEATURE_SCHEMA})
    listed = {str(entry["path"]): entry for entry in manifest.get("videos", [])}
    missing = [path for path in listed if not (root / path).is_file()]
    resized = [path for path, entry in listed.items() if path not in missing and "bytes" in entry
               and (root / path).stat().st_size != entry["bytes"]]
    extra = sorted({path.relative_to(root).as_posix() for path in (root / "videos").glob("*")} - set(listed))
    for label, paths in (("ausentes", missing), ("com tamanho divergente", resized), ("não listados", extra)):
        if paths:
            problems.append(f"{root}: {len(paths)} arquivo(s) {label}, ex. {paths[:3]}")
    if frames_path is not None:
        problems += _mismatches(manifest_path, manifest.get("sources", {}),
                                {"frames_sha256": sha256_file(frames_path) if frames_path.exists() else None})
    return problems


def bundle(root: Path, frames_path: Path | None) -> Verdict:
    return _produced((root,), lambda: bundle_problems(root, frames_path))


def motion_targets(root: Path, bundle_root: Path) -> Verdict:
    from gatefall.experiments.motion import SCHEMA

    def problems() -> list[str]:
        meta_path = root / "motion-manifest.json"
        if not meta_path.is_file():
            return [f"{meta_path}: ausente em alvos existentes"]
        meta = _read_json(meta_path)
        found = _mismatches(meta_path, meta, {"schema": SCHEMA,
                                              "bundle_sha256": sha256_file(bundle_root / "manifest.json")})
        missing = [entry["path"] for entry in meta.get("videos", []) if not (root / entry["path"]).is_file()]
        if missing:
            found.append(f"{root}: {len(missing)} alvo(s) ausente(s), ex. {missing[:3]}")
        return found

    return _produced((root,), problems)


def input_problems(run: RunSpec, config: dict) -> tuple[list[str], dict | None]:
    """Pacote e alvos do run: os atuais, ou ligados aos atuais por um atestado de equivalência.

    O atestado só é consultado quando o run não usa os artefatos atuais; devolve seu resumo.
    """
    from gatefall.experiments.motion.equivalence import EquivalenceError, attested_run

    bundle_manifest = run.bundle / "manifest.json"
    bundle_current = bundle_manifest.is_file() and config.get("manifest_sha256") == sha256_file(bundle_manifest)
    targets_current = True
    if run.targets is not None:
        targets_manifest = run.targets / "motion-manifest.json"
        recorded = (config.get("motion") or {}).get("targets_sha256")
        targets_current = targets_manifest.is_file() and recorded == sha256_file(targets_manifest)
    if bundle_current and targets_current:
        return [], None
    if run.equivalence is not None and run.equivalence.is_file():
        try:
            return [], attested_run(config, run.equivalence, run.bundle, run.targets)
        except EquivalenceError as error:
            return [f"{run.run_dir}: atestado não liga o run aos artefatos atuais: {error}"], None
    problems = [] if bundle_current else [f"{run.run_dir}: pacote {run.bundle} ausente ou alterado desde o treino"]
    if not targets_current:
        problems.append(f"{run.run_dir}: alvos {run.targets} ausentes ou alterados desde o treino")
    return problems, None


def run_problems(run: RunSpec) -> list[str]:
    """Identidade planejada, pacote e alvos atuais (ou atestados), hashes dos artefatos e proveniência."""
    from gatefall.study.registry import RUN_PROVENANCE, StudyExecutionError, verify_run_provenance

    required = ["config.json", "metrics.json", *(name for name, _ in RUN_ARTIFACTS[1:])]
    missing = [name for name in required if not (run.run_dir / name).is_file()]
    if missing:
        return [f"{run.run_dir}: run incompleto, sem {missing}; o treino recusa diretório existente"]
    config = _read_json(run.run_dir / "config.json")
    metrics = _read_json(run.run_dir / "metrics.json")
    problems = [f"{run.run_dir}: campo divergente da identidade planejada: {name}"
                for name in config_mismatches(run, config)]
    problems += input_problems(run, config)[0]
    altered = [name for name, field in RUN_ARTIFACTS if sha256_file(run.run_dir / name) != metrics.get(field)]
    if altered:
        problems.append(f"{run.run_dir}: artefatos não conferem com metrics.json: {altered}")
    if not problems and (run.run_dir / RUN_PROVENANCE).exists():
        try:
            verify_run_provenance(run)
        except StudyExecutionError as error:
            problems.append(str(error))
    return problems


def trained_run(run: RunSpec) -> Verdict:
    return _produced((run.run_dir,), lambda: run_problems(run))


def evaluation_problems(run: RunSpec, output: Path, target_bundle: Path | None) -> list[str]:
    problems = run_problems(run)
    if problems:
        return problems
    config = _read_json(run.run_dir / "config.json")
    metrics = _read_json(run.run_dir / "metrics.json")
    expected: dict[str, object] = {
        "split": "test", "selection_performed": False,
        "checkpoint_sha256": metrics["checkpoint_sha256"], "config_sha256": metrics["config_sha256"],
        "manifest_sha256": config["manifest_sha256"],
    }
    if target_bundle is not None:
        target_manifest = target_bundle / "manifest.json"
        expected |= {"adaptation": "none",
                     "target_manifest_sha256": sha256_file(target_manifest) if target_manifest.is_file() else None}
    report = _read_json(output)
    problems = _mismatches(output, report, expected)
    attested = input_problems(run, config)[1]
    if attested is not None:
        # Avaliação de run atestado precisa ter sido feita sobre o pacote atual com este atestado.
        recorded = report.get("equivalence") or {}
        problems += _mismatches(output, {f"equivalence.{key}": value for key, value in recorded.items()},
                                {"equivalence.sha256": attested["sha256"],
                                 "equivalence.current_manifest_sha256": attested["current_manifest_sha256"]})
    return problems


def evaluation(run: RunSpec, output: Path, target_bundle: Path | None = None) -> Verdict:
    return _produced((output,), lambda: evaluation_problems(run, output, target_bundle))
