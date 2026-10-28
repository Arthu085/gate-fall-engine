"""Atestado explícito de equivalência entre pacote/alvos históricos e os atuais de um run de movimento.

O atestado embute os quatro manifestos, para ser reconferido sem os artefatos originais. Só as
diferenças de metadados em `BUNDLE_ALLOWLIST` e `TARGETS_ALLOWLIST` são admitidas; qualquer
outra diferença, arquivo ausente, extra ou com bytes divergentes recusa o atestado.
"""

import hashlib
import json
from pathlib import Path

from gatefall.experiments.physics.data import ALL_SPLITS, load_bundle
from gatefall.hashing import sha256_file
from gatefall.provenance import git_state, utc_now

SCHEMA = "motion-equivalence-v1"
ROLES = ("original_bundle", "original_targets", "bundle", "targets")
MANIFEST_NAMES = {"bundle": "manifest.json", "targets": "motion-manifest.json"}
ADDED_SOURCES = ("pose_extractor", "dinov3_extractor")
BUNDLE_ALLOWLIST = (
    "code_provenance: pode diferir (revisão do código de preparo)",
    "splits: pode ser adicionado se igual a train/val/test, exigidas quando o campo não existia",
    "videos[].bytes: pode ser adicionado; conferido contra o tamanho real ao atestar",
    "sources.pose_extractor, sources.dinov3_extractor: podem ser adicionados se ausentes no original",
)
TARGETS_ALLOWLIST = (
    "code_provenance: pode diferir (revisão do código de preparo)",
    "bundle_sha256: remapeado; cada manifesto de alvos aponta para o seu pacote",
)
GUARANTEE = (
    "Ao atestar, todo HDF5 dos pacotes e todo NPZ dos alvos, original e atual, foi relido e tem o "
    "SHA-256 registrado, sem arquivos ausentes ou extras, e os registros por vídeo (identidade, "
    "partição, K, caminho, hash, pose de origem) são iguais e na mesma ordem. Fontes, grade, "
    "preparação e parâmetros dos alvos são iguais; só diferem os metadados da allowlist. O "
    "atestado não cobre o código ou o ambiente do treino histórico nem suas avaliações."
)


class EquivalenceError(ValueError):
    pass


def _short(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _compare_keys(prefix: str, original: dict, current: dict, rejected: list[str],
                  skip: tuple[str, ...] = ()) -> None:
    for key in sorted(set(original) | set(current)):
        if key in skip:
            continue
        if key not in original or key not in current:
            rejected.append(f"{prefix}{key}: presente só no {'atual' if key in current else 'original'}")
        elif original[key] != current[key]:
            rejected.append(f"{prefix}{key}: original {_short(original[key])} diverge do atual "
                            f"{_short(current[key])}")


def _video_differences(original: object, current: object, rejected: list[str]) -> int:
    if not isinstance(original, list) or not isinstance(current, list) or len(original) != len(current):
        rejected.append("videos: número de vídeos diverge")
        return 0
    for name, entries in (("original", original), ("atual", current)):
        for field in ("video_id", "path"):
            values = [entry.get(field) for entry in entries]
            if len(set(values)) != len(values):
                rejected.append(f"videos: {field} repetido no {name}; mapeamento ambíguo")
    added = 0
    for index, (old, new) in enumerate(zip(original, current)):
        extra = set(new) - set(old)
        if extra - {"bytes"} or set(old) - set(new) or any(old[field] != new[field] for field in old):
            rejected.append(f"videos[{index}] ({old.get('video_id')}): identidade, partição, caminho, "
                            "hash ou K diverge")
        added += "bytes" in extra
    return added


def bundle_differences(original: dict, current: dict) -> tuple[list[str], list[str]]:
    """(diferenças admitidas, diferenças recusadas) entre manifestos de pacote."""
    allowed: list[str] = []
    rejected: list[str] = []
    if original.get("code_provenance") != current.get("code_provenance"):
        allowed.append("code_provenance: diverge")
    if "splits" not in original and "splits" in current:
        present = {entry.get("split") for entry in original.get("videos", [])}
        if present == set(ALL_SPLITS) and current["splits"] == list(ALL_SPLITS):
            allowed.append("splits: adicionado")
        else:
            rejected.append(f"splits: {_short(current['splits'])} não corresponde às partições do original")
    added = _video_differences(original.get("videos"), current.get("videos"), rejected)
    if added:
        allowed.append(f"videos[].bytes: adicionado em {added} vídeo(s)")
    old_sources, new_sources = original.get("sources"), current.get("sources")
    if not isinstance(old_sources, dict) or not isinstance(new_sources, dict):
        rejected.append("sources: ausente ou malformado")
    else:
        fresh = tuple(key for key in ADDED_SOURCES if key not in old_sources and key in new_sources)
        allowed += [f"sources.{key}: adicionado" for key in fresh]
        _compare_keys("sources.", old_sources, new_sources, rejected, fresh)
    skip = ("code_provenance", "videos", "sources", *(("splits",) if "splits" not in original else ()))
    _compare_keys("", original, current, rejected, skip)
    return allowed, rejected


def targets_differences(original: dict, current: dict) -> tuple[list[str], list[str]]:
    allowed: list[str] = []
    rejected: list[str] = []
    if original.get("code_provenance") != current.get("code_provenance"):
        allowed.append("code_provenance: diverge")
    if original.get("bundle_sha256") != current.get("bundle_sha256"):
        allowed.append("bundle_sha256: remapeado")
    _compare_keys("", original, current, rejected, ("code_provenance", "bundle_sha256"))
    return allowed, rejected


def _link_problems(manifests: dict[str, dict], hashes: dict[str, str]) -> list[str]:
    problems = []
    for bundle, targets in (("original_bundle", "original_targets"), ("bundle", "targets")):
        if manifests[targets].get("bundle_sha256") != hashes[bundle]:
            problems.append(f"{targets}: bundle_sha256 não aponta para {bundle}; alvos de outro pacote")
    current = manifests["bundle"]
    for name, value in (("dataset", current.get("dataset")), ("visual_kind", current.get("visual_kind"))):
        if manifests["original_bundle"].get(name) != value:
            problems.append(f"{name} diverge entre pacote original e atual")
    return problems


def _differences(manifests: dict[str, dict]) -> tuple[dict[str, list[str]], list[str]]:
    bundle_allowed, bundle_rejected = bundle_differences(manifests["original_bundle"], manifests["bundle"])
    targets_allowed, targets_rejected = targets_differences(manifests["original_targets"], manifests["targets"])
    rejected = [f"pacote: {item}" for item in bundle_rejected] + [f"alvos: {item}" for item in targets_rejected]
    return {"bundle": bundle_allowed, "targets": targets_allowed}, rejected


def _physical_problems(root: Path, manifest: dict, manifest_name: str) -> list[str]:
    """Relê do disco cada arquivo listado e recusa ausentes, extras ou com bytes divergentes."""
    root = root.resolve()
    listed = {str(entry["path"]): entry for entry in manifest.get("videos", [])}
    found = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} - {manifest_name}
    problems = [f"{root}: {len(paths)} arquivo(s) {label}, ex. {paths[:3]}"
                for label, paths in (("ausentes", sorted(set(listed) - found)),
                                     ("não listados", sorted(found - set(listed)))) if paths]
    for relative, entry in listed.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            problems.append(f"{root}: caminho fora do diretório: {relative}")
        elif path.is_file():
            if sha256_file(path) != entry.get("sha256"):
                problems.append(f"{path}: SHA-256 real diverge do registrado")
            if "bytes" in entry and path.stat().st_size != entry["bytes"]:
                problems.append(f"{path}: tamanho real diverge de bytes registrado")
    return problems


def _manifest_path(role: str, root: Path) -> Path:
    return root / MANIFEST_NAMES[role.removeprefix("original_")]


def attest(original_bundle: Path, original_targets: Path, bundle: Path, targets: Path, output: Path) -> Path:
    """Relê os quatro diretórios e grava o atestado em `output` (novo); recusa qualquer evidência parcial."""
    if output.exists():
        raise FileExistsError(f"atestado já existe: {output}")
    roots = dict(zip(ROLES, (original_bundle, original_targets, bundle, targets)))
    missing = [str(_manifest_path(role, root)) for role, root in roots.items()
               if not _manifest_path(role, root).is_file()]
    if missing:
        raise EquivalenceError(f"manifesto(s) ausente(s): {missing}")
    texts = {role: _manifest_path(role, root).read_bytes().decode("utf-8") for role, root in roots.items()}
    hashes = {role: hashlib.sha256(text.encode("utf-8")).hexdigest() for role, text in texts.items()}
    manifests = {role: json.loads(text) for role, text in texts.items()}
    allowed, problems = _differences(manifests)
    problems += _link_problems(manifests, hashes)
    for role, root in roots.items():
        problems += _physical_problems(root, manifests[role], _manifest_path(role, root).name)
    if not problems:
        from gatefall.experiments.motion.data import load_targets

        try:
            load_targets(targets, bundle, *load_bundle(bundle))
        except ValueError as error:
            problems.append(f"alvos atuais não carregam sobre o pacote atual: {error}")
    if problems:
        raise EquivalenceError("equivalência recusada:\n  " + "\n  ".join(problems))
    record = {
        "schema": SCHEMA, "created_utc": utc_now(), "code": git_state(),
        "dataset": manifests["bundle"]["dataset"], "visual_kind": manifests["bundle"]["visual_kind"],
        "paths": {role: str(root) for role, root in roots.items()},
        "rehashed_files": {role: len(manifests[role]["videos"]) for role in ROLES},
        "allowlist": {"bundle": list(BUNDLE_ALLOWLIST), "targets": list(TARGETS_ALLOWLIST)},
        "allowed_differences": allowed, "guarantee": GUARANTEE,
        "manifests": {role: {"sha256": hashes[role], "text": texts[role]} for role in ROLES},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return output


def verify_attestation(path: Path, bundle: Path, targets: Path | None = None) -> dict:
    """Reconfere o atestado sem os artefatos originais e o liga ao pacote (e alvos) atuais em disco."""
    if not path.is_file():
        raise EquivalenceError(f"{path}: atestado ausente")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA or set(record.get("manifests", {})) != set(ROLES):
        raise EquivalenceError(f"{path}: schema ou papéis do atestado inválidos")
    texts = {role: record["manifests"][role]["text"] for role in ROLES}
    hashes = {role: hashlib.sha256(text.encode("utf-8")).hexdigest() for role, text in texts.items()}
    problems = [f"{role}: SHA-256 do manifesto embutido diverge do registrado"
                for role in ROLES if hashes[role] != record["manifests"][role]["sha256"]]
    if problems:
        raise EquivalenceError(f"{path}: " + "; ".join(problems))
    manifests = {role: json.loads(text) for role, text in texts.items()}
    allowed, problems = _differences(manifests)
    problems += _link_problems(manifests, hashes)
    if allowed != record.get("allowed_differences"):
        problems.append("resumo de diferenças admitidas não confere com os manifestos embutidos")
    if (record.get("dataset"), record.get("visual_kind")) != (manifests["bundle"].get("dataset"),
                                                              manifests["bundle"].get("visual_kind")):
        problems.append("dataset ou fonte visual do atestado diverge do pacote")
    current = {"bundle": bundle / "manifest.json"}
    if targets is not None:
        current["targets"] = targets / "motion-manifest.json"
    for role, manifest in current.items():
        if not manifest.is_file() or sha256_file(manifest) != hashes[role]:
            problems.append(f"{manifest}: ausente ou diferente do {role} atestado")
    if problems:
        raise EquivalenceError(f"{path}: " + "; ".join(problems))
    return {"path": str(path), "sha256": sha256_file(path), "dataset": record["dataset"],
            "visual_kind": record["visual_kind"],
            "original_manifest_sha256": hashes["original_bundle"],
            "original_targets_sha256": hashes["original_targets"],
            "current_manifest_sha256": hashes["bundle"], "current_targets_sha256": hashes["targets"]}


def attested_run(config: dict, path: Path, bundle: Path, targets: Path | None = None) -> dict:
    """Resumo do atestado quando ele liga exatamente o pacote e os alvos registrados no run aos atuais."""
    summary = verify_attestation(path, bundle, targets)
    motion = config.get("motion") or {}
    expected = {"manifest_sha256": (config.get("manifest_sha256"), summary["original_manifest_sha256"]),
                "targets_sha256": (motion.get("targets_sha256"), summary["original_targets_sha256"]),
                "dataset": (config.get("dataset"), summary["dataset"]),
                "visual_kind": (config.get("visual_kind"), summary["visual_kind"])}
    divergent = [name for name, (recorded, attested) in expected.items() if recorded != attested]
    if divergent:
        raise EquivalenceError(f"{path}: run não corresponde aos artefatos originais atestados: {divergent}")
    return summary
