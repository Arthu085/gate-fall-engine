"""Respostas futuras de treino, separadas das entradas causais e verificadas."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from gatefall.config import TARGET_FPS, TRAIN_STRIDE
from gatefall.experiments.motion import JOINTS, SCHEMA
from gatefall.experiments.physics.data import code_provenance, load_bundle, write_json
from gatefall.experiments.physics.kinematics import corrected_pose_features
from gatefall.hashing import sha256_file
from gatefall.pose.loading import PoseArrays, load_pose, pose_path
from gatefall.runs import validate_local_run_dir


def validate_options(horizons: list[int], confidence: float) -> None:
    if (not horizons or any(type(h) is not int or h < 1 for h in horizons)
            or horizons != sorted(set(horizons))):
        raise ValueError("horizontes devem ser inteiros positivos, únicos e crescentes")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confiança deve estar em [0,1]")


def build_targets(pose: PoseArrays, horizons: list[int], confidence: float) -> dict[str, np.ndarray]:
    validate_options(horizons, confidence)
    if pose.width <= 0 or pose.height <= 0 or pose.keypoints.shape != (pose.k, 17, 3):
        raise ValueError("dimensões inválidas na pose original")
    points = pose.keypoints[:, JOINTS, :]
    observed = (pose.person_found[:, None] & (points[..., 2] >= confidence)
                & np.isfinite(points).all(axis=-1))
    current = np.where(observed[..., None], points[..., :2], 0).astype(np.float32)
    current /= math.hypot(pose.width, pose.height)
    shape = (pose.k, len(horizons), len(JOINTS), 2)
    future = np.zeros(shape, dtype=np.float32)
    velocity = np.zeros_like(future)
    valid = np.zeros(shape[:-1], dtype=bool)
    for j, horizon in enumerate(horizons):
        if horizon >= pose.k:
            continue
        mask = observed[:-horizon] & observed[horizon:]
        valid[:-horizon, j] = mask
        future[:-horizon, j] = np.where(mask[..., None], current[horizon:], 0)
        velocity[:-horizon, j] = np.where(
            mask[..., None], (current[horizon:] - current[:-horizon]) / (horizon / TARGET_FPS), 0)
    return {"current": current, "future": future, "velocity": velocity, "valid": valid}


def prepare(bundle: Path, pose_root: Path, output: Path,
            horizons: list[int] | None = None, confidence: float = .5) -> Path:
    horizons = [1, 3] if horizons is None else horizons
    validate_options(horizons, confidence)
    bundle, pose_root, output = bundle.resolve(), pose_root.resolve(), output.resolve()
    validate_local_run_dir(output)
    if output.exists() or bundle == output or bundle in output.parents:
        raise ValueError("destino deve ser novo e externo ao bundle")
    manifest, videos = load_bundle(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".motion-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "targets"
        stage.mkdir()
        entries = []
        for entry in manifest["videos"]:
            if entry["split"] != "train":
                continue
            identity = entry["video_id"]
            raw = pose_path(identity, pose_root=pose_root).resolve()
            if pose_root not in raw.parents:
                raise ValueError("identidade aponta para fora da raiz de poses")
            digest = sha256_file(raw)
            source_hashes = manifest.get("sources", {}).get("files", {}).get(identity, {})
            if digest not in source_hashes.values():
                raise ValueError(f"{identity}: pose original não consta dos hashes do bundle")
            with h5py.File(raw, "r") as handle:
                if handle.attrs.get("video_id") != identity or handle.attrs.get("K") != entry["k"]:
                    raise ValueError("identidade/grade da pose original divergente")
            pose = load_pose(identity, pose_root=pose_root)
            reconstructed = corrected_pose_features(pose, manifest["preparation_config"]["max_gap_seconds"])
            if not np.allclose(reconstructed, videos[identity]["pose"], rtol=0, atol=1e-6):
                raise ValueError(f"{identity}: pose original não reproduz as entradas do bundle")
            arrays = build_targets(pose, horizons, confidence)
            name = hashlib.sha256(identity.encode()).hexdigest() + ".npz"
            np.savez_compressed(stage / name, current=arrays["current"], future=arrays["future"],
                                velocity=arrays["velocity"], valid=arrays["valid"])
            entries.append({"video_id": identity, "split": "train", "k": pose.k, "path": name,
                            "sha256": sha256_file(stage / name), "pose_sha256": digest})
        write_json(stage / "motion-manifest.json", {
            "schema": SCHEMA, "bundle_sha256": sha256_file(bundle / "manifest.json"),
            "target_fps": TARGET_FPS, "horizons": horizons, "confidence": confidence,
            "joints": list(JOINTS), "coordinates": "image_diagonal",
            "code_provenance": code_provenance(), "videos": entries,
        })
        load_targets(stage, bundle, manifest, videos)
        if output.exists():
            raise FileExistsError(output)
        os.replace(stage, output)
    return output


def load_targets(root: Path, bundle: Path, manifest: dict, videos: dict) -> tuple[dict, dict, dict]:
    root = root.resolve()
    meta = json.loads((root / "motion-manifest.json").read_text(encoding="utf-8"))
    validate_options(meta["horizons"], meta["confidence"])
    if (meta.get("schema") != SCHEMA or meta.get("target_fps") != TARGET_FPS
            or meta.get("bundle_sha256") != sha256_file(bundle / "manifest.json")
            or meta.get("joints") != list(JOINTS) or meta.get("coordinates") != "image_diagonal"):
        raise ValueError("pacote auxiliar incompatível com bundle/esquema")
    train_entries = {e["video_id"]: e for e in manifest["videos"] if e["split"] == "train"}
    targets = {}
    counts = np.zeros((len(manifest["label_names"]), len(meta["horizons"])), dtype=np.int64)
    possible = np.zeros_like(counts)
    for entry in meta["videos"]:
        identity = entry["video_id"]
        path = (root / entry["path"]).resolve()
        if (identity not in train_entries or identity in targets or entry["split"] != "train"
                or entry["k"] != train_entries[identity]["k"] or root not in path.parents
                or sha256_file(path) != entry["sha256"]):
            raise ValueError("identidade, caminho, split ou hash auxiliar inválido")
        if entry["pose_sha256"] not in manifest["sources"]["files"][identity].values():
            raise ValueError("hash da pose auxiliar não pertence ao bundle")
        k = entry["k"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        shapes = {"current": (k, 4, 2), "future": (k, len(meta["horizons"]), 4, 2),
                  "velocity": (k, len(meta["horizons"]), 4, 2),
                  "valid": (k, len(meta["horizons"]), 4)}
        if set(arrays) != set(shapes):
            raise ValueError("campos auxiliares inválidos")
        for key, shape in shapes.items():
            expected = np.dtype(bool) if key == "valid" else np.dtype(np.float32)
            if arrays[key].shape != shape or arrays[key].dtype != expected or not np.isfinite(arrays[key]).all():
                raise ValueError("shape, tipo ou valores auxiliares inválidos")
        for j, h in enumerate(meta["horizons"]):
            valid = arrays["valid"][:, j]
            if valid[max(0, k-h):].any():
                raise ValueError("alvo cruza final do vídeo")
            residual = arrays["future"][:, j] - arrays["current"] - h / TARGET_FPS * arrays["velocity"][:, j]
            if not np.allclose(residual[valid], 0, atol=1e-6):
                raise ValueError("alvos violam identidade cinemática")
        mask = arrays["valid"][..., None]
        if np.any(np.where(mask, 0, arrays["future"])) or np.any(np.where(mask, 0, arrays["velocity"])):
            raise ValueError("alvos inválidos devem estar zerados")
        labels = videos[identity]["labels"]
        for end in range(0, k, TRAIN_STRIDE):
            label = int(labels[end])
            if label != -1:
                counts[label] += arrays["valid"][end].sum(axis=-1)
                possible[label] += 4
        targets[identity] = arrays
    if set(targets) != set(train_entries) or counts.sum() == 0:
        raise ValueError("pacote incompleto ou sem supervisão válida nas janelas de treino")
    coverage = {"horizons": meta["horizons"], "label_names": manifest["label_names"],
                "valid_joint_targets_by_class": counts.tolist(),
                "possible_joint_targets_by_class": possible.tolist(),
                "total_valid": int(counts.sum()), "total_possible": int(possible.sum())}
    return meta, targets, coverage
