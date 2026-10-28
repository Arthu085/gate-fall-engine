"""Pacotes offline por vídeo com proveniência e falhas pareadas de treino."""

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from gatefall.config import NUM_CLASSES, TARGET_FPS
from gatefall.experiments.physics import EVIDENCE_NAMES, FEATURE_SCHEMA, SCHEMA_VERSION
from gatefall.experiments.physics.consistency import evidence_for_video
from gatefall.experiments.physics.kinematics import POSE_DIM, corrected_pose_features
from gatefall.hashing import sha256_file
from gatefall.pose.loading import PoseArrays
from gatefall.runs import validate_local_run_dir

VISUAL_DIMS = {"none": 0, "dinov3": 1536, "sam3": 10}
ALL_SPLITS = ("train", "val", "test")
POSE_EXTRACTOR_ATTRS = ("model_name", "ultralytics_version", "tracker_name", "target_fps")


def code_provenance() -> dict:
    package = Path(__file__).resolve().parents[2]
    files = {str(path.relative_to(package)): sha256_file(path)
             for path in sorted(package.rglob("*.py"))}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=package,
                                            stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_revision": revision, "git_dirty": dirty, "python_sources_sha256": files}


@dataclass
class RawVideo:
    video_id: str
    split: str
    labels: np.ndarray
    pose: PoseArrays
    visual: np.ndarray
    visual_quality: np.ndarray


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_attrs(path: Path, identity: str, k: int, source_split: str) -> dict:
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
    for name, expected in (("video_id", identity), ("K", k), ("split", source_split)):
        if attrs.get(name) != expected:
            raise ValueError(f"{path}: identidade/grade de origem divergente ({name})")
    return attrs


def validate_preparation(max_gap_seconds: float, outage_probability: float, outage_frames: int) -> None:
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds deve ser positivo e finito")
    if not math.isfinite(outage_probability) or not 0 <= outage_probability <= 1:
        raise ValueError("outage_probability deve estar em [0,1]")
    if outage_frames < 1:
        raise ValueError("outage_frames deve ser >=1")


def outage_mask(k: int, seed: int, key: str, probability: float, duration: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    mask = np.zeros(k, dtype=bool)
    for start in np.flatnonzero(rng.random(k) < probability):
        mask[start:min(k, start + duration)] = True
    return mask


def build_variants(
    record: RawVideo, visual_kind: str, seed: int, max_gap_seconds: float,
    outage_probability: float, outage_frames: int,
) -> dict[str, np.ndarray]:
    validate_preparation(max_gap_seconds, outage_probability, outage_frames)
    pose, visual = record.pose, record.visual
    if (visual_kind not in VISUAL_DIMS or visual.shape != (pose.k, VISUAL_DIMS[visual_kind])
            or record.visual_quality.shape != (pose.k,) or record.labels.shape != (pose.k,)):
        raise ValueError(f"{record.video_id}: shape de fonte incompatível")
    if (not np.isfinite(visual).all() or not np.isfinite(record.visual_quality).all()
            or not np.isin(record.labels, np.arange(-1, NUM_CLASSES)).all()
            or record.labels.dtype.kind not in "iu"):
        raise ValueError(f"{record.video_id}: fonte não finita ou rótulos inválidos")
    if record.split not in ("train", "val", "test"):
        raise ValueError("partição desconhecida")
    present = (visual[:, 0] == 1 if visual_kind == "sam3"
               else np.full(pose.k, visual_kind == "dinov3", dtype=bool))
    if visual_kind == "sam3" and not np.isin(visual[:, 0], (0, 1)).all():
        raise ValueError("presença SAM 3 deve ser binária")
    qv = record.visual_quality * present
    clean_pose = corrected_pose_features(pose, max_gap_seconds)
    evidence = evidence_for_video(pose, visual, qv, present, visual_kind, max_gap_seconds)
    result = {"labels": record.labels.astype(np.int64), "pose": clean_pose,
              "visual": visual.astype(np.float32), "evidence": evidence}
    if record.split != "train":
        return result
    p_changed = outage_mask(pose.k, seed, record.video_id + ":pose", outage_probability, outage_frames)
    p_changed &= pose.person_found
    v_changed = outage_mask(pose.k, seed, record.video_id + ":visual", outage_probability, outage_frames)
    v_changed &= present
    bad_points, bad_boxes = pose.keypoints.copy(), pose.bbox.copy()
    bad_points[p_changed] = 0; bad_boxes[p_changed] = 0
    bad_pose = PoseArrays(bad_points, bad_boxes, pose.person_found & ~p_changed,
                         pose.k, pose.width, pose.height)
    bad_visual = visual.copy(); bad_visual[v_changed] = 0
    bad_qv = qv.copy(); bad_qv[v_changed] = 0
    result.update(
        pose_corrupt=corrected_pose_features(bad_pose, max_gap_seconds),
        visual_corrupt=bad_visual.astype(np.float32),
        evidence_pose_corrupt=evidence_for_video(
            bad_pose, visual, qv, present, visual_kind, max_gap_seconds),
        evidence_visual_corrupt=evidence_for_video(
            pose, bad_visual, bad_qv, present & ~v_changed, visual_kind, max_gap_seconds),
        pose_changed=p_changed, visual_changed=v_changed,
    )
    return result


def prepare_records(
    records: Iterable[RawVideo], output: Path, *, dataset: str, visual_kind: str,
    label_names: tuple[str, ...], sources: dict, seed: int = 42,
    max_gap_seconds: float = .5, outage_probability: float = .05, outage_frames: int = 3,
    required_splits: tuple[str, ...] = ALL_SPLITS,
) -> Path:
    validate_preparation(max_gap_seconds, outage_probability, outage_frames)
    if not required_splits or not set(required_splits) <= set(ALL_SPLITS):
        raise ValueError("partições exigidas inválidas")
    output = output.resolve()
    validate_local_run_dir(output)
    if output.exists():
        raise FileExistsError(f"destino já existe; escolha outro: {output}")
    if visual_kind not in VISUAL_DIMS or len(label_names) != NUM_CLASSES:
        raise ValueError("modalidade ou nomes de classes inválidos")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".physics-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"; stage.mkdir()
        (stage / "videos").mkdir()
        videos: list[dict] = []
        seen: set[str] = set()
        for record in records:
            if record.video_id in seen:
                raise ValueError(f"vídeo duplicado/partições sobrepostas: {record.video_id}")
            seen.add(record.video_id)
            arrays = build_variants(record, visual_kind, seed, max_gap_seconds, outage_probability, outage_frames)
            relative = f"videos/{hashlib.sha256(record.video_id.encode()).hexdigest()}.h5"
            path = stage / relative
            with h5py.File(path, "w") as handle:
                for name, array in arrays.items():
                    handle.create_dataset(name, data=array)
                handle.attrs["video_id"] = record.video_id
                handle.attrs["split"] = record.split
            videos.append({"video_id": record.video_id, "split": record.split,
                           "path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                           "k": record.pose.k})
        if {v["split"] for v in videos} != set(required_splits):
            raise ValueError(f"pacote exige vídeos disjuntos em {'/'.join(required_splits)}")
        manifest = {
            "schema_version": SCHEMA_VERSION, "feature_schema": FEATURE_SCHEMA,
            "dataset": dataset, "visual_kind": visual_kind, "visual_dim": VISUAL_DIMS[visual_kind],
            "label_names": label_names, "evidence_names": EVIDENCE_NAMES,
            "target_fps": TARGET_FPS,
            "code_provenance": code_provenance(),
            "preparation_config": {"seed": seed, "max_gap_seconds": max_gap_seconds,
                                   "outage_probability": outage_probability, "outage_frames": outage_frames},
            "sources": sources, "videos": videos,
            "splits": [split for split in ALL_SPLITS if split in required_splits],
        }
        write_json(stage / "manifest.json", manifest)
        load_bundle(stage)
        if output.exists():
            raise FileExistsError(output)
        os.replace(stage, output)
    return output


def load_bundle(root: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("feature_schema") != FEATURE_SCHEMA
            or manifest.get("evidence_names") != list(EVIDENCE_NAMES)
            or manifest.get("target_fps") != TARGET_FPS
            or manifest.get("visual_kind") not in VISUAL_DIMS
            or manifest.get("visual_dim") != VISUAL_DIMS[manifest["visual_kind"]]
            or len(manifest.get("label_names", [])) != NUM_CLASSES):
        raise ValueError("manifest incompatível com o esquema experimental")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    splits: set[str] = set()
    for entry in manifest["videos"]:
        identity, split = entry["video_id"], entry["split"]
        path = (root / entry["path"]).resolve()
        if root not in path.parents or identity in arrays or split not in ALL_SPLITS:
            raise ValueError("caminho, identidade ou partição inválida no pacote")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"hash de vídeo incompatível: {identity}")
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("video_id") != identity or handle.attrs.get("split") != split:
                raise ValueError("identidade HDF5 diverge do manifest")
            data: dict[str, np.ndarray] = {
                str(name): np.asarray(cast(h5py.Dataset, handle[name])[()]) for name in handle}
        shapes = {"labels": (entry["k"],), "pose": (entry["k"], POSE_DIM),
                  "visual": (entry["k"], manifest["visual_dim"]), "evidence": (entry["k"], 8)}
        if split == "train":
            shapes.update(pose_corrupt=shapes["pose"], visual_corrupt=shapes["visual"],
                          evidence_pose_corrupt=shapes["evidence"], evidence_visual_corrupt=shapes["evidence"],
                          pose_changed=shapes["labels"], visual_changed=shapes["labels"])
        if set(data) != set(shapes) or entry["k"] < 1:
            raise ValueError("campos ou comprimento de vídeo incompatíveis")
        for name, shape in shapes.items():
            if data[name].shape != shape or not np.isfinite(data[name]).all():
                raise ValueError(f"{identity}/{name}: shape ou valores inválidos")
            if name.startswith("evidence") and np.any((data[name] < 0) | (data[name] > 1)):
                raise ValueError("evidência fora de [0,1]")
            if name.startswith("evidence") and not np.isin(data[name][:, 4:], (0, 1)).all():
                raise ValueError("validade/presença deve ser binária")
            if name.endswith("changed") and data[name].dtype.kind != "b":
                raise ValueError("máscara de corrupção deve ser booleana")
        if data["labels"].dtype.kind not in "iu" or not np.isin(data["labels"], np.arange(-1, NUM_CLASSES)).all():
            raise ValueError("rótulos inválidos")
        arrays[identity] = data; splits.add(split)
    # Pacotes anteriores ao campo `splits` sempre exigiram as três partições.
    if splits != set(manifest.get("splits", ALL_SPLITS)):
        raise ValueError("partições incompletas")
    return manifest, arrays


def prepare_dataset(dataset: str, visual_kind: str, output: Path, **options) -> Path:
    from gatefall.datasets import get_dataset
    from gatefall.dinov3 import storage as dino_storage
    from gatefall.dinov3.report import find_provenance_divergences
    from gatefall.dinov3.storage import dinov3_path, read_features
    from gatefall.features.quality_extract import current_provenance
    from gatefall.features.quality_storage import validate_existing_file as validate_quality
    from gatefall.features.quality_storage import quality_path, read_quality
    from gatefall.features.shared_le2i import shared_source
    from gatefall.pose.loading import load_pose, pose_path
    from gatefall.sam3.features import collect_sam3_provenance, load_v_t
    from gatefall.sam3.storage import sam3_path, read_sam_score

    adapter = get_dataset(dataset)
    source = shared_source(adapter)
    if visual_kind not in VISUAL_DIMS:
        raise ValueError("modalidade desconhecida")
    frames = adapter.load_frames()
    source_splits = {str(identity): str(group["split"].iloc[0])
                     for identity, group in source.load_frames().groupby("video_id")}
    sources: dict = {"frames_sha256": sha256_file(adapter.frames_path), "files": {}}
    pose_extractor: set[tuple] = set()
    if visual_kind == "sam3":
        sources["sam3_provenance"] = collect_sam3_provenance(source_splits, sam3_root=adapter.sam3_root)
    dino_attrs: dict[str, dict[str, object]] = {}

    def records() -> Iterable[RawVideo]:
        for video_id, group in frames.groupby("video_id", sort=True):
            identity = str(video_id)
            group = group.sort_values("frame_index")
            if group["split"].nunique() != 1 or not np.array_equal(group["frame_index"], np.arange(len(group))):
                raise ValueError(f"grade/partição inválida para {identity}")
            pose_file = pose_path(identity, pose_root=adapter.pose_root)
            pose_attrs = source_attrs(pose_file, identity, len(group), source_splits[identity])
            pose_extractor.add(tuple(str(pose_attrs.get(name)) for name in POSE_EXTRACTOR_ATTRS))
            pose = load_pose(identity, pose_root=adapter.pose_root)
            if pose.k != len(group):
                raise ValueError("pose e grade desalinhadas")
            paths = [pose_file]
            if visual_kind == "sam3":
                path = sam3_path(identity, sam3_root=adapter.sam3_root)
                source_attrs(path, identity, pose.k, source_splits[identity])
                visual = load_v_t(identity, sam3_root=adapter.sam3_root)
                qv = read_sam_score(path); paths.append(path)
            elif visual_kind == "dinov3":
                path = dinov3_path(identity, dinov3_root=adapter.dinov3_root)
                quality_file = quality_path(identity, quality_root=adapter.quality_root)
                attrs = source_attrs(path, identity, pose.k, source_splits[identity])
                dino_attrs[identity] = {name: attrs[name] for name in dino_storage.PROVENANCE_ATTR_NAMES
                                        if name in attrs}
                source_attrs(quality_file, identity, pose.k, source_splits[identity])
                reasons = dino_storage.validate_existing_file(
                    path, expected_k=pose.k, feature_dim=1536,
                    expected_attrs={"target_fps": TARGET_FPS})
                reasons += validate_quality(
                    quality_file, expected_k=pose.k,
                    expected_attrs=current_provenance(pose_source_digest=sha256_file(pose_file)))
                if reasons:
                    raise ValueError(f"{identity}: fontes incompatíveis: {'; '.join(reasons)}")
                visual = read_features(path); qv = read_quality(quality_file)[:, 1]
                paths += [path, quality_file]
            else:
                visual = np.zeros((pose.k, 0), dtype=np.float32); qv = np.zeros(pose.k, dtype=np.float32)
            sources["files"][identity] = {str(p): sha256_file(p) for p in paths}
            yield RawVideo(identity, str(group["split"].iloc[0]), group["label"].to_numpy(), pose, visual, qv)
        divergences = find_provenance_divergences(dino_attrs)
        if divergences:
            raise ValueError("proveniência DINOv3 divergente: " + "; ".join(divergences))
        if len(pose_extractor) != 1:
            raise ValueError(f"extratores de pose divergentes entre vídeos: {sorted(pose_extractor)}")
        sources["pose_extractor"] = dict(zip(POSE_EXTRACTOR_ATTRS, next(iter(pose_extractor))))
        if dino_attrs:
            sources["dinov3_extractor"] = next(iter(dino_attrs.values()))
    return prepare_records(records(), output, dataset=dataset, visual_kind=visual_kind,
                           label_names=adapter.label_names, sources=sources,
                           required_splits=required_splits(dataset), **options)


def required_splits(dataset: str) -> tuple[str, ...]:
    from gatefall.data.staged.specs import STAGED_SPECS

    spec = STAGED_SPECS.get(dataset)
    return ("test",) if spec is not None and spec.role == "external_test" else ALL_SPLITS
