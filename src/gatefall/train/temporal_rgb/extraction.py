"""R3D-18 congelado, controle estático e viabilidade antes da extração offline."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from gatefall.data.video_io import decode_frames
from gatefall.hashing import sha256_array, sha256_file
from gatefall.runs import REPOSITORY_ROOT
from gatefall.train.temporal_rgb.contract import (
    FEATURE_DIM, MODES, WEIGHT_FILENAME, contract, digest, guard_destination, implementation_sha256, setup,
)
from gatefall.train.temporal_rgb.data import Context, align_video, context_record
from gatefall.train.temporal_rgb.storage import cache_path, publish_cache, read_cache, write_json

WEIGHTS = R3D_18_Weights.KINETICS400_V1
MAX_DECODED_BYTES = 768 * 1024**2
MAX_EXTRACTION_HOURS = 24


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def weight_path(root: Path, download: bool) -> Path:
    path = root / "weights" / WEIGHT_FILENAME
    guard_destination(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError("pesos ausentes; execute smoke --download-weights antes")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".weights-", dir=path.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            torch.hub.download_url_to_file(WEIGHTS.url, str(temporary), hash_prefix="b3b3357e", progress=False)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if not sha256_file(path).startswith("b3b3357e"):
        raise ValueError("pesos não correspondem à identidade oficial KINETICS400_V1")
    return path


def load_backbone(path: Path, device: str) -> nn.Module:
    model = r3d_18(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten(1))
    backbone.requires_grad_(False)
    return backbone.eval().to(device)


def embed(model: nn.Module, rgb: np.ndarray, device: str) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != 16 or rgb.shape[-1] != 3:
        raise ValueError("o clipe deve ter formato uint8[16,H,W,3]")
    clip = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
    clip = WEIGHTS.transforms()(clip).unsqueeze(0).to(device)
    with torch.inference_mode():
        result = model(clip).cpu().numpy().astype(np.float32)
    if result.shape != (1, FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError("descritor R3D-18 inválido")
    return result[0]


def smoke(root: Path, device: str, download: bool) -> dict[str, Any]:
    settings = setup(device)
    path = weight_path(root, download)
    model = load_backbone(path, device)
    before = digest({name: sha256_array(value.cpu().numpy()) for name, value in model.state_dict().items()})
    rgb = np.random.default_rng(42).integers(0, 256, (16, 128, 171, 3), dtype=np.uint8)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    first = embed(model, rgb, device)
    start = time.perf_counter()
    second = embed(model, rgb, device)
    elapsed = time.perf_counter() - start
    static = embed(model, np.repeat(rgb[-1:], 16, axis=0), device)
    if not np.array_equal(first, second) or np.array_equal(first, static):
        raise RuntimeError("smoke não confirma repetição determinística ou distinção temporal")
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("backbone não está congelado em eval")
    after = digest({name: sha256_array(value.cpu().numpy()) for name, value in model.state_dict().items()})
    if before != after:
        raise RuntimeError("parâmetros ou buffers do backbone mudaram")
    record = {
        "status": "synthetic_pretrained_smoke_passed", "real_data_trained": False,
        "contract": contract(), "settings": settings, "weight_sha256": sha256_file(path),
        "weight_url": WEIGHTS.url, "preprocessing": repr(WEIGHTS.transforms()),
        "feature_shape": list(first.shape), "bitwise_repeat": True,
        "dynamic_static_distinct": True, "seconds_per_forward": elapsed,
        "backbone_state_unchanged": True, "backbone_state_sha256": before,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "frozen_parameters": sum(p.numel() for p in model.parameters()),
    }
    destination = root / f"smoke_{device}.json"
    guard_destination(destination)
    write_json(destination, record)
    return record


def video_inputs(context: Context, video_id: str) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    frames = cast(pd.DataFrame, context.frames[context.frames["video_id"] == video_id])
    rows = context.manifest[context.manifest["video_id"] == video_id]
    if len(rows) != 1 or frames.empty:
        raise ValueError(f"vídeo fora do train permitido: {video_id}")
    row = rows.iloc[0]
    alignment = align_video(frames, float(row["fps"]), int(row["n_frames_counted"]))
    path = context.adapter.resolve_video_path(str(row["relative_path"]))
    return path, alignment, {
        "video_id": video_id, "fps": float(row["fps"]),
        "source_frame_count": int(row["n_frames_counted"]),
        "width": int(row["width"]), "height": int(row["height"]),
    }


def resource_plan(context: Context, root: Path, benchmark: dict[str, Any]) -> dict[str, Any]:
    windows = 0
    largest_decode = 0
    for video_id in sorted(context.frames["video_id"].unique()):
        path, alignment, metadata = video_inputs(context, str(video_id))
        if not path.is_file():
            raise FileNotFoundError(path)
        windows += len(alignment["k_end"])
        decoded_bytes = len(np.unique(alignment["clip_src_indices"])) * metadata["width"] * metadata["height"] * 3
        largest_decode = max(largest_decode, decoded_bytes)
    estimated_cache = windows * (2 * FEATURE_DIM * 4 + 24 * 8 + 6 * 16 * 8 + 32)
    required_disk = estimated_cache * 2 + 1024**3
    free_disk = shutil.disk_usage(root).free
    available_ram = available_memory_bytes()
    hours = windows * 2 * benchmark["seconds_per_forward"] / 3600
    problems = []
    if largest_decode > MAX_DECODED_BYTES or largest_decode * 3 + 1024**3 > available_ram:
        problems.append("decodificação por vídeo excede o orçamento de RAM")
    if required_disk > free_disk:
        problems.append("disco livre insuficiente para cache e publicação atômica")
    if hours > MAX_EXTRACTION_HOURS:
        problems.append("estimativa de inferência excede 24 horas por protocolo")
    if benchmark["settings"]["device"] == "cuda":
        free_cuda, _ = torch.cuda.mem_get_info()
        if benchmark["peak_cuda_bytes"] * 1.5 > free_cuda:
            problems.append("VRAM livre insuficiente com margem de 50%")
    return {
        "windows": windows, "forward_passes": windows * 2,
        "estimated_feature_bytes_uncompressed": estimated_cache,
        "largest_decoded_video_bytes": largest_decode, "available_ram_bytes": available_ram,
        "required_free_disk_bytes": required_disk, "free_disk_bytes": free_disk,
        "estimated_forward_hours": hours, "max_forward_hours": MAX_EXTRACTION_HOURS,
        "timing_excludes_decode_and_io": True, "problems": problems, "feasible": not problems,
    }


def preflight(context: Context, root: Path, device: str) -> dict[str, Any]:
    settings = setup(device)
    smoke_path = root / f"smoke_{device}.json"
    benchmark = json.loads(smoke_path.read_text(encoding="utf-8"))
    path = weight_path(root, False)
    weights_digest = sha256_file(path)
    if benchmark["settings"] != settings or benchmark["weight_sha256"] != weights_digest or benchmark["contract"] != contract():
        raise ValueError("smoke desatualizado; execute smoke novamente no mesmo dispositivo")
    plan = resource_plan(context, root, benchmark)
    destination = root / context.protocol / "feasibility.json"
    guard_destination(destination)
    write_json(destination, {**context_record(context), **plan})
    if not plan["feasible"]:
        raise RuntimeError(f"extração recusada: {plan['problems']}")
    return plan


def extract(context: Context, root: Path, device: str) -> None:
    preflight(context, root, device)
    settings = setup(device)
    path = weight_path(root, False)
    weights_digest = sha256_file(path)
    model = load_backbone(path, device)
    cache_root = root / context.protocol / "features"
    for video_id in sorted(str(v) for v in context.frames["video_id"].unique()):
        video_path, alignment, metadata = video_inputs(context, video_id)
        source_digest = sha256_file(video_path)
        expected = {
            **context_record(context), **metadata, "contract": contract(), "settings": settings,
            "weight_sha256": weights_digest, "source_video_sha256": source_digest,
        }
        destination = cache_path(cache_root, video_id)
        guard_destination(destination)

        def produce() -> dict[str, np.ndarray]:
            indices = sorted(int(index) for index in np.unique(alignment["clip_src_indices"]))
            decoded = dict(zip(indices, decode_frames(video_path, indices), strict=True))
            if any(frame.shape != (metadata["height"], metadata["width"], 3) for frame in decoded.values()):
                raise ValueError("resolução decodificada diverge do manifesto")
            features: dict[str, list[np.ndarray]] = {mode: [] for mode in MODES}
            for dynamic, static in zip(alignment["clip_src_indices"], alignment["static_src_indices"], strict=True):
                for mode, clip in (("dynamic", dynamic), ("static", static)):
                    rgb = np.stack([decoded[int(index)] for index in clip])
                    features[mode].append(embed(model, rgb, device))
            if sha256_file(video_path) != source_digest:
                raise RuntimeError("vídeo mudou durante a extração")
            return {mode: np.stack(values) for mode, values in features.items()}

        publish_cache(destination, expected, alignment, produce)
        print(f"cache verificado: {context.protocol}/{video_id}", flush=True)


def load_features(
    context: Context, root: Path, windows: pd.DataFrame, mode: str,
    *, expected_settings: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, str]]:
    if mode not in MODES:
        raise ValueError("modo inválido")
    values: list[np.ndarray] = []
    hashes: dict[str, str] = {}
    settings_seen: str | None = None
    code_hash = implementation_sha256()
    lock_hash = sha256_file(REPOSITORY_ROOT / "uv.lock")
    weights_digest = sha256_file(weight_path(root, False))
    for video_id, group in windows.groupby("video_id", sort=False):
        video_id = str(video_id)
        path, alignment, metadata = video_inputs(context, video_id)
        cache = cache_path(root / context.protocol / "features", video_id)
        guard_destination(cache)
        with h5py.File(cache, "r") as stream:
            recorded = json.loads(str(stream.attrs["provenance"]))
        if recorded["settings"]["implementation_sha256"] != code_hash or recorded["settings"]["lock_sha256"] != lock_hash:
            raise ValueError("implementação ou lockfile mudou desde a extração")
        if expected_settings is not None and recorded["settings"] != expected_settings:
            raise ValueError("ambiente ou dispositivo mudou desde a extração")
        current = {
            **context_record(context), **metadata, "contract": contract(), "settings": recorded["settings"],
            "weight_sha256": weights_digest, "source_video_sha256": sha256_file(path),
        }
        identity = digest(recorded["settings"])
        if settings_seen is not None and settings_seen != identity:
            raise ValueError("features misturam ambientes de extração")
        settings_seen = identity
        arrays = read_cache(cache, current, alignment)
        ends = group["k_end"].to_numpy(dtype=np.int64)
        positions = np.searchsorted(arrays["k_end"], ends)
        if np.any(positions >= len(arrays["k_end"])) or not np.array_equal(arrays["k_end"][positions], ends):
            raise ValueError("cache não contém as janelas exatas da dobra")
        if not np.array_equal(arrays["labels"][positions], group["label"].to_numpy()):
            raise ValueError("rótulos do cache divergem")
        values.append(arrays[mode][positions])
        hashes[video_id] = sha256_file(cache)
    return np.concatenate(values), hashes
