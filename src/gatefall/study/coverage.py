"""Cobertura e alinhamento das features offline por vídeo, sem carregar backbones."""

import h5py

from gatefall.datasets import get_dataset
from gatefall.features.shared_le2i import shared_source

MODALITY_ROOTS = {"pose": "pose_root", "quality": "quality_root", "dinov3": "dinov3_root", "sam3": "sam3_root"}


def _check(path, identity: str, k: int, split: str) -> str:
    if not path.is_file():
        return "missing"
    try:
        with h5py.File(path, "r") as handle:
            attrs = dict(handle.attrs)
    except OSError:
        return "unreadable"
    if attrs.get("video_id") != identity or attrs.get("split") != split:
        return "identity_mismatch"
    return "ok" if attrs.get("K") == k else "length_mismatch"


def feature_coverage(dataset: str) -> dict:
    from gatefall.dinov3.storage import dinov3_path
    from gatefall.features.quality_storage import quality_path
    from gatefall.pose.loading import pose_path
    from gatefall.sam3.storage import sam3_path

    adapter = shared_source(get_dataset(dataset))
    frames = adapter.load_frames()
    resolvers = {
        "pose": lambda identity: pose_path(identity, pose_root=adapter.pose_root),
        "quality": lambda identity: quality_path(identity, quality_root=adapter.quality_root),
        "dinov3": lambda identity: dinov3_path(identity, dinov3_root=adapter.dinov3_root),
        "sam3": lambda identity: sam3_path(identity, sam3_root=adapter.sam3_root),
    }
    report: dict = {"dataset": dataset, "videos": 0, "modalities": {}}
    problems: dict[str, list[dict]] = {name: [] for name in resolvers}
    counts: dict[str, dict[str, int]] = {name: {} for name in resolvers}
    for video_id, group in frames.groupby("video_id", sort=True):
        identity = str(video_id); k = len(group); split = str(group["split"].iloc[0])
        report["videos"] += 1
        for name, resolve in resolvers.items():
            status = _check(resolve(identity), identity, k, split)
            counts[name][status] = counts[name].get(status, 0) + 1
            if status != "ok":
                problems[name].append({"video_id": identity, "status": status})
    for name in resolvers:
        report["modalities"][name] = {"root": str(getattr(adapter, MODALITY_ROOTS[name])),
                                      "counts": counts[name], "problems": problems[name][:50],
                                      "complete": counts[name].get("ok", 0) == report["videos"]}
    return report
