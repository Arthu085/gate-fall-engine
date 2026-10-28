"""Custo de extração por quadro em diretório isolado, sem tocar no cache de features."""

import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from gatefall.data.staged.specs import STAGED_SPECS
from gatefall.datasets import get_dataset
from gatefall.datasets.staged import StagedVideoAdapter
from gatefall.provenance import git_state, utc_now
from gatefall.study import STUDY_DIR_NAME
from gatefall.study.plan import VISUAL_EXTRACTORS

PROFILE_DIR = Path("runs/local") / STUDY_DIR_NAME / "profiles"
EXTRACTOR_ORDER = ("pose", "quality", "dinov3", "sam3")


def profiling_adapter(dataset: str, features_dir: Path) -> StagedVideoAdapter:
    adapter = get_dataset(dataset)
    if dataset not in STAGED_SPECS or not isinstance(adapter, StagedVideoAdapter):
        raise ValueError("perfil de extração só para datasets do estudo (upfall, gmdcsa24)")
    return type(adapter)(replace(STAGED_SPECS[dataset], features_dir=features_dir))


def _extract(name: str, video_id: str, adapter) -> None:
    if name == "pose":
        from gatefall.pose.extract import run_pose_extract
        from gatefall.pose.smoke import DEFAULT_MODEL
        run_pose_extract(video_id, DEFAULT_MODEL, False, adapter=adapter)
    elif name == "quality":
        from gatefall.features.quality_extract import run_quality_extract
        run_quality_extract(video_id, adapter=adapter)
    elif name == "dinov3":
        from gatefall.dinov3.extract import run_dinov3_extract
        run_dinov3_extract(video_id, adapter=adapter)
    else:
        from gatefall.sam3.extract import run_sam3_extract
        run_sam3_extract(video_id, adapter=adapter)


def profile_extraction(dataset: str, videos: int, output_dir: Path = PROFILE_DIR) -> Path:
    """Extrai `videos` vídeos de teste em diretório temporário e mede segundos por quadro.

    O primeiro vídeo inclui carga de modelo; `seconds_per_frame_warm` exclui-o.
    """
    import torch

    if videos < 2:
        raise ValueError("perfil exige ao menos 2 vídeos (1 de aquecimento)")
    frames = get_dataset(dataset).load_frames()
    test = frames[frames["split"] == "test"]
    counts = {str(identity): len(group) for identity, group in test.groupby("video_id", sort=True)}
    selected = sorted(counts)[:videos]
    result: dict = {"dataset": dataset, "created_utc": utc_now(), "code": git_state(), "videos": selected,
                    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    "extractors": {}}
    with tempfile.TemporaryDirectory(prefix="gatefall-profile-") as temporary:
        adapter = profiling_adapter(dataset, Path(temporary))
        for name in EXTRACTOR_ORDER:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            per_video = []
            for video_id in selected:
                started = time.perf_counter()
                _extract(name, video_id, adapter)
                per_video.append({"video_id": video_id, "frames": counts[video_id],
                                  "seconds": time.perf_counter() - started})
            warm = per_video[1:]
            result["extractors"][name] = {
                "per_video": per_video,
                "seconds_per_frame_warm": sum(v["seconds"] for v in warm) / sum(v["frames"] for v in warm),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"extraction-{dataset}-{utc_now().replace(':', '')}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def latest_profile(dataset: str, profile_dir: Path = PROFILE_DIR) -> dict | None:
    candidates = sorted(profile_dir.glob(f"extraction-{dataset}-*.json"))
    return json.loads(candidates[-1].read_text(encoding="utf-8")) if candidates else None


def extraction_seconds_per_frame(profile: dict | None, visual: str) -> float | None:
    if profile is None:
        return None
    return sum(profile["extractors"][name]["seconds_per_frame_warm"] for name in ("pose", *VISUAL_EXTRACTORS[visual]))
