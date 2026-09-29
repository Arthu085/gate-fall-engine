from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from gatefall.data.frames import read_frames
from gatefall.data.manifest import read_manifest
from gatefall.datasets.le2i import LE2I_LABEL_NAMES


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


@dataclass(frozen=True)
class _SyntheticDatasetAdapter:
    raw_dir: Path
    manifest_path: Path
    frames_path: Path
    dinov3_root: Path
    pose_root: Path = Path("pose")
    pose_stats_path: Path = Path("pose_stats.json")
    quality_root: Path = Path("quality")
    sam3_root: Path = Path("sam3")
    identifier: str = "le2i"
    label_names: tuple[str, ...] = LE2I_LABEL_NAMES

    def load_manifest(self) -> pd.DataFrame:
        return read_manifest(self.manifest_path)

    def load_frames(self) -> pd.DataFrame:
        return read_frames(self.frames_path)

    def video_paths(self) -> dict[str, Path]:
        manifest = self.load_manifest()
        return {
            str(video_id): self.resolve_video_path(str(relative_path))
            for video_id, relative_path in zip(
                manifest["video_id"], manifest["relative_path"]
            )
        }

    def resolve_video_path(self, relative_path: str) -> Path:
        return self.raw_dir / relative_path


class _FakeBackbone:
    def __init__(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> None:
        self._cls_token = cls_token
        self._patch_tokens = patch_tokens

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "x_norm_clstoken": self._cls_token,
            "x_norm_patchtokens": self._patch_tokens,
        }
