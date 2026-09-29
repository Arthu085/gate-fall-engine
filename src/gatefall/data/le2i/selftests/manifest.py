"""Selftest sintético do schema e da leitura de manifestos Le2i."""

import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import gatefall.data.le2i.manifest as manifest_module
import gatefall.data.le2i.verification as verification_module
from gatefall.data.manifest import MANIFEST_DTYPES
from gatefall.datasets.le2i import Le2iDatasetAdapter

LEGACY_STATUS = ("pose_status", "dino_status", "sam_status")


def _splits(raw_directory: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for index, split in enumerate(("train", "val", "test"), start=1):
        relative_path = Path("Coffee_room_01/Videos") / f"video ({index}).avi"
        video_path = raw_directory / relative_path
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"synthetic video {index}".encode())
        splits[split] = pd.DataFrame(
            [
                {
                    "path": f"Coffee_room_01/video_{index}",
                    "subject": index,
                    "cam": 1,
                    "label": 1,
                    "start": 0.0,
                    "end": 1.0,
                }
            ]
        )
    return splits


def _metadata() -> dict[str, object]:
    return {
        "r_frame_rate": "25/1",
        "avg_frame_rate": "25/1",
        "n_frames_header": 25,
        "n_frames_counted": 25,
        "duration_s": 1.0,
        "width": 640,
        "height": 480,
        "codec": "mpeg4",
    }


def run_manifest_selftest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw_directory = root / "raw"
        splits = _splits(raw_directory)
        with (
            patch.object(manifest_module, "load_annotation_splits", return_value=splits),
            patch.object(manifest_module, "probe_video", return_value=_metadata()),
            patch.object(manifest_module.shutil, "which", return_value="ffprobe"),
            patch.object(verification_module, "load_annotation_splits", return_value=splits),
        ):
            for protocol in ("cs", "cv"):
                adapter = Le2iDatasetAdapter(protocol=protocol, raw_dir=raw_directory)
                object.__setattr__(
                    adapter, "manifest_path", root / protocol / "manifest.parquet"
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest_module.ingest_le2i_dataset(force=False, adapter=adapter)

                manifest = adapter.load_manifest()
                assert list(manifest.columns) == list(MANIFEST_DTYPES)
                assert len(manifest) == 3
                assert manifest["split"].tolist() == ["train", "val", "test"]
                assert len(adapter.video_paths()) == 3

                with patch.object(verification_module, "LE2I_DATASET", adapter):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        verification_module.verify_le2i_manifest(adapter=adapter)
                assert "verify OK" in output.getvalue()

                legacy = manifest.assign(**{status: "pending" for status in LEGACY_STATUS})
                legacy_path = root / protocol / "legacy.parquet"
                legacy.to_parquet(legacy_path)
                object.__setattr__(adapter, "manifest_path", legacy_path)

                loaded_legacy = adapter.load_manifest()
                pd.testing.assert_frame_equal(
                    loaded_legacy[list(MANIFEST_DTYPES)], manifest
                )
                assert all(status in loaded_legacy for status in LEGACY_STATUS)
                assert len(adapter.video_paths()) == 3

                with patch.object(verification_module, "LE2I_DATASET", adapter):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        verification_module.verify_le2i_manifest(adapter=adapter)
                assert "verify OK" in output.getvalue()
                print(f"[PASS] {adapter.identifier}: ingest, verify e leitura legada")
