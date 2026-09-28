"""Casos sintéticos da auditoria de caixas manuais do Le2i."""

import json
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.data.le2i.manual_bbox import (
    discover_annotation_files,
    parse_annotation_file,
)
from gatefall.pose.loading import _write_synthetic_pose
from gatefall.pose.manual_bbox_audit import FrameRecord, _miss_runs, build_report


def _expect_error(operation: object, text: str) -> bool:
    if not callable(operation):
        return False
    try:
        operation()
    except ValueError as exc:
        return text in str(exc)
    return False


def run_selftest() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        raw_root = root / "raw"
        pose_root = root / "pose"
        annotation_path = raw_root / "Home_01" / "Annotation_files" / "video (1).txt"
        annotation_path.parent.mkdir(parents=True)
        annotation_path.write_text("1,1,292,152,311,240\n", encoding="utf-8")
        representative = parse_annotation_file(annotation_path, n_frames=5)
        assert representative.boxes[0].frame == 1
        assert representative.boxes[0].auxiliary == 1
        assert representative.boxes[0].xyxy == (292, 152, 311, 240)
        annotation_path.write_text(
            "\n  \t\n12\n\n1,77,10,10,20,20\n2,88,0,0,0,0\n \t \n"
            "3, 99, 0, 0, 10, 10\n3,100,20,20,30,30\n\n 7 \n \t\n"
            "4,101,-1,-1,-1,-1\n5,102,10,10,20,20\n\n",
            encoding="utf-8",
        )
        annotations = parse_annotation_file(annotation_path, n_frames=5)
        assert annotations.metadata_lines == 2
        assert annotations.zero_rows == 1 and annotations.negative_rows == 1
        assert len(annotations.boxes) == 4
        assert [box.auxiliary for box in annotations.boxes] == [77, 99, 100, 102]
        assert discover_annotation_files(raw_root) == {
            "home_01/video_1": annotation_path
        }

        manifest = pd.DataFrame(
            [
                {
                    "video_id": "home_01/video_1",
                    "env": "home_01",
                    "split": "train",
                    "n_frames_counted": 5,
                    "width": 100,
                    "height": 200,
                },
                {
                    "video_id": "office/video_2",
                    "env": "office",
                    "split": "test",
                    "n_frames_counted": 3,
                    "width": 100,
                    "height": 200,
                },
            ]
        )
        frames = pd.DataFrame(
            [
                {
                    "video_id": "home_01/video_1",
                    "frame_index": index,
                    "src_index": index,
                    "env": "home_01",
                    "split": "train",
                    "label": label,
                }
                for index, label in enumerate([0, 1, 2, 2, IGNORE_LABEL])
            ]
        )
        pose_bbox = np.asarray(
            [
                [10, 10, 20, 20],
                [10, 10, 20, 20],
                [20, 20, 30, 30],
                [10, 10, 20, 20],
                [10, 10, 20, 20],
            ],
            dtype=np.float32,
        )
        _write_synthetic_pose(
            pose_root,
            "home_01/video_1",
            k=5,
            bbox=pose_bbox,
            person_found=np.asarray([True, True, True, True, False]),
        )
        report = build_report(manifest, frames, raw_root, pose_root)
        json.dumps(report, allow_nan=False)
        coverage = report["coverage"]
        assert isinstance(coverage, dict) and coverage["annotated_videos"] == 1
        assert (
            coverage["corpus_videos"] == 2 and coverage["annotated_environments"] == 1
        )
        records = report["sampled_manual_present_frames"]
        assert isinstance(records, list) and len(records) == 3
        assert [record["manual_frame"] for record in records] == [1, 3, 5]
        assert records[1]["manual_box_count"] == 2 and records[1]["best_iou"] == 1.0
        assert records[2]["person_found"] is False and records[2]["best_iou"] is None
        overall = report["overall"]
        assert isinstance(overall, dict) and overall["pose_detection_recall"] == 2 / 3
        assert overall["pose_miss_frames"] == 1 and overall["multi_box_frames"] == 1
        assert cast(dict[str, object], overall["best_iou"])["mean"] == 1.0
        assert report["annotation_rows"] == {
            "zero_rows": 1,
            "negative_rows": 1,
            "metadata_lines": 2,
            "valid_box_rows": 4,
        }
        label_groups = cast(dict[str, dict[str, object]], report["by_label_group"])
        assert label_groups["fall_or_fallen"]["manual_present_frames"] == 1
        assert label_groups["other_labeled"]["manual_present_frames"] == 1
        assert label_groups["ignored"]["manual_present_frames"] == 1
        assert label_groups["other_labeled"]["pose_miss_frames"] == 0
        assert label_groups["ignored"]["pose_miss_frames"] == 1
        runs = report["pose_miss_runs"]
        assert isinstance(runs, dict) and runs["count"] == 1 and runs["max_length"] == 1
        miss_records = [
            cast(
                FrameRecord,
                {
                    **records[0],
                    "frame_index": index,
                    "person_found": False,
                    "best_iou": None,
                },
            )
            for index in (0, 1, 3)
        ]
        assert [run["length"] for run in _miss_runs(miss_records)] == [2, 1]

        duplicate_manifest = pd.concat(
            [manifest, manifest.iloc[[0]]], ignore_index=True
        )
        assert _expect_error(
            lambda: build_report(duplicate_manifest, frames, raw_root, pose_root),
            "video_id duplicado",
        )

        annotation_path.write_text("1,2,3,4,5\n", encoding="utf-8")
        assert _expect_error(
            lambda: parse_annotation_file(annotation_path, n_frames=5),
            "estrutura inválida",
        )
        annotation_path.write_text("1 2 3 4 5 6\n", encoding="utf-8")
        assert _expect_error(
            lambda: parse_annotation_file(annotation_path, n_frames=5),
            "estrutura inválida",
        )
        annotation_path.write_text("6,2,1,1,2,2\n", encoding="utf-8")
        assert _expect_error(
            lambda: parse_annotation_file(annotation_path, n_frames=5), "fora de 1..5"
        )
        annotation_path.write_text("1,2,10,10,5,20\n", encoding="utf-8")
        assert _expect_error(
            lambda: parse_annotation_file(annotation_path, n_frames=5),
            "positiva impossível",
        )
        annotation_path.write_text("1,2,10,10,10,20\n", encoding="utf-8")
        assert _expect_error(
            lambda: parse_annotation_file(annotation_path, n_frames=5),
            "positiva impossível",
        )
        annotation_path.write_text("1,2,10,10,101,20\n", encoding="utf-8")
        assert _expect_error(
            lambda: build_report(manifest, frames, raw_root, pose_root),
            "fora de 100x200",
        )
        annotation_path.unlink()
        assert discover_annotation_files(raw_root) == {}
        print(
            "[PASS] auditoria manual Le2i: parser, alinhamento, IoU, falhas, agregação e arquivos ausentes"
        )
