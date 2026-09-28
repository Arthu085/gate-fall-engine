"""CLI do relatório auditável de fatores de domínio por split (Le2i-CS/Le2i-CV)."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.data.le2i.annotations import load_annotation_splits
from gatefall.data.le2i.generalization import (
    build_generalization_report,
    fps_distribution,
    pose_coverage_and_quality,
    report_environment_camera_disjointness,
    resolution_distribution,
)
from gatefall.datasets import SUPPORTED_DATASET_IDENTIFIERS, get_dataset
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.pose.loading import _write_synthetic_pose


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def run_report(dataset_name: str, output: Path) -> None:
    adapter = cast(Le2iDatasetAdapter, get_dataset(dataset_name))
    manifest = adapter.load_manifest()
    frames = adapter.load_frames()
    splits = load_annotation_splits(protocol=adapter.protocol)
    result = build_generalization_report(
        manifest,
        frames,
        splits,
        adapter.pose_root,
        identifier=adapter.identifier,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output)
    print(f"{output}: relatório de generalização gravado")


def _build_fixture(pose_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    videos = [
        ("train", "Kitchen", 1, 1, "Kitchen/video_a", 320, 240, 10.0),
        ("train", "Kitchen", 1, 1, "Kitchen/video_b", 320, 240, 10.0),
        ("val", "Office", 2, 2, "Office/video_c", 640, 480, 25.0),
        ("test", "Coffee", 3, 3, "Coffee/video_d", 640, 480, 25.0),
    ]
    manifest_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for split, env, cam, subject, video_id, width, height, fps in videos:
        _write_synthetic_pose(pose_root, video_id, k=3)
        manifest_rows.append(
            {
                "video_id": video_id,
                "env": env,
                "cam": cam,
                "subject": subject,
                "split": split,
                "width": width,
                "height": height,
                "fps": fps,
            }
        )
        for frame_index in range(3):
            frame_rows.append(
                {
                    "video_id": video_id,
                    "split": split,
                    "env": env,
                    "subject": subject,
                    "frame_index": frame_index,
                    "label": 0,
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    frames = pd.DataFrame(frame_rows)
    splits = {
        "train": pd.DataFrame({"path": ["p1", "p2"]}),
        "val": pd.DataFrame({"path": ["p3"]}),
        "test": pd.DataFrame({"path": ["p4"]}),
    }
    return manifest, frames, splits


def _subject_overlap_fixture(pose_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest, frames, splits = _build_fixture(pose_root)
    manifest = manifest.copy()
    manifest.loc[manifest["video_id"] == "Office/video_c", "subject"] = 1
    frames = frames.copy()
    frames.loc[frames["video_id"] == "Office/video_c", "subject"] = 1
    return manifest, frames, splits


def _label_absent_from_train_fixture(
    pose_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest, frames, splits = _build_fixture(pose_root)
    frames = frames.copy()
    frames.loc[frames["video_id"] == "Coffee/video_d", "label"] = 5
    return manifest, frames, splits


def _overlapping_fixture(pose_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest, frames, splits = _build_fixture(pose_root)
    manifest = manifest.copy()
    manifest.loc[manifest["video_id"] == "Office/video_c", "env"] = "Kitchen"
    manifest.loc[manifest["video_id"] == "Office/video_c", "cam"] = 1
    frames = frames.copy()
    frames.loc[frames["video_id"] == "Office/video_c", "env"] = "Kitchen"
    return manifest, frames, splits


def check_resolution_and_fps_aggregation() -> bool:
    manifest = pd.DataFrame(
        {"width": [320, 320, 640], "height": [240, 240, 480], "fps": [10.0, 10.0, 25.0]}
    )
    resolution = resolution_distribution(manifest)
    fps = fps_distribution(manifest)
    return _check(
        "agregação de resolução e fps: contagens corretas a partir do manifesto",
        resolution == {"320x240": 2, "640x480": 1} and fps == {"10.0": 2, "25.0": 1},
    )


def check_pose_coverage_fraction() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp) / "pose"
        _write_synthetic_pose(
            pose_root, "EnvA/video_a", k=3, person_found=np.array([True, True, False])
        )
        _write_synthetic_pose(
            pose_root, "EnvA/video_b", k=3, person_found=np.array([True, True, True])
        )
        coverage, percentiles = pose_coverage_and_quality(
            ["EnvA/video_a", "EnvA/video_b"], pose_root
        )
        expected = 5 / 6
        ok = abs(coverage - expected) < 1e-9 and bool(percentiles)
    return _check(
        "cobertura de detecção de pose: fração de person_found bate com o esperado",
        ok,
    )


def check_disjoint_splits_pass() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        manifest, _frames, _splits = _build_fixture(Path(tmp) / "pose")
        per_video = cast(pd.DataFrame, manifest[["video_id", "env", "cam", "split"]].drop_duplicates())
        ok = report_environment_camera_disjointness(per_video, critical=True)
    return _check(
        "splits com ambiente/câmera disjuntos: passam mesmo com critical=True", ok
    )


def check_overlapping_environment_fails() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        manifest, _frames, _splits = _overlapping_fixture(Path(tmp) / "pose")
        per_video = cast(pd.DataFrame, manifest[["video_id", "env", "cam", "split"]].drop_duplicates())
        is_disjoint = report_environment_camera_disjointness(per_video, critical=True)
    return _check(
        "splits com ambiente sobreposto: disjointness detecta a sobreposição",
        not is_disjoint,
    )


def check_critical_overlap_exits_nonzero() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp) / "pose"
        manifest, frames, splits = _overlapping_fixture(pose_root)
        try:
            build_generalization_report(
                manifest, frames, splits, pose_root, identifier="le2i-cv"
            )
            exited_nonzero = False
        except SystemExit as exc:
            exited_nonzero = exc.code not in (0, None)
    return _check(
        "protocolo crítico (le2i-cv) com ambiente sobreposto: relatório sai "
        "com código != 0",
        exited_nonzero,
    )


def check_non_critical_overlap_does_not_exit() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp) / "pose"
        manifest, frames, splits = _overlapping_fixture(pose_root)
        result = build_generalization_report(
            manifest, frames, splits, pose_root, identifier="le2i"
        )
    return _check(
        "protocolo não crítico (le2i): ambiente sobreposto não interrompe o "
        "relatório",
        result["environment_camera_disjoint"] is False,
    )


def check_le2i_cv_shape_assertion_exits_on_mismatch() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp) / "pose"
        manifest, frames, splits = _build_fixture(pose_root)
        try:
            build_generalization_report(
                manifest, frames, splits, pose_root, identifier="le2i-cv"
            )
            exited_nonzero = False
        except SystemExit as exc:
            exited_nonzero = exc.code not in (0, None)
    return _check(
        "le2i-cv: contagem de vídeos/segmentos fora do formato esperado causa "
        "saída != 0",
        exited_nonzero,
    )


def check_report_shape_and_keys() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp) / "pose"
        manifest, frames, splits = _build_fixture(pose_root)
        result = build_generalization_report(
            manifest, frames, splits, pose_root, identifier="le2i"
        )
    expected_top_keys = {
        "splits",
        "environment_camera_disjoint",
        "subject_disjoint",
        "subjects_overlapping_across_splits",
        "narrative",
    }
    expected_split_keys = {
        "environments",
        "cameras",
        "n_videos",
        "n_segments",
        "n_windows_train_stride",
        "n_windows_eval_stride",
        "resolution_distribution",
        "fps_distribution",
        "pose_detection_coverage",
        "q_pose_percentiles",
        "labels_absent_from_train",
    }
    splits_block = cast(dict[str, dict[str, object]], result["splits"])
    splits_ok = set(splits_block) == {"test", "train", "val"}
    block_keys_ok = all(
        set(block) == expected_split_keys for block in splits_block.values()
    )
    return _check(
        "relatório: chaves de topo e de cada bloco de split completas",
        set(result) == expected_top_keys and splits_ok and block_keys_ok,
    )


def check_subject_disjoint_splits_report_disjoint() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp) / "pose"
        manifest, frames, splits = _build_fixture(pose_root)
        result = build_generalization_report(
            manifest, frames, splits, pose_root, identifier="le2i"
        )
    return _check(
        "subjects disjuntos entre splits: subject_disjoint=True e lista de "
        "sobreposição vazia",
        result["subject_disjoint"] is True
        and result["subjects_overlapping_across_splits"] == [],
    )


def check_subject_overlap_is_reported_not_asserted() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp) / "pose"
        manifest, frames, splits = _subject_overlap_fixture(pose_root)
        result = build_generalization_report(
            manifest, frames, splits, pose_root, identifier="le2i"
        )
    return _check(
        "subjects sobrepostos entre splits: reportado como fato "
        "(subject_disjoint=False, subject 1 listado), sem levantar SystemExit",
        result["subject_disjoint"] is False
        and result["subjects_overlapping_across_splits"] == [1],
    )


def check_labels_absent_from_train_is_flagged() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp) / "pose"
        manifest, frames, splits = _label_absent_from_train_fixture(pose_root)
        result = build_generalization_report(
            manifest, frames, splits, pose_root, identifier="le2i"
        )
    splits_block = cast(dict[str, dict[str, object]], result["splits"])
    test_labels_absent = cast(list[str], splits_block["test"]["labels_absent_from_train"])
    train_labels_absent = cast(list[str], splits_block["train"]["labels_absent_from_train"])
    return _check(
        "classe presente apenas no split de teste (lie_down) é sinalizada em "
        "labels_absent_from_train, igualmente em todos os blocos de split",
        "lie_down" in test_labels_absent
        and test_labels_absent == train_labels_absent,
    )


def run_generalization_selftest() -> None:
    checks = [
        check_resolution_and_fps_aggregation(),
        check_pose_coverage_fraction(),
        check_disjoint_splits_pass(),
        check_overlapping_environment_fails(),
        check_critical_overlap_exits_nonzero(),
        check_non_critical_overlap_does_not_exit(),
        check_le2i_cv_shape_assertion_exits_on_mismatch(),
        check_report_shape_and_keys(),
        check_subject_disjoint_splits_report_disjoint(),
        check_subject_overlap_is_reported_not_asserted(),
        check_labels_absent_from_train_is_flagged(),
    ]
    if not all(checks):
        print("\ngeneralization report selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\ngeneralization report selftest OK: todas as checagens passaram")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report",
        help="Gera o relatório de fatores de domínio a partir do dataset real",
    )
    report_parser.add_argument(
        "--dataset", default="le2i", choices=SUPPORTED_DATASET_IDENTIFIERS
    )
    report_parser.add_argument("--output", type=Path, required=True)
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas do relatório de generalização"
    )

    args = parser.parse_args()
    if args.command == "report":
        run_report(args.dataset, args.output)
    elif args.command == "selftest":
        run_generalization_selftest()


if __name__ == "__main__":
    main()
