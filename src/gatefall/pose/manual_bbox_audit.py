"""Diagnóstico pós-hoc de detecção e localização de pose com caixas manuais Le2i."""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.data.le2i.manual_bbox import (
    discover_annotation_files,
    parse_annotation_file,
)
from gatefall.datasets.le2i import LE2I_DATASET, LE2I_LABEL_NAMES
from gatefall.pose.loading import load_pose
from gatefall.pose.selection import bbox_iou
from gatefall.runs import validate_local_run_dir


class FrameRecord(TypedDict):
    video_id: str
    frame_index: int
    src_index: int
    manual_frame: int
    env: str
    split: str
    label: str
    label_group: str
    manual_box_count: int
    person_found: bool
    best_iou: float | None


class MissRun(TypedDict):
    video_id: str
    start_frame_index: int
    end_frame_index: int
    length: int


def _summary(records: list[FrameRecord]) -> dict[str, object]:
    present = len(records)
    found = [record for record in records if record["person_found"]]
    ious = np.asarray(
        [record["best_iou"] for record in found if record["best_iou"] is not None],
        dtype=np.float64,
    )
    multi_box = sum(record["manual_box_count"] > 1 for record in records)
    return {
        "manual_present_frames": present,
        "pose_found_frames": len(found),
        "pose_detection_recall": len(found) / present if present else None,
        "pose_miss_frames": present - len(found),
        "pose_miss_fraction": (present - len(found)) / present if present else None,
        "multi_box_frames": multi_box,
        "multi_box_fraction": multi_box / present if present else None,
        "manual_box_count_frequency": dict(
            sorted(Counter(record["manual_box_count"] for record in records).items())
        ),
        "best_iou": {
            "count": len(ious),
            "mean": float(np.mean(ious)) if len(ious) else None,
            "median": float(np.median(ious)) if len(ious) else None,
            "p25": float(np.percentile(ious, 25)) if len(ious) else None,
            "p75": float(np.percentile(ious, 75)) if len(ious) else None,
            "p95": float(np.percentile(ious, 95)) if len(ious) else None,
        },
    }


def _stratify(records: list[FrameRecord], field: str) -> dict[str, dict[str, object]]:
    groups: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    return {key: _summary(groups[key]) for key in sorted(groups)}


def _miss_runs(records: list[FrameRecord]) -> list[MissRun]:
    runs: list[MissRun] = []
    by_video: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        by_video[str(record["video_id"])].append(record)
    for video_id, video_records in sorted(by_video.items()):
        start: int | None = None
        end = -1
        for record in sorted(video_records, key=lambda row: row["frame_index"]):
            frame_index = record["frame_index"]
            if not record["person_found"]:
                if start is not None and frame_index != end + 1:
                    runs.append(
                        {
                            "video_id": video_id,
                            "start_frame_index": start,
                            "end_frame_index": end,
                            "length": end - start + 1,
                        }
                    )
                    start = None
                if start is None:
                    start = frame_index
                end = frame_index
            elif start is not None:
                runs.append(
                    {
                        "video_id": video_id,
                        "start_frame_index": start,
                        "end_frame_index": end,
                        "length": end - start + 1,
                    }
                )
                start = None
        if start is not None:
            runs.append(
                {
                    "video_id": video_id,
                    "start_frame_index": start,
                    "end_frame_index": end,
                    "length": end - start + 1,
                }
            )
    return runs


def build_report(
    manifest: pd.DataFrame, frames: pd.DataFrame, raw_root: Path, pose_root: Path
) -> dict[str, object]:
    files = discover_annotation_files(raw_root)
    manifest_by_video = manifest.set_index("video_id", verify_integrity=True)
    unknown = sorted(set(files) - set(str(value) for value in manifest_by_video.index))
    if unknown:
        raise ValueError(f"anotações manuais sem vídeo no manifesto: {unknown}")
    records: list[FrameRecord] = []
    annotation_rows = Counter[str](
        {"zero_rows": 0, "negative_rows": 0, "metadata_lines": 0, "valid_box_rows": 0}
    )
    annotated_by_env = Counter[str]()
    for video_id, path in sorted(files.items()):
        manifest_row = cast(pd.Series, manifest_by_video.loc[video_id])
        n_frames = int(cast(int, manifest_row["n_frames_counted"]))
        annotations = parse_annotation_file(path, n_frames=n_frames)
        annotation_rows["zero_rows"] += annotations.zero_rows
        annotation_rows["negative_rows"] += annotations.negative_rows
        annotation_rows["metadata_lines"] += annotations.metadata_lines
        annotation_rows["valid_box_rows"] += len(annotations.boxes)
        annotated_by_env[str(manifest_row["env"])] += 1
        boxes_by_frame: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        width = int(cast(int, manifest_row["width"]))
        height = int(cast(int, manifest_row["height"]))
        for box in annotations.boxes:
            x1, y1, x2, y2 = box.xyxy
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                raise ValueError(
                    f"caixa manual fora de {width}x{height} em {path}, quadro {box.frame}: {box.xyxy}"
                )
            boxes_by_frame[box.frame].append(box.xyxy)
        video_frames = cast(pd.DataFrame, frames[frames["video_id"] == video_id])
        if video_frames.empty:
            raise ValueError(f"vídeo anotado sem grade em frames.parquet: {video_id}")
        pose = load_pose(video_id, pose_root=pose_root)
        if (
            pose.k != len(video_frames)
            or pose.bbox.shape != (pose.k, 4)
            or pose.person_found.shape != (pose.k,)
        ):
            raise ValueError(f"grade e HDF5 de pose incompatíveis para {video_id}")
        for pose_index, (_, frame_row) in enumerate(
            video_frames.sort_values("frame_index").iterrows()
        ):
            frame_index = int(cast(int, frame_row["frame_index"]))
            if frame_index != pose_index:
                raise ValueError(
                    f"frame_index descontínuo em {video_id}: esperado {pose_index}, recebido {frame_index}"
                )
            manual_frame = int(cast(int, frame_row["src_index"])) + 1
            if not 1 <= manual_frame <= n_frames:
                raise ValueError(
                    f"src_index fora do vídeo em {video_id}, frame_index={frame_index}"
                )
            boxes = boxes_by_frame.get(manual_frame)
            if not boxes:
                continue
            found = bool(pose.person_found[pose_index])
            best_iou: float | None = None
            if found:
                pose_box = np.asarray(pose.bbox[pose_index], dtype=np.float64)
                if (
                    not np.isfinite(pose_box).all()
                    or pose_box[2] <= pose_box[0]
                    or pose_box[3] <= pose_box[1]
                ):
                    raise ValueError(
                        f"bbox de pose inválida em {video_id}, frame_index={frame_index}"
                    )
                best_iou = float(
                    np.max(bbox_iou(pose_box, np.asarray(boxes, dtype=np.float64)))
                )
            label_id = int(cast(int, frame_row["label"]))
            if label_id != IGNORE_LABEL and not 0 <= label_id < len(LE2I_LABEL_NAMES):
                raise ValueError(f"rótulo inválido em {video_id}: {label_id}")
            label = (
                "ignored" if label_id == IGNORE_LABEL else LE2I_LABEL_NAMES[label_id]
            )
            if label_id == IGNORE_LABEL:
                label_group = "ignored"
            elif label in ("fall", "fallen"):
                label_group = "fall_or_fallen"
            else:
                label_group = "other_labeled"
            records.append(
                {
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "src_index": int(cast(int, frame_row["src_index"])),
                    "manual_frame": manual_frame,
                    "env": str(frame_row["env"]),
                    "split": str(frame_row["split"]),
                    "label": label,
                    "label_group": label_group,
                    "manual_box_count": len(boxes),
                    "person_found": found,
                    "best_iou": best_iou,
                }
            )
    runs = _miss_runs(records)
    corpus_by_env = Counter(str(value) for value in manifest["env"])
    coverage_by_env = {
        env: {
            "annotated_videos": annotated_by_env[env],
            "corpus_videos": corpus_by_env[env],
        }
        for env in sorted(corpus_by_env)
    }
    return {
        "scope": "diagnóstico pós-hoc da detecção e localização de pose automática; não é ground truth para seleção de modelos nem classificador de oclusão",
        "identity_limit": "O maior IoU entre caixas manuais mede apenas localização; não estabelece identidade correta. A coluna auxiliar da anotação não tem semântica documentada usada pelo GateFall.",
        "coverage": {
            "annotated_videos": len(files),
            "corpus_videos": len(manifest),
            "annotated_environments": len(annotated_by_env),
            "by_environment": coverage_by_env,
            "distribution_reference": {
                "annotated_videos": 108,
                "corpus_videos": 190,
                "environments": 3,
            },
        },
        "annotation_rows": dict(annotation_rows),
        "overall": _summary(records),
        "by_environment": _stratify(records, "env"),
        "by_split": _stratify(records, "split"),
        "by_label": _stratify(records, "label"),
        "by_label_group": _stratify(records, "label_group"),
        "pose_miss_runs": {
            "count": len(runs),
            "max_length": max((run["length"] for run in runs), default=0),
            "length_frequency": dict(
                sorted(Counter(run["length"] for run in runs).items())
            ),
            "runs": runs,
        },
        "sampled_manual_present_frames": records,
    }


def run_report(output: Path | None) -> None:
    if output is not None:
        validate_local_run_dir(output.parent)
    manifest = LE2I_DATASET.load_manifest()
    frames = LE2I_DATASET.load_frames()
    report = build_report(
        manifest, frames, LE2I_DATASET.raw_dir, LE2I_DATASET.pose_root
    )
    coverage = cast(dict[str, object], report["coverage"])
    overall = cast(dict[str, object], report["overall"])
    print(
        f"Cobertura manual: {coverage['annotated_videos']}/{coverage['corpus_videos']} vídeos; {coverage['annotated_environments']} ambientes (referência da distribuição: 108/190, 3 ambientes)"
    )
    print(
        f"Quadros amostrados com caixa manual: {overall['manual_present_frames']}; recall de pose: {overall['pose_detection_recall']}; falhas: {overall['pose_miss_frames']} ({overall['pose_miss_fraction']})"
    )
    print(
        f"Quadros com múltiplas caixas: {overall['multi_box_frames']} ({overall['multi_box_fraction']}); IoU: {overall['best_iou']}"
    )
    runs = cast(dict[str, object], report["pose_miss_runs"])
    print(
        f"Linhas de anotação: {report['annotation_rows']}; sequências de falha: {runs['count']}; maior sequência: {runs['max_length']}"
    )
    for field in ("by_environment", "by_split", "by_label", "by_label_group"):
        print(f"{field}: {report[field]}")
    print(report["identity_limit"])
    if output is not None:
        if output.exists():
            raise FileExistsError(f"relatório já existe: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
        try:
            tmp_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_path, output)
        finally:
            tmp_path.unlink(missing_ok=True)
        print(f"Relatório JSON gravado: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--output", type=Path)
    subparsers.add_parser("selftest")
    args = parser.parse_args()
    try:
        if args.command == "report":
            run_report(args.output)
        else:
            from gatefall.pose.manual_bbox_audit_selftest import run_selftest

            run_selftest()
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
