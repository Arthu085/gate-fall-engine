"""Auditoria da seleção de pessoa nos artefatos de pose do Le2i.

Mede, a partir de uma raiz de pose já extraída, quadros com múltiplas
detecções, trocas da track selecionada, trocas de identidade plausíveis e o
enriquecimento dessas trocas nas caudas p99/p99.9 das derivadas temporais de
keypoints e bbox. Serve para comparar uma extração "antes" e uma "depois" da
mudança de política de seleção; não escreve nada fora do `--json` pedido.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from gatefall.pose.kinematics import build_pose_features, feature_blocks
from gatefall.pose.loading import pose_path

# Uma troca de track só conta como troca de identidade plausível quando a bbox
# selecionada também descola espacialmente: trocas de ID do ByteTrack sobre a
# mesma pessoa mantêm IoU alto e não são evidência de outra pessoa física.
SWITCH_IOU_THRESHOLD = 0.5

# Os dois conjuntos de quadros suspeitos medidos lado a lado: toda troca da
# track selecionada, e o subconjunto em que a bbox também descola.
EVENT_SETS = ("track_change", "identity_switch")

DERIVATIVE_BLOCKS = (
    "kp_velocity",
    "kp_acceleration",
    "bbox_velocity",
    "bbox_acceleration",
)

TAIL_PERCENTILES = (99.0, 99.9)


@dataclass
class VideoAudit:
    video_id: str
    k: int
    multi_person_frames: int
    person_found_frames: int
    event_masks: dict[str, np.ndarray]
    magnitudes: dict[str, np.ndarray] = field(default_factory=dict)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _discover_video_ids(pose_root: Path) -> list[str]:
    if not pose_root.is_dir():
        print(f"\nauditoria FALHOU: {pose_root} não existe", file=sys.stderr)
        sys.exit(1)
    video_ids = [
        f"{path.parent.name}/{path.stem}" for path in sorted(pose_root.glob("*/*.h5"))
    ]
    if not video_ids:
        print(f"\nauditoria FALHOU: nenhum .h5 em {pose_root}", file=sys.stderr)
        sys.exit(1)
    return video_ids


def _block_magnitudes(matrix: np.ndarray) -> dict[str, np.ndarray]:
    blocks = {name: (start, end) for name, start, end in feature_blocks()}
    magnitudes: dict[str, np.ndarray] = {}
    for name in DERIVATIVE_BLOCKS:
        start, end = blocks[name]
        magnitudes[name] = np.linalg.norm(matrix[:, start:end], axis=1)
    return magnitudes


def audit_video(video_id: str, *, pose_root: Path) -> VideoAudit:
    path = pose_path(video_id, pose_root=pose_root)
    with h5py.File(path, "r") as h5_file:
        bbox = cast(h5py.Dataset, h5_file["bbox"])[()]
        person_found = cast(h5py.Dataset, h5_file["person_found"])[()]
        n_detections = cast(h5py.Dataset, h5_file["n_detections"])[()]
        track_id = cast(h5py.Dataset, h5_file["track_id"])[()]
        k = int(cast(int, h5_file.attrs["K"]))

    event_masks = {name: np.zeros((k,), dtype=np.bool_) for name in EVENT_SETS}
    previous_index: int | None = None
    for index in range(k):
        if not person_found[index]:
            continue
        if previous_index is not None:
            previous_id = int(track_id[previous_index])
            current_id = int(track_id[index])
            if previous_id != -1 and current_id != -1 and previous_id != current_id:
                event_masks["track_change"][index] = True
                if _iou(bbox[index], bbox[previous_index]) < SWITCH_IOU_THRESHOLD:
                    event_masks["identity_switch"][index] = True
        previous_index = index

    matrix, _ = build_pose_features(video_id, pose_root=pose_root)

    return VideoAudit(
        video_id=video_id,
        k=k,
        multi_person_frames=int((n_detections > 1).sum()),
        person_found_frames=int(person_found.sum()),
        event_masks=event_masks,
        magnitudes=_block_magnitudes(matrix),
    )


def _enrichment(
    event_mask: np.ndarray, magnitude: np.ndarray, percentile: float
) -> dict[str, float | int]:
    threshold = float(np.percentile(magnitude, percentile))
    tail = magnitude >= threshold
    tail_count = int(tail.sum())
    events_in_tail = int((tail & event_mask).sum())
    total = int(event_mask.shape[0])
    event_rate = float(event_mask.sum()) / total if total > 0 else 0.0
    tail_rate = events_in_tail / tail_count if tail_count > 0 else 0.0
    return {
        "threshold": threshold,
        "tail_frames": tail_count,
        "event_frames_in_tail": events_in_tail,
        "enrichment": tail_rate / event_rate if event_rate > 0.0 else 0.0,
    }


def run_audit(pose_root: Path, label: str, json_path: Path | None) -> None:
    video_ids = _discover_video_ids(pose_root)
    audits = [audit_video(video_id, pose_root=pose_root) for video_id in video_ids]

    event_masks = {
        name: np.concatenate([a.event_masks[name] for a in audits])
        for name in EVENT_SETS
    }
    magnitudes = {
        name: np.concatenate([a.magnitudes[name] for a in audits])
        for name in DERIVATIVE_BLOCKS
    }

    total_frames = int(sum(a.k for a in audits))
    total_multi = int(sum(a.multi_person_frames for a in audits))
    total_found = int(sum(a.person_found_frames for a in audits))

    tails = {
        event_set: {
            block: {
                f"p{percentile:g}": _enrichment(
                    event_masks[event_set], magnitudes[block], percentile
                )
                for percentile in TAIL_PERCENTILES
            }
            for block in DERIVATIVE_BLOCKS
        }
        for event_set in EVENT_SETS
    }

    summary: dict[str, object] = {
        "label": label,
        "pose_root": str(pose_root),
        "videos": len(audits),
        "total_frames": total_frames,
        "person_found_frames": total_found,
        "multi_person_frames": total_multi,
        "switch_iou_threshold": SWITCH_IOU_THRESHOLD,
        "event_frames": {
            name: int(mask.sum()) for name, mask in event_masks.items()
        },
        "tails": tails,
    }

    print(f"\nAuditoria de seleção de pessoa ({label}) — {pose_root}")
    print(f"vídeos: {len(audits)}")
    print(f"quadros da grade: {total_frames}")
    pct_found = 100.0 * total_found / total_frames if total_frames > 0 else 0.0
    print(f"quadros com pessoa encontrada: {total_found} ({pct_found:.4f}%)")
    pct_multi = 100.0 * total_multi / total_frames if total_frames > 0 else 0.0
    print(f"quadros multi-pessoa (n_detections > 1): {total_multi} ({pct_multi:.4f}%)")
    for name in EVENT_SETS:
        count = int(event_masks[name].sum())
        pct = 100.0 * count / total_frames if total_frames > 0 else 0.0
        print(f"quadros marcados como {name}: {count} ({pct:.4f}%)")

    for event_set in EVENT_SETS:
        print(f"\nenriquecimento de {event_set} nas caudas:")
        for block in DERIVATIVE_BLOCKS:
            for percentile in TAIL_PERCENTILES:
                stats = tails[event_set][block][f"p{percentile:g}"]
                print(
                    f"  {block} p{percentile:g}: "
                    f"{stats['event_frames_in_tail']}/{stats['tail_frames']} quadros, "
                    f"enriquecimento {stats['enrichment']:.2f}x"
                )

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nresumo gravado em {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-root", default="data/features/le2i/pose")
    parser.add_argument("--label", default="atual")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    run_audit(
        Path(args.pose_root),
        args.label,
        Path(args.json) if args.json is not None else None,
    )


if __name__ == "__main__":
    main()
