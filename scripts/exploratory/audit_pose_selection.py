"""Auditoria da seleção de pessoa nos artefatos de pose do Le2i.

Mede, a partir de uma raiz de pose já extraída, quadros com múltiplas
detecções e candidatos a troca de identidade da pessoa selecionada, além do
enriquecimento desses candidatos nas caudas p99/p99.9 das derivadas temporais
de keypoints e bbox. A métrica principal é o candidato a troca de identidade;
as trocas cruas da track e o subconjunto com IoU baixo entram apenas como
diagnósticos adicionais. Serve para comparar uma extração "antes" e uma
"depois" da mudança de política de seleção; não escreve nada fora do `--json`
pedido.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from gatefall.pose.kinematics import build_pose_features, feature_blocks
from gatefall.pose.loading import pose_path

# Uma troca de track só entra no diagnóstico de IoU quando a bbox selecionada
# também descola espacialmente: trocas de ID do ByteTrack sobre a mesma pessoa
# mantêm IoU alto e não são evidência de outra pessoa física.
SWITCH_IOU_THRESHOLD = 0.5

PRIMARY_EVENT_SET = "candidate_switch"

# Diagnósticos subordinados à métrica principal: toda troca da track
# selecionada, e o subconjunto em que a bbox também descola.
ADDITIONAL_EVENT_SETS = ("track_change", "identity_switch")

EVENT_SETS = (PRIMARY_EVENT_SET, *ADDITIONAL_EVENT_SETS)

DERIVATIVE_BLOCKS = (
    "kp_velocity",
    "kp_acceleration",
    "bbox_velocity",
    "bbox_acceleration",
)

# Uma descontinuidade de posição no quadro t contamina a aceleração em t e em
# t+1, porque a segunda diferença ainda carrega o salto. A máscara de evento
# usada contra esses dois blocos é expandida um quadro à frente.
EXPANDED_BLOCKS = ("kp_acceleration", "bbox_acceleration")

TAIL_PERCENTILES = (99.0, 99.9)


@dataclass
class VideoAudit:
    video_id: str
    k: int
    multi_person_frames: int
    person_found_frames: int
    adjacent_changes: int
    adjacent_changes_multi: int
    post_gap_changes: int
    post_gap_changes_multi: int
    event_masks: dict[str, np.ndarray]
    expanded_event_masks: dict[str, np.ndarray]
    magnitudes: dict[str, np.ndarray]


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
        # Maior componente absoluta do bloco, não a norma L2: a evidência
        # procurada é um salto em alguma coordenada, e a norma dilui um salto
        # isolado entre 34 colunas de keypoints.
        magnitudes[name] = np.abs(matrix[:, start:end]).max(axis=1)
    return magnitudes


def _expand_forward(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    expanded[1:] |= mask[:-1]
    return expanded


def audit_video(video_id: str, *, pose_root: Path) -> VideoAudit:
    path = pose_path(video_id, pose_root=pose_root)
    with h5py.File(path, "r") as h5_file:
        bbox = cast(h5py.Dataset, h5_file["bbox"])[()]
        person_found = cast(h5py.Dataset, h5_file["person_found"])[()]
        n_detections = cast(h5py.Dataset, h5_file["n_detections"])[()]
        track_id = cast(h5py.Dataset, h5_file["track_id"])[()]
        k = int(cast(int, h5_file.attrs["K"]))

    event_masks = {name: np.zeros((k,), dtype=np.bool_) for name in EVENT_SETS}
    adjacent_changes = 0
    adjacent_changes_multi = 0
    post_gap_changes = 0
    post_gap_changes_multi = 0

    tracked = [
        index
        for index in range(k)
        if bool(person_found[index]) and int(track_id[index]) >= 0
    ]
    for previous_index, index in zip(tracked, tracked[1:]):
        if int(track_id[previous_index]) == int(track_id[index]):
            continue
        multi = int(n_detections[previous_index]) > 1 or int(n_detections[index]) > 1
        if index - previous_index == 1:
            adjacent_changes += 1
            adjacent_changes_multi += int(multi)
        else:
            post_gap_changes += 1
            post_gap_changes_multi += int(multi)
        if multi:
            event_masks[PRIMARY_EVENT_SET][index] = True

    previous_index = None
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
        adjacent_changes=adjacent_changes,
        adjacent_changes_multi=adjacent_changes_multi,
        post_gap_changes=post_gap_changes,
        post_gap_changes_multi=post_gap_changes_multi,
        event_masks=event_masks,
        expanded_event_masks={
            name: _expand_forward(mask) for name, mask in event_masks.items()
        },
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
        "event_frames": int(event_mask.sum()),
        "event_frames_in_tail": events_in_tail,
        "enrichment": tail_rate / event_rate if event_rate > 0.0 else 0.0,
    }


def _tails_for(
    mask: np.ndarray, expanded_mask: np.ndarray, magnitudes: dict[str, np.ndarray]
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        block: {
            f"p{percentile:g}": _enrichment(
                expanded_mask if block in EXPANDED_BLOCKS else mask,
                magnitudes[block],
                percentile,
            )
            for percentile in TAIL_PERCENTILES
        }
        for block in DERIVATIVE_BLOCKS
    }


def _print_tails(
    tails: dict[str, dict[str, dict[str, float | int]]], indent: str = "  "
) -> None:
    for block in DERIVATIVE_BLOCKS:
        for percentile in TAIL_PERCENTILES:
            stats = tails[block][f"p{percentile:g}"]
            print(
                f"{indent}{block} p{percentile:g}: "
                f"{stats['event_frames_in_tail']}/{stats['tail_frames']} quadros, "
                f"enriquecimento {stats['enrichment']:.2f}x "
                f"(máscara de {stats['event_frames']} quadros)"
            )


def run_audit(pose_root: Path, label: str, json_path: Path | None) -> None:
    video_ids = _discover_video_ids(pose_root)
    audits = [audit_video(video_id, pose_root=pose_root) for video_id in video_ids]

    event_masks = {
        name: np.concatenate([a.event_masks[name] for a in audits])
        for name in EVENT_SETS
    }
    expanded_event_masks = {
        name: np.concatenate([a.expanded_event_masks[name] for a in audits])
        for name in EVENT_SETS
    }
    magnitudes = {
        name: np.concatenate([a.magnitudes[name] for a in audits])
        for name in DERIVATIVE_BLOCKS
    }

    total_frames = int(sum(a.k for a in audits))
    total_multi = int(sum(a.multi_person_frames for a in audits))
    total_found = int(sum(a.person_found_frames for a in audits))
    adjacent_changes = int(sum(a.adjacent_changes for a in audits))
    adjacent_changes_multi = int(sum(a.adjacent_changes_multi for a in audits))
    post_gap_changes = int(sum(a.post_gap_changes for a in audits))
    post_gap_changes_multi = int(sum(a.post_gap_changes_multi for a in audits))

    tails = {
        name: _tails_for(event_masks[name], expanded_event_masks[name], magnitudes)
        for name in EVENT_SETS
    }

    summary: dict[str, object] = {
        "label": label,
        "pose_root": str(pose_root),
        "videos": len(audits),
        "total_frames": total_frames,
        "person_found_frames": total_found,
        "multi_person_frames": total_multi,
        "metrica_principal": {
            "event_set": PRIMARY_EVENT_SET,
            "adjacent_track_changes": adjacent_changes,
            "adjacent_track_changes_multi_person": adjacent_changes_multi,
            "post_gap_track_changes": post_gap_changes,
            "post_gap_track_changes_multi_person": post_gap_changes_multi,
            "candidate_switch_frames": int(event_masks[PRIMARY_EVENT_SET].sum()),
            "expanded_frames": int(expanded_event_masks[PRIMARY_EVENT_SET].sum()),
            "tails": tails[PRIMARY_EVENT_SET],
        },
        "diagnosticos_adicionais": {
            "switch_iou_threshold": SWITCH_IOU_THRESHOLD,
            "event_sets": {
                name: {
                    "event_frames": int(event_masks[name].sum()),
                    "expanded_frames": int(expanded_event_masks[name].sum()),
                    "tails": tails[name],
                }
                for name in ADDITIONAL_EVENT_SETS
            },
        },
    }

    print(f"\nAuditoria de seleção de pessoa ({label}) — {pose_root}")
    print(f"vídeos: {len(audits)}")
    print(f"quadros da grade: {total_frames}")
    pct_found = 100.0 * total_found / total_frames if total_frames > 0 else 0.0
    print(f"quadros com pessoa encontrada: {total_found} ({pct_found:.4f}%)")
    pct_multi = 100.0 * total_multi / total_frames if total_frames > 0 else 0.0
    print(f"quadros multi-pessoa (n_detections > 1): {total_multi} ({pct_multi:.4f}%)")

    candidates = int(event_masks[PRIMARY_EVENT_SET].sum())
    pct_candidates = 100.0 * candidates / total_frames if total_frames > 0 else 0.0
    print("\nMÉTRICA PRINCIPAL — candidatos a troca de identidade")
    print(
        f"trocas de track adjacentes: {adjacent_changes} "
        f"({adjacent_changes_multi} em contexto multi-pessoa)"
    )
    print(
        f"trocas de track após lacuna: {post_gap_changes} "
        f"({post_gap_changes_multi} em contexto multi-pessoa)"
    )
    print(f"candidatos a troca de identidade: {candidates} ({pct_candidates:.4f}%)")
    print(f"\nenriquecimento de {PRIMARY_EVENT_SET} nas caudas:")
    _print_tails(tails[PRIMARY_EVENT_SET])

    print("\ndiagnósticos adicionais (subordinados à métrica principal)")
    for name in ADDITIONAL_EVENT_SETS:
        count = int(event_masks[name].sum())
        pct = 100.0 * count / total_frames if total_frames > 0 else 0.0
        print(f"  quadros marcados como {name}: {count} ({pct:.4f}%)")
    for name in ADDITIONAL_EVENT_SETS:
        print(f"\n  enriquecimento de {name} nas caudas:")
        _print_tails(tails[name], indent="    ")

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
