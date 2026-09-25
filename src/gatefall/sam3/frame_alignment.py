"""Confere que decode_frames retorna o quadro correto para o src_index solicitado.

Espelha `dinov3/frame_alignment.py`, mas recomputa o vídeo inteiro (não só as
posições amostradas): a seleção de instância do SAM 3 tem estado causal
(`InstanceSelector`), então checar uma posição isolada sem o histórico de
seleção anterior não seria uma checagem válida da mesma decisão que a
extração tomou.

Roda o SAM 3 real via `Sam3RuntimeSegmenter` — não faz parte do CI (não é
chamado por `extract.py selftest`), é um comando manual de hardware real
(decisão do plano, Q6: mantido por paridade com o DINOv3).
"""

import sys
from pathlib import Path
from typing import Callable

import numpy as np

from gatefall.data.video_io import decode_frames
from gatefall.datasets import DatasetAdapter
from gatefall.sam3 import storage
from gatefall.sam3.dataset_guard import ensure_sam3_dataset_supported
from gatefall.sam3.extract import (
    Sam3ExtractError,
    run_frames_through_segmenter,
    select_src_indices,
)
from gatefall.sam3.runtime import (
    Sam3RuntimeSegmenter,
    Sam3Segmenter,
    resolve_checkpoint_path,
    resolve_runtime_project_dir,
)
from gatefall.sam3.storage import sam3_path

FIXED_SAMPLE_VIDEO_IDS: tuple[str, ...] = (
    "coffee_room_01/video_1",
    "home_01/video_1",
)


def grid_positions(k: int) -> list[int]:
    if k <= 0:
        return []
    seen: list[int] = []
    for p in (0, k // 2, k - 1):
        if p not in seen:
            seen.append(p)
    return seen


def _v_t_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def check_discriminative_match(
    fresh: np.ndarray,
    stored_at_position: np.ndarray,
    *,
    stored_prev: np.ndarray | None,
    stored_next: np.ndarray | None,
) -> tuple[bool, dict[str, float], list[str]]:
    distances = {"self": _v_t_distance(fresh, stored_at_position)}
    ok = True
    inconclusive: list[str] = []
    if stored_prev is not None:
        distances["prev"] = _v_t_distance(fresh, stored_prev)
        if distances["self"] > distances["prev"]:
            ok = False
        elif distances["self"] == distances["prev"]:
            inconclusive.append("prev")
    if stored_next is not None:
        distances["next"] = _v_t_distance(fresh, stored_next)
        if distances["self"] > distances["next"]:
            ok = False
        elif distances["self"] == distances["next"]:
            inconclusive.append("next")
    return ok, distances, inconclusive


def run_sam3_verify_frame_alignment(
    *,
    adapter: DatasetAdapter,
    runtime_project_dir_value: str | None,
    checkpoint_path_value: str | None,
    video_ids: tuple[str, ...] = FIXED_SAMPLE_VIDEO_IDS,
    segmenter: Sam3Segmenter | None = None,
    decode_video_frames: Callable[[Path, list[int]], list[np.ndarray]] | None = None,
) -> None:
    ensure_sam3_dataset_supported(adapter)

    decode_video_frames = decode_video_frames or decode_frames
    video_paths = adapter.video_paths()
    manifest = adapter.load_manifest()

    external_segmenter = segmenter
    runtime_project_dir = resolve_runtime_project_dir(runtime_project_dir_value)
    checkpoint_path = resolve_checkpoint_path(checkpoint_path_value)

    def _verify_video(active_segmenter: Sam3Segmenter, video_id: str, failures: list[str]) -> int:
        try:
            src_indices = select_src_indices(video_id, adapter=adapter)
        except Sam3ExtractError as exc:
            failures.append(str(exc))
            return 0
        k = len(src_indices)
        if video_id not in video_paths:
            failures.append(f"{video_id}: ausente do manifesto (video_paths)")
            return 0
        try:
            stored_v_t = storage.read_v_t(sam3_path(video_id, sam3_root=adapter.sam3_root))
        except (OSError, KeyError) as exc:
            failures.append(f"{video_id}: falha ao ler .h5 ({exc})")
            return 0
        if stored_v_t.shape[0] != k:
            failures.append(
                f"{video_id}: K armazenado ({stored_v_t.shape[0]}) != frames.parquet ({k})"
            )
            return 0

        manifest_row = manifest[manifest["video_id"] == video_id]
        if manifest_row.empty:
            failures.append(f"{video_id}: ausente do manifesto")
            return 0
        row = manifest_row.iloc[0]
        width, height = int(row["width"]), int(row["height"])

        frames_rgb = decode_video_frames(video_paths[video_id], src_indices)
        fresh_v_t, _, _ = run_frames_through_segmenter(
            frames_rgb, segmenter=active_segmenter, width=width, height=height
        )

        inconclusive_count = 0
        for position in grid_positions(k):
            stored_prev = stored_v_t[position - 1] if position > 0 else None
            stored_next = stored_v_t[position + 1] if position < k - 1 else None
            discriminative_ok, distances, inconclusive = check_discriminative_match(
                fresh_v_t[position],
                stored_v_t[position],
                stored_prev=stored_prev,
                stored_next=stored_next,
            )
            print(
                f"{video_id} k={position} (src_index={src_indices[position]}): "
                f"distances={distances}"
            )
            if inconclusive:
                inconclusive_count += 1
                print(
                    f"{video_id} k={position}: empate exato com {inconclusive} "
                    "— inconclusivo, não tratado como falha"
                )
            if not discriminative_ok:
                failures.append(
                    f"{video_id} k={position}: V_t recomputado não é "
                    "estritamente mais próximo da linha correta que das vizinhas"
                )
        if not np.array_equal(fresh_v_t, stored_v_t):
            failures.append(
                f"{video_id}: V_t recomputado do vídeo inteiro diverge do "
                "armazenado (checagem determinística fim a fim)"
            )
        return inconclusive_count

    failures: list[str] = []
    inconclusive_total = 0

    if external_segmenter is not None:
        for video_id in video_ids:
            inconclusive_total += _verify_video(external_segmenter, video_id, failures)
    else:
        with Sam3RuntimeSegmenter(
            runtime_project_dir=runtime_project_dir, checkpoint_path=checkpoint_path
        ) as live_segmenter:
            for video_id in video_ids:
                inconclusive_total += _verify_video(live_segmenter, video_id, failures)

    if failures:
        print("\nsam3 verify-frame-alignment FALHOU:", file=sys.stderr)
        for message in failures:
            print(f"  {message}", file=sys.stderr)
        sys.exit(1)
    if inconclusive_total == 0:
        print(
            "\nsam3 verify-frame-alignment OK: todas as posições amostradas "
            "batem e são discriminativas"
        )
    else:
        plural = "posições" if inconclusive_total > 1 else "posição"
        print(
            "\nsam3 verify-frame-alignment OK: todas as posições amostradas "
            f"batem, mas {inconclusive_total} {plural} amostrada(s) "
            "foi(ram) inconclusiva(s) por empate exato, não conta(m) como "
            "confirmação discriminativa"
        )
