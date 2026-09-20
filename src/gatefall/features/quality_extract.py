"""Extração offline dos proxies de qualidade (`q_pose`, `q_visual`) para HDF5.

Um sidecar por vídeo em `data/features/le2i/quality/<env>/<video>.h5`, com o
dataset `quality [K, 2]` alinhado quadro a quadro com `frames.parquet` — a
mesma grade temporal usada pelas features de pose e DINOv3.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from gatefall.config import TARGET_FPS
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.dinov3.backbone import RESIZE_SIZE
from gatefall.dinov3.dataset_guard import (
    DINOV3_SUPPORTED_DATASET_IDENTIFIERS,
    ensure_dinov3_dataset_supported,
)
from gatefall.features.quality_sequence import (
    DEFAULT_BATCH_SIZE,
    compute_quality_sequence,
)
from gatefall.features.quality_storage import (
    QUALITY_CHANNEL_NAMES,
    quality_path,
    read_quality,
    validate_existing_file,
    verify_written_file,
    write_quality_atomic,
)
from gatefall.hashing import sha256_file
from gatefall.pose.loading import pose_path

POSE_QUALITY_SOURCE = "gatefall.pose.quality.compute_pose_quality"
VISUAL_QUALITY_SOURCE = "gatefall.dinov3.quality.compute_visual_quality"


class QualityExtractSkipped(Exception):
    """Sidecar do vídeo já existe e é válido, e `--force` não foi passado."""


class QualityExtractError(Exception):
    """Falha ao processar um vídeo (dados ausentes, `video_id` desconhecido,
    verificação pós-escrita divergente, ou sidecar inválido sem `--force`)."""


@dataclass(frozen=True)
class QualityExtractResult:
    video_id: str
    env: str
    split: str
    k: int


def pose_source_sha256(video_id: str, *, pose_root: Path) -> str:
    """Digest do sidecar de pose do qual `q_pose` deriva.

    Sem ele, uma reextração de pose (rerun do YOLO-Pose, mudança de seleção ou
    de fórmula) deixaria o sidecar de qualidade intacto e silenciosamente
    superado: `validate_existing_file` não apontaria motivo, `extract-all`
    pularia o vídeo e o `quality_features_sha256` da receita do B1 continuaria
    batendo. Espelha `weights_sha256` na extração DINOv3.
    """
    path = pose_path(video_id, pose_root=pose_root)
    if not path.exists():
        raise QualityExtractError(
            f"\nquality extract FALHOU: sidecar de pose {path} não existe — rode "
            "`uv run python -m gatefall.pose.extract extract-all` primeiro"
        )
    return sha256_file(path)


def current_provenance(*, pose_source_digest: str) -> dict[str, object]:
    return {
        "channel_names": QUALITY_CHANNEL_NAMES,
        "target_fps": TARGET_FPS,
        "pose_quality_source": POSE_QUALITY_SOURCE,
        "visual_quality_source": VISUAL_QUALITY_SOURCE,
        "pose_source_sha256": pose_source_digest,
        "resize_height": RESIZE_SIZE,
        "resize_width": RESIZE_SIZE,
    }


def existing_sidecar_action(
    reasons: list[str], *, force: bool
) -> Literal["skip", "fail", "reextract"]:
    if force:
        return "reextract"
    return "fail" if reasons else "skip"


def select_src_indices(video_id: str, *, adapter: DatasetAdapter) -> list[int]:
    if not adapter.frames_path.exists():
        raise QualityExtractError(
            f"\nquality extract FALHOU: {adapter.frames_path} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro"
        )
    frames = adapter.load_frames()
    video_frames = cast(
        pd.DataFrame, frames[frames["video_id"] == video_id]
    ).sort_values("frame_index")
    if video_frames.empty:
        raise QualityExtractError(
            f"\nquality extract FALHOU: video_id '{video_id}' não encontrado "
            f"em {adapter.frames_path}"
        )
    return [int(value) for value in video_frames["src_index"]]


def run_quality_extract(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> QualityExtractResult:
    ensure_dinov3_dataset_supported(adapter)

    output_path = quality_path(video_id, quality_root=adapter.quality_root)
    src_indices = select_src_indices(video_id, adapter=adapter)
    k = len(src_indices)
    provenance = current_provenance(
        pose_source_digest=pose_source_sha256(video_id, pose_root=adapter.pose_root)
    )

    if output_path.exists():
        reasons = validate_existing_file(
            output_path, expected_k=k, expected_attrs=provenance
        )
        action = existing_sidecar_action(reasons, force=force)
        if action == "skip":
            raise QualityExtractSkipped(
                f"skip {output_path} (já existe e é válido, use --force para "
                "sobrescrever)"
            )
        if action == "fail":
            raise QualityExtractError(
                f"\nquality extract FALHOU: {output_path} existe mas é inválido "
                f"({'; '.join(reasons)}) — rode com --force para reextrair"
            )
        if reasons:
            print(f"revalidando e reextraindo {output_path}: {'; '.join(reasons)}")

    manifest = adapter.load_manifest()
    manifest_rows = cast(pd.DataFrame, manifest[manifest["video_id"] == video_id])
    if manifest_rows.empty:
        raise QualityExtractError(
            f"\nquality extract FALHOU: video_id '{video_id}' não encontrado "
            "no manifesto"
        )
    manifest_row = manifest_rows.iloc[0]

    video_paths = adapter.video_paths()
    if video_id not in video_paths:
        raise QualityExtractError(
            f"\nquality extract FALHOU: video_id '{video_id}' não encontrado "
            "no manifesto"
        )

    quality = compute_quality_sequence(
        video_id,
        pose_root=adapter.pose_root,
        video_path=video_paths[video_id],
        src_indices=src_indices,
        batch_size=batch_size,
    )

    attrs: dict[str, object] = {
        "video_id": video_id,
        "env": str(manifest_row["env"]),
        "split": str(manifest_row["split"]),
        "subject": int(manifest_row["subject"]),
        "K": k,
        "fps": float(manifest_row["fps"]),
        **provenance,
    }

    write_quality_atomic(output_path, quality, attrs)
    verify_written_file(output_path, quality=quality, attrs=attrs)

    print(f"\n{video_id}: K={k}, qualidade gravada em {output_path}")

    return QualityExtractResult(
        video_id=video_id,
        env=str(manifest_row["env"]),
        split=str(manifest_row["split"]),
        k=k,
    )


def run_quality_extract_all(
    *,
    adapter: DatasetAdapter,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> None:
    ensure_dinov3_dataset_supported(adapter)

    if not adapter.frames_path.exists():
        print(
            f"\nquality extract-all FALHOU: {adapter.frames_path} não existe — "
            "rode `uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = adapter.load_frames()
    video_ids = [
        str(video_id)
        for video_id in pd.unique(cast(pd.Series, frames["video_id"]))
    ]

    processed = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    for video_id in video_ids:
        try:
            run_quality_extract(
                video_id, adapter=adapter, batch_size=batch_size, force=force
            )
        except QualityExtractSkipped as exc:
            skipped += 1
            print(str(exc))
        except Exception as exc:
            failures.append((video_id, str(exc)))
            print(
                f"\nquality extract-all FALHOU em {video_id}: {exc}",
                file=sys.stderr,
            )
            continue
        else:
            processed += 1

    print("\nResumo quality extract-all")
    print(f"vídeos processados: {processed}")
    print(f"vídeos pulados (já existiam): {skipped}")
    print(f"vídeos com falha: {len(failures)}")

    if failures:
        print("\nquality extract-all: falhas por vídeo:", file=sys.stderr)
        for video_id, message in failures:
            print(f"  {video_id}: {message}", file=sys.stderr)
        sys.exit(1)


def run_quality_report(*, adapter: DatasetAdapter) -> None:
    ensure_dinov3_dataset_supported(adapter)
    frames = adapter.load_frames()
    per_video = cast(
        pd.DataFrame,
        frames.groupby("video_id").agg(
            split=("split", "first"), k=("frame_index", "size")
        ),
    )

    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    values_by_split: dict[str, list[np.ndarray]] = {}
    for video_id in per_video.index:
        video_id = str(video_id)
        path = quality_path(video_id, quality_root=adapter.quality_root)
        if not path.exists():
            missing.append(video_id)
            continue
        expected_k = int(per_video.loc[video_id, "k"])
        try:
            provenance = current_provenance(
                pose_source_digest=pose_source_sha256(
                    video_id, pose_root=adapter.pose_root
                )
            )
        except (QualityExtractError, OSError) as exc:
            invalid.append((video_id, str(exc).strip()))
            continue
        reasons = validate_existing_file(
            path, expected_k=expected_k, expected_attrs=provenance
        )
        if reasons:
            invalid.append((video_id, "; ".join(reasons)))
            continue
        split = str(per_video.loc[video_id, "split"])
        values_by_split.setdefault(split, []).append(read_quality(path))

    print(f"\nvídeos na grade: {len(per_video)}")
    print(f"sidecars ausentes: {len(missing)}")
    print(f"sidecars inválidos (layout ou proveniência): {len(invalid)}")

    for split in sorted(values_by_split):
        stacked = np.concatenate(values_by_split[split], axis=0)
        for channel, name in enumerate(QUALITY_CHANNEL_NAMES):
            column = stacked[:, channel]
            print(
                f"  split={split} {name}: n={column.shape[0]}, "
                f"p05={np.percentile(column, 5):.4f}, "
                f"p50={np.percentile(column, 50):.4f}, "
                f"p95={np.percentile(column, 95):.4f}"
            )

    if missing or invalid:
        print("\nquality report FALHOU", file=sys.stderr)
        for video_id in missing:
            print(f"  ausente: {video_id}", file=sys.stderr)
        for video_id, reason in invalid:
            print(f"  inválido: {video_id}: {reason}", file=sys.stderr)
        sys.exit(1)
    print(
        "\nquality report OK: todos os sidecars presentes, com layout válido e "
        "proveniência coerente"
    )


def run_selftest() -> None:
    from gatefall.features.quality_extract_selftest import run_quality_extract_selftest
    from gatefall.features.quality_sequence_selftest import (
        run_quality_sequence_selftest,
    )

    sequence_ok = run_quality_sequence_selftest()
    extract_ok = run_quality_extract_selftest()
    if not (sequence_ok and extract_ok):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="Extrai os proxies de qualidade de um vídeo e grava o .h5"
    )
    extract_parser.add_argument("--video-id", required=True)
    extract_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument(
        "--dataset", default="le2i", choices=DINOV3_SUPPORTED_DATASET_IDENTIFIERS
    )

    extract_all_parser = subparsers.add_parser(
        "extract-all",
        help="Extrai os proxies de qualidade de todos os vídeos de frames.parquet",
    )
    extract_all_parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    extract_all_parser.add_argument("--force", action="store_true")
    extract_all_parser.add_argument(
        "--dataset", default="le2i", choices=DINOV3_SUPPORTED_DATASET_IDENTIFIERS
    )

    report_parser = subparsers.add_parser(
        "report", help="Relata cobertura e distribuição dos sidecars de qualidade"
    )
    report_parser.add_argument(
        "--dataset", default="le2i", choices=DINOV3_SUPPORTED_DATASET_IDENTIFIERS
    )

    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas da montagem e do armazenamento"
    )

    args = parser.parse_args()
    if args.command == "selftest":
        run_selftest()
        return

    adapter = get_dataset(args.dataset)
    if args.command == "extract":
        try:
            run_quality_extract(
                args.video_id,
                adapter=adapter,
                batch_size=args.batch_size,
                force=args.force,
            )
        except QualityExtractSkipped as exc:
            print(str(exc))
            sys.exit(0)
        except QualityExtractError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    elif args.command == "extract-all":
        run_quality_extract_all(
            adapter=adapter, batch_size=args.batch_size, force=args.force
        )
    elif args.command == "report":
        run_quality_report(adapter=adapter)


if __name__ == "__main__":
    main()
