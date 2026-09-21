"""Extração offline de descritores V_t do SAM 3 de vídeos do Le2i para HDF5."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import TARGET_FPS
from gatefall.data.video_io import decode_frames
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.hashing import sha256_file
from gatefall.sam3.dataset_guard import (
    SAM3_SUPPORTED_DATASET_IDENTIFIERS,
    ensure_sam3_dataset_supported,
)
from gatefall.sam3.descriptors import V_T_DIM, compute_descriptor
from gatefall.sam3.runtime import (
    TEXT_PROMPT,
    Sam3RuntimeSegmenter,
    Sam3Segmenter,
    resolve_checkpoint_path,
    resolve_runtime_project_dir,
)
from gatefall.sam3.selection import InstanceSelector

MODEL_NAME = "facebook/sam3"


class Sam3ExtractSkipped(Exception):
    """.h5 do vídeo já existe e é válido (dados e proveniência batem) e --force não foi passado."""


class Sam3ExtractError(Exception):
    """Falha ao processar um vídeo (dados ausentes, video_id não encontrado, verificação
    pós-escrita divergente, ou .h5 existente inválido sem --force para reextrair)."""


@dataclass(frozen=True)
class Sam3ExtractResult:
    video_id: str
    env: str
    split: str
    k: int
    n_present: int


def select_src_indices(video_id: str, *, adapter: DatasetAdapter) -> list[int]:
    if not adapter.frames_path.exists():
        raise Sam3ExtractError(
            f"\nsam3 extract FALHOU: {adapter.frames_path} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro"
        )

    frames = adapter.load_frames()
    video_frames = cast(
        pd.DataFrame, frames[frames["video_id"] == video_id]
    ).sort_values("frame_index")
    if video_frames.empty:
        raise Sam3ExtractError(
            f"\nsam3 extract FALHOU: video_id '{video_id}' não encontrado "
            f"em {adapter.frames_path}"
        )

    return [int(x) for x in video_frames["src_index"]]


def _resolve_provenance_hash(explicit: str | None, path: Path) -> str:
    if explicit is not None:
        return explicit
    return sha256_file(path) if path.exists() else ""


def _warn_if_empty_provenance_hash(attr_name: str, value: str) -> None:
    if value == "":
        print(
            f"aviso: proveniência '{attr_name}' gravada como '' — o arquivo "
            "de hash não existia no momento em que foi calculado; este "
            "artefato terá proveniência incompleta",
            file=sys.stderr,
        )


def run_frames_through_segmenter(
    frames_rgb: list[np.ndarray], *, segmenter: Sam3Segmenter, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = len(frames_rgb)
    v_t = np.zeros((k, V_T_DIM), dtype=np.float32)
    sam_score = np.zeros((k,), dtype=np.float32)
    n_instances = np.zeros((k,), dtype=np.int16)

    selector = InstanceSelector()
    for position, frame in enumerate(frames_rgb):
        instances = segmenter.segment_frame(frame, TEXT_PROMPT)
        n_instances[position] = len(instances)
        selected_index = selector.select(instances)
        if selected_index is None:
            v_t[position] = compute_descriptor(None, frame_width=width, frame_height=height)
            continue
        selected = instances[selected_index]
        v_t[position] = compute_descriptor(
            selected.mask, frame_width=width, frame_height=height
        )
        sam_score[position] = selected.score

    return v_t, sam_score, n_instances


def run_sam3_extract(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    segmenter: Sam3Segmenter | None = None,
    runtime_project_dir_value: str | None = None,
    checkpoint_path_value: str | None = None,
    force: bool = False,
    checkpoint_sha256: str | None = None,
    runtime_lock_sha256: str | None = None,
) -> Sam3ExtractResult:
    ensure_sam3_dataset_supported(adapter)

    from gatefall.sam3.storage import (
        sam3_path,
        validate_existing_file,
        verify_written_file,
        write_sam3_atomic,
    )

    output_path = sam3_path(video_id, sam3_root=adapter.sam3_root)

    runtime_project_dir = resolve_runtime_project_dir(runtime_project_dir_value)
    checkpoint_path = resolve_checkpoint_path(checkpoint_path_value)

    external_segmenter = segmenter
    if external_segmenter is None:
        from gatefall.sam3.runtime import ensure_sam3_runtime_available

        ensure_sam3_runtime_available(runtime_project_dir, checkpoint_path)

    resolved_checkpoint_sha256 = _resolve_provenance_hash(checkpoint_sha256, checkpoint_path)
    resolved_runtime_lock_sha256 = _resolve_provenance_hash(
        runtime_lock_sha256, runtime_project_dir / "uv.lock"
    )

    src_indices = select_src_indices(video_id, adapter=adapter)
    k = len(src_indices)

    current_provenance: dict[str, object] = {
        "model_name": MODEL_NAME,
        "text_prompt": TEXT_PROMPT,
        "sam3_checkpoint_sha256": resolved_checkpoint_sha256,
        "sam3_runtime_lock_sha256": resolved_runtime_lock_sha256,
        "target_fps": TARGET_FPS,
    }

    if output_path.exists():
        reasons = validate_existing_file(
            output_path,
            expected_k=k,
            v_t_dim=V_T_DIM,
            expected_attrs=current_provenance,
        )
        if not reasons and not force:
            raise Sam3ExtractSkipped(
                f"skip {output_path} (já existe e é válido, use --force para "
                "sobrescrever)"
            )
        if reasons and not force:
            raise Sam3ExtractError(
                f"\nsam3 extract FALHOU: {output_path} existe mas é inválido "
                f"({'; '.join(reasons)}) — rode com --force para reextrair"
            )
        if reasons and force:
            print(f"revalidando e reextraindo {output_path}: {'; '.join(reasons)}")

    manifest = adapter.load_manifest()
    manifest_row = cast(pd.DataFrame, manifest[manifest["video_id"] == video_id])
    if manifest_row.empty:
        raise Sam3ExtractError(
            f"\nsam3 extract FALHOU: video_id '{video_id}' não encontrado "
            "no manifesto do Le2i"
        )
    manifest_row = manifest_row.iloc[0]

    video_paths = adapter.video_paths()
    if video_id not in video_paths:
        raise Sam3ExtractError(
            f"\nsam3 extract FALHOU: video_id '{video_id}' não encontrado "
            "no manifesto do Le2i"
        )

    frames_rgb = decode_frames(video_paths[video_id], src_indices)
    if len(frames_rgb) != k:
        raise Sam3ExtractError(
            f"decodificação de {video_id} retornou {len(frames_rgb)} quadros; "
            f"esperado {k}"
        )

    width = int(manifest_row["width"])
    height = int(manifest_row["height"])

    runtime_manifest: dict[str, object] = {}
    if external_segmenter is not None:
        maybe_manifest = getattr(external_segmenter, "runtime_manifest", None)
        if isinstance(maybe_manifest, dict):
            runtime_manifest = maybe_manifest
        v_t, sam_score, n_instances = run_frames_through_segmenter(
            frames_rgb, segmenter=external_segmenter, width=width, height=height
        )
    else:
        with Sam3RuntimeSegmenter(
            runtime_project_dir=runtime_project_dir, checkpoint_path=checkpoint_path
        ) as live_segmenter:
            runtime_manifest = live_segmenter.runtime_manifest
            resolved_runtime_lock_sha256 = _resolve_provenance_hash(
                runtime_lock_sha256, runtime_project_dir / "uv.lock"
            )
            current_provenance["sam3_runtime_lock_sha256"] = resolved_runtime_lock_sha256
            v_t, sam_score, n_instances = run_frames_through_segmenter(
                frames_rgb, segmenter=live_segmenter, width=width, height=height
            )

    _warn_if_empty_provenance_hash(
        "sam3_checkpoint_sha256", cast(str, current_provenance["sam3_checkpoint_sha256"])
    )
    _warn_if_empty_provenance_hash(
        "sam3_runtime_lock_sha256", cast(str, current_provenance["sam3_runtime_lock_sha256"])
    )

    attrs: dict[str, object] = {
        "video_id": video_id,
        "env": str(manifest_row["env"]),
        "split": str(manifest_row["split"]),
        "subject": int(manifest_row["subject"]),
        "K": k,
        "fps": float(manifest_row["fps"]),
        "width": width,
        "height": height,
        **current_provenance,
        **runtime_manifest,
    }

    write_sam3_atomic(output_path, v_t, sam_score, n_instances, attrs)
    verify_written_file(
        output_path, v_t=v_t, sam_score=sam_score, n_instances=n_instances, attrs=attrs
    )

    n_present = int(np.sum(v_t[:, 0]))
    print(f"\n{video_id}: K={k}, n_present={n_present}, features gravadas em {output_path}")

    return Sam3ExtractResult(
        video_id=video_id,
        env=str(manifest_row["env"]),
        split=str(manifest_row["split"]),
        k=k,
        n_present=n_present,
    )


def _run_extract_cli(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    runtime_project_dir_value: str | None,
    checkpoint_path_value: str | None,
    force: bool,
) -> None:
    try:
        run_sam3_extract(
            video_id,
            adapter=adapter,
            runtime_project_dir_value=runtime_project_dir_value,
            checkpoint_path_value=checkpoint_path_value,
            force=force,
        )
    except Sam3ExtractSkipped as exc:
        print(str(exc))
        sys.exit(0)
    except Sam3ExtractError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


def run_sam3_extract_all(
    *,
    adapter: DatasetAdapter,
    runtime_project_dir_value: str | None,
    checkpoint_path_value: str | None,
    force: bool = False,
) -> None:
    ensure_sam3_dataset_supported(adapter)

    if not adapter.frames_path.exists():
        print(
            f"\nsam3 extract-all FALHOU: {adapter.frames_path} não existe — "
            "rode `uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = adapter.load_frames()
    per_video = cast(
        pd.DataFrame,
        frames.groupby("video_id").agg(
            env=("env", "first"), split=("split", "first"), k=("frame_index", "size")
        ),
    )

    runtime_project_dir = resolve_runtime_project_dir(runtime_project_dir_value)
    checkpoint_path = resolve_checkpoint_path(checkpoint_path_value)

    from gatefall.sam3.runtime import ensure_sam3_runtime_available

    ensure_sam3_runtime_available(runtime_project_dir, checkpoint_path)

    processed = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    with Sam3RuntimeSegmenter(
        runtime_project_dir=runtime_project_dir, checkpoint_path=checkpoint_path
    ) as live_segmenter:
        checkpoint_sha256 = _resolve_provenance_hash(None, checkpoint_path)
        runtime_lock_sha256 = _resolve_provenance_hash(None, runtime_project_dir / "uv.lock")
        for video_id in per_video.index:
            video_id = str(video_id)
            try:
                run_sam3_extract(
                    video_id,
                    adapter=adapter,
                    segmenter=live_segmenter,
                    runtime_project_dir_value=runtime_project_dir_value,
                    checkpoint_path_value=checkpoint_path_value,
                    force=force,
                    checkpoint_sha256=checkpoint_sha256,
                    runtime_lock_sha256=runtime_lock_sha256,
                )
            except Sam3ExtractSkipped as exc:
                skipped += 1
                print(str(exc))
            except Exception as exc:
                failures.append((video_id, str(exc)))
                print(
                    f"\nsam3 extract-all FALHOU em {video_id}: {exc}",
                    file=sys.stderr,
                )
                if not live_segmenter.is_alive:
                    print(
                        "\nsam3 extract-all ABORTADO: o processo do runtime SAM 3 "
                        f"morreu ao processar {video_id} — os vídeos restantes "
                        "seriam relatados com falhas enganosas de pipe/EOF contra "
                        "um subprocesso morto; reinicie a extração em vez de "
                        "confiar nesta lista parcial",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                continue
            else:
                processed += 1

    print("\nResumo sam3 extract-all")
    print(f"vídeos processados: {processed}")
    print(f"vídeos pulados (já existiam): {skipped}")
    print(f"vídeos com falha: {len(failures)}")

    if failures:
        print("\nsam3 extract-all: falhas por vídeo:", file=sys.stderr)
        for video_id, message in failures:
            print(f"  {video_id}: {message}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Roda o SAM 3 sobre um vídeo do Le2i e grava um .h5",
    )
    extract_parser.add_argument("--video-id", required=True)
    extract_parser.add_argument("--runtime-dir", default=None)
    extract_parser.add_argument("--checkpoint", default=None)
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )

    extract_all_parser = subparsers.add_parser(
        "extract-all",
        help="Roda o SAM 3 sobre todos os vídeos do Le2i listados em frames.parquet",
    )
    extract_all_parser.add_argument("--runtime-dir", default=None)
    extract_all_parser.add_argument("--checkpoint", default=None)
    extract_all_parser.add_argument("--force", action="store_true")
    extract_all_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )

    report_parser = subparsers.add_parser(
        "report", help="Relata a cobertura das features SAM 3 extraídas"
    )
    report_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )

    verify_frame_alignment_parser = subparsers.add_parser(
        "verify-frame-alignment",
        help=(
            "Confere que decode_frames retorna o quadro correto para o "
            "src_index solicitado em uma amostra fixa de vídeos (roda "
            "hardware real do SAM 3, não faz parte do CI)"
        ),
    )
    verify_frame_alignment_parser.add_argument("--runtime-dir", default=None)
    verify_frame_alignment_parser.add_argument("--checkpoint", default=None)
    verify_frame_alignment_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )

    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas de descritores e armazenamento"
    )

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.sam3.extract_selftest import run_sam3_selftest

        run_sam3_selftest()
        return

    adapter = get_dataset(args.dataset)
    if args.command == "extract":
        _run_extract_cli(
            args.video_id,
            adapter=adapter,
            runtime_project_dir_value=args.runtime_dir,
            checkpoint_path_value=args.checkpoint,
            force=args.force,
        )
    elif args.command == "extract-all":
        run_sam3_extract_all(
            adapter=adapter,
            runtime_project_dir_value=args.runtime_dir,
            checkpoint_path_value=args.checkpoint,
            force=args.force,
        )
    elif args.command == "report":
        from gatefall.sam3.report import run_sam3_report

        run_sam3_report(adapter=adapter)
    elif args.command == "verify-frame-alignment":
        from gatefall.sam3.frame_alignment import run_sam3_verify_frame_alignment

        run_sam3_verify_frame_alignment(
            adapter=adapter,
            runtime_project_dir_value=args.runtime_dir,
            checkpoint_path_value=args.checkpoint,
        )


if __name__ == "__main__":
    main()
