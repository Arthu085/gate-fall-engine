"""Extração offline de features DINOv3 de vídeos do Le2i para HDF5."""

import argparse
import sys
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
import torchvision

from gatefall.config import TARGET_FPS
from gatefall.data.video_io import decode_frames
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.dinov3.backbone import (
    FEATURE_DIM,
    MODEL_NAME,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    RESIZE_SIZE,
    ensure_backbone_paths_exist,
    load_backbone,
    read_dinov3_repo_commit,
    resolve_repo_dir,
    resolve_weights_path,
)
from gatefall.dinov3.features import Dinov3Backbone, compute_features
from gatefall.dinov3.preprocessing import preprocess_frames
from gatefall.hashing import sha256_file

DEFAULT_BATCH_SIZE = 32


class Dinov3ExtractSkipped(Exception):
    """.h5 do vídeo já existe e é válido (dados e proveniência batem) e --force não foi passado."""


class Dinov3ExtractError(Exception):
    """Falha ao processar um vídeo (dados ausentes, video_id não encontrado, verificação
    pós-escrita divergente, ou .h5 existente inválido sem --force para reextrair)."""


@dataclass(frozen=True)
class Dinov3ExtractResult:
    video_id: str
    env: str
    split: str
    k: int


def _select_src_indices(video_id: str, *, adapter: DatasetAdapter) -> list[int]:
    if not adapter.frames_path.exists():
        raise Dinov3ExtractError(
            f"\ndinov3 extract FALHOU: {adapter.frames_path} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro"
        )

    frames = adapter.load_frames()
    video_frames = cast(
        pd.DataFrame, frames[frames["video_id"] == video_id]
    ).sort_values("frame_index")
    if video_frames.empty:
        raise Dinov3ExtractError(
            f"\ndinov3 extract FALHOU: video_id '{video_id}' não encontrado "
            f"em {adapter.frames_path}"
        )

    return [int(x) for x in video_frames["src_index"]]


def run_dinov3_extract(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    repo_dir_value: str | None = None,
    weights_path_value: str | None = None,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    backbone: torch.nn.Module | None = None,
    force: bool = False,
    weights_sha256: str | None = None,
    dinov3_repo_commit: str | None = None,
) -> Dinov3ExtractResult:
    from gatefall.dinov3.storage import (
        dinov3_path,
        validate_existing_file,
        verify_written_file,
        write_dinov3_features_atomic,
    )

    output_path = dinov3_path(video_id, dinov3_root=adapter.dinov3_root)

    repo_dir = resolve_repo_dir(repo_dir_value)
    weights_path = resolve_weights_path(weights_path_value)
    ensure_backbone_paths_exist(repo_dir, weights_path)

    resolved_weights_sha256 = (
        weights_sha256 if weights_sha256 is not None else sha256_file(weights_path)
    )
    resolved_dinov3_repo_commit = (
        dinov3_repo_commit
        if dinov3_repo_commit is not None
        else read_dinov3_repo_commit(repo_dir)
    )

    src_indices = _select_src_indices(video_id, adapter=adapter)
    k = len(src_indices)

    current_provenance: dict[str, object] = {
        "model_name": MODEL_NAME,
        "feature_dim": FEATURE_DIM,
        "weights_sha256": resolved_weights_sha256,
        "dinov3_repo_commit": resolved_dinov3_repo_commit,
        "resize_height": RESIZE_SIZE,
        "resize_width": RESIZE_SIZE,
        "normalize_mean": np.array(NORMALIZE_MEAN, dtype=np.float64),
        "normalize_std": np.array(NORMALIZE_STD, dtype=np.float64),
        "target_fps": TARGET_FPS,
    }

    if output_path.exists():
        reasons = validate_existing_file(
            output_path,
            expected_k=k,
            feature_dim=FEATURE_DIM,
            expected_attrs=current_provenance,
        )
        if not reasons and not force:
            raise Dinov3ExtractSkipped(
                f"skip {output_path} (já existe e é válido, use --force para "
                "sobrescrever)"
            )
        if reasons and not force:
            raise Dinov3ExtractError(
                f"\ndinov3 extract FALHOU: {output_path} existe mas é inválido "
                f"({'; '.join(reasons)}) — rode com --force para reextrair"
            )
        if reasons and force:
            print(f"revalidando e reextraindo {output_path}: {'; '.join(reasons)}")

    manifest = adapter.load_manifest()
    manifest_row = cast(pd.DataFrame, manifest[manifest["video_id"] == video_id])
    if manifest_row.empty:
        raise Dinov3ExtractError(
            f"\ndinov3 extract FALHOU: video_id '{video_id}' não encontrado "
            "no manifesto do Le2i"
        )
    manifest_row = manifest_row.iloc[0]

    video_paths = adapter.video_paths()
    if video_id not in video_paths:
        raise Dinov3ExtractError(
            f"\ndinov3 extract FALHOU: video_id '{video_id}' não encontrado "
            "no manifesto do Le2i"
        )

    frames_rgb = decode_frames(video_paths[video_id], src_indices)
    if len(frames_rgb) != k:
        raise Dinov3ExtractError(
            f"decodificação de {video_id} retornou {len(frames_rgb)} quadros; "
            f"esperado {k}"
        )

    resolved_device = device if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if backbone is None:
        backbone = load_backbone(repo_dir, weights_path, resolved_device)

    features = np.zeros((k, FEATURE_DIM), dtype=np.float16)
    for start in range(0, k, batch_size):
        end = min(start + batch_size, k)
        batch = preprocess_frames(frames_rgb[start:end]).to(resolved_device)
        features[start:end] = compute_features(
            cast(Dinov3Backbone, backbone), batch
        )

    attrs: dict[str, object] = {
        "video_id": video_id,
        "env": str(manifest_row["env"]),
        "split": str(manifest_row["split"]),
        "subject": int(manifest_row["subject"]),
        "K": k,
        "fps": float(manifest_row["fps"]),
        "width": int(manifest_row["width"]),
        "height": int(manifest_row["height"]),
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision.__version__,
        **current_provenance,
    }

    write_dinov3_features_atomic(output_path, features, attrs)
    verify_written_file(output_path, features=features, attrs=attrs)

    print(f"\n{video_id}: K={k}, features gravadas em {output_path}")

    return Dinov3ExtractResult(
        video_id=video_id,
        env=str(manifest_row["env"]),
        split=str(manifest_row["split"]),
        k=k,
    )


def _run_extract_cli(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    repo_dir_value: str | None,
    weights_path_value: str | None,
    batch_size: int,
    force: bool,
) -> None:
    try:
        run_dinov3_extract(
            video_id,
            adapter=adapter,
            repo_dir_value=repo_dir_value,
            weights_path_value=weights_path_value,
            batch_size=batch_size,
            force=force,
        )
    except Dinov3ExtractSkipped as exc:
        print(str(exc))
        sys.exit(0)
    except Dinov3ExtractError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


def run_dinov3_extract_all(
    *,
    adapter: DatasetAdapter,
    repo_dir_value: str | None,
    weights_path_value: str | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> None:
    if not adapter.frames_path.exists():
        print(
            f"\ndinov3 extract-all FALHOU: {adapter.frames_path} não existe — "
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_dir = resolve_repo_dir(repo_dir_value)
    weights_path = resolve_weights_path(weights_path_value)
    backbone = load_backbone(repo_dir, weights_path, device)
    weights_sha256 = sha256_file(weights_path)
    dinov3_repo_commit = read_dinov3_repo_commit(repo_dir)

    processed = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    for video_id in per_video.index:
        video_id = str(video_id)
        try:
            run_dinov3_extract(
                video_id,
                adapter=adapter,
                repo_dir_value=repo_dir_value,
                weights_path_value=weights_path_value,
                device=device,
                batch_size=batch_size,
                backbone=backbone,
                force=force,
                weights_sha256=weights_sha256,
                dinov3_repo_commit=dinov3_repo_commit,
            )
        except Dinov3ExtractSkipped as exc:
            skipped += 1
            print(str(exc))
        except Exception as exc:
            failures.append((video_id, str(exc)))
            print(
                f"\ndinov3 extract-all FALHOU em {video_id}: {exc}",
                file=sys.stderr,
            )
            continue
        else:
            processed += 1

    print("\nResumo dinov3 extract-all")
    print(f"vídeos processados: {processed}")
    print(f"vídeos pulados (já existiam): {skipped}")
    print(f"vídeos com falha: {len(failures)}")

    if failures:
        print("\ndinov3 extract-all: falhas por vídeo:", file=sys.stderr)
        for video_id, message in failures:
            print(f"  {video_id}: {message}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Roda o backbone DINOv3 sobre um vídeo do Le2i e grava um .h5",
    )
    extract_parser.add_argument("--video-id", required=True)
    extract_parser.add_argument("--repo-dir", default=None)
    extract_parser.add_argument("--weights", default=None)
    extract_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    extract_all_parser = subparsers.add_parser(
        "extract-all",
        help="Roda o backbone DINOv3 sobre todos os vídeos do Le2i listados em frames.parquet",
    )
    extract_all_parser.add_argument("--repo-dir", default=None)
    extract_all_parser.add_argument("--weights", default=None)
    extract_all_parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    extract_all_parser.add_argument("--force", action="store_true")
    extract_all_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    report_parser = subparsers.add_parser(
        "report", help="Relata a cobertura das features DINOv3 extraídas"
    )
    report_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    audit_parser = subparsers.add_parser(
        "audit", help="Audita a qualidade das features DINOv3 extraídas"
    )
    audit_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    verify_determinism_parser = subparsers.add_parser(
        "verify-determinism",
        help="Roda a extração de um vídeo duas vezes e confere que as features batem",
    )
    verify_determinism_parser.add_argument("--video-id", required=True)
    verify_determinism_parser.add_argument("--repo-dir", default=None)
    verify_determinism_parser.add_argument("--weights", default=None)
    verify_determinism_parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    verify_determinism_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas de pré-processamento e armazenamento"
    )

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.dinov3.extract_selftest import run_dinov3_selftest

        run_dinov3_selftest()
        return

    adapter = get_dataset(args.dataset)
    if args.command == "extract":
        _run_extract_cli(
            args.video_id,
            adapter=adapter,
            repo_dir_value=args.repo_dir,
            weights_path_value=args.weights,
            batch_size=args.batch_size,
            force=args.force,
        )
    elif args.command == "extract-all":
        run_dinov3_extract_all(
            adapter=adapter,
            repo_dir_value=args.repo_dir,
            weights_path_value=args.weights,
            batch_size=args.batch_size,
            force=args.force,
        )
    elif args.command == "report":
        from gatefall.dinov3.report import run_dinov3_report

        run_dinov3_report(adapter=adapter)
    elif args.command == "audit":
        from gatefall.dinov3.audit import run_dinov3_audit

        run_dinov3_audit(adapter=adapter)
    elif args.command == "verify-determinism":
        from gatefall.dinov3.determinism import run_dinov3_verify_determinism

        run_dinov3_verify_determinism(
            args.video_id,
            adapter=adapter,
            repo_dir_value=args.repo_dir,
            weights_path_value=args.weights,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
