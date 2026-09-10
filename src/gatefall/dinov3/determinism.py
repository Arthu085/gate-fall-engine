"""Verificação de determinismo da extração DINOv3: mesma entrada, mesma saída."""

import sys
import tempfile
from pathlib import Path

from gatefall.datasets import DatasetAdapter
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.dinov3.extract import DEFAULT_BATCH_SIZE, run_dinov3_extract
from gatefall.dinov3.storage import dinov3_path, read_features
from gatefall.hashing import sha256_array


def _adapter_with_dinov3_root(adapter: DatasetAdapter, dinov3_root: Path) -> DatasetAdapter:
    return Le2iDatasetAdapter(
        identifier=adapter.identifier,
        raw_dir=adapter.raw_dir,
        manifest_path=adapter.manifest_path,
        frames_path=adapter.frames_path,
        pose_root=adapter.pose_root,
        pose_stats_path=adapter.pose_stats_path,
        dinov3_root=dinov3_root,
        dinov3_stats_path=adapter.dinov3_stats_path,
        label_names=adapter.label_names,
    )


def _run_verify(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    output_root: Path,
    repo_dir_value: str | None,
    weights_path_value: str | None,
    batch_size: int,
) -> None:
    verify_adapter = _adapter_with_dinov3_root(adapter, output_root)
    output_path = dinov3_path(video_id, dinov3_root=verify_adapter.dinov3_root)

    run_dinov3_extract(
        video_id,
        adapter=verify_adapter,
        repo_dir_value=repo_dir_value,
        weights_path_value=weights_path_value,
        batch_size=batch_size,
        force=True,
    )
    first_hash = sha256_array(read_features(output_path))

    run_dinov3_extract(
        video_id,
        adapter=verify_adapter,
        repo_dir_value=repo_dir_value,
        weights_path_value=weights_path_value,
        batch_size=batch_size,
        force=True,
    )
    second_hash = sha256_array(read_features(output_path))

    print(f"\nhash da 1ª extração: {first_hash}")
    print(f"hash da 2ª extração: {second_hash}")

    if first_hash != second_hash:
        print(
            f"\ndinov3 verify-determinism FALHOU: hashes divergem para {video_id}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\ndinov3 verify-determinism OK: hashes idênticos para {video_id}")


def run_dinov3_verify_determinism(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    repo_dir_value: str | None,
    weights_path_value: str | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_dir_value: str | None = None,
) -> None:
    if output_dir_value is not None:
        output_root = Path(output_dir_value)
        canonical = output_root.resolve() == adapter.dinov3_root.resolve()
        if canonical:
            print(
                f"\nmodo CANÔNICO — sobrescrevendo o dataset real em {output_root}"
            )
        else:
            print(f"\nmodo NÃO CANÔNICO — diretório explícito {output_root}")
        _run_verify(
            video_id,
            adapter=adapter,
            output_root=output_root,
            repo_dir_value=repo_dir_value,
            weights_path_value=weights_path_value,
            batch_size=batch_size,
        )
        return

    with tempfile.TemporaryDirectory(
        prefix="gatefall-dinov3-verify-determinism-"
    ) as temporary_dir:
        output_root = Path(temporary_dir)
        print(
            "\nmodo padrão — extraindo em diretório temporário efêmero "
            f"{output_root} (use --output-dir para apontar para outro "
            "caminho, inclusive o canônico)"
        )
        _run_verify(
            video_id,
            adapter=adapter,
            output_root=output_root,
            repo_dir_value=repo_dir_value,
            weights_path_value=weights_path_value,
            batch_size=batch_size,
        )
