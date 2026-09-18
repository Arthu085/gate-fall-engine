"""Verificação de determinismo da extração DINOv3: mesma entrada, mesma saída."""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

from gatefall.datasets import DatasetAdapter
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.dinov3.dataset_guard import ensure_dinov3_dataset_supported
from gatefall.dinov3.extract import DEFAULT_BATCH_SIZE, run_dinov3_extract
from gatefall.dinov3.storage import dinov3_path, read_features
from gatefall.hashing import sha256_array


def adapter_with_dinov3_root(adapter: DatasetAdapter, dinov3_root: Path) -> DatasetAdapter:
    if not isinstance(adapter, Le2iDatasetAdapter):
        raise TypeError(
            f"adapter {type(adapter).__name__} não é um Le2iDatasetAdapter; "
            "adapter_with_dinov3_root não sabe derivar uma cópia dele"
        )
    return replace(adapter, dinov3_root=dinov3_root)


def _run_verify(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    output_root: Path,
    repo_dir_value: str | None,
    weights_path_value: str | None,
    batch_size: int,
) -> None:
    verify_adapter = adapter_with_dinov3_root(adapter, output_root)
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


def resolve_verify_determinism_output_root(
    output_dir_value: str | None, *, canonical_root: Path
) -> tuple[Path | None, str]:
    if output_dir_value is None:
        return None, "ephemeral"
    output_root = Path(output_dir_value)
    if output_root.resolve() == canonical_root.resolve():
        return output_root, "canonical"
    return output_root, "non_canonical"


def run_dinov3_verify_determinism(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    repo_dir_value: str | None,
    weights_path_value: str | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_dir_value: str | None = None,
    run_verify: Callable[..., None] | None = None,
) -> None:
    ensure_dinov3_dataset_supported(adapter)

    run_verify_fn = run_verify or _run_verify
    output_root, mode = resolve_verify_determinism_output_root(
        output_dir_value, canonical_root=adapter.dinov3_root
    )

    if mode != "ephemeral":
        assert output_root is not None
        if mode == "canonical":
            print(
                f"\nmodo CANÔNICO — sobrescrevendo o dataset real em {output_root}"
            )
        else:
            print(f"\nmodo NÃO CANÔNICO — diretório explícito {output_root}")
        run_verify_fn(
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
        run_verify_fn(
            video_id,
            adapter=adapter,
            output_root=output_root,
            repo_dir_value=repo_dir_value,
            weights_path_value=weights_path_value,
            batch_size=batch_size,
        )
