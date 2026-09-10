"""Verificação de determinismo da extração DINOv3: mesma entrada, mesma saída."""

import sys

from gatefall.datasets import DatasetAdapter
from gatefall.dinov3.extract import DEFAULT_BATCH_SIZE, run_dinov3_extract
from gatefall.dinov3.storage import dinov3_path, read_features
from gatefall.hashing import sha256_array


def run_dinov3_verify_determinism(
    video_id: str,
    *,
    adapter: DatasetAdapter,
    repo_dir_value: str | None,
    weights_path_value: str | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    output_path = dinov3_path(video_id, dinov3_root=adapter.dinov3_root)

    run_dinov3_extract(
        video_id,
        adapter=adapter,
        repo_dir_value=repo_dir_value,
        weights_path_value=weights_path_value,
        batch_size=batch_size,
        force=True,
    )
    first_hash = sha256_array(read_features(output_path))

    run_dinov3_extract(
        video_id,
        adapter=adapter,
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
