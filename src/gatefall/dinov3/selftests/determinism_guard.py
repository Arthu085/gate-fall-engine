import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from gatefall.datasets import DatasetAdapter
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.dinov3.audit import run_dinov3_audit
from gatefall.dinov3.backbone import FEATURE_DIM, configure_deterministic_inference
from gatefall.dinov3.dataset_guard import ensure_dinov3_dataset_supported
from gatefall.dinov3.determinism import (
    adapter_with_dinov3_root,
    resolve_verify_determinism_output_root,
    run_dinov3_verify_determinism,
)
from gatefall.dinov3.extract import run_dinov3_extract, run_dinov3_extract_all
from gatefall.dinov3.features import compute_features
from gatefall.dinov3.frame_alignment import run_dinov3_verify_frame_alignment
from gatefall.dinov3.report import run_dinov3_report
from gatefall.dinov3.selftests.fixtures import _check, _FakeBackbone, _SyntheticDatasetAdapter
from gatefall.hashing import sha256_array


def _check_determinism_plumbing() -> bool:
    try:
        configure_deterministic_inference()
        configure_deterministic_inference()
        no_exception = True
    except Exception:
        no_exception = False

    array = np.random.randn(5, FEATURE_DIM).astype(np.float16)
    hash_a = sha256_array(array)
    hash_b = sha256_array(array)
    mutated = array.copy()
    mutated[0, 0] += 1.0
    hash_mutated = sha256_array(mutated)

    cls_token = torch.arange(2 * 768, dtype=torch.float32).reshape(2, 768)
    patch_tokens = torch.zeros((2, 4, 768), dtype=torch.float32)
    backbone = _FakeBackbone(cls_token, patch_tokens)
    batch = torch.zeros((2, 3, 8, 8))
    features_a = compute_features(backbone, batch)
    features_b = compute_features(backbone, batch)
    pipeline_hash_a = sha256_array(features_a)
    pipeline_hash_b = sha256_array(features_b)

    ok = (
        no_exception
        and hash_a == hash_b
        and hash_a != hash_mutated
        and pipeline_hash_a == pipeline_hash_b
    )
    return _check(
        "determinismo: configure_deterministic_inference não lança, "
        "sha256_array é estável e sensível a mutação, pipeline fake repete hash",
        ok,
    )


def _check_configure_deterministic_inference_does_not_seed_rng() -> bool:
    torch.manual_seed(2024)
    before = torch.get_rng_state()
    configure_deterministic_inference()
    after = torch.get_rng_state()
    ok = torch.equal(before, after)
    return _check(
        "configure_deterministic_inference: não altera o estado do RNG global", ok
    )


def _check_adapter_with_dinov3_root() -> bool:
    base = Le2iDatasetAdapter()
    replacement_root = Path("synthetic-dinov3-root")
    replaced = adapter_with_dinov3_root(base, replacement_root)

    cs_ok = (
        replaced.dinov3_root == replacement_root
        and replaced.identifier == base.identifier
        and replaced.raw_dir == base.raw_dir
        and replaced.manifest_path == base.manifest_path
        and replaced.frames_path == base.frames_path
        and replaced.pose_root == base.pose_root
        and replaced.pose_stats_path == base.pose_stats_path
        and replaced.label_names == base.label_names
    )

    cv_replaced = adapter_with_dinov3_root(
        Le2iDatasetAdapter(protocol="cv"), replacement_root
    )
    cv_ok = (
        cv_replaced.dinov3_root == replacement_root
        and cv_replaced.identifier == "le2i-cv"
        and cv_replaced.manifest_path == Path("data/processed/le2i_cv/manifest.parquet")
        and cv_replaced.frames_path == Path("data/processed/le2i_cv/frames.parquet")
        and cv_replaced.pose_stats_path
        == Path("src/gatefall/features/stats/pose_le2i_cv.json")
    )

    return _check(
        "adapter_with_dinov3_root: troca só dinov3_root, mantendo os "
        "demais campos idênticos ao adapter base, e preserva o protocolo "
        "(um adapter cv volta como cv, com os caminhos derivados do cv)",
        cs_ok and cv_ok,
    )


def _check_dinov3_entry_points_reject_le2i_cv() -> bool:
    def rejects(call: Callable[[], object]) -> bool:
        try:
            call()
        except ValueError as exc:
            return "le2i-cv" in str(exc)
        except Exception:
            return False
        return False

    with tempfile.TemporaryDirectory() as temporary_dir:
        cv_dinov3_root = Path(temporary_dir) / "dinov3"
        cv_adapter = Le2iDatasetAdapter(protocol="cv", dinov3_root=cv_dinov3_root)

        rejected = [
            rejects(
                lambda: run_dinov3_extract("env1/video1", adapter=cv_adapter)
            ),
            rejects(
                lambda: run_dinov3_extract_all(
                    adapter=cv_adapter, repo_dir_value=None, weights_path_value=None
                )
            ),
            rejects(
                lambda: run_dinov3_verify_determinism(
                    "env1/video1",
                    adapter=cv_adapter,
                    repo_dir_value=None,
                    weights_path_value=None,
                )
            ),
        ]

        nothing_written_ok = not cv_dinov3_root.exists() and not any(
            Path(temporary_dir).iterdir()
        )

    accepts_cs = True
    try:
        ensure_dinov3_dataset_supported(Le2iDatasetAdapter())
        ensure_dinov3_dataset_supported(Le2iDatasetAdapter(protocol="cv"))
        ensure_dinov3_dataset_supported(
            _SyntheticDatasetAdapter(
                raw_dir=Path("raw"),
                manifest_path=Path("manifest.parquet"),
                frames_path=Path("frames.parquet"),
                dinov3_root=Path("dinov3"),
            )
        )
    except Exception:
        accepts_cs = False

    return _check(
        "guarda de protocolo: extração e determinismo DINOv3 rejeitam o "
        "adapter le2i-cv com ValueError sem criar nada sob "
        "adapter.dinov3_root, e a guarda continua aceitando qualquer adapter "
        "com identifier 'le2i' (checagem por identifier, não por isinstance)",
        all(rejected) and nothing_written_ok and accepts_cs,
    )


def _check_resolve_verify_determinism_output_root() -> bool:
    canonical_root = Path("data/features/le2i/dinov3")

    default_root, default_mode = resolve_verify_determinism_output_root(
        None, canonical_root=canonical_root
    )
    default_ok = default_root is None and default_mode == "ephemeral"

    canonical_value_root, canonical_mode = resolve_verify_determinism_output_root(
        str(canonical_root), canonical_root=canonical_root
    )
    canonical_ok = (
        canonical_value_root == canonical_root and canonical_mode == "canonical"
    )

    other_root = Path("data/features/le2i/dinov3-verify-tmp")
    other_value_root, other_mode = resolve_verify_determinism_output_root(
        str(other_root), canonical_root=canonical_root
    )
    other_ok = other_value_root == other_root and other_mode == "non_canonical"

    with tempfile.TemporaryDirectory() as temporary_dir:
        real_dinov3_root = Path(temporary_dir) / "dinov3"
        adapter = Le2iDatasetAdapter(dinov3_root=real_dinov3_root)

        recorded_roots: list[Path] = []

        def spy_run_verify(
            video_id: str,
            *,
            adapter: DatasetAdapter,
            output_root: Path,
            repo_dir_value: str | None,
            weights_path_value: str | None,
            batch_size: int,
        ) -> None:
            recorded_roots.append(output_root)

        run_dinov3_verify_determinism(
            "env1/video1",
            adapter=adapter,
            repo_dir_value=None,
            weights_path_value=None,
            run_verify=spy_run_verify,
        )

        ephemeral_root_disjoint_ok = (
            len(recorded_roots) == 1
            and recorded_roots[0].resolve() != real_dinov3_root.resolve()
            and real_dinov3_root.resolve() not in recorded_roots[0].resolve().parents
        )
        nothing_written_ok = not real_dinov3_root.exists()

    ok = (
        default_ok
        and canonical_ok
        and other_ok
        and ephemeral_root_disjoint_ok
        and nothing_written_ok
    )
    return _check(
        "resolve_verify_determinism_output_root: modo padrão não referencia "
        "dinov3_root e é efêmero; --output-dir igual ao canônico é "
        "identificado como CANÔNICO; outro caminho é NÃO CANÔNICO; e "
        "run_dinov3_verify_determinism em modo padrão não escreve nada sob "
        "adapter.dinov3_root", ok
    )
