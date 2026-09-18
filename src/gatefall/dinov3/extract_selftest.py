"""Selftest sintético da extração DINOv3 (pré-processamento, features, armazenamento).

Não toca em GPU, dataset real ou pesos do backbone — todas as entradas são
sintéticas.
"""

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
import torch

from gatefall.data.frames import read_frames
from gatefall.data.manifest import read_manifest
from gatefall.datasets import DatasetAdapter
from gatefall.datasets.le2i import LE2I_LABEL_NAMES, Le2iDatasetAdapter
from gatefall.dinov3 import storage
from gatefall.dinov3.audit import (
    DimensionStatsAccumulator,
    count_duplicate_consecutive_rows,
    count_non_finite,
    frame_index_is_contiguous,
    max_abs_and_headroom,
    run_dinov3_audit,
)
from gatefall.dinov3.backbone import (
    FEATURE_DIM,
    RESIZE_SIZE,
    configure_deterministic_inference,
)
from gatefall.dinov3.dataset_guard import ensure_dinov3_dataset_supported
from gatefall.dinov3.determinism import (
    adapter_with_dinov3_root,
    resolve_verify_determinism_output_root,
    run_dinov3_verify_determinism,
)
from gatefall.dinov3.extract import run_dinov3_extract, run_dinov3_extract_all
from gatefall.dinov3.features import Dinov3Backbone, compute_features
from gatefall.dinov3.frame_alignment import (
    check_discriminative_match,
    grid_positions,
    max_abs_diff_within_tolerance,
    run_dinov3_verify_frame_alignment,
)
from gatefall.dinov3.preprocessing import preprocess_frames
from gatefall.dinov3.report import find_provenance_divergences, run_dinov3_report
from gatefall.dinov3.storage import (
    Dinov3StorageError,
    dinov3_path,
    validate_existing_file,
    verify_written_file,
    write_dinov3_features_atomic,
)
from gatefall.hashing import sha256_array


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


@dataclass(frozen=True)
class _SyntheticDatasetAdapter:
    raw_dir: Path
    manifest_path: Path
    frames_path: Path
    dinov3_root: Path
    pose_root: Path = Path("pose")
    pose_stats_path: Path = Path("pose_stats.json")
    identifier: str = "le2i"
    label_names: tuple[str, ...] = LE2I_LABEL_NAMES

    def load_manifest(self) -> pd.DataFrame:
        return read_manifest(self.manifest_path)

    def load_frames(self) -> pd.DataFrame:
        return read_frames(self.frames_path)

    def video_paths(self) -> dict[str, Path]:
        manifest = self.load_manifest()
        return {
            str(video_id): self.resolve_video_path(str(relative_path))
            for video_id, relative_path in zip(
                manifest["video_id"], manifest["relative_path"]
            )
        }

    def resolve_video_path(self, relative_path: str) -> Path:
        return self.raw_dir / relative_path


class _FakeBackbone:
    def __init__(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> None:
        self._cls_token = cls_token
        self._patch_tokens = patch_tokens

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "x_norm_clstoken": self._cls_token,
            "x_norm_patchtokens": self._patch_tokens,
        }


def _check_preprocess_frames_shape_and_dtype() -> bool:
    ok = True
    for height, width in [(64, 64), (100, 200), (300, 150)]:
        frames = [
            np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        out = preprocess_frames(frames)
        ok = ok and (
            out.shape == (3, 3, RESIZE_SIZE, RESIZE_SIZE)
            and out.dtype == torch.float32
        )
    return _check(
        "preprocess_frames: shape [K,3,224,224] float32 para H,W variados", ok
    )


def _check_compute_features_formula() -> bool:
    batch_size = 2
    cls_token = torch.arange(batch_size * 768, dtype=torch.float32).reshape(
        batch_size, 768
    )
    patch_tokens = torch.zeros((batch_size, 4, 768), dtype=torch.float32)
    patch_tokens[0] = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(4, 1).expand(4, 768)
    patch_tokens[1] = torch.tensor([5.0, 6.0, 7.0, 8.0]).view(4, 1).expand(4, 768)
    expected_mean_patch = torch.tensor(
        [[2.5] * 768, [6.5] * 768], dtype=torch.float32
    )

    backbone = _FakeBackbone(cls_token, patch_tokens)
    features = compute_features(backbone, torch.zeros((batch_size, 3, 8, 8)))

    ok = (
        features.shape == (batch_size, FEATURE_DIM)
        and features.dtype == np.float16
        and np.allclose(
            features[:, :768], cls_token.numpy(), atol=1e-2
        )
        and np.allclose(
            features[:, 768:], expected_mean_patch.numpy(), atol=1e-2
        )
    )
    return _check(
        "compute_features: [B,1536] float16 com CLS nas primeiras 768 colunas "
        "e média dos patch tokens nas últimas 768",
        ok,
    )


def _check_storage_round_trip() -> bool:
    features = np.random.randn(5, FEATURE_DIM).astype(np.float16)
    attrs: dict[str, object] = {
        "video_id": "synthetic_env/synthetic_video",
        "K": 5,
        "normalize_mean": np.array([0.485, 0.456, 0.406], dtype=np.float64),
        "normalize_std": np.array([0.229, 0.224, 0.225], dtype=np.float64),
    }

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "synthetic_env" / "synthetic_video.h5"
        write_dinov3_features_atomic(path, features, attrs)
        try:
            verify_written_file(path, features=features, attrs=attrs)
            round_trip_ok = True
        except Dinov3StorageError:
            round_trip_ok = False

        mismatched_features = np.random.randn(5, FEATURE_DIM).astype(np.float16)
        try:
            verify_written_file(path, features=mismatched_features, attrs=attrs)
            mismatch_caught = False
        except Dinov3StorageError:
            mismatch_caught = True

    ok = round_trip_ok and mismatch_caught
    return _check(
        "storage: escrita atômica + releitura confere features e atributos "
        "(inclusive array-valued), e divergência é detectada",
        ok,
    )


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


def _check_audit_helpers() -> bool:
    with_nan = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    with_nan[1, 0] = np.nan
    non_finite_ok = count_non_finite(with_nan) == 1

    max_abs_array = np.zeros((2, 2), dtype=np.float32)
    max_abs_array[0, 0] = 100.0
    max_abs, headroom = max_abs_and_headroom(max_abs_array, ceiling=200.0)
    max_abs_ok = max_abs == 100.0 and headroom == 100.0

    duplicate_array = np.array(
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32
    )
    duplicate_ok = count_duplicate_consecutive_rows(duplicate_array) == 1

    contiguous_ok = frame_index_is_contiguous([0, 1, 2, 3]) and not frame_index_is_contiguous(
        [0, 2, 3, 5]
    )

    accumulator = DimensionStatsAccumulator()
    dead_dim_array = np.zeros((10, 3), dtype=np.float32)
    dead_dim_array[:, 0] = 5.0
    dead_dim_array[:, 1] = np.arange(10, dtype=np.float32)
    dead_dim_array[:, 2] = -3.0
    accumulator.update(dead_dim_array)
    dead = accumulator.dead_dimensions()
    dead_ok = set(dead) == {0, 2}

    ok = non_finite_ok and max_abs_ok and duplicate_ok and contiguous_ok and dead_ok
    return _check(
        "audit: count_non_finite, max_abs_and_headroom, "
        "count_duplicate_consecutive_rows, frame_index_is_contiguous e "
        "DimensionStatsAccumulator dão os resultados esperados", ok
    )


def _full_provenance_attrs(**overrides: object) -> dict[str, object]:
    attrs: dict[str, object] = {
        name: f"valor-{name}" for name in storage.PROVENANCE_ATTR_NAMES
    }
    attrs.update(overrides)
    return attrs


def _check_provenance_divergences() -> bool:
    homogeneous: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(),
        "env1/v2": _full_provenance_attrs(),
    }
    homogeneous_ok = find_provenance_divergences(homogeneous) == []

    heterogeneous: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(),
        "env1/v2": _full_provenance_attrs(weights_sha256="different"),
    }
    divergences = find_provenance_divergences(heterogeneous)
    heterogeneous_ok = len(divergences) == 1 and "env1/v2" in divergences[0]

    ok = homogeneous_ok and heterogeneous_ok
    return _check(
        "find_provenance_divergences: dataset homogêneo dá [] e vídeo com "
        "atributo divergente é nomeado", ok
    )


def _check_provenance_divergences_missing_attribute() -> bool:
    missing_in_reference: dict[str, dict[str, object]] = {
        "env1/v1": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
        "env1/v2": _full_provenance_attrs(),
    }
    divergences_missing_in_reference = find_provenance_divergences(missing_in_reference)
    missing_in_reference_ok = (
        len(divergences_missing_in_reference) == 1
        and "weights_sha256" in divergences_missing_in_reference[0]
        and "env1/v1" in divergences_missing_in_reference[0]
    )

    missing_in_candidate: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(),
        "env1/v2": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
    }
    divergences_missing_in_candidate = find_provenance_divergences(missing_in_candidate)
    missing_in_candidate_ok = (
        len(divergences_missing_in_candidate) == 1
        and "weights_sha256" in divergences_missing_in_candidate[0]
        and "env1/v2" in divergences_missing_in_candidate[0]
    )

    ok = missing_in_reference_ok and missing_in_candidate_ok
    return _check(
        "find_provenance_divergences: atributo ausente em apenas um dos "
        "arquivos (referência ou candidato) é reportado como divergência", ok
    )


def _check_provenance_divergences_missing_from_every_file() -> bool:
    missing_everywhere: dict[str, dict[str, object]] = {
        "env1/v1": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
        "env1/v2": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
    }
    divergences = find_provenance_divergences(missing_everywhere)
    missing_everywhere_ok = (
        len(divergences) == 2
        and all("weights_sha256" in message for message in divergences)
    )

    single_video_with_missing_attribute: dict[str, dict[str, object]] = {
        "env1/v1": {
            key: value
            for key, value in _full_provenance_attrs().items()
            if key != "weights_sha256"
        },
    }
    single_video_divergences = find_provenance_divergences(
        single_video_with_missing_attribute
    )
    single_video_ok = (
        len(single_video_divergences) == 1
        and "weights_sha256" in single_video_divergences[0]
        and "env1/v1" in single_video_divergences[0]
    )

    ok = missing_everywhere_ok and single_video_ok
    return _check(
        "find_provenance_divergences: atributo ausente em todos os arquivos "
        "(inclusive a referência) e dataset de um único vídeo são reportados", ok
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


def _check_grid_positions() -> bool:
    ok = (
        grid_positions(0) == []
        and grid_positions(1) == [0]
        and grid_positions(2) == [0, 1]
        and grid_positions(5) == [0, 2, 4]
        and grid_positions(62) == [0, 31, 61]
    )
    return _check(
        "grid_positions: início, meio e fim, sem duplicatas para K pequeno", ok
    )


def _check_max_abs_diff_within_tolerance() -> bool:
    stored = np.full((1, FEATURE_DIM), 5.86, dtype=np.float16)
    close = stored.copy()
    close[0, 0] = np.nextafter(np.float16(5.86), np.float16(0))
    close_ok, _, _ = max_abs_diff_within_tolerance(close, stored)

    far = stored.copy().astype(np.float32)
    far[0, 0] = 1.0
    far_ok, _, _ = max_abs_diff_within_tolerance(far, stored)

    ok = close_ok and not far_ok
    return _check(
        "max_abs_diff_within_tolerance: diferença de arredondamento float16 "
        "passa, diferença grande falha", ok
    )


def _check_check_discriminative_match() -> bool:
    stored_at_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    stored_prev = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    stored_next = np.array([-10.0, -20.0, -30.0], dtype=np.float32)

    match_ok, _, match_inconclusive = check_discriminative_match(
        stored_at_position,
        stored_at_position,
        stored_prev=stored_prev,
        stored_next=stored_next,
    )

    shifted_ok, _, _ = check_discriminative_match(
        stored_next,
        stored_at_position,
        stored_prev=stored_prev,
        stored_next=stored_next,
    )

    tie_ok, _, tie_inconclusive = check_discriminative_match(
        stored_at_position,
        stored_at_position,
        stored_prev=stored_prev,
        stored_next=stored_at_position.copy(),
    )

    ok = (
        match_ok
        and not match_inconclusive
        and not shifted_ok
        and tie_ok
        and tie_inconclusive == ["next"]
    )
    return _check(
        "check_discriminative_match: vetor recomputado igual ao esperado "
        "passa; simulação de deslocamento de um quadro falha; empate exato "
        "com uma vizinha é inconclusivo, não falha", ok
    )


def _check_validate_existing_file() -> bool:
    features = np.random.randn(5, FEATURE_DIM).astype(np.float16)
    expected_attrs: dict[str, object] = {
        "model_name": "dinov3_vitb16",
        "normalize_mean": np.array([0.485, 0.456, 0.406], dtype=np.float64),
    }
    attrs: dict[str, object] = {"K": 5, **expected_attrs}

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "synthetic_env" / "synthetic_video.h5"
        write_dinov3_features_atomic(path, features, attrs)

        valid_reasons = validate_existing_file(
            path, expected_k=5, feature_dim=FEATURE_DIM, expected_attrs=expected_attrs
        )
        valid_ok = valid_reasons == []

        wrong_k_reasons = validate_existing_file(
            path, expected_k=6, feature_dim=FEATURE_DIM, expected_attrs=expected_attrs
        )
        wrong_k_ok = len(wrong_k_reasons) > 0

        wrong_attr_reasons = validate_existing_file(
            path,
            expected_k=5,
            feature_dim=FEATURE_DIM,
            expected_attrs={"model_name": "different_model"},
        )
        wrong_attr_ok = (
            len(wrong_attr_reasons) > 0 and "model_name" in wrong_attr_reasons[0]
        )

    ok = valid_ok and wrong_k_ok and wrong_attr_ok
    return _check(
        "validate_existing_file: arquivo válido dá [], shape divergente e "
        "atributo divergente dão razões nomeadas", ok
    )


class _IndexAwareFakeBackbone:
    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean_pixel = x.mean(dim=(1, 2, 3))
        batch_size = x.shape[0]
        cls_token = mean_pixel.view(batch_size, 1).expand(batch_size, 768).clone()
        patch_tokens = (
            mean_pixel.view(batch_size, 1, 1).expand(batch_size, 4, 768).clone()
        )
        return {"x_norm_clstoken": cls_token, "x_norm_patchtokens": patch_tokens}


def _frame_for_src_index(src_index: int) -> np.ndarray:
    value = min(255, (src_index + 1) * 10)
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _build_frame_alignment_fixture(
    root: Path,
    *,
    video_id: str,
    k: int,
    include_manifest_row: bool = True,
    include_h5: bool = True,
    h5_row_count: int | None = None,
    stored_row_src_index: Callable[[int], int] | None = None,
) -> _SyntheticDatasetAdapter:
    env, _, video_name = video_id.partition("/")
    raw_dir = root / "raw"
    manifest_path = root / "manifest.parquet"
    frames_path = root / "frames.parquet"
    dinov3_root = root / "dinov3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "video_id": [video_id] * k,
            "frame_index": list(range(k)),
            "src_index": list(range(k)),
        }
    ).to_parquet(frames_path)

    if include_manifest_row:
        pd.DataFrame(
            {
                "video_id": [video_id],
                "relative_path": [f"{env}/{video_name}.avi"],
            }
        ).to_parquet(manifest_path)
    else:
        pd.DataFrame({"video_id": [], "relative_path": []}).to_parquet(manifest_path)

    if include_h5:
        row_count = h5_row_count if h5_row_count is not None else k
        resolve_src_index = stored_row_src_index or (lambda position: position)
        backbone = _IndexAwareFakeBackbone()
        stored = np.zeros((row_count, FEATURE_DIM), dtype=np.float16)
        for position in range(row_count):
            frame = _frame_for_src_index(resolve_src_index(position))
            batch = preprocess_frames([frame])
            stored[position] = compute_features(backbone, batch)[0]
        write_dinov3_features_atomic(
            dinov3_path(video_id, dinov3_root=dinov3_root), stored, {"K": row_count}
        )

    return _SyntheticDatasetAdapter(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        frames_path=frames_path,
        dinov3_root=dinov3_root,
    )


def _run_frame_alignment_and_get_exit_code(
    *,
    adapter: DatasetAdapter,
    video_ids: tuple[str, ...],
    backbone: Dinov3Backbone,
    decode_single_frame: Callable[[Path, int], np.ndarray],
) -> int | None:
    try:
        run_dinov3_verify_frame_alignment(
            adapter=adapter,
            repo_dir_value=None,
            weights_path_value=None,
            video_ids=video_ids,
            backbone=backbone,
            decode_single_frame=decode_single_frame,
        )
    except SystemExit as exc:
        return cast("int | None", exc.code)
    return None


def _check_run_dinov3_verify_frame_alignment_happy_path() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index),
        )
    ok = exit_code in (None, 0)
    return _check(
        "run_dinov3_verify_frame_alignment: caminho feliz sintético não "
        "reporta falha", ok
    )


def _check_run_dinov3_verify_frame_alignment_shifted() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index + 1),
        )
    ok = exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: quadro decodificado deslocado "
        "de uma posição é reportado como falha", ok
    )


class _FinelySpacedFakeBackbone:
    """Backbone sintético cujo valor de saída difere entre posições vizinhas
    por apenas alguns ULPs de float16 na magnitude de `base`, em vez de um
    passo cheio de intensidade de pixel — simula um deslocamento de quadro
    sutil o bastante para passar na checagem de tolerância isoladamente."""

    def __init__(
        self, reference_means: list[float], *, base: float, fine_step: float
    ) -> None:
        self._reference_means = reference_means
        self._base = base
        self._fine_step = fine_step

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean_pixel = x.mean(dim=(1, 2, 3))
        batch_size = x.shape[0]
        outputs = [
            self._base
            + self._fine_step
            * min(
                range(len(self._reference_means)),
                key=lambda i: abs(self._reference_means[i] - value),
            )
            for value in mean_pixel.tolist()
        ]
        output_tensor = torch.tensor(outputs, dtype=torch.float32)
        cls_token = output_tensor.view(batch_size, 1).expand(batch_size, 768).clone()
        patch_tokens = (
            output_tensor.view(batch_size, 1, 1).expand(batch_size, 4, 768).clone()
        )
        return {"x_norm_clstoken": cls_token, "x_norm_patchtokens": patch_tokens}


def _build_fine_grid_frame_alignment_fixture(
    root: Path, *, video_id: str, k: int, backbone: Dinov3Backbone
) -> _SyntheticDatasetAdapter:
    env, _, video_name = video_id.partition("/")
    raw_dir = root / "raw"
    manifest_path = root / "manifest.parquet"
    frames_path = root / "frames.parquet"
    dinov3_root = root / "dinov3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "video_id": [video_id] * k,
            "frame_index": list(range(k)),
            "src_index": list(range(k)),
        }
    ).to_parquet(frames_path)
    pd.DataFrame(
        {"video_id": [video_id], "relative_path": [f"{env}/{video_name}.avi"]}
    ).to_parquet(manifest_path)

    stored = np.zeros((k, FEATURE_DIM), dtype=np.float16)
    for position in range(k):
        frame = _frame_for_src_index(position)
        batch = preprocess_frames([frame])
        stored[position] = compute_features(backbone, batch)[0]
    write_dinov3_features_atomic(
        dinov3_path(video_id, dinov3_root=dinov3_root), stored, {"K": k}
    )

    return _SyntheticDatasetAdapter(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        frames_path=frames_path,
        dinov3_root=dinov3_root,
    )


def _check_run_dinov3_verify_frame_alignment_shifted_within_tolerance() -> bool:
    k = 3
    base = float(np.float16(5.0))
    fine_step = float(np.spacing(np.float16(5.0)))
    reference_means = [
        float(preprocess_frames([_frame_for_src_index(position)]).mean().item())
        for position in range(k)
    ]
    backbone = _FinelySpacedFakeBackbone(reference_means, base=base, fine_step=fine_step)

    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_fine_grid_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=k, backbone=backbone
        )
        stored = storage.read_features(
            dinov3_path(video_id, dinov3_root=adapter.dinov3_root)
        )

        shifted_batch = preprocess_frames([_frame_for_src_index(2)])
        shifted_fresh = compute_features(backbone, shifted_batch)[0]
        tolerance_alone_passes, _, _ = max_abs_diff_within_tolerance(
            shifted_fresh, stored[1]
        )

        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=backbone,
            decode_single_frame=lambda path, index: _frame_for_src_index(index + 1),
        )

    ok = tolerance_alone_passes and exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: deslocamento de um quadro sutil "
        "(poucos ULPs de float16) passa isoladamente na checagem de "
        "tolerância, mas ainda é pego pela checagem discriminativa por "
        "ficar mais próximo da linha vizinha do que da posição correta", ok
    )


def _check_run_dinov3_verify_frame_alignment_k_mismatch() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5, h5_row_count=4
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index),
        )
    ok = exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: K do .h5 divergente de "
        "frames.parquet é reportado como falha, sem exceção não tratada", ok
    )


def _check_run_dinov3_verify_frame_alignment_missing_manifest_row() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5, include_manifest_row=False
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index),
        )
    ok = exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: video_id ausente do manifesto é "
        "acumulado como falha, sem KeyError não tratado", ok
    )


def _check_run_dinov3_verify_frame_alignment_missing_h5() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        video_id = "env1/video1"
        adapter = _build_frame_alignment_fixture(
            Path(temporary_dir), video_id=video_id, k=5, include_h5=False
        )
        exit_code = _run_frame_alignment_and_get_exit_code(
            adapter=adapter,
            video_ids=(video_id,),
            backbone=_IndexAwareFakeBackbone(),
            decode_single_frame=lambda path, index: _frame_for_src_index(index),
        )
    ok = exit_code == 1
    return _check(
        "run_dinov3_verify_frame_alignment: .h5 ausente é acumulado como "
        "falha, sem exceção não tratada de leitura", ok
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
            rejects(lambda: run_dinov3_report(cv_adapter)),
            rejects(lambda: run_dinov3_audit(adapter=cv_adapter)),
            rejects(
                lambda: run_dinov3_verify_determinism(
                    "env1/video1",
                    adapter=cv_adapter,
                    repo_dir_value=None,
                    weights_path_value=None,
                )
            ),
            rejects(
                lambda: run_dinov3_verify_frame_alignment(
                    adapter=cv_adapter, repo_dir_value=None, weights_path_value=None
                )
            ),
        ]

        nothing_written_ok = not cv_dinov3_root.exists() and not any(
            Path(temporary_dir).iterdir()
        )

    accepts_cs = True
    try:
        ensure_dinov3_dataset_supported(Le2iDatasetAdapter())
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
        "guarda de protocolo: os seis pontos de entrada DINOv3 rejeitam o "
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


def run_dinov3_selftest() -> None:
    checks = [
        _check_preprocess_frames_shape_and_dtype(),
        _check_compute_features_formula(),
        _check_storage_round_trip(),
        _check_determinism_plumbing(),
        _check_audit_helpers(),
        _check_provenance_divergences(),
        _check_provenance_divergences_missing_attribute(),
        _check_provenance_divergences_missing_from_every_file(),
        _check_validate_existing_file(),
        _check_configure_deterministic_inference_does_not_seed_rng(),
        _check_grid_positions(),
        _check_max_abs_diff_within_tolerance(),
        _check_check_discriminative_match(),
        _check_run_dinov3_verify_frame_alignment_happy_path(),
        _check_run_dinov3_verify_frame_alignment_shifted(),
        _check_run_dinov3_verify_frame_alignment_shifted_within_tolerance(),
        _check_run_dinov3_verify_frame_alignment_k_mismatch(),
        _check_run_dinov3_verify_frame_alignment_missing_manifest_row(),
        _check_run_dinov3_verify_frame_alignment_missing_h5(),
        _check_adapter_with_dinov3_root(),
        _check_dinov3_entry_points_reject_le2i_cv(),
        _check_resolve_verify_determinism_output_root(),
    ]
    if not all(checks):
        print("\ndinov3 extract selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\ndinov3 extract selftest OK: todas as checagens passaram")
