"""Selftest sintético da extração SAM 3 (descritores, seleção, armazenamento).

Não toca em GPU, dataset real nem no runtime isolado `sam3_runtime/` — todas
as entradas são sintéticas e o segmentador é sempre um fake injetado via
`segmenter=`, nunca o `Sam3RuntimeSegmenter` real.
"""

import os
import re
import sys
import tempfile
import tomllib
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pandas as pd

import gatefall.sam3.extract as sam3_extract
from gatefall.data.frames import read_frames
from gatefall.data.manifest import read_manifest
from gatefall.datasets.le2i import LE2I_LABEL_NAMES, Le2iDatasetAdapter
from gatefall.sam3 import storage
from gatefall.sam3.dataset_guard import ensure_sam3_dataset_supported
from gatefall.sam3.descriptors import (
    V_T_DIM,
    ZERO_DESCRIPTOR,
    compute_descriptor,
)
from gatefall.sam3.extract import (
    Sam3ExtractError,
    run_frames_through_segmenter,
    run_sam3_extract,
    run_sam3_extract_all,
)
from gatefall.sam3.frame_alignment import run_sam3_verify_frame_alignment
from gatefall.sam3.report import find_provenance_divergences, run_sam3_report
from gatefall.sam3.runtime import (
    CHECKPOINT_PATH_ENV_VAR,
    RUNTIME_DIR_ENV_VAR,
    Sam3Instance,
    build_worker_invocation,
    ensure_sam3_runtime_available,
    resolve_checkpoint_path,
    resolve_runtime_project_dir,
)
from gatefall.sam3.storage import (
    Sam3StorageError,
    sam3_path,
    validate_existing_file,
    verify_written_file,
    write_sam3_atomic,
)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


@dataclass(frozen=True)
class _SyntheticDatasetAdapter:
    raw_dir: Path
    manifest_path: Path
    frames_path: Path
    sam3_root: Path
    pose_root: Path = Path("pose")
    pose_stats_path: Path = Path("pose_stats.json")
    dinov3_root: Path = Path("dinov3")
    quality_root: Path = Path("quality")
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


class _ScriptedSam3Segmenter:
    """Segmentador fake dirigido por script: um quadro por chamada, na ordem."""

    def __init__(self, script: list[list[Sam3Instance]]) -> None:
        self._script = script
        self._call_count = 0

    def segment_frame(self, frame_rgb: np.ndarray, text_prompt: str) -> list[Sam3Instance]:
        instances = self._script[self._call_count]
        self._call_count += 1
        return instances


def _rect_mask(
    x_min: int, y_min: int, x_max: int, y_max: int, *, height: int, width: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[y_min : y_max + 1, x_min : x_max + 1] = True
    return mask


def _build_fixture(
    root: Path, *, video_id: str, k: int, width: int = 100, height: int = 50
) -> _SyntheticDatasetAdapter:
    env, _, video_name = video_id.partition("/")
    raw_dir = root / "raw"
    manifest_path = root / "manifest.parquet"
    frames_path = root / "frames.parquet"
    sam3_root = root / "sam3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "video_id": [video_id] * k,
            "frame_index": list(range(k)),
            "src_index": list(range(k)),
        }
    ).to_parquet(frames_path)
    pd.DataFrame(
        {
            "video_id": [video_id],
            "relative_path": [f"{env}/{video_name}.avi"],
            "env": [env],
            "split": ["train"],
            "subject": [1],
            "fps": [10.0],
            "width": [width],
            "height": [height],
        }
    ).to_parquet(manifest_path)

    return _SyntheticDatasetAdapter(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        frames_path=frames_path,
        sam3_root=sam3_root,
    )


def _install_fake_decode_frames(
    frames_by_src: dict[int, np.ndarray]
) -> Callable[[], None]:
    original = sam3_extract.decode_frames

    def fake_decode_frames(video_path: Path, src_indices: list[int]) -> list[np.ndarray]:
        return [frames_by_src[i] for i in src_indices]

    sam3_extract.decode_frames = fake_decode_frames

    def _restore() -> None:
        sam3_extract.decode_frames = original

    return _restore


def _check_compute_descriptor_rectangle() -> bool:
    width, height = 100, 50
    mask = _rect_mask(10, 5, 29, 14, height=height, width=width)
    descriptor = compute_descriptor(mask, frame_width=width, frame_height=height)

    m00 = 20 * 10
    xbar, ybar = 19.5, 9.5
    mu20 = (20 ** 2 - 1) / 12
    mu02 = (10 ** 2 - 1) / 12
    r = abs(mu20 - mu02)
    expected = np.array(
        [
            1.0,
            m00 / (width * height),
            xbar / width,
            ybar / height,
            20 / width,
            10 / height,
            m00 / (20 * 10),
            float(np.sqrt(1.0 - (min(mu20, mu02) / max(mu20, mu02)))),
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )
    ok = bool(np.allclose(descriptor, expected, atol=1e-5))
    return _check(
        "compute_descriptor: retângulo sintético dá área/centroide/bbox/"
        "fill_ratio/excentricidade/ângulo com valores calculados à mão", ok
    )


def _check_compute_descriptor_square_is_isotropic() -> bool:
    width, height = 60, 60
    mask = _rect_mask(10, 10, 29, 29, height=height, width=width)
    descriptor = compute_descriptor(mask, frame_width=width, frame_height=height)
    ok = (
        abs(float(descriptor[7])) < 1e-6
        and abs(float(descriptor[8])) < 1e-6
        and abs(float(descriptor[9])) < 1e-6
    )
    return _check(
        "compute_descriptor: máscara quadrada (isotrópica) dá excentricidade "
        "e sin/cos_2theta exatamente 0", ok
    )


def _check_compute_descriptor_empty_mask() -> bool:
    empty_mask = np.zeros((50, 100), dtype=bool)
    ok = bool(np.array_equal(
        compute_descriptor(empty_mask, frame_width=100, frame_height=50), ZERO_DESCRIPTOR
    )) and bool(np.array_equal(
        compute_descriptor(None, frame_width=100, frame_height=50), ZERO_DESCRIPTOR
    ))
    return _check(
        "compute_descriptor: máscara vazia ou ausência de detecção dão "
        "exatamente ZERO_DESCRIPTOR (inclusive o canal 'present')", ok
    )


def _check_missing_frame_is_isolated_zero_no_forward_fill() -> bool:
    width, height = 100, 50
    mask_a = _rect_mask(10, 5, 19, 14, height=height, width=width)
    mask_b = _rect_mask(50, 20, 59, 29, height=height, width=width)
    script = [
        [Sam3Instance(mask=mask_a, score=0.9)],
        [],
        [Sam3Instance(mask=mask_b, score=0.9)],
    ]
    segmenter = _ScriptedSam3Segmenter(script)
    frames_rgb = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(3)]
    v_t, sam_score, n_instances = run_frames_through_segmenter(
        frames_rgb, segmenter=segmenter, width=width, height=height
    )
    ok = (
        bool(np.array_equal(v_t[1], ZERO_DESCRIPTOR))
        and not bool(np.array_equal(v_t[1], v_t[0]))
        and not bool(np.array_equal(v_t[1], v_t[2]))
        and float(sam_score[1]) == 0.0
        and int(n_instances[1]) == 0
    )
    return _check(
        "sequência presente/ausente/presente: o quadro sem detecção é "
        "exatamente zero e diferente dos dois vizinhos (sem forward-fill)", ok
    )


def _check_end_to_end_continuity_tracks_moving_instance() -> bool:
    # Fixture puramente de máscaras sintéticas — nenhum artefato de pose é
    # criado ou lido neste teste, condizente com V_t não depender de pose.
    width, height = 100, 60
    video_id = "coffee_room/video_continuity"
    k = 4

    target_boxes = [
        (10, 5, 19, 14),
        (12, 5, 21, 14),
        (14, 5, 23, 14),
        (16, 5, 25, 14),
    ]
    target_scores = [0.9, 0.4, 0.3, 0.2]
    distractor_boxes = [
        (70, 30, 79, 39),
        (71, 30, 80, 39),
        (72, 30, 81, 39),
        (73, 30, 82, 39),
    ]
    distractor_scores = [0.5, 0.99, 0.99, 0.99]

    script = [
        [
            Sam3Instance(
                mask=_rect_mask(*target_boxes[i], height=height, width=width),
                score=target_scores[i],
            ),
            Sam3Instance(
                mask=_rect_mask(*distractor_boxes[i], height=height, width=width),
                score=distractor_scores[i],
            ),
        ]
        for i in range(k)
    ]
    segmenter = _ScriptedSam3Segmenter(script)

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        adapter = _build_fixture(root, video_id=video_id, k=k, width=width, height=height)
        frames_by_src = {i: np.zeros((height, width, 3), dtype=np.uint8) for i in range(k)}
        restore = _install_fake_decode_frames(frames_by_src)
        try:
            result = run_sam3_extract(
                video_id,
                adapter=adapter,
                segmenter=segmenter,
                runtime_project_dir_value=str(root / "sam3_runtime_missing"),
                checkpoint_path_value=str(root / "checkpoint_missing.pt"),
            )
        finally:
            restore()

        output_path = sam3_path(video_id, sam3_root=adapter.sam3_root)
        stored_v_t = storage.read_v_t(output_path)
        stored_sam_score = storage.read_sam_score(output_path)

        with h5py.File(output_path, "r") as h5_file:
            checkpoint_attr = str(h5_file.attrs["sam3_checkpoint_sha256"])
            lock_attr = str(h5_file.attrs["sam3_runtime_lock_sha256"])

        frames_parquet = read_frames(adapter.frames_path)
        k_matches = result.k == len(frames_parquet) == k
    scores_track_target = bool(np.allclose(stored_sam_score, target_scores, atol=1e-6))
    expected_centroid_x = np.array(
        [((b[0] + b[2]) / 2.0) / width for b in target_boxes], dtype=np.float32
    )
    centroids_track_target = bool(np.allclose(stored_v_t[:, 2], expected_centroid_x, atol=1e-5))
    provenance_sentinels_ok = checkpoint_attr == "" and lock_attr == ""

    ok = (
        k_matches
        and scores_track_target
        and centroids_track_target
        and provenance_sentinels_ok
    )
    return _check(
        "fim a fim: instância contínua (score mais baixo) é seguida em vez "
        "do distrator (score mais alto e sem overlap), K bate com "
        "frames.parquet e sam3_checkpoint_sha256/sam3_runtime_lock_sha256 "
        "degradam para '' quando os arquivos reais não existem", ok
    )


def _check_storage_round_trip() -> bool:
    v_t = np.random.randn(5, V_T_DIM).astype(np.float32)
    sam_score = np.random.rand(5).astype(np.float32)
    n_instances = np.array([1, 2, 0, 1, 3], dtype=np.int16)
    attrs: dict[str, object] = {
        "video_id": "synthetic_env/synthetic_video",
        "K": 5,
        "model_name": "facebook/sam3",
    }

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "synthetic_env" / "synthetic_video.h5"
        write_sam3_atomic(path, v_t, sam_score, n_instances, attrs)
        try:
            verify_written_file(
                path, v_t=v_t, sam_score=sam_score, n_instances=n_instances, attrs=attrs
            )
            round_trip_ok = True
        except Sam3StorageError:
            round_trip_ok = False

        mismatched_v_t = np.random.randn(5, V_T_DIM).astype(np.float32)
        try:
            verify_written_file(
                path, v_t=mismatched_v_t, sam_score=sam_score, n_instances=n_instances, attrs=attrs
            )
            mismatch_caught = False
        except Sam3StorageError:
            mismatch_caught = True

    ok = round_trip_ok and mismatch_caught
    return _check(
        "storage: escrita atômica + releitura confere v_t/sam_score/"
        "n_instances e atributos bit-a-bit, e divergência é detectada", ok
    )


def _check_validate_existing_file() -> bool:
    v_t = np.random.randn(5, V_T_DIM).astype(np.float32)
    sam_score = np.zeros(5, dtype=np.float32)
    n_instances = np.zeros(5, dtype=np.int16)
    expected_attrs: dict[str, object] = {"model_name": "facebook/sam3"}
    attrs: dict[str, object] = {"K": 5, **expected_attrs}

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "synthetic_env" / "synthetic_video.h5"
        write_sam3_atomic(path, v_t, sam_score, n_instances, attrs)

        valid_reasons = validate_existing_file(
            path, expected_k=5, v_t_dim=V_T_DIM, expected_attrs=expected_attrs
        )
        valid_ok = valid_reasons == []

        wrong_k_reasons = validate_existing_file(
            path, expected_k=6, v_t_dim=V_T_DIM, expected_attrs=expected_attrs
        )
        wrong_k_ok = len(wrong_k_reasons) > 0

        wrong_attr_reasons = validate_existing_file(
            path,
            expected_k=5,
            v_t_dim=V_T_DIM,
            expected_attrs={"model_name": "different_model"},
        )
        wrong_attr_ok = (
            len(wrong_attr_reasons) > 0 and "model_name" in wrong_attr_reasons[0]
        )

    with tempfile.TemporaryDirectory() as temporary_dir:
        wrong_n_instances_dtype_path = Path(temporary_dir) / "synthetic_env" / "video.h5"
        write_sam3_atomic(
            wrong_n_instances_dtype_path,
            v_t,
            sam_score,
            n_instances.astype(np.int32),
            attrs,
        )
        wrong_n_instances_dtype_reasons = validate_existing_file(
            wrong_n_instances_dtype_path,
            expected_k=5,
            v_t_dim=V_T_DIM,
            expected_attrs=expected_attrs,
        )
        wrong_n_instances_dtype_ok = (
            len(wrong_n_instances_dtype_reasons) > 0
            and any("n_instances" in reason for reason in wrong_n_instances_dtype_reasons)
        )

    with tempfile.TemporaryDirectory() as temporary_dir:
        wrong_sam_score_dtype_path = Path(temporary_dir) / "synthetic_env" / "video.h5"
        write_sam3_atomic(
            wrong_sam_score_dtype_path,
            v_t,
            sam_score.astype(np.float64),
            n_instances,
            attrs,
        )
        wrong_sam_score_dtype_reasons = validate_existing_file(
            wrong_sam_score_dtype_path,
            expected_k=5,
            v_t_dim=V_T_DIM,
            expected_attrs=expected_attrs,
        )
        wrong_sam_score_dtype_ok = (
            len(wrong_sam_score_dtype_reasons) > 0
            and any("sam_score" in reason for reason in wrong_sam_score_dtype_reasons)
        )

    ok = (
        valid_ok
        and wrong_k_ok
        and wrong_attr_ok
        and wrong_n_instances_dtype_ok
        and wrong_sam_score_dtype_ok
    )
    return _check(
        "validate_existing_file: arquivo válido dá [], shape divergente, "
        "atributo divergente e dtype divergente de sam_score/n_instances "
        "dão razões nomeadas", ok
    )


def _full_provenance_attrs(**overrides: object) -> dict[str, object]:
    attrs: dict[str, object] = {
        name: f"valor-{name}" for name in storage.PROVENANCE_ATTR_NAMES
    }
    attrs.update(overrides)
    return attrs


def _check_provenance_divergence_heterogeneous_checkpoint() -> bool:
    homogeneous: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(),
        "env1/v2": _full_provenance_attrs(),
    }
    homogeneous_ok = find_provenance_divergences(homogeneous) == []

    heterogeneous: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(),
        "env1/v2": _full_provenance_attrs(sam3_checkpoint_sha256="different"),
    }
    divergences = find_provenance_divergences(heterogeneous)
    heterogeneous_ok = len(divergences) == 1 and "env1/v2" in divergences[0]

    ok = homogeneous_ok and heterogeneous_ok
    return _check(
        "find_provenance_divergences: dataset homogêneo dá [] e "
        "sam3_checkpoint_sha256 divergente entre dois vídeos sintéticos é "
        "nomeado", ok
    )


def _check_dataset_guard_rejects_le2i_cv() -> bool:
    def rejects(call: Callable[[], object]) -> bool:
        try:
            call()
        except ValueError as exc:
            return "le2i-cv" in str(exc)
        except Exception:
            return False
        return False

    with tempfile.TemporaryDirectory() as temporary_dir:
        cv_sam3_root = Path(temporary_dir) / "sam3"
        cv_adapter = Le2iDatasetAdapter(protocol="cv", sam3_root=cv_sam3_root)

        rejected = [
            rejects(lambda: run_sam3_extract("env1/video1", adapter=cv_adapter)),
            rejects(
                lambda: run_sam3_extract_all(
                    adapter=cv_adapter,
                    runtime_project_dir_value=None,
                    checkpoint_path_value=None,
                )
            ),
            rejects(lambda: run_sam3_report(cv_adapter)),
            rejects(
                lambda: run_sam3_verify_frame_alignment(
                    adapter=cv_adapter,
                    runtime_project_dir_value=None,
                    checkpoint_path_value=None,
                )
            ),
        ]

        nothing_written_ok = not cv_sam3_root.exists() and not any(
            Path(temporary_dir).iterdir()
        )

    accepts_cs = True
    try:
        ensure_sam3_dataset_supported(Le2iDatasetAdapter())
        ensure_sam3_dataset_supported(
            _SyntheticDatasetAdapter(
                raw_dir=Path("raw"),
                manifest_path=Path("manifest.parquet"),
                frames_path=Path("frames.parquet"),
                sam3_root=Path("sam3"),
            )
        )
    except Exception:
        accepts_cs = False

    return _check(
        "guarda de protocolo: os quatro pontos de entrada SAM 3 rejeitam o "
        "adapter le2i-cv com ValueError sem criar nada sob "
        "adapter.sam3_root, e a guarda continua aceitando qualquer adapter "
        "com identifier 'le2i' (checagem por identifier, não por isinstance)",
        all(rejected) and nothing_written_ok and accepts_cs,
    )


def _check_build_worker_invocation_project_equals_cwd() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        (root / "sam3_runtime").mkdir()
        original_cwd = Path.cwd()
        os.chdir(root)
        try:
            argv, cwd = build_worker_invocation(
                Path("sam3_runtime"), Path("checkpoint.pt")
            )
        finally:
            os.chdir(original_cwd)

        project_value = argv[argv.index("--project") + 1]
        ok = project_value == cwd == str((root / "sam3_runtime").resolve())
    return _check(
        "build_worker_invocation: com entradas relativas, --project e cwd "
        "são exatamente o mesmo caminho absoluto", ok
    )


def _check_build_worker_invocation_checkpoint_absolute_independent_of_cwd() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        (root / "sam3_runtime").mkdir()
        original_cwd = Path.cwd()
        os.chdir(root)
        try:
            argv, cwd = build_worker_invocation(
                Path("sam3_runtime"), Path("weights/checkpoint.pt")
            )
        finally:
            os.chdir(original_cwd)

        checkpoint_value = argv[argv.index("--checkpoint") + 1]
        ok = (
            Path(checkpoint_value).is_absolute()
            and checkpoint_value == str((root / "weights/checkpoint.pt").resolve())
            and checkpoint_value != cwd
        )
    return _check(
        "build_worker_invocation: --checkpoint é absoluto e independente do "
        "cwd do subprocesso", ok
    )


def _check_resolve_precedence_and_absolute_paths() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        original_cwd = Path.cwd()
        os.chdir(root)
        original_runtime_env = os.environ.pop(RUNTIME_DIR_ENV_VAR, None)
        original_checkpoint_env = os.environ.pop(CHECKPOINT_PATH_ENV_VAR, None)
        try:
            default_dir = resolve_runtime_project_dir(None)
            default_checkpoint = resolve_checkpoint_path(None)
            default_ok = default_dir.is_absolute() and default_checkpoint.is_absolute()

            os.environ[RUNTIME_DIR_ENV_VAR] = "env_runtime_dir"
            os.environ[CHECKPOINT_PATH_ENV_VAR] = "env_checkpoint.pt"
            env_dir = resolve_runtime_project_dir(None)
            env_checkpoint = resolve_checkpoint_path(None)
            env_ok = (
                env_dir == (root / "env_runtime_dir").resolve()
                and env_checkpoint == (root / "env_checkpoint.pt").resolve()
                and env_dir.is_absolute()
                and env_checkpoint.is_absolute()
            )

            cli_dir = resolve_runtime_project_dir("cli_runtime_dir")
            cli_checkpoint = resolve_checkpoint_path("cli_checkpoint.pt")
            cli_ok = (
                cli_dir == (root / "cli_runtime_dir").resolve()
                and cli_checkpoint == (root / "cli_checkpoint.pt").resolve()
                and cli_dir.is_absolute()
                and cli_checkpoint.is_absolute()
            )
        finally:
            os.chdir(original_cwd)
            if original_runtime_env is None:
                os.environ.pop(RUNTIME_DIR_ENV_VAR, None)
            else:
                os.environ[RUNTIME_DIR_ENV_VAR] = original_runtime_env
            if original_checkpoint_env is None:
                os.environ.pop(CHECKPOINT_PATH_ENV_VAR, None)
            else:
                os.environ[CHECKPOINT_PATH_ENV_VAR] = original_checkpoint_env

    ok = default_ok and env_ok and cli_ok
    return _check(
        "resolve_runtime_project_dir/resolve_checkpoint_path: precedência "
        "CLI > env > padrão se mantém e todo ramo devolve caminho absoluto "
        "a partir de entrada relativa", ok
    )


def _check_ensure_sam3_runtime_available_requires_uv_lock() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        runtime_dir = root / "sam3_runtime"
        runtime_dir.mkdir()
        (runtime_dir / "run_sam3.py").write_text("")
        checkpoint_path = root / "checkpoint.pt"
        checkpoint_path.write_text("")

        missing_lock_raised = False
        try:
            ensure_sam3_runtime_available(runtime_dir, checkpoint_path)
        except FileNotFoundError as exc:
            missing_lock_raised = "uv.lock" in str(exc)

        (runtime_dir / "uv.lock").write_text("")
        cleared_after_lock_present = True
        try:
            ensure_sam3_runtime_available(runtime_dir, checkpoint_path)
        except FileNotFoundError:
            cleared_after_lock_present = False

    ok = missing_lock_raised and cleared_after_lock_present
    return _check(
        "ensure_sam3_runtime_available: uv.lock ausente levanta "
        "FileNotFoundError e sua presença limpa a checagem", ok
    )


_WELL_FORMED_REQUIRED_PROVENANCE_VALUES: dict[str, str] = {
    "sam3_checkpoint_sha256": "a" * 64,
    "sam3_runtime_lock_sha256": "b" * 64,
    "sam3_source_revision": "c" * 40,
}


def _full_required_provenance_attrs(**overrides: object) -> dict[str, object]:
    attrs: dict[str, object] = dict(_WELL_FORMED_REQUIRED_PROVENANCE_VALUES)
    attrs.update(overrides)
    return attrs


def _check_find_invalid_required_provenance() -> bool:
    complete_ok = (
        storage.find_invalid_required_provenance(
            _full_required_provenance_attrs(), storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES
        )
        == []
    )

    empty_string_reasons = storage.find_invalid_required_provenance(
        _full_required_provenance_attrs(sam3_source_revision=""),
        storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES,
    )
    empty_string_ok = empty_string_reasons == ["sam3_source_revision: vazio"]

    attrs_missing_key = _full_required_provenance_attrs()
    del attrs_missing_key["sam3_runtime_lock_sha256"]
    missing_key_reasons = storage.find_invalid_required_provenance(
        attrs_missing_key, storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES
    )
    missing_key_ok = missing_key_reasons == ["sam3_runtime_lock_sha256: ausente"]

    malformed_revision_reasons = storage.find_invalid_required_provenance(
        _full_required_provenance_attrs(
            sam3_source_revision="ERRO_RESOLVENDO_REVISAO_SAM3: LookupError('...')"
        ),
        storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES,
    )
    malformed_revision_ok = (
        len(malformed_revision_reasons) == 1
        and malformed_revision_reasons[0].startswith("sam3_source_revision: formato inválido")
    )

    malformed_checkpoint_reasons = storage.find_invalid_required_provenance(
        _full_required_provenance_attrs(sam3_checkpoint_sha256="nao-e-um-sha256"),
        storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES,
    )
    malformed_checkpoint_ok = (
        len(malformed_checkpoint_reasons) == 1
        and malformed_checkpoint_reasons[0].startswith("sam3_checkpoint_sha256: formato inválido")
    )

    ok = (
        complete_ok
        and empty_string_ok
        and missing_key_ok
        and malformed_revision_ok
        and malformed_checkpoint_ok
    )
    return _check(
        "find_invalid_required_provenance: completo e bem formado dá [], "
        "valor ausente, valor vazio e valor não vazio mas mal formado "
        "(ex.: uma mensagem de erro em vez de um sha) são nomeados "
        "separadamente com o motivo específico", ok
    )


def _check_report_detects_structurally_invalid_h5() -> bool:
    video_id = "coffee_room/video_bad"
    k = 3

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        adapter = _build_fixture(root, video_id=video_id, k=k)
        pd.DataFrame(
            {
                "video_id": [video_id] * k,
                "frame_index": list(range(k)),
                "src_index": list(range(k)),
                "split": ["train"] * k,
            }
        ).to_parquet(adapter.frames_path)
        output_path = sam3_path(video_id, sam3_root=adapter.sam3_root)

        v_t = np.zeros((k, V_T_DIM), dtype=np.float32)
        sam_score_wrong_dtype = np.zeros(k, dtype=np.float64)
        n_instances = np.zeros(k, dtype=np.int16)
        attrs = {"K": k, **_full_required_provenance_attrs(), **{
            name: f"valor-{name}" for name in storage.PROVENANCE_ATTR_NAMES
        }}
        write_sam3_atomic(output_path, v_t, sam_score_wrong_dtype, n_instances, attrs)

        captured_stdout = StringIO()
        with redirect_stdout(captured_stdout):
            try:
                run_sam3_report(adapter)
            except SystemExit:
                pass

    report_output = captured_stdout.getvalue()
    ok = "[FAIL] nenhum .h5 estruturalmente inválido" in report_output
    return _check(
        "run_sam3_report: dataset 'sam_score' com dtype divergente (float64 "
        "em vez de float32) é detectado por validate_existing_file e reprovado "
        "na checagem dedicada", ok
    )


def _check_report_rejects_malformed_source_revision() -> bool:
    video_id = "coffee_room/video_malformed_revision"
    k = 3

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        adapter = _build_fixture(root, video_id=video_id, k=k)
        pd.DataFrame(
            {
                "video_id": [video_id] * k,
                "frame_index": list(range(k)),
                "src_index": list(range(k)),
                "split": ["train"] * k,
            }
        ).to_parquet(adapter.frames_path)
        output_path = sam3_path(video_id, sam3_root=adapter.sam3_root)

        v_t = np.zeros((k, V_T_DIM), dtype=np.float32)
        sam_score = np.zeros(k, dtype=np.float32)
        n_instances = np.zeros(k, dtype=np.int16)
        attrs: dict[str, object] = {
            "K": k,
            **_full_required_provenance_attrs(
                sam3_source_revision="ERRO_RESOLVENDO_REVISAO_SAM3: LookupError('...')"
            ),
            **{
                name: f"valor-{name}"
                for name in storage.PROVENANCE_ATTR_NAMES
                if name not in storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES
            },
        }
        write_sam3_atomic(output_path, v_t, sam_score, n_instances, attrs)

        captured_stdout = StringIO()
        with redirect_stdout(captured_stdout):
            try:
                run_sam3_report(adapter)
            except SystemExit:
                pass

    report_output = captured_stdout.getvalue()
    ok = (
        "[FAIL] nenhum atributo de proveniência obrigatório ausente, vazio "
        "ou malformado" in report_output
        and "sam3_source_revision: formato inválido" in report_output
    )
    return _check(
        "run_sam3_report: sam3_source_revision não vazio mas mal formado "
        "(ex.: uma mensagem de erro em vez de um sha de commit) reprova a "
        "checagem de proveniência obrigatória em vez de passar como não vazio",
        ok,
    )


def _check_report_rejects_malformed_checkpoint_digest() -> bool:
    video_id = "coffee_room/video_malformed_checkpoint"
    k = 3

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        adapter = _build_fixture(root, video_id=video_id, k=k)
        pd.DataFrame(
            {
                "video_id": [video_id] * k,
                "frame_index": list(range(k)),
                "src_index": list(range(k)),
                "split": ["train"] * k,
            }
        ).to_parquet(adapter.frames_path)
        output_path = sam3_path(video_id, sam3_root=adapter.sam3_root)

        v_t = np.zeros((k, V_T_DIM), dtype=np.float32)
        sam_score = np.zeros(k, dtype=np.float32)
        n_instances = np.zeros(k, dtype=np.int16)
        attrs: dict[str, object] = {
            "K": k,
            **_full_required_provenance_attrs(sam3_checkpoint_sha256="nao-e-um-sha256"),
            **{
                name: f"valor-{name}"
                for name in storage.PROVENANCE_ATTR_NAMES
                if name not in storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES
            },
        }
        write_sam3_atomic(output_path, v_t, sam_score, n_instances, attrs)

        captured_stdout = StringIO()
        with redirect_stdout(captured_stdout):
            try:
                run_sam3_report(adapter)
            except SystemExit:
                pass

    report_output = captured_stdout.getvalue()
    ok = (
        "[FAIL] nenhum atributo de proveniência obrigatório ausente, vazio "
        "ou malformado" in report_output
        and "sam3_checkpoint_sha256: formato inválido" in report_output
    )
    return _check(
        "run_sam3_report: sam3_checkpoint_sha256 não vazio mas com "
        "comprimento/alfabeto diferente de um sha256 hexadecimal reprova a "
        "checagem de proveniência obrigatória", ok
    )


def _check_missing_video_id_raises_extract_error() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        adapter = _build_fixture(root, video_id="env1/video1", k=3)
        try:
            run_sam3_extract(
                "env1/nao_existe",
                adapter=adapter,
                segmenter=_ScriptedSam3Segmenter([]),
            )
            raised = False
        except Sam3ExtractError:
            raised = True
    return _check(
        "run_sam3_extract: video_id ausente de frames.parquet levanta "
        "Sam3ExtractError sem tocar no disco", raised
    )


_SETUPTOOLS_PKG_RESOURCES_REMOVAL_VERSION = (82, 0, 0)


def _parse_leading_version_tuple(text: str) -> tuple[int, ...] | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _pad_version_tuples(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    length = max(len(left), len(right))
    return (
        left + (0,) * (length - len(left)),
        right + (0,) * (length - len(right)),
    )


def _version_lt(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    padded_left, padded_right = _pad_version_tuples(left, right)
    return padded_left < padded_right


def _version_le(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    padded_left, padded_right = _pad_version_tuples(left, right)
    return padded_left <= padded_right


def _split_leading_distribution_name(clause: str) -> tuple[str, str]:
    # O PEP 508 só anexa o nome da distribuição à primeira cláusula separada
    # por vírgula; separamos esse prefixo (e extras opcionais, ex.: `[foo]`)
    # do restante para identificar a distribuição e testar o operador em
    # qualquer posição da cláusula.
    if clause[:1] in "<>=!~":
        return "", clause
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\s*\[[^\]]*\])?\s*", clause)
    if match is None:
        return "", clause
    return match.group(1), clause[match.end():]


def _normalize_distribution_name(name: str) -> str:
    # PEP 503: nomes de distribuição são comparados em minúsculas com
    # sequências de `-`, `_` e `.` colapsadas em um único `-`.
    return re.sub(r"[-_.]+", "-", name).lower()


def _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
    pyproject_path: Path,
) -> bool:
    try:
        with pyproject_path.open("rb") as pyproject_file:
            pyproject_data = tomllib.load(pyproject_file)
    except (OSError, tomllib.TOMLDecodeError):
        return False

    tool_table = pyproject_data.get("tool", {})
    if not isinstance(tool_table, dict):
        return False
    uv_table = tool_table.get("uv", {})
    if not isinstance(uv_table, dict):
        return False
    constraints = uv_table.get("constraint-dependencies", [])
    if not isinstance(constraints, list):
        return False
    for constraint in constraints:
        if not isinstance(constraint, str):
            continue
        clauses = [part.strip() for part in constraint.split(",")]
        if not clauses:
            continue
        distribution_name, remainder = _split_leading_distribution_name(clauses[0])
        if _normalize_distribution_name(distribution_name) != "setuptools":
            continue
        clauses[0] = remainder
        for clause in clauses:
            if clause.startswith("<="):
                version = _parse_leading_version_tuple(clause[2:])
                if version is not None and _version_lt(
                    version, _SETUPTOOLS_PKG_RESOURCES_REMOVAL_VERSION
                ):
                    return True
            elif clause.startswith("<"):
                version = _parse_leading_version_tuple(clause[1:])
                if version is not None and _version_le(
                    version, _SETUPTOOLS_PKG_RESOURCES_REMOVAL_VERSION
                ):
                    return True
    return False


def _lock_resolves_setuptools_below_pkg_resources_removal(lock_path: Path) -> bool:
    try:
        with lock_path.open("rb") as lock_file:
            lock_data = tomllib.load(lock_file)
    except (OSError, tomllib.TOMLDecodeError):
        return False

    packages = lock_data.get("package", [])
    if not isinstance(packages, list):
        return False
    # `uv lock` pode gerar mais de um stanza `setuptools` quando a resolução
    # se bifurca por `resolution-markers` (versões de Python/plataformas
    # diferentes); avaliar só o primeiro stanza encontrado deixaria passar um
    # lock em que um stanza fica abaixo do teto e outro acima.
    found_setuptools_stanza = False
    for package in packages:
        if not isinstance(package, dict):
            continue
        if package.get("name") != "setuptools":
            continue
        found_setuptools_stanza = True
        version = _parse_leading_version_tuple(str(package.get("version", "")))
        if version is None:
            return False
        if not _version_lt(version, _SETUPTOOLS_PKG_RESOURCES_REMOVAL_VERSION):
            return False
    return found_setuptools_stanza


def _write_pyproject_with_constraint(root: Path, constraint: str) -> Path:
    path = root / "pyproject.toml"
    path.write_text(
        "[tool.uv]\n"
        f'constraint-dependencies = ["{constraint}"]\n'
    )
    return path


def _write_lock_with_setuptools_version(root: Path, version: str) -> Path:
    path = root / "uv.lock"
    path.write_text(
        "[[package]]\n"
        'name = "setuptools"\n'
        f'version = "{version}"\n'
    )
    return path


def _write_lock_with_two_setuptools_versions(
    root: Path, first_version: str, second_version: str
) -> Path:
    path = root / "uv.lock"
    path.write_text(
        "[[package]]\n"
        'name = "setuptools"\n'
        f'version = "{first_version}"\n'
        "[[package]]\n"
        'name = "setuptools"\n'
        f'version = "{second_version}"\n'
    )
    return path


def _write_lock_with_no_setuptools_stanza(root: Path) -> Path:
    path = root / "uv.lock"
    path.write_text(
        "[[package]]\n"
        'name = "torch"\n'
        'version = "2.0.0"\n'
    )
    return path


def _check_setuptools_ceiling_matches_distribution_name_not_substring() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        lookalikes_rejected = all(
            not _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, constraint)
            )
            for constraint in (
                "setuptools-scm<82",
                "setuptools_scm<82",
                "setuptools.scm<82",
                "mysetuptools<2",
                "setuptools--scm<82",
                "setuptools__scm<82",
                "setuptools-_.scm<82",
            )
        )
        # Regressão do bug de indexação de cláusula: só a cláusula 0 carrega o
        # nome da distribuição (PEP 508); um token de outra distribuição nas
        # cláusulas seguintes não pode ser lido como teto do setuptools.
        clause_indexing_false_positives_rejected = all(
            not _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, constraint)
            )
            for constraint in (
                "setuptools>=200,foopkg<82",
                "setuptools>=77.0.3,otherpkg<82",
            )
        )
        positives_accepted = all(
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, constraint)
            )
            for constraint in (
                "setuptools>=77.0.3,<82",
                "setuptools<82,>=77.0.3",
                "setuptools<82",
                "setuptools >= 77.0.3 , < 82",
                "setuptools[foo]<82",
                "SetupTools<82",
                "SETUPTOOLS<82",
                "  setuptools<82  ",
                "  setuptools>=77.0.3,<82  ",
            )
        )
        unsafe_ceiling_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, "setuptools>=77.0.3,<=82")
            )
        )
        absent_ceiling_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, "setuptools>=77.0.3")
            )
        )
        direct_reference_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(
                    root, "setuptools @ https://example.invalid/setuptools.whl"
                )
            )
        )
        no_version_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, "setuptools[foo]")
            )
        )
        empty_string_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, "")
            )
        )
        non_string_entry_path = root / "pyproject.toml"
        non_string_entry_path.write_text(
            "[tool.uv]\nconstraint-dependencies = [1]\n"
        )
        non_string_entry_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                non_string_entry_path
            )
        )
        lookalikes_only_list_path = root / "pyproject.toml"
        lookalikes_only_list_path.write_text(
            "[tool.uv]\n"
            'constraint-dependencies = ["setuptools-scm<82", "mysetuptools<2"]\n'
        )
        lookalikes_only_list_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                lookalikes_only_list_path
            )
        )
    ok = (
        lookalikes_rejected
        and clause_indexing_false_positives_rejected
        and positives_accepted
        and unsafe_ceiling_rejected
        and absent_ceiling_rejected
        and direct_reference_rejected
        and no_version_rejected
        and empty_string_rejected
        and non_string_entry_rejected
        and lookalikes_only_list_rejected
    )
    return _check(
        "_pyproject_declares_setuptools_ceiling_below_pkg_resources_removal: "
        "identifica a distribuição pelo nome normalizado (PEP 503), não por "
        "substring — nomes parecidos (setuptools-scm, setuptools_scm, "
        "setuptools.scm, mysetuptools, e variantes com separadores repetidos "
        "como setuptools--scm/setuptools__scm/setuptools-_.scm) não "
        "satisfazem o teto, uma cláusula 2+ com nome de outra distribuição "
        "(setuptools>=200,foopkg<82 / setuptools>=77.0.3,otherpkg<82) não é "
        "lida como teto do setuptools, as variantes reais (reordenada, "
        "cláusula única, com espaços, com extras, com maiúsculas/minúsculas "
        "trocadas, com espaços nas bordas) continuam passando, e um teto "
        "ausente, <=82, uma referência direta (`@ url`), extras sem versão, "
        "string vazia, entrada não textual e uma lista só com nomes "
        "parecidos continuam reprovando", ok
    )


def _check_setuptools_lock_version_boundary_and_corruption() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        version_82_rejected = not _lock_resolves_setuptools_below_pkg_resources_removal(
            _write_lock_with_setuptools_version(root, "82.0.0")
        )
        version_81_accepted = _lock_resolves_setuptools_below_pkg_resources_removal(
            _write_lock_with_setuptools_version(root, "81.0.0")
        )
        corrupted_lock_path = root / "uv.lock"
        corrupted_lock_path.write_text('package = "oops"\n')
        corrupted_rejected = not _lock_resolves_setuptools_below_pkg_resources_removal(
            corrupted_lock_path
        )
        no_setuptools_stanza_rejected = (
            not _lock_resolves_setuptools_below_pkg_resources_removal(
                _write_lock_with_no_setuptools_stanza(root)
            )
        )
        # Regressão do bug de "primeiro stanza só": `uv lock` pode gerar mais
        # de um stanza `setuptools` quando a resolução se bifurca por
        # `resolution-markers` (Python/plataforma diferentes). Um stanza
        # abaixo do teto e outro acima deve reprovar, em qualquer ordem.
        two_stanzas_low_then_high_rejected = (
            not _lock_resolves_setuptools_below_pkg_resources_removal(
                _write_lock_with_two_setuptools_versions(root, "81.0.0", "84.0.0")
            )
        )
        two_stanzas_high_then_low_rejected = (
            not _lock_resolves_setuptools_below_pkg_resources_removal(
                _write_lock_with_two_setuptools_versions(root, "84.0.0", "81.0.0")
            )
        )
    ok = (
        version_82_rejected
        and version_81_accepted
        and corrupted_rejected
        and no_setuptools_stanza_rejected
        and two_stanzas_low_then_high_rejected
        and two_stanzas_high_then_low_rejected
    )
    return _check(
        "_lock_resolves_setuptools_below_pkg_resources_removal: versão "
        "82.0.0 reprova, 81.0.0 passa, lock corrompido (`package` fora do "
        "formato de lista de tabelas) reprova sem levantar exceção, lock sem "
        "nenhum stanza setuptools reprova, e dois stanzas setuptools "
        "(bifurcação por resolution-markers) com um deles >= 82 reprovam em "
        "qualquer ordem — não basta o primeiro stanza encontrado passar", ok
    )


def _check_normalize_distribution_name_collapses_separator_runs() -> bool:
    # PEP 503 colapsa QUALQUER sequência de `-`, `_` e `.` em um único `-`;
    # esta asserção fixa esse comportamento (e não apenas separadores
    # isolados), pegando uma regressão de `[-_.]+` para `[-_.]`.
    ok = (
        _normalize_distribution_name("setuptools__scm")
        == _normalize_distribution_name("setuptools-scm")
        == _normalize_distribution_name("setuptools-_.scm")
        == "setuptools-scm"
    )
    return _check(
        "_normalize_distribution_name: sequências de separadores repetidos "
        "ou mistos (setuptools__scm, setuptools-_.scm) colapsam para o "
        "mesmo nome normalizado que setuptools-scm", ok
    )


def _check_sam3_runtime_lock_pins_setuptools_below_pkg_resources_removal() -> bool:
    root = Path(__file__).resolve().parents[3]
    pyproject_path = root / "sam3_runtime" / "pyproject.toml"
    lock_path = root / "sam3_runtime" / "uv.lock"

    pyproject_ok = _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
        pyproject_path
    )
    lock_ok = _lock_resolves_setuptools_below_pkg_resources_removal(lock_path)

    return _check(
        "sam3_runtime: uv.lock resolve setuptools abaixo de 82 (pkg_resources "
        "ainda existe) e sam3_runtime/pyproject.toml declara esse teto em "
        "[tool.uv].constraint-dependencies",
        pyproject_ok and lock_ok,
    )


def run_sam3_selftest() -> None:
    checks = [
        _check_compute_descriptor_rectangle(),
        _check_compute_descriptor_square_is_isotropic(),
        _check_compute_descriptor_empty_mask(),
        _check_missing_frame_is_isolated_zero_no_forward_fill(),
        _check_end_to_end_continuity_tracks_moving_instance(),
        _check_storage_round_trip(),
        _check_validate_existing_file(),
        _check_provenance_divergence_heterogeneous_checkpoint(),
        _check_dataset_guard_rejects_le2i_cv(),
        _check_build_worker_invocation_project_equals_cwd(),
        _check_build_worker_invocation_checkpoint_absolute_independent_of_cwd(),
        _check_resolve_precedence_and_absolute_paths(),
        _check_ensure_sam3_runtime_available_requires_uv_lock(),
        _check_find_invalid_required_provenance(),
        _check_report_detects_structurally_invalid_h5(),
        _check_report_rejects_malformed_source_revision(),
        _check_report_rejects_malformed_checkpoint_digest(),
        _check_missing_video_id_raises_extract_error(),
        _check_setuptools_ceiling_matches_distribution_name_not_substring(),
        _check_setuptools_lock_version_boundary_and_corruption(),
        _check_normalize_distribution_name_collapses_separator_runs(),
        _check_sam3_runtime_lock_pins_setuptools_below_pkg_resources_removal(),
    ]
    if not all(checks):
        print("\nsam3 extract selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nsam3 extract selftest OK: todas as checagens passaram")
