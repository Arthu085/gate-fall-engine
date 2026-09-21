"""Selftest sintético da extração SAM 3 (descritores, seleção, armazenamento).

Não toca em GPU, dataset real nem no runtime isolado `sam3_runtime/` — todas
as entradas são sintéticas e o segmentador é sempre um fake injetado via
`segmenter=`, nunca o `Sam3RuntimeSegmenter` real.
"""

import sys
import tempfile
from dataclasses import dataclass
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
from gatefall.sam3.runtime import Sam3Instance
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
        _check_missing_video_id_raises_extract_error(),
    ]
    if not all(checks):
        print("\nsam3 extract selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nsam3 extract selftest OK: todas as checagens passaram")
