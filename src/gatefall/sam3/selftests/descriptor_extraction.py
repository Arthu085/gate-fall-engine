"""Checagens sintéticas dos descritores e da extração SAM 3."""

import tempfile
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from gatefall.data.frames import read_frames
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.sam3 import storage
from gatefall.sam3.dataset_guard import ensure_sam3_dataset_supported
from gatefall.sam3.descriptors import ZERO_DESCRIPTOR, compute_descriptor
from gatefall.sam3.extract import (
    Sam3ExtractError,
    run_frames_through_segmenter,
    run_sam3_extract,
    run_sam3_extract_all,
)
from gatefall.sam3.frame_alignment import run_sam3_verify_frame_alignment
from gatefall.sam3.report import run_sam3_report
from gatefall.sam3.runtime import Sam3Instance
from gatefall.sam3.storage import sam3_path
from gatefall.sam3.selftests.fixtures import (
    _ManifestSam3Segmenter,
    _ScriptedSam3Segmenter,
    _SyntheticDatasetAdapter,
    _build_fixture,
    _check,
    _install_fake_decode_frames,
    _rect_mask,
)


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
            # `.get` em vez de indexação: o atributo ausente precisa reprovar
            # a checagem, não abortar o selftest inteiro.
            autocast_dtype_attr = h5_file.attrs.get("sam3_inference_autocast_dtype")

        frames_parquet = read_frames(adapter.frames_path)
        k_matches = result.k == len(frames_parquet) == k
    scores_track_target = bool(np.allclose(stored_sam_score, target_scores, atol=1e-6))
    expected_centroid_x = np.array(
        [((b[0] + b[2]) / 2.0) / width for b in target_boxes], dtype=np.float32
    )
    centroids_track_target = bool(np.allclose(stored_v_t[:, 2], expected_centroid_x, atol=1e-5))
    provenance_sentinels_ok = (
        checkpoint_attr == ""
        and lock_attr == ""
        and autocast_dtype_attr is not None
        and str(autocast_dtype_attr) == ""
    )

    ok = (
        k_matches
        and scores_track_target
        and centroids_track_target
        and provenance_sentinels_ok
    )
    return _check(
        "fim a fim: instância contínua (score mais baixo) é seguida em vez "
        "do distrator (score mais alto e sem overlap), K bate com "
        "frames.parquet e sam3_checkpoint_sha256/sam3_runtime_lock_sha256/"
        "sam3_inference_autocast_dtype degradam para '' quando os arquivos "
        "reais não existem e o segmentador fake não traz manifesto", ok
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
        ]

        nothing_written_ok = not cv_sam3_root.exists() and not any(
            Path(temporary_dir).iterdir()
        )

    accepts_cs = True
    try:
        ensure_sam3_dataset_supported(Le2iDatasetAdapter())
        ensure_sam3_dataset_supported(Le2iDatasetAdapter(protocol="cv"))
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
        "guarda de protocolo: os dois comandos de extração SAM 3 rejeitam o "
        "adapter le2i-cv com ValueError sem criar nada sob "
        "adapter.sam3_root, e a guarda continua aceitando qualquer adapter "
        "com identifier 'le2i' (checagem por identifier, não por isinstance)",
        all(rejected) and nothing_written_ok and accepts_cs,
    )


def _check_extract_persists_inference_autocast_dtype_from_runtime_manifest() -> bool:
    width, height = 100, 50
    k = 2
    frames_by_src = {i: np.zeros((height, width, 3), dtype=np.uint8) for i in range(k)}
    manifest: dict[str, object] = {
        "sam3_inference_autocast_dtype": "float16",
        "device": "cuda",
    }

    def extract_with(
        video_id: str, *, inference_autocast_dtype: str | None = None
    ) -> str:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            adapter = _build_fixture(
                root, video_id=video_id, k=k, width=width, height=height
            )
            segmenter = _ManifestSam3Segmenter([[] for _ in range(k)], manifest)
            restore = _install_fake_decode_frames(frames_by_src)
            try:
                run_sam3_extract(
                    video_id,
                    adapter=adapter,
                    segmenter=segmenter,
                    runtime_project_dir_value=str(root / "sam3_runtime_missing"),
                    checkpoint_path_value=str(root / "checkpoint_missing.pt"),
                    inference_autocast_dtype=inference_autocast_dtype,
                )
            finally:
                restore()
            output_path = sam3_path(video_id, sam3_root=adapter.sam3_root)
            with h5py.File(output_path, "r") as h5_file:
                return str(h5_file.attrs.get("sam3_inference_autocast_dtype"))

    from_manifest_ok = extract_with("coffee_room/video_autocast_manifest") == "float16"
    with_matching_kwarg_ok = (
        extract_with(
            "coffee_room/video_autocast_kwarg", inference_autocast_dtype="float16"
        )
        == "float16"
    )

    ok = from_manifest_ok and with_matching_kwarg_ok
    return _check(
        "run_sam3_extract: sam3_inference_autocast_dtype vem do manifesto de "
        "runtime do segmentador injetado, e um "
        "inference_autocast_dtype explícito idêntico grava o mesmo valor", ok
    )


def _check_extract_force_rejects_second_inference_autocast_dtype() -> bool:
    width, height = 100, 50
    k = 2
    video_id = "coffee_room/video_autocast_force"
    frames_by_src = {i: np.zeros((height, width, 3), dtype=np.uint8) for i in range(k)}

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        adapter = _build_fixture(root, video_id=video_id, k=k, width=width, height=height)
        output_path = sam3_path(video_id, sam3_root=adapter.sam3_root)

        def extract_with_manifest_dtype(dtype_name: str, *, force: bool) -> None:
            segmenter = _ManifestSam3Segmenter(
                [[] for _ in range(k)],
                {"sam3_inference_autocast_dtype": dtype_name},
            )
            restore = _install_fake_decode_frames(frames_by_src)
            try:
                run_sam3_extract(
                    video_id,
                    adapter=adapter,
                    segmenter=segmenter,
                    runtime_project_dir_value=str(root / "sam3_runtime_missing"),
                    checkpoint_path_value=str(root / "checkpoint_missing.pt"),
                    force=force,
                )
            finally:
                restore()

        extract_with_manifest_dtype("bfloat16", force=False)

        try:
            extract_with_manifest_dtype("float16", force=True)
        except Sam3ExtractError as exc:
            message = str(exc)
            mismatch_rejected = (
                video_id in message
                and "'bfloat16'" in message
                and "'float16'" in message
                and "conjunto inteiro" in message
            )
        else:
            mismatch_rejected = False

        with h5py.File(output_path, "r") as h5_file:
            preserved_ok = str(h5_file.attrs["sam3_inference_autocast_dtype"]) == "bfloat16"

        try:
            extract_with_manifest_dtype("bfloat16", force=True)
        except Sam3ExtractError:
            same_dtype_ok = False
        else:
            same_dtype_ok = True

    ok = mismatch_rejected and preserved_ok and same_dtype_ok
    return _check(
        "run_sam3_extract: `--force` de um vídeo único contra um .h5 já "
        "gravado em 'bfloat16' levanta Sam3ExtractError quando o worker "
        "reporta 'float16' (nomeando o vídeo, os dois dtypes e a reextração "
        "do conjunto inteiro) sem sobrescrever o artefato, e o mesmo dtype "
        "reextrai normalmente", ok
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
