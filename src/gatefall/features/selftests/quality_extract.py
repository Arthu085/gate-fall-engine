"""Selftest sintético do armazenamento dos sidecars de qualidade
(`quality_storage.py` / `quality_extract.py`). Não toca em dados reais nem
carrega backbone: grava HDF5 sintéticos em diretório temporário."""

import sys
import tempfile
from pathlib import Path

import numpy as np

from gatefall.features.quality_extract import (
    QualityExtractError,
    current_provenance,
    existing_sidecar_action,
    pose_source_sha256,
)
from gatefall.features.quality_storage import (
    QUALITY_CHANNELS,
    QualityStorageError,
    quality_path,
    quality_set_sha256,
    read_quality,
    validate_existing_file,
    verify_written_file,
    write_quality_atomic,
)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _provenance(pose_source_digest: str = "pose-sha256-sintetico") -> dict[str, object]:
    return current_provenance(pose_source_digest=pose_source_digest)


def _write_pose_source(pose_root: Path, video_id: str, payload: bytes) -> Path:
    env, _, video_name = video_id.partition("/")
    path = pose_root / env / f"{video_name}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _synthetic_quality(k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((k, QUALITY_CHANNELS)).astype(np.float32)


def check_path_is_grouped_by_video() -> bool:
    root = Path("data/features/le2i/quality")
    path = quality_path("coffee_room/video (1)", quality_root=root)
    ok = path == root / "coffee_room" / "video (1).h5"
    return _check(
        "quality_path agrupa por vídeo em <quality_root>/<env>/<video>.h5, "
        "nunca um arquivo por quadro",
        ok,
    )


def check_write_read_round_trip() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "coffee_room" / "video.h5"
        quality = _synthetic_quality(37)
        attrs = {"video_id": "coffee_room/video", "K": 37, **_provenance()}
        write_quality_atomic(path, quality, attrs)
        verify_written_file(path, quality=quality, attrs=attrs)
        loaded = read_quality(path)
        ok = (
            loaded.shape == (37, QUALITY_CHANNELS)
            and loaded.dtype == np.float32
            and bool(np.array_equal(loaded, quality))
            and not any(path.parent.glob("*.tmp"))
        )
    return _check(
        "write_quality_atomic grava [K,2] float32, verify_written_file aceita o "
        "arquivo recém-gravado e read_quality devolve os mesmos valores",
        ok,
    )


def check_validate_existing_file_detects_wrong_k_and_attrs() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "coffee_room" / "video.h5"
        quality = _synthetic_quality(20)
        provenance = _provenance()
        write_quality_atomic(path, quality, {"K": 20, **provenance})

        clean = validate_existing_file(path, expected_k=20, expected_attrs=provenance)
        wrong_k = validate_existing_file(
            path, expected_k=21, expected_attrs=provenance
        )
        divergent = dict(provenance)
        divergent["target_fps"] = 25.0
        wrong_attr = validate_existing_file(
            path, expected_k=20, expected_attrs=divergent
        )
        missing_attr = validate_existing_file(
            path, expected_k=20, expected_attrs={**provenance, "ausente": 1}
        )
        ok = (
            clean == []
            and wrong_k != []
            and wrong_attr != []
            and missing_attr != []
        )
    return _check(
        "validate_existing_file aceita o sidecar íntegro e recusa K divergente, "
        "atributo de proveniência divergente e atributo ausente",
        ok,
    )


def check_reextracted_pose_source_invalidates_sidecar() -> bool:
    video_id = "coffee_room/video"
    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp) / "pose"
        quality_root = Path(tmp) / "quality"
        _write_pose_source(pose_root, video_id, b"pose-original")

        provenance = _provenance(pose_source_sha256(video_id, pose_root=pose_root))
        path = quality_path(video_id, quality_root=quality_root)
        write_quality_atomic(path, _synthetic_quality(15), {"K": 15, **provenance})

        before = validate_existing_file(path, expected_k=15, expected_attrs=provenance)
        skips_before = existing_sidecar_action(before, force=False) == "skip"

        _write_pose_source(pose_root, video_id, b"pose-reextraida")
        after_provenance = _provenance(
            pose_source_sha256(video_id, pose_root=pose_root)
        )
        after = validate_existing_file(
            path, expected_k=15, expected_attrs=after_provenance
        )
        refuses_after = existing_sidecar_action(after, force=False) == "fail"

        missing_pose_raised = False
        try:
            pose_source_sha256("coffee_room/ausente", pose_root=pose_root)
        except QualityExtractError:
            missing_pose_raised = True

        ok = (
            before == []
            and skips_before
            and any("pose_source_sha256" in reason for reason in after)
            and refuses_after
            and missing_pose_raised
        )
    return _check(
        "sidecar de pose reextraído muda pose_source_sha256, faz "
        "validate_existing_file apontar o motivo e faz a extração falhar em vez "
        "de pular o vídeo sem --force",
        ok,
    )


def check_verify_written_file_rejects_tampered_values() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "coffee_room" / "video.h5"
        quality = _synthetic_quality(12)
        attrs = {"K": 12, **_provenance()}
        write_quality_atomic(path, quality, attrs)

        tampered = quality.copy()
        tampered[0, 0] = 1.0 - tampered[0, 0]
        raised = False
        try:
            verify_written_file(path, quality=tampered, attrs=attrs)
        except QualityStorageError:
            raised = True
    return _check(
        "verify_written_file recusa um arquivo cujos valores divergem do que o "
        "chamador diz ter gravado",
        raised,
    )


def check_quality_set_sha256_is_deterministic_and_sensitive() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        video_ids = ["coffee_room/a", "home/b"]
        for index, video_id in enumerate(video_ids):
            write_quality_atomic(
                quality_path(video_id, quality_root=root),
                _synthetic_quality(10, seed=index),
                {"K": 10, **_provenance()},
            )
        first = quality_set_sha256(video_ids, quality_root=root)
        second = quality_set_sha256(list(reversed(video_ids)), quality_root=root)

        write_quality_atomic(
            quality_path(video_ids[0], quality_root=root),
            _synthetic_quality(10, seed=99),
            {"K": 10, **_provenance()},
        )
        after_change = quality_set_sha256(video_ids, quality_root=root)

        missing_raised = False
        try:
            quality_set_sha256(["coffee_room/ausente"], quality_root=root)
        except FileNotFoundError:
            missing_raised = True

        ok = first == second and after_change != first and missing_raised
    return _check(
        "quality_set_sha256 independe da ordem dos video_ids, muda quando um "
        "sidecar é reextraído e falha quando um sidecar está ausente",
        ok,
    )


def run_quality_extract_selftest() -> bool:
    checks = [
        check_path_is_grouped_by_video(),
        check_write_read_round_trip(),
        check_validate_existing_file_detects_wrong_k_and_attrs(),
        check_reextracted_pose_source_invalidates_sidecar(),
        check_verify_written_file_rejects_tampered_values(),
        check_quality_set_sha256_is_deterministic_and_sensitive(),
    ]
    ok = all(checks)
    if not ok:
        print("\nquality extract selftest FALHOU", file=sys.stderr)
    else:
        print("\nquality extract selftest OK: todas as checagens passaram")
    return ok
