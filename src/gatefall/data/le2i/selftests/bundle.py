"""Casos sintéticos de preparação e verificação do pacote Le2i CS."""

import tempfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from gatefall.data.le2i.bundle import (
    FRAMES_PATH,
    MANIFEST_PATH,
    RAW_PATH,
    RECEIPT_PATH,
    BundleError,
    prepare_bundle,
    verify_bundle,
)
from gatefall.hashing import sha256_file


def _source(root: Path, relative_path: str = "Coffee_room_01/Videos/video (1).avi") -> Path:
    source = root / "source"
    manifest_path = source / MANIFEST_PATH
    frames_path = source / FRAMES_PATH
    manifest_path.parent.mkdir(parents=True)
    video_path = source / RAW_PATH / relative_path
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"synthetic video bytes\x00\xff")
    pd.DataFrame(
        [
            {
                "video_id": "coffee_room_01/video_1",
                "relative_path": relative_path,
                "sha256": sha256_file(video_path),
            }
        ]
    ).to_parquet(manifest_path)
    pd.DataFrame(
        [{"video_id": "coffee_room_01/video_1", "src_index": 0}]
    ).to_parquet(frames_path)
    return source


def _expect_failure(action: Callable[[], object], expected: str) -> None:
    try:
        action()
    except BundleError as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError(f"falha esperada: {expected}")


def run_bundle_selftest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source(root)
        bundle = root / "bundle"
        assert prepare_bundle(source, bundle) == 1
        for relative in (MANIFEST_PATH, FRAMES_PATH):
            assert (source / relative).read_bytes() == (bundle / relative).read_bytes()
        relative_video = Path("Coffee_room_01/Videos/video (1).avi")
        assert (source / RAW_PATH / relative_video).read_bytes() == (
            bundle / RAW_PATH / relative_video
        ).read_bytes()
        assert verify_bundle(bundle) == 1
        _expect_failure(lambda: prepare_bundle(source, bundle), "destino já existe")
        print("[PASS] metadados byte-idênticos, caminho bruto preservado e pacote válido")

        (bundle / FRAMES_PATH).write_bytes(b"alterado")
        _expect_failure(lambda: verify_bundle(bundle), "arquivo canônico alterado")
        print("[PASS] verificação rejeita parquet alterado")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bundle = root / "bundle"
        prepare_bundle(_source(root), bundle)
        video_path = bundle / RAW_PATH / "Coffee_room_01/Videos/video (1).avi"
        video_path.write_bytes(b"alterado")
        _expect_failure(lambda: verify_bundle(bundle), "sha256 divergente para vídeo")
        video_path.unlink()
        _expect_failure(lambda: verify_bundle(bundle), "vídeo referenciado ausente")
        print("[PASS] verificação rejeita vídeo alterado ou ausente")

        (bundle / RECEIPT_PATH).unlink()
        _expect_failure(lambda: verify_bundle(bundle), "recibo de integridade ausente")
        print("[PASS] verificação exige recibo de integridade")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source(root)
        (source / RAW_PATH / "Coffee_room_01/Videos/video (1).avi").unlink()
        bundle = root / "bundle"
        _expect_failure(lambda: prepare_bundle(source, bundle), "vídeo referenciado ausente")
        assert not bundle.exists()
        print("[PASS] vídeo ausente não publica pacote")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source(root)
        manifest_path = source / MANIFEST_PATH
        manifest = pd.read_parquet(manifest_path)
        manifest.loc[0, "relative_path"] = "../outside.avi"
        manifest.to_parquet(manifest_path)
        bundle = root / "bundle"
        _expect_failure(lambda: prepare_bundle(source, bundle), "relative_path inválido")
        assert not bundle.exists()
        manifest.loc[0, "relative_path"] = "/tmp/outside.avi"
        manifest.to_parquet(manifest_path)
        _expect_failure(lambda: prepare_bundle(source, bundle), "relative_path inválido")
        assert not bundle.exists()
        manifest.loc[0, "relative_path"] = "C:\\outside.avi"
        manifest.to_parquet(manifest_path)
        _expect_failure(lambda: prepare_bundle(source, bundle), "relative_path inválido")
        assert not bundle.exists()
        print("[PASS] caminhos relativos inseguros rejeitados")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source(root)
        manifest_path = source / MANIFEST_PATH
        manifest = pd.read_parquet(manifest_path)
        manifest.loc[0, "sha256"] = "0" * 64
        manifest.to_parquet(manifest_path)
        bundle = root / "bundle"
        _expect_failure(lambda: prepare_bundle(source, bundle), "sha256 divergente")
        assert not bundle.exists()
        assert not list(root.glob(".bundle.*"))
        print("[PASS] preparação incompleta remove staging e não publica pacote")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source(root)
        (source / FRAMES_PATH).unlink()
        _expect_failure(lambda: prepare_bundle(source, root / "bundle"), "arquivo canônico ausente")
        assert not (root / "bundle").exists()
        print("[PASS] metadados canônicos ausentes rejeitados")


if __name__ == "__main__":
    run_bundle_selftest()
