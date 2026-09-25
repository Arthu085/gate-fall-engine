"""Selftest sintético da leitura validada de V_t (`sam3/features.py`).

Grava `.h5` sintéticos com `storage.write_sam3_atomic` — mesmo schema da
extração real — e nunca sobe o runtime do SAM 3 nem toca no dataset real.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from gatefall.config import TARGET_FPS
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.sam3.features import collect_sam3_provenance, load_v_t, sam3_set_sha256
from gatefall.sam3.storage import sam3_path, write_sam3_atomic

_VALID_PROVENANCE: dict[str, object] = {
    "model_name": "facebook/sam3",
    "text_prompt": "person",
    "sam3_checkpoint_sha256": "a" * 64,
    "sam3_runtime_lock_sha256": "b" * 64,
    "sam3_source_revision": "c" * 40,
    "sam3_inference_autocast_dtype": "float16",
    "target_fps": TARGET_FPS,
}


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _write(
    root: Path,
    video_id: str,
    k: int,
    *,
    split: str = "train",
    v_t_dim: int = V_T_DIM,
    stored_video_id: str | None = None,
    **provenance_overrides: object,
) -> Path:
    path = sam3_path(video_id, sam3_root=root)
    v_t = (np.arange(k * v_t_dim, dtype=np.float32).reshape(k, v_t_dim) / 100.0)
    attrs: dict[str, object] = {
        "video_id": stored_video_id or video_id,
        "split": split,
        "K": k,
        **_VALID_PROVENANCE,
        **provenance_overrides,
    }
    write_sam3_atomic(
        path,
        v_t,
        np.ones(k, dtype=np.float32),
        np.ones(k, dtype=np.int16),
        attrs,
    )
    return path


def _raises(callback, error: type[Exception] = ValueError, fragment: str = "") -> bool:
    try:
        callback()
    except error as exc:
        return fragment in str(exc)
    return False


def check_load_v_t_returns_stored_rows() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "env_a/video_1", 12)
        v_t = load_v_t("env_a/video_1", sam3_root=root)
    expected = np.arange(12 * V_T_DIM, dtype=np.float32).reshape(12, V_T_DIM) / 100.0
    return _check(
        f"load_v_t devolve [K,{V_T_DIM}] float32 idêntico ao gravado, na ordem de linha do .h5",
        v_t.dtype == np.float32 and bool(np.array_equal(v_t, expected)),
    )


def check_load_v_t_rejects_identity_and_structure_errors() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "env_a/video_1", 8, stored_video_id="env_a/video_2")
        wrong_identity = _raises(
            lambda: load_v_t("env_a/video_1", sam3_root=root), fragment="video_2"
        )
        _write(root, "env_a/video_3", 8, v_t_dim=V_T_DIM - 1)
        wrong_dim = _raises(lambda: load_v_t("env_a/video_3", sam3_root=root))
        path = _write(root, "env_a/video_4", 8)
        with h5py.File(path, "a") as h5_file:
            h5_file.attrs["K"] = 9
        wrong_k = _raises(lambda: load_v_t("env_a/video_4", sam3_root=root))
        missing = _raises(
            lambda: load_v_t("env_a/absent", sam3_root=root), error=FileNotFoundError
        )
    return _check(
        "load_v_t recusa .h5 de outro vídeo, largura != 10, atributo K "
        "divergente das linhas e arquivo ausente",
        wrong_identity and wrong_dim and wrong_k and missing,
    )


def check_collect_provenance_accepts_homogeneous_set() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "env_a/video_1", 6, split="train")
        _write(root, "env_b/video_1", 6, split="test")
        provenance = collect_sam3_provenance(
            {"env_a/video_1": "train", "env_b/video_1": "test"}, sam3_root=root
        )
    return _check(
        "collect_sam3_provenance devolve a proveniência comum normalizada de "
        "um conjunto homogêneo",
        provenance["text_prompt"] == "person"
        and provenance["sam3_inference_autocast_dtype"] == "float16"
        and provenance["target_fps"] == str(TARGET_FPS),
    )


def check_collect_provenance_rejects_invalid_sets() -> bool:
    cases: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "split"
        _write(root, "env_a/video_1", 6, split="val")
        cases["split"] = _raises(
            lambda: collect_sam3_provenance({"env_a/video_1": "train"}, sam3_root=root),
            fragment="split",
        )

        root = Path(tmp) / "mixed_dtype"
        _write(root, "env_a/video_1", 6)
        _write(root, "env_a/video_2", 6, sam3_inference_autocast_dtype="bfloat16")
        cases["mixed_dtype"] = _raises(
            lambda: collect_sam3_provenance(
                {"env_a/video_1": "train", "env_a/video_2": "train"}, sam3_root=root
            ),
            fragment="sam3_inference_autocast_dtype",
        )

        root = Path(tmp) / "prompt"
        _write(root, "env_a/video_1", 6, text_prompt="person falling")
        cases["prompt"] = _raises(
            lambda: collect_sam3_provenance({"env_a/video_1": "train"}, sam3_root=root),
            fragment="text_prompt",
        )

        root = Path(tmp) / "malformed"
        _write(root, "env_a/video_1", 6, sam3_source_revision="not-a-sha")
        cases["malformed"] = _raises(
            lambda: collect_sam3_provenance({"env_a/video_1": "train"}, sam3_root=root),
            fragment="sam3_source_revision",
        )

        cases["missing"] = _raises(
            lambda: collect_sam3_provenance(
                {"env_a/absent": "train"}, sam3_root=Path(tmp) / "empty"
            ),
            error=FileNotFoundError,
        )
    return _check(
        "collect_sam3_provenance recusa split divergente da tabela de frames, "
        "mistura FP16/BF16, prompt fora de 'person', proveniência malformada "
        f"e .h5 ausente ({cases})",
        all(cases.values()),
    )


def check_set_sha256_tracks_reextraction() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "env_a/video_1", 6)
        _write(root, "env_b/video_1", 6)
        video_ids = ["env_b/video_1", "env_a/video_1"]
        first = sam3_set_sha256(video_ids, sam3_root=root)
        reordered = sam3_set_sha256(list(reversed(video_ids)), sam3_root=root)
        _write(root, "env_a/video_1", 7)
        after_reextraction = sam3_set_sha256(video_ids, sam3_root=root)
    return _check(
        "sam3_set_sha256 independe da ordem dos video_ids e muda quando um "
        ".h5 é reextraído",
        first == reordered and first != after_reextraction,
    )


def run_sam3_features_selftest() -> bool:
    checks = [
        check_load_v_t_returns_stored_rows(),
        check_load_v_t_rejects_identity_and_structure_errors(),
        check_collect_provenance_accepts_homogeneous_set(),
        check_collect_provenance_rejects_invalid_sets(),
        check_set_sha256_tracks_reextraction(),
    ]
    ok = all(checks)
    if not ok:
        print("\nsam3 features selftest FALHOU", file=sys.stderr)
    else:
        print("\nsam3 features selftest OK: todas as checagens passaram")
    return ok
