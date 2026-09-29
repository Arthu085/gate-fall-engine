"""Checagens sintéticas da proveniência e dos relatórios SAM 3."""

import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from gatefall.sam3 import storage
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.sam3.report import find_provenance_divergences, run_sam3_report
from gatefall.sam3.storage import sam3_path, write_sam3_atomic
from gatefall.sam3.selftests.fixtures import _build_fixture, _check


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


def _check_provenance_divergence_mixed_inference_autocast_dtype() -> bool:
    homogeneous: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(sam3_inference_autocast_dtype="bfloat16"),
        "env1/v2": _full_provenance_attrs(sam3_inference_autocast_dtype="bfloat16"),
    }
    homogeneous_ok = find_provenance_divergences(homogeneous) == []

    heterogeneous: dict[str, dict[str, object]] = {
        "env1/v1": _full_provenance_attrs(sam3_inference_autocast_dtype="bfloat16"),
        "env1/v2": _full_provenance_attrs(sam3_inference_autocast_dtype="float16"),
    }
    divergences = find_provenance_divergences(heterogeneous)
    heterogeneous_ok = (
        len(divergences) == 1
        and "env1/v2" in divergences[0]
        and "'sam3_inference_autocast_dtype'" in divergences[0]
    )

    ok = homogeneous_ok and heterogeneous_ok
    return _check(
        "find_provenance_divergences: dois vídeos extraídos com dtype de "
        "autocast diferente (bfloat16 vs float16) dão exatamente uma "
        "divergência nomeando o vídeo e 'sam3_inference_autocast_dtype', e o "
        "caso homogêneo dá []", ok
    )


_WELL_FORMED_REQUIRED_PROVENANCE_VALUES: dict[str, str] = {
    "sam3_checkpoint_sha256": "a" * 64,
    "sam3_runtime_lock_sha256": "b" * 64,
    "sam3_source_revision": "c" * 40,
    "sam3_inference_autocast_dtype": "bfloat16",
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


def _check_inference_autocast_dtype_provenance_format() -> bool:
    def invalid_for(**overrides: object) -> list[str]:
        return storage.find_invalid_required_provenance(
            _full_required_provenance_attrs(**overrides),
            storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES,
        )

    accepted_ok = all(
        invalid_for(sam3_inference_autocast_dtype=value) == []
        for value in ("bfloat16", "float16")
    )

    empty_ok = invalid_for(sam3_inference_autocast_dtype="") == [
        "sam3_inference_autocast_dtype: vazio"
    ]

    attrs_missing_key = _full_required_provenance_attrs()
    del attrs_missing_key["sam3_inference_autocast_dtype"]
    missing_ok = storage.find_invalid_required_provenance(
        attrs_missing_key, storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES
    ) == ["sam3_inference_autocast_dtype: ausente"]

    malformed_ok = True
    for value in ("float32", "torch.bfloat16", "bf16", "BFloat16"):
        reasons = invalid_for(sam3_inference_autocast_dtype=value)
        malformed_ok = (
            malformed_ok
            and len(reasons) == 1
            and reasons[0].startswith(
                "sam3_inference_autocast_dtype: formato inválido"
            )
        )

    ok = accepted_ok and empty_ok and missing_ok and malformed_ok
    return _check(
        "find_invalid_required_provenance: sam3_inference_autocast_dtype "
        "aceita exatamente 'bfloat16' e 'float16', e ausente, vazio ou fora "
        "do vocabulário (float32, torch.bfloat16, bf16, BFloat16) é nomeado "
        "com o motivo específico", ok
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


def _check_report_rejects_malformed_inference_autocast_dtype() -> bool:
    video_id = "coffee_room/video_malformed_autocast_dtype"
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
                sam3_inference_autocast_dtype="torch.bfloat16"
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
        and "sam3_inference_autocast_dtype: formato inválido" in report_output
    )
    return _check(
        "run_sam3_report: sam3_inference_autocast_dtype com um nome de dtype "
        "do torch ('torch.bfloat16') em vez do vocabulário gravado reprova a "
        "checagem de proveniência obrigatória", ok
    )
