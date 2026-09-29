"""Checagens sintéticas de SAM 3: runtime."""

import os
import tempfile
from pathlib import Path

from gatefall.sam3 import runtime, storage
from gatefall.sam3.runtime import (
    CHECKPOINT_PATH_ENV_VAR,
    RUNTIME_DIR_ENV_VAR,
    build_worker_invocation,
    ensure_sam3_runtime_available,
    resolve_checkpoint_path,
    resolve_runtime_project_dir,
)
from gatefall.sam3.selftests.fixtures import _check


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


def _check_select_inference_autocast_dtype_policy() -> bool:
    # Tabela pura: a política é decidida sem torch e sem GPU, então o
    # suporte a bf16 na CUDA entra como parâmetro, não como consulta.
    expected_by_case: dict[tuple[str, bool], str] = {
        ("cuda", True): "bfloat16",
        ("cuda", False): "float16",
        ("cpu", False): "bfloat16",
        ("cpu", True): "bfloat16",
    }
    selected_by_case = {
        (device, cuda_bf16_supported): runtime.select_inference_autocast_dtype_name(
            device, cuda_bf16_supported=cuda_bf16_supported
        )
        for device, cuda_bf16_supported in expected_by_case
    }
    table_ok = selected_by_case == expected_by_case

    pattern, _ = storage.REQUIRED_PROVENANCE_ATTR_FORMATS["sam3_inference_autocast_dtype"]
    vocabulary_ok = all(
        value in runtime.SAM3_INFERENCE_AUTOCAST_DTYPE_NAMES
        and pattern.match(value) is not None
        for value in selected_by_case.values()
    )

    ok = table_ok and vocabulary_ok
    return _check(
        "select_inference_autocast_dtype_name: cuda com bf16 suportado dá "
        "bfloat16, cuda sem bf16 dá float16 e cpu dá bfloat16 em qualquer "
        "caso, e todo valor devolvido pertence a "
        "SAM3_INFERENCE_AUTOCAST_DTYPE_NAMES e casa com o formato de "
        "proveniência exigido", ok
    )
