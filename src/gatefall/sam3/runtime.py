"""Fronteira de isolamento entre o GateFall e o runtime real do SAM 3.

O SAM 3 (facebookresearch/sam3, runtime oficial) roda em um uv sub-projeto
isolado em `sam3_runtime/`, com seu próprio `pyproject.toml` e `uv.lock`
commitado. Este módulo nunca importa `torch`/`sam3` real nem o processo
`sam3_runtime/run_sam3.py` — só troca bytes com ele por um subprocesso de
vida longa, um quadro por vez, via um protocolo de fio (wire protocol) simples
de mensagens com prefixo de tamanho.

Protocolo (todo inteiro é uint32 big-endian):
- Ao iniciar, o worker imprime uma única linha JSON em stdout com o
  manifesto de runtime (versões, hash do checkpoint,
  `sam3_inference_autocast_dtype`) antes de processar qualquer quadro.
- Por quadro: o cliente escreve um cabeçalho JSON com prefixo de tamanho
  (`{"height", "width", "text_prompt"}`) seguido do RGB cru do quadro
  (`height*width*3` bytes) também com prefixo de tamanho; o worker responde
  com um blob `.npz` com prefixo de tamanho contendo `masks` ([N,H,W] bool) e
  `scores` ([N] float32).
"""

import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import IO, Protocol

import numpy as np

TEXT_PROMPT = "person"

SAM3_INFERENCE_AUTOCAST_DTYPE_NAMES: tuple[str, ...] = ("bfloat16", "float16")


def select_inference_autocast_dtype_name(
    device: str, *, cuda_bf16_supported: bool
) -> str:
    """Política de precisão de inferência do SAM 3, duplicada no worker.

    `sam3.perflib.fused.addmm_act()` (upstream 2345a4a) converte a primeira
    projeção do `Mlp` para bfloat16 incondicionalmente e devolve BF16 para um
    `fc2` FP32; só um contexto de autocast alinha as duas metades. FP16 é o
    recuo para hardware pré-Ampere, onde o GEMM bfloat16 não é nativo.
    """
    if device == "cuda":
        return "bfloat16" if cuda_bf16_supported else "float16"
    return "bfloat16"

SAM3_RUNTIME_PROJECT_DIR = Path("sam3_runtime")
DEFAULT_CHECKPOINT_PATH = Path("data/scratch/weights/sam3/sam3_checkpoint.pt")

RUNTIME_DIR_ENV_VAR = "GATEFALL_SAM3_RUNTIME_DIR"
CHECKPOINT_PATH_ENV_VAR = "GATEFALL_SAM3_CHECKPOINT_PATH"

_LENGTH_PREFIX_FORMAT = ">I"
_LENGTH_PREFIX_SIZE = struct.calcsize(_LENGTH_PREFIX_FORMAT)


@dataclass(frozen=True, eq=False)
class Sam3Instance:
    mask: np.ndarray
    score: float


class Sam3Segmenter(Protocol):
    def segment_frame(
        self, frame_rgb: np.ndarray, text_prompt: str
    ) -> list[Sam3Instance]: ...


def resolve_runtime_project_dir(cli_value: str | None) -> Path:
    if cli_value is not None:
        return Path(cli_value).resolve()
    env_value = os.environ.get(RUNTIME_DIR_ENV_VAR)
    if env_value is not None:
        return Path(env_value).resolve()
    return SAM3_RUNTIME_PROJECT_DIR.resolve()


def resolve_checkpoint_path(cli_value: str | None) -> Path:
    if cli_value is not None:
        return Path(cli_value).resolve()
    env_value = os.environ.get(CHECKPOINT_PATH_ENV_VAR)
    if env_value is not None:
        return Path(env_value).resolve()
    return DEFAULT_CHECKPOINT_PATH.resolve()


def build_worker_invocation(
    runtime_project_dir: Path, checkpoint_path: Path
) -> tuple[list[str], str]:
    """Monta o argv/cwd do `uv run` do worker do SAM 3.

    `--project` e `cwd` devem ser exatamente o mesmo caminho absoluto: `uv`
    resolve `--project` contra o próprio `cwd` do subprocesso, então um valor
    relativo aqui aponta para o lugar errado assim que `cwd` já é o próprio
    diretório do sub-projeto.
    """
    runtime_project_dir = runtime_project_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()
    argv = [
        "uv",
        "run",
        "--project",
        str(runtime_project_dir),
        "python",
        "run_sam3.py",
        "--checkpoint",
        str(checkpoint_path),
    ]
    return argv, str(runtime_project_dir)


def ensure_sam3_runtime_available(runtime_project_dir: Path, checkpoint_path: Path) -> None:
    if not runtime_project_dir.exists():
        raise FileNotFoundError(
            f"sub-projeto do runtime SAM 3 não encontrado: {runtime_project_dir} — "
            f"passe --runtime-dir, defina {RUNTIME_DIR_ENV_VAR} ou crie-o no "
            "caminho padrão (`sam3_runtime/`) antes de rodar a extração"
        )
    run_script = runtime_project_dir / "run_sam3.py"
    if not run_script.exists():
        raise FileNotFoundError(
            f"worker do runtime SAM 3 não encontrado: {run_script} — "
            f"{runtime_project_dir} existe mas não contém `run_sam3.py`"
        )
    lock_file = runtime_project_dir / "uv.lock"
    if not lock_file.exists():
        raise FileNotFoundError(
            f"uv.lock do runtime SAM 3 não encontrado: {lock_file} — "
            f"{runtime_project_dir} existe mas não contém `uv.lock` commitado"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint do SAM 3 não encontrado: {checkpoint_path} — passe "
            f"--checkpoint, defina {CHECKPOINT_PATH_ENV_VAR} ou baixe-o no "
            "caminho padrão antes de rodar a extração"
        )


def _write_length_prefixed(stream: IO[bytes], payload: bytes) -> None:
    stream.write(struct.pack(_LENGTH_PREFIX_FORMAT, len(payload)))
    stream.write(payload)
    stream.flush()


def _read_length_prefixed(stream: IO[bytes]) -> bytes:
    header = stream.read(_LENGTH_PREFIX_SIZE)
    if len(header) != _LENGTH_PREFIX_SIZE:
        raise EOFError("runtime SAM 3 encerrou a conexão antes do esperado")
    (length,) = struct.unpack(_LENGTH_PREFIX_FORMAT, header)
    payload = stream.read(length)
    if len(payload) != length:
        raise EOFError("runtime SAM 3 encerrou a conexão no meio de uma mensagem")
    return payload


class Sam3RuntimeSegmenter:
    """Gerenciador de contexto que sobe e mantém vivo o worker `run_sam3.py`.

    Um único subprocesso é reaproveitado entre quadros (o carregamento do
    modelo é caro e o backbone permanece congelado — `CLAUDE.md`, invariante
    4), trocando apenas bytes pelo protocolo descrito no módulo.
    """

    def __init__(
        self,
        *,
        runtime_project_dir: Path | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self._runtime_project_dir = (
            runtime_project_dir.resolve()
            if runtime_project_dir is not None
            else resolve_runtime_project_dir(None)
        )
        self._checkpoint_path = (
            checkpoint_path.resolve()
            if checkpoint_path is not None
            else resolve_checkpoint_path(None)
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._runtime_manifest: dict[str, object] | None = None

    @property
    def runtime_manifest(self) -> dict[str, object]:
        if self._runtime_manifest is None:
            raise RuntimeError(
                "runtime_manifest só está disponível dentro do bloco `with`"
            )
        return self._runtime_manifest

    def __enter__(self) -> "Sam3RuntimeSegmenter":
        ensure_sam3_runtime_available(self._runtime_project_dir, self._checkpoint_path)
        argv, cwd = build_worker_invocation(self._runtime_project_dir, self._checkpoint_path)
        self._process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )
        assert self._process.stdout is not None
        manifest_line = self._process.stdout.readline()
        if not manifest_line:
            raise RuntimeError(
                "runtime SAM 3 encerrou antes de imprimir o manifesto de "
                "inicialização — verifique `uv sync` em sam3_runtime/"
            )
        try:
            self._runtime_manifest = json.loads(manifest_line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "runtime SAM 3 imprimiu um manifesto de inicialização que não "
                f"é JSON válido; linha bruta recebida: {manifest_line!r}"
            ) from exc
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None

        cleanup_error: BaseException | None = None
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError) as stdin_close_exc:
                cleanup_error = stdin_close_exc
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired as kill_wait_exc:
                cleanup_error = kill_wait_exc

        if cleanup_error is not None:
            print(
                "aviso: limpeza do processo do runtime SAM 3 encontrou um "
                f"problema ({cleanup_error!r}); processo pode ter ficado "
                "órfão — verifique manualmente",
                file=sys.stderr,
            )

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def segment_frame(
        self, frame_rgb: np.ndarray, text_prompt: str
    ) -> list[Sam3Instance]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Sam3RuntimeSegmenter usado fora do bloco `with`")

        height, width = frame_rgb.shape[0], frame_rgb.shape[1]
        header = json.dumps(
            {"height": height, "width": width, "text_prompt": text_prompt}
        ).encode("utf-8")
        _write_length_prefixed(self._process.stdin, header)
        _write_length_prefixed(
            self._process.stdin, np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes()
        )

        response_bytes = _read_length_prefixed(self._process.stdout)
        with np.load(BytesIO(response_bytes)) as npz_file:
            masks = npz_file["masks"]
            scores = npz_file["scores"]

        return [
            Sam3Instance(mask=masks[i].astype(bool), score=float(scores[i]))
            for i in range(masks.shape[0])
        ]
