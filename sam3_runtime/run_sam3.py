"""Worker de vida longa do SAM 3 real (facebookresearch/sam3, runtime oficial).

Roda isolado em `sam3_runtime/` (seu próprio `pyproject.toml` e `uv.lock`
commitado). Nunca importa `gatefall`: só troca bytes com o processo pai pelo
protocolo de fio descrito abaixo e em `gatefall.sam3.runtime`. Os dois lados
duplicam a implementação do protocolo de propósito — nenhum dos dois pode
importar o outro.

Protocolo (todo inteiro é uint32 big-endian):
- Ao iniciar, este processo imprime uma única linha JSON em stdout com o
  manifesto de runtime (`sam3_package_version`, `sam3_source_revision`,
  `torch_version_isolated`, `device`, `sam3_checkpoint_sha256`,
  `sam3_inference_autocast_dtype`) antes de processar qualquer quadro.
- Por quadro recebido em stdin: um cabeçalho JSON com prefixo de tamanho
  (`{"height", "width", "text_prompt"}`) seguido do RGB cru do quadro
  (`height*width*3` bytes) também com prefixo de tamanho. Responde em
  stdout com um blob `.npz` com prefixo de tamanho contendo `masks`
  ([N,H,W] bool) e `scores` ([N] float32).
"""

import argparse
import importlib.metadata
import json
import os
import struct
import sys
from io import BytesIO
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

_LENGTH_PREFIX_FORMAT = ">I"
_LENGTH_PREFIX_SIZE = struct.calcsize(_LENGTH_PREFIX_FORMAT)

_SAM3_PROCESSOR_RESOLUTION = 1008
_SAM3_CONFIDENCE_THRESHOLD = 0.5

_INFERENCE_AUTOCAST_DTYPE_NAMES: tuple[str, ...] = ("bfloat16", "float16")
_INFERENCE_AUTOCAST_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _select_inference_autocast_dtype_name(
    device: str, *, cuda_bf16_supported: bool
) -> str:
    # Duplicada de propósito em `gatefall.sam3.runtime`, como o protocolo de
    # fio: nenhum dos dois lados pode importar o outro.
    if device == "cuda":
        return "bfloat16" if cuda_bf16_supported else "float16"
    return "bfloat16"


def _resolve_inference_autocast_dtype_name(device: str) -> str:
    # `sam3.perflib.fused.addmm_act()` (upstream 2345a4a) converte a primeira
    # projeção do `Mlp` para bfloat16 incondicionalmente e devolve BF16 para
    # um `fc2` FP32; só autocast alinha as duas metades. `including_emulation`
    # precisa ser False: com o padrão True o torch 2.14 responde True em sm_75
    # apenas porque um tensor bfloat16 pode ser alocado, anulando em silêncio
    # o recuo para FP16 em hardware pré-Ampere.
    cuda_bf16_supported = device == "cuda" and torch.cuda.is_bf16_supported(
        including_emulation=False
    )
    return _select_inference_autocast_dtype_name(
        device, cuda_bf16_supported=cuda_bf16_supported
    )


def _write_length_prefixed(stream: BinaryIO, payload: bytes) -> None:
    stream.write(struct.pack(_LENGTH_PREFIX_FORMAT, len(payload)))
    stream.write(payload)
    stream.flush()


def _read_length_prefixed(stream: BinaryIO) -> bytes:
    header = stream.read(_LENGTH_PREFIX_SIZE)
    if len(header) != _LENGTH_PREFIX_SIZE:
        raise EOFError("conexão encerrada antes do esperado")
    (length,) = struct.unpack(_LENGTH_PREFIX_FORMAT, header)
    payload = stream.read(length)
    if len(payload) != length:
        raise EOFError("conexão encerrada no meio de uma mensagem")
    return payload


def _checkpoint_sha256(checkpoint: str) -> str:
    import hashlib
    from pathlib import Path

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        return ""
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Sam3SourceRevisionError(Exception):
    """Não foi possível resolver a revisão git da distribuição 'sam3' instalada."""


def _sam3_source_revision() -> str:
    try:
        distribution = importlib.metadata.distribution("sam3")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Sam3SourceRevisionError(
            "distribuição 'sam3' não encontrada em importlib.metadata — o "
            "pacote foi instalado corretamente?"
        ) from exc

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise Sam3SourceRevisionError(
            "direct_url.json ausente na distribuição 'sam3' instalada — ela "
            "não parece ter sido instalada a partir da URL git oficial "
            "(reinstalação a partir de wheel/cache local perde essa proveniência)"
        )

    try:
        direct_url = json.loads(direct_url_text)
        commit_id = direct_url["vcs_info"]["commit_id"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise Sam3SourceRevisionError(
            f"direct_url.json da distribuição 'sam3' não tem o formato "
            f"esperado (vcs_info.commit_id): {exc!r}"
        ) from exc

    if not commit_id:
        raise Sam3SourceRevisionError(
            "commit_id vazio em vcs_info de direct_url.json da distribuição 'sam3'"
        )
    return str(commit_id)


def _segment_frame(
    processor: Sam3Processor,
    frame_rgb: np.ndarray,
    text_prompt: str,
    *,
    autocast_device_type: str,
    autocast_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    image = Image.fromarray(frame_rgb, mode="RGB")
    with torch.autocast(device_type=autocast_device_type, dtype=autocast_dtype):
        state = processor.set_image(image)
        state = processor.set_text_prompt(text_prompt, state)

    masks = state["masks"].squeeze(1).to(torch.bool).cpu().numpy()
    scores = state["scores"].to(torch.float32).cpu().numpy()
    return masks, scores


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    real_stdout: BinaryIO = sys.stdout.buffer
    sys.stdout = sys.stderr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        bpe_path=None,
        device=device,
        eval_mode=True,
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    processor = Sam3Processor(
        model,
        resolution=_SAM3_PROCESSOR_RESOLUTION,
        device=device,
        confidence_threshold=_SAM3_CONFIDENCE_THRESHOLD,
    )

    try:
        sam3_source_revision = _sam3_source_revision()
    except Sam3SourceRevisionError as exc:
        print(f"sam3_runtime FALHOU: {exc}", file=sys.stderr)
        sys.exit(1)

    inference_autocast_dtype_name = _resolve_inference_autocast_dtype_name(device)
    inference_autocast_dtype = _INFERENCE_AUTOCAST_DTYPES[inference_autocast_dtype_name]

    manifest = {
        "sam3_package_version": importlib.metadata.version("sam3"),
        "sam3_source_revision": sam3_source_revision,
        "torch_version_isolated": torch.__version__,
        "device": device,
        "sam3_checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
        "sam3_inference_autocast_dtype": inference_autocast_dtype_name,
    }
    stdout = real_stdout
    stdout.write((json.dumps(manifest) + "\n").encode("utf-8"))
    stdout.flush()

    stdin: BinaryIO = sys.stdin.buffer
    while True:
        try:
            header_bytes = _read_length_prefixed(stdin)
        except EOFError:
            break
        header = json.loads(header_bytes.decode("utf-8"))
        height, width = int(header["height"]), int(header["width"])
        text_prompt = str(header["text_prompt"])

        frame_bytes = _read_length_prefixed(stdin)
        frame_rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)

        masks, scores = _segment_frame(
            processor,
            frame_rgb,
            text_prompt,
            autocast_device_type=device,
            autocast_dtype=inference_autocast_dtype,
        )

        buffer = BytesIO()
        np.savez(buffer, masks=masks, scores=scores)
        _write_length_prefixed(stdout, buffer.getvalue())


if __name__ == "__main__":
    main()
