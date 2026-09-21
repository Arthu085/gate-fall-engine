"""Worker de vida longa do SAM 3 real (facebook/sam3, via transformers).

Roda isolado em `sam3_runtime/` (seu próprio `pyproject.toml`, sem `uv.lock`
commitado — o operador roda `uv sync` aqui antes de extrair). Nunca importa
`gatefall`: só troca bytes com o processo pai pelo protocolo de fio descrito
abaixo e em `gatefall.sam3.runtime`. Os dois lados duplicam a implementação
do protocolo de propósito — nenhum dos dois pode importar o outro.

Protocolo (todo inteiro é uint32 big-endian):
- Ao iniciar, este processo imprime uma única linha JSON em stdout com o
  manifesto de runtime (`sam3_package_version`, `torch_version_isolated`,
  `device`) antes de processar qualquer quadro.
- Por quadro recebido em stdin: um cabeçalho JSON com prefixo de tamanho
  (`{"height", "width", "text_prompt"}`) seguido do RGB cru do quadro
  (`height*width*3` bytes) também com prefixo de tamanho. Responde em
  stdout com um blob `.npz` com prefixo de tamanho contendo `masks`
  ([N,H,W] bool) e `scores` ([N] float32).
"""

import argparse
import json
import os
import struct
import sys
from io import BytesIO
from typing import BinaryIO

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import Sam3Model, Sam3Processor

_LENGTH_PREFIX_FORMAT = ">I"
_LENGTH_PREFIX_SIZE = struct.calcsize(_LENGTH_PREFIX_FORMAT)


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


def _segment_frame(
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    frame_rgb: np.ndarray,
    text_prompt: str,
) -> tuple[np.ndarray, np.ndarray]:
    image = Image.fromarray(frame_rgb, mode="RGB")
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[(frame_rgb.shape[0], frame_rgb.shape[1])]
    )[0]

    masks = results["masks"].to(torch.bool).cpu().numpy()
    scores = results["scores"].to(torch.float32).cpu().numpy()
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
        scores = scores[np.newaxis, ...] if scores.ndim == 0 else scores
    return masks, scores


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    real_stdout: BinaryIO = sys.stdout.buffer
    sys.stdout = sys.stderr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained(args.checkpoint).to(device).eval()
    processor = Sam3Processor.from_pretrained(args.checkpoint)

    manifest = {
        "sam3_package_version": transformers.__version__,
        "torch_version_isolated": torch.__version__,
        "device": device,
        "sam3_checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
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

        masks, scores = _segment_frame(model, processor, device, frame_rgb, text_prompt)

        buffer = BytesIO()
        np.savez(buffer, masks=masks, scores=scores)
        _write_length_prefixed(stdout, buffer.getvalue())


if __name__ == "__main__":
    main()
