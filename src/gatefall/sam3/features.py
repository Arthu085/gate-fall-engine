"""Leitura validada dos descritores V_t do SAM 3 para consumo por armas de fusão.

Só lê os `.h5` já gravados por `gatefall.sam3.extract`: nunca sobe o runtime
isolado nem altera o schema persistido. Antes de qualquer janela ser montada,
confere a identidade de cada arquivo (`video_id`, `K`, `split`), a estrutura
(`storage.validate_existing_file`) e a proveniência homogênea do conjunto.
"""

import hashlib
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from gatefall.config import TARGET_FPS
from gatefall.hashing import sha256_file
from gatefall.sam3 import storage
from gatefall.sam3.descriptors import V_T_DIM
from gatefall.sam3.extract import MODEL_NAME
from gatefall.sam3.report import find_provenance_divergences
from gatefall.sam3.runtime import TEXT_PROMPT
from gatefall.sam3.storage import sam3_path


def _normalize_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _missing_file_error(path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"features SAM 3 não encontradas: {path}; rode "
        "`uv run python -m gatefall.sam3.extract extract-all` primeiro"
    )


def load_v_t(video_id: str, *, sam3_root: Path) -> np.ndarray:
    path = sam3_path(video_id, sam3_root=sam3_root)
    if not path.exists():
        raise _missing_file_error(path)
    try:
        with h5py.File(path, "r") as h5_file:
            stored_video_id = _normalize_attr(h5_file.attrs.get("video_id", ""))
            stored_k = int(cast(int, h5_file.attrs.get("K", -1)))
    except OSError as exc:
        raise ValueError(f"video_id={video_id!r}: {path} ilegível ({exc})") from exc
    if stored_video_id != video_id:
        raise ValueError(
            f"video_id={video_id!r}: {path} declara video_id={stored_video_id!r}"
        )
    reasons = storage.validate_existing_file(
        path, expected_k=stored_k, v_t_dim=V_T_DIM, expected_attrs={}
    )
    if reasons:
        raise ValueError(
            f"video_id={video_id!r}: {path} estruturalmente inválido: "
            f"{'; '.join(reasons)}"
        )
    v_t = storage.read_v_t(path)
    if not bool(np.isfinite(v_t).all()):
        raise ValueError(f"video_id={video_id!r}: {path} contém V_t não finito")
    return v_t.astype(np.float32)


def collect_sam3_provenance(
    split_by_video: dict[str, str], *, sam3_root: Path
) -> dict[str, str]:
    """Devolve a proveniência comum a todos os `.h5` do conjunto pedido.

    Falha com `ValueError` quando algum arquivo diverge do split da tabela de
    frames, traz proveniência obrigatória ausente/malformada, diverge da
    proveniência dos demais ou foge do contrato congelado de extração
    (modelo, prompt `"person"`, `TARGET_FPS`).
    """
    attrs_by_video: dict[str, dict[str, object]] = {}
    problems: list[str] = []
    for video_id in sorted(split_by_video):
        path = sam3_path(video_id, sam3_root=sam3_root)
        if not path.exists():
            raise _missing_file_error(path)
        with h5py.File(path, "r") as h5_file:
            attrs: dict[str, object] = {
                name: h5_file.attrs[name]
                for name in storage.PROVENANCE_ATTR_NAMES
                if name in h5_file.attrs
            }
            stored_split = _normalize_attr(h5_file.attrs.get("split", ""))
        if stored_split != split_by_video[video_id]:
            problems.append(
                f"{video_id}: split {stored_split!r} no .h5 diverge de "
                f"{split_by_video[video_id]!r} na tabela de frames"
            )
        normalized = {name: _normalize_attr(value) for name, value in attrs.items()}
        problems.extend(
            f"{video_id}: {reason}"
            for reason in storage.find_invalid_required_provenance(
                cast(dict[str, object], normalized),
                storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES,
            )
        )
        attrs_by_video[video_id] = attrs

    problems.extend(find_provenance_divergences(attrs_by_video))
    if problems:
        raise ValueError(
            "proveniência SAM 3 inválida: " + "; ".join(problems)
        )
    if not attrs_by_video:
        raise ValueError("nenhum vídeo informado para validar a proveniência SAM 3")

    reference = next(iter(attrs_by_video.values()))
    provenance = {
        name: _normalize_attr(reference[name]) for name in storage.PROVENANCE_ATTR_NAMES
    }
    contract = {
        "model_name": MODEL_NAME,
        "text_prompt": TEXT_PROMPT,
        "target_fps": str(TARGET_FPS),
    }
    contract_violations = [
        f"{name}={provenance[name]!r} (esperado {expected!r})"
        for name, expected in contract.items()
        if provenance[name] != expected
    ]
    if contract_violations:
        raise ValueError(
            "features SAM 3 fora do contrato de extração congelado: "
            + ", ".join(contract_violations)
        )
    return provenance


def sam3_set_sha256(video_ids: list[str], *, sam3_root: Path) -> str:
    """Digest determinístico do conjunto de `.h5` do SAM 3.

    Mesmo esquema de `quality_set_sha256`: sha256 das linhas
    `<video_id> <sha256 do arquivo>` ordenadas por `video_id`, sensível a
    qualquer reextração de qualquer vídeo.
    """
    digest = hashlib.sha256()
    for video_id in sorted(video_ids):
        path = sam3_path(video_id, sam3_root=sam3_root)
        if not path.exists():
            raise _missing_file_error(path)
        digest.update(f"{video_id} {sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()
