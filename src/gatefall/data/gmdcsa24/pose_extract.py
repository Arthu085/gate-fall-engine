"""Extração offline isolada de pose (YOLO-Pose + ByteTrack do braço A) dos vídeos elegíveis do GMDCSA24.

Implementa a etapa de extração autorizada em
`docs/analysis/gmdcsa24-weak-label-protocol.md#autorizacao-do-passo-4`: só os
145 vídeos incluídos pela saída validada do adapter, decodificados pelo pipe
passthrough do adapter e alinhados ao `src_index` de `frames.parquet`, com o
backbone congelado e a mesma inferência, seleção de pessoa e esquema HDF5 da
extração Le2i (`gatefall.pose.extract`). Grava só em
`data/features/gmdcsa24/pose/`, ignorado pelo Git e fora dos artefatos Le2i.
Não ajusta padronização, não monta treino e não calcula métrica de modelo.

Tudo falha fechado: adapter ausente ou inválido, E0 não refeito no commit
atual, checkout sujo, vídeo excluído, grade desalinhada, fonte com outro
SHA-256, modelo fora do contrato do braço A ou saída sobre artefato canônico.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

import gatefall.data.gmdcsa24.decoding as decoding_module
import gatefall.pose.extract as pose_extract_module
import gatefall.pose.selection as selection_module
from gatefall.config import TARGET_FPS
from gatefall.data.gmdcsa24.adapter import load_reference
from gatefall.data.gmdcsa24.decoding import GridFrameReader
from gatefall.data.gmdcsa24.provenance import DEFAULT_REVIEW_DIR, ensure_ignored_output
from gatefall.data.gmdcsa24.source import DEFAULT_ROOT, read_checkout_state, sha256_of
from gatefall.data.gmdcsa24.weak_label import (
    CANDIDATES_FILE,
    DEFAULT_OUTPUT_DIR,
    ENV,
    EXPECTED_EXCLUSIONS,
    FRAMES_FILE,
    MANIFEST_FILE,
    PUBLISHED_AGGREGATES,
    SPLIT,
    VIDEOS_FILE,
    WINDOW_FILES,
    Le2iReference,
    compare_expected,
    read_table,
    validate_outputs,
)
from gatefall.data.resampling import build_time_grid
from gatefall.datasets.le2i import LE2I_DATASET, PROTOCOL_PROCESSED_DIRS
from gatefall.hashing import sha256_dataframe
from gatefall.pose.extract import (
    N_KEYPOINTS,
    TRACKER_NAME,
    WEIGHTS_DIR,
    PoseExtractError,
    track_pose,
    write_pose_file,
)
from gatefall.pose.loading import pose_path
from gatefall.pose.smoke import DEFAULT_MODEL
from gatefall.runs import REPOSITORY_ROOT

DATASET = "gmdcsa24"
DEFAULT_POSE_DIR = Path("data/features/gmdcsa24/pose")
POSE_MANIFEST_FILE = "manifest.json"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / DEFAULT_MODEL

# A extração só grava num diretório próprio do GMDCSA24: nunca dentro, nem
# acima, das features Le2i, dos dados processados (Le2i-CS, Le2i-CV e a saída
# do adapter), dos rótulos, das fontes brutas, dos runs, das estatísticas de
# padronização ou da revisão de proveniência.
FORBIDDEN_POSE_DIRS = (
    LE2I_DATASET.pose_root.parent,
    *PROTOCOL_PROCESSED_DIRS.values(),
    DEFAULT_OUTPUT_DIR,
    Path("data/labels"),
    Path("data/raw"),
    Path("runs"),
    Path("src/gatefall/features/stats"),
    DEFAULT_REVIEW_DIR,
)

# Hashes de conteúdo publicados em "Resultado sobre os dados pinados" do
# protocolo; o item 2 do portão E0 exige que o build refeito os reproduza.
PUBLISHED_TABLE_SHA256 = {
    WINDOW_FILES["T"]: "dd904e1d21af41f36738157fb6bf61cc5792ffa8c6477f1e4dc8e86efff926a8",
    WINDOW_FILES["S"]: "bf9371715a152d7cbf795155f31a2c28ffef5f98410649aee5ed17c13862efcb",
    CANDIDATES_FILE: "4493cafd8772cc0df1435b9d188c5d6e67ffed26c0bcb4177c83ffeeedbd07c6",
}

# Contrato congelado do braço A. Os 190 HDF5 de pose do Le2i registram
# `yolo26n-pose.pt`, Ultralytics 8.4.131, `bytetrack.yaml` e 10 fps, mas não o
# hash dos pesos; o SHA-256 abaixo é o do `data/scratch/weights/yolo26n-pose.pt`
# local, o caminho que a extração Le2i resolve, com data anterior à dos HDF5 do Le2i.
FROZEN_POSE_CONTRACT: dict[str, object] = {
    "model_name": DEFAULT_MODEL,
    "weights_sha256": "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
    "ultralytics_version": "8.4.131",
    "tracker_name": TRACKER_NAME,
    "target_fps": TARGET_FPS,
    "n_keypoints": N_KEYPOINTS,
}

POSE_DATASETS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "keypoints": ((N_KEYPOINTS, 3), np.dtype(np.float32)),
    "bbox": ((4,), np.dtype(np.float32)),
    "person_found": ((), np.dtype(np.bool_)),
    "n_detections": ((), np.dtype(np.int16)),
    "track_id": ((), np.dtype(np.int32)),
}

CONDITIONS = ("adapter", "separação", "manifesto", "cobertura", "alinhamento", "HDF5")


@dataclass(frozen=True)
class ExtractionGate:
    """Expectativas fixas exigidas pela extração e pela validação.

    O selftest troca os valores publicados por expectativas sintéticas; o
    código de produção só usa `PINNED_GATE`."""

    pose_contract: dict[str, object]
    expected_exclusions: dict[str, list[str]]
    expected_aggregates: dict[str, object]
    table_sha256: dict[str, str]

    @property
    def included_videos(self) -> int:
        videos = cast(dict[str, int], self.expected_aggregates["videos"])
        return int(videos["included"])


PINNED_GATE = ExtractionGate(
    pose_contract=FROZEN_POSE_CONTRACT,
    expected_exclusions=EXPECTED_EXCLUSIONS,
    expected_aggregates=PUBLISHED_AGGREGATES,
    table_sha256=PUBLISHED_TABLE_SHA256,
)


@dataclass(frozen=True)
class AdapterInput:
    manifest: dict[str, object]
    manifest_sha256: str
    videos: pd.DataFrame
    frames: pd.DataFrame


@dataclass(frozen=True)
class VideoPlan:
    video_id: str
    subject: int
    relative_path: str
    source_sha256: str
    fps: float
    width: int
    height: int
    n_frames_decoded: int
    src_indices: list[int]
    grid_sha256: str


def pose_location_problems(pose_dir: Path) -> list[str]:
    resolved = pose_dir.resolve()
    problems: list[str] = []
    for forbidden in FORBIDDEN_POSE_DIRS:
        target = forbidden.resolve()
        if resolved == target or target in resolved.parents or resolved in target.parents:
            problems.append(f"{pose_dir} se sobrepõe a {forbidden}")
    try:
        ensure_ignored_output(pose_dir)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def read_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def section(manifest: dict[str, object], name: str) -> dict[str, object]:
    value = manifest.get(name)
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def load_adapter(adapter_dir: Path) -> AdapterInput:
    manifest_path = adapter_dir / MANIFEST_FILE
    return AdapterInput(
        manifest=read_json(manifest_path),
        manifest_sha256=sha256_of(manifest_path),
        videos=read_table(adapter_dir, VIDEOS_FILE),
        frames=read_table(adapter_dir, FRAMES_FILE),
    )


def adapter_problems(
    adapter_dir: Path, le2i: Le2iReference | None, gate: ExtractionGate
) -> list[str]:
    """As seis condições do adapter, os hashes publicados das tabelas e a seção `code` do E0."""
    results = validate_outputs(
        adapter_dir, le2i, gate.expected_exclusions, gate.expected_aggregates
    )
    problems = [
        f"adapter {condition}: {problem}"
        for condition, condition_problems in results.items()
        for problem in condition_problems
    ]
    if problems:
        return problems
    manifest = read_json(adapter_dir / MANIFEST_FILE)
    files = cast(dict[str, dict[str, object]], manifest.get("files", {}))
    for name, expected in gate.table_sha256.items():
        recorded = files.get(name, {}).get("content_sha256")
        if recorded != expected:
            problems.append(
                f"E0: {name} com hash {recorded!r}, o protocolo publica {expected!r}"
            )
    code = section(manifest, "code")
    if code.get("dirty") is not False:
        problems.append(f"E0: build do adapter com code.dirty = {code.get('dirty')!r}")
    commit = code.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        problems.append(f"E0: build do adapter sem code.commit válido ({commit!r})")
    return problems


def checkout_problems(adapter: AdapterInput, checkout: tuple[str, list[str]]) -> list[str]:
    """A extração roda no mesmo commit limpo em que o E0 refez o build do adapter."""
    head, status = checkout
    problems: list[str] = []
    if status:
        problems.append(
            f"checkout do GateFall com {len(status)} arquivo(s) modificado(s) ou não rastreado(s)"
        )
    recorded = section(adapter.manifest, "code").get("commit")
    if recorded != head:
        problems.append(
            f"E0 não refeito no commit atual: adapter gerado em {recorded!r}, HEAD {head!r}; "
            "rode `adapter build --force` e `adapter validate` neste checkout limpo"
        )
    return problems


def tracker_config_path() -> Path:
    return Path(ultralytics.__file__).parent / "cfg" / "trackers" / TRACKER_NAME


def model_identity(weights_path: Path) -> dict[str, object]:
    return {
        "model_name": weights_path.name,
        "weights_sha256": sha256_of(weights_path),
        "ultralytics_version": ultralytics.__version__,
        "tracker_name": TRACKER_NAME,
        "tracker_config_sha256": sha256_of(tracker_config_path()),
        "target_fps": TARGET_FPS,
        "n_keypoints": N_KEYPOINTS,
        "torch_version": torch.__version__,
    }


def contract_problems(identity: dict[str, object], gate: ExtractionGate) -> list[str]:
    observed = {key: identity.get(key) for key in gate.pose_contract}
    return compare_expected(gate.pose_contract, observed, "modelo.")


def extraction_code_sha256() -> str:
    """SHA-256 do código que decide a pose gravada: este módulo, a inferência
    compartilhada, a seleção de pessoa e o decodificador."""
    digest = hashlib.sha256()
    for module in (sys.modules[__name__], pose_extract_module, selection_module, decoding_module):
        digest.update(Path(inspect.getfile(module)).read_bytes())
    return digest.hexdigest()


def included_videos(videos: pd.DataFrame) -> pd.DataFrame:
    included = cast(pd.DataFrame, videos[~videos["excluded"]])
    return included.sort_values("video_id", ignore_index=True)


def eligibility_problems(video_ids: list[str], videos: pd.DataFrame) -> list[str]:
    known = {str(row["video_id"]): bool(row["excluded"]) for row in videos.to_dict("records")}
    problems: list[str] = []
    for video_id in video_ids:
        if video_id not in known:
            problems.append(f"{video_id}: fora de {VIDEOS_FILE}")
        elif known[video_id]:
            problems.append(f"{video_id}: vídeo excluído pela política do protocolo")
    return problems


def grid_sha256(video_frames: pd.DataFrame) -> str:
    alignment = cast(pd.DataFrame, video_frames[["frame_index", "time_s", "src_index"]])
    return sha256_dataframe(alignment.reset_index(drop=True))


def grid_problems(video: dict[str, object], video_frames: pd.DataFrame) -> list[str]:
    """Grade de um vídeo contra a receita do adapter e a contagem do decodificador."""
    video_id = str(video["video_id"])
    k = len(video_frames)
    src = video_frames["src_index"].to_numpy(np.int64)
    n_counted = int(cast(int, video["n_frames_counted"]))
    n_decoded = video["n_frames_decoded"]
    _, expected_src = build_time_grid(n_counted, float(cast(float, video["fps"])), TARGET_FPS)
    problems: list[str] = []
    if k == 0:
        return [f"{video_id}: sem quadros em {FRAMES_FILE}"]
    if k != int(cast(int, video["grid_frames"])):
        problems.append(f"{video_id}: {k} quadros na grade, grid_frames = {video['grid_frames']}")
    if not np.array_equal(video_frames["frame_index"].to_numpy(np.int64), np.arange(k)):
        problems.append(f"{video_id}: frame_index não é 0..K-1 em ordem")
    if not np.array_equal(src, expected_src.astype(np.int64)):
        problems.append(f"{video_id}: src_index diverge de build_time_grid")
    if np.any(np.diff(src) <= 0):
        problems.append(f"{video_id}: src_index repetido ou fora de ordem (quadro duplicado)")
    if n_decoded is None or n_decoded is pd.NA or int(cast(int, n_decoded)) != n_counted:
        problems.append(
            f"{video_id}: decodificador do adapter com {n_decoded} quadros, "
            f"ffprobe conta {n_counted}"
        )
    elif int(src.max()) >= n_counted:
        problems.append(f"{video_id}: src_index {int(src.max())} além dos {n_counted} quadros")
    return problems


def build_plans(adapter: AdapterInput) -> tuple[dict[str, VideoPlan], list[str]]:
    frames_by_video = {
        str(key): cast(pd.DataFrame, group).sort_values("frame_index")
        for key, group in adapter.frames.groupby("video_id")
    }
    plans: dict[str, VideoPlan] = {}
    problems: list[str] = []
    for video in included_videos(adapter.videos).to_dict("records"):
        video_id = str(video["video_id"])
        video_frames = frames_by_video.get(video_id, cast(pd.DataFrame, adapter.frames.iloc[0:0]))
        video_problems = grid_problems(video, video_frames)
        problems.extend(video_problems)
        if video_problems:
            continue
        plans[video_id] = VideoPlan(
            video_id=video_id,
            subject=int(video["subject"]),
            relative_path=str(video["relative_path"]),
            source_sha256=str(video["source_sha256"]),
            fps=float(video["fps"]),
            width=int(video["width"]),
            height=int(video["height"]),
            n_frames_decoded=int(video["n_frames_decoded"]),
            src_indices=[int(index) for index in video_frames["src_index"]],
            grid_sha256=grid_sha256(video_frames),
        )
    return plans, problems


def expected_attrs(
    plan: VideoPlan, identity: dict[str, object], provenance: dict[str, str]
) -> dict[str, object]:
    """Atributos do esquema de pose do Le2i mais a proveniência do GMDCSA24."""
    return {
        "video_id": plan.video_id,
        "env": ENV,
        "split": SPLIT,
        "subject": plan.subject,
        "K": len(plan.src_indices),
        "fps": plan.fps,
        "width": plan.width,
        "height": plan.height,
        "model_name": identity["model_name"],
        "ultralytics_version": identity["ultralytics_version"],
        "tracker_name": identity["tracker_name"],
        "target_fps": identity["target_fps"],
        "dataset": DATASET,
        "relative_path": plan.relative_path,
        "source_sha256": plan.source_sha256,
        "grid_sha256": plan.grid_sha256,
        "n_frames_decoded": plan.n_frames_decoded,
        "weights_sha256": identity["weights_sha256"],
        "tracker_config_sha256": identity["tracker_config_sha256"],
        "adapter_manifest_sha256": provenance["adapter_manifest_sha256"],
        "code_commit": provenance["code_commit"],
        "extraction_code_sha256": provenance["extraction_code_sha256"],
    }


def plain(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def pose_file_problems(path: Path, attrs: dict[str, object]) -> list[str]:
    """Estrutura, atributos e coerência interna de um HDF5 de pose já gravado."""
    k = int(cast(int, attrs["K"]))
    try:
        with h5py.File(path, "r") as h5_file:
            problems = [
                f"{path}: atributo {key} = {plain(h5_file.attrs.get(key))!r}, esperado {expected!r}"
                for key, expected in attrs.items()
                if plain(h5_file.attrs.get(key)) != expected
            ]
            arrays: dict[str, np.ndarray] = {}
            for name, (tail, dtype) in POSE_DATASETS.items():
                dataset = h5_file.get(name)
                if not isinstance(dataset, h5py.Dataset):
                    problems.append(f"{path}: dataset {name} ausente")
                elif dataset.shape != (k, *tail) or dataset.dtype != dtype:
                    problems.append(
                        f"{path}: dataset {name} com shape {dataset.shape} e dtype {dataset.dtype}, "
                        f"esperado {(k, *tail)} {dtype}"
                    )
                else:
                    arrays[name] = dataset[()]
    except OSError as exc:
        return [f"{path}: HDF5 ilegível ou incompleto ({exc})"]
    if problems:
        return problems
    found = arrays["person_found"]
    if not np.array_equal(found, arrays["n_detections"] > 0):
        problems.append(f"{path}: person_found diverge de n_detections > 0")
    if np.any(arrays["track_id"][~found] != -1):
        problems.append(f"{path}: track_id definido em quadro sem pessoa")
    if np.any(arrays["keypoints"][~found] != 0) or np.any(arrays["bbox"][~found] != 0):
        problems.append(f"{path}: keypoints ou bbox não nulos em quadro sem pessoa")
    if not (np.isfinite(arrays["keypoints"]).all() and np.isfinite(arrays["bbox"]).all()):
        problems.append(f"{path}: keypoints ou bbox não finitos")
    return problems


def extract_video(
    plan: VideoPlan,
    root: Path,
    pose_dir: Path,
    model: YOLO,
    attrs: dict[str, object],
    force: bool,
) -> int | None:
    """Extrai um vídeo e devolve quantos quadros têm pessoa; `None` quando um
    HDF5 já existente confere e `force` não foi passado. A fonte é conferida
    antes, então um HDF5 válido nunca dispensa a verificação do MP4."""
    source = root / plan.relative_path
    if not source.is_file():
        raise PoseExtractError(f"{source} não encontrado")
    if sha256_of(source) != plan.source_sha256:
        raise PoseExtractError(f"{source}: SHA-256 diverge do source_sha256 do adapter")
    output_path = pose_path(plan.video_id, pose_root=pose_dir)
    if output_path.exists() and not force:
        problems = pose_file_problems(output_path, attrs)
        if problems:
            raise PoseExtractError(
                f"{output_path} já existe e não confere ({problems[0]}); use --force"
            )
        return None
    reader = GridFrameReader(source, plan.src_indices, plan.width, plan.height)
    track = track_pose(model, reader, len(plan.src_indices))
    if reader.decoded_frames != plan.n_frames_decoded:
        raise PoseExtractError(
            f"{source}: decodificador entregou {reader.decoded_frames} quadros, "
            f"o adapter registrou {plan.n_frames_decoded}"
        )
    write_pose_file(output_path, track, attrs)
    return int(track.person_found.sum())


def file_record(pose_dir: Path, plan: VideoPlan) -> dict[str, object]:
    """Registro de um vídeo no manifesto, recalculado do plano e do HDF5 gravado."""
    path = pose_path(plan.video_id, pose_root=pose_dir)
    with h5py.File(path, "r") as h5_file:
        found = cast(h5py.Dataset, h5_file["person_found"])[()]
    return {
        "path": path.relative_to(pose_dir).as_posix(),
        "file_sha256": sha256_of(path),
        "K": int(found.shape[0]),
        "n_found": int(found.sum()),
        "source_sha256": plan.source_sha256,
        "grid_sha256": plan.grid_sha256,
        "n_frames_decoded": plan.n_frames_decoded,
    }


def file_records(
    pose_dir: Path, plans: dict[str, VideoPlan]
) -> dict[str, dict[str, object]]:
    return {video_id: file_record(pose_dir, plan) for video_id, plan in sorted(plans.items())}


def summarize(records: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "videos": len(records),
        "grid_frames": sum(int(cast(int, record["K"])) for record in records.values()),
        "frames_with_person": sum(int(cast(int, record["n_found"])) for record in records.values()),
    }


def write_manifest(
    pose_dir: Path,
    adapter_dir: Path,
    adapter: AdapterInput,
    identity: dict[str, object],
    provenance: dict[str, str],
    records: dict[str, dict[str, object]],
) -> Path:
    """Grava por último, só com a cobertura completa; sem ele a saída não valida."""
    files = cast(dict[str, dict[str, object]], adapter.manifest.get("files", {}))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET,
        "protocol": {
            key: section(adapter.manifest, "protocol").get(key) for key in ("path", "sha256")
        },
        "source": {
            key: section(adapter.manifest, "source").get(key)
            for key in ("repository", "tag", "commit", "root")
        },
        "adapter": {
            "dir": adapter_dir.as_posix(),
            "manifest_sha256": provenance["adapter_manifest_sha256"],
            "code_commit": section(adapter.manifest, "code").get("commit"),
            "table_sha256": {
                name: files.get(name, {}).get("content_sha256")
                for name in (*WINDOW_FILES.values(), CANDIDATES_FILE, FRAMES_FILE, VIDEOS_FILE)
            },
        },
        "model": identity,
        "code": {
            "commit": provenance["code_commit"],
            "dirty": False,
            "extraction_code_sha256": provenance["extraction_code_sha256"],
        },
        "decoder": decoding_module.__name__ + ".GridFrameReader (ffmpeg -fps_mode passthrough)",
        "summary": summarize(records),
        "videos": records,
    }
    manifest_path = pose_dir / POSE_MANIFEST_FILE
    tmp_path = manifest_path.with_suffix(f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, manifest_path)
    return manifest_path


def run_extract(
    root: Path,
    adapter_dir: Path,
    pose_dir: Path,
    video_ids: list[str],
    force: bool,
    *,
    le2i: Le2iReference | None,
    gate: ExtractionGate = PINNED_GATE,
    checkout: tuple[str, list[str]] | None = None,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    model: YOLO | None = None,
) -> bool:
    problems = pose_location_problems(pose_dir) + adapter_problems(adapter_dir, le2i, gate)
    if problems:
        return report_problems("extract", problems)
    adapter = load_adapter(adapter_dir)
    problems.extend(
        checkout_problems(adapter, checkout or read_checkout_state(REPOSITORY_ROOT))
    )
    manifest_path = pose_dir / POSE_MANIFEST_FILE
    if manifest_path.exists() and not force:
        problems.append(f"{manifest_path} já existe; rode `validate` ou use --force para refazer")
    problems.extend(eligibility_problems(video_ids, adapter.videos))
    identity = model_identity(weights_path) if weights_path.is_file() else None
    if identity is None:
        problems.append(
            f"{weights_path} não encontrado; a extração não baixa pesos, use os do braço A"
        )
    else:
        problems.extend(contract_problems(identity, gate))
    plans, grid = build_plans(adapter)
    problems.extend(grid)
    if len(included_videos(adapter.videos)) != gate.included_videos:
        problems.append(
            f"{VIDEOS_FILE} com {len(included_videos(adapter.videos))} vídeos incluídos, "
            f"o protocolo fixa {gate.included_videos}"
        )
    if problems or identity is None:
        return report_problems("extract", problems)

    provenance = {
        "adapter_manifest_sha256": adapter.manifest_sha256,
        "code_commit": str(section(adapter.manifest, "code")["commit"]),
        "extraction_code_sha256": extraction_code_sha256(),
    }
    manifest_path.unlink(missing_ok=True)
    if model is None:
        model = YOLO(str(weights_path))
    requested = video_ids or sorted(plans)
    failures: list[str] = []
    for video_id in requested:
        plan = plans[video_id]
        attrs = expected_attrs(plan, identity, provenance)
        try:
            n_found = extract_video(plan, root, pose_dir, model, attrs, force)
        except (PoseExtractError, OSError, ValueError) as exc:
            failures.append(f"{video_id}: {exc}")
            print(f"FALHA {video_id}: {exc}", file=sys.stderr)
            continue
        k = len(plan.src_indices)
        if n_found is None:
            print(f"skip {video_id}: HDF5 existente confere")
        else:
            print(f"{video_id}: K={k}, quadros com pessoa={n_found} ({100.0 * n_found / k:.1f}%)")
    if failures:
        return report_problems("extract", failures)

    missing = [
        video_id
        for video_id, plan in plans.items()
        if pose_file_problems(
            pose_path(video_id, pose_root=pose_dir), expected_attrs(plan, identity, provenance)
        )
    ]
    if missing:
        print(
            f"\nmanifesto não gravado: {len(plans) - len(missing)} de {len(plans)} vídeos "
            "extraídos e conferidos; rode `extract` sem --video-id para completar"
        )
        return bool(video_ids) or report_problems("extract", ["cobertura incompleta"])
    records = file_records(pose_dir, plans)
    path = write_manifest(pose_dir, adapter_dir, adapter, identity, provenance, records)
    print(f"\nmanifesto gravado em {path}")
    return report_validation(validate_pose_output(pose_dir, adapter_dir, le2i, gate))


def manifest_problems(
    manifest: dict[str, object], adapter: AdapterInput, gate: ExtractionGate
) -> list[str]:
    identity = section(manifest, "model")
    problems = contract_problems(identity, gate)
    adapter_section = section(manifest, "adapter")
    adapter_commit = section(adapter.manifest, "code").get("commit")
    if adapter_section.get("manifest_sha256") != adapter.manifest_sha256:
        problems.append(
            "adapter.manifest_sha256 diverge do manifesto atual do adapter; "
            "a pose foi extraída de outra saída do adapter"
        )
    if adapter_section.get("code_commit") != adapter_commit:
        problems.append(f"adapter.code_commit diverge de {adapter_commit!r}")
    code = section(manifest, "code")
    if code.get("commit") != adapter_commit or code.get("dirty") is not False:
        problems.append(
            f"code: extração em {code.get('commit')!r} (dirty = {code.get('dirty')!r}), "
            f"E0 em {adapter_commit!r}"
        )
    protocol = section(adapter.manifest, "protocol")
    if section(manifest, "protocol").get("sha256") != protocol.get("sha256"):
        problems.append("protocol.sha256 diverge do registrado pelo adapter")
    return problems


def coverage_problems(pose_dir: Path, plans: dict[str, VideoPlan], videos: pd.DataFrame) -> list[str]:
    expected = {
        pose_path(video_id, pose_root=pose_dir).relative_to(pose_dir).as_posix()
        for video_id in plans
    }
    found = {
        path.relative_to(pose_dir).as_posix()
        for path in (pose_dir.rglob("*") if pose_dir.is_dir() else [])
        if path.is_file() and path.relative_to(pose_dir).as_posix() != POSE_MANIFEST_FILE
    }
    excluded = {
        pose_path(str(video_id), pose_root=pose_dir).relative_to(pose_dir).as_posix()
        for video_id in videos.loc[videos["excluded"], "video_id"]
    }
    if not found:
        return [f"nenhum HDF5 em {pose_dir}: os {len(expected)} vídeos elegíveis estão ausentes"]
    problems = [f"vídeo excluído extraído: {path}" for path in sorted(found & excluded)]
    problems.extend(f"HDF5 ausente: {path}" for path in sorted(expected - found))
    problems.extend(
        f"arquivo inesperado ou incompleto: {path}" for path in sorted(found - expected - excluded)
    )
    return problems


def not_verified(conditions: tuple[str, ...]) -> dict[str, list[str]]:
    return {condition: ["não verificado: pré-requisito anterior reprovado"] for condition in conditions}


def validate_pose_output(
    pose_dir: Path,
    adapter_dir: Path,
    le2i: Le2iReference | None,
    gate: ExtractionGate = PINNED_GATE,
) -> dict[str, list[str]]:
    """Problemas por condição; a extração só vale com todas as listas vazias."""
    results: dict[str, list[str]] = {condition: [] for condition in CONDITIONS}
    results["separação"] = pose_location_problems(pose_dir)
    results["adapter"] = adapter_problems(adapter_dir, le2i, gate)
    if results["adapter"]:
        return {**results, **not_verified(("manifesto", "cobertura", "alinhamento", "HDF5"))}
    adapter = load_adapter(adapter_dir)
    plans, results["alinhamento"] = build_plans(adapter)
    if len(plans) != gate.included_videos:
        results["alinhamento"].append(
            f"{len(plans)} vídeos elegíveis alinhados, o protocolo fixa {gate.included_videos}"
        )
    manifest_path = pose_dir / POSE_MANIFEST_FILE
    if not manifest_path.is_file():
        results["manifesto"] = [f"{manifest_path} não encontrado; extração incompleta"]
        results["cobertura"] = coverage_problems(pose_dir, plans, adapter.videos)
        return {**results, **not_verified(("HDF5",))}
    manifest = read_json(manifest_path)
    results["manifesto"] = manifest_problems(manifest, adapter, gate)
    results["cobertura"] = coverage_problems(pose_dir, plans, adapter.videos)
    records = section(manifest, "videos")
    results["cobertura"].extend(
        f"vídeo elegível ausente do manifesto: {video_id}"
        for video_id in sorted(set(plans) - set(records))
    )
    results["cobertura"].extend(
        f"vídeo inesperado no manifesto: {video_id}"
        for video_id in sorted(set(records) - set(plans))
    )
    identity = section(manifest, "model")
    code = section(manifest, "code")
    provenance = {
        "adapter_manifest_sha256": adapter.manifest_sha256,
        "code_commit": str(code.get("commit")),
        "extraction_code_sha256": str(code.get("extraction_code_sha256")),
    }
    verified: dict[str, dict[str, object]] = {}
    for video_id, plan in sorted(plans.items()):
        path = pose_path(video_id, pose_root=pose_dir)
        if not path.is_file():
            continue
        file_problems = pose_file_problems(path, expected_attrs(plan, identity, provenance))
        results["HDF5"].extend(file_problems)
        if file_problems:
            continue
        verified[video_id] = file_record(pose_dir, plan)
        if video_id in records:
            label = f"manifesto videos.{video_id}."
            results["HDF5"].extend(compare_expected(verified[video_id], records[video_id], label))
    # Um HDF5 ausente ou inválido já reprova cobertura ou HDF5; o resumo só é
    # recalculado quando todos os vídeos elegíveis foram conferidos.
    if len(verified) == len(plans):
        results["manifesto"].extend(
            compare_expected(summarize(verified), manifest.get("summary"), "summary.")
        )
    return results


def report_problems(command: str, problems: list[str]) -> bool:
    print(f"\n{command} recusado:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    return False


def report_validation(results: dict[str, list[str]]) -> bool:
    print("\n=== validação da pose do GMDCSA24 ===")
    for condition, problems in results.items():
        print(f"[{'FAIL' if problems else 'PASS'}] {condition}")
        for problem in problems:
            print(f"  {problem}")
    print("\nnenhuma métrica de modelo é calculada; só cobertura e proveniência")
    return all(not problems for problems in results.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extrai a pose dos 145 vídeos elegíveis com o backbone congelado do braço A",
    )
    extract_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    extract_parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    extract_parser.add_argument("--output-dir", type=Path, default=DEFAULT_POSE_DIR)
    extract_parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Extrai só este vídeo elegível (repetível); o manifesto só sai com os 145",
    )
    extract_parser.add_argument("--force", action="store_true", help="Refaz HDF5 existentes")
    validate_parser = subparsers.add_parser(
        "validate", help="Confere cobertura, alinhamento, proveniência e estrutura da pose gravada"
    )
    validate_parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    validate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_POSE_DIR)
    subparsers.add_parser(
        "selftest", help="Verifica a extração com vídeos sintéticos e um modelo falso"
    )

    args = parser.parse_args()
    if args.command == "selftest":
        from gatefall.data.gmdcsa24.selftests.pose_extract import run_pose_extract_selftest

        run_pose_extract_selftest()
        return
    le2i = load_reference()
    if args.command == "extract":
        ok = run_extract(
            args.root, args.adapter_dir, args.output_dir, args.video_id, args.force, le2i=le2i
        )
    else:
        ok = report_validation(validate_pose_output(args.output_dir, args.adapter_dir, le2i))
    if not ok:
        print(f"\n{args.command} FALHOU", file=sys.stderr)
        sys.exit(1)
    print(f"\n{args.command} OK")


if __name__ == "__main__":
    main()
