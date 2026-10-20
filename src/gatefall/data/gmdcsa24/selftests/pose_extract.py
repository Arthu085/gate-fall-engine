"""Selftest sintético da extração de pose isolada do GMDCSA24 (`pose_extract.py`).

Reaproveita o clone sintético do selftest do adapter (seis vídeos de cinza
crescente, um de taxa variável e dois excluídos) e grava uma saída de adapter
válida num diretório temporário. O YOLO é trocado por um modelo falso: a
coordenada x dos keypoints é o cinza médio do quadro recebido, o que identifica
o quadro de origem, e o quadro preto (quadro 0) não tem detecção. O tracker
falso imita o ByteTrack: perde a track sem detecção e dá um ID novo na
reaquisição, de modo que um estado herdado de outro vídeo aparece no
`track_id`. Não toca no dataset real, no Hugging Face, no Le2i nem nos pesos
reais. Requer FFmpeg.
"""

import contextlib
import io
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import h5py
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from gatefall.data.gmdcsa24.decoding import GridFrameReader
from gatefall.data.gmdcsa24.pose_extract import (
    FROZEN_POSE_CONTRACT,
    POSE_MANIFEST_FILE,
    AdapterInput,
    ExtractionGate,
    VideoPlan,
    build_plans,
    expected_attrs,
    extract_video,
    extraction_code_sha256,
    grid_problems,
    load_adapter,
    model_identity,
    pose_location_problems,
    run_extract,
    validate_pose_output,
)
from gatefall.data.gmdcsa24.selftests.adapter import (
    _EVAL_HASHES,
    _EXPECTED_AGGREGATES,
    _EXPECTED_EXCLUSIONS,
    _VFR_VIDEO,
    _VIDEOS,
    _check,
    _encode,
    _le2i_reference,
    _omnifall,
    _original_segments,
    _provenance,
    _source_index,
    _video_path,
)
from gatefall.data.gmdcsa24.source import build_inventory, sha256_of
from gatefall.data.gmdcsa24.weak_label import (
    CANDIDATES_FILE,
    MANIFEST_FILE,
    WINDOW_FILES,
    build_tables,
    probe_sources,
    read_table,
    write_outputs,
    write_parquet,
)
from gatefall.data.video_io import probe_frame_count
from gatefall.pose.extract import PoseExtractError
from gatefall.pose.loading import pose_path
from gatefall.runs import REPOSITORY_ROOT

_COMMIT = "c" * 40
_Select = Callable[[dict[str, object]], dict[str, object]]
_LE2I = _le2i_reference(_EVAL_HASHES)


class _FakeTracker:
    """Perde a track sem detecção; a reaquisição recebe o próximo ID, como no ByteTrack."""

    def __init__(self) -> None:
        self.next_id = 1
        self.active: int | None = None

    def reset(self) -> None:
        self.next_id = 1
        self.active = None

    def assign(self, detected: bool) -> int | None:
        if not detected:
            self.active = None
            return None
        if self.active is None:
            self.active = self.next_id
            self.next_id += 1
        return self.active


class _FakeBoxes:
    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, ids: torch.Tensor | None) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.id = ids

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _FakeModel:
    """Uma detecção por quadro não preto, com keypoints x = cinza médio do quadro."""

    def __init__(self) -> None:
        self.predictor: SimpleNamespace | None = None
        self.calls = 0

    @property
    def tracker(self) -> _FakeTracker:
        if self.predictor is None:
            self.predictor = SimpleNamespace(trackers=[_FakeTracker()])
        return cast(_FakeTracker, self.predictor.trackers[0])

    def track(self, frame_bgr: np.ndarray, **_: object) -> list[SimpleNamespace]:
        self.calls += 1
        gray = float(frame_bgr.mean())
        track_id = self.tracker.assign(gray >= 1.0)
        if track_id is None:
            empty = _FakeBoxes(torch.zeros((0, 4)), torch.zeros((0,)), None)
            return [SimpleNamespace(boxes=empty, keypoints=None)]
        xy = torch.zeros((1, 17, 2))
        xy[:, :, 0] = gray
        xy[:, :, 1] = 1.0
        boxes = _FakeBoxes(
            torch.tensor([[10.0, 10.0, 30.0, 40.0]]),
            torch.tensor([0.9]),
            torch.tensor([float(track_id)]),
        )
        keypoints = SimpleNamespace(xy=xy, conf=torch.full((1, 17), 0.8))
        return [SimpleNamespace(boxes=boxes, keypoints=keypoints)]


@dataclass(frozen=True)
class _Setup:
    root: Path
    adapter_dir: Path
    weights: Path
    gate: ExtractionGate


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _extract(
    setup: _Setup,
    pose_dir: Path,
    model: _FakeModel,
    *,
    video_ids: list[str] | None = None,
    force: bool = False,
    adapter_dir: Path | None = None,
    gate: ExtractionGate | None = None,
    checkout: tuple[str, list[str]] = (_COMMIT, []),
    weights: Path | None = None,
) -> bool:
    with _quiet():
        return run_extract(
            setup.root,
            adapter_dir or setup.adapter_dir,
            pose_dir,
            video_ids or [],
            force,
            le2i=_LE2I,
            gate=gate or setup.gate,
            checkout=checkout,
            weights_path=weights or setup.weights,
            model=cast(YOLO, model),
        )


def _validate(setup: _Setup, pose_dir: Path, adapter_dir: Path | None = None) -> dict[str, list[str]]:
    return validate_pose_output(pose_dir, adapter_dir or setup.adapter_dir, _LE2I, setup.gate)


def _failed(results: dict[str, list[str]]) -> set[str]:
    return {condition for condition, problems in results.items() if problems}


def _plans(setup: _Setup) -> dict[str, VideoPlan]:
    plans, problems = build_plans(load_adapter(setup.adapter_dir))
    assert not problems, problems
    return plans


def _read(pose_dir: Path, video_id: str) -> dict[str, np.ndarray]:
    with h5py.File(pose_path(video_id, pose_root=pose_dir), "r") as h5_file:
        return {str(name): cast(h5py.Dataset, h5_file[name])[()] for name in h5_file.keys()}


def _step(video_id: str) -> int:
    return _VIDEOS[video_id][2]


def check_decoder(root: Path) -> list[bool]:
    path = _video_path(root, _VFR_VIDEO)
    _, n_frames, step, _ = _VIDEOS[_VFR_VIDEO]
    full = GridFrameReader(path, list(range(n_frames)), 64, 48)
    indices = [_source_index(frame, step) for frame in full]
    wanted = [0, 3, 14, 15, 16, 29]
    subset = GridFrameReader(path, wanted, 64, 48)
    picked = [_source_index(frame, step) for frame in subset]
    rejected: list[bool] = []
    for bad in ([0, 2, 2, 5], [0, 5, 3]):
        try:
            GridFrameReader(path, bad, 64, 48)
            rejected.append(False)
        except ValueError:
            rejected.append(True)
    return [
        _check(
            "taxa variável: o pipe sem passthrough duplica quadros; o leitor da grade entrega "
            "cada quadro uma vez e conta o total",
            probe_frame_count(path) > n_frames
            and indices == list(range(n_frames))
            and full.decoded_frames == n_frames,
        ),
        _check(
            "taxa variável: os quadros pedidos em volta do salto de PTS são os de mesmo índice "
            "de origem, e o vídeo é lido até o fim",
            picked == wanted and subset.decoded_frames == n_frames,
        ),
        _check("src_index repetido ou fora de ordem é recusado antes de decodificar", all(rejected)),
    ]


def check_grid_alignment(adapter: AdapterInput) -> list[bool]:
    included = cast(pd.DataFrame, adapter.videos[~adapter.videos["excluded"]])
    video = cast(dict[str, object], included.iloc[0].to_dict())
    frames = cast(pd.DataFrame, adapter.frames[adapter.frames["video_id"] == video["video_id"]])
    frames = frames.sort_values("frame_index").reset_index(drop=True)
    duplicated = frames.copy()
    duplicated.loc[2, "src_index"] = duplicated.loc[1, "src_index"]
    shifted = frames.assign(src_index=frames["src_index"] + 1)
    truncated = cast(pd.DataFrame, frames.iloc[:-1])
    decoded_off = {**video, "n_frames_decoded": int(cast(int, video["n_frames_counted"])) + 1}
    return [
        _check("grade íntegra do adapter não tem problema de alinhamento", grid_problems(video, frames) == []),
        _check(
            "grade com src_index repetido reprova como quadro duplicado",
            any("duplicado" in problem for problem in grid_problems(video, duplicated)),
        ),
        _check(
            "grade deslocada de um quadro reprova contra build_time_grid",
            any("build_time_grid" in problem for problem in grid_problems(video, shifted)),
        ),
        _check(
            "grade truncada e decodificador com contagem diferente do ffprobe reprovam",
            grid_problems(video, truncated) != [] and grid_problems(decoded_off, frames) != [],
        ),
    ]


def check_separation(setup: _Setup, tmp: Path) -> list[bool]:
    forbidden = [
        Path("data/features/le2i/pose"),
        Path("data/features/le2i"),
        Path("data/features"),
        Path("data/processed/gmdcsa24/pose"),
        Path("data/processed/le2i_cv"),
        Path("runs/local/gmdcsa24"),
        REPOSITORY_ROOT / "src/gatefall/gmdcsa24_pose",
    ]
    model = _FakeModel()
    refused = not _extract(setup, Path("data/features/le2i/pose"), model)
    return [
        _check(
            "saída recusada sobre as features Le2i, os dados processados, os runs, acima deles "
            "ou num caminho versionado; aceita num diretório próprio",
            all(pose_location_problems(path) for path in forbidden)
            and pose_location_problems(tmp / "pose") == [],
        ),
        _check("extract sobre a pose do Le2i é recusado sem rodar o modelo", refused and model.calls == 0),
    ]


def _with_manifest(source: Path, target: Path, mutate: Callable[[dict[str, object]], None]) -> Path:
    shutil.copytree(source, target)
    manifest_path = target / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def check_prerequisites(setup: _Setup, tmp: Path) -> list[bool]:
    """Cada pré-requisito violado recusa a extração antes de rodar o modelo ou gravar."""
    tampered_tables = tmp / "adapter_tampered"
    shutil.copytree(setup.adapter_dir, tampered_tables)
    windows = tampered_tables / WINDOW_FILES["T"]
    write_parquet(read_table(tampered_tables, WINDOW_FILES["T"]).iloc[1:], windows)
    dirty = _with_manifest(
        setup.adapter_dir, tmp / "adapter_dirty", lambda manifest: manifest.update(code={"commit": _COMMIT, "dirty": True})
    )
    other_weights = tmp / "other" / "yolo26n-pose.pt"
    other_weights.parent.mkdir()
    other_weights.write_bytes(b"outros pesos")
    excluded = sorted(_EXPECTED_EXCLUSIONS)[0]
    other_version = replace(
        setup.gate, pose_contract={**setup.gate.pose_contract, "ultralytics_version": "0.0.0"}
    )
    other_table = replace(
        setup.gate, table_sha256={**setup.gate.table_sha256, WINDOW_FILES["T"]: "0" * 64}
    )
    cases: dict[str, Callable[[Path, _FakeModel], bool]] = {
        "manifesto do adapter ausente": lambda out, model: _extract(
            setup, out, model, adapter_dir=tmp / "adapter_missing"
        ),
        "tabela do adapter adulterada": lambda out, model: _extract(
            setup, out, model, adapter_dir=tampered_tables
        ),
        "build do adapter com code.dirty": lambda out, model: _extract(
            setup, out, model, adapter_dir=dirty
        ),
        "HEAD diferente do commit do E0": lambda out, model: _extract(
            setup, out, model, checkout=("d" * 40, [])
        ),
        "checkout com arquivo não rastreado": lambda out, model: _extract(
            setup, out, model, checkout=(_COMMIT, ["?? notas.txt"])
        ),
        "vídeo excluído pedido": lambda out, model: _extract(
            setup, out, model, video_ids=[excluded]
        ),
        "vídeo desconhecido pedido": lambda out, model: _extract(
            setup, out, model, video_ids=["Subject_9/ADL/01"]
        ),
        "pesos com outro SHA-256": lambda out, model: _extract(
            setup, out, model, weights=other_weights
        ),
        "pesos ausentes": lambda out, model: _extract(
            setup, out, model, weights=tmp / "missing" / "yolo26n-pose.pt"
        ),
        "outra versão do Ultralytics no contrato": lambda out, model: _extract(
            setup, out, model, gate=other_version
        ),
        "hash publicado de T diferente": lambda out, model: _extract(
            setup, out, model, gate=other_table
        ),
    }
    checks: list[bool] = []
    for position, (name, run) in enumerate(cases.items()):
        model = _FakeModel()
        pose_dir = tmp / f"refused_{position}"
        checks.append(
            _check(
                f"pré-requisito: {name} recusa sem rodar o modelo nem gravar",
                not run(pose_dir, model) and model.calls == 0 and not pose_dir.exists(),
            )
        )
    return checks


def check_source_identity(setup: _Setup, tmp: Path) -> list[bool]:
    plans = _plans(setup)
    plan = plans[sorted(plans)[0]]
    identity = model_identity(setup.weights)
    provenance = {
        "adapter_manifest_sha256": "a" * 64,
        "code_commit": _COMMIT,
        "extraction_code_sha256": "e" * 64,
    }
    outcomes: list[bool] = []
    for name, changed in {
        "SHA-256 da fonte": replace(plan, source_sha256="0" * 64),
        "contagem do decodificador": replace(plan, n_frames_decoded=plan.n_frames_decoded + 1),
    }.items():
        pose_dir = tmp / f"identity_{name.split()[0]}"
        try:
            extract_video(
                changed,
                setup.root,
                pose_dir,
                cast(YOLO, _FakeModel()),
                expected_attrs(changed, identity, provenance),
                force=False,
            )
            outcomes.append(False)
        except PoseExtractError:
            outcomes.append(not pose_path(plan.video_id, pose_root=pose_dir).exists())
    return [
        _check(
            "fonte com outro SHA-256 ou decodificador com outra contagem falha sem gravar HDF5",
            all(outcomes),
        )
    ]


def check_extraction(setup: _Setup, tmp: Path) -> tuple[Path, list[bool]]:
    plans = _plans(setup)
    pose_dir = tmp / "pose"
    model = _FakeModel()
    first = sorted(plans)[0]
    partial_ok = _extract(setup, pose_dir, model, video_ids=[first])
    partial = _validate(setup, pose_dir)
    first_calls = model.calls
    partial_manifest = (pose_dir / POSE_MANIFEST_FILE).exists()
    full_ok = _extract(setup, pose_dir, model)
    clean = _validate(setup, pose_dir)
    grid_frames = {video_id: len(plan.src_indices) for video_id, plan in plans.items()}

    aligned = True
    for video_id, plan in plans.items():
        arrays = _read(pose_dir, video_id)
        found = arrays["person_found"]
        source = np.array(plan.src_indices)
        x = arrays["keypoints"][:, 0, 0]
        decoded = np.rint(x * 219 / (255 * _step(video_id))).astype(np.int64)
        aligned &= bool(np.array_equal(found, source != 0))
        aligned &= bool(np.array_equal(decoded[found], source[found]))

    first_ids = [int(_read(pose_dir, video_id)["track_id"][1]) for video_id in sorted(plans)]
    leak = _FakeModel()
    for gray in (0, 40, 0, 40):
        leak.track(np.full((48, 64, 3), gray, dtype=np.uint8))
    leaked_id = leak.tracker.active

    checks = [
        _check(
            "extração parcial (--video-id) grava o HDF5, não grava manifesto e não valida",
            partial_ok
            and not partial_manifest
            and {"manifesto", "cobertura"} <= _failed(partial)
            and first_calls == grid_frames[first],
        ),
        _check(
            "extração completa: só os vídeos incluídos, o HDF5 existente que confere é pulado, "
            "e a validação passa em todas as condições",
            full_ok
            and _failed(clean) == set()
            and model.calls == sum(grid_frames.values()),
        ),
        _check(
            "alinhamento ponta a ponta: o quadro de cada linha do HDF5 é o src_index da grade; "
            "o quadro preto não tem pessoa",
            aligned,
        ),
        _check(
            "isolamento do tracker: cada vídeo recomeça no track_id 1; sem reset, o tracker "
            "falso entregaria outro ID",
            first_ids == [1] * len(plans) and leaked_id == 2,
        ),
        _check(
            "nenhum arquivo para vídeo excluído",
            all(
                not pose_path(video_id, pose_root=pose_dir).exists()
                for video_id in _EXPECTED_EXCLUSIONS
            ),
        ),
    ]
    return pose_dir, checks


def check_existing_source(setup: _Setup, pose_dir: Path, tmp: Path) -> list[bool]:
    """Um HDF5 existente e válido não dispensa conferir o MP4 de origem."""
    plans = _plans(setup)
    plan = plans[sorted(plans)[0]]
    attrs = expected_attrs(
        plan,
        model_identity(setup.weights),
        {
            "adapter_manifest_sha256": load_adapter(setup.adapter_dir).manifest_sha256,
            "code_commit": _COMMIT,
            "extraction_code_sha256": extraction_code_sha256(),
        },
    )
    existing_dir = tmp / "pose_existing_source"
    shutil.copytree(pose_dir, existing_dir)
    output_path = pose_path(plan.video_id, pose_root=existing_dir)
    original_sha256 = sha256_of(output_path)

    def changed(root: Path) -> None:
        with (root / plan.relative_path).open("ab") as source:
            source.write(b"\0")

    def missing(root: Path) -> None:
        (root / plan.relative_path).unlink()

    outcomes: dict[str, bool] = {}
    sources: dict[str, Callable[[Path], None]] = {
        "unchanged": lambda root: None,
        "changed": changed,
        "missing": missing,
    }
    for name, mutate in sources.items():
        root = tmp / f"root_{name}"
        shutil.copytree(setup.root, root)
        mutate(root)
        model = _FakeModel()
        try:
            skipped = extract_video(plan, root, existing_dir, cast(YOLO, model), attrs, force=False)
            rejected = False
        except PoseExtractError:
            skipped, rejected = 0, True
        untouched = model.calls == 0 and sha256_of(output_path) == original_sha256
        outcomes[name] = untouched and (skipped is None if name == "unchanged" else rejected)
    return [
        _check(
            "HDF5 existente com a fonte inalterada é pulado sem rodar o modelo",
            outcomes["unchanged"],
        ),
        _check(
            "HDF5 existente e válido não dispensa a fonte: MP4 alterado ou ausente é recusado "
            "sem rodar o modelo nem sobrescrever o HDF5",
            outcomes["changed"] and outcomes["missing"],
        ),
    ]


def check_existing_outputs(setup: _Setup, pose_dir: Path, tmp: Path) -> list[bool]:
    plans = _plans(setup)
    total = sum(len(plan.src_indices) for plan in plans.values())
    repeat = _FakeModel()
    refused = not _extract(setup, pose_dir, repeat)

    forced_dir = tmp / "pose_forced"
    shutil.copytree(pose_dir, forced_dir)
    forced = _FakeModel()
    forced_ok = _extract(setup, forced_dir, forced, force=True)

    mismatch_dir = tmp / "pose_mismatch"
    shutil.copytree(pose_dir, mismatch_dir)
    (mismatch_dir / POSE_MANIFEST_FILE).unlink()
    target = sorted(plans)[1]
    with h5py.File(pose_path(target, pose_root=mismatch_dir), "a") as h5_file:
        h5_file.attrs["source_sha256"] = "0" * 64
    mismatch = _FakeModel()
    mismatch_refused = not _extract(setup, mismatch_dir, mismatch)
    repair = _FakeModel()
    repaired = _extract(setup, mismatch_dir, repair, video_ids=[target], force=True)
    return [
        _check("saída completa existente não é refeita sem --force", refused and repeat.calls == 0),
        _check(
            "--force refaz todos os vídeos e a saída volta a validar",
            forced_ok and forced.calls == total and _failed(_validate(setup, forced_dir)) == set(),
        ),
        _check(
            "HDF5 existente que não confere é recusado sem --force; --force num vídeo o refaz "
            "e o manifesto volta com a cobertura completa",
            mismatch_refused
            and repaired
            and repair.calls == len(plans[target].src_indices)
            and _failed(_validate(setup, mismatch_dir)) == set(),
        ),
    ]


def _variant(source: Path, target: Path, mutate: Callable[[Path], None]) -> Path:
    shutil.copytree(source, target)
    mutate(target)
    return target


def _edit_pose_manifest(update: Callable[[dict[str, object]], None]) -> Callable[[Path], None]:
    def mutate(pose_dir: Path) -> None:
        path = pose_dir / POSE_MANIFEST_FILE
        manifest = json.loads(path.read_text(encoding="utf-8"))
        update(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    return mutate


def check_incomplete_and_provenance(setup: _Setup, pose_dir: Path, tmp: Path) -> list[bool]:
    plans = _plans(setup)
    first, second = sorted(plans)[:2]
    excluded = sorted(_EXPECTED_EXCLUSIONS)[0]

    second_relative = pose_path(second, pose_root=Path()).as_posix()

    def first_path(root: Path) -> Path:
        return pose_path(first, pose_root=root)

    def write_bytes(path: Path, content: bytes) -> None:
        path.write_bytes(content)

    def resize_track_id(root: Path) -> None:
        with h5py.File(first_path(root), "a") as h5_file:
            del h5_file["track_id"]
            h5_file.create_dataset("track_id", data=np.zeros((1,), dtype=np.int32))

    def set_attr(name: str, value: object) -> Callable[[Path], None]:
        def mutate(root: Path) -> None:
            with h5py.File(pose_path(second, pose_root=root), "a") as h5_file:
                h5_file.attrs[name] = value

        return mutate

    def copy_excluded(root: Path) -> None:
        target = pose_path(excluded, pose_root=root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(first_path(root), target)

    def set_record(field: str, value: object) -> Callable[[dict[str, object]], None]:
        def update(manifest: dict[str, object]) -> None:
            cast(dict[str, dict[str, object]], manifest["videos"])[first][field] = value

        return update

    def set_section(name: str, field: str, value: object) -> Callable[[dict[str, object]], None]:
        def update(manifest: dict[str, object]) -> None:
            cast(dict[str, object], manifest[name])[field] = value

        return update

    def record(manifest: dict[str, object]) -> dict[str, object]:
        return cast(dict[str, dict[str, object]], manifest["videos"])[first]

    def summary(manifest: dict[str, object]) -> dict[str, object]:
        return cast(dict[str, object], manifest["summary"])

    def shift(select: _Select, field: str) -> Callable[[Path], None]:
        def update(manifest: dict[str, object]) -> None:
            select(manifest)[field] = int(cast(int, select(manifest)[field])) + 1

        return _edit_pose_manifest(update)

    def drop(select: _Select, field: str) -> Callable[[Path], None]:
        def update(manifest: dict[str, object]) -> None:
            del select(manifest)[field]

        return _edit_pose_manifest(update)

    def add(select: _Select, field: str) -> Callable[[Path], None]:
        return _edit_pose_manifest(lambda manifest: select(manifest).update({field: 0}))

    def drop_first_record(manifest: dict[str, object]) -> None:
        del cast(dict[str, object], manifest["videos"])[first]

    def add_excluded_record(manifest: dict[str, object]) -> None:
        cast(dict[str, object], manifest["videos"])[excluded] = dict(record(manifest))

    def drop_summary(manifest: dict[str, object]) -> None:
        del manifest["summary"]

    variants: dict[str, tuple[Callable[[Path], None], set[str]]] = {
        "HDF5 ausente": (lambda root: first_path(root).unlink(), {"cobertura"}),
        "temporário de escrita interrompida": (
            lambda root: write_bytes(first_path(root).with_suffix(".h5.123.tmp"), b"parcial"),
            {"cobertura"},
        ),
        "HDF5 truncado": (lambda root: write_bytes(first_path(root), b"\x89HDF"), {"HDF5"}),
        "dataset com K errado": (resize_track_id, {"HDF5"}),
        "manifesto ausente": (lambda root: (root / POSE_MANIFEST_FILE).unlink(), {"manifesto", "HDF5"}),
        "HDF5 de vídeo excluído": (copy_excluded, {"cobertura"}),
        "source_sha256 do HDF5": (set_attr("source_sha256", "0" * 64), {"HDF5"}),
        "commit gravado no HDF5": (set_attr("code_commit", "d" * 40), {"HDF5"}),
        "grid_sha256 do manifesto": (
            _edit_pose_manifest(set_record("grid_sha256", "0" * 64)),
            {"HDF5"},
        ),
        "path do vídeo no manifesto": (
            _edit_pose_manifest(set_record("path", second_relative)),
            {"HDF5"},
        ),
        "K do vídeo no manifesto": (shift(record, "K"), {"HDF5"}),
        "quadros com pessoa do vídeo no manifesto": (shift(record, "n_found"), {"HDF5"}),
        "campo ausente no registro do vídeo": (drop(record, "n_found"), {"HDF5"}),
        "campo extra no registro do vídeo": (add(record, "n_missing"), {"HDF5"}),
        "vídeo elegível ausente do manifesto": (
            _edit_pose_manifest(drop_first_record),
            {"cobertura"},
        ),
        "vídeo excluído no manifesto": (_edit_pose_manifest(add_excluded_record), {"cobertura"}),
        "total de vídeos do resumo": (shift(summary, "videos"), {"manifesto"}),
        "total de quadros do resumo": (shift(summary, "grid_frames"), {"manifesto"}),
        "total com pessoa do resumo": (shift(summary, "frames_with_person"), {"manifesto"}),
        "total ausente do resumo": (drop(summary, "frames_with_person"), {"manifesto"}),
        "total extra no resumo": (add(summary, "frames_without_person"), {"manifesto"}),
        "resumo ausente": (_edit_pose_manifest(drop_summary), {"manifesto"}),
        "SHA-256 dos pesos no manifesto": (
            _edit_pose_manifest(set_section("model", "weights_sha256", "0" * 64)),
            {"manifesto", "HDF5"},
        ),
        "commit da extração no manifesto": (
            _edit_pose_manifest(set_section("code", "commit", "d" * 40)),
            {"manifesto", "HDF5"},
        ),
    }
    checks: list[bool] = []
    for position, (name, (mutate, expected)) in enumerate(variants.items()):
        variant = _variant(pose_dir, tmp / f"variant_{position}", mutate)
        failed = _failed(_validate(setup, variant))
        checks.append(_check(f"validação reprova {name} ({', '.join(sorted(expected))})", failed == expected))

    other_adapter = _with_manifest(
        setup.adapter_dir,
        tmp / "adapter_rebuilt",
        lambda manifest: manifest.update(created_at="2026-10-20T00:00:00+00:00"),
    )
    rebuilt = _failed(_validate(setup, pose_dir, other_adapter))
    checks.append(
        _check(
            "pose extraída de outra saída do adapter (manifesto com outro SHA-256) reprova",
            rebuilt == {"manifesto", "HDF5"},
        )
    )
    checks.append(
        _check(
            "a saída original continua válida depois das variantes",
            _failed(_validate(setup, pose_dir)) == set(),
        )
    )
    return checks


def run_pose_extract_selftest() -> None:
    checks: list[bool] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        root = tmp / "GMDCSA24"
        for identifier, (fps, n_frames, step, variable_rate) in _VIDEOS.items():
            _encode(_video_path(root, identifier), fps, n_frames, step, variable_rate)
        adapter_dir = tmp / "adapter"
        with _quiet():
            sources = probe_sources(root, build_inventory(root).videos)
        tables = build_tables(sources, _omnifall(), _original_segments())
        write_outputs(
            adapter_dir,
            tables,
            {**_provenance(), "code": {"commit": _COMMIT, "dirty": False}},
            force=False,
        )
        weights = tmp / "weights" / "yolo26n-pose.pt"
        weights.parent.mkdir()
        weights.write_bytes(b"pesos sinteticos")
        files = json.loads((adapter_dir / MANIFEST_FILE).read_text(encoding="utf-8"))["files"]
        gate = ExtractionGate(
            pose_contract={**FROZEN_POSE_CONTRACT, "weights_sha256": sha256_of(weights)},
            expected_exclusions=_EXPECTED_EXCLUSIONS,
            expected_aggregates=_EXPECTED_AGGREGATES,
            table_sha256={
                name: files[name]["content_sha256"]
                for name in (WINDOW_FILES["T"], WINDOW_FILES["S"], CANDIDATES_FILE)
            },
        )
        setup = _Setup(root=root, adapter_dir=adapter_dir, weights=weights, gate=gate)

        checks.extend(check_decoder(root))
        checks.extend(check_grid_alignment(load_adapter(adapter_dir)))
        checks.extend(check_separation(setup, tmp))
        checks.extend(check_prerequisites(setup, tmp))
        checks.extend(check_source_identity(setup, tmp))
        pose_dir, extraction = check_extraction(setup, tmp)
        checks.extend(extraction)
        checks.extend(check_existing_source(setup, pose_dir, tmp))
        checks.extend(check_existing_outputs(setup, pose_dir, tmp))
        checks.extend(check_incomplete_and_provenance(setup, pose_dir, tmp))
    if not all(checks):
        print("\nselftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: todos os casos passaram")
