"""Selftest sintético da montagem de T/S e dos portões de pré-treino do piloto GMDCSA24.

Reaproveita o clone sintético do selftest do adapter e o modelo falso do
selftest da extração para gravar saídas de adapter e de pose válidas num
diretório temporário. O Le2i-CS também é sintético: quatro vídeos (dois de
treino, um de val e um de test) com grade, pose e padronização próprias, e um
controle treinado de verdade com a receita congelada do braço A. As features
passam por uma leitura espiã que registra cada vídeo lido, o que prova que
`test` nunca é lido. Não toca no dataset real, no Le2i real nem em `runs/`.
Requer FFmpeg.
"""

import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.frames import apply_frame_schema, write_frames
from gatefall.data.gmdcsa24.pose_extract import (
    FROZEN_POSE_CONTRACT,
    POSE_MANIFEST_FILE,
    ExtractionGate,
    run_extract,
)
from gatefall.data.gmdcsa24.selftests.adapter import (
    _EXPECTED_AGGREGATES,
    _EXPECTED_EXCLUSIONS,
    _VIDEOS,
    _check,
    _encode,
    _omnifall,
    _original_segments,
    _provenance,
    _video_path,
)
from gatefall.data.gmdcsa24.selftests.pose_extract import _COMMIT, _FakeModel, _quiet
from gatefall.data.gmdcsa24.source import build_inventory, sha256_of
from gatefall.data.gmdcsa24.weak_label import (
    CANDIDATES_FILE,
    LE2I_CS_TRAIN_WINDOWS,
    MANIFEST_FILE,
    WINDOW_FILES,
    build_tables,
    le2i_reference,
    probe_sources,
    read_table,
    write_outputs,
    write_parquet,
)
from gatefall.data.pose_dataset import PoseWindowDataset, WindowIndexDataset
from gatefall.data.windowing import window_frame_indices
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.multiseed_summary import _config_fingerprint
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, save_alarm_protocol
from gatefall.features.standardization import (
    StandardizationStats,
    compute_train_stats,
    load_stats,
    save_stats,
)
from gatefall.hashing import sha256_dataframe, sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM
from gatefall.pose.loading import pose_path
from gatefall.train.baseline_a.artifacts import validate_training_run
from gatefall.train.baseline_a.cli import _resolve_config
from gatefall.train.baseline_a.engine import run_training
from gatefall.train.gmdcsa24_weak_label.assembly import (
    Le2iSide,
    TreatmentSources,
    aligned_loader,
    arm_a_features,
    detection_coverage,
    gmdcsa24_windows,
    le2i_windows,
)
from gatefall.train.gmdcsa24_weak_label.cli import run_prepare, run_train
from gatefall.train.gmdcsa24_weak_label.gates import (
    PREPARED_MANIFEST_FILE,
    PREPARED_STATS_FILE,
    TREATMENT_RECORD_FILE,
    VALIDATION_SPLITS,
    Assessment,
    ControlRun,
    TrainingGate,
    TreatmentInputs,
    assess_data,
    assess_training,
    config_problems,
    location_problems,
    membership_problems,
)

WALK, FALL, FALLEN, SITTING, STAND_UP, STANDING, OTHER = (
    LE2I_LABEL_NAMES.index(name)
    for name in ("walk", "fall", "fallen", "sitting", "stand_up", "standing", "other")
)

# video_id: (split, env, subject, trechos de rótulo em quadros da grade).
_LE2I_VIDEOS: dict[str, tuple[str, str, int, list[tuple[int, int]]]] = {
    "coffee_room_01/video_1": ("train", "Coffee_room_01", 1, [(WALK, 15), (FALL, 5), (FALLEN, 20)]),
    "home_01/video_2": ("train", "Home_01", 2, [(SITTING, 18), (STAND_UP, 4), (STANDING, 14)]),
    "home_02/video_3": ("val", "Home_02", 3, [(WALK, 12), (FALL, 6), (FALLEN, 12)]),
    "office/video_4": ("test", "Office", 4, [(OTHER, 10), (FALL, 6), (FALLEN, 14)]),
}
_TEST_VIDEOS = {video for video, (split, *_) in _LE2I_VIDEOS.items() if split == "test"}
_SEED = 42


class _Spy:
    """Leitura de features do braço A que registra cada (vídeo, raiz de pose) lido."""

    def __init__(self) -> None:
        self.loaded: list[tuple[str, Path]] = []

    def __call__(self, video_id: str, pose_root: Path) -> np.ndarray:
        self.loaded.append((video_id, pose_root))
        return arm_a_features(video_id, pose_root)

    def videos(self) -> set[str]:
        return {video_id for video_id, _ in self.loaded}


@dataclass(frozen=True)
class _World:
    tmp: Path
    adapter_dir: Path
    pose_dir: Path
    le2i: Le2iSide
    gate: TrainingGate
    control: ControlRun
    spy: _Spy

    def inputs(self, treatment: str = "T", **changes: Any) -> TreatmentInputs:
        base = TreatmentInputs(
            treatment=treatment,
            adapter_dir=self.adapter_dir,
            pose_dir=self.pose_dir,
            training_dir=self.tmp / "training",
            le2i=self.le2i,
            checkout=("e" * 40, []),
            gate=self.gate,
            load_features=self.spy,
        )
        return replace(base, **changes)


def _failed(results: dict[str, list[str]]) -> set[str]:
    return {condition for condition, problems in results.items() if problems}


def _le2i_frames() -> pd.DataFrame:
    tables = []
    for video_id, (split, env, subject, spans) in _LE2I_VIDEOS.items():
        labels = np.concatenate([np.full(length, label) for label, length in spans])
        k = len(labels)
        tables.append(
            pd.DataFrame(
                {
                    "video_id": video_id,
                    "split": split,
                    "env": env,
                    "subject": subject,
                    "frame_index": np.arange(k),
                    "time_s": np.arange(k) / 10,
                    "src_index": np.arange(k) * 2,
                    "label": labels,
                    "gap_position": None,
                }
            )
        )
    return apply_frame_schema(pd.concat(tables, ignore_index=True))


def _write_le2i_pose(frames: pd.DataFrame, pose_root: Path) -> None:
    """Pose sintética determinística: caixa que desliza e um quadro sem pessoa por vídeo."""
    for position, (video_id, group) in enumerate(frames.groupby("video_id", sort=True)):
        k = len(group)
        rng = np.random.default_rng(position)
        t = np.arange(k, dtype=np.float32)
        bbox = np.stack([10 + t, 10 + 0.5 * t, 40 + t, 60 + 0.2 * t], axis=1).astype(np.float32)
        keypoints = np.zeros((k, 17, 3), dtype=np.float32)
        keypoints[:, :, 0] = bbox[:, None, 0] + rng.uniform(0, 30, (k, 17))
        keypoints[:, :, 1] = bbox[:, None, 1] + rng.uniform(0, 50, (k, 17))
        keypoints[:, :, 2] = rng.uniform(0.3, 1.0, (k, 17))
        person_found = np.ones(k, dtype=bool)
        person_found[3] = False
        keypoints[~person_found] = 0
        bbox[~person_found] = 0
        path = pose_path(str(video_id), pose_root=pose_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("keypoints", data=keypoints)
            h5_file.create_dataset("bbox", data=bbox)
            h5_file.create_dataset("person_found", data=person_found)
            h5_file.attrs["K"] = k
            h5_file.attrs["width"] = 160
            h5_file.attrs["height"] = 120


def _build(tmp: Path) -> _World:
    root = tmp / "GMDCSA24"
    for identifier, (fps, n_frames, step, variable_rate) in _VIDEOS.items():
        _encode(_video_path(root, identifier), fps, n_frames, step, variable_rate)
    with _quiet():
        sources = probe_sources(root, build_inventory(root).videos)
    tables = build_tables(sources, _omnifall(), _original_segments())

    frames = _le2i_frames()
    frames_path = tmp / "le2i" / "frames.parquet"
    write_frames(frames, frames_path)
    le2i_pose = tmp / "le2i_pose"
    _write_le2i_pose(frames, le2i_pose)
    # V2 do adapter fixa as 5 219 janelas publicadas do treino Le2i-CS; o portão
    # de treino confere a contagem sintética pelo `TrainingGate`.
    reference = replace(
        le2i_reference(frames, pd.DataFrame({"sha256": ["0" * 64]})),
        train_windows=LE2I_CS_TRAIN_WINDOWS,
    )

    def loader(video_id: str) -> np.ndarray:
        return arm_a_features(video_id, le2i_pose)

    stats_path = tmp / "stats" / "pose_le2i_cs.json"
    with _quiet():
        save_stats(
            compute_train_stats(
                PoseWindowDataset(frames, "train", TRAIN_STRIDE, loader), frames_path
            ),
            stats_path,
            force=False,
        )
    le2i = Le2iSide(frames, frames_path, le2i_pose, stats_path, reference)

    adapter_dir = tmp / "adapter"
    write_outputs(
        adapter_dir,
        tables,
        {
            **_provenance(),
            "le2i_cs": {"eval_index_sha256": dict(reference.eval_index_sha256)},
            "code": {"commit": _COMMIT, "dirty": False},
        },
        force=False,
    )
    weights = tmp / "weights" / "yolo26n-pose.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"pesos sinteticos")
    files = json.loads((adapter_dir / MANIFEST_FILE).read_text(encoding="utf-8"))["files"]
    extraction = ExtractionGate(
        pose_contract={**FROZEN_POSE_CONTRACT, "weights_sha256": sha256_of(weights)},
        expected_exclusions=_EXPECTED_EXCLUSIONS,
        expected_aggregates=_EXPECTED_AGGREGATES,
        table_sha256={
            name: files[name]["content_sha256"]
            for name in (WINDOW_FILES["T"], WINDOW_FILES["S"], CANDIDATES_FILE)
        },
    )
    pose_dir = tmp / "pose"
    with _quiet():
        extracted = run_extract(
            root,
            adapter_dir,
            pose_dir,
            [],
            False,
            le2i=reference,
            gate=extraction,
            checkout=(_COMMIT, []),
            weights_path=weights,
            model=cast(YOLO, _FakeModel()),
        )
    if not extracted:
        raise RuntimeError("selftest: a extração sintética de pose falhou")

    control_dir = tmp / "control" / "seed_42"
    config = _resolve_config(_SEED, stats_path, sha256_file(stats_path))
    with _quiet():
        run_training(
            input_dim=POSE_FEATURE_DIM,
            train_source=PoseWindowDataset(frames, "train", TRAIN_STRIDE, loader),
            val_source=PoseWindowDataset(frames, "val", EVAL_STRIDE, loader),
            test_source=PoseWindowDataset(frames, "test", EVAL_STRIDE, loader),
            stats=load_stats(stats_path),
            config=config,
            run_dir=control_dir,
            force=False,
            label_names=LE2I_LABEL_NAMES,
        )
        save_alarm_protocol(
            BASELINE_A_ALARM_PROTOCOL, control_dir / "alarm_protocol.yaml", force=False
        )

    funnel = cast(dict[str, int], _EXPECTED_AGGREGATES["funnel"])
    n_le2i = len(le2i_windows(frames, "train"))
    gate = TrainingGate(
        extraction=extraction,
        extraction_commit=_COMMIT,
        le2i_train_windows=n_le2i,
        training_windows={treatment: n_le2i + funnel[treatment] for treatment in WINDOW_FILES},
        eval_index_sha256=dict(reference.eval_index_sha256),
        control_checkpoint_sha256={_SEED: sha256_file(control_dir / "checkpoint.pt")},
        control_config_fingerprint=_config_fingerprint(config),
    )
    return _World(tmp, adapter_dir, pose_dir, le2i, gate, ControlRun(control_dir), _Spy())


def _assess_data(inputs: TreatmentInputs) -> Assessment:
    with _quiet():
        return assess_data(inputs)


def _assess_training(inputs: TreatmentInputs, control: ControlRun, seed: int = _SEED) -> Assessment:
    with _quiet():
        return assess_training(inputs, seed, control)


def _validates(run_dir: Path, splits: tuple[str, ...] | None = None) -> bool:
    try:
        if splits is None:
            validate_training_run(run_dir)
        else:
            validate_training_run(run_dir, splits=splits)
    except RuntimeError:
        return False
    return True


def _keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(frame["video_id"].astype(str), frame["k_end"].astype(int)))


def check_membership(world: _World) -> list[bool]:
    gate = world.gate
    t = _assess_data(world.inputs("T"))
    s = _assess_data(world.inputs("S"))
    t_sources = t.sources
    s_sources = s.sources
    if t_sources is None or s_sources is None:
        return [_check("portões de dados de T e S passam sobre a saída íntegra", False)]
    tables = {name: gmdcsa24_windows(world.adapter_dir, name) for name in WINDOW_FILES}
    included = set(
        read_table(world.adapter_dir, "videos.parquet").query("~excluded")["video_id"].astype(str)
    )
    n_le2i = gate.le2i_train_windows
    t_windows = t_sources.train.windows
    off_total = replace(
        gate, training_windows={**gate.training_windows, "T": gate.training_windows["T"] + 1}
    )
    off_le2i = replace(gate, le2i_train_windows=n_le2i + 1)
    return [
        _check("portões de dados de T e S passam sobre a saída íntegra", t.passed and s.passed),
        _check(
            "treino de T = treino Le2i-CS inalterado seguido de windows_t.parquet, sem outra janela",
            t_windows.iloc[:n_le2i]
            .reset_index(drop=True)
            .equals(t_sources.le2i_train.reset_index(drop=True))
            and _keys(t_windows.iloc[n_le2i:]) == _keys(tables["T"])
            and len(t_windows) == gate.training_windows["T"]
            and len(s_sources.train) == gate.training_windows["S"],
        ),
        _check(
            "S está contido em T, e T e S só têm vídeos incluídos pelo adapter",
            _keys(tables["S"]) <= _keys(tables["T"])
            and set(tables["T"]["video_id"].astype(str)) <= included
            and not (set(tables["T"]["video_id"].astype(str)) & set(_EXPECTED_EXCLUSIONS)),
        ),
        _check(
            "total do treino aumentado ou do treino Le2i-CS diferente do fixado reprova tratamento",
            _failed(_assess_data(world.inputs("T", gate=off_total)).results) == {"tratamento"}
            and "tratamento" in _failed(_assess_data(world.inputs("T", gate=off_le2i)).results),
        ),
    ]


def check_alignment(world: _World) -> list[bool]:
    inputs = world.inputs("T")
    assessment = _assess_data(inputs)
    sources = assessment.sources
    if sources is None:
        return [_check("alinhamento: fontes montadas", False)]
    n_le2i = len(sources.le2i_train)
    row = sources.train.windows.iloc[n_le2i]
    video_id, k_end, n_frames = str(row["video_id"]), int(row["k_end"]), int(row["n_frames"])
    world.spy.loaded.clear()
    window, label, _ = sources.train[n_le2i]
    expected = arm_a_features(video_id, world.pose_dir)[window_frame_indices(k_end, n_frames)]
    le2i_video = str(sources.train.windows.iloc[0]["video_id"])
    sources.train[0]
    roots = dict(world.spy.loaded)
    shifted = sources.gmdcsa24.assign(n_frames=sources.gmdcsa24["n_frames"] + 1)
    try:
        aligned_loader(shifted, inputs.roots, arm_a_features)(video_id)
        loader_refused = False
    except ValueError:
        loader_refused = True
    return [
        _check(
            "janela do GMDCSA24: features do braço A da pose do GMDCSA24 nos quadros da janela",
            np.array_equal(window, expected.astype(np.float32))
            and label == int(row["label"])
            and roots == {video_id: world.pose_dir, le2i_video: world.le2i.pose_root},
        ),
        _check(
            "K do índice diferente do K da pose reprova a leitura e o tratamento",
            loader_refused
            and any(
                "K da pose" in problem
                for problem in membership_problems(inputs, replace(sources, gmdcsa24=shifted))
            ),
        ),
    ]


def _expected_weights(labels: np.ndarray) -> list[float]:
    counts = np.bincount(labels, minlength=len(LE2I_LABEL_NAMES))
    return [float(np.float32(1.0 / count)) if count else 0.0 for count in counts]


@dataclass(frozen=True)
class _Prepared:
    manifest: dict[str, Any]
    stats: StandardizationStats
    sources: TreatmentSources


def _prepare(world: _World, treatment: str) -> tuple[bool, _Prepared | None]:
    inputs = world.inputs(treatment)
    with _quiet():
        ok = run_prepare(inputs, force=False)
    sources = _assess_data(inputs).sources
    if not ok or sources is None:
        return False, None
    manifest = json.loads(
        (inputs.prepared_dir / PREPARED_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    return True, _Prepared(manifest, load_stats(inputs.prepared_dir / PREPARED_STATS_FILE), sources)


def _frames_of(source: PoseWindowDataset | WindowIndexDataset) -> np.ndarray:
    return np.concatenate([source[i][0] for i in range(len(source))]).astype(np.float64)


def check_statistics(world: _World) -> list[bool]:
    control_stats_sha = sha256_file(world.le2i.stats_path)
    t_ok, t = _prepare(world, "T")
    s_ok, s = _prepare(world, "S")
    if t is None or s is None:
        return [_check("prepare de T e S grava padronização e manifesto", False)]
    train = t.sources.train
    stack = _frames_of(train)
    with_val = np.concatenate([stack, _frames_of(t.sources.val)])
    kept = ~np.asarray(t.stats.excluded_mask) & ~np.asarray(t.stats.guarded_mask)
    mean = np.asarray(t.stats.mean)[kept]
    weights = cast(list[float], t.manifest["class_weights"])
    with _quiet():
        refused = not run_prepare(world.inputs("T"), force=False)
    return [
        _check(
            "prepare de T e S grava padronização e manifesto em diretórios próprios", t_ok and s_ok
        ),
        _check(
            "padronização de T = média e desvio de todos os quadros das janelas do treino "
            "aumentado, e só delas (com val, a média muda)",
            np.allclose(mean, stack.mean(axis=0)[kept])
            and np.allclose(np.asarray(t.stats.std)[kept], stack.std(axis=0)[kept])
            and not np.allclose(mean, with_val.mean(axis=0)[kept])
            and t.stats.window_count == len(train)
            and t.stats.frames_hash == sha256_dataframe(train.windows),
        ),
        _check(
            "pesos da perda = 1/contagem sobre o treino aumentado, zero para classe ausente",
            weights == _expected_weights(train.windows["label"].to_numpy(np.int64))
            and weights[LE2I_LABEL_NAMES.index("lie_down")] == 0.0
            and weights != _expected_weights(t.sources.le2i_train["label"].to_numpy(np.int64)),
        ),
        _check(
            "T e S têm padronização e pesos próprios; o JSON do controle não muda",
            t.stats != s.stats
            and t.manifest["class_weights"] != s.manifest["class_weights"]
            and sha256_file(world.le2i.stats_path) == control_stats_sha,
        ),
        _check("prepare não sobrescreve uma preparação existente sem --force", refused),
    ]


def _copy_with(source: Path, target: Path, mutate: Callable[[Path], None]) -> Path:
    shutil.copytree(source, target)
    mutate(target)
    return target


def _edit_json(name: str, update: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def mutate(directory: Path) -> None:
        path = directory / name
        data = json.loads(path.read_text(encoding="utf-8"))
        update(data)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return mutate


def check_split_isolation(world: _World) -> list[bool]:
    sources = _assess_data(world.inputs("T")).sources
    if sources is None:
        return [_check("isolamento: fontes montadas", False)]
    train = sources.train.windows
    frames = world.le2i.frames

    def le2i_variant(changed: pd.DataFrame) -> Le2iSide:
        reference = replace(
            le2i_reference(changed, pd.DataFrame({"sha256": ["0" * 64]})),
            train_windows=LE2I_CS_TRAIN_WINDOWS,
        )
        return replace(world.le2i, frames=changed, reference=reference)

    val_mask = frames["split"] == "val"
    relabeled = frames.copy()
    relabeled.loc[val_mask & (relabeled["frame_index"] == 20), "label"] = WALK
    leaked_id = frames.copy()
    leaked_id.loc[val_mask, "video_id"] = sorted(
        set(train.iloc[len(sources.le2i_train) :]["video_id"].astype(str))
    )[0]
    stats_dir = Path("src/gatefall/features/stats/gmdcsa24_training")
    return [
        _check(
            "treino aumentado só com split train e sem vídeo de val/test do Le2i-CS",
            set(train["split"].astype(str)) == {"train"}
            and not (set(train["video_id"].astype(str)) & world.le2i.reference.eval_video_ids),
        ),
        _check(
            "rótulo de val alterado (hash do índice de val diferente do publicado) reprova V2",
            "V2" in _failed(_assess_data(world.inputs("T", le2i=le2i_variant(relabeled))).results),
        ),
        _check(
            "video_id do GMDCSA24 na val do Le2i-CS reprova V2",
            "V2" in _failed(_assess_data(world.inputs("T", le2i=le2i_variant(leaked_id))).results),
        ),
        _check(
            "JSON do tratamento sobre as estatísticas canônicas e run sobre o controle Le2i são recusados",
            "V2" in _failed(_assess_data(world.inputs("T", training_dir=stats_dir)).results)
            and location_problems(Path("runs/local/le2i/baseline_a")) != []
            and location_problems(Path("runs/reference/le2i/baseline_a")) != []
            and location_problems(world.tmp / "runs" / "t" / "seed_42") == [],
        ),
    ]


def check_provenance(world: _World) -> list[bool]:
    tmp = world.tmp / "provenance"
    tmp.mkdir()

    def set_code(commit: str, dirty: bool) -> Callable[[dict[str, object]], None]:
        return lambda manifest: manifest.update(code={"commit": commit, "dirty": dirty})

    def shift_summary(manifest: dict[str, object]) -> None:
        summary = cast(dict[str, int], manifest["summary"])
        summary["videos"] += 1

    def truncate_windows(directory: Path) -> None:
        write_parquet(
            read_table(directory, WINDOW_FILES["T"]).iloc[1:], directory / WINDOW_FILES["T"]
        )

    variants = {
        "adapter gerado em outro commit": {
            "adapter_dir": _copy_with(
                world.adapter_dir,
                tmp / "adapter_commit",
                _edit_json(MANIFEST_FILE, set_code("d" * 40, False)),
            )
        },
        "adapter com code.dirty": {
            "adapter_dir": _copy_with(
                world.adapter_dir,
                tmp / "adapter_dirty",
                _edit_json(MANIFEST_FILE, set_code(_COMMIT, True)),
            )
        },
        "tabela de T adulterada": {
            "adapter_dir": _copy_with(world.adapter_dir, tmp / "adapter_windows", truncate_windows)
        },
        "manifesto da pose adulterado": {
            "pose_dir": _copy_with(
                world.pose_dir, tmp / "pose_summary", _edit_json(POSE_MANIFEST_FILE, shift_summary)
            )
        },
        "commit da extração validada diferente": {
            "gate": replace(world.gate, extraction_commit="d" * 40)
        },
    }
    checks = [
        _check(
            f"proveniência reprova {name}",
            "proveniência" in _failed(_assess_data(world.inputs("T", **changes)).results),
        )
        for name, changes in variants.items()
    ]
    clean_other_head = _assess_data(world.inputs("T", checkout=("f" * 40, [])))
    dirty = _assess_data(world.inputs("T", checkout=("f" * 40, ["?? notas.txt"])))
    checks.extend(
        [
            _check(
                "commit de treino diferente do commit da extração, em checkout limpo, passa sem "
                "refazer adapter nem pose",
                clean_other_head.passed,
            ),
            _check(
                "checkout com arquivo não rastreado reprova", _failed(dirty.results) == {"checkout"}
            ),
        ]
    )
    with _quiet():
        refused = not run_prepare(
            world.inputs("T", checkout=("f" * 40, ["?? notas.txt"]), training_dir=tmp / "training"),
            force=False,
        )
    checks.append(
        _check(
            "prepare recusa portão reprovado sem gravar nada",
            refused and not (tmp / "training").exists(),
        )
    )

    prepared = world.inputs("T").prepared_dir

    def other_adapter(manifest: dict[str, object]) -> None:
        cast(dict[str, object], manifest["adapter"])["manifest_sha256"] = "0" * 64

    def tamper_stats(directory: Path) -> None:
        path = directory / PREPARED_STATS_FILE
        stats = json.loads(path.read_text(encoding="utf-8"))
        stats["mean"][0] += 1.0
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    def resign_stats(directory: Path) -> None:
        tamper_stats(directory)
        _edit_json(
            PREPARED_MANIFEST_FILE,
            lambda manifest: cast(dict[str, object], manifest["standardization"]).update(
                sha256=sha256_file(directory / PREPARED_STATS_FILE)
            ),
        )(directory)

    for name, mutate in {
        "manifesto preparado de outra saída do adapter": _edit_json(
            PREPARED_MANIFEST_FILE, other_adapter
        ),
        "JSON de padronização adulterado": tamper_stats,
        "JSON de padronização adulterado com SHA-256 refeito no manifesto": resign_stats,
    }.items():
        training_dir = tmp / f"training_{len(checks)}"
        _copy_with(prepared, training_dir / "t", mutate)
        results = _assess_training(
            world.inputs("T", training_dir=training_dir), control=world.control
        ).results
        checks.append(
            _check(
                f"padronização e pesos reprova {name}", "padronização e pesos" in _failed(results)
            )
        )
    return checks


def check_control(world: _World) -> list[bool]:
    tmp = world.tmp / "controls"
    tmp.mkdir()
    inputs = world.inputs("T")
    clean = _assess_training(inputs, control=world.control)
    other_sha = replace(world.gate, control_checkpoint_sha256={_SEED: "0" * 64})
    other_fingerprint = replace(world.gate, control_config_fingerprint="0" * 64)

    def epochs(directory: Path) -> None:
        path = directory / "config.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["epochs"] = 29
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def alarm(directory: Path) -> None:
        save_alarm_protocol(
            replace(BASELINE_A_ALARM_PROTOCOL, trigger_consecutive=2),
            directory / "alarm_protocol.yaml",
            force=True,
        )

    with _quiet():
        epochs_control = ControlRun(_copy_with(world.control.run_dir, tmp / "epochs", epochs))
        alarm_control = ControlRun(_copy_with(world.control.run_dir, tmp / "alarm", alarm))
    treatment = clean.config
    control_config = _resolve_config(
        _SEED, world.le2i.stats_path, sha256_file(world.le2i.stats_path)
    )
    return [
        _check(
            "controle publicado da mesma seed: V3 e V4 passam e todos os portões passam",
            clean.passed,
        ),
        _check(
            "checkpoint do controle diferente do publicado reprova V3; retreinado nesta máquina passa",
            _failed(
                _assess_training(world.inputs("T", gate=other_sha), control=world.control).results
            )
            == {"V3"}
            and _assess_training(
                world.inputs("T", gate=other_sha),
                control=ControlRun(world.control.run_dir, retrained=True),
            ).passed,
        ),
        _check(
            "controle de outra seed ou com configuração fora da receita reprova V3",
            "V3" in _failed(_assess_training(inputs, seed=43, control=world.control).results)
            and "V3" in _failed(_assess_training(inputs, control=epochs_control).results),
        ),
        _check(
            "fingerprint do controle diferente do publicado reprova V4",
            _failed(
                _assess_training(
                    world.inputs("T", gate=other_fingerprint), control=world.control
                ).results
            )
            == {"V4"},
        ),
        _check(
            "alarme do controle recalibrado reprova V2",
            _failed(_assess_training(inputs, control=alarm_control).results) == {"V2"},
        ),
        _check(
            "V4: o tratamento só difere do controle no JSON de padronização; outra diferença reprova",
            treatment is not None
            and config_problems(treatment, control_config) == []
            and len(config_problems(replace(treatment, lr=treatment.lr * 2), control_config)) == 1
            and treatment.standardization_stats_path
            == str(inputs.prepared_dir / PREPARED_STATS_FILE),
        ),
    ]


def check_training(world: _World) -> list[bool]:
    world.spy.loaded.clear()
    inputs = world.inputs("T")
    run_dir = world.tmp / "runs" / "t" / "seed_42"
    with _quiet():
        trained = run_train(inputs, _SEED, world.control, run_dir, force=False)
    loaded = world.spy.videos()
    if not trained:
        return [
            _check("treino de T com a receita congelada termina e grava o run e o registro", False)
        ]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / TREATMENT_RECORD_FILE).read_text(encoding="utf-8"))
    control_metrics = json.loads(
        (world.control.run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    checkpoint = sha256_file(run_dir / "checkpoint.pt")

    def add_test(metrics: dict[str, Any]) -> None:
        metrics["final"]["test"] = metrics["final"]["val"]

    with_test = _copy_with(
        run_dir, world.tmp / "runs" / "with_test", _edit_json("metrics.json", add_test)
    )

    world.spy.loaded.clear()
    with _quiet():
        rerun = run_train(inputs, _SEED, world.control, run_dir, force=False)
    return [
        _check("treino de T com a receita congelada termina e grava o run e o registro", trained),
        _check(
            "nenhuma feature de test do Le2i-CS é lida em preflight e treino; val e treino são",
            not (loaded & _TEST_VIDEOS) and {"home_02/video_3", "coffee_room_01/video_1"} <= loaded,
        ),
        _check(
            "o run de T avalia só train e val; o controle canônico continua avaliando test",
            set(metrics["final"]) == set(VALIDATION_SPLITS)
            and set(control_metrics["final"]) == {"train", "val", "test"}
            and record.get("evaluated_splits") == list(VALIDATION_SPLITS),
        ),
        _check(
            "o run de T valida só sem test: metrics com test reprova, e o validador canônico o recusa",
            _validates(run_dir, VALIDATION_SPLITS)
            and not _validates(with_test, VALIDATION_SPLITS)
            and not _validates(run_dir),
        ),
        _check(
            "o registro liga o run à preparação, ao índice de treino e ao controle pareado",
            record.get("treatment") == "T"
            and record.get("seed") == _SEED
            and record.get("prepared_manifest_sha256")
            == sha256_file(inputs.prepared_dir / PREPARED_MANIFEST_FILE)
            and record["control"]["checkpoint_sha256"]
            == world.gate.control_checkpoint_sha256[_SEED],
        ),
        _check(
            "run completo não é refeito sem --force",
            rerun and sha256_file(run_dir / "checkpoint.pt") == checkpoint,
        ),
    ]


def check_coverage(world: _World) -> list[bool]:
    tables = {name: gmdcsa24_windows(world.adapter_dir, name) for name in WINDOW_FILES}
    hashes = {name: sha256_dataframe(frame) for name, frame in tables.items()}
    records = json.loads((world.pose_dir / POSE_MANIFEST_FILE).read_text(encoding="utf-8"))[
        "videos"
    ]
    videos, summary = detection_coverage(records, tables, world.pose_dir)
    expected = {
        video: (int(record["K"]) - 1) / int(record["K"]) for video, record in records.items()
    }
    return [
        _check(
            "cobertura por vídeo: o quadro preto de cada vídeo sintético fica sem pessoa",
            dict(zip(videos["video_id"], videos["coverage"])) == expected
            and videos["coverage"].is_monotonic_increasing,
        ),
        _check(
            "cobertura por janela soma as janelas de T e S, e o relatório não muda T nem S",
            all(
                summary[name]["windows"]
                == cast(int, summary[name]["windows_full_detection"])
                + cast(int, summary[name]["windows_partial_detection"])
                + cast(int, summary[name]["windows_without_detection"])
                == len(tables[name])
                for name in tables
            )
            and {
                name: sha256_dataframe(gmdcsa24_windows(world.adapter_dir, name)) for name in tables
            }
            == hashes,
        ),
    ]


def run_treatment_selftest() -> None:
    checks: list[bool] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        world = _build(Path(tmp_dir))
        checks.extend(check_membership(world))
        checks.extend(check_alignment(world))
        checks.extend(check_statistics(world))
        checks.extend(check_split_isolation(world))
        checks.extend(check_provenance(world))
        checks.extend(check_control(world))
        checks.extend(check_training(world))
        checks.extend(check_coverage(world))
    if not all(checks):
        print("\nselftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: todos os casos passaram")
