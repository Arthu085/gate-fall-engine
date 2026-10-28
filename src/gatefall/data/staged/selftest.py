"""Contratos sintéticos dos datasets encenados: partição, alinhamento temporal e rótulos."""

import json
import tempfile
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from PIL import Image

from gatefall.config import IGNORE_LABEL, TARGET_FPS
from gatefall.data.omnifall.provenance import write_annotation_provenance
from gatefall.data.staged.labels import (
    LABELS_FILENAME,
    PROVENANCE_FILENAME,
    REFERENCE_SPLIT_FILES,
    UnverifiableLabelsError,
    map_to_gatefall,
)
from gatefall.data.staged.specs import (
    GMDCSA24_SPEC,
    GMDCSA24_SPLIT,
    UPFALL_SPEC,
    UPFALL_SPLIT,
    StagedDatasetSpec,
    SubjectSplit,
)

ORIGIN = datetime(2018, 7, 4, 12, 4, 17, 738369)


def _rows(frame: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    return cast(pd.DataFrame, frame[frame[column] == value])


def _expect(error: type[BaseException], action, message: str) -> None:
    try:
        action()
    except error:
        return
    raise AssertionError(message)


def _frame_name(instant: datetime) -> str:
    return instant.strftime("%Y-%m-%dT%H_%M_%S.%f") + ".png"


def _write_sequence(directory: Path, n_frames: int, tag: int, drop: range | None = None) -> None:
    directory.mkdir(parents=True)
    rng = np.random.default_rng(len(str(directory)))
    elapsed = 0.0
    for index in range(n_frames):
        elapsed += 1 / 18 + float(rng.uniform(-0.004, 0.004)) if index else 0.0
        if drop is not None and index in drop:
            continue
        pixel = np.full((6, 8, 3), index % 256, dtype=np.uint8)
        pixel[-1, -1] = tag
        Image.fromarray(pixel).save(directory / _frame_name(ORIGIN + timedelta(seconds=elapsed)))


def _world(root: Path) -> StagedDatasetSpec:
    spec = replace(
        UPFALL_SPEC,
        raw_dir=root / "raw",
        processed_dir=root / "processed",
        features_dir=root / "features",
        labels_dir=root / "labels",
        split=SubjectSplit("selftest-v1", train=(1,), val=(2,), test=(3,)),
    )
    rows = []
    for subject in (1, 2, 3):
        for camera in (1, 2):
            video_id = f"Subject{subject}/Activity1/Trial1/Subject{subject}Activity1Trial1Camera{camera}"
            drop = range(20, 30) if (subject, camera) == (1, 1) else None
            _write_sequence(spec.raw_dir / video_id, 70, 10 * subject + camera, drop)
            rows += [
                (video_id, 0, 0.0, 1.0, subject, camera),
                (video_id, 1, 1.0, 1.8, subject, camera),
                (video_id, 2, 1.8, 3.0, subject, camera),
                (video_id, 12, 3.0, 3.6, subject, camera),
            ]
    spec.labels_dir.mkdir(parents=True)
    labels = pd.DataFrame(rows, columns=["path", "label", "start", "end", "subject", "cam"])
    labels["dataset"] = "up_fall"
    labels.to_csv(spec.labels_dir / LABELS_FILENAME, index=False)
    for split, filename in REFERENCE_SPLIT_FILES.items():
        subjects = {"train": (1, 2), "validation": (2,), "test": (3,)}[split]
        labels[labels["subject"].isin(subjects)].to_csv(spec.labels_dir / filename, index=False)
    write_annotation_provenance(
        provenance_path=spec.labels_dir / PROVENANCE_FILENAME,
        files_directory=spec.labels_dir,
        filenames=[LABELS_FILENAME, *REFERENCE_SPLIT_FILES.values()],
        dataset_repo_id="simplexsigil2/omnifall",
        revision="68e5cee56a4bad38cca4aea791cac248f96e79a0",
        configs=["labels", "up_fall-cs"],
    )
    return spec


def test_frozen_splits() -> None:
    assert sorted(UPFALL_SPLIT.train + UPFALL_SPLIT.val + UPFALL_SPLIT.test) == list(range(1, 18))
    assert set(UPFALL_SPLIT.test) == {2, 4, 15, 16} and UPFALL_SPLIT.val == (11,)
    assert GMDCSA24_SPLIT.train == () and GMDCSA24_SPLIT.val == ()
    _expect(ValueError, lambda: SubjectSplit("bad", (1,), (1,), ()), "sujeito repetido aceito")
    _expect(ValueError, lambda: UPFALL_SPLIT.split_of(18), "sujeito desconhecido aceito")
    assert GMDCSA24_SPEC.source_relative_path("Subject_2/Fall/07") == Path("Subject 2/Fall/07.mp4")
    assert GMDCSA24_SPEC.parse_video_id("Subject_2/Fall/07")["clip"] == "07"
    _expect(ValueError, lambda: UPFALL_SPEC.parse_video_id("Subject1/Activity1/Trial1/Subject2Activity1Trial1Camera1"),
            "identidade inconsistente aceita")
    print("[PASS] partições congeladas disjuntas por sujeito e teste externo reservado")


def test_timestamps_and_staging(root: Path) -> None:
    from gatefall.data.staged.sequences import SequenceError, parse_frame_timestamp, stage_upfall_archive

    assert parse_frame_timestamp("2018-07-04T12_04_17.738369") == ORIGIN
    _expect(SequenceError, lambda: parse_frame_timestamp("frame_0001"), "nome sem tempo aceito")
    archive = root / "Subject9Activity2Trial3Camera1.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../../escape/" + _frame_name(ORIGIN), b"x")
        handle.writestr("nested/" + _frame_name(ORIGIN + timedelta(seconds=.1)), b"y")
        handle.writestr("notes.txt", b"z")
    target = stage_upfall_archive(archive, root / "staged")
    assert sorted(p.name for p in target.iterdir()) == sorted(
        [_frame_name(ORIGIN), _frame_name(ORIGIN + timedelta(seconds=.1))])
    assert not (root / "escape").exists()
    _expect(FileExistsError, lambda: stage_upfall_archive(archive, root / "staged"), "sobrescrita aceita")
    print("[PASS] carimbos de tempo verificáveis e extração sem escapar do destino")


def _assert_latest_at_or_before(times: np.ndarray, src: np.ndarray, source_times: np.ndarray) -> None:
    """Cada passo usa o último quadro de origem com tempo <= t (nunca um posterior)."""
    from gatefall.data.staged.timeline import CAUSAL_TOLERANCE_S

    chosen = source_times[src]
    assert np.all(chosen <= times + CAUSAL_TOLERANCE_S), "quadro de origem posterior ao instante de predição"
    following = np.minimum(src + 1, len(source_times) - 1)
    last = src == len(source_times) - 1
    assert np.all(last | (source_times[following] > times + CAUSAL_TOLERANCE_S)), "existe quadro mais recente <= t"


def test_causal_alignment() -> None:
    from gatefall.data.resampling import build_time_grid
    from gatefall.data.staged.timeline import (
        FutureObservationError,
        VideoGrid,
        build_video_frames,
        grid_from_timestamps,
        grid_from_video,
    )

    # Quadro 1e-6 s depois de t=0,1 não pode ser usado; empate exato em t=0,2 é passado.
    stamps = np.array([0.0, 0.05, 0.1000001, 0.18, 0.2, 0.55, 0.61])
    grid = grid_from_timestamps(stamps, 0.7)
    assert grid.src_indices.tolist() == [0, 1, 4, 4, 4, 4, 5]
    assert np.allclose(grid.lag_s, [0, .05, 0, .1, .2, .3, .05])
    rng = np.random.default_rng(2026)
    for _ in range(200):
        steps = rng.uniform(0.01, 0.2, rng.integers(5, 80))
        steps[rng.random(len(steps)) < .1] += rng.uniform(.1, .6)
        source = np.concatenate([[0.0], np.cumsum(steps)])
        snapped = rng.random(len(source)) < .3
        source[snapped] = np.round(source[snapped] * TARGET_FPS) / TARGET_FPS
        source = np.unique(source)
        if len(source) < 2:
            continue
        grid = grid_from_timestamps(source, float(source[-1] + .05))
        _assert_latest_at_or_before(grid.times_s, grid.src_indices, source)
    for fps in (7.5, 10.0, 12.5, 15.0, 23.976, 25.0, 29.97, 30.0, 60.0):
        for n_frames in (1, 7, 31, 250):
            grid = grid_from_video(n_frames, fps)
            _assert_latest_at_or_before(grid.times_s, grid.src_indices, np.arange(n_frames) / fps)
    # Contrato histórico do Le2i preservado: quadro mais próximo, que pode estar à frente de t.
    times, nearest = build_time_grid(250, 25.0, TARGET_FPS)
    lookahead = nearest / 25.0 - times
    assert lookahead.max() > 0 and lookahead.max() <= .5 / 25.0 + 1e-12
    future = VideoGrid(np.array([0.0, 0.1]), np.array([0, 1]), np.array([0.0, 0.12]))
    _expect(FutureObservationError, lambda: build_video_frames(
        video_id="x", split="test", env="e", subject=1, grid=future, duration_s=.2,
        segments=pd.DataFrame({"start": [0.0], "end": [.2], "label": [0]})), "quadro futuro aceito")
    _expect(ValueError, lambda: grid_from_timestamps(np.array([.1, .2]), 1.0), "origem diferente de 0 aceita")
    print("[PASS] alinhamento causal: último quadro <= t em sequências irregulares, lacunas, empates e vídeo; "
          "Le2i mantém o quadro mais próximo")


def test_build_and_audit(spec: StagedDatasetSpec) -> None:
    from gatefall.data.frame_source import decode_dataset_frames
    from gatefall.data.staged.audit import run_audit
    from gatefall.data.staged.build import build_manifest, build_timegrid, staged_paths
    from gatefall.datasets.staged import StagedImageSequenceAdapter

    build_manifest(spec)
    build_timegrid(spec)
    paths = staged_paths(spec)
    manifest = pd.read_parquet(paths.manifest)
    identities = pd.read_parquet(paths.identities)
    frames = pd.read_parquet(paths.frames)
    assert dict(zip(manifest["subject"], manifest["split"])) == {1: "train", 2: "val", 3: "test"}
    assert identities.groupby("recording_id")["split"].nunique().max() == 1
    assert (identities.groupby("recording_id").size() == 2).all()

    video = "Subject2/Activity1/Trial1/Subject2Activity1Trial1Camera1"
    part = _rows(frames, "video_id", video).sort_values("frame_index")
    assert np.allclose(part["time_s"], np.arange(len(part)) / TARGET_FPS)
    labels = part["label"].to_numpy()
    assert labels[0] == 0 and labels[12] == 1 and labels[25] == 2
    assert np.all(labels[part["time_s"].to_numpy() >= 3.0] == IGNORE_LABEL), "classe 12 deve virar IGNORE"
    index = pd.read_parquet(paths.sequence_index)
    for video_id, rows in frames.groupby("video_id"):
        stamps = _rows(index, "video_id", video_id).sort_values("src_index")["timestamp_s"].to_numpy()
        _assert_latest_at_or_before(rows["time_s"].to_numpy(), rows["src_index"].to_numpy(), stamps)

    gapped_id = "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1"
    source_gap = _rows(_rows(frames, "video_id", gapped_id), "gap_position", "source_gap")
    assert len(source_gap) > 0 and (source_gap["label"] != IGNORE_LABEL).all()
    gap_index = _rows(index, "video_id", gapped_id).sort_values("src_index")
    gap_stamps = gap_index["timestamp_s"].to_numpy()[source_gap["src_index"].to_numpy()]
    assert np.all(gap_stamps <= source_gap["time_s"].to_numpy() + 1e-9), "lacuna não pode usar quadro futuro"

    adapter = StagedImageSequenceAdapter(spec)
    decoded = decode_dataset_frames(adapter, video, [0, 5, 3], decode_video=lambda *_: [])
    assert [int(frame[0, 0, 0]) for frame in decoded] == [0, 5, 3]

    report = run_audit(spec)
    assert report["passed"], report["problems"]
    assert report["omnifall_reference_split"]["subjects_in_multiple_splits"] == {2: ["train", "val"]}
    assert report["coverage"]["train"]["fall_events"] == 2
    assert report["coverage"]["train"]["source_gap_timesteps"] > 0
    provenance = json.loads(paths.provenance.read_text())
    assert provenance["split"]["version"] == "selftest-v1" and len(provenance["split_sha256"]) == 64
    print("[PASS] manifesto, grade 10 fps causal em frames.parquet, lacunas, IGNORE explícito e auditoria")


def test_failures(spec: StagedDatasetSpec) -> None:
    from gatefall.data.staged.audit import leakage_problems
    from gatefall.data.staged.build import build_manifest, staged_paths
    from gatefall.data.staged.timeline import build_video_frames, grid_from_video

    manifest = pd.read_parquet(staged_paths(spec).manifest)
    identities = pd.read_parquet(staged_paths(spec).identities)
    leaked = manifest.copy()
    leaked.loc[leaked.index[0], "split"] = "test"
    assert any("subject" in problem for problem in leakage_problems(leaked, identities))
    leaked_views = identities.copy()
    leaked_views.loc[leaked_views.index[0], "split"] = "test"
    assert any("sincronizadas" in problem for problem in leakage_problems(manifest, leaked_views))
    duplicated = manifest.copy()
    duplicated.loc[duplicated.index[1], "sha256"] = duplicated["sha256"].iloc[0]
    assert any("idêntico" in problem for problem in leakage_problems(duplicated, identities))

    segments = pd.DataFrame({"start": [0.0], "end": [9.0], "label": [0]})
    _expect(UnverifiableLabelsError, lambda: build_video_frames(
        video_id="x", split="test", env="e", subject=1, grid=grid_from_video(30, 10.0),
        duration_s=3.0, segments=segments), "rótulo além da fonte aceito")
    mapped, unsupported = map_to_gatefall(pd.DataFrame({"label": [1, 10, 15, 9]}))
    assert unsupported == 2 and mapped["label"].tolist() == [1, IGNORE_LABEL, IGNORE_LABEL, 9]

    missing = replace(spec, processed_dir=spec.processed_dir.parent / "missing")
    victim = spec.raw_dir / "Subject3/Activity1/Trial1/Subject3Activity1Trial1Camera2"
    victim.rename(victim.with_name("hidden"))
    _expect(FileNotFoundError, lambda: build_manifest(missing), "fonte ausente aceita")
    victim.with_name("hidden").rename(victim)
    labels_file = spec.labels_dir / LABELS_FILENAME
    labels_file.write_text(labels_file.read_text() + "\n")
    _expect(ValueError, lambda: build_manifest(missing), "anotação adulterada aceita")
    print("[PASS] vazamento, vistas separadas, duplicatas, rótulo inverificável, fonte ausente e adulteração recusados")


def run_selftest() -> None:
    test_frozen_splits()
    test_causal_alignment()
    with tempfile.TemporaryDirectory(prefix="gatefall-staged-") as temporary:
        root = Path(temporary)
        test_timestamps_and_staging(root)
        spec = _world(root / "world")
        test_build_and_audit(spec)
        test_failures(spec)
    print("staged selftest PASS")


if __name__ == "__main__":
    run_selftest()
