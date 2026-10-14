import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.hashing import sha256_file
from gatefall.runs import REPOSITORY_ROOT
from gatefall.train.internal_folds.cli import run_prepare, run_train
from gatefall.train.internal_folds.folds import (
    ARMS,
    DEVELOPMENT_SPLIT,
    EVALUATION_SPLIT_MAPPING,
    FOLDS_FILE,
    RETAINED_SPLIT,
    build_fold_partitions,
    fold_run_dir,
    membership_sha256,
    stats_dir,
    summary_dir,
)
from gatefall.train.internal_folds.selftests.fixtures import (
    OFFICIAL_HELD_OUT_VIDEOS,
    QUALITY,
    RecordingLoaders,
    _check,
    official_train_frames,
    pose_features,
    raises,
    synthetic_workspace,
    visual_features,
)
from gatefall.train.internal_folds.stats import fit_fold_stats, load_fold_stats
from gatefall.train.internal_folds.summary import (
    SUMMARY_JSON_FILE,
    apply_criterion,
    build_summary,
    predicted_other_fraction,
    write_summary,
)
from gatefall.train.internal_folds.training import FOLD_RUN_FILE

OFFICIAL_STATS = sorted((REPOSITORY_ROOT / "src/gatefall/features/stats").glob("*.json"))


def _window_mean(frames, split: str, loader) -> np.ndarray:
    source = PoseWindowDataset(frames, split, TRAIN_STRIDE, loader)
    windows = np.concatenate([source[index][0] for index in range(len(source))]).astype(np.float64)
    return windows.mean(axis=0)


def _window_count(frames, split: str, stride: int) -> int:
    return len(PoseWindowDataset(frames, split, stride, pose_features))


def _selftest_normalization_provenance() -> bool:
    with synthetic_workspace() as (root, frames_path, manifest_path):
        official_before = {path: sha256_file(path) for path in OFFICIAL_STATS}
        partitions = build_fold_partitions(official_train_frames())
        input_sha256 = {"frames": sha256_file(frames_path), "manifest": sha256_file(manifest_path)}
        fit_ok = True
        for partition in partitions:
            recorder = RecordingLoaders()
            loaders = recorder.loaders()
            stats = fit_fold_stats(
                partition, stats_dir(root, partition.fold), frames_path, input_sha256,
                loaders.pose, loaders.visual, force=False,
            )
            development_mean = _window_mean(partition.frames, DEVELOPMENT_SPLIT, pose_features)
            pooled_frames = partition.frames.assign(split=DEVELOPMENT_SPLIT)
            pooled_mean = _window_mean(pooled_frames, DEVELOPMENT_SPLIT, pose_features)
            standardized = ~np.asarray(stats.pose.excluded_mask) & ~np.asarray(stats.pose.guarded_mask)
            visual_mean = _window_mean(partition.frames, DEVELOPMENT_SPLIT, visual_features)
            provenance = json.loads((stats_dir(root, partition.fold) / "provenance.json").read_text())
            fit_ok &= (
                set(recorder.loaded_videos()) == set(partition.development_video_ids)
                and np.allclose(np.asarray(stats.pose.mean)[standardized], development_mean[standardized])
                and not np.allclose(np.asarray(stats.pose.mean)[standardized], pooled_mean[standardized])
                and np.allclose(np.asarray(stats.dinov3.mean), visual_mean, atol=1e-4)
                and provenance["development_video_ids"] == list(partition.development_video_ids)
                and provenance["fit_split"] == DEVELOPMENT_SPLIT
                and stats.window_count == _window_count(partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE)
            )

        first, second = partitions[0], partitions[1]
        swapped = root / "swapped"
        shutil.copytree(stats_dir(root, first.fold), swapped)
        swapped_refused = raises(
            ValueError, lambda: load_fold_stats(second, swapped, frames_path, input_sha256), "proveniência diverge"
        )
        tampered = stats_dir(root, first.fold) / "dinov3.json"
        tampered.write_text(tampered.read_text().replace("[", "[ ", 1))
        tampered_refused = raises(
            ValueError, lambda: load_fold_stats(first, stats_dir(root, first.fold), frames_path, input_sha256), "files"
        )
        stale_inputs = {**input_sha256, "frames": "0" * 64}
        stale_refused = raises(
            ValueError, lambda: load_fold_stats(second, stats_dir(root, second.fold), frames_path, stale_inputs),
            "input_sha256",
        )
        official_untouched = {path: sha256_file(path) for path in OFFICIAL_STATS} == official_before
        return _check(
            "as estatísticas de pose e DINOv3 de cada dobra são ajustadas só nos vídeos de desenvolvimento, "
            "diferem da média que incluiria o ambiente retido, registram a proveniência e recusam estatística "
            "de outra dobra, adulterada ou de entradas antigas; as estatísticas oficiais não mudam",
            fit_ok and swapped_refused and tampered_refused and stale_refused and official_untouched,
        )


def _metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _total_support(split_metrics: dict) -> int:
    return sum(split_metrics["support"].values())


def _train(root: Path, frames_path: Path, manifest_path: Path, folds: list[str], arms: list[str], force: bool,
           recorder: RecordingLoaders | None = None) -> None:
    loaders = (recorder or RecordingLoaders()).loaders()
    run_train(root, frames_path, manifest_path, "pilot", folds, arms, force=force, loaders=loaders, quality=QUALITY)


def _isolation_check(root: Path, partitions: list, recorder: RecordingLoaders) -> bool:
    isolation_ok = not (recorder.loaded_videos() & set(OFFICIAL_HELD_OUT_VIDEOS))
    for partition in partitions:
        development_windows = _window_count(partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE)
        retained_windows = _window_count(partition.frames, RETAINED_SPLIT, EVAL_STRIDE)
        for arm in ARMS:
            run_dir = fold_run_dir(root, partition.fold, arm, 42)
            final = _metrics(run_dir)["final"]
            config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            stats_paths = [value for key, value in config.items() if key.endswith("standardization_stats_path")]
            document = json.loads((run_dir / FOLD_RUN_FILE).read_text(encoding="utf-8"))
            isolation_ok &= (
                _total_support(final["train"]) == development_windows
                and _total_support(final["val"]) == retained_windows
                and final["val"] == final["test"]
                and all(str(stats_dir(root, partition.fold)) in path for path in stats_paths)
                and document["fold"] == partition.fold
                and document["canonical"] is False
                and document["evaluation_split_mapping"]["test"] == RETAINED_SPLIT
                and document["artifacts_sha256"]["checkpoint.pt"] == sha256_file(run_dir / "checkpoint.pt")
            )
    return _check(
        "o treino de A, B0 e B1 usa só o desenvolvimento da dobra em stride 4, avalia só o ambiente retido em "
        "stride 1 nas posições val/test, não lê val/test oficiais e publica fold_run.json ligado aos artefatos",
        isolation_ok,
    )


def _lifecycle_check(root: Path, frames_path: Path, manifest_path: Path, fold: str) -> bool:
    run_dir = fold_run_dir(root, fold, "A", 42)
    sidecar = run_dir / FOLD_RUN_FILE
    original = sidecar.read_text(encoding="utf-8")
    checkpoint = sha256_file(run_dir / "checkpoint.pt")
    _train(root, frames_path, manifest_path, [fold], ["A"], force=False)
    skip_ok = sha256_file(run_dir / "checkpoint.pt") == checkpoint and sidecar.read_text(encoding="utf-8") == original

    def refused(content: str | None, fragment: str) -> bool:
        if content is None:
            sidecar.unlink()
        else:
            sidecar.write_text(content, encoding="utf-8")
        failed = raises(RuntimeError, lambda: _train(root, frames_path, manifest_path, [fold], ["A"], False), fragment)
        preserved = sha256_file(run_dir / "checkpoint.pt") == checkpoint and (
            not sidecar.exists() if content is None else sidecar.read_text(encoding="utf-8") == content
        )
        return failed and preserved

    recorded = json.loads(original)
    cases = {
        "ausente": refused(None, "ausente"),
        "corrompido": refused("{ não é json", "ilegível"),
        "seed": refused(json.dumps({**recorded, "seed": 43}), "seed"),
        "dobra": refused(json.dumps({**recorded, "fold": "retain_EnvB"}), "fold"),
        "artefatos": refused(
            json.dumps({**recorded, "artifacts_sha256": {**recorded["artifacts_sha256"], "checkpoint.pt": "0" * 64}}),
            "artifacts_sha256",
        ),
        "sem artefatos": refused(
            json.dumps({key: value for key, value in recorded.items() if key != "artifacts_sha256"}), "artifacts"
        ),
    }
    summary_refused = raises(RuntimeError, lambda: build_summary(root, "pilot"), FOLD_RUN_FILE)

    marker = run_dir / "marker.txt"
    marker.write_text("run anterior", encoding="utf-8")
    _train(root, frames_path, manifest_path, [fold], ["A"], force=True)
    retrained = not marker.exists() and json.loads(sidecar.read_text(encoding="utf-8")) == recorded

    partial = fold_run_dir(root, fold, "B0", 42)
    (partial / "checkpoint.pt").unlink()
    partial_refused = raises(
        RuntimeError, lambda: _train(root, frames_path, manifest_path, [fold], ["B0"], False), "run parcial"
    )
    _train(root, frames_path, manifest_path, [fold], ["B0"], force=True)
    failed_cases = [name for name, ok in cases.items() if not ok]
    if failed_cases:
        print(f"  casos de fold_run.json não recusados: {failed_cases}")
    return _check(
        "run completo e íntegro é pulado sem reescrever nada; sem --force, fold_run.json ausente, corrompido, "
        "de outra seed ou dobra ou desligado dos artefatos interrompe o treino sem criar metadados e preserva o "
        "checkpoint; o resumo também recusa; --force retreina de fato; run parcial é recusado",
        skip_ok and not failed_cases and summary_refused and retrained and partial_refused,
    )


def _tamper_json(path: Path, mutate: Callable[[dict], None]) -> Callable[[], None]:
    original = path.read_bytes()
    data = json.loads(original)
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")

    def restore() -> None:
        path.write_bytes(original)

    return restore


def _rehash_membership(document: dict) -> None:
    document["membership_sha256"] = membership_sha256(document["folds"])


def _summary_tamper_check(root: Path) -> bool:
    folds_path = root / FOLDS_FILE
    fold = json.loads(folds_path.read_text(encoding="utf-8"))["folds"][0]["fold"]
    provenance_path = stats_dir(root, fold) / "provenance.json"
    sidecar = fold_run_dir(root, fold, "B1", 42) / FOLD_RUN_FILE

    def move_retained_video(document: dict) -> None:
        record = document["folds"][0]
        video_id = record["retained_video_ids"].pop()
        record["development_video_ids"] = sorted([*record["development_video_ids"], video_id])
        _rehash_membership(document)

    def shift_count(field: str) -> Callable[[dict], None]:
        def mutate(document: dict) -> None:
            document["folds"][0][field] += 1
            _rehash_membership(document)

        return mutate

    cases: dict[str, tuple[Path, Callable[[dict], None]]] = {
        "input_sha256 de frames": (folds_path, lambda d: d["input_sha256"].update(frames="1" * 64)),
        "input_sha256 com papel extra": (folds_path, lambda d: d["input_sha256"].update(val="2" * 64)),
        "membership_sha256": (folds_path, lambda d: d.update(membership_sha256="3" * 64)),
        "evaluation_split_mapping": (folds_path, lambda d: d["evaluation_split_mapping"].update(test="test")),
        "official_splits_read": (folds_path, lambda d: d.update(official_splits_read=["train", "val"])),
        "critério": (folds_path, lambda d: d.update(reproduction_criterion="outro critério")),
        "seed_sets": (folds_path, lambda d: d["seed_sets"].update(assessment=[42, 43])),
        "window_contract": (folds_path, lambda d: d["window_contract"].update(train_stride=1)),
        "campo extra": (folds_path, lambda d: d.update(note="x")),
        "composição gravada": (folds_path, move_retained_video),
        "janelas de desenvolvimento": (folds_path, shift_count("development_windows_train_stride")),
        "janelas retidas": (folds_path, shift_count("retained_windows_eval_stride")),
        "quality_features_sha256": (folds_path, lambda d: d.update(quality_features_sha256="4" * 64)),
        "proveniência: window_count": (provenance_path, lambda d: d.update(window_count=d["window_count"] + 1)),
        "proveniência: input_sha256": (provenance_path, lambda d: d["input_sha256"].update(manifest="5" * 64)),
        "proveniência: vídeos": (provenance_path, lambda d: d["development_video_ids"].pop()),
        "fold_run: evaluation_split_mapping": (sidecar, lambda d: d["evaluation_split_mapping"].update(val="val")),
        "fold_run: estatísticas": (sidecar, lambda d: d["standardization_stats_sha256"].update(pose="6" * 64)),
    }
    accepted: list[str] = []
    for name, (path, mutate) in cases.items():
        restore = _tamper_json(path, mutate)
        try:
            if not raises((ValueError, RuntimeError), lambda: build_summary(root, "pilot")):
                accepted.append(name)
        finally:
            restore()
    if accepted:
        print(f"  adulterações aceitas pelo resumo: {accepted}")
    no_output = not summary_dir(root).exists()
    return _check(
        "o resumo recusa folds.json com input_sha256, membership_sha256, mapeamento de avaliação, contrato, "
        "composição ou contagens de janelas inconsistentes, proveniência de estatística divergente e "
        "fold_run.json adulterado, sem gravar saída",
        not accepted and no_output,
    )


def _tree_digest(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _summary_check(root: Path, frames_path: Path) -> bool:
    before = _tree_digest(root)
    processed = frames_path.parent
    hidden = processed.with_name("processed_hidden")
    processed.rename(hidden)
    try:
        built = build_summary(root, "pilot")
    finally:
        hidden.rename(processed)
    read_only = _tree_digest(root) == before
    written = write_summary(root, "pilot", force=False)
    summary = json.loads((summary_dir(root) / "pilot" / SUMMARY_JSON_FILE).read_text(encoding="utf-8"))
    not_overwritten = write_summary(root, "pilot", force=False) is False
    assessment_missing = raises(FileNotFoundError, lambda: build_summary(root, "assessment"), "runs ausentes")
    document = json.loads((root / FOLDS_FILE).read_text(encoding="utf-8"))
    return _check(
        "o resumo do piloto é somente leitura, não abre frames nem manifesto, não aplica o critério, não "
        "sobrescreve saída existente e só relata metadados reconciliados",
        read_only
        and built == summary
        and written
        and not_overwritten
        and assessment_missing
        and summary["criterion"]["status"] == "not_applicable"
        and summary["canonical"] is False
        and summary["input_sha256"] == document["input_sha256"]
        and summary["membership_sha256"] == document["membership_sha256"]
        and summary["evaluation_split_mapping"] == EVALUATION_SPLIT_MAPPING
        and len(summary["folds"]) == 3
        and all(len(block["runs"]) == 3 and "comparison" not in block for block in summary["folds"]),
    )


def _selftest_trained_workspace() -> list[bool]:
    with synthetic_workspace() as (root, frames_path, manifest_path):
        partitions = run_prepare(
            root, frames_path, manifest_path, force=False, loaders=RecordingLoaders().loaders(), quality=QUALITY
        )
        recorder = RecordingLoaders()
        _train(root, frames_path, manifest_path, [p.fold for p in partitions], list(ARMS), False, recorder)
        return [
            _isolation_check(root, partitions, recorder),
            _lifecycle_check(root, frames_path, manifest_path, partitions[0].fold),
            _summary_tamper_check(root),
            _summary_check(root, frames_path),
        ]


def _selftest_criterion() -> bool:
    matrix = [[0] * 10 for _ in range(10)]
    matrix[0][0], matrix[0][9], matrix[8][9], matrix[9][9], matrix[9][0] = 6, 2, 2, 5, 1
    fraction, outside_other = predicted_other_fraction({"confusion_matrix": matrix})

    def fold(name: str, a: list[float], b0: list[float], b1: list[float]) -> dict:
        runs = [
            {"arm": arm, "seed": 42 + index, "predicted_other_fraction": value}
            for arm, values in (("A", a), ("B0", b0), ("B1", b1))
            for index, value in enumerate(values)
        ]
        return {"fold": name, "runs": runs}

    reproduced = apply_criterion(
        [
            fold("f1", [0.1, 0.2], [0.21, 0.5], [0.0, 0.9]),
            fold("f2", [0.1, 0.3], [0.31, 0.4], [0.3, 0.4]),
            fold("f3", [0.1, 0.3], [0.2, 0.9], [0.5, 0.6]),
        ]
    )
    not_reproduced = apply_criterion(
        [
            fold("f1", [0.1, 0.2], [0.2, 0.5], [0.0, 0.9]),
            fold("f2", [0.1, 0.3], [0.31, 0.4], [0.3, 0.4]),
            fold("f3", [0.1, 0.3], [0.2, 0.9], [0.5, 0.6]),
        ]
    )
    return _check(
        "fração predita other = coluna other fora da linha other / janelas fora de other; o critério exige "
        "mínimo de B0 (ou B1) estritamente acima do máximo de A em pelo menos duas das três dobras",
        fraction == 4 / 10
        and outside_other == 10
        and reproduced["reproducing_folds"] == {"B0": ["f1", "f2"], "B1": ["f3"]}
        and reproduced["reproduced"] is True
        and not_reproduced["reproducing_folds"] == {"B0": ["f2"], "B1": ["f3"]}
        and not_reproduced["reproduced"] is False,
    )


def run_training_selftest() -> bool:
    results = [
        _selftest_normalization_provenance(),
        *_selftest_trained_workspace(),
        _selftest_criterion(),
    ]
    return all(results)
