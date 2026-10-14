import json
import shutil
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
    RETAINED_SPLIT,
    build_fold_partitions,
    fold_run_dir,
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
            ValueError, lambda: load_fold_stats(first, stats_dir(root, first.fold), frames_path, input_sha256), "hash"
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


def _selftest_training_isolation_and_summary() -> bool:
    with synthetic_workspace() as (root, frames_path, manifest_path):
        recorder = RecordingLoaders()
        partitions = run_prepare(
            root, frames_path, manifest_path, force=False, loaders=recorder.loaders(), quality=QUALITY
        )
        recorder = RecordingLoaders()
        run_train(root, frames_path, manifest_path, "pilot", [p.fold for p in partitions], list(ARMS),
                  force=False, loaders=recorder.loaders(), quality=QUALITY)
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
                )

        first_fold = partitions[0].fold
        run_dir = fold_run_dir(root, first_fold, "A", 42)
        checkpoint_before = sha256_file(run_dir / "checkpoint.pt")
        run_train(root, frames_path, manifest_path, "pilot", [first_fold], ["A"], force=False,
                  loaders=RecordingLoaders().loaders(), quality=QUALITY)
        skip_ok = sha256_file(run_dir / "checkpoint.pt") == checkpoint_before
        sidecar = run_dir / FOLD_RUN_FILE
        sidecar.write_text(sidecar.read_text().replace(first_fold, "retain_other"), encoding="utf-8")
        sidecar_refused = raises(
            RuntimeError,
            lambda: run_train(root, frames_path, manifest_path, "pilot", [first_fold], ["A"], force=False,
                              loaders=RecordingLoaders().loaders(), quality=QUALITY),
            "diverge",
        )
        summary_refused = raises(RuntimeError, lambda: build_summary(root, "pilot"), FOLD_RUN_FILE)
        partial = fold_run_dir(root, first_fold, "B0", 42)
        (partial / "checkpoint.pt").unlink()
        partial_refused = raises(
            RuntimeError,
            lambda: run_train(root, frames_path, manifest_path, "pilot", [first_fold], ["B0"], force=False,
                              loaders=RecordingLoaders().loaders(), quality=QUALITY),
            "run parcial",
        )
        run_train(root, frames_path, manifest_path, "pilot", [first_fold], ["A", "B0"], force=True,
                  loaders=RecordingLoaders().loaders(), quality=QUALITY)
        protection_ok = skip_ok and sidecar_refused and summary_refused and partial_refused

        written = write_summary(root, "pilot", force=False)
        summary = json.loads((summary_dir(root) / "pilot" / SUMMARY_JSON_FILE).read_text(encoding="utf-8"))
        not_overwritten = write_summary(root, "pilot", force=False) is False
        assessment_missing = raises(FileNotFoundError, lambda: build_summary(root, "assessment"), "runs ausentes")
        summary_ok = (
            written
            and not_overwritten
            and assessment_missing
            and summary["criterion"]["status"] == "not_applicable"
            and summary["canonical"] is False
            and len(summary["folds"]) == 3
            and all(len(block["runs"]) == 3 and "comparison" not in block for block in summary["folds"])
        )
        return _check(
            "o treino de A, B0 e B1 usa só o desenvolvimento da dobra em stride 4, avalia só o ambiente retido "
            "em stride 1 nas posições val/test, não lê val/test oficiais, preserva runs completos, recusa runs "
            "parciais ou de outra dobra sem --force e o resumo do piloto não aplica o critério",
            isolation_ok and protection_ok and summary_ok,
        )


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
        _selftest_training_isolation_and_summary(),
        _selftest_criterion(),
    ]
    return all(results)
