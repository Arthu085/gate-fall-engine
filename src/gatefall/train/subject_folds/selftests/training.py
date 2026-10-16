import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.hashing import sha256_file
from gatefall.runs import REPOSITORY_ROOT
from gatefall.train.internal_folds.folds import DEVELOPMENT_SPLIT, RETAINED_SPLIT, membership_sha256
from gatefall.train.internal_folds.selftests.fixtures import _check, raises
from gatefall.train.internal_folds.training import FOLD_RUN_FILE, publish_fold_run_document
from gatefall.train.subject_folds.cli import run_prepare, run_train
from gatefall.train.subject_folds.folds import (
    FOLDS_FILE,
    RESTRICTED_LABELS,
    SubjectFoldPartition,
    build_subject_fold_partitions,
    fold_run_dir,
    stats_dir,
    summary_dir,
)
from gatefall.train.subject_folds.selftests.fixtures import (
    OFFICIAL_HELD_OUT_VIDEOS,
    SYNTHETIC_POSE_SHA256,
    RecordingPose,
    official_train_frames,
    pose_features,
    synthetic_workspace,
)
from gatefall.train.subject_folds.stats import PROVENANCE_FILE, fit_fold_stats, load_fold_stats
from gatefall.train.subject_folds.summary import SUMMARY_JSON_FILE, build_summary, write_summary

OFFICIAL_STATS = sorted((REPOSITORY_ROOT / "src/gatefall/features/stats").glob("*.json"))


def _window_mean(frames: pd.DataFrame, split: str) -> np.ndarray:
    source = PoseWindowDataset(frames, split, TRAIN_STRIDE, pose_features)
    windows = np.concatenate([source[index][0] for index in range(len(source))]).astype(np.float64)
    return windows.mean(axis=0)


def _window_count(frames: pd.DataFrame, split: str, stride: int) -> int:
    return len(PoseWindowDataset(frames, split, stride, pose_features))


def _selftest_normalization_provenance() -> bool:
    with synthetic_workspace() as (root, frames_path, _manifest_path, _pose_root):
        official_before = {path: sha256_file(path) for path in OFFICIAL_STATS}
        partitions = build_subject_fold_partitions(official_train_frames())
        input_sha256 = {"frames": sha256_file(frames_path), "manifest": "f" * 64}
        fit_ok = True
        for partition in partitions:
            recorder = RecordingPose()
            stats = fit_fold_stats(
                partition, stats_dir(root, partition.fold), frames_path, input_sha256, SYNTHETIC_POSE_SHA256,
                recorder.inputs().loader, force=False,
            )
            standardized = ~np.asarray(stats.stats.excluded_mask) & ~np.asarray(stats.stats.guarded_mask)
            development_mean = _window_mean(partition.frames, DEVELOPMENT_SPLIT)
            pooled_mean = _window_mean(partition.frames.assign(split=DEVELOPMENT_SPLIT), DEVELOPMENT_SPLIT)
            provenance = json.loads((stats_dir(root, partition.fold) / PROVENANCE_FILE).read_text())
            fit_ok &= (
                recorder.loaded_videos() == set(partition.development_video_ids)
                and np.allclose(np.asarray(stats.stats.mean)[standardized], development_mean[standardized])
                and not np.allclose(np.asarray(stats.stats.mean)[standardized], pooled_mean[standardized])
                and provenance["development_video_ids"] == list(partition.development_video_ids)
                and provenance["development_subjects"] == list(partition.development_subjects)
                and provenance["fit_split"] == DEVELOPMENT_SPLIT
                and provenance["stride"] == TRAIN_STRIDE
                and stats.window_count == _window_count(partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE)
            )

        first, second = partitions[0], partitions[1]
        swapped = root / "swapped"
        shutil.copytree(stats_dir(root, first.fold), swapped)
        record = second.membership_record
        cases = {
            "estatística de outra dobra": raises(
                ValueError,
                lambda: load_fold_stats(record, swapped, input_sha256, SYNTHETIC_POSE_SHA256),
                "proveniência diverge",
            ),
            "entradas antigas": raises(
                ValueError,
                lambda: load_fold_stats(
                    record, stats_dir(root, second.fold), {**input_sha256, "frames": "0" * 64}, SYNTHETIC_POSE_SHA256
                ),
                "input_sha256",
            ),
            "pose reextraída": raises(
                ValueError,
                lambda: load_fold_stats(record, stats_dir(root, second.fold), input_sha256, "1" * 64),
                "pose_features_sha256",
            ),
        }
        tampered = stats_dir(root, first.fold) / "pose.json"
        tampered.write_text(tampered.read_text().replace("[", "[ ", 1))
        cases["pose.json adulterado"] = raises(
            ValueError,
            lambda: load_fold_stats(first.membership_record, stats_dir(root, first.fold), input_sha256,
                                    SYNTHETIC_POSE_SHA256),
            "pose_stats_sha256",
        )
        failed = [name for name, ok in cases.items() if not ok]
        if failed:
            print(f"  estatísticas não recusadas: {failed}")
        official_untouched = {path: sha256_file(path) for path in OFFICIAL_STATS} == official_before
        return _check(
            "a estatística de pose de cada dobra é ajustada só nos vídeos de desenvolvimento em stride 4, difere da "
            "média que incluiria o grupo retido, registra a proveniência e recusa estatística de outra dobra, "
            "adulterada, de entradas antigas ou de pose reextraída; as estatísticas oficiais não mudam",
            fit_ok and not failed and official_untouched,
        )


def _metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _train(root: Path, frames_path: Path, manifest_path: Path, pose_root: Path, folds: list[str], force: bool,
           recorder: RecordingPose | None = None) -> None:
    pose = (recorder or RecordingPose()).inputs()
    run_train(root, frames_path, manifest_path, pose_root, folds, ["A"], [42], force=force, pose=pose)


def _isolation_check(root: Path, partitions: list[SubjectFoldPartition], recorder: RecordingPose) -> bool:
    isolation_ok = not (recorder.loaded_videos() & set(OFFICIAL_HELD_OUT_VIDEOS))
    document = json.loads((root / FOLDS_FILE).read_text(encoding="utf-8"))
    for partition in partitions:
        run_dir = fold_run_dir(root, partition.fold, "A", 42)
        final = _metrics(run_dir)["final"]
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        sidecar = json.loads((run_dir / FOLD_RUN_FILE).read_text(encoding="utf-8"))
        isolation_ok &= (
            sum(final["train"]["support"].values()) == _window_count(partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE)
            and sum(final["val"]["support"].values()) == _window_count(partition.frames, RETAINED_SPLIT, EVAL_STRIDE)
            and final["val"] == final["test"]
            and str(stats_dir(root, partition.fold)) in config["standardization_stats_path"]
            and config["seed"] == 42
            and sidecar["fold"] == partition.fold
            and sidecar["retained_subjects"] == list(partition.retained_subjects)
            and sidecar["canonical"] is False
            and sidecar["evaluation_split_mapping"]["test"] == RETAINED_SPLIT
            and sidecar["membership_sha256"] == document["membership_sha256"]
            and sidecar["input_sha256"] == document["input_sha256"]
            and sidecar["artifacts_sha256"]["checkpoint.pt"] == sha256_file(run_dir / "checkpoint.pt")
        )
    return _check(
        "o braço A treina só no desenvolvimento da dobra em stride 4, avalia só o grupo retido em stride 1 nas "
        "posições val/test, não lê val/test oficiais e publica fold_run.json ligado à composição e aos artefatos",
        isolation_ok,
    )


def _lifecycle_check(root: Path, frames_path: Path, manifest_path: Path, pose_root: Path, fold: str) -> bool:
    run_dir = fold_run_dir(root, fold, "A", 42)
    sidecar = run_dir / FOLD_RUN_FILE
    original = sidecar.read_text(encoding="utf-8")
    checkpoint = sha256_file(run_dir / "checkpoint.pt")
    _train(root, frames_path, manifest_path, pose_root, [fold], force=False)
    skip_ok = sha256_file(run_dir / "checkpoint.pt") == checkpoint and sidecar.read_text(encoding="utf-8") == original

    def refused(content: str | None, fragment: str) -> bool:
        if content is None:
            sidecar.unlink()
        else:
            sidecar.write_text(content, encoding="utf-8")
        failed = raises(
            RuntimeError, lambda: _train(root, frames_path, manifest_path, pose_root, [fold], False), fragment
        )
        return failed and sha256_file(run_dir / "checkpoint.pt") == checkpoint

    recorded = json.loads(original)
    cases = {
        "ausente": refused(None, "ausente"),
        "corrompido": refused("{ não é json", "ilegível"),
        "dobra": refused(json.dumps({**recorded, "fold": "retain_s1"}), "fold"),
        "composição": refused(json.dumps({**recorded, "membership_sha256": "0" * 64}), "membership_sha256"),
        "artefatos": refused(
            json.dumps({**recorded, "artifacts_sha256": {**recorded["artifacts_sha256"], "checkpoint.pt": "0" * 64}}),
            "artifacts_sha256",
        ),
    }
    failed_cases = [name for name, ok in cases.items() if not ok]
    if failed_cases:
        print(f"  casos de fold_run.json não recusados: {failed_cases}")
    summary_refused = raises(RuntimeError, lambda: build_summary(root), FOLD_RUN_FILE)

    marker = run_dir / "marker.txt"
    marker.write_text("run anterior", encoding="utf-8")
    _train(root, frames_path, manifest_path, pose_root, [fold], force=True)
    retrained = not marker.exists() and json.loads(sidecar.read_text(encoding="utf-8")) == recorded

    (run_dir / "checkpoint.pt").rename(run_dir / "checkpoint.bak")
    partial_refused = raises(
        RuntimeError, lambda: _train(root, frames_path, manifest_path, pose_root, [fold], False), "run parcial"
    )
    (run_dir / "checkpoint.bak").rename(run_dir / "checkpoint.pt")
    return _check(
        "run íntegro é pulado sem reescrever nada; sem --force, fold_run.json ausente, corrompido, de outra dobra "
        "ou composição ou desligado dos artefatos interrompe o treino e preserva o checkpoint; o resumo também "
        "recusa; --force retreina de forma determinística; run parcial é recusado",
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


def _folds_cases(root: Path) -> dict[str, tuple[Path, Callable[[dict], None]]]:
    path = root / FOLDS_FILE

    def recorded(mutate: Callable[[dict], None]) -> Callable[[dict], None]:
        def apply(document: dict) -> None:
            mutate(document["folds"][0])
            _rehash_membership(document)

        return apply

    def move_retained_video(record: dict) -> None:
        video_id = record["retained_video_ids"].pop()
        record["development_video_ids"] = sorted([*record["development_video_ids"], video_id])

    def drop_standing(record: dict) -> None:
        record["retained_support_eval_stride"]["walk"] += record["retained_support_eval_stride"]["standing"]
        record["retained_support_eval_stride"]["standing"] = 0

    def shift_support(record: dict) -> None:
        record["development_support_train_stride"]["walk"] += 1
        record["development_support_train_stride"]["fall"] -= 1

    return {
        "input_sha256 de frames": (path, lambda d: d["input_sha256"].update(frames="1" * 64)),
        "input_sha256 com papel extra": (path, lambda d: d["input_sha256"].update(val="2" * 64)),
        "pose_features_sha256": (path, lambda d: d.update(pose_features_sha256="3" * 64)),
        "membership_sha256": (path, lambda d: d.update(membership_sha256="4" * 64)),
        "evaluation_split_mapping": (path, lambda d: d["evaluation_split_mapping"].update(test="test")),
        "official_splits_read": (path, lambda d: d.update(official_splits_read=["train", "test"])),
        "grupos retidos": (path, lambda d: d.update(retained_subject_groups=[[0], [1, 3], [4, 6, 8]])),
        "seeds": (path, lambda d: d.update(seeds=[42, 43])),
        "braços": (path, lambda d: d.update(arms=["A", "B0"])),
        "window_contract": (path, lambda d: d["window_contract"].update(train_stride=1)),
        "campo extra": (path, lambda d: d.update(pooled_score=0.8)),
        "composição": (path, recorded(move_retained_video)),
        "sujeitos retidos": (path, recorded(lambda r: r.update(retained_subjects=[1]))),
        "janelas de desenvolvimento": (
            path, recorded(lambda r: r.update(development_windows_train_stride=r["development_windows_train_stride"] + 1))
        ),
        "suporte deslocado": (path, recorded(shift_support)),
        "classe restrita ausente": (path, recorded(drop_standing)),
    }


def _republish_after(run_dir: Path, mutate_metrics: Callable[[dict], None]) -> Callable[[], None]:
    """Altera `metrics.json` e regrava `fold_run.json` coerente, simulando adulteração com hashes refeitos."""
    sidecar = run_dir / FOLD_RUN_FILE
    original_sidecar = sidecar.read_bytes()
    restore_metrics = _tamper_json(run_dir / "metrics.json", mutate_metrics)
    document = {key: value for key, value in json.loads(original_sidecar).items() if key != "artifacts_sha256"}
    publish_fold_run_document(run_dir, document)

    def restore() -> None:
        restore_metrics()
        sidecar.write_bytes(original_sidecar)

    return restore


def _summary_tamper_check(root: Path) -> bool:
    fold = "retain_s0"
    provenance_path = stats_dir(root, fold) / PROVENANCE_FILE
    run_dir = fold_run_dir(root, fold, "A", 42)
    sidecar = run_dir / FOLD_RUN_FILE
    cases = _folds_cases(root)
    cases.update({
        "proveniência: window_count": (provenance_path, lambda d: d.update(window_count=d["window_count"] + 1)),
        "proveniência: input_sha256": (provenance_path, lambda d: d["input_sha256"].update(manifest="5" * 64)),
        "proveniência: vídeos": (provenance_path, lambda d: d["development_video_ids"].pop()),
        "proveniência: sujeitos": (provenance_path, lambda d: d.update(development_subjects=[0, 1])),
        "fold_run: evaluation_split_mapping": (sidecar, lambda d: d["evaluation_split_mapping"].update(val="val")),
        "fold_run: estatística": (sidecar, lambda d: d["standardization_stats_sha256"].update(pose="6" * 64)),
        "fold_run: pose": (sidecar, lambda d: d.update(pose_features_sha256="7" * 64)),
    })
    accepted: list[str] = []
    for name, (path, mutate) in cases.items():
        restore = _tamper_json(path, mutate)
        try:
            if not raises((ValueError, RuntimeError), lambda: build_summary(root)):
                accepted.append(name)
        finally:
            restore()

    def diverge_test(metrics: dict) -> None:
        metrics["final"]["test"]["macro_f1_restricted"] = 0.999

    def shift_retained_support(metrics: dict) -> None:
        for split in ("val", "test"):
            metrics["final"][split]["support"]["walk"] += 1

    def shift_train_support(metrics: dict) -> None:
        metrics["final"]["train"]["support"]["walk"] += 1

    for name, mutate, fragment in (
        ("val ≠ test", diverge_test, "mesmo grupo retido"),
        ("suporte do retido", shift_retained_support, ""),
        ("suporte de treino", shift_train_support, ""),
    ):
        restore = _republish_after(run_dir, mutate)
        try:
            if not raises(RuntimeError, lambda: build_summary(root), fragment):
                accepted.append(name)
        finally:
            restore()

    checkpoint = run_dir / "checkpoint.pt"
    original_checkpoint = checkpoint.read_bytes()
    checkpoint.write_bytes(original_checkpoint[:-16])
    try:
        if not raises(RuntimeError, lambda: build_summary(root), "checkpoint"):
            accepted.append("checkpoint truncado")
    finally:
        checkpoint.write_bytes(original_checkpoint)
    if accepted:
        print(f"  adulterações aceitas pelo resumo: {accepted}")
    return _check(
        "o resumo recusa folds.json com contrato, hashes, grupos, composição, contagens ou suporte inconsistentes "
        "ou com classe restrita ausente, proveniência divergente, fold_run.json adulterado, val diferente de "
        "test, suporte por classe divergente do registrado e checkpoint corrompido, sem gravar saída",
        not accepted and not summary_dir(root).exists(),
    )


def _tree_digest(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _summary_check(root: Path, frames_path: Path) -> bool:
    before = _tree_digest(root)
    processed = frames_path.parent
    hidden = processed.with_name("processed_hidden")
    processed.rename(hidden)
    try:
        built = build_summary(root)
    finally:
        hidden.rename(processed)
    read_only = _tree_digest(root) == before
    written = write_summary(root, force=False)
    path = summary_dir(root) / "baseline" / SUMMARY_JSON_FILE
    summary = json.loads(path.read_text(encoding="utf-8"))
    not_overwritten = write_summary(root, force=False) is False
    document = json.loads((root / FOLDS_FILE).read_text(encoding="utf-8"))
    blocks_ok = len(summary["folds"]) == 3 and all(
        len(block["runs"]) == 1
        and list(block["runs"][0]["retained_per_class"]) == list(RESTRICTED_LABELS)
        and all(entry["support"] > 0 for entry in block["runs"][0]["retained_per_class"].values())
        and block["runs"][0]["retained_support"] == fold["retained_support_eval_stride"]
        and block["limitations"]
        for block, fold in zip(summary["folds"], document["folds"])
    )
    return _check(
        "o resumo é somente leitura, não abre frames, relata macro-F1 restrita, F1 e suporte por classe e "
        "limitações de cada dobra, sem escore agregado, e não sobrescreve saída existente",
        read_only
        and built == summary
        and written
        and not_overwritten
        and blocks_ok
        and summary["pooled_score"] is None
        and summary["canonical"] is False
        and summary["input_sha256"] == document["input_sha256"]
        and summary["membership_sha256"] == document["membership_sha256"],
    )


def _selftest_trained_workspace() -> list[bool]:
    with synthetic_workspace() as (root, frames_path, manifest_path, pose_root):
        partitions = run_prepare(root, frames_path, manifest_path, pose_root, False, RecordingPose().inputs())
        missing_runs = raises(FileNotFoundError, lambda: build_summary(root), "runs ausentes")
        recorder = RecordingPose()
        _train(root, frames_path, manifest_path, pose_root, [p.fold for p in partitions], False, recorder)
        stale_pose = raises(
            ValueError,
            lambda: run_train(root, frames_path, manifest_path, pose_root, ["retain_s0"], ["A"], [42], False,
                              RecordingPose(sha256="9" * 64).inputs()),
            "pose mudaram",
        )
        return [
            _check("o resumo recusa runs ausentes e o treino recusa pose diferente da registrada no prepare",
                   missing_runs and stale_pose),
            _isolation_check(root, partitions, recorder),
            _lifecycle_check(root, frames_path, manifest_path, pose_root, partitions[0].fold),
            _summary_tamper_check(root),
            _summary_check(root, frames_path),
        ]


def run_training_selftest() -> bool:
    results = [_selftest_normalization_provenance(), *_selftest_trained_workspace()]
    return all(results)
