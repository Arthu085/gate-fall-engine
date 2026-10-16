import json
import shutil
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL, TRAIN_STRIDE
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis import standing_sitting_diagnosis, stratified_error_diagnosis
from gatefall.eval.analysis.classification_diagnosis import DiagnosisInput
from gatefall.eval.analysis.selftests.classification_diagnosis import POSE_STATS, _final_split
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.eval.analysis.selftests.stratified_error_diagnosis import _raises, _tree_hashes
from gatefall.eval.analysis.standing_sitting_diagnosis import (
    CONSIDER_VERDICT,
    INSUFFICIENT_VERDICT,
    INTERIOR,
    NOT_EVALUABLE,
    NOT_SUSTAINED,
    OFFICIAL_TRAIN,
    OFFICIAL_VAL,
    POSE_ABSENT,
    POSE_PRESENT,
    REPORT_JSON_FILE,
    REPORT_SUMMARY_FILE,
    REPORT_WINDOWS_FILE,
    SUBJECT_FOLD,
    SUSTAINED,
    TRANSITION,
    UNRESOLVED_VERDICT,
    VIDEO_END,
    evaluate_rule,
    join_attributes,
    labeled_windows,
    overall_verdict,
    pair_breakdown,
    pair_verdict,
    pose_files,
    run_diagnosis,
    support_rule,
)
from gatefall.eval.shared.event_artifacts import EventRunConfig
from gatefall.eval.shared.orchestration import EventEvaluation, Predictions
from gatefall.hashing import sha256_file
from gatefall.pose.loading import PoseArrays
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.subject_folds.cli import run_prepare, run_train
from gatefall.train.subject_folds.folds import BASELINE_VARIANT, build_subject_fold_partitions
from gatefall.train.subject_folds.selftests.fixtures import (
    TRAIN_VIDEOS,
    VIDEO_SPEC,
    RawRecordingPose,
    frames_table,
    official_train_frames,
    synthetic_workspace,
)

WALK, SITTING, STAND_UP, STANDING = 0, 4, 7, 8
TEST_VIDEOS = frozenset(video_id for video_id, split, _env, _subject in VIDEO_SPEC if split == "test")

# Vídeo com 30 quadros `standing` seguidos de 30 `walk`, sem pose nos quadros
# 0..9 e 30..44. Categorias esperadas: k_end 10 sem pose (borda replicada);
# 24 interior (5 quadros até o fim do trecho); 25 e 29 transição pela margem
# final; 35 transição por janela misturada; 44 misturada, mas sem pose na
# maioria dos índices (a pose ausente tem precedência); 59 interior no fim do
# vídeo, onde não há trecho seguinte.
CATEGORY_LABELS = [STANDING] * 30 + [WALK] * 30
CATEGORY_POSE_MISSING = [*range(0, 10), *range(30, 45)]
EXPECTED_CATEGORIES = {10: POSE_ABSENT, 24: INTERIOR, 25: TRANSITION, 29: TRANSITION, 35: TRANSITION,
                       44: POSE_ABSENT, 59: INTERIOR}


def _pose(n_frames: int, missing: list[int]) -> PoseArrays:
    frames = np.arange(n_frames, dtype=np.float32)
    bbox = np.stack([frames, np.zeros(n_frames), frames + 20.0, np.full(n_frames, 40.0)], axis=1)
    keypoints = np.zeros((n_frames, 17, 3), dtype=np.float32)
    keypoints[:, :, 2] = 0.5
    person_found = np.ones(n_frames, dtype=bool)
    person_found[missing] = False
    return PoseArrays(keypoints=keypoints, bbox=bbox.astype(np.float32), person_found=person_found, k=n_frames,
                      width=320, height=240)


def _category_frames() -> pd.DataFrame:
    return pd.DataFrame([{"video_id": "EnvA/v1", "split": "val", "env": "EnvA", "subject": 5, "frame_index": index,
                          "label": label} for index, label in enumerate(CATEGORY_LABELS)])


def _category_windows() -> pd.DataFrame:
    pose = _pose(len(CATEGORY_LABELS), CATEGORY_POSE_MISSING)
    return labeled_windows(_category_frames(), EVAL_STRIDE, frozenset({STANDING, WALK}), lambda _video_id: pose)


def _selftest_window_categories() -> bool:
    windows = _category_windows().set_index("k_end")
    categories_ok = all(windows.loc[k_end, "category"] == category for k_end, category in EXPECTED_CATEGORIES.items())
    following_ok = (windows.loc[29, "following_label"] == "walk" and windows.loc[59, "following_label"] == VIDEO_END)
    fractions_ok = (
        windows.loc[10, "detected_fraction"] == 1 / 24
        and windows.loc[24, "detected_fraction"] == 15 / 24
        and windows.loc[44, "detected_fraction"] == 9 / 24
    )
    stride_ok = list(labeled_windows(_category_frames(), TRAIN_STRIDE, frozenset({STANDING}),
                                     lambda _video_id: _pose(60, []))["k_end"]) == list(range(0, 30, TRAIN_STRIDE))
    short_pose = _raises(
        ValueError,
        lambda: labeled_windows(_category_frames(), EVAL_STRIDE, frozenset({STANDING}), lambda _video_id: _pose(59, [])),
        "EnvA/v1",
    )
    return _check(
        "categoria exclusiva por janela: pose ausente tem precedência; transição por janela misturada ou pela margem "
        "final só quando há trecho seguinte; interior no fim do vídeo; fração de detecção inclui quadros "
        "replicados; stride e K de pose divergente",
        categories_ok and following_ok and fractions_ok and stride_ok and short_pose,
    )


def _breakdown_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    attributes = _category_windows()
    standing = cast(pd.DataFrame, attributes[attributes["label"] == STANDING])
    errors = {10, 25, 29}
    predictions = standing.assign(pred_label=[WALK if int(k_end) in errors else STANDING
                                              for k_end in standing["k_end"]])
    matrix = np.zeros((10, 10), dtype=np.int64)
    matrix[STANDING][WALK] = len(errors)
    matrix[STANDING][STANDING] = len(standing) - len(errors)
    return join_attributes(predictions, standing, "sintético"), matrix


def _selftest_pair_breakdown() -> bool:
    windows, matrix = _breakdown_inputs()
    block = pair_breakdown(windows, "standing", "walk", matrix, "sintético")
    categories = {row["stratum"]: row for row in block["strata"]["category"]}
    unit = block["units"]["subject_5"]
    counted = (
        block["counts"] == {"n_true": 30, "n_error": 3, "n_correct": 27, "n_other_outcome": 0}
        and categories[TRANSITION]["n_error"] == 2
        and categories[POSE_ABSENT]["n_error"] == 1
        and unit[TRANSITION]["n_error"] == 2 and unit[POSE_PRESENT]["n_error"] == 2
        and unit[POSE_ABSENT]["n_true"] + unit[POSE_PRESENT]["n_true"] == 30
        and block["destinations"] == {"walk": 3, "standing": 27}
        and block["profiles"]["error"]["n"] == 3
    )
    empty = pair_breakdown(windows, "sitting", "stand_up", matrix, "sintético")
    empty_ok = empty["counts"]["n_true"] == 0 and empty["units"] == {} and empty["profiles"]["error"]["n"] == 0
    tampered = matrix.copy()
    tampered[STANDING][WALK] += 1
    tamper_fails = _raises(ValueError, lambda: pair_breakdown(windows, "standing", "walk", tampered, "sintético"),
                           "metrics.json")
    attributes = _category_windows()
    attributes = cast(pd.DataFrame, attributes[attributes["label"] == STANDING])
    predictions = cast(pd.DataFrame, windows[["video_id", "env", "subject", "k_end", "label", "pred_label"]])
    missing_row = _raises(ValueError, lambda: join_attributes(predictions, attributes.iloc[1:], "sintético"),
                          "não alinham")
    relabeled = predictions.assign(label=np.where(predictions["k_end"] == 3, WALK, predictions["label"]))
    label_mismatch = _raises(ValueError, lambda: join_attributes(relabeled, attributes, "sintético"), "não alinham")
    return _check(
        "o par conta erros, acertos, destinos e unidades por categoria, reproduz a linha de confusion_matrix, falha "
        "com matriz adulterada e recusa atributos sem predição ou com rótulo divergente",
        counted and empty_ok and tamper_fails and missing_row and label_mismatch,
    )


def _counts(transition: tuple[int, float], interior: tuple[int, float], absent: tuple[int, float] = (0, 0)) -> dict:
    strata = {TRANSITION: transition, INTERIOR: interior, POSE_ABSENT: absent,
              POSE_PRESENT: (transition[0] + interior[0], transition[1] + interior[1])}
    total = transition[0] + interior[0] + absent[0]
    errors = transition[1] + interior[1] + absent[1]
    return {"n_true": total, "n_error": errors,
            **{name: {"n_true": n_true, "n_error": n_error} for name, (n_true, n_error) in strata.items()}}


def _fold_unit(name: str, counts: dict) -> dict:
    return {"unit": name, "evidence": SUBJECT_FOLD, "models": [("A_seed42", counts)]}


def _selftest_rules_and_verdicts() -> bool:
    concentrated = _counts((20, 10), (20, 2))
    boundary = _counts((20, 6), (20, 4))
    diffuse = _counts((20, 2), (20, 2))
    sustained = evaluate_rule("R1", TRANSITION, INTERIOR, [_fold_unit(f"u{i}", concentrated) for i in range(3)])
    at_threshold = evaluate_rule("R1", TRANSITION, INTERIOR, [_fold_unit(f"u{i}", boundary) for i in range(3)])
    two_of_three = evaluate_rule("R1", TRANSITION, INTERIOR,
                                 [_fold_unit("u0", concentrated), _fold_unit("u1", concentrated),
                                  _fold_unit("u2", diffuse)])
    one_of_three = evaluate_rule("R1", TRANSITION, INTERIOR,
                                 [_fold_unit("u0", concentrated), _fold_unit("u1", diffuse),
                                  _fold_unit("u2", diffuse)])
    small = _counts((9, 9), (20, 0))
    not_evaluable = evaluate_rule("R1", TRANSITION, INTERIOR,
                                  [_fold_unit("u0", concentrated), _fold_unit("u1", concentrated),
                                   _fold_unit("u2", small)])

    def val_unit(meeting: int) -> dict:
        models = [(arm, concentrated if index < meeting else diffuse)
                  for index, arm in enumerate(("A", "B0", "B1", "C0", "C1"))]
        return {"unit": "official_val/subject_5", "evidence": OFFICIAL_VAL, "models": models}

    four_arms = evaluate_rule("R1", TRANSITION, INTERIOR, [val_unit(4)])["units"][0]
    three_arms = evaluate_rule("R1", TRANSITION, INTERIOR, [val_unit(3)])["units"][0]
    rules_ok = (
        sustained["status"] == SUSTAINED
        and at_threshold["status"] == SUSTAINED
        and two_of_three["status"] == SUSTAINED
        and one_of_three["status"] == NOT_SUSTAINED
        and not_evaluable["status"] == NOT_EVALUABLE and not_evaluable["n_eligible_units"] == 2
        and four_arms["met"] and four_arms["model_agreement"] == 0.8
        and not three_arms["met"]
    )

    units = [_fold_unit(f"u{i}", concentrated) for i in range(3)]
    common = {"n_windows": 100, "subjects": [0, 1, 6]}
    adequate = support_rule({**common, "share_of_labeled": 0.03}, units)
    rare = support_rule({**common, "share_of_labeled": 0.01}, units)
    few_units = support_rule({**common, "share_of_labeled": 0.03}, units[:2])
    support_ok = not adequate["limited"] and rare["limited"] and few_units["limited"]
    verdicts_ok = (
        pair_verdict(rare, sustained) == INSUFFICIENT_VERDICT
        and pair_verdict(adequate, sustained) == CONSIDER_VERDICT
        and pair_verdict(adequate, not_evaluable) == INSUFFICIENT_VERDICT
        and pair_verdict(adequate, one_of_three) == UNRESOLVED_VERDICT
        and overall_verdict([UNRESOLVED_VERDICT, CONSIDER_VERDICT]) == CONSIDER_VERDICT
        and overall_verdict([INSUFFICIENT_VERDICT, INSUFFICIENT_VERDICT]) == INSUFFICIENT_VERDICT
        and overall_verdict([INSUFFICIENT_VERDICT, UNRESOLVED_VERDICT]) == UNRESOLVED_VERDICT
    )
    return _check(
        "regras: diferença de taxa e fração de erros no limiar contam, 2/3 das unidades sustentam e 1/3 não, menos "
        "de três unidades elegíveis não avaliam, a unidade do val exige 80% dos braços, suporte limitado por fração "
        "no train ou por unidades, e vereditos por par e geral seguem a precedência declarada",
        rules_ok and support_ok and verdicts_ok,
    )


# Predições sintéticas do val: standing→walk e sitting→stand_up em janelas
# alternadas, deslocadas pela seed; as demais janelas são corretas.
def _val_prediction(label: int, k_end: int, seed: int) -> int:
    if label == STANDING and (k_end + seed) % 2 == 0:
        return WALK
    if label == SITTING and (k_end + seed) % 3 == 0:
        return STAND_UP
    return max(label, 0)


def _val_predictions(frames: pd.DataFrame, seed: int) -> Predictions:
    windows = build_window_index(cast(pd.DataFrame, frames[frames["split"] == "val"]), stride=EVAL_STRIDE,
                                 drop_ignored=False)
    video_ids = [str(video_id) for video_id in windows["video_id"]]
    k_ends = [int(k_end) for k_end in windows["k_end"]]
    labels = [int(label) for label in windows["label"]]
    return video_ids, k_ends, labels, [_val_prediction(label, k_end, seed)
                                       for label, k_end in zip(labels, k_ends)]


def _labeled_block(true_labels: list[int], pred_labels: list[int]) -> dict:
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(pred_labels, dtype=np.int64)
    labeled = y_true != IGNORE_LABEL
    return _final_split(y_true[labeled], y_pred[labeled])


def _write_run(run_dir: Path, arm: str, seed: int, frames: pd.DataFrame) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_key = "standardization_stats_path" if arm == "A" else "pose_standardization_stats_path"
    config = {"run_name": f"synthetic_{arm}_{seed}", "arm": arm, "seed": seed, stats_key: POSE_STATS["le2i"]}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint-{arm}-{seed}".encode())
    _video_ids, _k_ends, true_labels, pred_labels = _val_predictions(frames, seed)
    val = _labeled_block(true_labels, pred_labels)
    train_labels = build_window_index(official_train_frames(frames), stride=TRAIN_STRIDE)["label"].tolist()
    metrics = {
        "run_name": f"synthetic_{arm}_{seed}",
        "checkpoint_sha256": sha256_file(checkpoint),
        "final": {"train": _labeled_block(train_labels, train_labels), "val": val, "test": val},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


class Recorder:
    def __init__(self) -> None:
        self.splits: list[str] = []
        self.raw = RawRecordingPose()

    def load_evaluation(self, arm: str, dataset: str, run_dir: Path) -> EventEvaluation:
        frames = frames_table()
        seed = int(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["seed"])

        def evaluate_split(split: str) -> tuple[int, Predictions]:
            self.splits.append(split)
            predictions = _val_predictions(frames, seed)
            return len(predictions[0]), predictions

        return EventEvaluation(cast(EventRunConfig, None), lambda: (frames, evaluate_split))


def _diagnose(paths: tuple[Path, Path, Path, Path], runs: list[DiagnosisInput], output_dir: Path, recorder: Recorder,
              force: bool = False, fold_root: Path | None = None) -> bool:
    root, frames_path, manifest_path, pose_root = paths
    raw_loader = recorder.raw.inputs().raw_loader
    assert raw_loader is not None
    return run_diagnosis(runs, output_dir, fold_root=fold_root or root, frames_path=frames_path,
                         manifest_path=manifest_path, pose_root=pose_root, force=force, load=raw_loader,
                         pose=recorder.raw.inputs())


def _train_folds(paths: tuple[Path, Path, Path, Path]) -> None:
    root, frames_path, manifest_path, pose_root = paths
    run_prepare(root, frames_path, manifest_path, pose_root, False, RawRecordingPose().inputs())
    folds = [partition.fold for partition in build_subject_fold_partitions(official_train_frames())]
    run_train(root, frames_path, manifest_path, pose_root, folds, ["A"], [42], False, RawRecordingPose().inputs(),
              BASELINE_VARIANT)


def _report_checks(report: dict, windows: pd.DataFrame, summary: str, runs_root: Path, recorder: Recorder) -> bool:
    missing = report["missing_evidence"]
    folds = report["subject_folds"]["folds"]
    sources = {item["source"] for item in missing}
    read_videos = set(recorder.raw.raw_calls) | set(recorder.raw.calls)
    return (
        report["canonical"] is False
        and report["read_only"] is True
        and report["official_splits_read"] == ["train", "val"]
        and [(entry["arm"], entry["seed"]) for entry in report["official_val"]["runs"]] == [("A", 42), ("A", 43)]
        and report["official_val"]["arms"]["A"]["n_seeds"] == 2
        and all(entry["reconciliation"]["ok"] for entry in report["official_val"]["runs"])
        and len(folds) == 3
        and all(all(block["clean_reproduction"]["matches"].values()) for block in folds)
        and report["train_composition"]["support_reconciled_with"] == [entry["source"] for entry in
                                                                       report["official_val"]["runs"]]
        and recorder.splits == ["val", "val"]
        and not read_videos & TEST_VIDEOS
        and "EnvVal/v5" in recorder.raw.raw_calls
        and set(recorder.raw.calls) <= set(TRAIN_VIDEOS)
        and not set(windows["video_id"]) & TEST_VIDEOS
        and set(windows["evidence"]) == {OFFICIAL_VAL, SUBJECT_FOLD, OFFICIAL_TRAIN}
        and str(runs_root / "b0" / "checkpoint.pt") in next(item["missing_files"] for item in missing
                                                            if item["source"] == str(runs_root / "b0"))
        and {"B1", "C0", "C1", "A seed 44", "A seed 45", "A seed 46"} <= sources
        and report["decision"]["verdict"] in {CONSIDER_VERDICT, INSUFFICIENT_VERDICT, UNRESOLVED_VERDICT}
        and set(report["decision"]["pairs"]) == {"standing->walk", "sitting->stand_up"}
        and "não canônico" in summary and "## Evidência ausente" in summary
    )


def _selftest_end_to_end() -> bool:
    with synthetic_workspace() as paths, ExitStack() as stack:
        root, frames_path, _manifest_path, _pose_root = paths
        _train_folds(paths)
        recorder = Recorder()
        stack.enter_context(patch.object(standing_sitting_diagnosis, "load_evaluation", recorder.load_evaluation))
        stack.enter_context(patch.object(stratified_error_diagnosis, "_frames_path", lambda dataset: frames_path))
        workspace = root.parent
        frames = frames_table()
        runs_root = workspace / "runs"
        _write_run(runs_root / "a42", "A", 42, frames)
        _write_run(runs_root / "a43", "A", 43, frames)
        _write_run(runs_root / "b0", "B0", 42, frames)
        (runs_root / "b0" / "checkpoint.pt").unlink()
        runs = [DiagnosisInput("le2i", "A", runs_root / "a42"), DiagnosisInput("le2i", "A", runs_root / "a43"),
                DiagnosisInput("le2i", "B0", runs_root / "b0")]
        before = _tree_hashes(workspace)
        output_dir = workspace / "out"
        written = _diagnose(paths, runs, output_dir, recorder)
        report = json.loads((output_dir / REPORT_JSON_FILE).read_text(encoding="utf-8"))
        windows = pd.read_csv(output_dir / REPORT_WINDOWS_FILE)
        summary = (output_dir / REPORT_SUMMARY_FILE).read_text(encoding="utf-8")
        recorded = written and _report_checks(report, windows, summary, runs_root, recorder)
        unchanged = {path: digest for path, digest in _tree_hashes(workspace).items()
                     if not path.startswith("out")} == before

        _diagnose(paths, runs, workspace / "repeat", Recorder())
        deterministic = all((workspace / "repeat" / name).read_bytes() == (output_dir / name).read_bytes()
                            for name in (REPORT_JSON_FILE, REPORT_SUMMARY_FILE, REPORT_WINDOWS_FILE))
        previous = (output_dir / REPORT_JSON_FILE).read_bytes()
        skipped = not _diagnose(paths, runs[:1], output_dir, Recorder())
        preserved = (output_dir / REPORT_JSON_FILE).read_bytes() == previous
        forced = _diagnose(paths, runs[:1], output_dir, Recorder(), force=True)

        guards = all(
            _raises(ValueError, lambda target=target: _diagnose(paths, runs, target, Recorder()), text)
            for target, text in ((REFERENCE_RUN_ROOT / "stage6a", "sobrepõe"), (root / "diagnostics", "sobrepõe"),
                                 (runs_root / "a42" / "x", "sobrepõe"),
                                 (REPOSITORY_ROOT / "stage6a_selftest", "runs/local"))
        )
        cv_rejected = _raises(ValueError, lambda: _diagnose(
            paths, [DiagnosisInput("le2i-cv", "A", runs_root / "a42")], workspace / "cv", Recorder()), "Le2i-CS")
        shutil.copytree(runs_root / "a42", runs_root / "a42_copy")
        duplicate = _raises(ValueError, lambda: _diagnose(
            paths, [runs[0], DiagnosisInput("le2i", "A", runs_root / "a42_copy")], workspace / "dup", Recorder()),
            "mais de uma vez") and not (workspace / "dup").exists()

        metrics_path = runs_root / "a43" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        stored = metrics["final"]["val"]["confusion_matrix"]
        stored[STANDING][WALK], stored[STANDING][STANDING] = stored[STANDING][WALK] + 1, stored[STANDING][STANDING] - 1
        metrics["final"]["val"] = _final_split(*_matrix_labels(stored))
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        tamper_fails = _raises(ValueError, lambda: _diagnose(paths, runs[1:2], workspace / "tampered", Recorder()),
                               "diverge de metrics.json") and not (workspace / "tampered").exists()

        _diagnose(paths, runs[:1], workspace / "no_folds", Recorder(), fold_root=workspace / "absent_folds")
        no_folds = json.loads((workspace / "no_folds" / REPORT_JSON_FILE).read_text(encoding="utf-8"))
        folds_reported = (
            no_folds["subject_folds"]["folds"] == []
            and any(item["reason"] == "folds.json ausente" for item in no_folds["missing_evidence"])
        )
        pose_scope = {path.relative_to(workspace / "pose").with_suffix("").as_posix()
                      for path in pose_files(frames_path, workspace / "pose")} == {*TRAIN_VIDEOS, "EnvVal/v5"}
    return _check(
        "execução ponta a ponta com dobras sintéticas treinadas: grava JSON/resumo/janelas, reconcilia val, train e "
        "retidos, lê só train e val (nunca vídeos de test), registra evidência ausente sem inventar resultados, é "
        "determinística, não muda entradas, respeita --force, recusa saídas protegidas, Le2i-CV e duplicatas e não "
        "grava após divergência com metrics.json",
        recorded and unchanged and deterministic and skipped and preserved and forced and guards and cv_rejected
        and duplicate and tamper_fails and folds_reported and pose_scope,
    )


def _matrix_labels(matrix: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(true, pred) for true, row in enumerate(matrix) for pred, count in enumerate(row) for _ in range(count)]
    return np.array([true for true, _ in pairs]), np.array([pred for _, pred in pairs])


def run_selftest() -> bool:
    checks: list[Callable[[], bool]] = [
        _selftest_window_categories,
        _selftest_pair_breakdown,
        _selftest_rules_and_verdicts,
        _selftest_end_to_end,
    ]
    results = [check() for check in checks]
    print("\nstanding_sitting_diagnosis selftest " + ("OK: todas as checagens passaram" if all(results) else "FALHOU"))
    return all(results)
