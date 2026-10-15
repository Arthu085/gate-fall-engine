import json
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import cast
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import yaml

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis import stratified_error_diagnosis, window_context_diagnosis
from gatefall.eval.analysis.classification_diagnosis import DiagnosisInput
from gatefall.eval.analysis.selftests.classification_diagnosis import POSE_STATS, _final_split
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.eval.analysis.selftests.stratified_error_diagnosis import _raises, _tree_hashes
from gatefall.eval.analysis.stratified_error_diagnosis import analyze_split_with_table
from gatefall.eval.analysis.window_context_diagnosis import (
    CORRECT,
    ERROR,
    WINDOW_CONTEXT_CSV_FILE,
    WINDOW_CONTEXT_JSON_FILE,
    WINDOW_CONTEXT_SUMMARY_FILE,
    WINDOW_CONTEXT_WINDOWS_FILE,
    _with_dimensions,
    check_conservation,
    join_context,
    pair_analysis,
    person_found_loader,
    run_window_context_diagnosis,
    split_context,
    video_context,
)
from gatefall.eval.shared.event_artifacts import EventRunConfig
from gatefall.eval.shared.orchestration import EventEvaluation, Predictions
from gatefall.hashing import sha256_file
from gatefall.runs import REFERENCE_RUN_ROOT

OTHER, WALK, FALL, SITTING, STAND_UP, STANDING = 9, 0, 1, 4, 7, 8

# (video_id, split, env, subject, rótulo por quadro). Em EnvA/v1, o primeiro
# trecho `other` começa no quadro 0 (janelas puras com replicação de borda) e
# o segundo começa no quadro 40, depois de `walk` (janelas 40..62
# misturadas, 63..69 puras). EnvB/v2 tem um trecho IGNORE_LABEL entre `walk`
# e `sitting`.
VIDEO_SPEC: tuple[tuple[str, str, str, int, list[int]], ...] = (
    ("EnvA/v1", "test", "EnvA", 1, [OTHER] * 10 + [WALK] * 30 + [OTHER] * 30),
    ("EnvB/v2", "test", "EnvB", 2,
     [STANDING] * 5 + [WALK] * 20 + [IGNORE_LABEL] * 5 + [SITTING] * 10 + [STAND_UP] * 5),
    ("EnvA/v3", "val", "EnvA", 3, [WALK] * 10 + [OTHER] * 30 + [SITTING] * 20),
)
POSE_MISSING_FRAMES: dict[str, set[int]] = {"EnvA/v1": {0, 1, 2, 41}}
# other→walk em test: 1 e 3 (puras, com borda), 40..45 (misturadas), 65 e 66
# (puras, meio do trecho); other→fall em 5. A janela IGNORE_LABEL 27 de v2
# prevê walk e não pode entrar em nenhuma contagem.
PREDICTIONS: dict[tuple[str, int], int] = {
    ("EnvA/v1", 1): WALK,
    ("EnvA/v1", 3): WALK,
    ("EnvA/v1", 5): FALL,
    **{("EnvA/v1", k_end): WALK for k_end in range(40, 46)},
    ("EnvA/v1", 65): WALK,
    ("EnvA/v1", 66): WALK,
    ("EnvB/v2", 2): WALK,
    ("EnvB/v2", 27): WALK,
    ("EnvB/v2", 32): STAND_UP,
    ("EnvA/v3", 12): WALK,
}

PredictionMutation = Callable[[Predictions], Predictions]


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": video_id, "split": split, "env": env, "subject": subject,
             "frame_index": frame_index, "label": label}
            for video_id, split, env, subject, labels in VIDEO_SPEC
            for frame_index, label in enumerate(labels)
        ]
    )


def _person_found(video_id: str) -> np.ndarray:
    n_frames = next(len(labels) for spec_id, _split, _env, _subject, labels in VIDEO_SPEC if spec_id == video_id)
    person_found = np.ones(n_frames, dtype=bool)
    person_found[sorted(POSE_MISSING_FRAMES.get(video_id, set()))] = False
    return person_found


def _predictions(frames: pd.DataFrame, split: str) -> Predictions:
    windows = build_window_index(
        cast(pd.DataFrame, frames[frames["split"] == split]), stride=EVAL_STRIDE, drop_ignored=False
    )
    video_ids = [str(video_id) for video_id in windows["video_id"]]
    k_ends = [int(k_end) for k_end in windows["k_end"]]
    true_labels = [int(label) for label in windows["label"]]
    pred_labels = [
        PREDICTIONS.get((video_id, k_end), max(label, 0))
        for video_id, k_end, label in zip(video_ids, k_ends, true_labels)
    ]
    return video_ids, k_ends, true_labels, pred_labels


def _evaluator(frames: pd.DataFrame, mutate: PredictionMutation | None = None) -> Callable[[str], tuple[int, Predictions]]:
    def evaluate_split(split: str) -> tuple[int, Predictions]:
        predictions = _predictions(frames, split)
        return len(predictions[0]), mutate(predictions) if mutate else predictions

    return evaluate_split


def _stored_block(frames: pd.DataFrame, split: str) -> dict:
    _video_ids, _k_ends, true_labels, pred_labels = _predictions(frames, split)
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(pred_labels, dtype=np.int64)
    labeled = y_true != IGNORE_LABEL
    return _final_split(y_true[labeled], y_pred[labeled])


def _labeled(split: str = "test") -> tuple[pd.DataFrame, np.ndarray]:
    frames = _frames()
    stored = _stored_block(frames, split)
    table, _analysis = analyze_split_with_table(frames, _evaluator(frames), split, stored, "sintético")
    windows = join_context(table, split_context(frames, split, _person_found), "sintético")
    labeled = _with_dimensions(cast(pd.DataFrame, windows[windows["label"] != IGNORE_LABEL]))
    return labeled, np.asarray(stored["confusion_matrix"], dtype=np.int64)


def _row(rows: list[dict], stratum: str) -> dict:
    return next(row for row in rows if row["stratum"] == stratum)


def _selftest_context_features() -> bool:
    labels = np.array([OTHER] * 5 + [WALK] * 30 + [IGNORE_LABEL] * 3 + [OTHER] * 10, dtype=np.int64)
    person_found = np.ones(len(labels), dtype=bool)
    person_found[:3] = False
    context = video_context(labels, person_found).set_index("k_end")

    def at(k_end: int) -> pd.Series:
        return cast(pd.Series, context.loc[k_end])

    first, boundary, near_boundary, inside, last_walk, after_ignore, ignored = (
        at(0), at(5), at(27), at(28), at(34), at(38), at(36)
    )
    return _check(
        "deriva pureza, fração do rótulo final com quadros replicados, fronteiras, trecho anterior, borda e pose",
        len(context) == len(labels)
        and first["purity"] == "pure" and first["n_final_label_frames"] == 24
        and first["n_replicated_frames"] == 23 and first["edge_padding"] == "padded"
        and first["preceding_label"] == "video_start" and first["frames_to_segment_end"] == 4
        and first["pose_final"] == "missing" and at(3)["pose_final"] == "detected"
        # k_end=5: 19 cópias do quadro 0 e os quadros 1..4 são `other`; só o 5 é `walk`.
        and boundary["n_final_label_frames"] == 1 and boundary["purity"] == "mixed"
        and boundary["n_replicated_frames"] == 18 and boundary["preceding_label"] == "other"
        and boundary["frames_since_segment_start"] == 0 and boundary["start_distance"] == "00-05"
        and near_boundary["n_final_label_frames"] == 23
        and abs(near_boundary["final_label_fraction"] - 23 / 24) < 1e-12
        and near_boundary["start_distance"] == "12-22" and near_boundary["edge_padding"] == "unpadded"
        and inside["purity"] == "pure" and inside["start_distance"] == "23+"
        and last_walk["frames_to_segment_end"] == 0 and last_walk["segment_length"] == 30
        and last_walk["end_distance"] == "00-05"
        and after_ignore["n_final_label_frames"] == 1 and after_ignore["n_ignored_frames"] == 3
        and after_ignore["preceding_label"] == "IGNORE_LABEL" and after_ignore["purity"] == "mixed"
        and ignored["preceding_label"] == "walk" and ignored["n_final_label_frames"] == 2,
    )


def _selftest_pair_strata_and_ignored_labels() -> bool:
    labeled, matrix = _labeled()
    pair = pair_analysis(labeled, "other", "walk", matrix, "sintético")
    strata = pair["strata"]
    control = pair_analysis(labeled, "sitting", "stand_up", matrix, "sintético")
    sitting_after_ignore = labeled[(labeled["video_id"] == "EnvB/v2") & (labeled["k_end"] == 30)].iloc[0]
    return _check(
        "separa other→walk de other correto por pureza, borda, pose, fronteira, trecho anterior e ambiente, "
        "sem contar janelas IGNORE_LABEL",
        pair["counts"] == {"n_true": 40, "n_error": 10, "n_correct": 29, "n_other_outcome": 1}
        and _row(strata["purity"], "mixed")["n_true"] == 23
        and _row(strata["purity"], "mixed")["n_error"] == 6
        and _row(strata["purity"], "pure")["n_error"] == 4
        and abs(_row(strata["purity"], "mixed")["error_rate"] - 6 / 23) < 1e-12
        and abs(pair["profiles"][ERROR]["mixed_share"] - 0.6) < 1e-12
        and abs(pair["profiles"][CORRECT]["mixed_share"] - 17 / 29) < 1e-12
        and abs(pair["profiles"][ERROR]["edge_padded_share"] - 0.2) < 1e-12
        and abs(pair["profiles"][ERROR]["pose_missing_share"] - 0.2) < 1e-12
        and _row(strata["start_distance"], "00-05")["n_error"] == 8
        and _row(strata["start_distance"], "23+")["n_error"] == 2
        and _row(strata["preceding_label"], "video_start")["n_true"] == 10
        and _row(strata["preceding_label"], "walk")["n_error"] == 8
        and [row["stratum"] for row in strata["environment_purity"]] == ["EnvA/mixed", "EnvA/pure"]
        and [row["stratum"] for row in strata["video"]] == ["EnvA/v1"]
        and control["counts"]["n_error"] == 1
        and _row(control["strata"]["preceding_label"], "IGNORE_LABEL")["n_true"] == 10
        and sitting_after_ignore["purity"] == "mixed" and sitting_after_ignore["n_ignored_frames"] == 5
        and int((labeled["label"] == IGNORE_LABEL).sum()) == 0
        and len(labeled) == 115 - 5,
    )


def _selftest_alignment_failures() -> bool:
    frames = _frames()
    stored = _stored_block(frames, "test")
    table, _analysis = analyze_split_with_table(frames, _evaluator(frames), "test", stored, "s")
    context = split_context(frames, "test", _person_found)

    def drop_last(p: Predictions) -> Predictions:
        return p[0][:-1], p[1][:-1], p[2][:-1], p[3][:-1]

    with tempfile.TemporaryDirectory() as tmp:
        pose_root = Path(tmp)
        (pose_root / "EnvA").mkdir()
        with h5py.File(pose_root / "EnvA" / "v1.h5", "w") as h5_file:
            h5_file.create_dataset("person_found", data=np.ones(69, dtype=bool))
        short_pose = person_found_loader(pose_root)
        pose_mismatch = _raises(
            ValueError,
            lambda: split_context(cast(pd.DataFrame, frames[frames["video_id"] == "EnvA/v1"]), "test", short_pose),
            "person_found",
        )
    return _check(
        "recusa contexto sem predição, predição sem contexto, pose com comprimento divergente e predições "
        "desalinhadas do índice de janelas",
        _raises(ValueError, lambda: join_context(table, context.iloc[1:], "s"), "não alinha")
        and _raises(ValueError, lambda: join_context(table.iloc[1:], context, "s"), "não alinha")
        and pose_mismatch
        and _raises(
            ValueError,
            lambda: analyze_split_with_table(frames, _evaluator(frames, drop_last), "test", stored, "s"),
            "não alinham",
        ),
    )


def _selftest_conservation() -> bool:
    labeled, matrix = _labeled()
    pair = pair_analysis(labeled, "other", "walk", matrix, "sintético")
    corrupted = json.loads(json.dumps(pair["strata"]))
    corrupted["purity"][0]["n_error"] += 1
    shifted = matrix.copy()
    shifted[OTHER][WALK] -= 1
    shifted[OTHER][OTHER] += 1
    sums_match = all(
        sum(row[field] for row in rows) == pair["counts"][field]
        for rows in pair["strata"].values()
        for field in ("n_true", "n_error", "n_correct", "n_other_outcome")
    )
    return _check(
        "conserva as contagens do par em todo estrato e falha quando estrato ou matriz divergem",
        pair["conservation"]["ok"] is True
        and pair["conservation"]["n_error"] == int(matrix[OTHER][WALK])
        and sums_match
        and _raises(
            ValueError,
            lambda: check_conservation(pair["counts"], corrupted, matrix, OTHER, WALK, "s"),
            "soma de n_error por purity",
        )
        and _raises(
            ValueError,
            lambda: check_conservation(pair["counts"], pair["strata"], shifted, OTHER, WALK, "s"),
            "divergem de metrics.json",
        ),
    )


def _write_run(run_dir: Path, arm: str, seed: int, frames: pd.DataFrame, with_checkpoint: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_key = "standardization_stats_path" if arm == "A" else "pose_standardization_stats_path"
    config = {"run_name": f"synthetic_{arm}_{seed}", "arm": arm, "seed": seed, stats_key: POSE_STATS["le2i"]}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint-{arm}-{seed}".encode())
    final = {split: _stored_block(frames, split) for split in ("val", "test")}
    metrics = {
        "run_name": f"synthetic_{arm}_{seed}",
        "checkpoint_sha256": sha256_file(checkpoint),
        "final": {"train": final["val"], **final},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if not with_checkpoint:
        checkpoint.unlink()


def _write_pose(pose_root: Path, frames: pd.DataFrame) -> None:
    for video_id in frames["video_id"].unique():
        env, _, name = str(video_id).partition("/")
        (pose_root / env).mkdir(parents=True, exist_ok=True)
        with h5py.File(pose_root / env / f"{name}.h5", "w") as h5_file:
            h5_file.create_dataset("person_found", data=_person_found(str(video_id)))


def _selftest_end_to_end() -> bool:
    frames = _frames()
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        root = Path(tmp)
        frames_path = root / "data" / "frames.parquet"
        frames_path.parent.mkdir(parents=True)
        frames.to_parquet(frames_path)
        pose_root = root / "data" / "pose"
        _write_pose(pose_root, frames)

        def fake_load(arm: str, dataset: str, run_dir: Path) -> EventEvaluation:
            return EventEvaluation(cast(EventRunConfig, None), lambda: (frames, _evaluator(frames)))

        stack.enter_context(patch.object(window_context_diagnosis, "load_evaluation", fake_load))
        stack.enter_context(patch.object(window_context_diagnosis, "_frames_path", lambda dataset: frames_path))
        stack.enter_context(patch.object(window_context_diagnosis, "_pose_root", lambda dataset: pose_root))
        stack.enter_context(patch.object(stratified_error_diagnosis, "_frames_path", lambda dataset: frames_path))

        runs_root = root / "runs"
        _write_run(runs_root / "a42", "A", 42, frames)
        _write_run(runs_root / "a43", "A", 43, frames)
        _write_run(runs_root / "b0", "B0", 42, frames, with_checkpoint=False)
        runs = [
            DiagnosisInput("le2i", "A", runs_root / "a42"),
            DiagnosisInput("le2i", "A", runs_root / "a43"),
            DiagnosisInput("le2i", "B0", runs_root / "b0"),
        ]
        before = _tree_hashes(root)
        output_dir = root / "analysis"
        written = run_window_context_diagnosis(runs, output_dir)
        report = json.loads((output_dir / WINDOW_CONTEXT_JSON_FILE).read_text(encoding="utf-8"))
        csv = pd.read_csv(output_dir / WINDOW_CONTEXT_CSV_FILE)
        windows = pd.read_csv(output_dir / WINDOW_CONTEXT_WINDOWS_FILE)
        summary = (output_dir / WINDOW_CONTEXT_SUMMARY_FILE).read_text(encoding="utf-8")
        inputs_untouched = {
            path: digest for path, digest in _tree_hashes(root).items() if not path.startswith("analysis")
        } == before

        missing = report["missing_inputs"]
        aggregate = report["arms"]["A"]["splits"]["test"]["other->walk"]
        consistency = report["cross_run_consistency"]["test"]["other->walk"]
        recorded = (
            written
            and report["canonical"] is False
            and [entry["seed"] for entry in report["runs"]] == [42, 43]
            and report["arms"]["A"]["n_seeds"] == 2
            and aggregate["counts"]["n_error"]["mean"] == 10
            and aggregate["counts"]["n_error"]["std"] == 0
            and consistency["error_mixed_share_above_correct"] == 2
            and consistency["mixed_error_rate_above_pure"] == 2
            and report["runs"][0]["splits"]["val"]["pairs"]["other->walk"]["counts"]["n_error"] == 1
            and any(path.endswith("EnvA/v1.h5") for path in report["input_sha256"])
            and any(item["arm"] == "B0" and item["missing_files"] == [str(runs_root / "b0" / "checkpoint.pt")]
                    for item in missing)
            and {item["seed"] for item in missing if item["arm"] == "A"} == {44, 45, 46}
            and any(item["arm"] == "B1" and item["reason"] == "nenhum run fornecido" for item in missing)
            and list(windows.columns[-2:]) == ["pred_A_seed42", "pred_A_seed43"]
            and len(windows) == 110 + 60
            and "IGNORE_LABEL" not in set(windows["true_label"])
            and set(csv["record"]) == {"run", "arm_seed_aggregate"}
            and "não canônico" in summary
            and "## val" in summary and "## test" in summary
        )

        guards = _raises(ValueError, lambda: run_window_context_diagnosis(runs, REFERENCE_RUN_ROOT / "ctx"), "sobrepõe")
        guards = guards and _raises(ValueError, lambda: run_window_context_diagnosis(runs, runs_root / "a42" / "x"),
                                    "sobrepõe")
        _write_run(runs_root / "a42_copy", "A", 42, frames)
        duplicate = _raises(
            ValueError,
            lambda: run_window_context_diagnosis(
                [runs[0], DiagnosisInput("le2i", "A", runs_root / "a42_copy")], root / "dup"
            ),
            "mais de uma vez",
        ) and not (root / "dup").exists()
        cv_rejected = _raises(
            ValueError,
            lambda: run_window_context_diagnosis([DiagnosisInput("le2i-cv", "A", runs_root / "a42")], root / "cv"),
            "Le2i-CS",
        )

        previous = (output_dir / WINDOW_CONTEXT_WINDOWS_FILE).read_text(encoding="utf-8")
        skipped = not run_window_context_diagnosis(runs[:1], output_dir)
        preserved = (output_dir / WINDOW_CONTEXT_WINDOWS_FILE).read_text(encoding="utf-8") == previous
        forced = run_window_context_diagnosis(runs[:1], output_dir, force=True)

        tampered = runs_root / "tampered"
        _write_run(tampered, "A", 44, frames)
        metrics = json.loads((tampered / "metrics.json").read_text(encoding="utf-8"))
        metrics["final"]["test"] = _final_split(np.array([0, 1]), np.array([0, 1]))
        (tampered / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        tamper_fails = _raises(
            ValueError,
            lambda: run_window_context_diagnosis([DiagnosisInput("le2i", "A", tampered)], root / "tampered_out"),
            "diverge de metrics.json",
        ) and not (root / "tampered_out").exists()

        (pose_root / "EnvA" / "v3.h5").unlink()
        run_window_context_diagnosis(runs[:1], root / "no_pose")
        no_pose = json.loads((root / "no_pose" / WINDOW_CONTEXT_JSON_FILE).read_text(encoding="utf-8"))
        pose_reported = (
            no_pose["runs"] == []
            and no_pose["arms"] == {}
            and str(pose_root / "EnvA" / "v3.h5") in no_pose["missing_inputs"][0]["missing_files"]
        )
    return _check(
        "execução ponta a ponta grava JSON/CSV/janelas/resumo, registra runs, seeds e pose ausentes sem "
        "inventar resultados, não muda entradas, recusa saída protegida, duplicatas e Le2i-CV, respeita "
        "--force e não grava após divergência com metrics.json",
        recorded and inputs_untouched and guards and duplicate and cv_rejected and skipped and preserved
        and forced and tamper_fails and pose_reported,
    )


def run_window_context_diagnosis_selftest() -> bool:
    checks = [
        _selftest_context_features(),
        _selftest_pair_strata_and_ignored_labels(),
        _selftest_alignment_failures(),
        _selftest_conservation(),
        _selftest_end_to_end(),
    ]
    ok = all(checks)
    if ok:
        print("\nwindow_context_diagnosis selftest OK: todas as checagens passaram")
    else:
        print("\nwindow_context_diagnosis selftest FALHOU", file=sys.stderr)
    return ok


def run_selftest() -> None:
    if not run_window_context_diagnosis_selftest():
        sys.exit(1)
