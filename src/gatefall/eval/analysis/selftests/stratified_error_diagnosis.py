import json
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis import stratified_error_diagnosis
from gatefall.eval.analysis.classification_diagnosis import DiagnosisInput
from gatefall.eval.analysis.selftests.classification_diagnosis import POSE_STATS, _final_split
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.eval.analysis.stratified_error_diagnosis import (
    STRATIFIED_CSV_FILE,
    STRATIFIED_JSON_FILE,
    STRATIFIED_SUMMARY_FILE,
    analyze_split,
    reconcile,
    run_stratified_diagnosis,
)
from gatefall.eval.shared.event_artifacts import EventRunConfig
from gatefall.eval.shared.orchestration import EventEvaluation, Predictions
from gatefall.hashing import sha256_file
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT

# (video_id, split, env, subject, rótulo por quadro). O sujeito 1 aparece em
# EnvA e EnvB; o sujeito 3 em EnvA e EnvC. Rótulos: 0=walk, 1=fall, 2=fallen,
# 3=sit_down, 4=sitting, 7=stand_up, 8=standing, 9=other.
VIDEO_SPEC: tuple[tuple[str, str, str, int, list[int]], ...] = (
    ("a1", "test", "EnvA", 1, [0, 0, 1, 1, 2, IGNORE_LABEL]),
    ("a2", "test", "EnvA", 2, [0, 0, 0, 8, 8, 8]),
    ("b1", "test", "EnvB", 1, [IGNORE_LABEL, 9, 9, 9, 4, 4]),
    ("v1", "val", "EnvA", 3, [0, 0, 0, 3, 4, 4]),
    ("v2", "val", "EnvC", 3, [0, 0, 7, 8, 8, 8]),
)
# Erros conhecidos em test: fall→fallen e standing→walk em EnvA;
# sitting→other e sitting→fall em EnvB (fall não existe em EnvB). A janela
# IGNORE_LABEL de b1 prevê fall e não pode virar falso positivo.
PREDICTIONS: dict[tuple[str, int], int] = {
    ("a1", 2): 2,
    ("a2", 3): 0,
    ("b1", 4): 9,
    ("b1", 5): 1,
    ("b1", 0): 1,
    ("v2", 2): 8,
}

PredictionMutation = Callable[[Predictions], Predictions]


def _frames(spec: tuple[tuple[str, str, str, int, list[int]], ...] = VIDEO_SPEC) -> pd.DataFrame:
    rows = [
        {"video_id": video_id, "split": split, "env": env, "subject": subject,
         "frame_index": frame_index, "label": label}
        for video_id, split, env, subject, labels in spec
        for frame_index, label in enumerate(labels)
    ]
    return pd.DataFrame(rows)


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


def _evaluator(
    frames: pd.DataFrame, mutate: PredictionMutation | None = None, usable_offset: int = 0
) -> Callable[[str], tuple[int, Predictions]]:
    def evaluate_split(split: str) -> tuple[int, Predictions]:
        predictions = _predictions(frames, split)
        usable = len(predictions[0]) + usable_offset
        return usable, mutate(predictions) if mutate else predictions

    return evaluate_split


def _stored_block(frames: pd.DataFrame, split: str) -> dict:
    _video_ids, _k_ends, true_labels, pred_labels = _predictions(frames, split)
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(pred_labels, dtype=np.int64)
    labeled = y_true != IGNORE_LABEL
    return _final_split(y_true[labeled], y_pred[labeled])


def _analysis(split: str = "test") -> dict:
    frames = _frames()
    return analyze_split(frames, _evaluator(frames), split, _stored_block(frames, split), "sintético")


def _groups(analysis: dict, level: str) -> dict[str, dict]:
    return {group["group"]: group for group in analysis["levels"][level]}


def _raises(exception: type[BaseException], action: Callable[[], object], text: str = "") -> bool:
    try:
        action()
    except exception as exc:
        return text in str(exc)
    return False


def _selftest_grouping() -> bool:
    analysis = _analysis()
    environments = _groups(analysis, "environment")
    subjects = _groups(analysis, "environment_subject")
    videos = _groups(analysis, "video")
    env_a = np.asarray(environments["EnvA"]["confusion_matrix"])
    env_b = np.asarray(environments["EnvB"]["confusion_matrix"])
    a1_fall = subjects["EnvA/subject_1"]["per_class"]["fall"]
    return _check(
        "agrupa por ambiente, (ambiente, sujeito) e vídeo com suporte, matriz, erros e recall conhecidos",
        list(environments) == ["EnvA", "EnvB"]
        and list(subjects) == ["EnvA/subject_1", "EnvA/subject_2", "EnvB/subject_1"]
        and list(videos) == ["a1", "a2", "b1"]
        and environments["EnvA"]["n_labeled_windows"] == 11
        and environments["EnvA"]["n_videos"] == 2
        and environments["EnvB"]["n_labeled_windows"] == 5
        and env_a[1][2] == 1
        and env_a[8][0] == 1
        and env_b[4][9] == 1
        and env_b[4][1] == 1
        and environments["EnvA"]["n_errors"] == 2
        and environments["EnvB"]["n_errors"] == 2
        and a1_fall["support"] == 2
        and a1_fall["recall"] == 0.5
        and videos["b1"]["env"] == "EnvB"
        and videos["b1"]["subject"] == 1
        and environments["EnvB"]["dominant_confusions"][0]["true_label"] == "sitting"
        and analysis["reconciliation"]["ok"] is True,
    )


def _selftest_label_exclusion_and_restricted_macro() -> bool:
    analysis = _analysis()
    overall = analysis["overall"]
    env_b = _groups(analysis, "environment")["EnvB"]
    # EnvB: other com tp=3, fp=1 → F1 6/7; sitting e fall com F1 0; as cinco
    # restritas ausentes também entram com F1 0 na média de oito classes.
    expected_env_b_macro = (6 / 7) / 8
    return _check(
        "exclui IGNORE_LABEL das métricas e mantém a Macro-F1 restrita fixa de oito classes",
        overall["n_windows"] == 18
        and overall["n_ignored_windows"] == 2
        and overall["n_labeled_windows"] == 16
        and overall["per_class"]["fall"]["fp"] == 1
        and sum(sum(row) for row in overall["confusion_matrix"]) == 16
        and abs(env_b["macro_f1_restricted"] - expected_env_b_macro) < 1e-12
        and env_b["n_ignored_windows"] == 1,
    )


def _selftest_identity_alignment() -> bool:
    frames = _frames()
    stored = _stored_block(frames, "test")

    def fails(mutate: PredictionMutation | None, text: str, frames_override: pd.DataFrame | None = None,
              usable_offset: int = 0) -> bool:
        source = frames if frames_override is None else frames_override
        return _raises(
            ValueError,
            lambda: analyze_split(source, _evaluator(frames, mutate, usable_offset), "test", stored, "sintético"),
            text,
        )

    def drop_last(p: Predictions) -> Predictions:
        return p[0][:-1], p[1][:-1], p[2][:-1], p[3][:-1]

    def duplicate_first(p: Predictions) -> Predictions:
        return [p[0][0], *p[0][1:]], [p[1][0], p[1][0], *p[1][2:]], p[2], p[3]

    def extra_window(p: Predictions) -> Predictions:
        return [*p[0][:-1], "zz"], [*p[1][:-1], 0], p[2], p[3]

    def wrong_true_label(p: Predictions) -> Predictions:
        return p[0], p[1], [9, *p[2][1:]], p[3]

    def invalid_prediction(p: Predictions) -> Predictions:
        return p[0], p[1], p[2], [10, *p[3][1:]]

    def ragged(p: Predictions) -> Predictions:
        return p[0], p[1], p[2], p[3][:-1]

    ambiguous_env = frames.copy()
    ambiguous_env.loc[(ambiguous_env["video_id"] == "a1") & (ambiguous_env["frame_index"] == 3), "env"] = "EnvB"
    return _check(
        "recusa janela ausente, duplicada, extra, rótulo divergente, predição inválida, contagem divergente "
        "e vídeo com ambiente ambíguo",
        fails(drop_last, "sem predição: 1")
        and fails(duplicate_first, "duplicada")
        and fails(extra_window, "sem janela: 1")
        and fails(wrong_true_label, "rótulo verdadeiro")
        and fails(invalid_prediction, "fora de 0..9")
        and fails(ragged, "tamanhos diferentes")
        and fails(None, "avaliador relatou", usable_offset=1)
        and fails(None, "ambíguo", frames_override=ambiguous_env),
    )


def _selftest_aggregate_reconciliation() -> bool:
    frames = _frames()
    video_ids, k_ends, true_labels, pred_labels = _predictions(frames, "test")
    moved = list(pred_labels)
    moved[video_ids.index("a2") + 3] = 8
    y_true = np.asarray(true_labels, dtype=np.int64)
    labeled = y_true != IGNORE_LABEL
    stored_moved = _final_split(y_true[labeled], np.asarray(moved, dtype=np.int64)[labeled])
    stored_short = _final_split(y_true[labeled][1:], np.asarray(pred_labels, dtype=np.int64)[labeled][1:])

    analysis = _analysis()
    stored = _stored_block(frames, "test")
    stored_validated = {"confusion_matrix": stored["confusion_matrix"], "per_class": stored["per_class"],
                        "macro_f1_restricted": stored["macro_f1_restricted"]}
    dropped_group = {**analysis["levels"], "video": analysis["levels"]["video"][:-1]}
    corrupted_support = json.loads(json.dumps(analysis["levels"]))
    corrupted_support["environment"][0]["per_class"]["walk"]["support"] += 1
    return _check(
        "falha quando matriz, suporte ou soma de subgrupos divergem de metrics.json",
        _raises(ValueError, lambda: analyze_split(frames, _evaluator(frames), "test", stored_moved, "s"),
                "diverge de metrics.json")
        and _raises(ValueError, lambda: analyze_split(frames, _evaluator(frames), "test", stored_short, "s"),
                    "diverge de metrics.json")
        and _raises(ValueError, lambda: reconcile(dropped_group, analysis["overall"], stored_validated, "s"),
                    "por video")
        and _raises(ValueError, lambda: reconcile(corrupted_support, analysis["overall"], stored_validated, "s"),
                    "suportes por environment"),
    )


def _selftest_missing_classes() -> bool:
    analysis = _analysis()
    env_b = _groups(analysis, "environment")["EnvB"]
    a2 = _groups(analysis, "video")["a2"]
    return _check(
        "identifica classes ausentes por subgrupo e classes ausentes porém preditas",
        env_b["absent_restricted_labels"] == ["walk", "fall", "fallen", "sit_down", "stand_up", "standing"]
        and "lie_down" in env_b["absent_labels"]
        and env_b["predicted_absent_labels"] == ["fall"]
        and env_b["per_class"]["fall"]["f1"] == 0.0
        and a2["absent_restricted_labels"] == ["fall", "fallen", "sit_down", "sitting", "stand_up", "other"]
        and a2["predicted_absent_labels"] == [],
    )


def _selftest_ambiguous_subject_ids() -> bool:
    test = _analysis("test")
    val = _analysis("val")
    subjects = _groups(test, "environment_subject")
    val_subjects = _groups(val, "environment_subject")
    return _check(
        "trata o mesmo ID bruto de sujeito em ambientes diferentes como grupos distintos e o reporta",
        test["coverage"]["subject_ids_in_multiple_environments"] == {"1": ["EnvA", "EnvB"]}
        and subjects["EnvA/subject_1"]["n_labeled_windows"] == 5
        and subjects["EnvB/subject_1"]["n_labeled_windows"] == 5
        and val["coverage"]["subject_ids_in_multiple_environments"] == {"3": ["EnvA", "EnvC"]}
        and list(val_subjects) == ["EnvA/subject_3", "EnvC/subject_3"]
        and test["coverage"]["n_environment_subject_groups"] == 3,
    )


def _write_run(run_dir: Path, dataset: str, arm: str, frames: pd.DataFrame, with_checkpoint: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_key = "standardization_stats_path" if arm == "A" else "pose_standardization_stats_path"
    config = {"run_name": f"synthetic_{arm}", "arm": arm, "seed": 43, stats_key: POSE_STATS[dataset]}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint-{arm}".encode())
    metrics = {
        "run_name": f"synthetic_{arm}",
        "checkpoint_sha256": sha256_file(checkpoint),
        "final": {split: _stored_block(frames, split) for split in ("train", "val", "test")},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if not with_checkpoint:
        checkpoint.unlink()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _selftest_end_to_end_partial_and_read_only() -> bool:
    frames = _frames()
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        root = Path(tmp)
        frames_path = root / "data" / "frames.parquet"
        frames_path.parent.mkdir(parents=True)
        frames.to_parquet(frames_path)
        loaded: list[tuple[str, str]] = []

        def fake_load(arm: str, dataset: str, run_dir: Path) -> EventEvaluation:
            loaded.append((arm, dataset))
            return EventEvaluation(cast(EventRunConfig, None), lambda: (frames, _evaluator(frames)))

        stack.enter_context(patch.object(stratified_error_diagnosis, "load_evaluation", fake_load))
        stack.enter_context(patch.object(stratified_error_diagnosis, "_frames_path", lambda dataset: frames_path))

        runs_root = root / "runs"
        cs_a = runs_root / "le2i" / "a"
        cv_b0 = runs_root / "le2i_cv" / "b0"
        cs_c1 = runs_root / "le2i" / "c1"
        _write_run(cs_a, "le2i", "A", frames)
        _write_run(cv_b0, "le2i-cv", "B0", frames)
        _write_run(cs_c1, "le2i", "C1", frames, with_checkpoint=False)
        runs = [
            DiagnosisInput("le2i", "A", cs_a),
            DiagnosisInput("le2i-cv", "B0", cv_b0),
            DiagnosisInput("le2i", "C1", cs_c1),
        ]
        before = _tree_hashes(root)
        output_dir = root / "analysis"
        written = run_stratified_diagnosis(runs, output_dir)
        report = json.loads((output_dir / STRATIFIED_JSON_FILE).read_text(encoding="utf-8"))
        csv = pd.read_csv(output_dir / STRATIFIED_CSV_FILE)
        summary = (output_dir / STRATIFIED_SUMMARY_FILE).read_text(encoding="utf-8")
        inputs_untouched = {
            path: digest for path, digest in _tree_hashes(root).items() if not path.startswith("analysis")
        } == before

        protocols = report["protocols"]
        missing = {(item["dataset"], item["arm"]): item for item in report["missing_inputs"]}
        partial = (
            len(protocols["le2i"]["arms"]["A"]) == 1
            and len(protocols["le2i-cv"]["arms"]["B0"]) == 1
            and protocols["le2i"]["arms"]["C1"] == []
            and missing[("le2i", "C1")]["missing_files"] == [str(cs_c1 / "checkpoint.pt")]
            and missing[("le2i-cv", "A")]["reason"] == "nenhum run fornecido"
            and len(report["missing_inputs"]) == 8
            and sorted(loaded) == [("A", "le2i"), ("B0", "le2i-cv")]
        )
        entry = protocols["le2i"]["arms"]["A"][0]
        recorded = (
            report["canonical"] is False
            and entry["seed"] == 43
            and entry["checkpoint_sha256"] == sha256_file(cs_a / "checkpoint.pt")
            and str((cs_a / "checkpoint.pt").resolve()) in report["input_sha256"]
            and entry["splits"]["test"]["reconciliation"]["ok"] is True
            and set(csv["level"]) == {"split", "environment", "environment_subject", "video"}
            and "(não canônico)" in summary
            and "Hipóteses (não verificadas)" in summary
        )

        guarded = [cs_a, cs_a / "out", REFERENCE_RUN_ROOT / "stratified", REPOSITORY_ROOT / "src" / "x",
                   REPOSITORY_ROOT / "data" / "x"]
        guards = all(
            _raises(ValueError, lambda target=target: run_stratified_diagnosis(runs, target), "sobrepõe")
            for target in guarded
        )
        nothing_created = not (cs_a / "out").exists() and not (REFERENCE_RUN_ROOT / "stratified").exists()
        duplicate = _raises(ValueError, lambda: run_stratified_diagnosis([runs[0], runs[0]], root / "dup"),
                            "mais de uma vez")

        previous = (output_dir / STRATIFIED_CSV_FILE).read_text(encoding="utf-8")
        skipped = not run_stratified_diagnosis(runs[:1], output_dir)
        preserved = (output_dir / STRATIFIED_CSV_FILE).read_text(encoding="utf-8") == previous
        forced = run_stratified_diagnosis(runs[:1], output_dir, force=True)

        tampered_dir = runs_root / "le2i" / "tampered"
        _write_run(tampered_dir, "le2i", "B1", frames)
        metrics = json.loads((tampered_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics["final"]["test"] = _final_split(np.array([0, 1]), np.array([0, 1]))
        (tampered_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        tampered_output = root / "tampered_out"
        tamper_fails = _raises(
            ValueError,
            lambda: run_stratified_diagnosis([DiagnosisInput("le2i", "B1", tampered_dir)], tampered_output),
            "diverge de metrics.json",
        ) and not tampered_output.exists()
    return _check(
        "execução ponta a ponta registra runs parciais sem fabricar resultados, seed e hashes, não muda "
        "entradas, recusa saída protegida e duplicatas, respeita --force e não grava após divergência",
        written and inputs_untouched and partial and recorded and guards and nothing_created and duplicate
        and skipped and preserved and forced and tamper_fails,
    )


def run_stratified_error_diagnosis_selftest() -> bool:
    checks = [
        _selftest_grouping(),
        _selftest_label_exclusion_and_restricted_macro(),
        _selftest_identity_alignment(),
        _selftest_aggregate_reconciliation(),
        _selftest_missing_classes(),
        _selftest_ambiguous_subject_ids(),
        _selftest_end_to_end_partial_and_read_only(),
    ]
    ok = all(checks)
    if ok:
        print("\nstratified_error_diagnosis selftest OK: todas as checagens passaram")
    else:
        print("\nstratified_error_diagnosis selftest FALHOU", file=sys.stderr)
    return ok


def run_selftest() -> None:
    if not run_stratified_error_diagnosis_selftest():
        sys.exit(1)
