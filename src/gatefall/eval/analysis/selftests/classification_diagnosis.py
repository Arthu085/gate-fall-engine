import copy
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis.classification_diagnosis import (
    CANONICAL_REFERENCE,
    DIAGNOSIS_CSV_FILE,
    DIAGNOSIS_JSON_FILE,
    DIAGNOSIS_SUMMARY_FILE,
    LOCAL_RUN,
    MULTISEED_SUMMARY,
    DiagnosisInput,
    build_diagnosis,
    diagnose_multiseed_summary,
    diagnose_run,
    diagnose_split,
    dominant_confusions,
    labels_from_confusion_matrix,
    run_diagnosis,
    snapshot_inputs,
    verify_inputs_unchanged,
    write_outputs,
)
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.runs import REFERENCE_RUN_ROOT
from gatefall.train.shared.metrics import (
    RESTRICTED_CLASSES,
    classification_summary,
    confusion_matrix,
    restricted_macro_f1,
    support_by_name,
)

POSE_STATS = {
    "le2i": "src/gatefall/features/stats/pose_le2i_cs.json",
    "le2i-cv": "src/gatefall/features/stats/pose_le2i_cv.json",
}


def _known_matrix() -> list[list[int]]:
    # walk 8 corretas + 2→standing; fall 3 + 1→fallen; standing 1 + 3→walk;
    # demais restritas 5 corretas; lie_down/lying sem suporte.
    matrix = [[0] * 10 for _ in range(10)]
    matrix[0][0], matrix[0][8] = 8, 2
    matrix[1][1], matrix[1][2] = 3, 1
    matrix[2][2] = 4
    matrix[8][8], matrix[8][0] = 1, 3
    for class_id in (3, 4, 7, 9):
        matrix[class_id][class_id] = 5
    return matrix


def _final_split(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    summary = classification_summary(y_true, y_pred, LE2I_LABEL_NAMES)
    macro_f1, f1_by_class = restricted_macro_f1(y_true, y_pred)
    return {
        "macro_f1_restricted": macro_f1,
        "f1_by_class": {str(class_id): value for class_id, value in f1_by_class.items()},
        "support": support_by_name(y_true, LE2I_LABEL_NAMES),
        "confusion_matrix": summary["confusion_matrix"],
        "per_class": summary["per_class"],
    }


def _final_block(error_offset: int) -> dict:
    final: dict[str, dict] = {}
    for split_offset, split in enumerate(("train", "val", "test")):
        matrix = copy.deepcopy(_known_matrix())
        matrix[9][9] += 10 - 3 * split_offset
        matrix[9][4] += split_offset + error_offset
        final[split] = _final_split(*labels_from_confusion_matrix(matrix))
    return final


def _write_run(run_dir: Path, dataset: str, arm: str, seed: int, error_offset: int = 0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_key = "standardization_stats_path" if arm == "A" else "pose_standardization_stats_path"
    config = {"run_name": f"synthetic_{arm}", "arm": arm, "seed": seed, stats_key: POSE_STATS[dataset]}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    metrics = {
        "run_name": f"synthetic_{arm}",
        "final": _final_block(error_offset),
        "checkpoint_sha256": f"checkpoint-{arm}-{seed}-{error_offset}",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def _summary_payload(arm: str, seeds: list[int], run_dir_prefix: str) -> dict:
    seed_reports = [
        {
            "seed": seed,
            "run_dir": f"{run_dir_prefix}/seed_{seed}",
            "checkpoint_sha256": f"checkpoint-{arm}-{seed}-{seed - 42}",
            "classification": _final_block(seed - 42),
        }
        for seed in seeds
    ]
    aggregate = {
        split: {
            "macro_f1_restricted": {
                "n": len(seeds),
                "mean": float(
                    np.mean([report["classification"][split]["macro_f1_restricted"] for report in seed_reports])
                ),
            }
        }
        for split in ("train", "val", "test")
    }
    return {
        "arm": arm,
        "config_fingerprint_sha256": "synthetic",
        "n_seeds": len(seeds),
        "seeds": seed_reports,
        "aggregate": {"classification": aggregate},
    }


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _raises(exception: type[BaseException], action: Callable[[], object]) -> bool:
    try:
        action()
    except exception:
        return True
    return False


def _selftest_metrics_match_independent_formula() -> bool:
    matrix = _known_matrix()
    diagnosis = diagnose_split(_final_split(*labels_from_confusion_matrix(matrix)), LE2I_LABEL_NAMES, "sintético")
    counts = np.asarray(matrix)
    expected_f1: dict[int, float] = {}
    ok = True
    for class_id in range(10):
        tp = counts[class_id, class_id]
        precision = tp / counts[:, class_id].sum() if counts[:, class_id].sum() else 0.0
        recall = tp / counts[class_id].sum() if counts[class_id].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        expected_f1[class_id] = f1
        row = diagnosis["per_class"][LE2I_LABEL_NAMES[class_id]]
        ok = ok and abs(row["precision"] - precision) < 1e-12 and abs(row["recall"] - recall) < 1e-12
        ok = ok and abs(row["f1"] - f1) < 1e-12 and row["support"] == counts[class_id].sum()
    expected_macro = float(np.mean([expected_f1[class_id] for class_id in RESTRICTED_CLASSES]))
    deficit = diagnosis["macro_f1_deficit"]
    deficit_sum = sum(item["deficit"] for item in deficit)
    walk = diagnosis["per_class"]["walk"]
    return _check(
        "precision/recall/F1/suporte e Macro-F1 restrita reproduzem a fórmula independente; "
        "déficit soma 1 − Macro-F1 e exclui lie_down/lying",
        ok
        and abs(walk["precision"] - 8 / 11) < 1e-12
        and abs(walk["recall"] - 0.8) < 1e-12
        and abs(diagnosis["macro_f1_restricted"] - expected_macro) < 1e-12
        and abs(deficit_sum - (1 - expected_macro)) < 1e-12
        and {item["label"] for item in deficit}.isdisjoint({"lie_down", "lying"})
        and deficit[0]["label"] == "standing"
        and not diagnosis["per_class"]["lie_down"]["included_in_macro_f1"],
    )


def _selftest_confusion_interpretation() -> bool:
    matrix = _known_matrix()
    y_true, y_pred = labels_from_confusion_matrix(matrix)
    round_trip = confusion_matrix(y_true, y_pred).tolist() == matrix
    confusions = dominant_confusions(matrix, LE2I_LABEL_NAMES)
    pairs = [(item["true_label"], item["predicted_label"], item["count"]) for item in confusions]
    first = confusions[0]
    tie_matrix = [[0] * 10 for _ in range(10)]
    tie_matrix[4][3] = tie_matrix[2][1] = 2
    tie_pairs = [(item["true_id"], item["predicted_id"]) for item in dominant_confusions(tie_matrix, LE2I_LABEL_NAMES)]
    return _check(
        "confusões dominantes ignoram a diagonal, ordenam por contagem e desempatam por classe; "
        "frações por suporte verdadeiro, predito e total de erros",
        round_trip
        and pairs == [("standing", "walk", 3), ("walk", "standing", 2), ("fall", "fallen", 1)]
        and abs(first["share_of_true_support"] - 0.75) < 1e-12
        and abs(first["share_of_predicted"] - 3 / 11) < 1e-12
        and abs(first["share_of_all_errors"] - 0.5) < 1e-12
        and tie_pairs == [(2, 1), (4, 3)]
        and dominant_confusions([[0] * 10 for _ in range(10)], LE2I_LABEL_NAMES) == [],
    )


def _selftest_invalid_inputs_rejected() -> bool:
    valid = _final_split(*labels_from_confusion_matrix(_known_matrix()))

    def mutated(change: Callable[[dict], None]) -> dict:
        block = copy.deepcopy(valid)
        change(block)
        return block

    bad_blocks = [
        mutated(lambda block: block["confusion_matrix"].pop()),
        mutated(lambda block: block["confusion_matrix"][0].__setitem__(1, -1)),
        mutated(lambda block: block["confusion_matrix"][0].__setitem__(1, 1.5)),
        mutated(lambda block: block.__setitem__("macro_f1_restricted", block["macro_f1_restricted"] + 0.01)),
        mutated(lambda block: block["per_class"]["fall"].__setitem__("recall", 0.0)),
        mutated(lambda block: block["support"].__setitem__("walk", 1)),
        mutated(lambda block: block.pop("per_class")),
    ]
    blocks_rejected = all(
        _raises(ValueError, lambda block=block: diagnose_split(block, LE2I_LABEL_NAMES, "sintético"))
        for block in bad_blocks
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_run(root / "a", "le2i", "A", 42)
        wrong_arm = _raises(ValueError, lambda: diagnose_run(root / "a", "le2i", "B0", LOCAL_RUN, LE2I_LABEL_NAMES))
        wrong_protocol = _raises(
            ValueError, lambda: diagnose_run(root / "a", "le2i-cv", "A", LOCAL_RUN, LE2I_LABEL_NAMES)
        )
        reference_as_run = _raises(
            ValueError,
            lambda: build_diagnosis(
                [DiagnosisInput("le2i", "A", REFERENCE_RUN_ROOT / "le2i/baseline_a")], [], None
            ),
        )

        summary_path = root / "summary" / "multiseed_summary.json"
        payload = _summary_payload("B1", [42, 43], str(root / "seeds"))

        def summary_rejected(change: Callable[[dict], None], dataset: str = "le2i") -> bool:
            broken = copy.deepcopy(payload)
            change(broken)
            _write_summary(summary_path, broken)
            return _raises(
                ValueError, lambda: diagnose_multiseed_summary(summary_path, dataset, "B1", LE2I_LABEL_NAMES)
            )

        summary_checks = [
            summary_rejected(lambda broken: broken.__setitem__("arm", "B0")),
            summary_rejected(lambda broken: broken["seeds"][1].__setitem__("seed", 42)),
            summary_rejected(lambda broken: broken.__setitem__("n_seeds", 5)),
            summary_rejected(lambda broken: broken["seeds"][0].__setitem__("run_dir", "runs/local/le2i_cv/baseline_b1")),
            summary_rejected(lambda broken: broken["seeds"][0].__setitem__("run_dir", "runs/local/le2i/baseline_b1"), "le2i-cv"),
            summary_rejected(
                lambda broken: broken["aggregate"]["classification"]["test"]["macro_f1_restricted"].__setitem__("mean", 0.0)
            ),
        ]
    return _check(
        "entradas inválidas são recusadas: matriz malformada/negativa/não inteira, métricas gravadas "
        "divergentes, arma ou protocolo trocados, referência como --run, sumário com arma, seed, "
        "n_seeds, run_dir de outro protocolo ou agregado divergentes",
        blocks_rejected and wrong_arm and wrong_protocol and reference_as_run and all(summary_checks),
    )


def _selftest_non_finite_stored_values_rejected() -> bool:
    # NaN torna qualquer comparação falsa; sem checagem explícita um valor
    # gravado NaN passaria como "não divergente" do recálculo.
    non_finite = (float("nan"), float("inf"), float("-inf"))
    valid = _final_split(*labels_from_confusion_matrix(_known_matrix()))
    setters: list[Callable[[dict, float], None]] = [
        lambda block, value: block.__setitem__("macro_f1_restricted", value),
        lambda block, value: block["f1_by_class"].__setitem__("0", value),
        lambda block, value: block["per_class"]["fall"].__setitem__("precision", value),
        lambda block, value: block["support"].__setitem__("walk", value),
    ]

    def block_rejected(setter: Callable[[dict, float], None], value: float) -> bool:
        block = copy.deepcopy(valid)
        setter(block, value)
        return _raises(ValueError, lambda: diagnose_split(block, LE2I_LABEL_NAMES, "sintético"))

    blocks_rejected = all(block_rejected(setter, value) for setter in setters for value in non_finite)
    valid_block_accepted = not _raises(ValueError, lambda: diagnose_split(valid, LE2I_LABEL_NAMES, "sintético"))

    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "summary" / "multiseed_summary.json"
        payload = _summary_payload("C1", [42, 43], str(Path(tmp) / "seeds"))

        def summary_rejected(change: Callable[[dict], None]) -> bool:
            broken = copy.deepcopy(payload)
            change(broken)
            _write_summary(summary_path, broken)
            return _raises(
                ValueError, lambda: diagnose_multiseed_summary(summary_path, "le2i", "C1", LE2I_LABEL_NAMES)
            )

        aggregate_rejected = all(
            summary_rejected(
                lambda broken, split=split, field=field, value=value: broken["aggregate"]["classification"][split][
                    "macro_f1_restricted"
                ].__setitem__(field, value)
            )
            for split in ("train", "test")
            for field in ("n", "mean")
            for value in non_finite
        )
        seed_metric_rejected = summary_rejected(
            lambda broken: broken["seeds"][1]["classification"]["val"].__setitem__("macro_f1_restricted", float("nan"))
        )
        _write_summary(summary_path, payload)
        valid_summary_accepted = not _raises(
            ValueError, lambda: diagnose_multiseed_summary(summary_path, "le2i", "C1", LE2I_LABEL_NAMES)
        )
    return _check(
        "NaN e ±inf gravados são recusados em Macro-F1 restrita, f1_by_class, per_class, support, "
        "métricas por seed e agregado multi-seed (n e mean); valores finitos válidos seguem aceitos",
        blocks_rejected
        and valid_block_accepted
        and aggregate_rejected
        and seed_metric_rejected
        and valid_summary_accepted,
    )


def _selftest_missing_data_reported() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reference_root = root / "reference"
        _write_run(reference_root / "baseline_a", "le2i", "A", 42)
        summary_path = root / "summaries" / "a" / "multiseed_summary.json"
        _write_summary(summary_path, _summary_payload("A", [42, 43, 45], str(root / "seeds")))
        report, _ = build_diagnosis([], [DiagnosisInput("le2i", "A", summary_path)], reference_root)
        missing = {(item["dataset"], item["arm"], item["input"]) for item in report["missing_inputs"]}
        seeds_reason = [item["reason"] for item in report["missing_inputs"] if item["input"] == "seeds"]
        reference_missing = all(("le2i", arm, CANONICAL_REFERENCE) in missing for arm in ("B0", "B1", "C0", "C1"))
        cv_missing = all(("le2i-cv", arm, LOCAL_RUN) in missing for arm in ("A", "B0", "B1", "C0", "C1"))
        cv_has_no_reference = not any(
            item["dataset"] == "le2i-cv" and item["input"] == CANONICAL_REFERENCE for item in report["missing_inputs"]
        )

        disabled, _ = build_diagnosis([], [], None)
        disabled_reason = all(
            item["reason"] == "referências canônicas desabilitadas"
            for item in disabled["missing_inputs"]
            if item["input"] == CANONICAL_REFERENCE
        )

        incomplete = root / "incomplete"
        _write_run(incomplete, "le2i", "C0", 42)
        metrics = json.loads((incomplete / "metrics.json").read_text(encoding="utf-8"))
        del metrics["final"]["test"]
        (incomplete / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        missing_split = _raises(
            ValueError, lambda: diagnose_run(incomplete, "le2i", "C0", LOCAL_RUN, LE2I_LABEL_NAMES)
        )
        missing_file = _raises(
            FileNotFoundError, lambda: diagnose_run(root / "absent", "le2i", "C0", LOCAL_RUN, LE2I_LABEL_NAMES)
        )
    return _check(
        "entradas ausentes são reportadas sem supor referência CV: referências CS por arma, runs CV, "
        "sumários e seeds 44/46 faltantes; split ou arquivo ausente falha explicitamente",
        reference_missing
        and cv_missing
        and cv_has_no_reference
        and ("le2i", "A", MULTISEED_SUMMARY) not in missing
        and seeds_reason == [f"seeds esperadas ausentes em {summary_path}: [44, 46]"]
        and disabled_reason
        and missing_split
        and missing_file,
    )


def _selftest_end_to_end_protocols_and_protection() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reference_root = root / "reference"
        _write_run(reference_root / "baseline_b1", "le2i", "B1", 42)
        cs_run = root / "local" / "cs_b1_seed43"
        cv_run = root / "local" / "cv_b1"
        _write_run(cs_run, "le2i", "B1", 43, error_offset=1)
        _write_run(cv_run, "le2i-cv", "B1", 42, error_offset=4)
        summary_path = root / "summaries" / "b1" / "multiseed_summary.json"
        _write_summary(summary_path, _summary_payload("B1", [42, 43, 44, 45, 46], str(root / "seeds")))
        runs = [DiagnosisInput("le2i", "B1", cs_run), DiagnosisInput("le2i-cv", "B1", cv_run)]
        summaries = [DiagnosisInput("le2i", "B1", summary_path)]
        inputs = [
            *(reference_root / "baseline_b1" / name for name in ("config.yaml", "metrics.json")),
            *(path / name for path in (cs_run, cv_run) for name in ("config.yaml", "metrics.json")),
            summary_path,
        ]
        before = snapshot_inputs(inputs)
        mtimes = {path: path.stat().st_mtime_ns for path in inputs}

        output_dir = root / "out"
        written = run_diagnosis(runs, summaries, output_dir, reference_root)
        report = json.loads((output_dir / DIAGNOSIS_JSON_FILE).read_text(encoding="utf-8"))
        summary_text = (output_dir / DIAGNOSIS_SUMMARY_FILE).read_text(encoding="utf-8")
        cs_sources = report["protocols"]["le2i"]["arms"]["B1"]
        cv_sources = report["protocols"]["le2i-cv"]["arms"]["B1"]
        protocols_separate = (
            [run["source_kind"] for run in cs_sources["runs"]] == [CANONICAL_REFERENCE, LOCAL_RUN]
            and [run["source"] for run in cv_sources["runs"]] == [str(cv_run)]
            and not cv_sources["multiseed_summaries"]
            and all(run["dataset"] == "le2i-cv" for run in cv_sources["runs"])
        )
        reference_match = [
            (item["seed"], item["same_checkpoint_sha256"]) for item in report["canonical_reference_matches"]
        ] == [(42, True)]
        variation = cs_sources["multiseed_summaries"][0]["seed_variation"]["test"]["macro_f1_restricted"]
        rendered = all(
            marker in summary_text
            for marker in ("## Le2i-CS", "## Le2i-CV", "Observações confirmadas", "Hipóteses", "Entradas ausentes", "Limitações")
        )
        inputs_untouched = snapshot_inputs(inputs) == before and all(
            path.stat().st_mtime_ns == mtime for path, mtime in mtimes.items()
        ) and report["input_sha256"] == before

        guarded_outputs = [
            REFERENCE_RUN_ROOT / "classification_diagnosis",
            reference_root / "baseline_b1",
            cs_run,
            cs_run / "diagnosis",
            summary_path.parent,
            root,
        ]
        guards = all(
            _raises(ValueError, lambda target=target: run_diagnosis(runs, summaries, target, reference_root))
            for target in guarded_outputs
        )
        nothing_created = not (REFERENCE_RUN_ROOT / "classification_diagnosis").exists() and not (
            cs_run / "diagnosis"
        ).exists()

        previous = (output_dir / DIAGNOSIS_CSV_FILE).read_text(encoding="utf-8")
        skipped = not write_outputs(output_dir, {"x": 1}, [], "novo", force=False)
        preserved = (output_dir / DIAGNOSIS_CSV_FILE).read_text(encoding="utf-8") == previous
        forced = write_outputs(output_dir, {"x": 1}, [], "novo", force=True)
        overwritten = (output_dir / DIAGNOSIS_SUMMARY_FILE).read_text(encoding="utf-8") == "novo"

        snapshot = snapshot_inputs([cs_run / "metrics.json"])
        (cs_run / "metrics.json").write_text("{}", encoding="utf-8")
        tamper_detected = _raises(RuntimeError, lambda: verify_inputs_unchanged(snapshot))
    return _check(
        "execução ponta a ponta mantém Le2i-CS e Le2i-CV separados, liga referência à seed 42, "
        "não muda entradas, recusa saída sobre referências/runs/sumários, respeita --force e detecta adulteração",
        written
        and protocols_separate
        and reference_match
        and variation["n"] == 5
        and rendered
        and inputs_untouched
        and guards
        and nothing_created
        and skipped
        and preserved
        and forced
        and overwritten
        and tamper_detected,
    )


def run_classification_diagnosis_selftest() -> bool:
    checks = [
        _selftest_metrics_match_independent_formula(),
        _selftest_confusion_interpretation(),
        _selftest_invalid_inputs_rejected(),
        _selftest_non_finite_stored_values_rejected(),
        _selftest_missing_data_reported(),
        _selftest_end_to_end_protocols_and_protection(),
    ]
    ok = all(checks)
    if ok:
        print("\nclassification_diagnosis selftest OK: todas as checagens passaram")
    else:
        print("\nclassification_diagnosis selftest FALHOU", file=sys.stderr)
    return ok


def run_selftest() -> None:
    if not run_classification_diagnosis_selftest():
        sys.exit(1)
