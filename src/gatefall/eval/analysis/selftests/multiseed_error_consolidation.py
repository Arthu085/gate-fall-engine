import copy
import hashlib
import json
import statistics
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis import stratified_error_diagnosis
from gatefall.eval.analysis.classification_diagnosis import DiagnosisInput
from gatefall.eval.analysis.multiseed_error_consolidation import (
    CONSOLIDATION_CSV_FILE,
    CONSOLIDATION_JSON_FILE,
    CONSOLIDATION_SUMMARY_FILE,
    INCONCLUSIVE,
    STABLE,
    STABLE_IN_AVAILABLE_SEEDS,
    UNSTABLE,
    build_consolidation,
    run_consolidation,
)
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.eval.analysis.selftests.stratified_error_diagnosis import (
    VIDEO_SPEC,
    _frames,
    _raises,
    _stored_from,
    _tree_hashes,
)
from gatefall.eval.analysis.stratified_error_diagnosis import analyze_split, build_report
from gatefall.eval.shared.orchestration import Predictions
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.shared.metrics import restricted_macro_f1

SEEDS = (42, 43, 44, 45, 46)
Errors = dict[tuple[str, int], int]

# Janelas sintéticas (ver VIDEO_SPEC): em test, b1 (EnvB) tem other em k 1–3 e
# sitting em k 4–5; a1/a2 (EnvA) têm walk, fall, fallen e standing. Em val,
# v2 (EnvC) tem stand_up em k 2. Rótulos: 0=walk, 1=fall, 2=fallen, 8=standing,
# 9=other.
B0_COMMON_TEST: Errors = {("b1", 4): 9, ("b1", 5): 9}
B0_SEED_TEST: dict[int, Errors] = {
    42: {("a1", 2): 2},
    43: {("a2", 3): 0},
    44: {("a1", 3): 2},
    45: {("a2", 0): 8},
    46: {},
}
VAL_COMMON: Errors = {("v2", 2): 8}
A_TEST: Errors = {("a1", 2): 2}
# Só a seed 43 de A prevê fall em EnvB, onde fall não existe.
A_SEED_43_TEST: Errors = {("b1", 4): 1}


def _predictions(frames: pd.DataFrame, split: str, errors: Errors) -> Predictions:
    windows = build_window_index(
        cast(pd.DataFrame, frames[frames["split"] == split]), stride=EVAL_STRIDE, drop_ignored=False
    )
    video_ids = [str(video_id) for video_id in windows["video_id"]]
    k_ends = [int(k_end) for k_end in windows["k_end"]]
    true_labels = [int(label) for label in windows["label"]]
    pred_labels = [
        errors.get((video_id, k_end), max(label, 0))
        for video_id, k_end, label in zip(video_ids, k_ends, true_labels)
    ]
    return video_ids, k_ends, true_labels, pred_labels


def _split_analysis(frames: pd.DataFrame, split: str, errors: Errors) -> dict:
    predictions = _predictions(frames, split, errors)
    return analyze_split(
        frames, lambda _split: (len(predictions[0]), predictions), split, _stored_from(predictions), "sintético"
    )


def _entry(
    dataset: str,
    arm: str,
    seed: int,
    errors: dict[str, Errors],
    frames: pd.DataFrame | None = None,
    source: str | None = None,
    frames_sha256: str = "f" * 64,
) -> dict:
    frames = _frames() if frames is None else frames
    source = source or f"synthetic/{dataset}/{arm}/seed_{seed}"
    checkpoint = hashlib.sha256(f"{dataset}/{arm}/{seed}".encode()).hexdigest()
    return {
        "dataset": dataset,
        "protocol": {"le2i": "Le2i-CS", "le2i-cv": "Le2i-CV"}[dataset],
        "arm": arm,
        "seed": seed,
        "run_name": f"synthetic_{arm}",
        "source": source,
        "checkpoint_sha256": checkpoint,
        "input_sha256": {
            f"{source}/config.yaml": "c" * 64,
            f"{source}/metrics.json": "d" * 64,
            f"{source}/checkpoint.pt": checkpoint,
            f"data/processed/{dataset}/frames.parquet": frames_sha256,
        },
        "splits": {split: _split_analysis(frames, split, errors.get(split, {})) for split in ("val", "test")},
    }


def _report(entries: list[dict]) -> dict:
    by_source = {Path(entry["source"]): entry for entry in entries}
    with ExitStack() as stack:
        stack.enter_context(patch.object(stratified_error_diagnosis, "missing_prerequisites", lambda item: []))
        stack.enter_context(
            patch.object(stratified_error_diagnosis, "analyze_run", lambda item: copy.deepcopy(by_source[item.path]))
        )
        return build_report([DiagnosisInput(entry["dataset"], entry["arm"], Path(entry["source"])) for entry in entries])


def _b0_entry(seed: int, dataset: str = "le2i", **kwargs) -> dict:
    return _entry(dataset, "B0", seed, {"val": VAL_COMMON, "test": {**B0_COMMON_TEST, **B0_SEED_TEST[seed]}},
                  **kwargs)


def _a_entry(seed: int) -> dict:
    return _entry("le2i", "A", seed, {"test": {**A_TEST, **(A_SEED_43_TEST if seed == 43 else {})}})


def _scenario_entries() -> list[dict]:
    return [
        *(_a_entry(seed) for seed in SEEDS),
        *(_b0_entry(seed) for seed in SEEDS),
        *(_entry("le2i-cv", "B1", seed, {"val": VAL_COMMON, "test": B0_COMMON_TEST}) for seed in (42, 43, 44)),
    ]


def _write(path: Path, report: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    return path


def _consolidate(*reports: dict) -> dict:
    return build_consolidation(
        [(Path(f"synthetic/report_{index}.json"), report) for index, report in enumerate(reports)], SEEDS
    )


def _group(consolidation: dict, dataset: str, arm: str, split: str, level: str, name: str) -> dict:
    groups = consolidation["protocols"][dataset]["arms"][arm]["splits"][split][level]
    return next(group for group in groups if group["group"] == name)


def _expected_split_macro(seed: int) -> float:
    frames = _frames()
    _video_ids, _k_ends, true_labels, pred_labels = _predictions(
        frames, "test", {**B0_COMMON_TEST, **B0_SEED_TEST[seed]}
    )
    y_true = np.asarray(true_labels)
    labeled = y_true != IGNORE_LABEL
    return restricted_macro_f1(y_true[labeled], np.asarray(pred_labels)[labeled])[0]


def _selftest_aggregation() -> bool:
    consolidation = _consolidate(_report(_scenario_entries()))
    env_b = _group(consolidation, "le2i", "B0", "test", "environment", "EnvB")
    env_a = _group(consolidation, "le2i", "B0", "test", "environment", "EnvA")
    overall = _group(consolidation, "le2i", "B0", "test", "split", "all")
    subject = _group(consolidation, "le2i", "B0", "test", "environment_subject", "EnvB/subject_1")
    a_env_b = _group(consolidation, "le2i", "A", "test", "environment", "EnvB")
    expected = [_expected_split_macro(seed) for seed in SEEDS]
    # EnvB em B0: other com tp=3, fp=2 → F1 0,75; sitting com F1 0; média
    # fixa de oito classes restritas → 0,75/8 em todas as seeds.
    sitting = env_b["per_class"]["sitting"]
    consistency = env_b["dominant_confusion_consistency"]
    return _check(
        "agrega Macro-F1 por seed (média, desvio amostral), padrões por classe e consistência de confusões "
        "sem somar janelas entre seeds",
        abs(env_b["macro_f1_restricted"]["mean"] - 0.75 / 8) < 1e-12
        and env_b["macro_f1_restricted"]["std"] == 0.0
        and env_b["macro_f1_restricted"]["n"] == 5
        and env_b["n_labeled_windows"] == 5
        and overall["n_labeled_windows"] == 16
        and abs(overall["macro_f1_restricted"]["mean"] - statistics.mean(expected)) < 1e-12
        and abs(overall["macro_f1_restricted"]["std"] - statistics.stdev(expected)) < 1e-12
        and overall["macro_f1_restricted"]["by_seed"]["46"] == expected[-1]
        and overall["n_errors"]["by_seed"] == {"42": 3, "43": 3, "44": 3, "45": 3, "46": 2}
        and subject["macro_f1_restricted"]["mean"] == env_b["macro_f1_restricted"]["mean"]
        and sitting["support"] == 2
        and sitting["recall"]["mean"] == 0.0
        and sitting["main_error_target"]["modal"] == "other"
        and sitting["main_error_target"]["modal_seeds"] == 5
        and consistency["reference_top1"] == "sitting->other"
        and consistency["reference_top1_seeds"] == 5
        and consistency["reference_share_of_true_support"]["mean"] == 1.0
        and consistency["status"] == STABLE
        and overall["dominant_confusion_consistency"]["status"] == STABLE
        and env_a["dominant_confusion_consistency"]["reference_top1"] == "fall->fallen"
        and env_a["dominant_confusion_consistency"]["reference_top1_seeds"] == 2
        and env_a["dominant_confusion_consistency"]["top1_by_seed"]["46"] is None
        and env_a["dominant_confusion_consistency"]["status"] == UNSTABLE
        and _group(consolidation, "le2i", "A", "val", "split", "all")["dominant_confusion_consistency"]["status"]
        == INCONCLUSIVE
        and env_b["absent_restricted_labels"] == ["walk", "fall", "fallen", "sit_down", "stand_up", "standing"]
        and any("mecanicamente menor" in warning for warning in env_b["warnings"])
        and env_b["residual_prediction"]["rate_of_non_residual_windows"]["mean"] == 1.0
        and env_b["residual_prediction"]["share_of_errors"]["mean"] == 1.0
        and overall["residual_prediction"]["rate_of_non_residual_windows"]["by_seed"]["42"] == 2 / 13
        and a_env_b["residual_prediction"]["rate_of_non_residual_windows"]["max"] == 0.0
        and a_env_b["per_class"]["fall"]["seeds_predicting_absent_class"] == [43]
        and "fall ausente no grupo, mas predita nas seeds [43]" in a_env_b["warnings"],
    )


def _selftest_findings_distinguish_supported_and_hypotheses() -> bool:
    consolidation = _consolidate(_report(_scenario_entries()))
    stability = {
        (item["dataset"], item["arm"], item["split"], item["group"]): item for item in consolidation["focus_stability"]
    }
    texts = [item["text"] for item in consolidation["observations"]]
    hypotheses = [item["text"] for item in consolidation["hypotheses"]]
    return _check(
        "separa estabilidade sustentada de B0/B1 de hipóteses e limita os achados às armas foco",
        stability[("le2i", "B0", "test", "EnvB")]["supported"] is True
        and stability[("le2i", "B0", "test", "EnvA")]["supported"] is False
        and stability[("le2i-cv", "B1", "test", "EnvB")]["status"] == STABLE_IN_AVAILABLE_SEEDS
        and {item["arm"] for item in consolidation["focus_stability"]} == {"B0", "B1"}
        and any("sitting→other, é a dominante em 5 de 5 seeds" in text for text in texts)
        and any("Le2i-CS/B0 test, EnvB" in text and "(A: " in text for text in texts)
        and any("Le2i-CS/B0 test, EnvB" in text and "classe residual" in text for text in hypotheses)
        and any("Le2i-CS/B0 test, EnvB" in text and "excede a de qualquer seed de A" in text for text in hypotheses)
        and stability[("le2i", "B0", "test", "EnvB")]["residual_above_comparison_in_all_seeds"] is True
        and stability[("le2i-cv", "B1", "test", "EnvB")]["residual_above_comparison_in_all_seeds"] is None
        and any("Le2i-CS/B0 test, EnvA" in text and "depende da seed" in text for text in hypotheses),
    )


def _selftest_incomplete_seeds() -> bool:
    consolidation = _consolidate(_report(_scenario_entries()))
    cv_b1 = consolidation["protocols"]["le2i-cv"]["arms"]["B1"]
    missing = {(item["dataset"], item["arm"]): item for item in consolidation["missing_inputs"]}
    cs_c0 = consolidation["protocols"]["le2i"]["arms"]["C0"]
    return _check(
        "relata seeds e armas ausentes explicitamente e agrega só as seeds disponíveis",
        cv_b1["coverage"]["seeds"] == [42, 43, 44]
        and cv_b1["coverage"]["missing_seeds"] == [45, 46]
        and cv_b1["coverage"]["complete"] is False
        and _group(consolidation, "le2i-cv", "B1", "test", "split", "all")["macro_f1_restricted"]["n"] == 3
        and missing[("le2i-cv", "B1")]["reason"] == "cobertura de seeds incompleta"
        and missing[("le2i-cv", "B1")]["missing_seeds"] == [45, 46]
        and missing[("le2i", "C0")]["reason"] == "nenhuma seed fornecida"
        and cs_c0["splits"] == {}
        and len(consolidation["missing_inputs"]) == 8
        and any("seeds ou runs ausentes" in text for text in consolidation["limitations"]),
    )


def _selftest_incompatible_support() -> bool:
    base = [_b0_entry(seed) for seed in SEEDS[:2]]
    more_sitting = tuple(
        (video_id, split, env, subject, [IGNORE_LABEL, 9, 9, 4, 4, 4] if video_id == "b1" else labels)
        for video_id, split, env, subject, labels in VIDEO_SPEC
    )
    other_env = tuple(
        (video_id, split, "EnvD" if video_id == "b1" else env, subject, labels)
        for video_id, split, env, subject, labels in VIDEO_SPEC
    )
    support_entry = _entry("le2i", "A", 44, {}, frames=_frames(more_sitting))
    coverage_entry = _entry("le2i", "A", 44, {}, frames=_frames(other_env))
    frames_entry = _entry("le2i", "A", 44, {}, frames_sha256="e" * 64)
    cross_protocol_ok = _consolidate(_report([*base, _entry("le2i-cv", "A", 44, {}, frames=_frames(more_sitting))]))
    stride_report = _report([_a_entry(45)])
    stride_report["eval_stride"] = 2
    return _check(
        "recusa suporte, cobertura de grupos, frames ou parâmetros de relatório incompatíveis dentro do protocolo",
        _raises(ValueError, lambda: _consolidate(_report([*base, support_entry])), "suporte incompatível")
        and _raises(ValueError, lambda: _consolidate(_report([*base, coverage_entry])), "cobertura de grupos")
        and _raises(ValueError, lambda: _consolidate(_report([*base, frames_entry])), "frames.parquet diverge")
        and _raises(ValueError, lambda: _consolidate(_report(base), stride_report), "incompatível")
        and cross_protocol_ok["protocols"]["le2i-cv"]["arms"]["A"]["coverage"]["seeds"] == [44],
    )


def _selftest_duplicate_identities_and_provenance() -> bool:
    first = _b0_entry(42)
    same_checkpoint = _b0_entry(43)
    same_checkpoint["checkpoint_sha256"] = first["checkpoint_sha256"]
    same_checkpoint["input_sha256"][f"{same_checkpoint['source']}/checkpoint.pt"] = first["checkpoint_sha256"]
    moved = _b0_entry(42, source="synthetic/elsewhere/b0")
    unexpected_seed = _entry("le2i", "B0", 47, {})
    wrong_checkpoint = _b0_entry(44)
    wrong_checkpoint["input_sha256"][f"{wrong_checkpoint['source']}/checkpoint.pt"] = "0" * 64
    other_protocol = _b0_entry(45, source="runs/local/le2i_cv/b0_seed45")
    failed = _b0_entry(46)
    failed["splits"]["test"]["reconciliation"]["ok"] = False
    claims_identity = _b0_entry(46)
    claims_identity["splits"]["val"]["reconciliation"]["proves_historical_window_predictions"] = True
    tampered = _b0_entry(46)
    tampered["splits"]["test"]["levels"]["environment"][0]["macro_f1_restricted"] += 0.01
    dropped_group = _b0_entry(46)
    dropped_group["splits"]["test"]["levels"]["video"].pop()
    other_analysis = _report([_b0_entry(43)])
    other_analysis["analysis"] = "classification_diagnosis"
    return _check(
        "recusa identidade de seed ou checkpoint duplicada, seed inesperada, procedência inválida, "
        "reconciliação falha e relatório adulterado",
        _raises(ValueError, lambda: _consolidate(_report([first]), _report([moved])), "identidade de seed duplicada")
        and _raises(ValueError, lambda: _consolidate(_report([first, same_checkpoint])), "checkpoint_sha256 duplicado")
        and _raises(ValueError, lambda: _consolidate(_report([unexpected_seed])), "fora do conjunto esperado")
        and _raises(ValueError, lambda: _consolidate(_report([wrong_checkpoint])), "input_sha256")
        and _raises(ValueError, lambda: _consolidate(_report([other_protocol])), "outro protocolo")
        and _raises(ValueError, lambda: _consolidate(_report([failed])), "reconciliação")
        and _raises(ValueError, lambda: _consolidate(_report([claims_identity])), "reconciliação")
        and _raises(ValueError, lambda: _consolidate(_report([tampered])), "inconsistente")
        and _raises(ValueError, lambda: _consolidate(_report([dropped_group])), "soma das matrizes por video")
        and _raises(ValueError, lambda: _consolidate(other_analysis), "stratified_error_diagnosis"),
    )


def _mutated_report(mutate: Callable[[dict], None]) -> dict:
    # B0 seed 42 em test: o split inteiro tem duas confusões dominantes
    # (sitting→other com 2 janelas, fall→fallen com 1) e EnvB não tem fall.
    report = _report([_b0_entry(42)])
    mutate(report["protocols"]["le2i"]["arms"]["B0"][0]["splits"]["test"])
    return report


def _env_b(split: dict) -> dict:
    return next(group for group in split["levels"]["environment"] if group["group"] == "EnvB")


def _set(target: Callable[[dict], dict], key: str, value: object) -> Callable[[dict], None]:
    def mutate(split: dict) -> None:
        target(split)[key] = value

    return mutate


def _overall(split: dict) -> dict:
    return split["overall"]


def _top_confusion(split: dict) -> dict:
    return split["overall"]["dominant_confusions"][0]


def _walk(split: dict) -> dict:
    return split["overall"]["per_class"]["walk"]


GROUP_METADATA_CASES: tuple[tuple[str, Callable[[dict], None], str], ...] = (
    ("ordem trocada", lambda split: split["overall"]["dominant_confusions"].reverse(), "dominant_confusions[0]"),
    ("contagem", _set(_top_confusion, "count", 3), "dominant_confusions[0].count"),
    ("rótulo", _set(_top_confusion, "predicted_label", "walk"), "dominant_confusions[0].predicted_label"),
    ("fração", _set(_top_confusion, "share_of_true_support", 0.5), "share_of_true_support"),
    ("fração não finita", _set(_top_confusion, "share_of_all_errors", float("inf")), "não finito"),
    ("lista truncada", lambda split: split["overall"]["dominant_confusions"].pop(), "esperado lista com 2"),
    ("ausente mas predita", _set(_env_b, "predicted_absent_labels", ["fall"]), "predicted_absent_labels"),
    ("Macro-F1 NaN", _set(_overall, "macro_f1_restricted", float("nan")), "não finito"),
    ("Macro-F1 ausente", lambda split: split["overall"].pop("macro_f1_restricted"), "ausente"),
    ("revocação infinita", _set(_walk, "recall", float("-inf")), "per_class[walk].recall"),
    ("precisão textual", _set(_walk, "precision", "1.0"), "não numérico"),
    ("suporte booleano", _set(_walk, "support", True), "per_class[walk].support"),
    ("erros não inteiros", _set(_overall, "n_errors", 3.0), "n_errors"),
    ("janelas totais", lambda split: split["overall"].update(n_windows=split["overall"]["n_windows"] + 1),
     "n_windows: gravado"),
    ("janelas ignoradas negativas", _set(_overall, "n_ignored_windows", -1), "não inteiras"),
)


def _selftest_group_metadata_integrity() -> bool:
    rejected = {
        name: _raises(ValueError, lambda mutate=mutate: _consolidate(_mutated_report(mutate)), text)
        for name, mutate, text in GROUP_METADATA_CASES
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        published: list[bool] = []
        for name, mutate in (("nan", _set(_overall, "macro_f1_restricted", float("nan"))),
                             ("confusion", _set(_top_confusion, "count", 3))):
            path = root / name / "stratified_error_diagnosis.json"
            path.parent.mkdir()
            # json.dumps grava NaN literal como o json.load padrão aceita.
            path.write_text(json.dumps(_mutated_report(mutate)), encoding="utf-8")
            output_dir = root / f"{name}_out"
            published.append(
                not _raises(ValueError, lambda: run_consolidation([path], output_dir), "inconsistente")
                or output_dir.exists()
            )
    failed = [name for name, ok in rejected.items() if not ok]
    if failed:
        print(f"  mutações aceitas: {failed}")
    return _check(
        "recusa confusões dominantes adulteradas (ordem, contagem, rótulo, fração, lista), classes ausentes "
        "preditas divergentes, métricas ausentes, não numéricas, booleanas ou não finitas e contagens de "
        "janelas inconsistentes, antes de publicar qualquer saída",
        not failed and not any(published),
    )


def _selftest_reproducibility() -> bool:
    entries = _scenario_entries()
    combined = _consolidate(_report(entries))
    by_seed = _consolidate(*(_report([entry]) for entry in reversed(entries)))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = [_write(root / "reports" / f"seed_{index}.json", _report([entry])) for index, entry in enumerate(entries)]
        run_consolidation(paths, root / "first")
        run_consolidation(list(reversed(paths)), root / "second")
        identical = all(
            (root / "first" / name).read_bytes() == (root / "second" / name).read_bytes()
            for name in (CONSOLIDATION_JSON_FILE, CONSOLIDATION_CSV_FILE, CONSOLIDATION_SUMMARY_FILE)
        )
    same_aggregates = all(
        combined["protocols"][dataset]["arms"][arm]["splits"] == by_seed["protocols"][dataset]["arms"][arm]["splits"]
        for dataset in combined["protocols"]
        for arm in combined["protocols"][dataset]["arms"]
    )
    return _check(
        "gera saídas idênticas byte a byte independentemente da ordem e da partição dos relatórios",
        identical and same_aggregates,
    )


def _selftest_protected_outputs() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "runs" / "b0_seed42"
        entries = [_b0_entry(42, source=str(run_dir)), *(_b0_entry(seed) for seed in SEEDS[1:])]
        report_path = _write(root / "reports" / "stratified_error_diagnosis.json", _report(entries))
        before = _tree_hashes(root)
        guarded = [root / "reports", root / "reports" / "out", run_dir, run_dir / "out",
                   REFERENCE_RUN_ROOT / "consolidation", REPOSITORY_ROOT / "src" / "x", REPOSITORY_ROOT / "data" / "x"]
        guards = all(
            _raises(ValueError, lambda target=target: run_consolidation([report_path], target), "sobrepõe")
            for target in guarded
        )
        nothing_created = not (root / "reports" / "out").exists() and not (
            REFERENCE_RUN_ROOT / "consolidation"
        ).exists()
        duplicate = _raises(ValueError, lambda: run_consolidation([report_path, report_path], root / "dup"),
                            "mais de uma vez")

        output_dir = root / "analysis"
        written = run_consolidation([report_path], output_dir)
        report = json.loads((output_dir / CONSOLIDATION_JSON_FILE).read_text(encoding="utf-8"))
        csv = pd.read_csv(output_dir / CONSOLIDATION_CSV_FILE)
        summary = (output_dir / CONSOLIDATION_SUMMARY_FILE).read_text(encoding="utf-8")
        inputs_untouched = {
            path: digest for path, digest in _tree_hashes(root).items() if not path.startswith("analysis")
        } == before
        recorded = (
            report["canonical"] is False
            and report["read_only"] is True
            and str(report_path.resolve()) in report["input_sha256"]
            and "video" not in set(csv["level"])
            and set(csv["level"]) == {"coverage", "split", "environment", "environment_subject"}
            and "(não canônico)" in summary
            and "Observações confirmadas" in summary
            and "Hipóteses (não verificadas)" in summary
        )

        previous = (output_dir / CONSOLIDATION_CSV_FILE).read_text(encoding="utf-8")
        skipped = not run_consolidation([report_path], output_dir)
        preserved = (output_dir / CONSOLIDATION_CSV_FILE).read_text(encoding="utf-8") == previous
        forced = run_consolidation([report_path], output_dir, force=True)

        invalid = json.loads(report_path.read_text(encoding="utf-8"))
        invalid["protocols"]["le2i"]["arms"]["B0"][0]["splits"]["test"]["reconciliation"]["ok"] = False
        invalid_path = _write(root / "invalid" / "stratified_error_diagnosis.json", invalid)
        invalid_output = root / "invalid_out"
        invalid_fails = _raises(ValueError, lambda: run_consolidation([invalid_path], invalid_output),
                                "reconciliação") and not invalid_output.exists()
    return _check(
        "recusa saída sobre relatórios, runs, referências, src e data; não muta entradas; respeita --force e "
        "não grava após entrada inválida",
        guards and nothing_created and duplicate and written and inputs_untouched and recorded and skipped
        and preserved and forced and invalid_fails,
    )


def run_multiseed_error_consolidation_selftest() -> bool:
    checks = [
        _selftest_aggregation(),
        _selftest_findings_distinguish_supported_and_hypotheses(),
        _selftest_incomplete_seeds(),
        _selftest_incompatible_support(),
        _selftest_duplicate_identities_and_provenance(),
        _selftest_group_metadata_integrity(),
        _selftest_reproducibility(),
        _selftest_protected_outputs(),
    ]
    ok = all(checks)
    if ok:
        print("\nmultiseed_error_consolidation selftest OK: todas as checagens passaram")
    else:
        print("\nmultiseed_error_consolidation selftest FALHOU", file=sys.stderr)
    return ok


def run_selftest() -> None:
    if not run_multiseed_error_consolidation_selftest():
        sys.exit(1)
