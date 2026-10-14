import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.eval.analysis import internal_fold_feasibility
from gatefall.eval.analysis.internal_fold_feasibility import (
    FEASIBILITY_JSON_FILE,
    FEASIBILITY_SUMMARY_FILE,
    build_folds,
    build_report,
    read_train_partition,
    run_assessment,
    validate_official_train_partition,
)
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.hashing import sha256_file
from gatefall.runs import REFERENCE_RUN_ROOT

W, F, FN, SD, ST, SU, SG, OT = 0, 1, 2, 3, 4, 7, 8, 9
I = IGNORE_LABEL

# (video_id, split, env, cam, subject, rótulo por quadro). O sujeito bruto 1
# aparece em EnvA e EnvB. EnvC só tem `other` fora do prefixo ignorado e
# nenhum vídeo tem `fall`. Os vídeos de val/test usam rótulos e ambientes que
# não existem em train para denunciar qualquer leitura fora do split.
VIDEO_SPEC: tuple[tuple[str, str, str, int, int, list[int]], ...] = (
    ("a1", "train", "EnvA", 1, 1, [W, W, W, W, SD, SD, ST, ST]),
    ("a2", "train", "EnvA", 1, 2, [I, I, W, W, SG, SG, SU, SU]),
    ("b1", "train", "EnvB", 2, 1, [W, W, W, W, OT, OT, OT, OT]),
    ("b2", "train", "EnvB", 2, 3, [FN, FN, FN, FN, W, W, W, W]),
    ("c1", "train", "EnvC", 3, 4, [I, I, I, I, OT, OT, OT, OT]),
    ("v1", "val", "EnvVal", 6, 9, [5, 5, 5, 5, 6, 6, 6, 6]),
    ("t1", "test", "EnvTest", 7, 9, [6, 6, 6, 6, 5, 5, 5, 5]),
)


def _frames(spec: tuple[tuple[str, str, str, int, int, list[int]], ...] = VIDEO_SPEC) -> pd.DataFrame:
    rows = [
        {"video_id": video_id, "split": split, "env": env, "subject": subject, "frame_index": index, "label": label}
        for video_id, split, env, _cam, subject, labels in spec
        for index, label in enumerate(labels)
    ]
    return pd.DataFrame(rows)


def _manifest(spec: tuple[tuple[str, str, str, int, int, list[int]], ...] = VIDEO_SPEC) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": video_id, "split": split, "env": env, "cam": cam, "subject": subject,
             "width": 320, "height": 240, "fps": 25.0}
            for video_id, split, env, cam, subject, _labels in spec
        ]
    )


def _train_only(table: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, table[table["split"] == "train"]).reset_index(drop=True)


def _write_inputs(root: Path) -> tuple[Path, Path]:
    frames_path = root / "processed" / "frames.parquet"
    manifest_path = root / "processed" / "manifest.parquet"
    frames_path.parent.mkdir(parents=True)
    _frames().to_parquet(frames_path)
    _manifest().to_parquet(manifest_path)
    return frames_path, manifest_path


SYNTHETIC_EXPECTED = {
    "environments": ("EnvA", "EnvB", "EnvC"),
    "n_videos": 5,
    "windows_total_eval_stride": 40,
    "windows_labeled_eval_stride": 34,
}


def _raises(error: type[Exception], action: Callable[[], object], fragment: str) -> bool:
    try:
        action()
    except error as exc:
        return fragment in str(exc)
    return False


def _report() -> dict:
    return build_report(_train_only(_frames()), _train_only(_manifest()))


def _fold(report: dict, retained: str) -> dict:
    return next(fold for fold in report["folds"] if fold["retained_environment"] == retained)


def _selftest_fold_construction() -> bool:
    folds = build_folds(["EnvC", "EnvA", "EnvB"])
    retained = [fold["retained_environment"] for fold in folds]
    partitions_ok = all(
        sorted([cast(str, fold["retained_environment"]), *cast(list[str], fold["development_environments"])])
        == ["EnvA", "EnvB", "EnvC"]
        and fold["retained_environment"] not in cast(list[str], fold["development_environments"])
        for fold in folds
    )
    report = _report()
    videos_disjoint = all(
        fold["development_train"]["n_videos"] + fold["retained"]["n_videos"] == 5 for fold in report["folds"]
    )
    return _check(
        "cada ambiente de train é retido em exatamente uma dobra e os dois outros formam o treino de "
        "desenvolvimento, sem vídeo dos dois lados",
        retained == ["EnvA", "EnvB", "EnvC"] and partitions_ok and videos_disjoint,
    )


def _selftest_window_counts_and_ignore_semantics() -> bool:
    fold = _fold(_report(), "EnvC")
    development = fold["development_train"]["windows"]
    retained = fold["retained"]["windows"]
    # EnvA+EnvB: 4 vídeos × 8 quadros; a2 tem 2 quadros IGNORE_LABEL.
    # Stride 4 termina em k_end 0 e 4: a2 k_end 0 é ignorada.
    # EnvC: c1 tem 4 quadros ignorados; stride 4 ignora k_end 0.
    expected_development = {"train_stride": (8, 7, 1, 6), "eval_stride": (32, 30, 2, 26)}
    expected_retained = {"train_stride": (2, 1, 1, 0), "eval_stride": (8, 4, 4, 0)}

    def matches(block: dict, expected: dict) -> bool:
        return all(
            (block[name]["total"], block[name]["labeled"], block[name]["ignored"], block[name]["labeled_outside_other"])
            == values
            for name, values in expected.items()
        )

    return _check(
        "janelas rotuladas, ignoradas e fora de other seguem build_window_index nos dois strides, com "
        "IGNORE_LABEL decidido pelo último quadro",
        matches(development, expected_development) and matches(retained, expected_retained),
    )


def _selftest_class_support_and_identities() -> bool:
    report = _report()
    fold = _fold(report, "EnvC")
    development = fold["development_train"]
    support = development["restricted_class_support"]
    walk_ok = (support["walk"]["windows_eval_stride"], support["walk"]["videos"], support["walk"]["identities"]) == (
        14, 4, 4
    )
    sitting_ok = (support["sitting"]["windows_eval_stride"], support["sitting"]["videos"]) == (2, 1)
    excluded_ok = all(
        entry["windows_eval_stride"] == 0 for entry in development["excluded_class_support"].values()
    )
    identities_ok = development["identities"] == ["EnvA/1", "EnvA/2", "EnvB/1", "EnvB/3"]
    official_ok = report["official_train"]["n_identities"] == 5 and report["official_train"]["n_videos"] == 5
    return _check(
        "suporte por classe conta janelas, vídeos e identidades (ambiente, sujeito); o mesmo ID bruto em "
        "dois ambientes vira duas identidades e classes excluídas ficam fora das restritas",
        walk_ok and sitting_ok and excluded_ok and identities_ok and official_ok,
    )


def _selftest_missing_and_single_support_flags() -> bool:
    fold = _fold(_report(), "EnvA")
    development = fold["development_train"]
    retained = fold["retained"]
    missing_ok = development["missing_restricted_classes"] == ["fall", "sit_down", "sitting", "stand_up", "standing"]
    retained_missing_ok = retained["missing_restricted_classes"] == ["fall", "fallen", "other"]
    single_ok = "sit_down" in retained["single_video_classes"] and "fallen" in development["single_video_classes"]
    text = " ".join(fold["limitations"])
    flagged = "sem nenhuma janela" in text and "um único vídeo" in text and "IDs brutos" in text and "[1]" in text
    return _check(
        "classes restritas ausentes e classes de um único vídeo são marcadas nos dois lados, e IDs brutos "
        "repetidos entre lados são relatados sem tratá-los como a mesma pessoa",
        missing_ok and retained_missing_ok and single_ok and flagged,
    )


# EnvX só tem `other`; EnvY e EnvZ não têm `other`.
BLOCKER_SPEC: tuple[tuple[str, str, str, int, int, list[int]], ...] = (
    ("x1", "train", "EnvX", 1, 1, [OT] * 8),
    ("y1", "train", "EnvY", 2, 2, [W, W, W, W, SG, SG, SG, SG]),
    ("z1", "train", "EnvZ", 3, 3, [W, W, W, W, ST, ST, ST, ST]),
)


def _selftest_criterion_blockers() -> bool:
    report = build_report(_frames(BLOCKER_SPEC), _manifest(BLOCKER_SPEC))
    blocked = _fold(report, "EnvX")["criterion_blockers"]
    computable = _fold(report, "EnvY")["criterion_blockers"]
    return _check(
        "o critério fica bloqueado quando o ambiente retido não tem janela fora de other ou quando o treino de "
        "desenvolvimento não tem other",
        len(blocked) == 2
        and any("denominador" in text for text in blocked)
        and any("não tem janela other" in text for text in blocked)
        and computable == [],
    )


def _selftest_reads_only_official_train() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        frames_path, manifest_path = _write_inputs(Path(tmp))
        frames = read_train_partition(frames_path)
        manifest = read_train_partition(manifest_path)
    only_train = set(frames["split"]) == {"train"} and set(manifest["split"]) == {"train"}
    no_foreign_rows = not set(frames["env"]) & {"EnvVal", "EnvTest"} and not set(frames["label"]) & {5, 6}
    report = build_report(frames, manifest)
    return _check(
        "a leitura filtra split == train no parquet: rótulos e ambientes de val/test nunca entram no relatório",
        only_train and no_foreign_rows and report["official_splits_read"] == ["train"],
    )


def _selftest_official_partition_guard() -> bool:
    frames = _train_only(_frames())
    manifest = _train_only(_manifest())
    with patch.object(internal_fold_feasibility, "EXPECTED_OFFICIAL_TRAIN", SYNTHETIC_EXPECTED):
        accepted = _raises(ValueError, lambda: validate_official_train_partition(frames, manifest), "") is False
        wrong_envs = _raises(
            ValueError,
            lambda: validate_official_train_partition(
                cast(pd.DataFrame, frames[frames["env"] != "EnvC"]), manifest
            ),
            "ambientes do train oficial",
        )
        missing_video = _raises(
            ValueError,
            lambda: validate_official_train_partition(
                frames, cast(pd.DataFrame, manifest[manifest["video_id"] != "a1"])
            ),
            "divergem entre frames e manifesto",
        )
    with patch.object(internal_fold_feasibility, "EXPECTED_OFFICIAL_TRAIN", {**SYNTHETIC_EXPECTED, "n_videos": 6}):
        video_count = _raises(ValueError, lambda: validate_official_train_partition(frames, manifest), "vídeos")
    with patch.object(
        internal_fold_feasibility, "EXPECTED_OFFICIAL_TRAIN", {**SYNTHETIC_EXPECTED, "windows_labeled_eval_stride": 33}
    ):
        window_count = _raises(
            ValueError, lambda: validate_official_train_partition(frames, manifest), "janelas de train"
        )
    ambiguous = _frames()
    ambiguous.loc[(ambiguous["video_id"] == "a1") & (ambiguous["frame_index"] == 0), "subject"] = 7
    conflict = _raises(
        ValueError, lambda: build_report(_train_only(ambiguous), _train_only(_manifest())), "ambíguo"
    )
    return _check(
        "a partição train precisa bater com os ambientes, vídeos e janelas esperados do Le2i-CV e cada vídeo "
        "precisa ter ambiente e sujeito únicos",
        accepted and wrong_envs and missing_video and video_count and window_count and conflict,
    )


def _selftest_missing_prerequisites() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "out"
        written = run_assessment(root / "absent" / "frames.parquet", root / "absent" / "manifest.parquet", output)
        report = json.loads((output / FEASIBILITY_JSON_FILE).read_text(encoding="utf-8"))
        summary = (output / FEASIBILITY_SUMMARY_FILE).read_text(encoding="utf-8")
    no_numbers = report["folds"] == [] and "official_train" not in report and "input_sha256" not in report
    return _check(
        "dados locais ausentes geram relatório com os pré-requisitos faltantes e nenhum número inventado",
        written
        and report["status"] == "missing_prerequisites"
        and len(report["missing_prerequisites"]) == 2
        and no_numbers
        and "Nenhum número foi calculado" in summary,
    )


def _selftest_end_to_end_read_only() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames_path, manifest_path = _write_inputs(root)
        before = {path: sha256_file(path) for path in (frames_path, manifest_path)}
        output = root / "out"
        with patch.object(internal_fold_feasibility, "EXPECTED_OFFICIAL_TRAIN", SYNTHETIC_EXPECTED):
            written = run_assessment(frames_path, manifest_path, output)
            report = json.loads((output / FEASIBILITY_JSON_FILE).read_text(encoding="utf-8"))
            summary = (output / FEASIBILITY_SUMMARY_FILE).read_text(encoding="utf-8")
            (output / FEASIBILITY_SUMMARY_FILE).write_text("sentinela", encoding="utf-8")
            skipped = not run_assessment(frames_path, manifest_path, output)
            preserved = (output / FEASIBILITY_SUMMARY_FILE).read_text(encoding="utf-8") == "sentinela"
            forced = run_assessment(frames_path, manifest_path, output, force=True)
            protected = [REFERENCE_RUN_ROOT / "feasibility", frames_path.parent / "out"]
            guards = all(
                _raises(ValueError, lambda target=target: run_assessment(frames_path, manifest_path, target), "protegido")
                for target in protected
            )
        after = {path: sha256_file(path) for path in (frames_path, manifest_path)}
    contract = (
        report["canonical"] is False
        and report["read_only"] is True
        and report["status"] == "assessed"
        and len(report["folds"]) == 3
        and set(report["input_sha256"].values()) == set(before.values())
        and report["approval_required"]
    )
    return _check(
        "execução ponta a ponta grava JSON e Markdown não canônicos, registra hashes, não altera entradas, "
        "recusa saída protegida e respeita --force",
        written and contract and "retém EnvC" in summary and skipped and preserved and forced and guards
        and before == after and not (REFERENCE_RUN_ROOT / "feasibility").exists(),
    )


def run_internal_fold_feasibility_selftest() -> bool:
    checks = [
        _selftest_fold_construction(),
        _selftest_window_counts_and_ignore_semantics(),
        _selftest_class_support_and_identities(),
        _selftest_missing_and_single_support_flags(),
        _selftest_criterion_blockers(),
        _selftest_reads_only_official_train(),
        _selftest_official_partition_guard(),
        _selftest_missing_prerequisites(),
        _selftest_end_to_end_read_only(),
    ]
    ok = all(checks)
    if ok:
        print("\ninternal_fold_feasibility selftest OK: todas as checagens passaram")
    else:
        print("\ninternal_fold_feasibility selftest FALHOU", file=sys.stderr)
    return ok


def run_selftest() -> None:
    if not run_internal_fold_feasibility_selftest():
        sys.exit(1)
