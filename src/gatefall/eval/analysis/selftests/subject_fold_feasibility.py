import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis import subject_fold_feasibility
from gatefall.eval.analysis.internal_fold_feasibility import read_train_partition, validate_official_train_partition
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.eval.analysis.subject_fold_feasibility import (
    FEASIBILITY_JSON_FILE,
    FEASIBILITY_SUMMARY_FILE,
    FEASIBLE_VERDICT,
    LOSO_SCHEME,
    NOT_FEASIBLE_VERDICT,
    POOLED_SCHEME,
    build_report,
    check_fold_isolation,
    pooled_minority_groups,
    run_assessment,
    validate_subject_identifiers,
)
from gatefall.hashing import sha256_file
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT

W, F, FN, SD, ST, SU, SG, OT = 0, 1, 2, 3, 4, 7, 8, 9
I = IGNORE_LABEL

# Com 8 quadros, o stride 4 termina só em k_end 0 e 4. Os padrões P e Q
# cobrem as oito classes em stride 1; com as variantes rotacionadas P2 e Q2,
# as oito também aparecem em stride 4.
P = [W, W, F, F, FN, FN, SD, SD]
Q = [ST, ST, SU, SU, SG, SG, OT, OT]
P2 = [F, F, W, W, SD, SD, FN, FN]
Q2 = [SU, SU, ST, ST, OT, OT, SG, SG]
P_IGNORED = [W, I, F, F, FN, FN, SD, SD]

Spec = tuple[tuple[str, str, str, int, int, list[int]], ...]

# (video_id, split, env, cam, subject, rótulo por quadro). Os sujeitos 1 e 2
# cobrem as oito classes com dois vídeos cada; o sujeito 2 tem um quadro
# ignorado, logo menos janelas rotuladas. Os sujeitos 3 e 4 não têm
# `stand_up` nem `standing`. Os IDs brutos 1 (EnvA, EnvB) e 4 (EnvA, EnvC)
# aparecem em dois ambientes. Val e test usam ambientes, sujeito e rótulos que
# não existem em train para denunciar qualquer leitura fora do split.
VIDEO_SPEC: Spec = (
    ("s1p", "train", "EnvA", 1, 1, P),
    ("s1q", "train", "EnvA", 1, 1, Q),
    ("s1p2", "train", "EnvB", 2, 1, P2),
    ("s1q2", "train", "EnvB", 2, 1, Q2),
    ("s2p", "train", "EnvA", 1, 2, P_IGNORED),
    ("s2q", "train", "EnvA", 1, 2, Q),
    ("s2p2", "train", "EnvA", 1, 2, P2),
    ("s2q2", "train", "EnvA", 1, 2, Q2),
    ("s3a", "train", "EnvB", 2, 3, [W, W, W, W, OT, OT, OT, OT]),
    ("s3b", "train", "EnvB", 2, 3, [F, F, F, F, FN, FN, FN, FN]),
    ("s4a", "train", "EnvA", 1, 4, [W, W, W, W, SD, SD, ST, ST]),
    ("s4c", "train", "EnvC", 3, 4, [OT] * 8),
    ("v1", "val", "EnvVal", 6, 9, [5] * 8),
    ("t1", "test", "EnvTest", 7, 9, [6] * 8),
)

SYNTHETIC_EXPECTED: dict[str, object] = {
    "environments": ("EnvA", "EnvB", "EnvC"),
    "n_videos": 12,
    "windows_total_eval_stride": 96,
    "windows_labeled_eval_stride": 95,
}


def _frames(spec: Spec = VIDEO_SPEC) -> pd.DataFrame:
    return pd.DataFrame([
        {"video_id": video_id, "split": split, "env": env, "subject": subject, "frame_index": index, "label": label}
        for video_id, split, env, _cam, subject, labels in spec
        for index, label in enumerate(labels)
    ])


def _manifest(spec: Spec = VIDEO_SPEC) -> pd.DataFrame:
    return pd.DataFrame([
        {"video_id": video_id, "split": split, "env": env, "cam": cam, "subject": subject,
         "width": 320, "height": 240, "fps": 25.0}
        for video_id, split, env, cam, subject, _labels in spec
    ])


def _train_only(table: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, table[table["split"] == "train"]).reset_index(drop=True)


def _report(spec: Spec = VIDEO_SPEC) -> dict:
    return build_report(_train_only(_frames(spec)), _train_only(_manifest(spec)))


def _scheme(report: dict, name: str) -> dict:
    return next(scheme for scheme in report["schemes"] if scheme["scheme"] == name)


def _fold(report: dict, scheme: str, fold: str) -> dict:
    return next(entry for entry in _scheme(report, scheme)["folds"] if entry["fold"] == fold)


def _raises(error: type[Exception], action: Callable[[], object], fragment: str) -> bool:
    try:
        action()
    except error as exc:
        return fragment in str(exc)
    return False


def _write_inputs(root: Path) -> tuple[Path, Path]:
    frames_path = root / "processed" / "frames.parquet"
    manifest_path = root / "processed" / "manifest.parquet"
    frames_path.parent.mkdir(parents=True)
    _frames().to_parquet(frames_path)
    _manifest().to_parquet(manifest_path)
    return frames_path, manifest_path


def _selftest_identity_audit() -> bool:
    identity = _report()["subject_identity"]
    per_subject = {entry["raw_subject"]: entry for entry in identity["per_subject"]}
    return _check(
        "a auditoria de identidade agrupa pelo ID bruto, lista os IDs em mais de um ambiente, conta os pares "
        "(ambiente, sujeito) e não afirma identidade global",
        identity["grouping_unit"] == "raw_subject"
        and identity["raw_subjects"] == [1, 2, 3, 4]
        and identity["raw_subjects_spanning_environments"] == [1, 4]
        and identity["rejected_grouping"]["raw_subjects_split_across_sides_if_global"] == [1, 4]
        and identity["n_environment_subject_pairs"] == 6
        and identity["global_identity_verifiable"] is False
        and per_subject[4]["environments"] == ["EnvA", "EnvC"]
        and per_subject[2]["windows_labeled_eval_stride"] == 31
        and per_subject[1]["n_videos"] == 4,
    )


def _with_frame_value(frames: pd.DataFrame, video_id: str, index: int, value: object) -> pd.DataFrame:
    changed = frames.copy()
    changed["subject"] = changed["subject"].astype(object)
    changed.loc[(changed["video_id"] == video_id) & (changed["frame_index"] == index), "subject"] = value
    return changed


def _selftest_ambiguous_identities() -> bool:
    frames = _train_only(_frames())
    manifest = _train_only(_manifest())
    two_subjects = _raises(
        ValueError, lambda: build_report(_with_frame_value(frames, "s3a", 2, 4), manifest), "ambíguo nos frames: ['s3a']"
    )
    missing = _raises(
        ValueError, lambda: build_report(_with_frame_value(frames, "s1q", 0, None), manifest), "sem sujeito: ['s1q']"
    )
    fractional = _raises(
        ValueError,
        lambda: validate_subject_identifiers(_with_frame_value(frames, "s4a", 3, 1.5), manifest),
        "inteiro não negativo nos vídeos ['s4a']",
    )
    negative_manifest = manifest.copy()
    negative_manifest.loc[negative_manifest["video_id"] == "s2q", "subject"] = -1
    negative = _raises(
        ValueError, lambda: validate_subject_identifiers(frames, negative_manifest), "manifesto: sujeito"
    )
    mismatched_manifest = manifest.copy()
    mismatched_manifest.loc[mismatched_manifest["video_id"] == "s4c", "subject"] = 3
    mismatch = _raises(
        ValueError,
        lambda: validate_official_train_partition(frames, mismatched_manifest, SYNTHETIC_EXPECTED),
        "divergem entre frames e manifesto de train: ['s4c']",
    )
    return _check(
        "identidades ambíguas interrompem a execução: vídeo com dois sujeitos, sujeito ausente, fracionário ou "
        "negativo e sujeito divergente entre frames e manifesto",
        two_subjects and missing and fractional and negative and mismatch,
    )


def _selftest_fold_isolation() -> bool:
    report = _report()
    all_videos = 12
    isolated = all(
        sorted(fold["retained_subjects"] + fold["development_subjects"]) == [1, 2, 3, 4]
        and not set(fold["retained_subjects"]) & set(fold["development_subjects"])
        and fold["isolation"]["shared_video_ids"] == []
        and fold["isolation"]["shared_raw_subjects"] == []
        and fold["development_train"]["n_videos"] + fold["retained"]["n_videos"] == all_videos
        for scheme in report["schemes"]
        for fold in scheme["folds"]
    )
    manifest = _train_only(_manifest())
    duplicated = pd.concat([manifest, manifest.loc[manifest["video_id"] == "s3a"].assign(subject=1)])
    leaked = _raises(ValueError, lambda: check_fold_isolation(duplicated, [1, 2, 4], [3]), "vídeos nos dois lados=['s3a']")
    overlapping = _raises(ValueError, lambda: check_fold_isolation(manifest, [1, 2, 3], [3, 4]), "sujeitos nos dois lados=[3]")
    uncovered = _raises(ValueError, lambda: check_fold_isolation(manifest, [1, 2], [3]), "vídeos fora da dobra=['s4a', 's4c']")
    shared_ok = _fold(report, POOLED_SCHEME, "retain_s1")["isolation"]["shared_environments"] == ["EnvA", "EnvB"]
    return _check(
        "cada dobra particiona os sujeitos e os vídeos do train sem interseção; vídeo nos dois lados, sujeito "
        "repetido ou vídeo fora da dobra interrompem a execução, e ambientes compartilhados são relatados",
        isolated and leaked and overlapping and uncovered and shared_ok,
    )


def _selftest_scheme_construction() -> bool:
    report = _report()
    loso = _scheme(report, LOSO_SCHEME)
    pooled = _scheme(report, POOLED_SCHEME)
    tied_spec = tuple(
        (video_id, split, env, cam, subject, P if video_id == "s2p" else labels)
        for video_id, split, env, cam, subject, labels in VIDEO_SPEC
    )
    tied_groups, tied_moved = pooled_minority_groups(
        build_window_index(_train_only(_frames(tied_spec)), stride=1, drop_ignored=False)
    )
    return _check(
        "deixar um sujeito de fora gera uma dobra por ID bruto; o esquema agregado isola os sujeitos com as oito "
        "classes em mais de um vídeo e move o menor deles para o grupo sem cobertura (empate pelo menor ID)",
        loso["groups"] == [[1], [2], [3], [4]]
        and pooled["groups"] == [[1], [2, 3, 4]]
        and pooled["moved_to_pool"] == [2]
        and tied_groups == [(2,), (1, 3, 4)]
        and tied_moved == [1],
    )


def _selftest_class_support_failures() -> bool:
    report = _report()
    loso = _scheme(report, LOSO_SCHEME)
    pooled = _scheme(report, POOLED_SCHEME)
    failing = _fold(report, LOSO_SCHEME, "retain_s3")
    expected_missing = ["sit_down", "sitting", "stand_up", "standing"]
    pooled_fold = _fold(report, POOLED_SCHEME, "retain_s2-3-4")
    singleton_fold = _fold(report, POOLED_SCHEME, "retain_s1")
    rare_ok = (
        pooled_fold["rare_retained_classes"]["single_raw_subject"] == ["stand_up", "standing"]
        and singleton_fold["rare_retained_classes"]["single_raw_subject"] == []
        and any("único sujeito bruto" in text for text in singleton_fold["limitations"])
    )
    flags_ok = (
        failing["eight_class_macro_f1_comparable"] is False
        and failing["missing_classes"]["retained"] == expected_missing
        and any("sem nenhuma janela em stride 1" in text for text in failing["limitations"])
        and loso["n_comparable_folds"] == 2
        and loso["supports_controlled_comparison"] is False
        and pooled["supports_controlled_comparison"] is True
        and report["verdict"] == FEASIBLE_VERDICT
        and report["augmentation_feasibility"]["candidate_scheme"] == POOLED_SCHEME
    )
    starved_spec = tuple(entry for entry in VIDEO_SPEC if entry[4] in (3, 4))
    starved = _report(starved_spec)
    starved_ok = (
        starved["verdict"] == NOT_FEASIBLE_VERDICT
        and _scheme(starved, POOLED_SCHEME)["folds"] == []
        and _scheme(starved, LOSO_SCHEME)["n_comparable_folds"] == 0
        and starved["augmentation_feasibility"]["candidate_scheme"] is None
        and starved["augmentation_feasibility"]["data_blockers"]
    )
    summary = subject_fold_feasibility.render_summary(starved)
    return _check(
        "classe ausente em um lado torna a macro-F1 de oito classes não comparável; classes de um único sujeito "
        "só são marcadas em lados com vários sujeitos; sem dobras completas o veredito é not_feasible",
        flags_ok and rare_ok and starved_ok and "não forma pelo menos dois grupos" in summary,
    )


def _selftest_reads_only_official_train() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        frames_path, manifest_path = _write_inputs(Path(tmp))
        frames = read_train_partition(frames_path)
        manifest = read_train_partition(manifest_path)
    report: dict = build_report(frames, manifest)
    return _check(
        "a leitura filtra split == train: ambientes, sujeitos e rótulos de val/test nunca entram no relatório",
        set(frames["split"]) == {"train"}
        and set(manifest["split"]) == {"train"}
        and 9 not in report["subject_identity"]["raw_subjects"]
        and not set(frames["label"]) & {5, 6}
        and report["official_splits_read"] == ["train"],
    )


def _selftest_missing_prerequisites() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "out"
        written = run_assessment(root / "absent" / "frames.parquet", root / "absent" / "manifest.parquet", output)
        report = json.loads((output / FEASIBILITY_JSON_FILE).read_text(encoding="utf-8"))
        summary = (output / FEASIBILITY_SUMMARY_FILE).read_text(encoding="utf-8")
    return _check(
        "dados locais ausentes geram relatório com os pré-requisitos faltantes e nenhum número calculado",
        written
        and report["status"] == "missing_prerequisites"
        and report["verdict"] == NOT_FEASIBLE_VERDICT
        and len(report["missing_prerequisites"]) == 2
        and "subject_identity" not in report
        and "input_sha256" not in report
        and "Nenhum número foi calculado" in summary,
    )


def _selftest_deterministic_and_read_only() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames_path, manifest_path = _write_inputs(root)
        before = {path: sha256_file(path) for path in (frames_path, manifest_path)}
        first, second = root / "out1", root / "out2"
        with patch.object(subject_fold_feasibility, "EXPECTED_CS_TRAIN", SYNTHETIC_EXPECTED):
            written = run_assessment(frames_path, manifest_path, first) and run_assessment(
                frames_path, manifest_path, second
            )
        names = (FEASIBILITY_JSON_FILE, FEASIBILITY_SUMMARY_FILE)
        identical = all((first / name).read_bytes() == (second / name).read_bytes() for name in names)
        report = json.loads((first / FEASIBILITY_JSON_FILE).read_text(encoding="utf-8"))
        after = {path: sha256_file(path) for path in (frames_path, manifest_path)}
    contract = (
        report["canonical"] is False
        and report["read_only"] is True
        and report["stage"] == "4A"
        and set(report["input_sha256"].values()) == set(before.values())
    )
    return _check(
        "duas execuções sobre as mesmas entradas gravam bytes idênticos, registram os SHA-256 e não alteram as "
        "entradas",
        written and identical and contract and before == after,
    )


def _selftest_protected_outputs() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames_path, manifest_path = _write_inputs(root)
        output = root / "out"
        protected = [
            REFERENCE_RUN_ROOT / "subject_folds",
            frames_path.parent / "out",
            REPOSITORY_ROOT / "docs" / "subject_folds",
            REPOSITORY_ROOT / "src" / "subject_folds",
            REPOSITORY_ROOT / "data" / "subject_folds",
        ]
        outside_local = REPOSITORY_ROOT / "runs" / "subject_folds_selftest"
        with patch.object(subject_fold_feasibility, "EXPECTED_CS_TRAIN", SYNTHETIC_EXPECTED):
            guards = all(
                _raises(ValueError, lambda target=target: run_assessment(frames_path, manifest_path, target), "protegido")
                for target in protected
            )
            local_only = _raises(
                ValueError, lambda: run_assessment(frames_path, manifest_path, outside_local), "runs/local"
            )
            run_assessment(frames_path, manifest_path, output)
            (output / FEASIBILITY_SUMMARY_FILE).write_text("sentinela", encoding="utf-8")
            skipped = not run_assessment(frames_path, manifest_path, output)
            preserved = (output / FEASIBILITY_SUMMARY_FILE).read_text(encoding="utf-8") == "sentinela"
            forced = run_assessment(frames_path, manifest_path, output, force=True)
            replaced = (output / FEASIBILITY_SUMMARY_FILE).read_text(encoding="utf-8") != "sentinela"
    nothing_created = not any(path.exists() for path in [*protected, outside_local])
    return _check(
        "a saída recusa referências, src/, data/, docs/, o diretório das entradas e o repositório fora de "
        "runs/local; sem --force nada é sobrescrito",
        guards and local_only and skipped and preserved and forced and replaced and nothing_created,
    )


def run_subject_fold_feasibility_selftest() -> bool:
    checks = [
        _selftest_identity_audit(),
        _selftest_ambiguous_identities(),
        _selftest_fold_isolation(),
        _selftest_scheme_construction(),
        _selftest_class_support_failures(),
        _selftest_reads_only_official_train(),
        _selftest_missing_prerequisites(),
        _selftest_deterministic_and_read_only(),
        _selftest_protected_outputs(),
    ]
    ok = all(checks)
    if ok:
        print("\nsubject_fold_feasibility selftest OK: todas as checagens passaram")
    else:
        print("\nsubject_fold_feasibility selftest FALHOU", file=sys.stderr)
    return ok


def run_selftest() -> None:
    if not run_subject_fold_feasibility_selftest():
        sys.exit(1)
