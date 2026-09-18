"""Selftest sintético das estatísticas de duração de segmentos (`verification.py`).

Não toca no manifesto real — os splits são sintéticos, no formato
`path,label,start,end,subject,cam,dataset`. O foco é travar o comportamento
de `compute_segment_duration_stats` (interpolação de quantil do pandas) e
garantir que o relatório train-only nunca seja afetado por val/test.
"""

import contextlib
import io
import sys
from typing import cast

import pandas as pd

from gatefall.data.le2i.verification import (
    compute_segment_duration_stats,
    report_segment_duration_by_class_other_splits,
    report_subject_disjointness,
    report_train_duration_by_class,
)

_FALL_LABEL = 1
_COLUMNS = ["path", "label", "start", "end", "subject", "cam", "dataset"]


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _make_segments(
    labels: list[int], starts: list[float], ends: list[float]
) -> pd.DataFrame:
    n = len(labels)
    return pd.DataFrame(
        {
            "path": [f"video_{i}.avi" for i in range(n)],
            "label": labels,
            "start": starts,
            "end": ends,
            "subject": [i % 3 for i in range(n)],
            "cam": ["cam1"] * n,
            "dataset": ["le2i-cs"] * n,
        }
    )


def _make_empty_segments() -> pd.DataFrame:
    return pd.DataFrame({column: [] for column in _COLUMNS}).astype(
        {"label": "int64", "start": "float64", "end": "float64"}
    )


def check_known_value_quantiles() -> bool:
    # durações 1,2,3,4 -> pinam a interpolação linear padrão do pandas
    # (p25=1.75, p75=3.25) contra uma futura troca silenciosa de método.
    segments = _make_segments(
        labels=[_FALL_LABEL] * 4,
        starts=[0.0, 0.0, 0.0, 0.0],
        ends=[1.0, 2.0, 3.0, 4.0],
    )
    stats = compute_segment_duration_stats(segments)
    row = stats.loc[_FALL_LABEL]
    ok = (
        int(cast(int, row["count"])) == 4
        and float(cast(float, row["min"])) == 1.0
        and float(cast(float, row["p25"])) == 1.75
        and float(cast(float, row["median"])) == 2.5
        and float(cast(float, row["p75"])) == 3.25
        and float(cast(float, row["max"])) == 4.0
    )
    return _check(
        "valores conhecidos: fall=[1,2,3,4] -> count=4, min=1.0, p25=1.75, "
        "median=2.5, p75=3.25, max=4.0",
        ok,
    )


def check_train_report_independent_of_other_splits() -> bool:
    train = _make_segments(
        labels=[_FALL_LABEL, _FALL_LABEL, 0],
        starts=[0.0, 0.0, 0.0],
        ends=[1.5, 2.5, 5.0],
    )

    splits_empty_others = {
        "train": train,
        "val": _make_empty_segments(),
        "test": _make_empty_segments(),
    }
    splits_extreme_others = {
        "train": train.copy(deep=True),
        "val": _make_segments(
            labels=[_FALL_LABEL], starts=[0.0], ends=[9999.0]
        ),
        "test": _make_segments(
            labels=[_FALL_LABEL], starts=[0.0], ends=[0.0001]
        ),
    }

    with contextlib.redirect_stdout(io.StringIO()) as buffer_empty:
        report_train_duration_by_class(splits_empty_others)
    with contextlib.redirect_stdout(io.StringIO()) as buffer_extreme:
        report_train_duration_by_class(splits_extreme_others)

    ok = buffer_empty.getvalue() == buffer_extreme.getvalue()
    return _check(
        "relatório train-only: val/test byte-diferentes (vazio vs. outlier "
        "extremo) não mudam uma linha da saída, pois train é idêntico",
        ok,
    )


def check_empty_dataframe_returns_without_raising() -> bool:
    empty = _make_empty_segments()
    stats = compute_segment_duration_stats(empty)
    ok = bool(stats.empty)
    return _check(
        "dataframe vazio: compute_segment_duration_stats não levanta exceção "
        "e retorna resultado vazio",
        ok,
    )


def check_absent_fall_label_returns_without_raising() -> bool:
    segments = _make_segments(labels=[0, 0], starts=[0.0, 0.0], ends=[1.0, 2.0])
    stats = compute_segment_duration_stats(segments)
    ok = _FALL_LABEL not in stats.index
    return _check(
        "sem segmentos fall: compute_segment_duration_stats não levanta "
        "exceção e o rótulo fall fica ausente no resultado",
        ok,
    )


def check_other_splits_report_runs_without_raising() -> bool:
    splits = {
        "train": _make_segments(
            labels=[_FALL_LABEL], starts=[0.0], ends=[2.0]
        ),
        "val": _make_segments(labels=[_FALL_LABEL], starts=[0.0], ends=[3.0]),
        "test": _make_empty_segments(),
    }
    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        report_segment_duration_by_class_other_splits(splits)
    return _check(
        "relatório val/test/pooled roda sem exceção mesmo com um split vazio",
        len(buffer.getvalue()) > 0,
    )


def check_cs_overlap_reports_cross_subject_undercut() -> bool:
    train = _make_segments(labels=[_FALL_LABEL], starts=[0.0], ends=[1.0])
    test = _make_segments(labels=[_FALL_LABEL], starts=[0.0], ends=[1.0])
    splits = {"train": train, "test": test}

    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        report_subject_disjointness(splits, protocol="cs")
    output = buffer.getvalue()

    ok = (
        "não é verdadeiramente cross-subject" in output
        and "esperado no protocolo le2i-cv" not in output
    )
    return _check(
        "le2i-cs com subjects sobrepostos: conclusão aponta violação de "
        "cross-subject",
        ok,
    )


def check_cv_overlap_reports_expected_and_informative() -> bool:
    train = _make_segments(labels=[_FALL_LABEL], starts=[0.0], ends=[1.0])
    test = _make_segments(labels=[_FALL_LABEL], starts=[0.0], ends=[1.0])
    splits = {"train": train, "test": test}

    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        report_subject_disjointness(splits, protocol="cv")
    output = buffer.getvalue()

    ok = (
        "esperado no protocolo le2i-cv" in output
        and "informativo" in output
        and "não é verdadeiramente cross-subject" not in output
    )
    return _check(
        "le2i-cv com subjects sobrepostos: conclusão indica overlap "
        "esperado e informativo, sem alegar quebra do protocolo",
        ok,
    )


def check_disjoint_subjects_report_identical_across_protocols() -> bool:
    train = _make_segments(labels=[_FALL_LABEL], starts=[0.0], ends=[1.0])
    test = cast(pd.DataFrame, train.copy(deep=True))
    test["subject"] = test["subject"] + 100
    splits = {"train": train, "test": test}

    with contextlib.redirect_stdout(io.StringIO()) as buffer_cs:
        report_subject_disjointness(splits, protocol="cs")
    with contextlib.redirect_stdout(io.StringIO()) as buffer_cv:
        report_subject_disjointness(splits, protocol="cv")

    expected = "conclusão: os conjuntos de subjects são disjuntos entre todos os splits."
    ok = expected in buffer_cs.getvalue() and expected in buffer_cv.getvalue()
    return _check(
        "sem overlap de subjects: cs e cv produzem a mesma conclusão de "
        "disjunção",
        ok,
    )


def check_train_duration_heading_differs_by_protocol() -> bool:
    splits = {
        "train": _make_segments(
            labels=[_FALL_LABEL], starts=[0.0], ends=[2.0]
        )
    }

    with contextlib.redirect_stdout(io.StringIO()) as buffer_cs:
        report_train_duration_by_class(splits, protocol="cs")
    with contextlib.redirect_stdout(io.StringIO()) as buffer_cv:
        report_train_duration_by_class(splits, protocol="cv")

    output_cs = buffer_cs.getvalue()
    output_cv = buffer_cv.getvalue()

    ok = (
        "evidência de seleção de WINDOW_FRAMES" in output_cs
        and "diagnóstico apenas" not in output_cs
        and "diagnóstico apenas" in output_cv
        and "WINDOW_FRAMES=24" in output_cv
        and "congelado" in output_cv
        and "evidência de seleção de WINDOW_FRAMES" not in output_cv
    )
    return _check(
        "cabeçalho de duração train difere entre cs (evidência de seleção) "
        "e cv (diagnóstico/WINDOW_FRAMES congelado)",
        ok,
    )


def check_train_duration_numbers_identical_across_protocols() -> bool:
    splits = {
        "train": _make_segments(
            labels=[_FALL_LABEL] * 4,
            starts=[0.0, 0.0, 0.0, 0.0],
            ends=[1.0, 2.0, 3.0, 4.0],
        )
    }

    with contextlib.redirect_stdout(io.StringIO()) as buffer_cs:
        report_train_duration_by_class(splits, protocol="cs")
    with contextlib.redirect_stdout(io.StringIO()) as buffer_cv:
        report_train_duration_by_class(splits, protocol="cv")

    expected_line = (
        "fall (train): count=4, min=1.0000, p25=1.7500, median=2.5000, "
        "p75=3.2500, max=4.0000"
    )
    ok = expected_line in buffer_cs.getvalue() and expected_line in buffer_cv.getvalue()
    return _check(
        "estatísticas de duração idênticas entre cs e cv para o mesmo "
        "insumo (cálculo não passou a depender do protocolo)",
        ok,
    )


def run_verification_selftest() -> None:
    checks = [
        check_known_value_quantiles(),
        check_train_report_independent_of_other_splits(),
        check_empty_dataframe_returns_without_raising(),
        check_absent_fall_label_returns_without_raising(),
        check_other_splits_report_runs_without_raising(),
        check_cs_overlap_reports_cross_subject_undercut(),
        check_cv_overlap_reports_expected_and_informative(),
        check_disjoint_subjects_report_identical_across_protocols(),
        check_train_duration_heading_differs_by_protocol(),
        check_train_duration_numbers_identical_across_protocols(),
    ]
    if not all(checks):
        print("\nverification selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nverification selftest OK: todos os casos passaram")
