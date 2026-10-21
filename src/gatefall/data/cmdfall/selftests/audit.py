"""Selftest sintético da auditoria de metadados do CMDFall (`metadata.py`, `source.py`).

Não acessa a rede nem os CSVs reais: os segmentos são montados em memória no
layout `colors/S<sessão>P<sujeito>K<câmera>` do OmniFall, com os desvios que a
auditoria precisa detectar (vista divergente, vista ausente, gaps,
sobreposições, segmento contido e splits que vazam sessão ou sujeito).
"""

import sys
import tempfile
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.config import TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.cmdfall.metadata import (
    UNASSIGNED_SPLIT,
    annotate_segments,
    boundary_summary,
    cs_split_problems,
    grid_summary,
    inventory,
    label_summary,
    schema_problems,
    session_split_map,
    shared_units_across_splits,
    single_view,
    split_coverage,
    taxonomy_problems,
    video_table,
    view_duplication,
    window_eligibility,
)
from gatefall.data.cmdfall.source import source_problems
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.hashing import sha256_file

WALK, FALL, FALLEN, STANDING, OTHER = 0, 1, 2, 8, 9


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _cell(frame: pd.DataFrame, key: tuple[str, str], column: str) -> int:
    return int(cast(int, frame.at[key, column]))


def _rows(session: int, subject: int, cams: list[int], segments: list[tuple[int, float, float]]):
    return [
        {
            "path": f"colors/S{session}P{subject}K{cam}",
            "label": label,
            "start": start,
            "end": end,
            "subject": subject,
            "cam": cam,
            "dataset": "cmdfall",
        }
        for cam in cams
        for label, start, end in segments
    ]


# Sessão 1: walk [0, 3) e fall [3, 4) em duas vistas idênticas.
SESSION_1 = [(WALK, 0.0, 3.0), (FALL, 3.0, 4.0)]
# Sessão 2: início em 1 s, fronteira com jitter de 0,0001 s, gap de 0,5 s,
# sobreposição de 0,3 s (walk→fallen) e um segmento contido em fallen.
SESSION_2 = [
    (STANDING, 1.0, 2.0),
    (OTHER, 2.0001, 4.0),
    (WALK, 4.5, 6.0),
    (FALLEN, 5.7, 8.0),
    (STANDING, 6.0, 7.0),
]


def _synthetic_raw() -> pd.DataFrame:
    rows = _rows(1, 1, [1, 2], SESSION_1) + _rows(2, 2, [2], SESSION_2)
    return pd.DataFrame(rows)


def check_schema() -> list[bool]:
    raw = _synthetic_raw()
    broken = raw.copy()
    broken.loc[0, "path"] = "colors/S1P1"
    broken.loc[1, "subject"] = 7
    broken.loc[2, "cam"] = 9
    broken.loc[3, "label"] = 12
    broken.loc[4, "end"] = broken.loc[4, "start"]
    broken.loc[5, "dataset"] = "le2i"
    problems = schema_problems(broken)
    shared_session = pd.concat(
        [raw, pd.DataFrame(_rows(1, 3, [1], [(WALK, 0.0, 1.0)]))], ignore_index=True
    )
    label2id = pd.DataFrame({"label": list(LE2I_LABEL_NAMES), "id": range(len(LE2I_LABEL_NAMES))})
    swapped = label2id.copy()
    swapped.loc[[0, 1], "label"] = ["fall", "walk"]
    return [
        _check("CSV sintético válido não tem problema de schema", schema_problems(raw) == []),
        _check(
            "path inválido, subject/cam divergentes do path, label fora de 0..9, "
            "end <= start e dataset errado são relatados",
            len(problems) == 6,
        ),
        _check(
            "sessão com dois sujeitos é relatada",
            any("mais de um sujeito" in problem for problem in schema_problems(shared_session)),
        ),
        _check(
            "label2id na ordem do GateFall passa; ids trocados falham",
            taxonomy_problems(label2id) == [] and len(taxonomy_problems(swapped)) == 2,
        ),
        _check(
            "coluna ausente interrompe a validação",
            schema_problems(raw.drop(columns=["cam"])) == ["colunas ausentes: ['cam']"],
        ),
    ]


def check_views() -> list[bool]:
    segments = annotate_segments(_synthetic_raw())
    divergent_raw = _synthetic_raw()
    divergent_raw.loc[
        (divergent_raw["path"] == "colors/S1P1K2") & (divergent_raw["label"] == FALL), "end"
    ] = 4.5
    divergent = view_duplication(annotate_segments(divergent_raw))
    duplication = view_duplication(segments)
    sv = single_view(segments)
    summary = inventory(segments)
    labels = label_summary(segments)
    return [
        _check(
            "vistas idênticas: 2 sessões sem divergência; câmera 1 ausente na sessão 2",
            duplication.identical_sessions == 2
            and duplication.divergent_sessions == []
            and duplication.missing_views == {2: [1]},
        ),
        _check("vista com um fim deslocado marca a sessão como divergente", divergent.divergent_sessions == [1]),
        _check(
            "uma vista por sessão usa a menor câmera e remove a duplicata",
            len(sv) == len(SESSION_1) + len(SESSION_2)
            and set(sv["path"]) == {"colors/S1P1K1", "colors/S2P2K2"},
        ),
        _check(
            "inventário conta vídeos, sessões, sujeitos e vistas por sessão",
            summary["videos"] == 3
            and summary["sessions"] == 2
            and summary["subjects"] == 2
            and summary["sessions_by_view_count"] == {1: 1, 2: 1},
        ),
        _check(
            "rótulos: segmentos SV, MV e duração em uma vista",
            int(labels.loc["walk", "segments"]) == 2
            and int(labels.loc["walk", "mv_segments"]) == 3
            and abs(float(labels.loc["walk", "hours"]) * 3600 - 4.5) < 1e-9
            and int(labels.loc["walk", "segments_ge_window"]) == 1
            and int(labels.loc["standing", "sessions"]) == 1,
        ),
    ]


def check_boundaries() -> list[bool]:
    summary = boundary_summary(single_view(annotate_segments(_synthetic_raw())))
    return [
        _check(
            "jitter de 0,0001 s conta como fronteira contígua",
            summary.boundaries == 5 and summary.contiguous == 2,
        ),
        _check(
            "gap de 0,5 s, sobreposição de 0,3 s e segmento contido ficam acima de um passo da grade",
            summary.gaps_at_least_step == 1
            and summary.overlaps_at_least_step == 2
            and summary.large_overlap_pairs == {"fallen->standing": 1, "walk->fallen": 1},
        ),
        _check(
            "segmento contido no anterior é contado e não vira gap",
            summary.contained_segments == 1 and summary.sessions_with_overlap == 1,
        ),
        _check(
            "início em 1 s é deslocamento inicial, não gap interior",
            summary.sessions_with_leading_offset == 1
            and abs(summary.leading_offset_s - 1.0) < 1e-9
            and abs(summary.interior_gap_s - 0.5001) < 1e-6,
        ),
    ]


def check_windows() -> list[bool]:
    sv = single_view(annotate_segments(_synthetic_raw()))
    session_split = {1: "train", 2: "test"}
    windows = window_eligibility(sv, session_split)
    grid = grid_summary(sv, session_split)
    walk = windows.loc[(1, "train", "walk")]
    fall = windows.loc[(1, "train", "fall")]
    strided_walk = windows.loc[(TRAIN_STRIDE, "train", "walk")]
    test_standing = windows.loc[(1, "test", "standing")]
    return [
        _check(
            f"walk [0, 3 s) a 10 fps: 30 janelas, todas puras, {WINDOW_FRAMES - 1} com replicação de borda",
            int(walk["windows"]) == 30
            and int(walk["pure"]) == 30
            and int(walk["padded"]) == WINDOW_FRAMES - 1,
        ),
        _check(
            "fall de 1 s depois de walk: 10 janelas, nenhuma pura",
            int(fall["windows"]) == 10 and int(fall["pure"]) == 0,
        ),
        _check(
            f"stride {TRAIN_STRIDE} mantém k_end 0, 4, ..., 28 de walk",
            int(strided_walk["windows"]) == 8,
        ),
        _check(
            "quadros antes do primeiro segmento, no gap e sob o jitter que cruza t=2,0 s são IGNORE",
            int(test_standing["with_ignore"]) == int(test_standing["windows"]) == 10
            and int(grid.loc["test", "ignore_frames"]) == 10 + 1 + 5,
        ),
        _check(
            "sobreposição resolvida pelo menor início, como em labels_for_grid",
            int(grid.loc["test", "overlap_resolved_frames"]) == 3 + 10,
        ),
    ]


def check_splits() -> list[bool]:
    segments = annotate_segments(
        pd.concat(
            [_synthetic_raw(), pd.DataFrame(_rows(3, 2, [1], [(WALK, 0.0, 1.0)]))],
            ignore_index=True,
        )
    )
    videos = video_table(segments)
    clean = {
        "train": pd.DataFrame({"path": ["colors/S1P1K1", "colors/S1P1K2"]}),
        "val": pd.DataFrame({"path": []}),
        "test": pd.DataFrame({"path": ["colors/S2P2K2", "colors/S3P2K1"]}),
    }
    leaking = {
        "train": pd.DataFrame({"path": ["colors/S1P1K1", "colors/S3P2K1"]}),
        "val": pd.DataFrame({"path": ["colors/S1P1K2"]}),
        "test": pd.DataFrame({"path": ["colors/S2P2K2", "colors/S2P2K2", "colors/S9P9K9"]}),
    }
    problems = cs_split_problems(videos, leaking)
    by_view = {
        "train": pd.DataFrame({"path": ["colors/S1P1K1", "colors/S3P2K1"]}),
        "val": pd.DataFrame({"path": ["colors/S1P1K2"]}),
        "test": pd.DataFrame({"path": ["colors/S2P2K2"]}),
    }
    session_split = session_split_map(videos, by_view)
    coverage = split_coverage(single_view(segments), session_split_map(videos, clean))
    return [
        _check("splits por sujeito sem vazamento passam", cs_split_problems(videos, clean) == []),
        _check(
            "path repetido, path desconhecido, sessão e sujeito em dois splits são relatados",
            any("repetidos" in problem for problem in problems)
            and any("sem segmento" in problem for problem in problems)
            and any("session" in problem for problem in problems)
            and any("subject" in problem for problem in problems),
        ),
        _check(
            "sessão com vistas em splits diferentes não recebe split",
            session_split[1] == UNASSIGNED_SPLIT and session_split[2] == "test",
        ),
        _check(
            "split por câmera compartilha sessões e sujeitos entre splits",
            shared_units_across_splits(videos, by_view) == {"session": 1, "subject": 2},
        ),
        _check(
            "cobertura por split conta sessões e sujeitos de cada classe",
            _cell(coverage, ("test", "walk"), "sessions") == 2
            and _cell(coverage, ("test", "walk"), "subjects") == 1
            and _cell(coverage, ("val", "walk"), "segments") == 0,
        ),
    ]


def check_sources() -> list[bool]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "labels").mkdir()
        good = root / "labels" / "a.csv"
        good.write_text("path\n")
        expected = {"labels/a.csv": sha256_file(good), "labels/b.csv": "0" * 64}
        problems = source_problems(root, expected)
        good.write_text("path\nx\n")
        tampered = source_problems(root, {"labels/a.csv": expected["labels/a.csv"]})
    return [
        _check(
            "fonte ausente é relatada; fonte com SHA-256 esperado passa",
            len(problems) == 1 and "ausente" in problems[0],
        ),
        _check("fonte alterada depois do pin é relatada", len(tampered) == 1 and "divergente" in tampered[0]),
    ]


def run_audit_selftest() -> None:
    results = (
        check_schema()
        + check_views()
        + check_boundaries()
        + check_windows()
        + check_splits()
        + check_sources()
    )
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passaram")
    if passed != len(results):
        sys.exit(1)
