"""Selftest sintético da auditoria de metadados do CMDFall (`metadata.py`, `source.py`).

Não acessa a rede nem os CSVs reais: os segmentos são montados em memória no
layout `colors/S<sessão>P<sujeito>K<câmera>` do OmniFall, com os desvios que a
auditoria precisa detectar (vista divergente, vista ausente, gaps,
sobreposições, segmento contido e splits que vazam sessão ou sujeito). As
verificações de publicação gravam esses CSVs num repositório temporário, com
áreas protegidas e links simbólicos próprios, nunca no checkout real.
"""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest import mock

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
from gatefall.data.cmdfall.audit import run_report, write_report
from gatefall.data.cmdfall.source import (
    CS_SPLIT_FILES,
    CV_SPLIT_FILES,
    DEFAULT_REPORT_PATH,
    DEFAULT_SOURCE_DIR,
    LABEL2ID_FILE,
    SCRATCH_ROOT,
    SEGMENTS_FILE,
    destination_problems,
    fetch_sources,
    source_problems,
)
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


VALID_CS_SPLITS = {
    "train": ["colors/S1P1K1", "colors/S1P1K2"],
    "val": [],
    "test": ["colors/S2P2K2"],
}
CV_SPLITS = {"train": ["colors/S1P1K1"], "val": ["colors/S1P1K2"], "test": ["colors/S2P2K2"]}
CANONICAL_FILES = (
    "runs/reference/baseline_a/metrics.json",
    "src/gatefall/report.json",
    "data/labels/omnifall/report.json",
)


def _write_world(root: Path, raw: pd.DataFrame, cs_splits: dict[str, list[str]]) -> dict[str, str]:
    """Monta um repositório temporário com as fontes no layout do OmniFall e
    devolve o SHA-256 de cada CSV, que o selftest usa como pin."""
    label2id = pd.DataFrame({"label": list(LE2I_LABEL_NAMES), "id": range(len(LE2I_LABEL_NAMES))})
    files: dict[str, pd.DataFrame] = {SEGMENTS_FILE: raw, LABEL2ID_FILE: label2id}
    files |= {CS_SPLIT_FILES[name]: pd.DataFrame({"path": paths}) for name, paths in cs_splits.items()}
    files |= {CV_SPLIT_FILES[name]: pd.DataFrame({"path": paths}) for name, paths in CV_SPLITS.items()}
    source = root / DEFAULT_SOURCE_DIR
    for name, frame in files.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(source / name, index=False)
    for relative in CANONICAL_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("canonical\n")
    return {name: sha256_file(source / name) for name in files}


def _quiet_report(root: Path, expected: dict[str, str], output: Path = DEFAULT_REPORT_PATH) -> list[str]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_report(root / DEFAULT_SOURCE_DIR, output, root, expected)


def _report_files(root: Path) -> list[Path]:
    return sorted(path for path in (root / SCRATCH_ROOT).iterdir() if path.is_file())


def check_report_publication() -> list[bool]:
    divergent = _synthetic_raw()
    divergent.loc[(divergent["path"] == "colors/S1P1K2") & (divergent["label"] == FALL), "end"] = 4.5
    out_of_taxonomy = _synthetic_raw()
    out_of_taxonomy.loc[0, "label"] = 12
    leaking_splits = {**VALID_CS_SPLITS, "train": ["colors/S1P1K1"], "val": ["colors/S1P1K2"]}
    invalid_cases = {
        "vistas sincronizadas divergentes": (divergent, VALID_CS_SPLITS),
        "split cs com sessão e sujeito em dois splits": (_synthetic_raw(), leaking_splits),
        "label fora da taxonomia": (out_of_taxonomy, VALID_CS_SPLITS),
    }
    results: list[bool] = []
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        valid = base / "valid"
        problems = _quiet_report(valid, _write_world(valid, _synthetic_raw(), VALID_CS_SPLITS))
        report_path = valid / DEFAULT_REPORT_PATH
        results.append(
            _check(
                "metadados válidos publicam o relatório",
                problems == []
                and json.loads(report_path.read_text())["inventory"]["sessions"] == 2
                and _report_files(valid) == [report_path],
            )
        )

        for index, (name, (raw, splits)) in enumerate(invalid_cases.items()):
            root = base / f"invalid_{index}"
            problems = _quiet_report(root, _write_world(root, raw, splits))
            results.append(
                _check(
                    f"{name}: report falha sem gravar audit.json nem temporário",
                    any(problem.startswith("metadados:") for problem in problems)
                    and _report_files(root) == [],
                )
            )

        stale = base / "stale"
        expected = _write_world(stale, divergent, VALID_CS_SPLITS)
        previous = stale / DEFAULT_REPORT_PATH
        previous.write_text("relatório anterior\n")
        problems = _quiet_report(stale, expected)
        results.append(
            _check(
                "relatório anterior fica intacto quando a nova validação falha",
                bool(problems) and previous.read_text() == "relatório anterior\n",
            )
        )

        tampered = base / "tampered"
        expected = _write_world(tampered, _synthetic_raw(), VALID_CS_SPLITS)
        (tampered / DEFAULT_SOURCE_DIR / CS_SPLIT_FILES["val"]).write_text("path\ncolors/S1P1K2\n")
        problems = _quiet_report(tampered, expected)
        results.append(
            _check(
                "fonte alterada depois do pin falha antes de calcular e não grava relatório",
                any(problem.startswith("fontes:") for problem in problems)
                and _report_files(tampered) == [],
            )
        )
    return results


def check_destination_guard() -> list[bool]:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        root = base / "repo"
        expected = _write_world(root, _synthetic_raw(), VALID_CS_SPLITS)
        outside = base / "outside.json"
        outside.write_text("fora do repositório\n")
        (root / SCRATCH_ROOT / "escape").symlink_to(root / "runs/reference", target_is_directory=True)
        (root / SCRATCH_ROOT / "link.json").symlink_to(root / CANONICAL_FILES[0])
        guarded = [root / relative for relative in CANONICAL_FILES] + [outside]
        snapshot = {path: path.read_bytes() for path in guarded}
        tree_before = sorted(str(path) for path in base.rglob("*"))

        rejected = [
            *(Path(relative) for relative in CANONICAL_FILES),
            Path("runs/reference/new.json"),
            outside,
            SCRATCH_ROOT / "../../labels/omnifall/report.json",
            SCRATCH_ROOT / "escape/new.json",
            SCRATCH_ROOT / "link.json",
            SCRATCH_ROOT / "audit.txt",
            SCRATCH_ROOT,
            DEFAULT_SOURCE_DIR / "audit.json",
        ]
        outcomes = {destination: _quiet_report(root, expected, destination) for destination in rejected}
        unchanged = all(path.read_bytes() == content for path, content in snapshot.items())
        tree_after = sorted(str(path) for path in base.rglob("*"))

        nested = SCRATCH_ROOT / "runs/audit.json"
        accepted = _quiet_report(root, expected, nested) == [] and (root / nested).is_file()

        symlinked_root = base / "symlinked"
        (symlinked_root / "runs/reference").mkdir(parents=True)
        (symlinked_root / "data/scratch").mkdir(parents=True)
        (symlinked_root / SCRATCH_ROOT).symlink_to(
            symlinked_root / "runs/reference", target_is_directory=True
        )
        symlinked_problems = destination_problems(DEFAULT_REPORT_PATH, symlinked_root)

    return [
        _check(
            "destinos protegidos, externos, com .., por link simbólico, sem .json, "
            "diretório ou dentro das fontes são recusados",
            all(
                problems and all(problem.startswith("destino:") for problem in problems)
                for problems in outcomes.values()
            ),
        ),
        _check(
            "destinos recusados ficam intactos e nenhum arquivo novo aparece",
            unchanged and tree_before == tree_after,
        ),
        _check(
            "links simbólicos para runs/reference são resolvidos e acusados como área protegida",
            all(
                any("área protegida runs/reference" in problem for problem in outcomes[destination])
                for destination in (SCRATCH_ROOT / "escape/new.json", SCRATCH_ROOT / "link.json")
            ),
        ),
        _check("subdiretório de data/scratch/cmdfall continua aceito", accepted),
        _check(
            "data/scratch/cmdfall apontando para runs/reference é recusado",
            any("área protegida runs/reference" in problem for problem in symlinked_problems),
        ),
        _check(
            "fetch só grava fontes dentro de data/scratch/cmdfall",
            destination_problems(DEFAULT_SOURCE_DIR, root) == []
            and bool(destination_problems(Path("runs/reference/omnifall"), root))
            and bool(destination_problems(Path("data/labels/cmdfall"), root)),
        ),
    ]


def _write_canonical(root: Path) -> None:
    for relative in CANONICAL_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("canonical\n")


def _canonical_bytes(root: Path) -> dict[str, bytes]:
    return {relative: (root / relative).read_bytes() for relative in CANONICAL_FILES}


def _tree(base: Path) -> list[str]:
    return sorted(str(path) for path in base.rglob("*"))


def _leftover_temporaries(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*.tmp") if not path.is_symlink())


def check_secure_publication() -> list[bool]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "repo"
        expected = _write_world(root, _synthetic_raw(), VALID_CS_SPLITS)
        canonical = _canonical_bytes(root)
        predictable = root / SCRATCH_ROOT / "audit.json.tmp"
        predictable.symlink_to(root / CANONICAL_FILES[0])
        problems = _quiet_report(root, expected)
        report_path = root / DEFAULT_REPORT_PATH
        symlink_ignored = (
            problems == []
            and json.loads(report_path.read_text())["inventory"]["videos"] == 3
            and _canonical_bytes(root) == canonical
            and predictable.is_symlink()
            and predictable.readlink() == root / CANONICAL_FILES[0]
            and _leftover_temporaries(root / SCRATCH_ROOT) == []
        )

        predictable.unlink(missing_ok=True)
        report_path.write_text("relatório anterior\n")
        with mock.patch(
            "gatefall.data.cmdfall.source.os.replace", side_effect=OSError("falha simulada")
        ), contextlib.redirect_stdout(io.StringIO()):
            try:
                write_report({"inventory": {}}, report_path)
                raised = False
            except OSError:
                raised = True
        preserved = (
            raised
            and report_path.read_text() == "relatório anterior\n"
            and _leftover_temporaries(root / SCRATCH_ROOT) == []
        )
    return [
        _check(
            "link simbólico em audit.json.tmp para runs/reference não é seguido nem tocado",
            symlink_ignored,
        ),
        _check(
            "falha na troca atômica remove o temporário e preserva o relatório anterior",
            preserved,
        ),
    ]


def _fake_hub(payloads: dict[str, bytes], calls: list[str]) -> Callable[[str, Path], Path]:
    def download(filename: str, staging_dir: Path) -> Path:
        calls.append(filename)
        path = staging_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[filename])
        return path

    return download


def _quiet_fetch(
    root: Path, payloads: dict[str, bytes], expected: dict[str, str], calls: list[str]
) -> list[str]:
    with contextlib.redirect_stdout(io.StringIO()):
        return fetch_sources(DEFAULT_SOURCE_DIR, root, expected, _fake_hub(payloads, calls))


def check_fetch_confinement() -> list[bool]:
    filenames = [SEGMENTS_FILE, LABEL2ID_FILE, *CS_SPLIT_FILES.values(), *CV_SPLIT_FILES.values()]
    payloads = {filename: f"path\n{filename}\n".encode() for filename in filenames}
    expected = {filename: hashlib.sha256(payload).hexdigest() for filename, payload in payloads.items()}
    source = Path(DEFAULT_SOURCE_DIR)

    def attack(name: str, base: Path, plant: Callable[[Path], None]) -> tuple[bool, str]:
        root = base / name
        _write_canonical(root)
        (root / source).mkdir(parents=True)
        plant(root)
        canonical = _canonical_bytes(root)
        tree = _tree(base)
        calls: list[str] = []
        problems = _quiet_fetch(root, payloads, expected, calls)
        ok = (
            bool(problems)
            and all(problem.startswith("destino:") for problem in problems)
            and calls == []
            and _canonical_bytes(root) == canonical
            and _tree(base) == tree
        )
        return ok, " | ".join(problems)

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        valid = base / "valid"
        _write_canonical(valid)
        calls: list[str] = []
        fetched = (
            _quiet_fetch(valid, payloads, expected, calls) == []
            and calls == filenames
            and source_problems(valid / source, expected) == []
            and _leftover_temporaries(valid / SCRATCH_ROOT) == []
        )

        outside = base / "outside"
        outside.mkdir()
        child_dir, child_dir_problems = attack(
            "child_dir",
            base,
            lambda root: (root / source / "labels").symlink_to(
                root / "runs/reference", target_is_directory=True
            ),
        )
        child_outside, _ = attack(
            "child_outside",
            base,
            lambda root: (root / source / "splits").symlink_to(outside, target_is_directory=True),
        )

        def plant_file(root: Path) -> None:
            target = root / source / CS_SPLIT_FILES["train"]
            target.parent.mkdir(parents=True)
            target.symlink_to(root / CANONICAL_FILES[0])

        child_file, child_file_problems = attack("child_file", base, plant_file)

        relocated = base / "relocated"
        _write_canonical(relocated)
        (relocated / "data/scratch/elsewhere").mkdir(parents=True)
        (relocated / SCRATCH_ROOT).symlink_to(
            relocated / "data/scratch/elsewhere", target_is_directory=True
        )
        tree = _tree(base)
        relocated_calls: list[str] = []
        relocated_problems = _quiet_fetch(relocated, payloads, expected, relocated_calls)
        relocated_rejected = (
            any("diretório físico" in problem for problem in relocated_problems)
            and relocated_calls == []
            and _tree(base) == tree
        )

        mismatch = base / "mismatch"
        _write_canonical(mismatch)
        wrong = {**expected, LABEL2ID_FILE: "0" * 64}
        mismatch_problems = _quiet_fetch(mismatch, payloads, wrong, [])
        unpublished = any(problem.startswith("fontes:") for problem in mismatch_problems) and not (
            mismatch / SCRATCH_ROOT
        ).exists()

    return [
        _check("fetch publica as fontes verificadas sem deixar temporário", fetched),
        _check(
            "subdiretório das fontes ligado a runs/reference é recusado antes do download",
            child_dir and "área protegida runs/reference" in child_dir_problems,
        ),
        _check("subdiretório das fontes ligado para fora do repositório é recusado", child_outside),
        _check(
            "CSV de destino ligado a um arquivo de runs/reference é recusado e o arquivo fica intacto",
            child_file and "área protegida runs/reference" in child_file_problems,
        ),
        _check(
            "data/scratch/cmdfall ligado a outro diretório do repositório é recusado",
            relocated_rejected,
        ),
        _check(
            "download com SHA-256 divergente não publica nenhum CSV",
            unpublished,
        ),
    ]


def run_audit_selftest() -> None:
    results = (
        check_schema()
        + check_views()
        + check_boundaries()
        + check_windows()
        + check_splits()
        + check_sources()
        + check_report_publication()
        + check_destination_guard()
        + check_secure_publication()
        + check_fetch_confinement()
    )
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passaram")
    if passed != len(results):
        sys.exit(1)
