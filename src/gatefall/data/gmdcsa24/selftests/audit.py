"""Selftest sintético da auditoria do GMDCSA24 (`source.py`, `annotations.py`, `probe.py`).

Não toca no dataset real nem no Hugging Face: o clone é montado num diretório
temporário com a árvore `Subject N/{ADL,Fall}/NN.mp4` e os CSVs por sujeito, e
os segmentos OmniFall são um DataFrame em memória. O CSV sintético reproduz os
desvios de formato observados na tag v2.1. Também roda as checagens da
revisão de proveniência em `selftests/provenance.py`. Requer FFmpeg e Git.
"""

import contextlib
import io
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gatefall.config import WINDOW_FRAMES
from gatefall.data.gmdcsa24.annotations import (
    compare_fall_onsets,
    compare_identities,
    overlap_matrix,
    parse_classes,
    parse_original_csv,
    read_original_annotations,
    summarize_coverage,
    with_label_names,
)
from gatefall.data.gmdcsa24.audit import run_inventory, run_verify
from gatefall.data.gmdcsa24.probe import (
    PROBE_COLUMNS,
    grid_row,
    max_pts_drift,
    probe_checkout,
    report_probe,
)
from gatefall.data.gmdcsa24.selftests.provenance import run_provenance_checks
from gatefall.data.gmdcsa24.source import build_inventory, find_duplicate_videos

_SHORT_FRAMES = 19
_LONG_FRAMES = 30
_HEADER = "File Name,Length (seconds),Time of Recording,Attire,Description, Classes\n"


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _encode_video(path: Path, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=32x32:rate=10:duration={n_frames / 10}",
            "-frames:v",
            str(n_frames),
            "-c:v",
            "mpeg4",
            str(path),
        ],
        check=True,
    )


def _write_synthetic_checkout(root: Path) -> None:
    _encode_video(root / "Subject 1/ADL/01.mp4", _LONG_FRAMES)
    _encode_video(root / "Subject 1/Fall/01.mp4", _SHORT_FRAMES)
    # Cópia byte a byte de outro vídeo, para a busca de duplicatas.
    (root / "Subject 2/Fall").mkdir(parents=True)
    (root / "Subject 2/Fall/01.mp4").write_bytes((root / "Subject 1/Fall/01.mp4").read_bytes())
    (root / "Subject 1/Fall/02.mp4").write_bytes(b"not a video")
    (root / "README.md").write_text("readme")
    (root / "notes.txt").write_text("fora do layout")
    # ADL lista 02.mp4, que não existe no clone.
    (root / "Subject 1/ADL.csv").write_text(
        _HEADER
        + "01.mp4,03,Day,Shirt,Walking, Walking[0 to 2]; Reading[1 to 3]\n"
        + "02.mp4,05,Day,Shirt,Sitting, Sitting[0 to 5]\n"
    )
    (root / "Subject 1/Fall.csv").write_text(
        _HEADER
        + "01.mp4,02,Night,Shirt,Fall, Fall (FW)[1 to 2]; Standing[0 to 1]\n"
        + "02.mp4,xx,Night,Shirt,Fall, Falling (BW[0.5 to 2]\n"
    )
    (root / "Subject 2/Fall.csv").write_text(
        _HEADER + "01.mp4,02,Day,Shirt,Fall, Falling (SW)[0.8 to 2]; Standing[0 to 0.7]\n"
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def _omnifall_segments() -> pd.DataFrame:
    rows = [
        ("Subject_1/ADL/01", 0, 0.0, 2.5),
        ("Subject_1/Fall/01", 8, 0.0, 0.6),
        ("Subject_1/Fall/01", 1, 0.6, 1.5),
        ("Subject_1/Fall/01", 2, 1.5, 1.9),
        ("Subject_1/Fall/02", 1, 0.5, 1.0),
        ("Subject_2/Fall/01", 1, 0.8, 1.9),
    ]
    frame = pd.DataFrame(rows, columns=["video_id", "label", "start", "end"])
    frame["subject"] = frame["video_id"].str.extract(r"^Subject_(\d+)/")[0].astype(int)
    return frame.assign(cam=1, dataset="GMDCSA24")


def _omnifall_splits(segments: pd.DataFrame) -> pd.DataFrame:
    videos = segments[["video_id", "subject"]].drop_duplicates()
    return pd.concat(
        [videos.assign(protocol=protocol, split="train") for protocol in ("cs", "cv")],
        ignore_index=True,
    )


def check_parse_classes() -> list[bool]:
    cases = {
        "end before start": parse_classes("Sitting[9.7 to 3]; Walking[0 to 1]"),
        "open paren": parse_classes("Falling (BW[5.9 to 10]; Walking[0 to 1.3]"),
        "missing to": parse_classes("Eating[0.9 4.6]"),
        "open bracket": parse_classes("Standing[2.4 to 3.6; 6.6 to 7.9; Walking[3.6 to 5.2]"),
        "no interval": parse_classes("Reading; Sitting[0.8 to 8.1]"),
        "alias": parse_classes("Fall (FW)[0 to 3]"),
        "trailing junk": parse_classes("Falling (SW)[1.8 to 5]:; Walking[0 to 1.8];"),
    }

    def spans(key: str) -> list[tuple[str, float, float]]:
        return [
            (str(item["label"]), float(str(item["start"])), float(str(item["end"])))
            for item in cases[key][0]
        ]

    return [
        _check(
            "fim antes do início é descartado e relatado",
            spans("end before start") == [("Walking", 0.0, 1.0)]
            and cases["end before start"][1] == ["Sitting: fim 3 não posterior ao início 9.7"],
        ),
        _check(
            "parêntese não fechado mantém direção e intervalo",
            cases["open paren"][0][0]["direction"] == "BW"
            and spans("open paren")[0] == ("Falling", 5.9, 10.0)
            and cases["open paren"][1] == ["Falling (BW: parêntese não fechado"],
        ),
        _check(
            "intervalo sem 'to' é lido e relatado",
            spans("missing to") == [("Eating", 0.9, 4.6)] and len(cases["missing to"][1]) == 1,
        ),
        _check(
            "colchete não fechado termina antes do próximo rótulo",
            spans("open bracket")
            == [("Standing", 2.4, 3.6), ("Standing", 6.6, 7.9), ("Walking", 3.6, 5.2)]
            and cases["open bracket"][1] == ["Standing: colchete não fechado"],
        ),
        _check(
            "rótulo sem intervalo não inventa segmento",
            spans("no interval") == [("Sitting", 0.8, 8.1)]
            and cases["no interval"][1] == ["Reading: rótulo sem intervalo"],
        ),
        _check(
            "'Fall' vira 'Falling' com aviso",
            spans("alias") == [("Falling", 0.0, 3.0)] and len(cases["alias"][1]) == 1,
        ),
        _check(
            "lixo após ']' e ';' final são ignorados sem problema",
            spans("trailing junk") == [("Falling", 1.8, 5.0), ("Walking", 0.0, 1.8)]
            and cases["trailing junk"][1] == [],
        ),
    ]


def check_original_csv() -> bool:
    parsed = parse_original_csv(
        _HEADER + "01.mp4,04,Day,Shirt,Desc, with comma, Walking[0 to 4]\n\n", 3, "ADL"
    )
    return _check(
        "campo extra vira problema e o excedente é lido como Classes",
        parsed.videos["video_id"].tolist() == ["Subject_3/ADL/01"]
        and parsed.issues["issue"].tolist() == ["número de campos"]
        and parsed.segments["label"].tolist() == ["Walking"],
    )


def check_inventory(root: Path) -> list[bool]:
    inventory = build_inventory(root)
    original = read_original_annotations(root, inventory.annotation_files)
    omnifall = _omnifall_segments()
    problems = compare_identities(
        {
            "clone": set(inventory.videos["video_id"]),
            "CSV original": set(original.videos["video_id"]),
            "OmniFall labels": set(omnifall["video_id"]),
        }
    )
    with contextlib.redirect_stdout(io.StringIO()):
        inventory_ok = run_inventory(root, omnifall, _omnifall_splits(omnifall))
    return [
        _check(
            "vídeos, CSVs e arquivos fora do layout separados; .git ignorado",
            inventory.videos["video_id"].tolist()
            == ["Subject_1/ADL/01", "Subject_1/Fall/01", "Subject_1/Fall/02", "Subject_2/Fall/01"]
            and len(inventory.annotation_files) == 3
            and inventory.unrecognized == ["notes.txt"],
        ),
        _check(
            "cópia byte a byte agrupada por SHA-256",
            find_duplicate_videos(root, inventory.videos)
            == [["Subject_1/Fall/01", "Subject_2/Fall/01"]],
        ),
        _check(
            "vídeo listado no CSV e ausente do clone e do OmniFall é relatado",
            problems
            == [
                "Subject_1/ADL/02: ausente em clone",
                "Subject_1/ADL/02: ausente em OmniFall labels",
            ],
        ),
        _check(
            "Length ilegível e parêntese aberto viram problemas",
            sorted(original.issues.loc[original.issues["video_id"] == "Subject_1/Fall/02", "issue"])
            == ["duração ilegível", "formato de Classes"],
        ),
        _check("inventory falha com divergência, duplicata e arquivo fora do layout", not inventory_ok),
    ]


def check_coverage_and_cross_check(root: Path) -> list[bool]:
    inventory = build_inventory(root)
    original = read_original_annotations(root, inventory.annotation_files)
    durations = pd.Series({"Subject_1/ADL/01": 3.0, "Subject_1/Fall/01": 2.0})
    every = summarize_coverage(original.segments, durations).set_index("video_id")
    postures = summarize_coverage(
        original.segments, durations, frozenset({"Walking", "Standing", "Falling"})
    ).set_index("video_id")
    named = with_label_names(_omnifall_segments())
    matrix = overlap_matrix(original.segments, named)
    onsets = compare_fall_onsets(original.segments, named, inventory.videos).set_index("video_id")
    return [
        _check(
            "atividade sobre postura conta como sobreposição só sem filtro de postura",
            float(every.loc["Subject_1/ADL/01", "overlap_s"]) == 1.0
            and float(postures.loc["Subject_1/ADL/01", "overlap_s"]) == 0.0
            and float(postures.loc["Subject_1/ADL/01", "gap_s"]) == 1.0,
        ),
        _check(
            "Falling original se divide entre `fall` e `fallen` na matriz",
            math.isclose(float(matrix.loc["Falling", "fall"]), 0.5 + 0.5 + 1.1)
            and math.isclose(float(matrix.loc["Falling", "fallen"]), 0.4)
            and math.isclose(float(matrix.loc["Standing", "standing"]), 0.6),
        ),
        _check(
            "delta de início = primeiro `fall` OmniFall − Falling original",
            math.isclose(float(onsets.loc["Subject_1/Fall/01", "onset_delta_s"]), -0.4)
            and float(onsets.loc["Subject_2/Fall/01", "onset_delta_s"]) == 0.0
            and int(onsets.loc["Subject_1/ADL/01", "omnifall_fall_segments"]) == 0,
        ),
    ]


def check_probe(root: Path) -> list[bool]:
    probes = probe_checkout(root, build_inventory(root).videos).set_index("video_id")
    short = probes.loc["Subject_1/Fall/01"]
    long = probes.loc["Subject_1/ADL/01"]
    corrupt = probes.loc["Subject_1/Fall/02"]
    declared = pd.Series({"Subject_1/ADL/01": 3.0})
    with contextlib.redirect_stdout(io.StringIO()):
        empty_ok = report_probe(pd.DataFrame(columns=PROBE_COLUMNS), declared)
        with_failure_ok = report_probe(probes.reset_index(), declared)
        decoded_ok = report_probe(probes.drop(index="Subject_1/Fall/02").reset_index(), declared)
    return [
        _check(
            f"{_SHORT_FRAMES} e {_LONG_FRAMES} quadros a 10 fps viram grades do mesmo tamanho",
            int(short["grid_frames"]) == _SHORT_FRAMES
            and int(long["grid_frames"]) == _LONG_FRAMES
            and math.isclose(float(long["duration_s"]), 3.0),
        ),
        _check(
            "vídeo CFR tem desvio de PTS nulo; PTS irregular é medido",
            float(long["max_pts_drift_s"]) < 1e-6
            and math.isclose(max_pts_drift(np.array([0.0, 0.1, 0.25, 0.3]), 10.0), 0.05),
        ),
        _check(
            "vídeo corrompido vira erro relatado sem abortar a sondagem",
            isinstance(corrupt["error"], str) and bool(pd.isna(corrupt["grid_frames"])),
        ),
        _check(
            "sondagem vazia ou com falha não é sucesso; só decodificados é",
            not empty_ok and not with_failure_ok and decoded_ok,
        ),
    ]


def check_grid_row() -> bool:
    segments = pd.DataFrame(
        {"label": [0, 1, 2], "start": [0.0, 2.0, 2.8], "end": [1.5, 2.8, 3.0]}
    )
    row = grid_row("v", _LONG_FRAMES, 3.05, segments)
    # Grade 0,0..2,9 s: gap [1,5; 2,0) = 5 quadros; `fall` nos índices 20..27,
    # dos quais 23..27 têm janela causal completa.
    return _check(
        f"grade de {_LONG_FRAMES}: quadros sem replicação, ignorados e `fall` completos",
        row["unpadded_frames"] == _LONG_FRAMES - WINDOW_FRAMES + 1
        and row["ignored_frames"] == 5
        and row["fall_frames"] == 8
        and row["unpadded_fall_frames"] == 5
        and row["fall_onset_s"] == 2.0
        and row["first_fall_index"] == 20
        and math.isclose(float(str(row["label_end_minus_duration_s"])), -0.05),
    )


def check_verify(root: Path) -> bool:
    with contextlib.redirect_stdout(io.StringIO()):
        not_git_ok = run_verify(root / "Subject 1")
    with contextlib.redirect_stdout(io.StringIO()):
        wrong_commit_ok = run_verify(root)
    return _check(
        "verify falha fora de um clone e em repositório sem o commit pinado",
        not not_git_ok and not wrong_commit_ok,
    )


def run_audit_selftest() -> None:
    checks = [*check_parse_classes(), check_original_csv(), check_grid_row()]
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "GMDCSA24"
        _write_synthetic_checkout(root)
        checks.extend(check_inventory(root))
        checks.extend(check_coverage_and_cross_check(root))
        checks.extend(check_probe(root))
        checks.append(check_verify(root))
    checks.extend(run_provenance_checks())
    if not all(checks):
        print("\nselftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: todos os casos passaram")
