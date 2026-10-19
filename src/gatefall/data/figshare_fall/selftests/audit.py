"""Selftest sintético da auditoria do Figshare 28596332 (`inventory.py`, `probe.py`).

Não toca no dataset real: o zip é montado num diretório temporário com a
mesma árvore `VideoDataset/{ADL,Fall}/SBJ_xx_LOCy/ACTa_S_n/` do registro, além
dos desvios observados no arquivo publicado (artefatos `__MACOSX`, timelapses
fora das gravações, gravação sem vídeo, vídeo duplicado e `metadata.json`
divergente). Requer FFmpeg.
"""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.config import WINDOW_FRAMES
from gatefall.data.figshare_fall.inventory import (
    build_inventory,
    compute_file_digests,
    find_metadata_mismatches,
    find_probable_duplicate_videos,
    has_identical_content,
    summarize_subjects,
)
from gatefall.data.figshare_fall.audit import run_probe
from gatefall.data.figshare_fall.probe import (
    PROBE_COLUMNS,
    count_unpadded_windows,
    probe_videos,
    report_probe,
    select_spread_sample,
)

_SHORT_FRAMES = 19
_LONG_FRAMES = 30


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _encode_video(directory: Path, name: str, n_frames: int) -> bytes:
    path = directory / name
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
    return path.read_bytes()


def _metadata(subject: int, location: int, action: int, side: str, attempt: int) -> bytes:
    return json.dumps(
        {
            "subjectId": subject,
            "locationId": location,
            "actionId": action,
            "side": side,
            "attempt": attempt,
        }
    ).encode()


def _write_synthetic_archive(directory: Path) -> Path:
    short_video = _encode_video(directory, "short.mp4", _SHORT_FRAMES)
    long_video = _encode_video(directory, "long.mp4", _LONG_FRAMES)
    members: dict[str, bytes] = {
        "VideoDataset/timelapse.mp4": long_video,
        "VideoDataset/.DS_Store": b"x",
        "__MACOSX/VideoDataset/._timelapse.mp4": b"x",
        "VideoDataset/Fall/all_csvs/20240101000000.csv": b"a,b\n",
        "VideoDataset/Fall/all_plots/20240101000000.png": b"png",
        "VideoDataset/readme.md": b"?",
        # Fall do sujeito 1 no local 1 com dois vídeos idênticos (como o
        # "(1).mp4" do registro publicado).
        "VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/metadata.json": _metadata(1, 1, 1, "L", 1),
        "VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/a.mp4": short_video,
        "VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/a (1).mp4": short_video,
        "VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/a.txt": b"time\n",
        # ADL do mesmo sujeito em outro local.
        "VideoDataset/ADL/SBJ_01_LOC3/ACT13_B_1/metadata.json": _metadata(1, 3, 13, "B", 1),
        "VideoDataset/ADL/SBJ_01_LOC3/ACT13_B_1/b.mp4": long_video,
        "VideoDataset/ADL/SBJ_01_LOC3/ACT13_B_1/b.txt": b"time\n",
        # Gravação sem vídeo e com subjectId divergente do caminho.
        "VideoDataset/ADL/SBJ_02_LOC4/ACT14_R_2/metadata.json": _metadata(9, 4, 14, "R", 2),
        "VideoDataset/ADL/SBJ_02_LOC4/ACT14_R_2/c.txt": b"time\n",
        # Gravação sem metadata.json e com vídeo corrompido.
        "VideoDataset/Fall/SBJ_03_LOC2/ACT10_F_1/d.mp4": b"not a video",
    }
    path = directory / "VideoDataset.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("VideoDataset/", b"")
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def check_inventory(archive: zipfile.ZipFile) -> list[bool]:
    inventory = build_inventory(archive.infolist())
    recordings = inventory.recordings.set_index("recording_id")
    return [
        _check(
            "4 diretórios de gravação (2 Fall, 2 ADL), artefatos __MACOSX/.DS_Store ignorados",
            len(recordings) == 4
            and recordings["category"].value_counts().to_dict() == {"Fall": 2, "ADL": 2}
            and inventory.platform_artifacts == 2,
        ),
        _check(
            "timelapse fora de gravação conta como vídeo solto, não como gravação",
            inventory.loose_videos == ["VideoDataset/timelapse.mp4"]
            and "VideoDataset/timelapse.mp4" not in set(inventory.videos["member"]),
        ),
        _check(
            "all_csvs/all_plots são auxiliares; membro fora do layout é sinalizado",
            inventory.unrecognized_members == ["VideoDataset/readme.md"],
        ),
        _check(
            "contagem por gravação: 2 vídeos, 0 vídeo e campos do caminho parseados",
            int(recordings.loc["Fall/SBJ_01_LOC1/ACT1_L_1", "n_videos"]) == 2
            and int(recordings.loc["ADL/SBJ_02_LOC4/ACT14_R_2", "n_videos"]) == 0
            and int(recordings.loc["ADL/SBJ_02_LOC4/ACT14_R_2", "attempt"]) == 2
            and recordings.loc["ADL/SBJ_01_LOC3/ACT13_B_1", "side"] == "B",
        ),
        _check(
            "prováveis duplicatas agrupadas por (CRC-32, tamanho)",
            find_probable_duplicate_videos(inventory.videos)
            == [
                [
                    "VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/a (1).mp4",
                    "VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/a.mp4",
                ]
            ],
        ),
        # CRC-32 e tamanho iguais não provam igualdade; o SHA-256 do conteúdo
        # é que confirma ou refuta, inclusive para um grupo de vídeos diferentes.
        _check(
            "SHA-256 do conteúdo confirma a cópia e refuta vídeos diferentes",
            has_identical_content(
                archive,
                [
                    "VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/a.mp4",
                    "VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/a (1).mp4",
                ],
            )
            and not has_identical_content(
                archive,
                [
                    "VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/a.mp4",
                    "VideoDataset/ADL/SBJ_01_LOC3/ACT13_B_1/b.mp4",
                ],
            ),
        ),
    ]


def check_metadata_and_subjects(archive: zipfile.ZipFile) -> list[bool]:
    inventory = build_inventory(archive.infolist())
    problems = find_metadata_mismatches(archive, inventory.recordings)
    summary = summarize_subjects(inventory.recordings).set_index("subject")
    return [
        _check(
            "metadata.json divergente e ausente são relatados; os coerentes não",
            problems
            == [
                "ADL/SBJ_02_LOC4/ACT14_R_2: subjectId=9 no metadata.json, 2 no caminho",
                "Fall/SBJ_03_LOC2/ACT10_F_1: metadata.json ausente",
            ],
        ),
        _check(
            "sujeito 1 tem Fall no local 1 e ADL no local 3",
            summary.loc[1, "locations_fall"] == "1"
            and summary.loc[1, "locations_adl"] == "3"
            and int(summary.loc[1, "n_fall"]) == 1
            and int(summary.loc[2, "n_fall"]) == 0,
        ),
    ]


def check_probe(archive: zipfile.ZipFile) -> list[bool]:
    inventory = build_inventory(archive.infolist())
    probes = probe_videos(archive, list(inventory.videos["member"])).set_index("member")
    short = probes.loc["VideoDataset/Fall/SBJ_01_LOC1/ACT1_L_1/a.mp4"]
    long = probes.loc["VideoDataset/ADL/SBJ_01_LOC3/ACT13_B_1/b.mp4"]
    corrupt = probes.loc["VideoDataset/Fall/SBJ_03_LOC2/ACT10_F_1/d.mp4"]
    return [
        _check(
            f"{_SHORT_FRAMES} quadros a 10 fps -> grade {_SHORT_FRAMES}, nenhuma "
            f"janela de {WINDOW_FRAMES} sem replicação de borda",
            int(short["n_frames_counted"]) == _SHORT_FRAMES
            and int(short["grid_frames"]) == _SHORT_FRAMES
            and int(short["unpadded_windows"]) == 0,
        ),
        _check(
            f"{_LONG_FRAMES} quadros a 10 fps -> {_LONG_FRAMES - WINDOW_FRAMES + 1} "
            "janelas sem replicação de borda",
            int(long["grid_frames"]) == _LONG_FRAMES
            and int(long["unpadded_windows"]) == _LONG_FRAMES - WINDOW_FRAMES + 1,
        ),
        _check(
            "vídeo corrompido vira erro relatado, sem abortar a sondagem",
            isinstance(corrupt["error"], str) and pd.isna(corrupt["grid_frames"]),
        ),
        _check(
            "count_unpadded_windows satura em 0",
            count_unpadded_windows(5) == 0
            and count_unpadded_windows(WINDOW_FRAMES) == 1,
        ),
    ]


def _raises_value_error(members: list[str], limit: int) -> bool:
    try:
        select_spread_sample(members, limit)
    except ValueError:
        return True
    return False


def check_spread_sample() -> list[bool]:
    members = [f"m{index:02d}" for index in range(10)]
    return [
        _check(
            "amostra espalhada é determinística, inclui as pontas e respeita limites",
            select_spread_sample(list(reversed(members)), 3) == ["m00", "m04", "m09"]
            and select_spread_sample(members, 1) == ["m00"]
            and select_spread_sample(members, None) == members
            and select_spread_sample(members, 50) == members,
        ),
        _check(
            "limit 0 ou negativo levanta ValueError em vez de amostra vazia",
            _raises_value_error(members, 0) and _raises_value_error(members, -3),
        ),
    ]


def _quiet_report_probe(probes: pd.DataFrame) -> bool:
    with contextlib.redirect_stdout(io.StringIO()):
        return report_probe(probes)


def check_probe_outcome(archive: zipfile.ZipFile) -> list[bool]:
    inventory = build_inventory(archive.infolist())
    probes = probe_videos(archive, list(inventory.videos["member"]))
    decoded = cast(pd.DataFrame, probes[probes["error"].isna()])
    failed = cast(pd.DataFrame, probes[probes["error"].notna()])
    return [
        _check(
            "sondagem sem nenhum vídeo não é sucesso",
            not _quiet_report_probe(pd.DataFrame(columns=PROBE_COLUMNS)),
        ),
        _check(
            "sondagem só com falhas não é sucesso; só com decodificados é",
            not _quiet_report_probe(failed) and _quiet_report_probe(decoded),
        ),
    ]


def check_probe_cli(directory: Path, archive_path: Path) -> list[bool]:
    # Zip só com um vídeo solto: a seleção de vídeos de gravação fica vazia.
    empty_archive = directory / "sem_gravacoes.zip"
    with zipfile.ZipFile(empty_archive, "w") as archive:
        archive.writestr("VideoDataset/timelapse.mp4", b"x")
    with contextlib.redirect_stdout(io.StringIO()):
        empty_ok = run_probe(empty_archive, None)

    exit_codes = [
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gatefall.data.figshare_fall.audit",
                "probe",
                "--archive",
                str(archive_path),
                "--limit",
                limit,
            ],
            capture_output=True,
        ).returncode
        for limit in ("0", "-1")
    ]
    return [
        _check("zip sem vídeos de gravação faz probe falhar", not empty_ok),
        _check(
            "CLI rejeita --limit 0 e --limit -1 antes de sondar",
            exit_codes == [2, 2],
        ),
    ]


def check_file_digests(path: Path) -> bool:
    payload = path.read_bytes()
    size, md5, sha256 = compute_file_digests(path)
    return _check(
        "tamanho, MD5 e SHA-256 batem com hashlib sobre os mesmos bytes",
        size == len(payload)
        and md5 == hashlib.md5(payload).hexdigest()
        and sha256 == hashlib.sha256(payload).hexdigest(),
    )


def run_audit_selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = _write_synthetic_archive(Path(tmp_dir))
        with zipfile.ZipFile(archive_path) as archive:
            checks = [
                *check_inventory(archive),
                *check_metadata_and_subjects(archive),
                *check_probe(archive),
                *check_probe_outcome(archive),
            ]
        checks.extend(check_spread_sample())
        checks.extend(check_probe_cli(Path(tmp_dir), archive_path))
        checks.append(check_file_digests(archive_path))
    if not all(checks):
        print("\nselftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: todos os casos passaram")
