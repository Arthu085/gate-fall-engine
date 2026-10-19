"""Inventário e verificações do `VideoDataset.zip` do Figshare 28596332 (Ursul, 2025).

Tudo aqui lê o arquivo zip local sem extraí-lo para `data/`: o diretório
central do zip dá os nomes, tamanhos e CRC-32 de cada membro, e os
`metadata.json` são lidos em memória. Nada é gravado em disco.
"""

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import pandas as pd

FIGSHARE_ARTICLE_ID = 28596332
FIGSHARE_VERSION = 2
FIGSHARE_DOI = "10.6084/m9.figshare.28596332.v2"
ARCHIVE_FILE_ID = 52990358
ARCHIVE_NAME = "VideoDataset.zip"
ARCHIVE_SIZE_BYTES = 2_529_520_868
ARCHIVE_MD5 = "c784167d08f2fa94e3afd36cec758e1f"
ARCHIVE_DOWNLOAD_URL = f"https://ndownloader.figshare.com/files/{ARCHIVE_FILE_ID}"
DEFAULT_ARCHIVE_PATH = Path("data/raw/figshare_fall") / ARCHIVE_NAME

# Valores anunciados na descrição pública do registro; 999 + 1017 = 2016,
# não 2017.
ADVERTISED_RECORDINGS = 2017
ADVERTISED_FALLS = 999
ADVERTISED_ADLS = 1017
ADVERTISED_SUBJECTS = 29

RECORDING_PATTERN = re.compile(
    r"^VideoDataset/(?P<category>ADL|Fall)/SBJ_(?P<subject>\d+)_LOC(?P<location>\d+)/"
    r"ACT(?P<action>\d+)_(?P<side>[A-Z])_(?P<attempt>\d+)/(?P<filename>[^/]+)$"
)

RECORDING_COLUMNS = [
    "recording_id",
    "category",
    "subject",
    "location",
    "action",
    "side",
    "attempt",
    "n_videos",
    "n_sensor_logs",
    "n_metadata",
]
VIDEO_COLUMNS = ["member", "recording_id", "crc", "size"]


@dataclass(frozen=True)
class ArchiveInventory:
    recordings: pd.DataFrame
    videos: pd.DataFrame
    loose_videos: list[str]
    unrecognized_members: list[str]
    platform_artifacts: int


def is_platform_artifact(name: str) -> bool:
    return name.startswith("__MACOSX/") or PurePosixPath(name).name == ".DS_Store"


def build_inventory(infos: list[zipfile.ZipInfo]) -> ArchiveInventory:
    recordings: dict[str, dict[str, object]] = {}
    video_rows: list[dict[str, object]] = []
    loose_videos: list[str] = []
    unrecognized: list[str] = []
    platform_artifacts = 0

    for info in infos:
        name = info.filename
        if info.is_dir():
            continue
        if is_platform_artifact(name):
            platform_artifacts += 1
            continue

        match = RECORDING_PATTERN.match(name)
        if match is None:
            if name.endswith(".mp4"):
                loose_videos.append(name)
            elif not re.match(r"^VideoDataset/(ADL|Fall)/all_(csvs|plots)/", name):
                unrecognized.append(name)
            continue

        recording_id = name.rsplit("/", 1)[0].removeprefix("VideoDataset/")
        recording = recordings.setdefault(
            recording_id,
            {
                "recording_id": recording_id,
                "category": match["category"],
                "subject": int(match["subject"]),
                "location": int(match["location"]),
                "action": int(match["action"]),
                "side": match["side"],
                "attempt": int(match["attempt"]),
                "n_videos": 0,
                "n_sensor_logs": 0,
                "n_metadata": 0,
            },
        )
        filename = match["filename"]
        if filename.endswith(".mp4"):
            recording["n_videos"] = cast(int, recording["n_videos"]) + 1
            video_rows.append(
                {
                    "member": name,
                    "recording_id": recording_id,
                    "crc": info.CRC,
                    "size": info.file_size,
                }
            )
        elif filename.endswith(".txt"):
            recording["n_sensor_logs"] = cast(int, recording["n_sensor_logs"]) + 1
        elif filename == "metadata.json":
            recording["n_metadata"] = cast(int, recording["n_metadata"]) + 1
        else:
            unrecognized.append(name)

    recordings_table = pd.DataFrame(list(recordings.values()), columns=RECORDING_COLUMNS)
    videos_table = pd.DataFrame(video_rows, columns=VIDEO_COLUMNS)
    return ArchiveInventory(
        recordings=cast(
            pd.DataFrame, recordings_table.sort_values("recording_id", ignore_index=True)
        ),
        videos=cast(pd.DataFrame, videos_table.sort_values("member", ignore_index=True)),
        loose_videos=sorted(loose_videos),
        unrecognized_members=sorted(unrecognized),
        platform_artifacts=platform_artifacts,
    )


def read_inventory(archive: zipfile.ZipFile) -> ArchiveInventory:
    return build_inventory(archive.infolist())


def find_metadata_mismatches(
    archive: zipfile.ZipFile, recordings: pd.DataFrame
) -> list[str]:
    """Compara cada `metadata.json` com os campos codificados no caminho da gravação."""
    problems: list[str] = []
    for row in recordings.to_dict("records"):
        recording_id = str(row["recording_id"])
        member = f"VideoDataset/{recording_id}/metadata.json"
        try:
            metadata = json.loads(archive.read(member))
        except KeyError:
            problems.append(f"{recording_id}: metadata.json ausente")
            continue
        except json.JSONDecodeError as exc:
            problems.append(f"{recording_id}: metadata.json inválido ({exc})")
            continue

        expected = {
            "subjectId": row["subject"],
            "locationId": row["location"],
            "actionId": row["action"],
            "side": row["side"],
            "attempt": row["attempt"],
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                problems.append(
                    f"{recording_id}: {key}={metadata.get(key)!r} no metadata.json, "
                    f"{value!r} no caminho"
                )
    return problems


def find_duplicate_videos(videos: pd.DataFrame) -> list[list[str]]:
    """Agrupa vídeos com mesmo (CRC-32, tamanho), indício de cópia byte a byte."""
    groups = videos.groupby(["crc", "size"])["member"].apply(list)
    return sorted(
        sorted(cast(list[str], members)) for members in groups if len(members) > 1
    )


def summarize_subjects(recordings: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por sujeito: locais por categoria e contagem de gravações."""
    rows: list[dict[str, object]] = []
    for subject, group in recordings.groupby("subject"):
        row: dict[str, object] = {"subject": subject}
        for category in ("Fall", "ADL"):
            locations = group.loc[group["category"] == category, "location"]
            row[f"n_{category.lower()}"] = len(locations)
            row[f"locations_{category.lower()}"] = ",".join(
                str(location) for location in sorted(set(locations))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_file_digests(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return size, md5.hexdigest(), sha256.hexdigest()


def report_counts(inventory: ArchiveInventory) -> None:
    recordings = inventory.recordings
    videos = inventory.videos
    print("\n=== contagens anunciadas x arquivo ===")
    print(
        f"anunciado: {ADVERTISED_RECORDINGS} gravações = {ADVERTISED_FALLS} quedas + "
        f"{ADVERTISED_ADLS} ADL (soma {ADVERTISED_FALLS + ADVERTISED_ADLS})"
    )
    n_mp4 = len(videos) + len(inventory.loose_videos)
    print(f"arquivos .mp4 no zip (sem __MACOSX): {n_mp4}")
    print(f"  dentro de diretórios de gravação: {len(videos)}")
    print(f"  fora de diretórios de gravação: {len(inventory.loose_videos)}")
    for member in inventory.loose_videos:
        print(f"    {member}")
    top_folders = [
        member.split("/")[1] if member.count("/") > 1 else "<raiz>"
        for member in [*videos["member"], *inventory.loose_videos]
    ]
    print("arquivos .mp4 por pasta de topo (gravações + soltos):")
    print(pd.Series(top_folders).value_counts().sort_index().to_string())
    print(f"diretórios de gravação: {len(recordings)}")
    print(recordings.groupby("category").size().to_string())
    with_video = recordings[recordings["n_videos"] > 0]
    print(f"gravações com ao menos um vídeo: {len(with_video)}")
    print(with_video.groupby("category").size().to_string())
    print(f"artefatos __MACOSX/.DS_Store ignorados: {inventory.platform_artifacts}")
    if inventory.unrecognized_members:
        print(f"membros não reconhecidos ({len(inventory.unrecognized_members)}):")
        for member in inventory.unrecognized_members:
            print(f"  {member}")


def report_recording_anomalies(inventory: ArchiveInventory) -> None:
    recordings = inventory.recordings
    print("\n=== gravações com composição atípica ===")
    print(
        recordings.groupby(["n_videos", "n_sensor_logs", "n_metadata"])
        .size()
        .to_string()
    )
    for label, mask in (
        ("sem vídeo", recordings["n_videos"] == 0),
        ("com mais de um vídeo", recordings["n_videos"] > 1),
    ):
        subset = recordings[mask]
        print(f"{label} ({len(subset)}):")
        for recording_id in subset["recording_id"]:
            print(f"  {recording_id}")


def report_duplicates(inventory: ArchiveInventory) -> None:
    print("\n=== vídeos com mesmo CRC-32 e tamanho ===")
    duplicates = find_duplicate_videos(inventory.videos)
    if not duplicates:
        print("nenhum")
    recording_of = dict(
        zip(inventory.videos["member"], inventory.videos["recording_id"])
    )
    for group in duplicates:
        recordings = {recording_of[member] for member in group}
        scope = "mesma gravação" if len(recordings) == 1 else "gravações distintas"
        print(f"{scope}:")
        for member in group:
            print(f"  {member}")


def report_actions(recordings: pd.DataFrame) -> None:
    print("\n=== ações por categoria (granularidade do rótulo: gravação) ===")
    print(
        pd.crosstab(recordings["action"], recordings["category"]).to_string()
    )
    shared = set(recordings[recordings["category"] == "Fall"]["action"]) & set(
        recordings[recordings["category"] == "ADL"]["action"]
    )
    print(f"ações presentes em Fall e ADL: {sorted(shared) or 'nenhuma'}")
    print("lados por categoria:")
    print(pd.crosstab(recordings["side"], recordings["category"]).to_string())


def report_subjects_and_locations(recordings: pd.DataFrame) -> None:
    print("\n=== sujeitos e locais ===")
    subjects = sorted(recordings["subject"].unique())
    print(f"sujeitos distintos: {len(subjects)} (anunciado {ADVERTISED_SUBJECTS})")
    summary = summarize_subjects(recordings)
    print(summary.to_string(index=False))
    both = summary[(summary["n_fall"] > 0) & (summary["n_adl"] > 0)]
    print(f"sujeitos com Fall e ADL: {len(both)}")
    moved = both[both["locations_fall"] != both["locations_adl"]]
    print(f"sujeitos cujo local difere entre Fall e ADL: {len(moved)}")
    print("\nlocal x categoria (gravações):")
    print(pd.crosstab(recordings["location"], recordings["category"]).to_string())


def report_metadata(archive: zipfile.ZipFile, recordings: pd.DataFrame) -> bool:
    print("\n=== metadata.json x caminho ===")
    problems = find_metadata_mismatches(archive, recordings)
    for problem in problems:
        print(f"  {problem}")
    if problems:
        print(f"{len(problems)} divergências")
        return False
    print(f"OK: {len(recordings)} metadata.json coerentes com o caminho")
    return True
