"""Anotações temporais do GMDCSA24: CSV original do autor e segmentos do OmniFall.

O CSV original (`Subject N/{ADL,Fall}.csv`) traz, por vídeo, uma coluna
`Classes` em texto livre, como `Falling (SW)[1.8 to 5]; Walking[0 to 1.8]`.
As classes são multirrótulo (ex.: `Reading` sobreposto a `Sitting`) e não
seguem a taxonomia de dez classes do OmniFall. O parser aqui é tolerante: cada
desvio de formato vira um problema relatado, nunca um intervalo inventado.
"""

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.data.gmdcsa24.source import OMNIFALL_DATASET_NAME, video_id
from gatefall.data.intervals import sweep_gaps_and_overlap
from gatefall.data.le2i.annotations import DATASET_ID, LABEL_CONFIG_NAME, OMNIFALL_REVISION
from gatefall.data.omnifall.annotations import (
    annotation_split_dataframe,
    load_annotation_config,
)
from gatefall.datasets.le2i import LE2I_LABEL_NAMES

ORIGINAL_HEADER = [
    "File Name",
    "Length (seconds)",
    "Time of Recording",
    "Attire",
    "Description",
    "Classes",
]
# Um rótulo, a direção opcional da queda (SW/FW/BW) e o corpo entre colchetes.
# O corpo sem `]` termina antes do próximo rótulo, como em
# `Standing[2.4 to 3.6; 6.6 to 7.9; Walking[...]`.
SEGMENT_PATTERN = re.compile(
    r"(?P<label>[A-Z][a-z]+)\s*"
    r"(?:\((?P<direction>[A-Z]{2})(?P<close_paren>\))?)?\s*"
    r"(?:\[(?P<body>[^\[\]]*?)(?:(?P<close_bracket>\])|(?=;\s*[A-Z])|\s*$))?"
)
INTERVAL_PATTERN = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s*(?P<to>to)?\s*(?P<end>\d+(?:\.\d+)?)\s*$"
)
LABEL_ALIASES = {"Fall": "Falling"}
POSTURE_LABELS = frozenset({"Falling", "Sitting", "Sleeping", "Standing", "Walking"})
FALL_LABEL = "Falling"
OMNIFALL_FALL = LE2I_LABEL_NAMES.index("fall")
OMNIFALL_SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}
OMNIFALL_PROTOCOLS = ("cs", "cv")

ORIGINAL_VIDEO_COLUMNS = [
    "video_id",
    "subject",
    "category",
    "file_name",
    "length_s",
    "time_of_recording",
    "attire",
]
SEGMENT_COLUMNS = ["video_id", "label", "direction", "start", "end"]
ISSUE_COLUMNS = ["video_id", "issue", "detail"]


@dataclass(frozen=True)
class OriginalAnnotations:
    videos: pd.DataFrame
    segments: pd.DataFrame
    issues: pd.DataFrame


def parse_interval(text: str) -> tuple[float, float, list[str]]:
    match = INTERVAL_PATTERN.match(text)
    if match is None:
        raise ValueError(f"intervalo ilegível: {text!r}")
    problems = [] if match["to"] else [f"intervalo sem 'to': {text.strip()!r}"]
    return float(match["start"]), float(match["end"]), problems


def parse_classes(text: str) -> tuple[list[dict[str, object]], list[str]]:
    """Converte a coluna `Classes` em segmentos `[start, end)` e problemas de formato."""
    segments: list[dict[str, object]] = []
    problems: list[str] = []
    for match in SEGMENT_PATTERN.finditer(text):
        raw_label = match["label"]
        label = LABEL_ALIASES.get(raw_label, raw_label)
        if label != raw_label:
            problems.append(f"rótulo {raw_label!r} lido como {label!r}")
        if match["direction"] and not match["close_paren"]:
            problems.append(f"{raw_label} ({match['direction']}: parêntese não fechado")
        body = match["body"]
        if body is None:
            problems.append(f"{raw_label}: rótulo sem intervalo")
            continue
        if not match["close_bracket"]:
            problems.append(f"{raw_label}: colchete não fechado")
        for part in filter(str.strip, body.split(";")):
            try:
                start, end, interval_problems = parse_interval(part)
            except ValueError as exc:
                problems.append(f"{raw_label}: {exc}")
                continue
            problems.extend(f"{raw_label}: {problem}" for problem in interval_problems)
            if end <= start:
                problems.append(f"{raw_label}: fim {end:g} não posterior ao início {start:g}")
                continue
            segments.append(
                {"label": label, "direction": match["direction"], "start": start, "end": end}
            )
    if not segments and not problems:
        problems.append(f"nenhum rótulo reconhecido em {text!r}")
    return segments, problems


def parse_length(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse_original_csv(text: str, subject: int, category: str) -> OriginalAnnotations:
    video_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    rows = [row for row in csv.reader(io.StringIO(text)) if any(field.strip() for field in row)]
    header = [field.strip() for field in rows[0]] if rows else []
    if header != ORIGINAL_HEADER:
        issue_rows.append(
            {
                "video_id": f"Subject_{subject}/{category}",
                "issue": "cabeçalho inesperado",
                "detail": repr(header),
            }
        )
    for row in rows[1:]:
        file_name = row[0].strip()
        identifier = video_id(subject, category, file_name.removesuffix(".mp4"))
        if len(row) != len(ORIGINAL_HEADER):
            issue_rows.append(
                {
                    "video_id": identifier,
                    "issue": "número de campos",
                    "detail": f"{len(row)} campos; colunas além da 5ª lidas como Classes",
                }
            )
        padded = row + [""] * (len(ORIGINAL_HEADER) - len(row))
        fields = [*padded[:5], ",".join(padded[5:])]
        length_s = parse_length(fields[1])
        video_rows.append(
            {
                "video_id": identifier,
                "subject": subject,
                "category": category,
                "file_name": file_name,
                "length_s": length_s,
                "time_of_recording": fields[2].strip(),
                "attire": fields[3].strip(),
            }
        )
        if length_s is None:
            issue_rows.append(
                {"video_id": identifier, "issue": "duração ilegível", "detail": fields[1]}
            )
        segments, problems = parse_classes(fields[5])
        segment_rows.extend({"video_id": identifier, **segment} for segment in segments)
        issue_rows.extend(
            {"video_id": identifier, "issue": "formato de Classes", "detail": problem}
            for problem in problems
        )
    return OriginalAnnotations(
        videos=pd.DataFrame(video_rows, columns=ORIGINAL_VIDEO_COLUMNS),
        segments=pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS),
        issues=pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS),
    )


def read_original_annotations(root: Path, annotation_files: list[str]) -> OriginalAnnotations:
    parsed: list[OriginalAnnotations] = []
    for relative in annotation_files:
        subject_part, category_file = relative.split("/")
        subject = int(subject_part.removeprefix("Subject "))
        category = category_file.removesuffix(".csv")
        text = (root / relative).read_text(encoding="utf-8-sig")
        parsed.append(parse_original_csv(text, subject, category))
    return OriginalAnnotations(
        videos=concat([item.videos for item in parsed], ORIGINAL_VIDEO_COLUMNS),
        segments=concat([item.segments for item in parsed], SEGMENT_COLUMNS),
        issues=concat([item.issues for item in parsed], ISSUE_COLUMNS),
    )


def concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return cast(pd.DataFrame, pd.concat(non_empty, ignore_index=True))


def load_omnifall_segments() -> pd.DataFrame:
    """Segmentos do GMDCSA24 no config `labels` do OmniFall, na revisão pinada do Le2i."""
    labels = annotation_split_dataframe(
        load_annotation_config(DATASET_ID, LABEL_CONFIG_NAME, OMNIFALL_REVISION), "train"
    )
    segments = cast(pd.DataFrame, labels[labels["dataset"] == OMNIFALL_DATASET_NAME])
    return segments.rename(columns={"path": "video_id"}).reset_index(drop=True)


def load_omnifall_splits() -> pd.DataFrame:
    """Uma linha por (protocolo, split, vídeo) dos configs `gmdcsa24-{cs,cv}`."""
    rows: list[pd.DataFrame] = []
    for protocol in OMNIFALL_PROTOCOLS:
        config = load_annotation_config(DATASET_ID, f"gmdcsa24-{protocol}", OMNIFALL_REVISION)
        for split, name in OMNIFALL_SPLIT_NAMES.items():
            frame = annotation_split_dataframe(config, split)
            videos = cast(pd.DataFrame, frame[["path", "subject"]].drop_duplicates())
            rows.append(
                videos.rename(columns={"path": "video_id"}).assign(protocol=protocol, split=name)
            )
    return concat(rows, ["video_id", "subject", "protocol", "split"])


def compare_identities(named_sets: dict[str, set[str]]) -> list[str]:
    """Relata cada vídeo ausente de alguma das fontes."""
    every = set().union(*named_sets.values())
    problems: list[str] = []
    for name, members in named_sets.items():
        for missing in sorted(every - members):
            problems.append(f"{missing}: ausente em {name}")
    return problems


def summarize_coverage(
    segments: pd.DataFrame, durations: pd.Series, labels: frozenset[str] | None = None
) -> pd.DataFrame:
    """Gap e sobreposição por vídeo em `[0, duração)`, opcionalmente só entre `labels`."""
    rows: list[dict[str, object]] = []
    for identifier, duration in durations.items():
        group = cast(pd.DataFrame, segments[segments["video_id"] == identifier])
        if labels is not None:
            group = cast(pd.DataFrame, group[group["label"].isin(labels)])
        intervals = list(zip(group["start"].astype(float), group["end"].astype(float)))
        gap_s, overlap_s, gap_intervals = sweep_gaps_and_overlap(intervals, float(duration))
        rows.append(
            {
                "video_id": identifier,
                "duration_s": float(duration),
                "gap_s": gap_s,
                "overlap_s": overlap_s,
                "max_gap_s": max((end - start for start, end in gap_intervals), default=0.0),
                "beyond_s": max(0.0, max((end for _, end in intervals), default=0.0) - duration),
            }
        )
    return pd.DataFrame(
        rows, columns=["video_id", "duration_s", "gap_s", "overlap_s", "max_gap_s", "beyond_s"]
    )


def with_label_names(segments: pd.DataFrame) -> pd.DataFrame:
    """Troca o id inteiro do OmniFall pelo nome da classe."""
    return segments.assign(label=[LE2I_LABEL_NAMES[int(label)] for label in segments["label"]])


def overlap_matrix(original: pd.DataFrame, omnifall: pd.DataFrame) -> pd.DataFrame:
    """Segundos de sobreposição entre cada classe original e cada classe OmniFall."""
    totals: dict[tuple[str, str], float] = {}
    omnifall_by_video = {key: group for key, group in omnifall.groupby("video_id")}
    for row in original.to_dict("records"):
        targets = omnifall_by_video.get(row["video_id"])
        if targets is None:
            continue
        overlap = np.minimum(float(row["end"]), targets["end"].to_numpy(float)) - np.maximum(
            float(row["start"]), targets["start"].to_numpy(float)
        )
        for target, seconds in zip(targets["label"], overlap):
            if seconds > 0:
                key = (str(row["label"]), str(target))
                totals[key] = totals.get(key, 0.0) + float(seconds)
    if not totals:
        return pd.DataFrame()
    matrix = cast(pd.DataFrame, pd.Series(totals).unstack(fill_value=0.0))
    return matrix.reindex(columns=[name for name in LE2I_LABEL_NAMES if name in matrix.columns])


def compare_fall_onsets(
    original: pd.DataFrame, omnifall: pd.DataFrame, videos: pd.DataFrame
) -> pd.DataFrame:
    """Início de `Falling` no CSV original x início do primeiro `fall` do OmniFall."""
    original_onset = original[original["label"] == FALL_LABEL].groupby("video_id")["start"].min()
    omnifall_onset = omnifall[omnifall["label"] == "fall"].groupby("video_id")["start"].min()
    omnifall_falls = omnifall[omnifall["label"] == "fall"].groupby("video_id").size()
    table = cast(pd.DataFrame, videos[["video_id", "category"]]).drop_duplicates()
    table = table.set_index("video_id")
    table["original_onset_s"] = original_onset
    table["omnifall_onset_s"] = omnifall_onset
    table["omnifall_fall_segments"] = omnifall_falls
    table["omnifall_fall_segments"] = table["omnifall_fall_segments"].fillna(0).astype(int)
    table["onset_delta_s"] = table["omnifall_onset_s"] - table["original_onset_s"]
    return cast(pd.DataFrame, table.reset_index())


def report_original(annotations: OriginalAnnotations) -> None:
    videos = annotations.videos
    segments = annotations.segments
    issues = annotations.issues
    print("\n=== CSV original: colunas por vídeo ===")
    print(f"linhas de vídeo: {len(videos)}")
    print("Time of Recording por sujeito:")
    print(pd.crosstab(videos["time_of_recording"], videos["subject"]).to_string())
    print(f"Attire distintos: {videos['attire'].nunique()}")
    print("Length (seconds) declarado:")
    print(videos["length_s"].astype(float).describe(percentiles=[0.05, 0.5, 0.95]).to_string())

    print("\n=== CSV original: classes (multirrótulo, texto livre) ===")
    print(f"segmentos válidos: {len(segments)}")
    counts = segments.groupby(["label", "video_id"]).size().groupby("label").size()
    seconds = (segments["end"] - segments["start"]).groupby(segments["label"]).sum()
    print(pd.DataFrame({"videos": counts, "segundos": seconds.round(1)}).to_string())
    directions = cast(pd.Series, segments.loc[segments["label"] == FALL_LABEL, "direction"])
    print("direção de Falling:")
    print(directions.fillna("<sem>").value_counts().to_string())

    print(f"\nproblemas de formato ({len(issues)} em {issues['video_id'].nunique()} vídeos):")
    for row in issues.to_dict("records"):
        print(f"  {row['video_id']}: {row['issue']}: {row['detail']}")

    durations = cast(pd.Series, videos.set_index("video_id")["length_s"].astype(float))
    every = summarize_coverage(segments, durations)
    postures = summarize_coverage(segments, durations, POSTURE_LABELS)
    print("\n=== CSV original: cobertura sobre [0, Length) declarado ===")
    print(f"vídeos com gap > 0: {(every['gap_s'] > 0).sum()} de {len(every)}")
    print(f"vídeos com gap >= 1 s: {(every['max_gap_s'] >= 1).sum()}")
    print(f"vídeos com segmento além de Length: {(every['beyond_s'] > 0).sum()}")
    print(
        "vídeos com sobreposição entre classes de postura "
        f"{sorted(POSTURE_LABELS)}: {(postures['overlap_s'] > 0).sum()}"
    )
    print(
        "vídeos com sobreposição de qualquer classe (inclui atividade sobre postura): "
        f"{(every['overlap_s'] > 0).sum()}"
    )
    print("fração de [0, Length) sem rótulo, por vídeo:")
    print((every["gap_s"] / every["duration_s"]).describe(percentiles=[0.5, 0.95]).to_string())


def report_omnifall(omnifall: pd.DataFrame, splits: pd.DataFrame) -> None:
    print(f"\n=== OmniFall {DATASET_ID}@{OMNIFALL_REVISION[:7]}: {OMNIFALL_DATASET_NAME} ===")
    print(f"segmentos: {len(omnifall)} em {omnifall['video_id'].nunique()} vídeos")
    print("classes (segmentos, segundos):")
    lengths = (omnifall["end"] - omnifall["start"]).astype(float)
    seconds = lengths.groupby(omnifall["label"]).sum().round(1)
    table = pd.DataFrame({"segmentos": omnifall["label"].value_counts(), "segundos": seconds})
    print(table.reindex([n for n in LE2I_LABEL_NAMES if n in table.index]).to_string())
    print("segundos por classe e sujeito:")
    per_subject = lengths.groupby([omnifall["label"], omnifall["subject"]]).sum().unstack(fill_value=0.0)
    order = [name for name in LE2I_LABEL_NAMES if name in per_subject.index]
    print(cast(pd.DataFrame, per_subject).reindex(order).round(1).to_string())
    print(f"cam distintos: {sorted(int(cam) for cam in omnifall['cam'].unique())}")
    path_subject = omnifall["video_id"].str.extract(r"^Subject_(\d+)/")[0].astype(int)
    print(f"subject divergente do caminho: {(path_subject != omnifall['subject']).sum()}")

    durations = cast(pd.Series, omnifall.groupby("video_id")["end"].max())
    coverage = summarize_coverage(omnifall, durations)
    print("cobertura sobre [0, último fim) (a duração real exige `probe`):")
    print(f"  vídeos com gap > 0: {(coverage['gap_s'] > 0).sum()}")
    print(f"  vídeos com gap >= 0,1 s: {(coverage['max_gap_s'] >= 0.1).sum()}")
    print(f"  vídeos com sobreposição > 0: {(coverage['overlap_s'] > 0).sum()}")
    print(f"  vídeos com sobreposição >= 0,1 s: {(coverage['overlap_s'] >= 0.1).sum()}")
    flagged = cast(
        pd.DataFrame, coverage[(coverage["max_gap_s"] >= 0.1) | (coverage["overlap_s"] >= 0.1)]
    )
    for row in flagged.to_dict("records"):
        print(
            f"    {row['video_id']}: gap {row['gap_s']:.3f} s, "
            f"sobreposição {row['overlap_s']:.3f} s"
        )

    print("\nsplits OmniFall (vídeos por sujeito):")
    print(
        splits.assign(subject=splits["subject"].astype(int))
        .groupby(["protocol", "split", "subject"])
        .size()
        .to_string()
    )


def report_cross_check(
    original: OriginalAnnotations, omnifall: pd.DataFrame, videos: pd.DataFrame
) -> None:
    print("\n=== classe original x classe OmniFall (segundos sobrepostos, % da linha) ===")
    matrix = overlap_matrix(original.segments, omnifall)
    shares = matrix.div(matrix.sum(axis=1), axis=0) * 100
    print(shares.round(0).astype(int).to_string())

    print("\n=== início da queda: Falling original x primeiro `fall` OmniFall ===")
    onsets = compare_fall_onsets(original.segments, omnifall, videos)
    for category, group in onsets.groupby("category"):
        with_fall = group[group["omnifall_fall_segments"] > 0]
        with_original = group["original_onset_s"].notna()
        print(
            f"{category}: {len(group)} vídeos, {with_original.sum()} com Falling original, "
            f"{len(with_fall)} com `fall` OmniFall "
            f"({(with_fall['omnifall_fall_segments'] > 1).sum()} com mais de um)"
        )
    falls = cast(pd.DataFrame, onsets[onsets["category"] == "Fall"])
    missing = cast(
        pd.DataFrame, falls[falls["original_onset_s"].isna() | falls["omnifall_onset_s"].isna()]
    )
    for row in missing.to_dict("records"):
        print(
            f"  {row['video_id']}: Falling original={row['original_onset_s']:.2f}, "
            f"`fall` OmniFall={row['omnifall_onset_s']:.2f}"
        )
    delta = cast(pd.Series, falls["onset_delta_s"]).dropna().astype(float).round(3)
    print("OmniFall − original (s), vídeos Fall com ambos:")
    print(delta.describe(percentiles=[0.05, 0.5, 0.95]).to_string())
    print(f"|delta| > 0,5 s: {(delta.abs() > 0.5).sum()} de {len(delta)}")
