"""Fatos verificáveis só pelos CSVs CMDFall do OmniFall: identidade, vistas, rótulos, fronteiras, splits e janelas potenciais."""

from dataclasses import dataclass
from itertools import combinations
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import (
    EVAL_STRIDE,
    IGNORE_LABEL,
    NUM_CLASSES,
    TARGET_FPS,
    TRAIN_STRIDE,
    WINDOW_FRAMES,
)
from gatefall.data.intervals import sweep_gaps_and_overlap, tag_gap_positions
from gatefall.data.resampling import GridLabels, labels_for_grid
from gatefall.data.windowing import window_end_indices, window_frame_indices
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.train.shared.metrics import RESTRICTED_CLASSES

SEGMENT_COLUMNS = ["path", "label", "start", "end", "subject", "cam", "dataset"]
# colors/S<sessão>P<sujeito>K<câmera>; a leitura de S como sessão de gravação
# vem só do layout dos CSVs (cada S tem um único P e até 7 K).
PATH_PATTERN = r"^colors/S(\d+)P(\d+)K(\d+)$"
DATASET_NAME = "cmdfall"
PRIORITY_LABELS = ("other", "walk", "standing", "fall", "fallen")
SPLIT_NAMES = ("train", "val", "test")
UNASSIGNED_SPLIT = "unassigned"
# Diferenças de até 1 ms entre fim e início consecutivos vêm da formatação
# decimal dos CSVs (ex.: 0,00014 s) e contam como fronteira contígua.
BOUNDARY_TOLERANCE_S = 1e-3
GRID_STEP_S = 1.0 / TARGET_FPS
WINDOW_SECONDS = WINDOW_FRAMES / TARGET_FPS

# statistics.md e README.md do OmniFall na revisão fixada; SV conta uma vista
# por sessão, MV soma todas as vistas sincronizadas.
PUBLISHED_STATISTICS: dict[str, float] = {
    "videos": 384,
    "sv_segments": 6026,
    "mv_segments": 42143,
    "sv_hours": 7.12,
    "mv_hours": 49.80,
}


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return cast(pd.DataFrame, frame[mask])


def _distinct(frame: pd.DataFrame, name: str) -> int:
    return int(_column(frame, name).nunique())


def schema_problems(raw: pd.DataFrame) -> list[str]:
    missing = [column for column in SEGMENT_COLUMNS if column not in raw.columns]
    if missing:
        return [f"colunas ausentes: {missing}"]

    paths = _column(raw, "path").astype(str)
    tokens = paths.str.extract(PATH_PATTERN)
    matched = tokens.notna().all(axis=1).to_numpy()
    parsed = cast(pd.DataFrame, tokens[matched]).astype(int).to_numpy()
    path_values = paths.to_numpy()
    matched_paths = path_values[matched]
    labels = _column(raw, "label").to_numpy()
    starts = _column(raw, "start").to_numpy()
    ends = _column(raw, "end").to_numpy()

    subjects_by_session: dict[int, set[int]] = {}
    for session, subject in parsed[:, :2]:
        subjects_by_session.setdefault(int(session), set()).add(int(subject))

    checks: list[tuple[str, list[str]]] = [
        (f"path fora de {PATH_PATTERN}", list(path_values[~matched])),
        (
            "subject diverge do P do path",
            list(matched_paths[_column(raw, "subject").to_numpy()[matched] != parsed[:, 1]]),
        ),
        (
            "cam diverge do K do path",
            list(matched_paths[_column(raw, "cam").to_numpy()[matched] != parsed[:, 2]]),
        ),
        (
            f"dataset diferente de {DATASET_NAME!r}",
            list(path_values[_column(raw, "dataset").astype(str).to_numpy() != DATASET_NAME]),
        ),
        (
            f"label fora de 0..{NUM_CLASSES - 1}",
            list(path_values[(labels < 0) | (labels >= NUM_CLASSES)]),
        ),
        ("start negativo", list(path_values[starts < 0])),
        ("end <= start", list(path_values[ends <= starts])),
        (
            "sessão com mais de um sujeito",
            [f"S{session}" for session, subjects in subjects_by_session.items() if len(subjects) > 1],
        ),
    ]
    return [
        f"{description}: {len(items)} ocorrências (ex.: {items[0]})"
        for description, items in checks
        if items
    ]


def taxonomy_problems(label2id: pd.DataFrame) -> list[str]:
    published = dict(
        zip(_column(label2id, "id").astype(int), _column(label2id, "label").astype(str))
    )
    return [
        f"id {label_id}: OmniFall {published.get(label_id)!r}, GateFall {name!r}"
        for label_id, name in enumerate(LE2I_LABEL_NAMES)
        if published.get(label_id) != name
    ]


def annotate_segments(raw: pd.DataFrame) -> pd.DataFrame:
    tokens = _column(raw, "path").astype(str).str.extract(PATH_PATTERN).astype(int)
    segments = raw.assign(
        session=tokens[0], duration=_column(raw, "end") - _column(raw, "start")
    )
    return segments.sort_values(["path", "start", "end"], kind="stable").reset_index(drop=True)


def video_table(segments: pd.DataFrame) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        segments.drop_duplicates("path")[["path", "session", "subject", "cam"]],
    ).reset_index(drop=True)


def single_view(segments: pd.DataFrame) -> pd.DataFrame:
    """Uma vista por sessão (a de menor câmera); só representa a sessão inteira
    quando `view_duplication` não encontra vistas divergentes."""
    reference_cam = segments.groupby("session")["cam"].transform("min")
    return _rows(segments, _column(segments, "cam") == reference_cam).reset_index(drop=True)


@dataclass(frozen=True)
class ViewDuplication:
    sessions: int
    identical_sessions: int
    divergent_sessions: list[int]
    missing_views: dict[int, list[int]]
    cameras: list[int]


def view_duplication(segments: pd.DataFrame) -> ViewDuplication:
    cameras = sorted(int(cam) for cam in _column(segments, "cam").unique())
    divergent: list[int] = []
    missing: dict[int, list[int]] = {}
    for session, group in segments.groupby("session"):
        sequences = {
            int(cast(int, cam)): tuple(
                zip(_column(view, "label"), _column(view, "start"), _column(view, "end"))
            )
            for cam, view in group.groupby("cam")
        }
        if len(set(sequences.values())) > 1:
            divergent.append(int(cast(int, session)))
        absent = [cam for cam in cameras if cam not in sequences]
        if absent:
            missing[int(cast(int, session))] = absent
    n_sessions = _distinct(segments, "session")
    return ViewDuplication(
        sessions=n_sessions,
        identical_sessions=n_sessions - len(divergent),
        divergent_sessions=divergent,
        missing_views=missing,
        cameras=cameras,
    )


def inventory(segments: pd.DataFrame) -> dict[str, object]:
    videos = video_table(segments)
    sessions = videos.drop_duplicates("session")
    sessions_per_subject = cast(pd.Series, sessions.groupby("subject").size())
    views_per_session = cast(pd.Series, videos.groupby("session").size()).value_counts()
    return {
        "segments": len(segments),
        "videos": len(videos),
        "subjects": _distinct(videos, "subject"),
        "sessions": _distinct(videos, "session"),
        "cameras": sorted(int(cam) for cam in _column(videos, "cam").unique()),
        "subjects_with_multiple_sessions": sorted(
            int(cast(int, subject))
            for subject, count in sessions_per_subject.items()
            if count > 1
        ),
        "sessions_by_view_count": {
            int(cast(int, views)): int(count)
            for views, count in sorted(views_per_session.items())
        },
    }


def _hours(frame: pd.DataFrame) -> float:
    return float(_column(frame, "duration").sum()) / 3600


def published_comparison(segments: pd.DataFrame) -> list[tuple[str, float, float, bool]]:
    sv = single_view(segments)
    observed: dict[str, float] = {
        "videos": float(_distinct(segments, "path")),
        "sv_segments": float(len(sv)),
        "mv_segments": float(len(segments)),
        "sv_hours": round(_hours(sv), 2),
        "mv_hours": round(_hours(segments), 2),
    }
    return [
        (name, published, observed[name], observed[name] == published)
        for name, published in PUBLISHED_STATISTICS.items()
    ]


def _label_row(rows: pd.DataFrame) -> dict[str, float]:
    durations = _column(rows, "duration").to_numpy()
    p25, median, p75 = (
        np.quantile(durations, [0.25, 0.5, 0.75]) if len(durations) else (np.nan,) * 3
    )
    return {
        "segments": len(rows),
        "hours": float(durations.sum()) / 3600,
        "p25_s": float(p25),
        "median_s": float(median),
        "p75_s": float(p75),
        "segments_ge_window": int((durations >= WINDOW_SECONDS).sum()),
        "sessions": _distinct(rows, "session"),
        "subjects": _distinct(rows, "subject"),
    }


def _with_label(frame: pd.DataFrame, label_id: int) -> pd.DataFrame:
    return _rows(frame, _column(frame, "label") == label_id)


def label_summary(segments: pd.DataFrame) -> pd.DataFrame:
    sv = single_view(segments)
    rows: list[dict[str, object]] = []
    for label_id, name in enumerate(LE2I_LABEL_NAMES):
        rows.append(
            {
                "label": name,
                "priority": name in PRIORITY_LABELS,
                "restricted": label_id in RESTRICTED_CLASSES,
                **_label_row(_with_label(sv, label_id)),
                "mv_segments": len(_with_label(segments, label_id)),
            }
        )
    return pd.DataFrame(rows).set_index("label")


@dataclass(frozen=True)
class BoundarySummary:
    boundaries: int
    contiguous: int
    gaps_below_step: int
    gaps_at_least_step: int
    overlaps_below_step: int
    overlaps_at_least_step: int
    contained_segments: int
    interior_gap_s: float
    overlap_s: float
    sessions_with_leading_offset: int
    leading_offset_s: float
    sessions_with_overlap: int
    large_overlap_pairs: dict[str, int]


def boundary_summary(sv: pd.DataFrame) -> BoundarySummary:
    """Fronteiras entre segmentos consecutivos de uma vista por sessão, ordenados
    por início. O fim real do vídeo não está no CSV: a extensão anotada termina
    no maior `end`, então não existe gap final mensurável."""
    ordered = sv.sort_values(["session", "start", "end"], kind="stable").reset_index(drop=True)
    by_session = ordered.groupby("session")
    ends = _column(ordered, "end")
    delta = cast(pd.Series, by_session["start"].shift(-1)) - ends
    next_label = cast(pd.Series, by_session["label"].shift(-1))
    previous_max_end = cast(
        pd.Series, by_session["end"].transform(lambda values: values.cummax().shift(1))
    )
    contained = ends <= previous_max_end + BOUNDARY_TOLERANCE_S

    measured = delta.dropna().to_numpy()
    large_overlap = (delta <= -GRID_STEP_S + BOUNDARY_TOLERANCE_S) & delta.notna()
    pairs: dict[str, int] = {}
    for label, following in zip(
        _column(ordered, "label")[large_overlap], next_label[large_overlap]
    ):
        key = f"{LE2I_LABEL_NAMES[int(label)]}->{LE2I_LABEL_NAMES[int(following)]}"
        pairs[key] = pairs.get(key, 0) + 1

    interior_gap_s = 0.0
    overlap_s = 0.0
    leading_offset_s = 0.0
    leading_sessions = 0
    for _session, group in by_session:
        extent = float(_column(group, "end").max())
        _, session_overlap_s, gap_intervals = sweep_gaps_and_overlap(
            list(zip(_column(group, "start"), _column(group, "end"))), extent
        )
        overlap_s += session_overlap_s
        for start, end, position in tag_gap_positions(gap_intervals, extent):
            if position == "leading" and end - start > BOUNDARY_TOLERANCE_S:
                leading_sessions += 1
                leading_offset_s += end - start
            elif position == "interior":
                interior_gap_s += end - start

    overlapping = (delta < -BOUNDARY_TOLERANCE_S) | contained
    small = GRID_STEP_S - BOUNDARY_TOLERANCE_S
    return BoundarySummary(
        boundaries=len(measured),
        contiguous=int((np.abs(measured) <= BOUNDARY_TOLERANCE_S).sum()),
        gaps_below_step=int(((measured > BOUNDARY_TOLERANCE_S) & (measured < small)).sum()),
        gaps_at_least_step=int((measured >= small).sum()),
        overlaps_below_step=int(((measured < -BOUNDARY_TOLERANCE_S) & (measured > -small)).sum()),
        overlaps_at_least_step=int((measured <= -small).sum()),
        contained_segments=int(contained.sum()),
        interior_gap_s=interior_gap_s,
        overlap_s=overlap_s,
        sessions_with_leading_offset=leading_sessions,
        leading_offset_s=leading_offset_s,
        sessions_with_overlap=_distinct(_rows(ordered, overlapping), "session"),
        large_overlap_pairs=dict(sorted(pairs.items(), key=lambda item: (-item[1], item[0]))),
    )


def session_grid(session_segments: pd.DataFrame) -> GridLabels:
    """Grade de TARGET_FPS sobre a extensão anotada [0, maior `end`), com a
    mesma regra de `labels_for_grid` do Le2i. O número real de quadros depende
    da duração do vídeo, que só o vídeo informa."""
    n_frames = int(np.floor(float(_column(session_segments, "end").max()) * TARGET_FPS))
    times = np.arange(n_frames, dtype=np.float64) / TARGET_FPS
    return labels_for_grid(
        cast(pd.DataFrame, session_segments[["start", "end", "label"]]), times, IGNORE_LABEL
    )


def _split_of_path(splits: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {
        str(path): name for name, frame in splits.items() for path in _column(frame, "path")
    }


def _with_split(videos: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    split_of_path = _split_of_path(splits)
    return videos.assign(
        split=[split_of_path.get(str(path)) for path in _column(videos, "path")]
    )


def session_split_map(videos: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> dict[int, str]:
    split_of_path = _split_of_path(splits)
    mapping: dict[int, str] = {}
    for session, group in videos.groupby("session"):
        names = {split_of_path.get(str(path), UNASSIGNED_SPLIT) for path in _column(group, "path")}
        mapping[int(cast(int, session))] = names.pop() if len(names) == 1 else UNASSIGNED_SPLIT
    return mapping


def _units_in_several_splits(assigned: pd.DataFrame, unit: str) -> int:
    listed = cast(pd.DataFrame, assigned.dropna(subset=["split"]))
    spread = cast(pd.Series, listed.groupby(unit)["split"].nunique())
    return int((spread > 1).sum())


def cs_split_problems(videos: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> list[str]:
    listed = {name: [str(path) for path in _column(frame, "path")] for name, frame in splits.items()}
    problems = [
        f"{name}: {len(paths) - len(set(paths))} paths repetidos"
        for name, paths in listed.items()
        if len(paths) != len(set(paths))
    ]
    for first, second in combinations(listed, 2):
        shared = set(listed[first]) & set(listed[second])
        if shared:
            problems.append(f"{first} ∩ {second}: {len(shared)} paths (ex.: {sorted(shared)[0]})")

    known = {str(path) for path in _column(videos, "path")}
    union: set[str] = set().union(*map(set, listed.values()))
    if union - known:
        problems.append(f"{len(union - known)} paths de split sem segmento no CSV")
    if known - union:
        problems.append(f"{len(known - union)} vídeos fora de todos os splits")

    assigned = _with_split(videos, splits)
    for unit in ("session", "subject"):
        crossing = _units_in_several_splits(assigned, unit)
        if crossing:
            problems.append(f"{crossing} valores de {unit} em mais de um split")
    return problems


def _membership_row(split: str, members: pd.DataFrame) -> dict[str, object]:
    return {
        "split": split,
        "videos": len(members),
        "sessions": _distinct(members, "session"),
        "subjects": _distinct(members, "subject"),
        "cameras": ",".join(str(cam) for cam in sorted(_column(members, "cam").unique())),
    }


def split_overview(videos: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    assigned = _with_split(videos, splits)
    split_column = _column(assigned, "split")
    rows = [_membership_row(name, _rows(assigned, split_column == name)) for name in splits]
    rows.append(_membership_row(UNASSIGNED_SPLIT, _rows(assigned, split_column.isna())))
    return pd.DataFrame(rows).set_index("split")


def shared_units_across_splits(
    videos: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> dict[str, int]:
    assigned = _with_split(videos, splits)
    return {unit: _units_in_several_splits(assigned, unit) for unit in ("session", "subject")}


def _session_splits(frame: pd.DataFrame, session_split: dict[int, str]) -> list[str]:
    return [
        session_split.get(int(session), UNASSIGNED_SPLIT) for session in _column(frame, "session")
    ]


def split_coverage(sv: pd.DataFrame, session_split: dict[int, str]) -> pd.DataFrame:
    assigned = sv.assign(split=_session_splits(sv, session_split))
    rows: list[dict[str, object]] = []
    for split in (*SPLIT_NAMES, UNASSIGNED_SPLIT):
        in_split = _rows(assigned, _column(assigned, "split") == split)
        if split == UNASSIGNED_SPLIT and in_split.empty:
            continue
        for label_id, name in enumerate(LE2I_LABEL_NAMES):
            rows.append(
                {"split": split, "label": name, **_label_row(_with_label(in_split, label_id))}
            )
    return pd.DataFrame(rows).set_index(["split", "label"])


def grid_summary(sv: pd.DataFrame, session_split: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session, group in sv.groupby("session"):
        grid = session_grid(group)
        rows.append(
            {
                "split": session_split.get(int(cast(int, session)), UNASSIGNED_SPLIT),
                "frames": len(grid.labels),
                "ignore_frames": int((grid.labels == IGNORE_LABEL).sum()),
                "overlap_resolved_frames": grid.n_overlap_resolved,
                "segments_without_frame": grid.n_segments_skipped,
            }
        )
    return cast(pd.DataFrame, pd.DataFrame(rows).groupby("split").sum())


def window_eligibility(sv: pd.DataFrame, session_split: dict[int, str]) -> pd.DataFrame:
    """Janelas causais de WINDOW_FRAMES quadros que terminam em `k_end`, como em
    `build_window_index`: rótulo do quadro final, replicação de borda no início
    e `IGNORE_LABEL` no quadro final descartado. Uma vista por sessão."""
    records: list[pd.DataFrame] = []
    for session, group in sv.groupby("session"):
        grid = session_grid(group).labels
        n_frames = len(grid)
        for stride in (EVAL_STRIDE, TRAIN_STRIDE):
            ends = window_end_indices(n_frames, stride)
            final_labels = grid[ends]
            kept = final_labels != IGNORE_LABEL
            ends, final_labels = ends[kept], final_labels[kept]
            if not len(ends):
                continue
            frames = np.stack([window_frame_indices(int(k_end), n_frames) for k_end in ends])
            window_labels = grid[frames]
            records.append(
                pd.DataFrame(
                    {
                        "stride": stride,
                        "split": session_split.get(int(cast(int, session)), UNASSIGNED_SPLIT),
                        "label": [LE2I_LABEL_NAMES[int(label)] for label in final_labels],
                        "session": int(cast(int, session)),
                        "subject": int(_column(group, "subject").iloc[0]),
                        "pure": (window_labels == final_labels[:, None]).all(axis=1),
                        "padded": ends < WINDOW_FRAMES - 1,
                        "with_ignore": (window_labels == IGNORE_LABEL).any(axis=1),
                    }
                )
            )
    if not records:
        return pd.DataFrame(
            columns=["windows", "pure", "padded", "with_ignore", "sessions", "subjects"]
        )
    windows = pd.concat(records, ignore_index=True)
    return cast(
        pd.DataFrame,
        windows.groupby(["stride", "split", "label"]).agg(
            windows=("pure", "size"),
            pure=("pure", "sum"),
            padded=("padded", "sum"),
            with_ignore=("with_ignore", "sum"),
            sessions=("session", "nunique"),
            subjects=("subject", "nunique"),
        ),
    )
