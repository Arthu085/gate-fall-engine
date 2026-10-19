"""Revisão visual da proveniência das anotações OmniFall nos MP4 do GMDCSA24 v2.1.

A planilha de revisão tem dois grupos de colunas. `AUTO_COLUMNS` é evidência
gerada automaticamente: fronteira anotada, rótulos esperados, hash da fonte e
quadros por PTS real. `HUMAN_COLUMNS` é o julgamento do revisor. Só as colunas
humanas decidem a aprovação; a evidência automática nunca confirma, sozinha,
que uma anotação está correta.
"""

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import TARGET_FPS
from gatefall.data.gmdcsa24.annotations import compare_fall_onsets, summarize_coverage
from gatefall.data.gmdcsa24.probe import read_frame_times
from gatefall.data.gmdcsa24.source import sha256_of

# Um passo da grade de 10 fps: a tolerância documentada na análise de viabilidade.
REVIEW_TOLERANCE_S = 1 / TARGET_FPS
SAMPLE_SALT = "gmdcsa24-provenance-v1"
VIDEOS_PER_STRATUM = 2
LONG_GAP_S = 0.5
ONSET_DELTA_S = 1.5
BOUNDARY_EPSILON_S = 1e-3
DEFAULT_REVIEW_DIR = Path("data/scratch/gmdcsa24/provenance")
REVIEW_SHEET = "review.csv"
CONTACT_SHEET_DIR = "contact_sheets"

FALL = "fall"
UNLABELED = "sem_rotulo"
FALL_ONSET = "fall_onset"
FALL_END = "fall_end"
TRANSITION = "transition"
GAP_START = "gap_start"
GAP_END = "gap_end"
CFR_FLAG = "desvio_cfr"

PENDING = "pending"
CONFIRMED = "confirmed"
DISCREPANCY = "discrepancy"
EXCLUDED = "excluded"
OUTCOMES = (PENDING, CONFIRMED, DISCREPANCY, EXCLUDED)

AUTO_COLUMNS = [
    "case_id",
    "video_id",
    "subject",
    "category",
    "selection",
    "boundary",
    "annotated_s",
    "label_before",
    "label_after",
    "source_sha256",
    "fps",
    "max_pts_drift_s",
    "cfr_frame_index",
    "cfr_frame_pts_s",
    "pts_frame_index",
    "pts_frame_s",
    "cfr_offset_s",
    "auto_flag",
]
HUMAN_COLUMNS = ["review_outcome", "observed_boundary_s", "reviewer", "reviewed_at", "notes"]
REVIEW_COLUMNS = AUTO_COLUMNS + HUMAN_COLUMNS


@dataclass(frozen=True)
class Boundary:
    time_s: float
    kind: str
    label_before: str
    label_after: str


@dataclass(frozen=True)
class SelectedVideo:
    video_id: str
    selection: str
    all_boundaries: bool


@dataclass
class ReviewStatus:
    pending: list[str] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return bool(self.confirmed or self.excluded) and not self.pending and not self.blocked


def empty_segments(segments: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, segments.iloc[0:0])


def labels_at(segments: pd.DataFrame, time_s: float, skip_start: float | None = None) -> str:
    """Rótulos que cobrem `time_s` em `[start, end)`, em ordem de `start`, unidos por `+`."""
    starts = segments["start"].to_numpy(float)
    mask = (starts <= time_s) & (time_s < segments["end"].to_numpy(float))
    if skip_start is not None:
        mask &= ~np.isclose(starts, skip_start)
    covering = cast(pd.DataFrame, segments[mask])
    labels = covering.sort_values("start", kind="stable")["label"]
    return "+".join(str(label) for label in labels) or UNLABELED


def boundary_kind(label_before: str, label_after: str) -> str:
    if label_after == FALL:
        return FALL_ONSET
    if label_before == FALL:
        return FALL_END
    if label_after == UNLABELED:
        return GAP_START
    if label_before == UNLABELED:
        return GAP_END
    return TRANSITION


def video_boundaries(segments: pd.DataFrame) -> list[Boundary]:
    """Fronteiras anotadas de um vídeo: início de cada segmento e fim seguido de lacuna.

    O rótulo anterior ignora o próprio segmento que começa, para que a
    sobreposição de arredondamento (< 0,1 s) entre segmentos vizinhos não vire
    duas fronteiras."""
    boundaries: dict[float, Boundary] = {}
    last_end = float(cast(float, segments["end"].max())) if not segments.empty else 0.0
    for start in sorted(set(segments["start"].astype(float))):
        if start <= 0:
            continue
        before = labels_at(segments, start - BOUNDARY_EPSILON_S, skip_start=start)
        starting = segments[np.isclose(segments["start"].to_numpy(float), start)]
        after = "+".join(str(label) for label in starting["label"])
        if before != after:
            boundaries[start] = Boundary(start, boundary_kind(before, after), before, after)
    for end in sorted(set(segments["end"].astype(float))):
        if end >= last_end or end in boundaries:
            continue
        if labels_at(segments, end + BOUNDARY_EPSILON_S) == UNLABELED:
            before = labels_at(segments, end - BOUNDARY_EPSILON_S)
            boundaries[end] = Boundary(end, boundary_kind(before, UNLABELED), before, UNLABELED)
    return [boundaries[time] for time in sorted(boundaries)]


def sample_key(identifier: str) -> str:
    """Ordem pseudoaleatória estável, sem RNG: SHA-256 do sal mais o `video_id`."""
    return hashlib.sha256(f"{SAMPLE_SALT}:{identifier}".encode()).hexdigest()


def anomaly_reasons(
    videos: pd.DataFrame,
    omnifall: pd.DataFrame,
    original_segments: pd.DataFrame,
    probes: pd.DataFrame,
) -> dict[str, list[str]]:
    """Motivos objetivos de inclusão obrigatória, recalculados a cada execução."""
    reasons: dict[str, list[str]] = {}

    def add(identifier: object, reason: str) -> None:
        reasons.setdefault(str(identifier), []).append(reason)

    decoded = probes[probes["error"].isna()]
    durations = cast(pd.Series, decoded.set_index("video_id")["duration_s"].astype(float))
    coverage = summarize_coverage(omnifall, durations)
    for row in coverage.to_dict("records"):
        if row["overlap_s"] >= REVIEW_TOLERANCE_S - 1e-9:
            add(row["video_id"], f"sobreposição de {row['overlap_s']:.2f} s")
        if row["max_gap_s"] >= LONG_GAP_S:
            add(row["video_id"], f"lacuna de {row['max_gap_s']:.2f} s")

    onsets = compare_fall_onsets(original_segments, omnifall, videos)
    for row in cast(pd.DataFrame, onsets[onsets["category"] == "Fall"]).to_dict("records"):
        if row["omnifall_fall_segments"] == 0:
            add(row["video_id"], "vídeo Fall sem `fall` no OmniFall")
        elif pd.isna(row["original_onset_s"]):
            add(row["video_id"], "`fall` sem intervalo Falling no CSV original")
        elif abs(float(row["onset_delta_s"])) >= ONSET_DELTA_S:
            add(row["video_id"], f"início do `fall` {row['onset_delta_s']:+.2f} s do autor")

    for identifier, drift in zip(decoded["video_id"], decoded["max_pts_drift_s"].astype(float)):
        if drift > REVIEW_TOLERANCE_S:
            add(identifier, f"desvio de PTS de {drift:.3f} s")
    return reasons


def stratum_eligible(category: str, boundaries: list[Boundary]) -> bool:
    kinds = {boundary.kind for boundary in boundaries}
    return FALL_ONSET in kinds if category == "Fall" else TRANSITION in kinds


def select_videos(
    videos: pd.DataFrame,
    boundaries_by_video: dict[str, list[Boundary]],
    anomalies: dict[str, list[str]],
) -> list[SelectedVideo]:
    """Anomalias e desvios de PTS entram sempre; cada estrato sujeito × categoria
    recebe `VIDEOS_PER_STRATUM` vídeos a mais, pela ordem de `sample_key`."""
    selected = [
        SelectedVideo(identifier, "; ".join(reasons), all_boundaries=True)
        for identifier, reasons in sorted(anomalies.items())
    ]
    for keys, group in videos.groupby(["subject", "category"]):
        subject, category = cast(tuple[int, str], keys)
        candidates = sorted(
            (
                str(identifier)
                for identifier in group["video_id"]
                if identifier not in anomalies
                and stratum_eligible(str(category), boundaries_by_video.get(identifier, []))
            ),
            key=sample_key,
        )
        selected.extend(
            SelectedVideo(identifier, f"estrato S{subject}/{category}", all_boundaries=False)
            for identifier in candidates[:VIDEOS_PER_STRATUM]
        )
    return sorted(selected, key=lambda item: item.video_id)


def boundaries_to_review(category: str, video: SelectedVideo, boundaries: list[Boundary]) -> list[Boundary]:
    """Todas as fronteiras de uma anomalia; início e fim do `fall` num vídeo Fall
    do estrato; a primeira transição sem `fall` num vídeo ADL do estrato."""
    if video.all_boundaries:
        return boundaries
    if category == "Fall":
        return [boundary for boundary in boundaries if boundary.kind in (FALL_ONSET, FALL_END)]
    return [boundary for boundary in boundaries if boundary.kind == TRANSITION][:1]


def case_id(identifier: str, boundary: Boundary) -> str:
    return f"{identifier.replace('/', '_')}_{boundary.kind}_{round(boundary.time_s * 1000):06d}ms"


def locate_boundary_frames(
    frame_times: np.ndarray, fps: float, time_s: float
) -> dict[str, object]:
    """Quadro que a grade CFR associa a `time_s` contra o primeiro quadro com PTS >= `time_s`.

    `cfr_offset_s` é o PTS real do quadro CFR menos o instante anotado: o erro
    temporal que `build_time_grid` cometeria nessa fronteira."""
    relative = frame_times - frame_times[0]
    last = len(relative) - 1
    cfr_index = int(np.clip(np.round(time_s * fps), 0, last))
    pts_index = int(min(np.searchsorted(relative, time_s - 1e-6, side="left"), last))
    offset = float(relative[cfr_index] - time_s)
    return {
        "cfr_frame_index": cfr_index,
        "cfr_frame_pts_s": round(float(relative[cfr_index]), 4),
        "pts_frame_index": pts_index,
        "pts_frame_s": round(float(relative[pts_index]), 4),
        "cfr_offset_s": round(offset, 4),
        "auto_flag": CFR_FLAG if abs(offset) > REVIEW_TOLERANCE_S else "",
    }


def build_review_plan(
    root: Path,
    videos: pd.DataFrame,
    omnifall: pd.DataFrame,
    original_segments: pd.DataFrame,
    probes: pd.DataFrame,
) -> pd.DataFrame:
    """Planilha de revisão com evidência automática e colunas humanas `pending`.

    `omnifall` usa nomes de classe (`with_label_names`)."""
    by_video = {str(key): group for key, group in omnifall.groupby("video_id")}
    boundaries_by_video = {key: video_boundaries(group) for key, group in by_video.items()}
    anomalies = anomaly_reasons(videos, omnifall, original_segments, probes)
    indexed_videos = videos.set_index("video_id")
    indexed_probes = probes.set_index("video_id")
    rows: list[dict[str, object]] = []
    for video in select_videos(videos, boundaries_by_video, anomalies):
        info = indexed_videos.loc[video.video_id]
        probe = indexed_probes.loc[video.video_id]
        category = str(info["category"])
        boundaries = boundaries_to_review(category, video, boundaries_by_video.get(video.video_id, []))
        if not boundaries:
            continue
        path = root / str(info["relative_path"])
        frame_times = read_frame_times(path)
        digest = sha256_of(path)
        fps = float(probe["fps"])
        for boundary in boundaries:
            rows.append(
                {
                    "case_id": case_id(video.video_id, boundary),
                    "video_id": video.video_id,
                    "subject": int(info["subject"]),
                    "category": category,
                    "selection": video.selection,
                    "boundary": boundary.kind,
                    "annotated_s": round(boundary.time_s, 4),
                    "label_before": boundary.label_before,
                    "label_after": boundary.label_after,
                    "source_sha256": digest,
                    "fps": round(fps, 4),
                    "max_pts_drift_s": round(float(probe["max_pts_drift_s"]), 4),
                    **locate_boundary_frames(frame_times, fps, boundary.time_s),
                    "review_outcome": PENDING,
                    "observed_boundary_s": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def ensure_ignored_output(path: Path) -> None:
    """Recusa gravar quadros identificáveis num caminho que o Git versionaria.

    Fora de um repositório Git, `check-ignore` sai com 128 e o caminho é aceito."""
    anchor = path.resolve()
    while not anchor.exists():
        anchor = anchor.parent
    if not anchor.is_dir():
        anchor = anchor.parent
    result = subprocess.run(
        ["git", "-C", str(anchor), "check-ignore", "-q", "--no-index", str(path.resolve())],
        capture_output=True,
        text=True,
    )
    if result.returncode == 1:
        raise ValueError(
            f"{path} não é ignorado pelo Git; quadros de vídeo identificáveis só podem ser "
            "gravados num diretório ignorado, como data/scratch/"
        )


def check_review_target(review_dir: Path, force: bool) -> None:
    ensure_ignored_output(review_dir)
    sheet = review_dir / REVIEW_SHEET
    if sheet.exists() and not force:
        raise FileExistsError(
            f"{sheet} já existe e pode conter julgamentos de revisor; use --force para recriá-lo"
        )


def write_review_sheet(plan: pd.DataFrame, review_dir: Path, force: bool) -> Path:
    check_review_target(review_dir, force)
    sheet = review_dir / REVIEW_SHEET
    review_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(sheet, index=False)
    return sheet


def read_review_sheet(review_dir: Path) -> pd.DataFrame:
    sheet = pd.read_csv(review_dir / REVIEW_SHEET, dtype=str, keep_default_na=False)
    missing = [column for column in REVIEW_COLUMNS if column not in sheet.columns]
    if missing:
        raise ValueError(f"planilha de revisão sem as colunas {missing}")
    return sheet


def contact_sheet_path(review_dir: Path, identifier: str) -> Path:
    return review_dir / CONTACT_SHEET_DIR / f"{identifier}.png"


def parse_seconds(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def evaluate_case(row: dict[str, str]) -> tuple[str, str]:
    """Estado de um caso a partir só das colunas humanas e da marca `auto_flag`.

    Devolve (estado, motivo), com estado em pending, confirmed, excluded ou blocked."""
    outcome = row["review_outcome"].strip()
    if outcome not in OUTCOMES:
        return "blocked", f"review_outcome inválido {outcome!r}"
    if outcome == PENDING:
        return "pending", "sem revisão humana"
    if not row["reviewer"].strip() or not row["reviewed_at"].strip():
        return "blocked", "revisão sem reviewer ou reviewed_at"
    if outcome == EXCLUDED:
        if not row["notes"].strip():
            return "blocked", "exclusão sem justificativa em notes"
        return "excluded", row["notes"].strip()
    if outcome == DISCREPANCY:
        return "blocked", "divergência relatada pelo revisor; investigar antes de aprovar"
    observed = parse_seconds(row["observed_boundary_s"])
    if observed is None:
        return "blocked", "confirmado sem observed_boundary_s numérico"
    delta = observed - float(row["annotated_s"])
    if abs(delta) > REVIEW_TOLERANCE_S + 1e-9:
        return "blocked", f"fronteira observada a {delta:+.3f} s da anotada (> {REVIEW_TOLERANCE_S:g} s)"
    if row["auto_flag"] == CFR_FLAG:
        return "blocked", (
            f"grade CFR desloca a fronteira em {float(row['cfr_offset_s']):+.3f} s; "
            "a anotação pode estar certa, mas o vídeo só entra excluído ou com grade por PTS"
        )
    return "confirmed", f"fronteira observada a {delta:+.3f} s"


def check_auto_evidence(
    sheet: pd.DataFrame, omnifall: pd.DataFrame, digests: dict[str, str]
) -> dict[str, str]:
    """Confere a evidência automática contra o OmniFall pinado e o hash atual da fonte.

    Devolve `case_id` → problema; qualquer problema bloqueia o caso."""
    by_video = {str(key): group for key, group in omnifall.groupby("video_id")}
    problems: dict[str, str] = {}
    for row in sheet.to_dict("records"):
        identifier = str(row["video_id"])
        if digests.get(identifier) != row["source_sha256"]:
            problems[str(row["case_id"])] = "SHA-256 da fonte diverge da planilha"
            continue
        expected = {
            case_id(identifier, boundary): boundary
            for boundary in video_boundaries(by_video.get(identifier, empty_segments(omnifall)))
        }
        boundary = expected.get(str(row["case_id"]))
        if boundary is None or (boundary.label_before, boundary.label_after) != (
            row["label_before"],
            row["label_after"],
        ):
            problems[str(row["case_id"])] = "fronteira ou rótulos divergem do OmniFall pinado"
    return problems


def summarize_review(sheet: pd.DataFrame, evidence_problems: dict[str, str]) -> ReviewStatus:
    status = ReviewStatus()
    for row in sheet.to_dict("records"):
        identifier = str(row["case_id"])
        if identifier in evidence_problems:
            status.blocked.append(f"{identifier}: {evidence_problems[identifier]}")
            continue
        state, reason = evaluate_case(cast(dict[str, str], row))
        entry = f"{identifier}: {reason}"
        {
            "pending": status.pending,
            "confirmed": status.confirmed,
            "excluded": status.excluded,
            "blocked": status.blocked,
        }[state].append(entry)
    return status


def source_digests(root: Path, sheet: pd.DataFrame, videos: pd.DataFrame) -> dict[str, str]:
    paths = cast(pd.Series, videos.set_index("video_id")["relative_path"])
    return {
        str(identifier): sha256_of(root / str(paths[identifier]))
        for identifier in sorted(set(sheet["video_id"]))
        if identifier in paths.index
    }


def report_plan(plan: pd.DataFrame) -> None:
    print(f"\n=== amostra de revisão de proveniência (sal {SAMPLE_SALT!r}) ===")
    print(f"vídeos: {plan['video_id'].nunique()}; fronteiras (casos): {len(plan)}")
    per_stratum = plan.drop_duplicates("video_id").groupby(["subject", "category"]).size()
    print("vídeos por sujeito e categoria:")
    print(per_stratum.to_string())
    print("casos por tipo de fronteira:")
    print(plan["boundary"].value_counts().to_string())
    print(f"casos com |desvio CFR| > {REVIEW_TOLERANCE_S:g} s: {(plan['auto_flag'] == CFR_FLAG).sum()}")
    for video, group in plan.groupby("video_id", sort=True):
        first = group.iloc[0]
        print(f"  {video} [{first['selection']}] sha256 {str(first['source_sha256'])[:12]}")
        for row in group.to_dict("records"):
            flag = f" [{row['auto_flag']}]" if row["auto_flag"] else ""
            print(
                f"    {row['boundary']} {float(row['annotated_s']):.3f} s "
                f"{row['label_before']} → {row['label_after']}; quadro CFR "
                f"#{row['cfr_frame_index']} ({float(row['cfr_offset_s']):+.3f} s){flag}"
            )


def report_status(status: ReviewStatus, sheet: pd.DataFrame, review_dir: Path) -> None:
    rendered = sum(
        contact_sheet_path(review_dir, str(identifier)).is_file() for identifier in sheet["case_id"]
    )
    print(f"\n=== estado da revisão de proveniência ({review_dir / REVIEW_SHEET}) ===")
    print(
        f"casos: {len(sheet)}; evidência automática renderizada: {rendered} "
        "(não confirma anotação)"
    )
    print(f"confirmados por revisor humano: {len(status.confirmed)}")
    print(f"excluídos com justificativa: {len(status.excluded)}")
    print(f"pendentes: {len(status.pending)}")
    print(f"bloqueados: {len(status.blocked)}")
    for title, entries in (("bloqueados", status.blocked), ("excluídos", status.excluded)):
        if entries:
            print(f"{title}:")
            for entry in entries:
                print(f"  {entry}")
    if status.excluded:
        excluded = sheet[sheet["review_outcome"].str.strip() == EXCLUDED]["video_id"]
        print(f"vídeos excluídos do piloto: {sorted(set(excluded))}")
    verdict = "APROVADA" if status.approved else ("BLOQUEADA" if status.blocked else "PENDENTE")
    print(f"proveniência: {verdict}")
