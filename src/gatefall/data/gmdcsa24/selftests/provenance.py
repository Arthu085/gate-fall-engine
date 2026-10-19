"""Selftest sintético da revisão de proveniência (`provenance.py`, `review_render.py`).

Monta vídeos sintéticos num diretório temporário, um deles com taxa variável
(salto de 0,4 s no PTS a partir do quadro 15), e segmentos OmniFall em
memória. Não toca no dataset real nem no Hugging Face. Requer FFmpeg e Git.
"""

import contextlib
import io
import math
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from PIL import Image

from gatefall.data.gmdcsa24.probe import probe_checkout, read_frame_times
from gatefall.data.gmdcsa24.provenance import (
    CFR_FLAG,
    CONFIRMED,
    DISCREPANCY,
    EXCLUDED,
    FALL_END,
    FALL_ONSET,
    GAP_END,
    GAP_START,
    HUMAN_COLUMNS,
    PENDING,
    REVIEW_COLUMNS,
    TRANSITION,
    UNLABELED,
    Boundary,
    SelectedVideo,
    ReviewPlan,
    ReviewStatus,
    anomaly_reasons,
    build_review_plan,
    contact_sheet_path,
    ensure_ignored_output,
    evaluate_case,
    excluded_videos,
    extend_review_sheet,
    locate_boundary_frames,
    read_review_sheet,
    review_status,
    select_videos,
    video_boundaries,
    write_review_sheet,
)
from gatefall.data.gmdcsa24.review_render import decode_scaled_frames, render_review
from gatefall.data.gmdcsa24.source import build_inventory, sha256_of

_FRAMES = 30
_JUMP_FRAME = 15
_JUMP_S = 0.4
_VFR_VIDEO = "Subject_1/Fall/01"
_STRATUM_STEMS = ("02", "03", "04")
_REVIEWER = {"reviewer": "revisor", "reviewed_at": "2026-10-19"}


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _segments(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["video_id", "label", "start", "end"])


def _encode(path: Path, variable_rate: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Com taxa variável, o quadro N tem PTS N/10 s, mais 0,4 s a partir do 15.
    jump = f"+gte(N\\,{_JUMP_FRAME})*{_JUMP_S}" if variable_rate else ""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x48:rate=10:duration={_FRAMES / 10}",
            "-vf",
            f"settb=1/100,setpts='(N/10{jump})/TB'",
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "1/100",
            "-c:v",
            "mpeg4",
            str(path),
        ],
        check=True,
    )


def check_boundaries() -> list[bool]:
    dense = video_boundaries(
        _segments(
            [
                ("v", "walk", 0.0, 1.0),
                ("v", "fall", 1.0, 2.0),
                ("v", "fallen", 1.98, 3.0),
                ("v", "sitting", 3.5, 4.0),
            ]
        )
    )
    overlapped = video_boundaries(
        _segments(
            [
                ("v", "stand_up", 0.0, 7.0),
                ("v", "sitting", 2.5, 5.0),
                ("v", "standing", 6.0, 7.0),
            ]
        )
    )
    return [
        _check(
            "fronteiras: início e fim de `fall`, lacuna; sobreposição de arredondamento não duplica",
            dense
            == [
                Boundary(1.0, FALL_ONSET, "walk", "fall"),
                Boundary(1.98, FALL_END, "fall", "fallen"),
                Boundary(3.0, GAP_START, "fallen", UNLABELED),
                Boundary(3.5, GAP_END, UNLABELED, "sitting"),
            ],
        ),
        _check(
            "sobreposição real mantém o rótulo coberto e o fim do segmento interno",
            overlapped
            == [
                Boundary(2.5, TRANSITION, "stand_up", "sitting"),
                Boundary(6.0, TRANSITION, "stand_up", "standing"),
            ],
        ),
    ]


def check_selection() -> list[bool]:
    rows = []
    boundaries: dict[str, list[Boundary]] = {}
    for subject in (1, 2):
        for category in ("ADL", "Fall"):
            for stem in ("01", "02", "03", "04"):
                identifier = f"Subject_{subject}/{category}/{stem}"
                rows.append({"video_id": identifier, "subject": subject, "category": category})
                kind = FALL_ONSET if category == "Fall" else TRANSITION
                # O vídeo 04 não tem a fronteira que o estrato exige.
                boundaries[identifier] = [] if stem == "04" else [Boundary(1.0, kind, "a", "b")]
    videos = pd.DataFrame(rows)
    anomalies = {"Subject_1/Fall/02": ["lacuna de 0.60 s"], "Subject_2/ADL/04": ["x"]}
    first, short = select_videos(videos, boundaries, anomalies)
    shuffled = select_videos(videos.sample(frac=1.0, random_state=7), boundaries, anomalies)
    stratified = [video for video in first if not video.all_boundaries]
    per_stratum = pd.Series(
        [video.video_id.rsplit("/", 1)[0] for video in stratified]
    ).value_counts()

    def picks(prefix: str, selection: list[SelectedVideo]) -> list[SelectedVideo]:
        return [video for video in selection if video.video_id.startswith(prefix)]

    # S1/ADL tem três elegíveis (01–03); S1/Fall só dois, porque 02 é anomalia.
    adl_excluded = frozenset({picks("Subject_1/ADL/", stratified)[0].video_id})
    replaced, replaced_short = select_videos(videos, boundaries, anomalies, adl_excluded)
    fall_excluded = frozenset({picks("Subject_1/Fall/", stratified)[0].video_id})
    _, fall_short = select_videos(videos, boundaries, anomalies, fall_excluded)
    adl_after = picks("Subject_1/ADL/", [video for video in replaced if not video.all_boundaries])
    return [
        _check(
            "seleção determinística e independente da ordem das linhas",
            (first, short) == shuffled and short == (),
        ),
        _check(
            "anomalias entram com todas as fronteiras, fora da cota do estrato",
            {video.video_id for video in first if video.all_boundaries} == set(anomalies)
            and not any(video.video_id in anomalies for video in stratified),
        ),
        _check(
            "dois vídeos por sujeito × categoria, só entre os elegíveis",
            per_stratum.to_dict()
            == {"Subject_1/ADL": 2, "Subject_1/Fall": 2, "Subject_2/ADL": 2, "Subject_2/Fall": 2}
            and not any(video.video_id.endswith("/04") for video in stratified),
        ),
        _check(
            "vídeo de estrato excluído continua exigido e puxa o próximo candidato",
            len(adl_after) == 3
            and adl_excluded <= {video.video_id for video in adl_after}
            and sum(video.selection.endswith(", reposição") for video in adl_after) == 1
            and replaced_short == (),
        ),
        _check("estrato sem candidato de reposição fica curto", fall_short == ("S1/Fall",)),
    ]


def check_anomaly_reasons() -> bool:
    videos = pd.DataFrame(
        {
            "video_id": ["A/ADL/01", "A/Fall/01", "A/Fall/02", "A/Fall/03"],
            "category": ["ADL", "Fall", "Fall", "Fall"],
        }
    )
    omnifall = _segments(
        [
            ("A/ADL/01", "walk", 0.0, 1.1),
            ("A/ADL/01", "sitting", 1.0, 2.0),
            ("A/ADL/01", "standing", 2.6, 3.0),
            ("A/Fall/01", "lying", 0.0, 3.0),
            ("A/Fall/02", "walk", 0.0, 1.0),
            ("A/Fall/02", "fall", 1.0, 3.0),
            ("A/Fall/03", "walk", 0.0, 0.4),
            ("A/Fall/03", "fall", 0.4, 3.0),
        ]
    )
    original = pd.DataFrame(
        {"video_id": ["A/Fall/03"], "label": ["Falling"], "start": [2.0], "end": [3.0]}
    )
    probes = pd.DataFrame(
        {
            "video_id": videos["video_id"],
            "duration_s": [3.0] * 4,
            "max_pts_drift_s": [0.02, 0.12, 0.0, 0.0],
            "error": [None] * 4,
        }
    )
    reasons = anomaly_reasons(videos, omnifall, original, probes)
    return _check(
        "anomalias: sobreposição >= 0,1 s, lacuna >= 0,5 s, Fall sem `fall`, sem Falling, "
        "início divergente e desvio de PTS",
        reasons
        == {
            "A/ADL/01": ["sobreposição de 0.10 s", "lacuna de 0.60 s"],
            "A/Fall/01": ["vídeo Fall sem `fall` no OmniFall", "desvio de PTS de 0.120 s"],
            "A/Fall/02": ["`fall` sem intervalo Falling no CSV original"],
            "A/Fall/03": ["início do `fall` -1.60 s do autor"],
        },
    )


def check_locate_frames() -> bool:
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.55, 0.65, 0.75])
    shifted = locate_boundary_frames(times, 10.0, 0.5)
    aligned = locate_boundary_frames(times, 10.0, 0.2)
    return _check(
        "quadro CFR contra primeiro quadro por PTS; desvio > 0,1 s marcado",
        shifted["cfr_frame_index"] == 5
        and shifted["pts_frame_index"] == 4
        and math.isclose(float(str(shifted["cfr_offset_s"])), 0.15)
        and shifted["auto_flag"] == CFR_FLAG
        and aligned["cfr_offset_s"] == 0.0
        and aligned["auto_flag"] == "",
    )


def _human(outcome: str, observed: str = "", notes: str = "", flag: str = "") -> dict[str, str]:
    return {
        "annotated_s": "1.0",
        "auto_flag": flag,
        "cfr_offset_s": "0.15" if flag else "0.0",
        "review_outcome": outcome,
        "observed_boundary_s": observed,
        **({"reviewer": "", "reviewed_at": ""} if outcome == PENDING else _REVIEWER),
        "notes": notes,
    }


def check_review_states() -> bool:
    states = {
        "pendente": evaluate_case(_human(PENDING))[0],
        "confirmado na tolerância": evaluate_case(_human(CONFIRMED, "1.08"))[0],
        "confirmado fora da tolerância": evaluate_case(_human(CONFIRMED, "1.15"))[0],
        "confirmado sem instante": evaluate_case(_human(CONFIRMED))[0],
        "confirmado sem revisor": evaluate_case({**_human(CONFIRMED, "1.0"), "reviewer": ""})[0],
        "data de revisão inválida": evaluate_case(
            {**_human(CONFIRMED, "1.0"), "reviewed_at": "ontem"}
        )[0],
        "divergência": evaluate_case(_human(DISCREPANCY, "1.4"))[0],
        "desvio CFR confirmado": evaluate_case(_human(CONFIRMED, "1.0", flag=CFR_FLAG))[0],
        "desvio CFR excluído": evaluate_case(_human(EXCLUDED, notes="PTS", flag=CFR_FLAG))[0],
        "exclusão sem nota": evaluate_case(_human(EXCLUDED))[0],
        "outcome inválido": evaluate_case(_human("ok"))[0],
    }
    expected = {
        "pendente": "pending",
        "confirmado na tolerância": "confirmed",
        "confirmado fora da tolerância": "blocked",
        "confirmado sem instante": "blocked",
        "confirmado sem revisor": "blocked",
        "data de revisão inválida": "blocked",
        "divergência": "blocked",
        "desvio CFR confirmado": "blocked",
        "desvio CFR excluído": "excluded",
        "exclusão sem nota": "blocked",
        "outcome inválido": "blocked",
    }
    return _check(
        "estados: só revisão humana na tolerância aprova; divergência e desvio CFR bloqueiam",
        states == expected,
    )


def check_ignored_output(tmp: Path) -> bool:
    repository = tmp / "repo"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text("scratch/\n")

    def refused(path: Path) -> bool:
        try:
            ensure_ignored_output(path)
        except ValueError:
            return True
        return False

    return _check(
        "quadros só em diretório ignorado pelo Git ou fora de repositório",
        refused(repository / "frames")
        and not refused(repository / "scratch" / "review")
        and not refused(tmp / "outside"),
    )


def _write_review_fixture(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clone com um vídeo de taxa variável (anomalia) e três candidatos por estrato.

    Os candidatos de estrato são cópias de um único MP4 de taxa constante."""
    _encode(root / "Subject 1/Fall/01.mp4", variable_rate=True)
    template = root.parent / "cfr.mp4"
    _encode(template, variable_rate=False)
    rows = [
        (_VFR_VIDEO, "walk", 0.0, 1.4),
        (_VFR_VIDEO, "fall", 1.4, 2.2),
        (_VFR_VIDEO, "fallen", 2.2, 3.0),
    ]
    for subject in (1, 2):
        for category in ("ADL", "Fall"):
            for stem in _STRATUM_STEMS:
                target = root / f"Subject {subject}/{category}/{stem}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(template, target)
                identifier = f"Subject_{subject}/{category}/{stem}"
                if category == "ADL":
                    rows += [(identifier, "walk", 0.0, 1.0), (identifier, "sitting", 1.0, 3.0)]
                else:
                    rows += [
                        (identifier, "walk", 0.0, 1.0),
                        (identifier, "fall", 1.0, 2.0),
                        (identifier, "fallen", 2.0, 3.0),
                    ]
    omnifall = _segments(rows)
    falls = omnifall[omnifall["label"] == "fall"]
    original = pd.DataFrame(
        {
            "video_id": falls["video_id"],
            "label": "Falling",
            "start": falls["start"],
            "end": 3.0,
        }
    )
    videos = build_inventory(root).videos
    return videos, omnifall, original


def check_plan_and_render(tmp: Path) -> list[bool]:
    root = tmp / "GMDCSA24"
    videos, omnifall, original = _write_review_fixture(root)
    probes = probe_checkout(root, videos)
    with contextlib.redirect_stdout(io.StringIO()):
        plan = build_review_plan(root, videos, omnifall, original, probes)
    cases = plan.cases
    onset = cases.set_index("case_id").loc["Subject_1_Fall_01_fall_onset_001400ms"]
    stratified = cast(pd.DataFrame, cases[cases["video_id"] != _VFR_VIDEO]).drop_duplicates(
        "video_id"
    )

    review_dir = tmp / "review"
    write_review_sheet(cases, review_dir, force=False)
    try:
        write_review_sheet(cases, review_dir, force=False)
        overwrite_refused = False
    except FileExistsError:
        overwrite_refused = True
    sheet = read_review_sheet(review_dir)
    rendered = dict(render_review(root, review_dir, sheet, videos, omnifall))
    image = Image.open(contact_sheet_path(review_dir, str(onset.name)))

    fall_path = root / "Subject 1/Fall/01.mp4"
    relative = read_frame_times(fall_path)
    expected_frames = int((np.abs(relative - relative[0] - 1.4) <= 0.5 + 1e-9).sum())
    decoded = decode_scaled_frames(fall_path, list(range(_FRAMES)), 32, 24)
    try:
        decode_scaled_frames(fall_path, [_FRAMES], 32, 24)
        exhausted = False
    except EOFError:
        exhausted = True
    return [
        _check(
            "plano: anomalia com todas as fronteiras e dois vídeos por estrato, tudo `pending`",
            cases.columns.tolist() == REVIEW_COLUMNS
            and plan.short_strata == ()
            and len(cases) == 14
            and stratified.groupby(["subject", "category"]).size().tolist() == [2, 2, 2, 2]
            and sorted(cases.loc[cases["video_id"] == _VFR_VIDEO, "boundary"])
            == [FALL_END, FALL_ONSET]
            and set(cases["review_outcome"]) == {PENDING}
            and (cases[HUMAN_COLUMNS[1:]] == "").all().all(),
        ),
        # 30 quadros em 3,4 s dão 8,82 fps médios: a grade CFR põe 1,4 s no
        # quadro 12 (PTS 1,2 s), mas o quadro com PTS 1,4 s é o 14.
        _check(
            "taxa variável: quadro CFR a -0,2 s da fronteira é marcado; hash da fonte gravado",
            onset["auto_flag"] == CFR_FLAG
            and int(onset["cfr_frame_index"]) == 12
            and int(onset["pts_frame_index"]) == 14
            and math.isclose(float(onset["cfr_offset_s"]), -0.2, abs_tol=1e-3)
            and onset["source_sha256"] == sha256_of(fall_path),
        ),
        _check("planilha existente não é sobrescrita sem --force", overwrite_refused),
        _check(
            "decodificação sem duplicar quadros de vídeo com taxa variável",
            len(decoded) == _FRAMES and len(relative) == _FRAMES and exhausted,
        ),
        _check(
            "contact sheet por caso com os quadros de PTS na janela de ±0,5 s",
            rendered[str(onset.name)] == expected_frames
            and set(rendered) == set(cases["case_id"])
            and image.width > 0,
        ),
        *check_review_integrity(
            tmp / "integrity",
            sheet,
            lambda excluded: build_review_plan(root, videos, omnifall, original, probes, excluded),
        ),
    ]


def _roundtrip(sheet: pd.DataFrame, review_dir: Path) -> pd.DataFrame:
    write_review_sheet(sheet, review_dir, force=True)
    return read_review_sheet(review_dir)


def _complete(sheet: pd.DataFrame) -> pd.DataFrame:
    """Confirma cada fronteira no instante anotado e exclui o vídeo de taxa variável."""
    done = sheet.copy()
    vfr = done["video_id"] == _VFR_VIDEO
    done.loc[~vfr, "review_outcome"] = CONFIRMED
    done.loc[~vfr, "observed_boundary_s"] = done.loc[~vfr, "annotated_s"]
    done.loc[vfr, "review_outcome"] = EXCLUDED
    done.loc[vfr, "notes"] = "grade CFR desloca a queda"
    for column, value in _REVIEWER.items():
        done[column] = value
    return done


def _exclude(sheet: pd.DataFrame, identifiers: set[str]) -> pd.DataFrame:
    changed = sheet.copy()
    rows = changed["video_id"].isin(sorted(identifiers))
    changed.loc[rows, "review_outcome"] = EXCLUDED
    changed.loc[rows, "observed_boundary_s"] = ""
    changed.loc[rows, "notes"] = "oclusão"
    for column, value in _REVIEWER.items():
        changed.loc[rows, column] = value
    return changed


def _blocked_by(status: ReviewStatus, text: str) -> bool:
    return not status.approved and any(text in entry for entry in status.blocked)


def check_review_integrity(
    review_dir: Path, untouched: pd.DataFrame, rebuild: Callable[[frozenset[str]], ReviewPlan]
) -> list[bool]:
    """Toda planilha passa por CSV antes de ser avaliada, como no `review-status`."""

    def status_of(sheet: pd.DataFrame) -> ReviewStatus:
        stored = _roundtrip(sheet, review_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            plan = rebuild(excluded_videos(stored))
        return review_status(stored, plan)

    complete = _complete(untouched)
    confirmed_rows = complete.index[complete["review_outcome"] == CONFIRMED]
    first = confirmed_rows[0]
    flagged = complete.index[complete["auto_flag"] == CFR_FLAG]
    stratum_video = str(complete.loc[first, "video_id"])

    pending = status_of(untouched)
    approved = status_of(complete)
    reset = complete.copy()
    reset.loc[first, ["review_outcome", "observed_boundary_s"]] = [PENDING, ""]
    without_pending = reset.drop(index=first)
    without_flagged = complete.drop(index=flagged)
    duplicated = pd.concat([complete, complete.loc[[first]]], ignore_index=True)
    extra = pd.concat(
        [complete, complete.loc[[first]].assign(case_id="Subject_2_ADL_02_transition_009999ms")],
        ignore_index=True,
    )
    shifted = complete.copy()
    shifted.loc[first, "annotated_s"] = str(float(shifted.loc[first, "annotated_s"]) + 0.05)
    shifted.loc[first, "observed_boundary_s"] = shifted.loc[first, "annotated_s"]
    cleared = complete.copy()
    cleared.loc[flagged, ["auto_flag", "review_outcome", "notes"]] = ["", CONFIRMED, ""]
    cleared.loc[flagged, "observed_boundary_s"] = cleared.loc[flagged, "annotated_s"]
    offset = complete.copy()
    offset.loc[first, "cfr_offset_s"] = "0.05"
    all_excluded = _exclude(complete, set(complete["video_id"]))

    excluded_once = _exclude(complete, {stratum_video})
    with contextlib.redirect_stdout(io.StringIO()):
        extended = extend_review_sheet(
            _roundtrip(excluded_once, review_dir), rebuild(frozenset({stratum_video}))
        )
    added = cast(pd.DataFrame, extended[~extended["case_id"].isin(complete["case_id"])])
    replacement = str(added["video_id"].iloc[0]) if not added.empty else ""
    kept = extended.set_index("case_id").loc[complete["case_id"], "review_outcome"]
    replaced = _roundtrip(extended, review_dir)
    new_rows = replaced["case_id"].isin(added["case_id"])
    replaced.loc[new_rows, "review_outcome"] = CONFIRMED
    replaced.loc[new_rows, "observed_boundary_s"] = replaced.loc[new_rows, "annotated_s"]
    for column, value in _REVIEWER.items():
        replaced.loc[new_rows, column] = value
    try:
        extend_review_sheet(duplicated, rebuild(frozenset()))
        extend_refused = False
    except ValueError:
        extend_refused = True
    write_review_sheet(complete.assign(extra_column=""), review_dir, force=True)
    try:
        read_review_sheet(review_dir)
        extra_column_refused = False
    except ValueError:
        extra_column_refused = True

    replaced_status = status_of(replaced)
    exhausted = status_of(_exclude(replaced, {replacement}))
    excluded_status = status_of(all_excluded)
    return [
        _check(
            "planilha intocada continua pendente, sem bloqueio",
            not pending.approved and len(pending.pending) == 14 and not pending.blocked,
        ),
        _check(
            "revisão completa e íntegra aprova: confirmados e exclusão justificada",
            approved.approved and len(approved.confirmed) == 12 and len(approved.excluded) == 2,
        ),
        _check(
            "apagar caso pendente ou com `desvio_cfr` não aprova",
            _blocked_by(status_of(without_pending), "ausente")
            and _blocked_by(status_of(without_flagged), "ausente"),
        ),
        _check(
            "caso repetido ou fora da amostra bloqueia",
            _blocked_by(status_of(duplicated), "repetido")
            and _blocked_by(status_of(extra), "fora da amostra"),
        ),
        _check(
            "instante anotado alterado bloqueia, mesmo confirmado",
            _blocked_by(status_of(shifted), "['annotated_s']"),
        ),
        _check(
            "apagar `desvio_cfr` para confirmar o caso bloqueia",
            len(flagged) >= 1 and _blocked_by(status_of(cleared), "auto_flag"),
        ),
        _check(
            "offset CFR alterado bloqueia",
            _blocked_by(status_of(offset), "['cfr_offset_s']"),
        ),
        _check(
            "revisão toda excluída não aprova e exige reposições",
            not excluded_status.confirmed and _blocked_by(excluded_status, "ausente"),
        ),
        _check(
            "exclusão em estrato exige reposição; --extend a acrescenta `pending` e mantém julgamentos",
            _blocked_by(status_of(excluded_once), "ausente")
            and replacement != ""
            and set(added["review_outcome"]) == {PENDING}
            and added["selection"].str.endswith(", reposição").all()
            and kept.tolist() == _exclude(complete, {stratum_video})["review_outcome"].tolist()
            and replaced_status.approved,
        ),
        _check(
            "estrato sem candidato de reposição bloqueia",
            _blocked_by(exhausted, "estrato S"),
        ),
        _check(
            "--extend recusa planilha adulterada; coluna nova é recusada",
            extend_refused and extra_column_refused,
        ),
        _check(
            "planilha vazia não aprova",
            _blocked_by(status_of(untouched.iloc[0:0]), "ausente"),
        ),
    ]


def run_provenance_checks() -> list[bool]:
    checks = [
        *check_boundaries(),
        *check_selection(),
        check_anomaly_reasons(),
        check_locate_frames(),
        check_review_states(),
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        checks.append(check_ignored_output(tmp))
        checks.extend(check_plan_and_render(tmp))
    return checks
