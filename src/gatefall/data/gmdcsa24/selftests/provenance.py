"""Selftest sintético da revisão de proveniência (`provenance.py`, `review_render.py`).

Monta vídeos sintéticos num diretório temporário, um deles com taxa variável
(salto de 0,4 s no PTS a partir do quadro 15), e segmentos OmniFall em
memória. Não toca no dataset real nem no Hugging Face. Requer FFmpeg e Git.
"""

import contextlib
import io
import math
import subprocess
import tempfile
from pathlib import Path

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
    anomaly_reasons,
    build_review_plan,
    check_auto_evidence,
    contact_sheet_path,
    ensure_ignored_output,
    evaluate_case,
    locate_boundary_frames,
    select_videos,
    summarize_review,
    video_boundaries,
    write_review_sheet,
)
from gatefall.data.gmdcsa24.review_render import decode_scaled_frames, render_review
from gatefall.data.gmdcsa24.source import build_inventory, sha256_of

_FRAMES = 30
_JUMP_FRAME = 15
_JUMP_S = 0.4


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
    first = select_videos(videos, boundaries, anomalies)
    shuffled = select_videos(videos.sample(frac=1.0, random_state=7), boundaries, anomalies)
    stratified = [video for video in first if not video.all_boundaries]
    per_stratum = pd.Series(
        [video.video_id.rsplit("/", 1)[0] for video in stratified]
    ).value_counts()
    return [
        _check("seleção determinística e independente da ordem das linhas", first == shuffled),
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
        "reviewer": "" if outcome == PENDING else "revisor",
        "reviewed_at": "" if outcome == PENDING else "2026-10-19",
        "notes": notes,
    }


def check_review_states() -> list[bool]:
    states = {
        "pendente": evaluate_case(_human(PENDING))[0],
        "confirmado na tolerância": evaluate_case(_human(CONFIRMED, "1.08"))[0],
        "confirmado fora da tolerância": evaluate_case(_human(CONFIRMED, "1.15"))[0],
        "confirmado sem instante": evaluate_case(_human(CONFIRMED))[0],
        "confirmado sem revisor": evaluate_case({**_human(CONFIRMED, "1.0"), "reviewer": ""})[0],
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
        "divergência": "blocked",
        "desvio CFR confirmado": "blocked",
        "desvio CFR excluído": "excluded",
        "exclusão sem nota": "blocked",
        "outcome inválido": "blocked",
    }

    def sheet(*rows: dict[str, str]) -> pd.DataFrame:
        return pd.DataFrame([{"case_id": f"c{i}", **row} for i, row in enumerate(rows)])

    pending = summarize_review(sheet(_human(PENDING), _human(CONFIRMED, "1.0")), {})
    approved = summarize_review(
        sheet(_human(CONFIRMED, "0.95"), _human(EXCLUDED, notes="PTS", flag=CFR_FLAG)), {}
    )
    evidence_blocked = summarize_review(sheet(_human(CONFIRMED, "1.0")), {"c0": "hash"})
    return [
        _check(
            "estados: só revisão humana na tolerância aprova; divergência e desvio CFR bloqueiam",
            states == expected,
        ),
        _check(
            "caso pendente impede aprovação sem bloquear",
            not pending.approved and len(pending.pending) == 1 and not pending.blocked,
        ),
        _check(
            "confirmados e exclusões justificadas aprovam; evidência divergente bloqueia",
            approved.approved and not evidence_blocked.approved and len(evidence_blocked.blocked) == 1,
        ),
        _check("planilha vazia não aprova", not summarize_review(sheet(), {}).approved),
    ]


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


def check_plan_and_render(tmp: Path) -> list[bool]:
    root = tmp / "GMDCSA24"
    _encode(root / "Subject 1/Fall/01.mp4", variable_rate=True)
    _encode(root / "Subject 1/ADL/01.mp4", variable_rate=False)
    videos = build_inventory(root).videos
    probes = probe_checkout(root, videos)
    omnifall = _segments(
        [
            ("Subject_1/ADL/01", "walk", 0.0, 1.0),
            ("Subject_1/ADL/01", "sitting", 1.0, 3.0),
            ("Subject_1/Fall/01", "walk", 0.0, 1.4),
            ("Subject_1/Fall/01", "fall", 1.4, 2.2),
            ("Subject_1/Fall/01", "fallen", 2.2, 3.0),
        ]
    )
    original = pd.DataFrame(
        {"video_id": ["Subject_1/Fall/01"], "label": ["Falling"], "start": [1.4], "end": [3.0]}
    )
    with contextlib.redirect_stdout(io.StringIO()):
        plan = build_review_plan(root, videos, omnifall, original, probes)
    indexed = plan.set_index("case_id")
    onset = indexed.loc["Subject_1_Fall_01_fall_onset_001400ms"]

    review_dir = tmp / "review"
    write_review_sheet(plan, review_dir, force=False)
    try:
        write_review_sheet(plan, review_dir, force=False)
        overwrite_refused = False
    except FileExistsError:
        overwrite_refused = True
    sheet = pd.read_csv(review_dir / "review.csv", dtype=str, keep_default_na=False)
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

    digests = {
        identifier: sha256_of(root / path)
        for identifier, path in zip(videos["video_id"], videos["relative_path"])
    }
    tampered = sheet.copy()
    tampered.loc[0, "label_after"] = "standing"
    return [
        _check(
            "planilha: Fall com início e fim de `fall`, ADL com transição, tudo `pending`",
            plan.columns.tolist() == REVIEW_COLUMNS
            and sorted(plan["boundary"]) == [FALL_END, FALL_ONSET, TRANSITION]
            and set(plan["review_outcome"]) == {PENDING}
            and (plan[HUMAN_COLUMNS[1:]] == "").all().all(),
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
            and set(rendered) == set(plan["case_id"])
            and image.width > 0,
        ),
        _check(
            "evidência automática confere com OmniFall e hash; adulteração é detectada",
            check_auto_evidence(sheet, omnifall, digests) == {}
            and set(check_auto_evidence(tampered, omnifall, digests)) == {sheet.loc[0, "case_id"]}
            and set(check_auto_evidence(sheet, omnifall, {}))
            == set(sheet["case_id"]),
        ),
    ]


def run_provenance_checks() -> list[bool]:
    checks = [
        *check_boundaries(),
        *check_selection(),
        check_anomaly_reasons(),
        check_locate_frames(),
        *check_review_states(),
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        checks.append(check_ignored_output(tmp))
        checks.extend(check_plan_and_render(tmp))
    return checks
