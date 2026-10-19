"""Selftest sintético do adapter de rótulos fracos (`weak_label.py`, `decoding.py`).

Monta um clone temporário com seis vídeos de cinza crescente (o nível de cinza
identifica o quadro de origem) e segmentos OmniFall em memória. Um vídeo tem
taxa variável e outro tem lacuna longa, e os dois saem pela política de
exclusão. Os agregados esperados foram contados à mão a partir dos segmentos
abaixo. Não toca no dataset real, no Hugging Face nem no Le2i. Requer FFmpeg.
"""

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.data.frames import apply_frame_schema
from gatefall.data.gmdcsa24.decoding import count_decoded_frames, decode_frames
from gatefall.data.gmdcsa24.source import build_inventory
from gatefall.data.gmdcsa24.weak_label import (
    CANDIDATES_FILE,
    FRAMES_FILE,
    LE2I_CS_TRAIN_WINDOWS,
    MANIFEST_FILE,
    TABLE_FILES,
    VIDEOS_FILE,
    WINDOW_FILES,
    AdapterTables,
    Le2iReference,
    build_tables,
    compare_expected,
    le2i_reference,
    near_boundary,
    output_location_problems,
    probe_sources,
    read_table,
    v2_problems,
    v5_problems,
    validate_outputs,
    video_frames,
    write_outputs,
    write_parquet,
)
from gatefall.data.video_io import probe_frame_count
from gatefall.data.video_metadata import count_video_frames
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.runs import REPOSITORY_ROOT

# Ids inteiros do OmniFall.
WALK, FALL, FALLEN, SITTING, LIE_DOWN, LYING, STANDING, OTHER = (
    LE2I_LABEL_NAMES.index(name)
    for name in ("walk", "fall", "fallen", "sitting", "lie_down", "lying", "standing", "other")
)

# video_id: (fps, quadros, passo de luma por quadro, taxa variável).
_VIDEOS = {
    "Subject_1/Fall/01": (15, 72, 3, False),
    "Subject_1/ADL/01": (10, 40, 5, False),
    "Subject_2/Fall/01": (10, 30, 7, True),
    "Subject_2/ADL/01": (10, 40, 5, False),
    "Subject_3/Fall/01": (10, 50, 4, False),
    "Subject_3/ADL/01": (10, 40, 5, False),
}
_VFR_VIDEO = "Subject_2/Fall/01"
_JUMP_FRAME = 15
_JUMP_S = 0.4

_SEGMENTS = [
    # Grade de 48 quadros a 15 fps; fronteiras em 2,8 e 3,5 s. k_end 40 fica a
    # exatamente 0,5 s da fronteira e sai de S.
    ("Subject_1/Fall/01", WALK, 0.0, 2.8),
    ("Subject_1/Fall/01", FALL, 2.8, 3.5),
    ("Subject_1/Fall/01", FALLEN, 3.5, 4.8),
    # `lie_down` só em janelas com replicação de borda; lacuna curta de 0,25 s
    # deixa k_end 28 como IGNORE_LABEL; `lying` sai de T.
    ("Subject_1/ADL/01", LIE_DOWN, 0.0, 0.8),
    ("Subject_1/ADL/01", SITTING, 0.8, 2.6),
    ("Subject_1/ADL/01", STANDING, 2.6, 2.75),
    ("Subject_1/ADL/01", LYING, 3.0, 4.0),
    # Taxa variável: desvio de PTS de 0,2 s no quadro 15, vídeo excluído.
    ("Subject_2/Fall/01", WALK, 0.0, 1.0),
    ("Subject_2/Fall/01", FALL, 1.0, 2.0),
    ("Subject_2/Fall/01", FALLEN, 2.0, 3.4),
    # Sem fronteira: S mantém todas as janelas de T.
    ("Subject_2/ADL/01", OTHER, 0.0, 4.0),
    ("Subject_3/Fall/01", WALK, 0.0, 2.0),
    ("Subject_3/Fall/01", FALL, 2.0, 4.0),
    ("Subject_3/Fall/01", FALLEN, 4.0, 5.0),
    # Lacuna de 0,6 s: vídeo excluído.
    ("Subject_3/ADL/01", WALK, 0.0, 1.0),
    ("Subject_3/ADL/01", SITTING, 1.6, 4.0),
]
_ORIGINAL_FALLS = [
    ("Subject_1/Fall/01", 2.7, 3.5),
    ("Subject_2/Fall/01", 1.0, 2.0),
    ("Subject_3/Fall/01", 2.0, 4.0),
]

_EXPECTED_EXCLUSIONS = {
    # max |PTS − i/fps| com fps médio 30/3,4 s: no quadro 15, 1,9 − 1,7 s.
    "Subject_2/Fall/01": ["desvio de PTS de 0.200 s"],
    "Subject_3/ADL/01": ["lacuna de 0.60 s"],
}


def _classes(**counts: int) -> dict[str, int]:
    return {name: counts.get(name, 0) for name in LE2I_LABEL_NAMES}


# Janelas em stride 4 por vídeo, na ordem de _VIDEOS: rotuladas 12, 9, 9, 10,
# 13, 9; com replicação de borda 6, 6, 6, 6, 6, 5; T 6, 1, 0, 4, 7, 0; S 1, 0,
# 0, 4, 3, 0.
_EXPECTED_AGGREGATES: dict[str, object] = {
    "excluded_videos": sorted(_EXPECTED_EXCLUSIONS),
    "funnel": {
        "labeled": 62,
        "edge_padded": 35,
        "dropped_class": 2,
        "excluded_video": 7,
        "T": 18,
        "near_boundary": 10,
        "S": 8,
    },
    "class_counts": {
        "T": _classes(walk=1, fall=6, fallen=6, sitting=1, other=4),
        "S": _classes(fall=2, fallen=2, other=4),
    },
    "videos": {
        "source": 6,
        "excluded": 2,
        "included": 4,
        "with_window_T": 4,
        "with_fall_T": 2,
        "with_fall_S": 1,
    },
    "subjects": {
        "T": {"S1": 7, "S2": 4, "S3": 7},
        "T_fall": {"S1": 2, "S2": 0, "S3": 4},
    },
}


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _video_path(root: Path, identifier: str) -> Path:
    subject, category, stem = identifier.split("/")
    return root / subject.replace("_", " ") / category / f"{stem}.mp4"


def _encode(path: Path, fps: int, n_frames: int, step: int, variable_rate: bool) -> None:
    """Quadro N com luma 16 + N·passo; com taxa variável, PTS N/10 s mais 0,4 s a partir do 15."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timing = (
        [
            "-vf",
            f"settb=1/100,setpts='(N/10+gte(N\\,{_JUMP_FRAME})*{_JUMP_S})/TB'",
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "1/100",
        ]
        if variable_rate
        else []
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=64x48:r={fps}:d={n_frames / fps},format=yuv420p,"
            f"geq=lum='16+N*{step}':cb=128:cr=128",
            *timing,
            "-frames:v",
            str(n_frames),
            "-c:v",
            "mpeg4",
            "-q:v",
            "1",
            str(path),
        ],
        check=True,
    )


def _source_index(frame: np.ndarray, step: int) -> int:
    """Quadro de origem pelo cinza médio: luma 16 + N·passo vira RGB N·passo·255/219."""
    return int(round(float(frame.mean()) * 219 / (255 * step)))


def _omnifall() -> pd.DataFrame:
    frame = pd.DataFrame(_SEGMENTS, columns=["video_id", "label", "start", "end"])
    subject = frame["video_id"].str.extract(r"^Subject_(\d+)/")[0].astype(int)
    return frame.assign(subject=subject, cam=1, dataset="GMDCSA24")


def _original_segments() -> pd.DataFrame:
    frame = pd.DataFrame(_ORIGINAL_FALLS, columns=["video_id", "start", "end"])
    columns = ["video_id", "label", "direction", "start", "end"]
    return cast(pd.DataFrame, frame.assign(label="Falling", direction="FW")[columns])


def _le2i_reference(recorded: dict[str, str]) -> Le2iReference:
    return Le2iReference(
        eval_video_ids=frozenset({"Coffee_room_01/Videos/video (1)"}),
        source_sha256=frozenset({"0" * 64}),
        train_windows=LE2I_CS_TRAIN_WINDOWS,
        eval_index_sha256=recorded,
    )


_EVAL_HASHES = {"val": "a" * 64, "test": "b" * 64}


def _provenance(review_after: str | None = None) -> dict[str, object]:
    return {
        "le2i_cs": {"eval_index_sha256": _EVAL_HASHES},
        "review_sheet": {"path": "review.csv", "sha256_before": None, "sha256_after": review_after},
    }


def _validate(output_dir: Path, le2i: Le2iReference | None) -> dict[str, list[str]]:
    return validate_outputs(output_dir, le2i, _EXPECTED_EXCLUSIONS, _EXPECTED_AGGREGATES)


def check_near_boundary() -> bool:
    k_end = np.array([15, 18, 19])
    return _check(
        "fronteira: 0,5 s exato sai, inclusive com erro de ponto flutuante; 0,6 s fica; "
        "vídeo sem fronteira mantém tudo",
        near_boundary(k_end, [1.0, 1.3]).tolist() == [True, True, False]
        and near_boundary(k_end, []).tolist() == [False, False, False],
    )


def check_temporal_alignment(root: Path) -> list[bool]:
    segments = _omnifall()
    v1 = cast(pd.DataFrame, segments[segments["video_id"] == "Subject_1/Fall/01"])
    frames = video_frames("Subject_1/Fall/01", 1, 72, 15.0, v1)
    labels = frames["label"].to_numpy()
    src = frames["src_index"].to_numpy()
    path = _video_path(root, "Subject_1/Fall/01")
    wanted = [int(src[k]) for k in (0, 1, 3, 28, 47)]
    decoded = [_source_index(frame, 3) for frame in decode_frames(path, wanted, 64, 48)]
    return [
        _check(
            "grade de 10 fps sobre 72 quadros a 15 fps: K = 48, time_s = k/10, "
            "src_index = round(1,5·k) com arredondamento par",
            len(frames) == 48
            and np.allclose(frames["time_s"].to_numpy(), np.arange(48) / 10)
            and [int(src[k]) for k in (1, 3, 47)] == [2, 4, 70],
        ),
        _check(
            "rótulo no instante k/10 em [start, end): `fall` de k = 28 a 34, `fallen` em 35",
            labels[27] == WALK and labels[28] == FALL and labels[34] == FALL and labels[35] == FALLEN,
        ),
        _check(
            "o quadro decodificado em src_index é o quadro de origem de mesmo índice",
            decoded == wanted,
        ),
        _check(
            "esquema de quadros do Le2i, split train e env GMDCSA24",
            set(frames["split"]) == {"train"} and set(frames["env"]) == {"GMDCSA24"},
        ),
    ]


def check_duplicate_frames(root: Path) -> list[bool]:
    path = _video_path(root, _VFR_VIDEO)
    _, n_frames, step, _ = _VIDEOS[_VFR_VIDEO]
    decoded = decode_frames(path, list(range(n_frames)), 64, 48)
    return [
        _check(
            "taxa variável: o pipe sem passthrough duplica quadros; o do adapter não",
            probe_frame_count(path) > n_frames
            and count_decoded_frames(path, 64, 48) == count_video_frames(path) == n_frames,
        ),
        _check(
            "taxa variável: cada quadro decodificado é o quadro de origem de mesmo índice",
            [_source_index(frame, step) for frame in decoded] == list(range(n_frames)),
        ),
    ]


def check_tables(tables: AdapterTables) -> list[bool]:
    videos = tables.videos.set_index("video_id")
    candidates = tables.candidates
    v2_windows = cast(pd.DataFrame, candidates[candidates["video_id"] == "Subject_1/ADL/01"])
    excluded_windows = cast(pd.DataFrame, candidates[candidates["excluded_video"]])
    return [
        _check(
            "exclusão recalculada pela política: taxa variável e lacuna longa, com motivo",
            {
                str(key): str(value).split("; ")
                for key, value in videos.loc[videos["excluded"], "exclusion_reasons"].items()
            }
            == _EXPECTED_EXCLUSIONS,
        ),
        _check(
            "vídeo excluído sai inteiro: sem quadros gravados e sem janela em T ou S",
            set(tables.frames["video_id"]).isdisjoint(_EXPECTED_EXCLUSIONS)
            and not bool(excluded_windows[["in_T", "in_S"]].to_numpy().any())
            and len(tables.frames) == 48 + 40 + 40 + 50,
        ),
        _check(
            "IGNORE_LABEL não vira candidata; `lie_down` com replicação de borda conta como borda",
            28 not in v2_windows["k_end"].tolist()
            and bool((v2_windows["label"] != IGNORE_LABEL).all())
            and v2_windows.loc[v2_windows["label"] == LIE_DOWN, "edge_padded"].tolist() == [True, True],
        ),
        _check(
            "agregados contados à mão: funil, classes, vídeos e sujeitos",
            compare_expected(_EXPECTED_AGGREGATES, tables.summary) == [],
        ),
        _check(
            "V5: decodificador do adapter e ffprobe -count_frames contam o mesmo em todo vídeo",
            v5_problems(tables.videos) == [],
        ),
    ]


def check_le2i_reference() -> list[bool]:
    rows = []
    for identifier, split in (("train_video", "train"), ("val_video", "val"), ("test_video", "test")):
        for index in range(30):
            rows.append(
                {
                    "video_id": identifier,
                    "split": split,
                    "env": "Office",
                    "subject": 1,
                    "frame_index": index,
                    "time_s": index / 10,
                    "src_index": index,
                    "label": WALK,
                    "gap_position": None,
                }
            )
    frames = apply_frame_schema(pd.DataFrame(rows))
    manifest = pd.DataFrame({"sha256": ["1" * 64, "2" * 64, "3" * 64]})
    reference = le2i_reference(frames, manifest)
    changed_train = frames.copy()
    changed_train.loc[changed_train["split"] == "train", "label"] = FALL
    changed_val = frames.copy()
    changed_val.loc[changed_val["split"] == "val", "label"] = FALL
    return [
        _check(
            "referência Le2i: val/test, SHA-256 da fonte e janelas de treino em stride 4",
            reference.eval_video_ids == {"val_video", "test_video"}
            and reference.source_sha256 == {"1" * 64, "2" * 64, "3" * 64}
            and reference.train_windows == 8,
        ),
        _check(
            "hash do índice de val/test muda só com val/test, nunca com o treino",
            le2i_reference(changed_train, manifest).eval_index_sha256 == reference.eval_index_sha256
            and le2i_reference(changed_val, manifest).eval_index_sha256["val"]
            != reference.eval_index_sha256["val"]
            and le2i_reference(changed_val, manifest).eval_index_sha256["test"]
            == reference.eval_index_sha256["test"],
        ),
    ]


def check_output_location(tmp: Path) -> bool:
    return _check(
        "saída recusada dentro dos artefatos Le2i, acima deles ou num caminho versionado",
        output_location_problems(Path("data/processed/le2i/gmdcsa24")) != []
        and output_location_problems(Path("data/processed")) != []
        and output_location_problems(REPOSITORY_ROOT / "src/gatefall/gmdcsa24_out") != []
        and output_location_problems(tmp / "gmdcsa24") == [],
    )


def check_validation(tables: AdapterTables, tmp: Path) -> list[bool]:
    output_dir = tmp / "gmdcsa24"
    le2i = _le2i_reference(_EVAL_HASHES)
    write_outputs(output_dir, tables, _provenance(), force=False)
    clean = _validate(output_dir, le2i)
    try:
        write_outputs(output_dir, tables, _provenance(), force=False)
        refused = False
    except FileExistsError:
        refused = True

    off_by_one = json.loads(json.dumps(_EXPECTED_AGGREGATES))
    off_by_one["funnel"]["T"] += 1
    extra = json.loads(json.dumps(_EXPECTED_AGGREGATES))
    extra["videos"]["with_fall_S_extra"] = 0
    w2_off = validate_outputs(output_dir, le2i, _EXPECTED_EXCLUSIONS, off_by_one)["W2"]
    w2_extra = validate_outputs(output_dir, le2i, _EXPECTED_EXCLUSIONS, extra)["W2"]
    missing_exclusion = {"Subject_2/Fall/01": _EXPECTED_EXCLUSIONS["Subject_2/Fall/01"]}
    other_reason = {**_EXPECTED_EXCLUSIONS, "Subject_3/ADL/01": ["lacuna de 0.61 s"]}
    w1_missing = validate_outputs(output_dir, le2i, missing_exclusion, _EXPECTED_AGGREGATES)["W1"]
    w1_reason = validate_outputs(output_dir, le2i, other_reason, _EXPECTED_AGGREGATES)["W1"]

    checks = [
        _check(
            "saída gravada valida: integridade, proveniência, W1, W2, V2 e V5 sem problema",
            all(not problems for problems in clean.values())
            and set(clean) == {"integridade", "proveniência", "W1", "W2", "V2", "V5"},
        ),
        _check("saída existente não é sobrescrita sem --force", refused),
        _check(
            "W2 exato: um a mais no funil reprova com o valor esperado e o obtido",
            w2_off == ["agregados.funnel.T: esperado 19, obtido 18"],
        ),
        _check(
            "W2 exato: agregado esperado ausente da saída reprova",
            w2_extra == ["agregados.videos.with_fall_S_extra: ausente na saída (esperado 0)"],
        ),
        _check(
            "W1: exclusão a mais ou com outro motivo reprova",
            len(w1_missing) == 1 and len(w1_reason) == 1,
        ),
    ]

    tampered_dir = tmp / "tampered"
    write_outputs(tampered_dir, tables, _provenance(), force=False)
    windows_path = tampered_dir / WINDOW_FILES["T"]
    write_parquet(read_table(tampered_dir, WINDOW_FILES["T"]).iloc[1:], windows_path)
    dropped_row = _validate(tampered_dir, le2i)["integridade"]
    write_outputs(tampered_dir, tables, _provenance(), force=True)
    manifest_path = tampered_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary"]["funnel"]["S"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    edited_summary = _validate(tampered_dir, le2i)["integridade"]
    manifest_path.unlink()
    no_manifest = _validate(tampered_dir, le2i)
    review_dir = tmp / "review_changed"
    write_outputs(review_dir, tables, _provenance(review_after="c" * 64), force=False)
    review_changed = _validate(review_dir, le2i)["W1"]
    checks.extend(
        [
            _check(
                "integridade: janela removida de windows_t.parquet reprova",
                any(WINDOW_FILES["T"] in problem for problem in dropped_row),
            ),
            _check(
                "integridade: resumo editado no manifesto reprova",
                any(problem.startswith("summary.funnel.S") for problem in edited_summary),
            ),
            _check(
                "integridade: saída sem manifesto não valida",
                list(no_manifest) == ["integridade"] and len(no_manifest["integridade"]) == 1,
            ),
            _check(
                "W1: review.csv alterado durante a execução reprova",
                review_changed == ["review.csv mudou durante a execução do adapter"],
            ),
        ]
    )
    checks.extend(check_separation(output_dir, le2i))
    checks.extend(check_provenance_tampering(output_dir, le2i, tmp))
    return checks


def check_separation(output_dir: Path, le2i: Le2iReference) -> list[bool]:
    manifest = json.loads((output_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    tables = {name: read_table(output_dir, name) for name in TABLE_FILES}
    leaked = {**tables, FRAMES_FILE: tables[FRAMES_FILE].assign(split="val")}
    candidates = tables[CANDIDATES_FILE]
    excluded_window = cast(pd.DataFrame, candidates[candidates["excluded_video"]]).iloc[[0]]
    excluded_in_t = {
        **tables,
        WINDOW_FILES["T"]: pd.concat(
            [tables[WINDOW_FILES["T"]], excluded_window[tables[WINDOW_FILES["T"]].columns]]
        ),
    }
    gmdcsa_video = str(tables[VIDEOS_FILE]["video_id"].iloc[0])
    gmdcsa_sha = str(tables[VIDEOS_FILE]["source_sha256"].iloc[0])
    variants = {
        "video_id do GMDCSA24 em val/test": replace(
            le2i, eval_video_ids=le2i.eval_video_ids | {gmdcsa_video}
        ),
        "mesmo SHA-256 de um vídeo Le2i": replace(
            le2i, source_sha256=le2i.source_sha256 | {gmdcsa_sha}
        ),
        "treino Le2i-CS diferente do publicado": replace(
            le2i, train_windows=LE2I_CS_TRAIN_WINDOWS - 1
        ),
        "índice de val do Le2i-CS mudou desde o build": replace(
            le2i, eval_index_sha256={**_EVAL_HASHES, "val": "d" * 64}
        ),
    }
    checks = [
        _check(
            f"V2 reprova com {name}",
            len(v2_problems(output_dir, manifest, tables, variant)) == 1,
        )
        for name, variant in variants.items()
    ]
    checks.extend(
        [
            _check(
                "V2 reprova quadro do GMDCSA24 fora do split train",
                len(v2_problems(output_dir, manifest, leaked, le2i)) == 1,
            ),
            _check(
                "V2 reprova janela de vídeo excluído em T",
                len(v2_problems(output_dir, manifest, excluded_in_t, le2i)) == 1,
            ),
            _check(
                "V2 reprova sem a referência Le2i-CS",
                len(v2_problems(output_dir, manifest, tables, None)) == 1,
            ),
        ]
    )
    return checks


def _set_first_exclusion(field: str, value: object) -> Callable[[dict[str, object]], None]:
    def mutate(manifest: dict[str, object]) -> None:
        cast(list[dict[str, object]], manifest["excluded_videos"])[0][field] = value

    return mutate


def _set_field(section: str, field: str, value: object) -> Callable[[dict[str, object]], None]:
    def mutate(manifest: dict[str, object]) -> None:
        cast(dict[str, object], manifest[section])[field] = value

    return mutate


def check_provenance_tampering(output_dir: Path, le2i: Le2iReference, tmp: Path) -> list[bool]:
    """Cada adulteração, isolada numa cópia da saída, reprova só a proveniência;
    o validate não reescreve o manifesto e a saída original continua válida."""
    mutations = {
        "margem de fronteira": _set_field("parameters", "boundary_margin_s", 0.6),
        "commit do clone": _set_field("source", "commit", "0" * 40),
        "revisão do OmniFall": _set_field("labels", "revision", "0" * 40),
        "identidade do protocolo": _set_field("protocol", "sha256", "0" * 64),
        "motivo de exclusão": _set_first_exclusion("reasons", ["lacuna de 9.99 s"]),
        "SHA-256 da fonte excluída": _set_first_exclusion("source_sha256", "f" * 64),
    }
    checks: list[bool] = []
    for position, (name, mutate) in enumerate(mutations.items()):
        tampered_dir = tmp / f"provenance_{position}"
        shutil.copytree(output_dir, tampered_dir)
        manifest_path = tampered_dir / MANIFEST_FILE
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutate(manifest)
        tampered_text = json.dumps(manifest, indent=2, ensure_ascii=False)
        manifest_path.write_text(tampered_text, encoding="utf-8")
        results = _validate(tampered_dir, le2i)
        failed = {condition for condition, problems in results.items() if problems}
        checks.append(
            _check(
                f"proveniência: {name} adulterado reprova, sem reescrever o manifesto",
                failed == {"proveniência"}
                and len(results["proveniência"]) == 1
                and manifest_path.read_text(encoding="utf-8") == tampered_text,
            )
        )
    untouched = _validate(output_dir, le2i)
    checks.append(
        _check(
            "proveniência: a saída não adulterada continua válida",
            all(not problems for problems in untouched.values()),
        )
    )
    return checks


def run_adapter_selftest() -> None:
    checks = [check_near_boundary(), *check_le2i_reference()]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        root = tmp / "GMDCSA24"
        for identifier, (fps, n_frames, step, variable_rate) in _VIDEOS.items():
            _encode(_video_path(root, identifier), fps, n_frames, step, variable_rate)
        checks.extend(check_temporal_alignment(root))
        checks.extend(check_duplicate_frames(root))
        with contextlib.redirect_stdout(io.StringIO()):
            sources = probe_sources(root, build_inventory(root).videos)
        tables = build_tables(sources, _omnifall(), _original_segments())
        checks.extend(check_tables(tables))
        checks.append(check_output_location(tmp))
        checks.extend(check_validation(tables, tmp))
    if not all(checks):
        print("\nselftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nselftest OK: todos os casos passaram")
