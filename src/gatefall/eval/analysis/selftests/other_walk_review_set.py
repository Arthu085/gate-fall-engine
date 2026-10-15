import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from gatefall.data.video_io import decode_frames
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis import other_walk_review_set
from gatefall.eval.analysis.multiseed_summary import ARMS
from gatefall.eval.analysis.qualitative import _with_caption_strip
from gatefall.eval.analysis.other_walk_review_set import (
    CONTROL,
    ERROR,
    FRAMES_CSV_FILE,
    KEY_CSV_FILE,
    KEY_DIR,
    SELECTION_JSON_FILE,
    WORKSHEET_CSV_FILE,
    WORKSHEET_DIR,
    ReviewSources,
    plan_review_set,
    records,
    run_review_set,
    tile_lines,
    tile_origin,
    window_mapping,
)
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.eval.analysis.selftests.stratified_error_diagnosis import _raises, _tree_hashes
from gatefall.eval.analysis.window_context_diagnosis import WINDOW_COLUMNS, label_name, video_context
from gatefall.hashing import sha256_file
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT

OTHER, WALK = LE2I_LABEL_NAMES.index("other"), LE2I_LABEL_NAMES.index("walk")
N_SEEDS = 5
SEEDS = tuple(range(42, 42 + N_SEEDS))
FRAME_SIZE = (32, 24)

# (video_id, split, env, subject, rótulo por quadro).
# EnvL/v1: trecho A `other` [0, 49] (início do vídeo), trecho B `other`
# [70, 109] depois de `walk`. EnvC/v3: trechos `other` [0, 39] e [60, 99].
VIDEO_SPEC: tuple[tuple[str, str, str, int, list[int]], ...] = (
    ("EnvL/v1", "test", "Lecture_room", 1, [OTHER] * 50 + [WALK] * 20 + [OTHER] * 40 + [WALK] * 10),
    ("EnvC/v3", "test", "Coffee_room_02", 3, [OTHER] * 40 + [WALK] * 20 + [OTHER] * 40),
    ("EnvO/v4", "test", "Office", 4, [WALK] * 10 + [OTHER] * 40),
    ("EnvV/v5", "val", "Lecture_room", 5, [WALK] * 10 + [OTHER] * 40),
)

# (video_id, k_end) → (rótulo, votos por braço entre 5 seeds, votos do braço A).
# Janelas `other` fora daqui recebem 3 votos `other` e 2 `walk` por braço,
# sem persistência.
PREDICTION_SPEC: dict[tuple[str, int], tuple[str, int, int]] = {
    # Trecho A: erro persistente e puro em 30..40; em 41..45 o braço A só
    # tem 3 de 5 seeds com `walk`, abaixo do limiar.
    **{("EnvL/v1", k): ("walk", 5, 5) for k in range(30, 41)},
    **{("EnvL/v1", k): ("walk", 5, 3) for k in range(41, 46)},
    # Trecho B: erro persistente misturado em 70..72 e controle puro em
    # 95..109; 95 fica a 23 quadros de 72 e é preterido por sobreposição.
    **{("EnvL/v1", k): ("walk", 5, 5) for k in range(70, 73)},
    **{("EnvL/v1", k): ("other", 5, 5) for k in range(95, 110)},
    # EnvC/v3: 4 de 5 seeds em todo braço atingem o limiar; em 31 o braço A
    # cai para 3 de 5.
    **{("EnvC/v3", k): ("walk", 4, 4) for k in range(25, 31)},
    ("EnvC/v3", 31): ("walk", 4, 3),
    **{("EnvC/v3", k): ("other", 5, 5) for k in range(85, 100)},
    # Office não é prioritário: nada é selecionado, mas a cobertura conta.
    **{("EnvO/v4", k): ("walk", 5, 5) for k in range(40, 45)},
    **{("EnvO/v4", k): ("other", 5, 5) for k in range(45, 50)},
    # val: erro persistente contabilizado e numerado à parte de test.
    **{("EnvV/v5", k): ("walk", 5, 5) for k in range(40, 44)},
}
EXPECTED_SELECTION = {
    ("test", ERROR, "EnvL/v1", 30),
    ("test", ERROR, "EnvL/v1", 72),
    ("test", CONTROL, "EnvL/v1", 96),
    ("test", ERROR, "EnvC/v3", 25),
    ("test", CONTROL, "EnvC/v3", 85),
    ("val", ERROR, "EnvV/v5", 40),
}


def _src_index(frame_index: int) -> int:
    return (5 * frame_index) // 2


def _frames(spec: Sequence[tuple[str, str, str, int, list[int]]] = VIDEO_SPEC) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": video_id, "split": split, "env": env, "subject": subject, "frame_index": k,
             "time_s": k / 10, "src_index": _src_index(k), "label": label, "gap_position": None}
            for video_id, split, env, subject, labels in spec
            for k, label in enumerate(labels)
        ]
    )


def _prediction(video_id: str, k_end: int, true_label: str, arm: str, seed_index: int,
                spec: dict[tuple[str, int], tuple[str, int, int]]) -> str:
    if true_label != "other":
        return true_label
    label, votes, arm_a_votes = spec.get((video_id, k_end), ("other", 3, 3))
    n_votes = arm_a_votes if arm == "A" else votes
    if seed_index < n_votes:
        return label
    return "walk" if label == "other" else "sitting"


def _windows(spec: Sequence[tuple[str, str, str, int, list[int]]] = VIDEO_SPEC,
             predictions: dict[tuple[str, int], tuple[str, int, int]] = PREDICTION_SPEC) -> pd.DataFrame:
    tables = []
    for video_id, split, env, subject, labels in spec:
        context = video_context(np.asarray(labels, dtype=np.int64), np.ones(len(labels), dtype=bool))
        context = context.assign(video_id=video_id, split=split, env=env, subject=subject,
                                 true_label=[label_name(labels[k]) for k in context["k_end"]])
        tables.append(context)
    windows = cast(pd.DataFrame, pd.concat(tables, ignore_index=True)[WINDOW_COLUMNS])
    for arm in ARMS:
        for seed_index, seed in enumerate(SEEDS):
            windows[f"pred_{arm}_seed{seed}"] = [
                _prediction(str(video_id), int(k_end), str(true_label), arm, seed_index, predictions)
                for video_id, k_end, true_label in zip(windows["video_id"], windows["k_end"], windows["true_label"])
            ]
    return windows


def _color(src_index: int) -> tuple[int, int, int]:
    return (src_index % 256, (37 * src_index) % 256, (src_index // 256) * 40 % 256)


def _frame(src_index: int) -> np.ndarray:
    width, height = FRAME_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = _color(src_index)
    return frame


def _fake_decode(_path: Path, src_indices: Sequence[int]) -> list[np.ndarray]:
    return [_frame(index) for index in src_indices]


def _write_video(path: Path, n_src_frames: int) -> None:
    width, height = FRAME_SIZE
    raw = b"".join(_frame(index).tobytes() for index in range(n_src_frames))
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", "25", "-i", "-", "-c:v", "rawvideo", "-pix_fmt", "bgr24", str(path)],
        input=raw,
        check=True,
    )


def _sources(root: Path, frames: pd.DataFrame, windows: pd.DataFrame, write_videos: bool = True) -> ReviewSources:
    frames_path = root / "data" / "frames.parquet"
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(frames_path)
    windows_csv = root / "stage3a" / "window_context_windows.csv"
    windows_csv.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(windows_csv, index=False)
    raw_dir = root / "raw"
    raw_dir.mkdir(exist_ok=True)
    video_paths = {str(video_id): raw_dir / f"{str(video_id).replace('/', '__')}.avi"
                   for video_id in sorted(set(frames["video_id"]))}
    if write_videos:
        for path in video_paths.values():
            path.write_bytes(b"video")
    return ReviewSources(
        windows_csv=windows_csv,
        frames_path=frames_path,
        video_paths=video_paths,
        video_sha256={video_id: sha256_file(path) for video_id, path in video_paths.items() if path.is_file()},
        protected=(raw_dir,),
    )


def _selection(plan) -> set[tuple[str, str, str, int]]:
    return {(item["split"], item["role"], item["video_id"], int(item["k_end"])) for _id, item, _map in plan.reviews}


def _read_outputs(output_dir: Path) -> dict[str, str]:
    return {name: (output_dir / name).read_text(encoding="utf-8") for name in (
        f"{KEY_DIR}/{SELECTION_JSON_FILE}", f"{KEY_DIR}/{KEY_CSV_FILE}",
        f"{WORKSHEET_DIR}/{WORKSHEET_CSV_FILE}", f"{WORKSHEET_DIR}/{FRAMES_CSV_FILE}",
    )}


def _selftest_deterministic_selection() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        windows = _windows()
        sources = _sources(root / "a", _frames(), windows)
        plan = plan_review_set(sources)
        selected = _selection(plan)
        ids = [review_id for review_id, _item, _mapping in plan.reviews]

        shuffled = windows.sample(frac=1.0, random_state=7)
        prediction_columns = [column for column in shuffled.columns if column.startswith("pred_")]
        shuffled = cast(pd.DataFrame, shuffled[[*WINDOW_COLUMNS, *reversed(prediction_columns)]])
        shuffled_sources = _sources(root / "b", _frames().sample(frac=1.0, random_state=3), shuffled)
        out_a, out_b = root / "out_a", root / "out_b"
        with patch.object(other_walk_review_set, "LOCAL_OUTPUT_ROOT", root):
            run_review_set(sources, out_a, decode_fn=_fake_decode)
            run_review_set(shuffled_sources, out_b, decode_fn=_fake_decode)
        first, second = _read_outputs(out_a), _read_outputs(out_b)
        report = json.loads(first.pop(f"{KEY_DIR}/{SELECTION_JSON_FILE}"))
        shuffled_report = json.loads(second.pop(f"{KEY_DIR}/{SELECTION_JSON_FILE}"))
        # Caminhos e bytes das entradas embaralhadas diferem por construção.
        report.pop("input_sha256")
        shuffled_report.pop("input_sha256")
        coverage = report["coverage"]
        worksheet = pd.read_csv(out_a / WORKSHEET_DIR / WORKSHEET_CSV_FILE)
        exclusions = report["exclusions"]

        with patch.object(other_walk_review_set, "MAX_SEGMENTS_PER_GROUP", 1):
            capped = plan_review_set(sources)
        capped_reasons = {(item["video_id"], int(item["segment_start"]), item["role"], item["reason"])
                          for item in capped.exclusions}

    return all([
        _check("seleção: um item por trecho e papel, com o representante esperado", selected == EXPECTED_SELECTION),
        _check("seleção: limiar inclusivo de 4/5 seeds por braço; 3/5 em um braço não conta",
               coverage["test"]["Coffee_room_02"][ERROR]["n_windows"] == 6
               and coverage["test"]["Lecture_room"][ERROR]["n_windows"] == 14),
        _check("seleção: controle sobreposto ao erro do mesmo vídeo cede ao próximo candidato do trecho",
               ("test", CONTROL, "EnvL/v1", 96) in selected and ("test", CONTROL, "EnvL/v1", 95) not in selected),
        _check("seleção: nenhuma janela selecionada do mesmo vídeo a menos de 24 quadros de outra",
               all(abs(a[3] - b[3]) >= 24 for a in selected for b in selected if a != b and a[2] == b[2])),
        _check("seleção: ambiente não prioritário fica fora e entra só na cobertura",
               not any(item[2] == "EnvO/v4" for item in selected)
               and coverage["test"]["Office"][ERROR]["n_segments"] == 1
               and coverage["test"]["Office"][ERROR]["n_selected"] == 0 and not coverage["test"]["Office"]["focus"]),
        _check("seleção: val e test contabilizados e numerados separadamente",
               sorted(ids) == ["test-01", "test-02", "test-03", "test-04", "test-05", "val-01"]
               and coverage["val"]["Lecture_room"][ERROR]["n_selected"] == 1
               and coverage["test"]["Lecture_room"][ERROR]["n_selected"] == 2),
        _check("seleção: ordem de linhas e de colunas da entrada não altera nenhuma saída",
               first == second and report == shuffled_report),
        _check("seleção: sem exclusões com o limite padrão", exclusions == []),
        _check("seleção: limite por (split, ambiente, papel) registra o trecho preterido",
               ("EnvL/v1", 70, ERROR, "cap") in capped_reasons and len(capped.reviews) == 5),
        _check("planilha: sem papel, predições, votos nem frações de concordância",
               not [column for column in worksheet.columns
                        if column in {"role"} or "pred" in column or "votes" in column or "share" in column]),
        _check("planilha: campos de revisão vazios e rótulo verdadeiro registrado",
               bool(worksheet[list(other_walk_review_set.REVIEW_FIELDS)].isna().to_numpy().all())
               and set(worksheet["true_label"]) == {"other"}),
        _check("chave: concordância por braço registrada",
               {item["walk_votes_A"] for item in report["items"] if item["role"] == ERROR} == {5, 4}
               and all(item["n_runs"] == 25 for item in report["items"])),
    ])


def _alignment_spec() -> tuple[tuple[str, str, str, int, list[int]], ...]:
    return (("EnvL/v9", "test", "Lecture_room", 9, [OTHER] * 30 + [WALK] * 10),)


def _selftest_exact_frame_alignment() -> bool:
    spec = _alignment_spec()
    predictions = {("EnvL/v9", 10): ("walk", 5, 5)}
    frames = _frames(spec)
    windows = _windows(spec, predictions)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root, frames, windows, write_videos=False)
        video_path = sources.video_paths["EnvL/v9"]
        _write_video(video_path, n_src_frames=_src_index(len(spec[0][4]) - 1) + 3)
        sources = ReviewSources(
            windows_csv=sources.windows_csv, frames_path=sources.frames_path, video_paths=sources.video_paths,
            video_sha256={"EnvL/v9": sha256_file(video_path)}, protected=sources.protected,
        )
        output_dir = root / "out"
        with patch.object(other_walk_review_set, "LOCAL_OUTPUT_ROOT", root):
            run_review_set(sources, output_dir, decode_fn=decode_frames)
        sheet = np.array(Image.open(output_dir / WORKSHEET_DIR / "sheets" / "test-01.png").convert("RGB"))
        frame_rows = pd.read_csv(output_dir / WORKSHEET_DIR / FRAMES_CSV_FILE)
        worksheet = pd.read_csv(output_dir / WORKSHEET_DIR / WORKSHEET_CSV_FILE)

        video = frames.sort_values("frame_index").reset_index(drop=True)
        mapping = window_mapping(video, 10)
        tiles = [_with_caption_strip(_frame(int(src)), tile_lines(row))
                 for src, row in zip(mapping["src_index"], records(mapping))]
        cell_height = max(tile.shape[0] for tile in tiles)
        cell_width = max(tile.shape[1] for tile in tiles)
        tile_colors = [tuple(int(value) for value in sheet[top + 2, left + 2])
                       for top, left in (tile_origin(index, cell_height, cell_width) for index in range(len(tiles)))]

        tampered = windows.copy()
        tampered.loc[(tampered["video_id"] == "EnvL/v9") & (tampered["k_end"] == 10), "n_final_label_frames"] = 23
        tampered_sources = _sources(root / "tampered", frames, tampered)
        shifted = windows.copy()
        shifted.loc[shifted["video_id"] == "EnvL/v9", "segment_end"] = 30
        shifted_sources = _sources(root / "shifted", frames, shifted)
        relabeled = frames.copy()
        relabeled.loc[relabeled["frame_index"] == 10, "label"] = WALK
        relabeled_sources = _sources(root / "relabeled", relabeled, windows)
        tampered_fails = _raises(ValueError, lambda: plan_review_set(tampered_sources), "rótulo final divergem")
        shifted_fails = _raises(ValueError, lambda: plan_review_set(shifted_sources), "não corresponde")
        relabeled_fails = _raises(ValueError, lambda: plan_review_set(relabeled_sources), "não é other")

    expected_src = [_src_index(max(0, k)) for k in range(10 - 23, 11)]
    return all([
        _check("alinhamento: mapeamento frame_index → src_index com replicação de borda",
               mapping["src_index"].tolist() == expected_src
               and mapping["frame_index"].tolist() == [max(0, k) for k in range(-13, 11)]
               and mapping["replicated"].tolist() == [True] * 13 + [False] * 11),
        _check("alinhamento: cada ladrilho da folha decodificada pelo ffmpeg é o quadro src_index esperado",
               tile_colors == [_color(src) for src in expected_src]),
        _check("alinhamento: review_frames.csv repete o mapeamento e os tempos da grade",
               frame_rows["src_index"].tolist() == expected_src
               and np.allclose(frame_rows["time_s"].to_numpy(), mapping["time_s"].to_numpy())
               and frame_rows["label"].tolist() == ["other"] * 24),
        _check("alinhamento: planilha registra janela, src_index e tempos extremos",
               worksheet.iloc[0][["window_frame_start", "window_frame_end", "src_index_start", "src_index_end"]]
               .tolist() == [0, 10, 0, 25] and float(worksheet.iloc[0]["time_end_s"]) == 1.0),
        _check("alinhamento: contagem do rótulo final divergente do CSV falha",
               tampered_fails),
        _check("alinhamento: trecho do CSV divergente de frames.parquet falha",
               shifted_fails),
        _check("alinhamento: rótulo do quadro final divergente falha",
               relabeled_fails),
    ])


def _selftest_missing_sources() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root, _frames(), _windows())
        missing_csv = ReviewSources(**{**sources.__dict__, "windows_csv": root / "absent" / "windows.csv"})
        missing_frames = ReviewSources(**{**sources.__dict__, "frames_path": root / "absent" / "frames.parquet"})
        sources.video_paths["EnvC/v3"].unlink()
        output_dir = root / "out"
        with patch.object(other_walk_review_set, "LOCAL_OUTPUT_ROOT", root):
            raised_csv = _raises(FileNotFoundError, lambda: run_review_set(missing_csv, root / "o1"), "Etapa 3A")
            raised_frames = _raises(FileNotFoundError, lambda: run_review_set(missing_frames, root / "o2"),
                                    "frames.parquet")
            run_review_set(sources, output_dir, decode_fn=_fake_decode)
            sources.video_paths["EnvL/v1"].write_bytes(b"altered")
            mismatch = _raises(ValueError, lambda: run_review_set(sources, root / "o3", decode_fn=_fake_decode),
                               "diverge do manifesto")
        report = json.loads((output_dir / KEY_DIR / SELECTION_JSON_FILE).read_text(encoding="utf-8"))
        worksheet = pd.read_csv(output_dir / WORKSHEET_DIR / WORKSHEET_CSV_FILE, keep_default_na=False)
        sheets = sorted(path.name for path in (output_dir / WORKSHEET_DIR / "sheets").iterdir())
        coffee = [item for item in report["items"] if item["video_id"] == "EnvC/v3"]
        nothing_written = not (root / "o1").exists() and not (root / "o2").exists() and not (root / "o3").exists()
    return all([
        _check("ausência: window_context_windows.csv ausente falha antes de gravar", raised_csv),
        _check("ausência: frames.parquet ausente falha antes de gravar", raised_frames),
        _check("ausência: vídeo ausente registrado em missing_inputs, sem folha inventada",
               [item["video_id"] for item in report["missing_inputs"]] == ["EnvC/v3"]
               and all(item["sheet"] is None for item in coffee) and len(coffee) == 2
               and set(worksheet.loc[worksheet["video_id"] == "EnvC/v3", "sheet"]) == {""}),
        _check("ausência: folhas dos vídeos presentes continuam gravadas", len(sheets) == 4),
        _check("ausência: SHA-256 do vídeo divergente do manifesto falha", mismatch),
        _check("ausência: nenhuma saída gravada nas falhas", nothing_written),
    ])


def _selftest_protected_outputs() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root, _frames(), _windows())
        protected = [
            REFERENCE_RUN_ROOT / "review",
            REPOSITORY_ROOT / "src" / "review",
            REPOSITORY_ROOT / "data" / "review",
            REPOSITORY_ROOT / "docs" / "review",
            sources.windows_csv.parent / "review",
            sources.frames_path.parent,
            root / "raw" / "review",
            root,
        ]
        refused = all(_raises(ValueError, lambda path=path: run_review_set(sources, path, decode_fn=_fake_decode),
                              "protegido") for path in protected)
        versioned = _raises(ValueError, lambda: run_review_set(sources, REPOSITORY_ROOT / "review_out",
                                                                decode_fn=_fake_decode), "runs/local")
        before = _tree_hashes(root)
        output_dir = root / "local" / "review"
        with patch.object(other_walk_review_set, "LOCAL_OUTPUT_ROOT", root / "local"):
            written = run_review_set(sources, output_dir, decode_fn=_fake_decode)
            first = _tree_hashes(output_dir)
            skipped = not run_review_set(sources, output_dir, decode_fn=_fake_decode)
            unchanged = _tree_hashes(output_dir) == first
            forced = run_review_set(sources, output_dir, force=True, decode_fn=_fake_decode)
        inputs_unchanged = {name: digest for name, digest in _tree_hashes(root).items()
                            if not name.startswith("local/")} == before
    return all([
        _check("saída: referências, src, data, docs, entradas e vídeos brutos recusados", refused),
        _check("saída: diretório versionado do repositório fora de runs/local recusado",
               versioned and not (REPOSITORY_ROOT / "review_out").exists()),
        _check("saída: sem --force preserva saída existente e com --force regrava",
               written and skipped and unchanged and forced),
        _check("saída: entradas inalteradas", inputs_unchanged),
    ])


SELFTESTS: tuple[Callable[[], bool], ...] = (
    _selftest_deterministic_selection,
    _selftest_exact_frame_alignment,
    _selftest_missing_sources,
    _selftest_protected_outputs,
)


def run_selftest() -> None:
    results = [selftest() for selftest in SELFTESTS]
    if not all(results):
        sys.exit(1)
    print("selftest do conjunto de revisão other→walk: todos os casos passaram")
