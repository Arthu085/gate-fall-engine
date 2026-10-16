import copy
import io
import json
import os
import stat
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

from gatefall.config import EVAL_STRIDE, TRAIN_STRIDE
from gatefall.data.video_io import decode_frames
from gatefall.datasets.le2i import LE2I_LABEL_NAMES
from gatefall.eval.analysis import other_walk_review_set, standing_sitting_review_set
from gatefall.eval.analysis.classification_diagnosis import EXPECTED_SEEDS
from gatefall.eval.analysis.multiseed_summary import ARMS
from gatefall.eval.analysis.other_walk_review_set import (
    FRAMES_CSV_FILE,
    KEY_CSV_FILE,
    KEY_DIR,
    SELECTION_JSON_FILE,
    SELECTION_SUMMARY_FILE,
    WORKSHEET_CSV_FILE,
    WORKSHEET_DIR,
    ReviewSources,
    records,
    tile_lines,
    tile_origin,
)
from gatefall.eval.analysis.qualitative import _with_caption_strip
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.eval.analysis.selftests.stratified_error_diagnosis import _raises, _tree_hashes
from gatefall.eval.analysis.standing_sitting_diagnosis import (
    FOCUS_CLASS_IDS,
    FOCUS_TRUE_IDS,
    OFFICIAL_TRAIN,
    OFFICIAL_VAL,
    REPORT_JSON_FILE,
    REPORT_WINDOWS_FILE,
    SUBJECT_FOLD,
    labeled_windows,
    window_rows,
)
from gatefall.eval.analysis.standing_sitting_review_set import (
    CANDIDATES_UNAVAILABLE,
    CONTROL,
    ERROR,
    KEY_DIR_MODE,
    KEY_FILE_MODE,
    NO_CANDIDATE,
    PREDICTED_FOR,
    REVIEW_FIELDS,
    SUBJECT_MATCH,
    UNMATCHED,
    VIDEO_MATCH,
    guarded_decoder,
    plan_review_set,
    run_review_set,
    sheet_header,
    window_mapping,
)
from gatefall.hashing import sha256_file
from gatefall.pose.loading import PoseArrays
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT

WALK, SITTING, STAND_UP, STANDING = (
    LE2I_LABEL_NAMES.index(name) for name in ("walk", "sitting", "stand_up", "standing")
)
FRAME_SIZE = (32, 24)
FOLDS: tuple[tuple[str, tuple[int, ...]], ...] = (("retain_s0", (0,)), ("retain_s1", (1,)))

# (video_id, split, env, subject, rótulo por quadro, quadros sem pessoa detectada).
# EnvV: val oficial, sujeito 5. EnvT: train oficial; os sujeitos 0 e 1 são os
# retidos das dobras retain_s0 e retain_s1, o sujeito 3 pertence a uma dobra
# ausente da Etapa 6A e o sujeito 2 não é retido. EnvX/t0 é de test.
VIDEO_SPEC: tuple[tuple[str, str, str, int, list[int], list[int]], ...] = (
    ("EnvV/v1", "val", "Home_01", 5, [STANDING] * 200 + [WALK] * 20 + [SITTING] * 200 + [STAND_UP] * 10, []),
    ("EnvV/v2", "val", "Home_01", 5, [STANDING] * 100 + [WALK] * 10, []),
    ("EnvT/a0", "train", "Office", 0, [SITTING] * 100 + [STAND_UP] * 10, list(range(50, 91))),
    ("EnvT/a1", "train", "Office", 0, [SITTING] * 60 + [STAND_UP] * 10, []),
    ("EnvT/b0", "train", "Office", 1, [STANDING] * 80 + [WALK] * 10, []),
    ("EnvT/c0", "train", "Office", 3, [SITTING] * 50 + [WALK] * 10, []),
    ("EnvT/d0", "train", "Office", 2, [WALK] * 40 + [STANDING] * 30, []),
    ("EnvX/t0", "test", "Lecture_room", 9, [STANDING] * 60 + [WALK] * 10, []),
)
TEST_VIDEO = "EnvX/t0"

# (video_id, k_end) → (papel, votos por braço quando diferentes de 5 de 5).
# Janelas fora daqui: 3 de 5 seeds corretas e 2 de 5 no erro, sem persistência.
VAL_SPEC: dict[tuple[str, int], tuple[str, dict[str, int]]] = {
    ("EnvV/v1", 10): ("error", {}),
    ("EnvV/v1", 100): ("error", {"A": 4}),
    ("EnvV/v1", 60): ("error", {"B1": 3}),
    ("EnvV/v2", 30): ("error", {}),
    ("EnvV/v1", 197): ("error", {}),
    ("EnvV/v1", 150): ("correct", {}),
    ("EnvV/v2", 70): ("correct", {}),
    ("EnvV/v2", 97): ("correct", {}),
    ("EnvV/v1", 199): ("correct", {}),
    ("EnvV/v1", 230): ("error", {}),
    ("EnvV/v1", 240): ("correct", {}),
    ("EnvV/v1", 300): ("error", {}),
}
# Dobras: um modelo; janelas fora daqui são previstas como `lying`.
FOLD_SPEC: dict[tuple[str, int], str] = {
    ("EnvT/a0", 70): "error",
    ("EnvT/a0", 95): "correct",
    ("EnvT/a0", 30): "error",
    ("EnvT/a1", 40): "correct",
    ("EnvT/b0", 20): "error",
    ("EnvT/b0", 60): "correct",
}
EXPECTED_SELECTION = {
    (OFFICIAL_VAL, ERROR, "EnvV/v1", 197),
    (OFFICIAL_VAL, ERROR, "EnvV/v1", 10),
    (OFFICIAL_VAL, ERROR, "EnvV/v2", 30),
    (OFFICIAL_VAL, ERROR, "EnvV/v1", 230),
    (OFFICIAL_VAL, ERROR, "EnvV/v1", 300),
    (OFFICIAL_VAL, CONTROL, "EnvV/v2", 97),
    (OFFICIAL_VAL, CONTROL, "EnvV/v1", 150),
    (OFFICIAL_VAL, CONTROL, "EnvV/v2", 70),
    (SUBJECT_FOLD, ERROR, "EnvT/a0", 70),
    (SUBJECT_FOLD, ERROR, "EnvT/a0", 30),
    (SUBJECT_FOLD, ERROR, "EnvT/b0", 20),
    (SUBJECT_FOLD, CONTROL, "EnvT/a0", 95),
    (SUBJECT_FOLD, CONTROL, "EnvT/a1", 40),
    (SUBJECT_FOLD, CONTROL, "EnvT/b0", 60),
}
# Erro → (controle, nível de pareamento, motivo da ausência de controle).
EXPECTED_MATCHES: dict[tuple[str, int], tuple[tuple[str, int] | None, str, str | None]] = {
    ("EnvV/v1", 197): (("EnvV/v2", 97), SUBJECT_MATCH, None),
    ("EnvV/v1", 10): (("EnvV/v1", 150), VIDEO_MATCH, None),
    ("EnvV/v2", 30): (("EnvV/v2", 70), VIDEO_MATCH, None),
    ("EnvV/v1", 230): (None, UNMATCHED, CANDIDATES_UNAVAILABLE),
    ("EnvV/v1", 300): (None, UNMATCHED, NO_CANDIDATE),
    ("EnvT/a0", 70): (("EnvT/a0", 95), VIDEO_MATCH, None),
    ("EnvT/a0", 30): (("EnvT/a1", 40), SUBJECT_MATCH, None),
    ("EnvT/b0", 20): (("EnvT/b0", 60), VIDEO_MATCH, None),
}


def _src_index(frame_index: int) -> int:
    return (5 * frame_index) // 2


def _frames(spec: Sequence[tuple[str, str, str, int, list[int], list[int]]] = VIDEO_SPEC) -> pd.DataFrame:
    return pd.DataFrame([
        {"video_id": video_id, "split": split, "env": env, "subject": subject, "frame_index": k, "time_s": k / 10,
         "src_index": _src_index(k), "label": label, "gap_position": None}
        for video_id, split, env, subject, labels, _missing in spec
        for k, label in enumerate(labels)
    ])


def _pose(n_frames: int, missing: list[int]) -> PoseArrays:
    keypoints = np.zeros((n_frames, 17, 3), dtype=np.float32)
    keypoints[:, :, 2] = 0.8
    person_found = np.ones(n_frames, dtype=bool)
    person_found[missing] = False
    bbox = np.tile(np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float32), (n_frames, 1))
    return PoseArrays(keypoints=keypoints, bbox=bbox, person_found=person_found, k=n_frames, width=32, height=24)


POSES = {video_id: _pose(len(labels), missing) for video_id, _split, _env, _subject, labels, missing in VIDEO_SPEC}


def _val_prediction(true_id: int, video_id: str, k_end: int, arm: str, seed_index: int) -> int:
    true_label = LE2I_LABEL_NAMES[true_id]
    confusable = PREDICTED_FOR[true_label]
    role, votes = VAL_SPEC.get((video_id, k_end), ("default", {}))
    if role == "default":
        label = true_label if seed_index < 3 else confusable
    else:
        target, alternate = (confusable, true_label) if role == "error" else (true_label, confusable)
        label = target if seed_index < votes.get(arm, 5) else alternate
    return LE2I_LABEL_NAMES.index(label)


def _fold_prediction(true_id: int, video_id: str, k_end: int) -> int:
    role = FOLD_SPEC.get((video_id, k_end))
    if role is None:
        return LE2I_LABEL_NAMES.index("lying")
    return LE2I_LABEL_NAMES.index(PREDICTED_FOR[LE2I_LABEL_NAMES[true_id]]) if role == "error" else true_id


def _pair_counts(table: pd.DataFrame) -> dict:
    pairs = {}
    for true_id, predicted in ((STANDING, "walk"), (SITTING, "stand_up")):
        subset = cast(pd.DataFrame, table[table["label"] == true_id])
        n_error = int((subset["pred_label"] == LE2I_LABEL_NAMES.index(predicted)).sum())
        n_correct = int((subset["pred_label"] == true_id).sum())
        pairs[f"{LE2I_LABEL_NAMES[true_id]}->{predicted}"] = {"counts": {
            "n_true": len(subset), "n_error": n_error, "n_correct": n_correct,
            "n_other_outcome": len(subset) - n_error - n_correct,
        }}
    return pairs


def _stage6a(frames: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """CSV e JSON no formato da Etapa 6A, com atributos de janela calculados pelo próprio código da Etapa 6A."""
    def load(video_id: str) -> PoseArrays:
        return POSES[video_id]

    val_windows = labeled_windows(cast(pd.DataFrame, frames[frames["split"] == "val"]), EVAL_STRIDE, FOCUS_TRUE_IDS,
                                  load)
    tables = []
    runs = []
    for arm in ARMS:
        for seed_index, seed in enumerate(EXPECTED_SEEDS):
            predictions = [_val_prediction(int(label), str(video_id), int(k_end), arm, seed_index)
                           for label, video_id, k_end in zip(val_windows["label"], val_windows["video_id"],
                                                              val_windows["k_end"])]
            table = val_windows.assign(pred_label=predictions, evidence=OFFICIAL_VAL, fold="", arm=arm, seed=seed)
            tables.append(table)
            runs.append({"arm": arm, "seed": seed, "source": f"runs/synthetic/{arm}/seed_{seed}",
                         "checkpoint_sha256": f"{arm}-{seed}", "pairs": _pair_counts(table)})
    blocks = []
    for fold, subjects in FOLDS:
        retained = cast(pd.DataFrame, frames[(frames["split"] == "train") & frames["subject"].isin(list(subjects))])
        windows = labeled_windows(retained, EVAL_STRIDE, FOCUS_TRUE_IDS, load)
        predictions = [_fold_prediction(int(label), str(video_id), int(k_end))
                       for label, video_id, k_end in zip(windows["label"], windows["video_id"], windows["k_end"])]
        table = windows.assign(pred_label=predictions, evidence=SUBJECT_FOLD, fold=fold, arm="A", seed=42)
        tables.append(table)
        blocks.append({"evidence": SUBJECT_FOLD, "fold": fold, "arm": "A", "seed": 42,
                       "retained_subjects": list(subjects), "run_dir": f"runs/synthetic/{fold}",
                       "checkpoint_sha256": fold, "pairs": _pair_counts(table)})
    train = labeled_windows(cast(pd.DataFrame, frames[frames["split"] == "train"]), TRAIN_STRIDE, FOCUS_CLASS_IDS, load)
    train = train.assign(evidence=OFFICIAL_TRAIN, fold="", arm="", seed="", pred_label=None)
    csv = pd.concat([window_rows(table) for table in (*tables, train)], ignore_index=True)
    # Ida e volta pelo texto do CSV, como a Etapa 6B lê a saída gravada da Etapa 6A.
    csv = pd.read_csv(io.StringIO(csv.to_csv(index=False)), dtype=str, keep_default_na=False)
    report = {
        "analysis": "standing_sitting_error_diagnosis", "stage": "6A", "canonical": False, "read_only": True,
        "dataset": "le2i", "protocol": "Le2i-CS", "official_val": {"runs": runs}, "subject_folds": {"folds": blocks},
        "decision": {"verdict": "unresolved_limitation"}, "missing_evidence": [],
    }
    return csv, report


FRAMES = _frames()
STAGE6A_CSV, STAGE6A_REPORT = _stage6a(FRAMES)


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


def _sources(root: Path, frames: pd.DataFrame = FRAMES, csv: pd.DataFrame = STAGE6A_CSV,
             report: dict = STAGE6A_REPORT) -> ReviewSources:
    frames_path = root / "data" / "frames.parquet"
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(frames_path)
    diagnosis_dir = root / "stage6a"
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    csv.to_csv(diagnosis_dir / REPORT_WINDOWS_FILE, index=False)
    (diagnosis_dir / REPORT_JSON_FILE).write_text(json.dumps(report), encoding="utf-8")
    raw_dir = root / "raw"
    raw_dir.mkdir(exist_ok=True)
    video_paths = {video_id: raw_dir / f"{video_id.replace('/', '__')}.avi"
                   for video_id, *_rest in VIDEO_SPEC}
    for video_id, path in video_paths.items():
        path.write_bytes(f"video {video_id}".encode())
    return ReviewSources(
        windows_csv=diagnosis_dir / REPORT_WINDOWS_FILE,
        frames_path=frames_path,
        video_paths=video_paths,
        video_sha256={video_id: sha256_file(path) for video_id, path in video_paths.items()},
        extra_inputs=(diagnosis_dir / REPORT_JSON_FILE,),
        protected=(raw_dir,),
    )


def _selection(plan) -> set[tuple[str, str, str, int]]:
    return {(item["evidence"], item["role"], item["video_id"], int(item["k_end"])) for _id, item, _map in plan.reviews}


def _run(sources: ReviewSources, output_dir: Path, root: Path, decode_fn=_fake_decode, force: bool = False) -> bool:
    with patch.object(other_walk_review_set, "LOCAL_OUTPUT_ROOT", root):
        return run_review_set(sources, output_dir, force=force, decode_fn=decode_fn)


def _text_outputs(output_dir: Path) -> dict[str, str]:
    return {name: (output_dir / name).read_text(encoding="utf-8") for name in (
        f"{KEY_DIR}/{SELECTION_JSON_FILE}", f"{KEY_DIR}/{KEY_CSV_FILE}", f"{KEY_DIR}/{SELECTION_SUMMARY_FILE}",
        f"{WORKSHEET_DIR}/{WORKSHEET_CSV_FILE}", f"{WORKSHEET_DIR}/{FRAMES_CSV_FILE}",
    )}


def _comparable(outputs: dict[str, str]) -> dict[str, object]:
    """Saídas sem os campos que dependem do caminho e dos bytes das entradas."""
    report = json.loads(outputs[f"{KEY_DIR}/{SELECTION_JSON_FILE}"])
    report.pop("input_sha256")
    for field in ("windows_csv", "report_json"):
        report["source"].pop(field)
    return {**{name: text for name, text in outputs.items() if not name.endswith(".json")}, "report": report}


def _key(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / KEY_DIR / KEY_CSV_FILE, keep_default_na=False)


def _selftest_selection_and_matching() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root)
        plan = plan_review_set(sources)
        selected = _selection(plan)
        output_dir = root / "out"
        _run(sources, output_dir, root)
        report = json.loads((output_dir / KEY_DIR / SELECTION_JSON_FILE).read_text(encoding="utf-8"))
        key = _key(output_dir)
        with patch.object(standing_sitting_review_set, "MAX_ERRORS_PER_STRATUM", 1):
            capped = plan_review_set(sources)

    by_id = {row["review_id"]: row for row in records(key)}
    observed_matches = {}
    for row in by_id.values():
        if row["role"] != ERROR:
            continue
        partner = by_id.get(row["matched_review_id"])
        control = None if partner is None else (partner["video_id"], int(partner["k_end"]))
        observed_matches[(row["video_id"], int(row["k_end"]))] = (
            control, row["match_level"], row["unmatched_reason"] or None)
    controls_consistent = all(
        by_id[row["matched_review_id"]]["matched_review_id"] == row["review_id"]
        and all(by_id[row["matched_review_id"]][field] == row[field]
                for field in ("evidence", "fold", "subject", "true_label", "category"))
        for row in by_id.values() if row["role"] == CONTROL
    )
    val_strata = {(entry["true_label"], entry["category"]): entry for entry in report["coverage"][OFFICIAL_VAL]}
    standing_interior = val_strata[("standing", "interior")]
    items = sorted(selected, key=lambda item: (item[0], item[2], item[3]))
    capped_reasons = [(item["true_label"], item["category"], item["reason"]) for item in capped.exclusions]
    return all([
        _check("seleção: erros e controles esperados nas duas evidências", selected == EXPECTED_SELECTION),
        _check("seleção: limiar inclusivo de 4/5 seeds por braço; 3/5 em um braço não conta",
               standing_interior[ERROR]["n_windows"] == 3 and standing_interior[ERROR]["n_segments"] == 2),
        _check("seleção: um item por trecho, estrato e papel, o de maior fração mínima entre braços",
               (OFFICIAL_VAL, ERROR, "EnvV/v1", 100) not in selected),
        _check("seleção: janelas do mesmo vídeo a pelo menos 24 quadros umas das outras",
               all(abs(a[3] - b[3]) >= 24 for a in items for b in items if a != b and a[2] == b[2])),
        _check("pareamento: mesmo vídeo antes do mesmo sujeito; categoria e classe sempre iguais",
               observed_matches == EXPECTED_MATCHES and controls_consistent),
        _check("pareamento: erro sem controle registrado, sem relaxar categoria nem sujeito",
               standing_interior["match_levels"] == {VIDEO_MATCH: 2, SUBJECT_MATCH: 0, UNMATCHED: 0}
               and val_strata[("sitting", "interior")]["match_levels"][UNMATCHED] == 1
               and val_strata[("sitting", "transition")]["match_levels"][UNMATCHED] == 1),
        _check("pareamento: pose ausente é uma categoria pareada como as demais",
               (SUBJECT_FOLD, CONTROL, "EnvT/a0", 95) in selected
               and EXPECTED_MATCHES[("EnvT/a0", 70)][1] == VIDEO_MATCH),
        _check("evidências: val e dobras numeradas e contabilizadas separadamente",
               sorted(by_id) == [*(f"fold-{index:02d}" for index in range(1, 7)),
                                 *(f"val-{index:02d}" for index in range(1, 9))]
               and set(report["coverage"]) == {OFFICIAL_VAL, SUBJECT_FOLD}
               and {entry["fold"] for entry in report["coverage"][SUBJECT_FOLD]} == {"retain_s0", "retain_s1"}),
        _check("evidências: dobra ausente da Etapa 6A registrada, sem seleção inventada",
               [item["source"] for item in report["missing_evidence"]] == ["retain_s3-4-6-8"]
               and not any(item["video_id"] == "EnvT/c0" for item in report["items"])),
        _check("seleção: limite por estrato registra o trecho preterido",
               capped_reasons == [("standing", "interior", "cap")] and len(capped.reviews) == 12),
        _check("chave: procedência dos 27 modelos da Etapa 6A registrada",
               len(report["source"]["models"]) == 27 and report["source"]["verdict"] == "unresolved_limitation"),
    ])


def _selftest_determinism() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root / "a")
        csv = STAGE6A_CSV.sample(frac=1.0, random_state=11)
        shuffled = _sources(root / "b", FRAMES.sample(frac=1.0, random_state=5), csv)
        _run(sources, root / "out_a", root)
        _run(shuffled, root / "out_b", root)
        _run(sources, root / "out_c", root)
        first, second, third = (_text_outputs(root / name) for name in ("out_a", "out_b", "out_c"))
        sheets_a = _tree_hashes(root / "out_a" / WORKSHEET_DIR / "sheets")
        sheets_c = _tree_hashes(root / "out_c" / WORKSHEET_DIR / "sheets")
    return all([
        _check("determinismo: duas execuções sobre as mesmas entradas são idênticas byte a byte",
               first == third and sheets_a == sheets_c),
        _check("determinismo: ordem das linhas do CSV e de frames.parquet não altera as saídas",
               _comparable(first) == _comparable(second)),
    ])


def _test_variant_frames() -> pd.DataFrame:
    frames = FRAMES.copy()
    test = frames["split"] == "test"
    frames.loc[test, "label"] = WALK
    frames.loc[test, "env"] = "Coffee_room_01"
    return frames


def _selftest_split_isolation() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root / "a")
        decoded: list[Path] = []

        def spy(path: Path, src_indices: Sequence[int]) -> list[np.ndarray]:
            decoded.append(path)
            return _fake_decode(path, src_indices)

        _run(sources, root / "out_a", root, decode_fn=spy)
        altered = _sources(root / "b", _test_variant_frames())
        altered.video_paths[TEST_VIDEO].write_bytes(b"different test video")
        _run(altered, root / "out_b", root)
        report = json.loads((root / "out_a" / KEY_DIR / SELECTION_JSON_FILE).read_text(encoding="utf-8"))
        same = _comparable(_text_outputs(root / "out_a")) == _comparable(_text_outputs(root / "out_b"))

        moved = STAGE6A_CSV.copy()
        moved.loc[(moved["evidence"] == OFFICIAL_VAL) & (moved["video_id"] == "EnvV/v2"), "video_id"] = TEST_VIDEO
        moved_fails = _raises(ValueError, lambda: _run(_sources(root / "c", csv=moved), root / "out_c", root),
                              "fora de train/val")
        swapped = STAGE6A_CSV.copy()
        swapped.loc[swapped["evidence"] == SUBJECT_FOLD, "evidence"] = "official_test"
        swapped_fails = _raises(ValueError, lambda: _run(_sources(root / "d", csv=swapped), root / "out_d", root),
                                "evidência desconhecida")
        relabeled = FRAMES.copy()
        relabeled.loc[relabeled["video_id"] == "EnvT/b0", "split"] = "val"
        split_fails = _raises(ValueError, lambda: _run(_sources(root / "e", relabeled), root / "out_e", root),
                              "divergem de frames.parquet")
        guard = guarded_decoder(_fake_decode, {sources.video_paths["EnvV/v1"].resolve(): "val"}, {})
        guard_refuses = _raises(RuntimeError, lambda: guard(sources.video_paths[TEST_VIDEO], [0]), "fora de train/val")
        hashed = set(report["input_sha256"])
        nothing_written = not any((root / name).exists() for name in ("out_c", "out_d", "out_e"))
    return all([
        _check("isolamento: frames.parquet lido só em train/val",
               report["split_access"]["frames_parquet_rows_read"] == ["train", "val"]),
        _check("isolamento: vídeo de test nunca decodificado nem hasheado",
               all(TEST_VIDEO.replace("/", "__") not in str(path) for path in decoded) and bool(decoded)
               and not any("EnvX__t0" in name for name in hashed)
               and set(report["split_access"]["videos_decoded_by_split"]) == {"train", "val"}),
        _check("isolamento: rótulos, ambiente e vídeo de test alterados não mudam nenhuma saída", same),
        _check("isolamento: janela da Etapa 6A em vídeo de test falha", moved_fails),
        _check("isolamento: evidência fora de val oficial e dobras falha", swapped_fails),
        _check("isolamento: vídeo de dobra que não está em train falha", split_fails),
        _check("isolamento: decodificação de vídeo fora de train/val recusada", guard_refuses),
        _check("isolamento: nenhuma saída gravada nas falhas", nothing_written),
    ])


def _consistent(csv: pd.DataFrame, evidence: str, video_id: str, k_end: int, field: str, value: str) -> pd.DataFrame:
    changed = csv.copy()
    mask = (changed["evidence"] == evidence) & (changed["video_id"] == video_id) & (changed["k_end"] == str(k_end))
    changed.loc[mask, field] = value
    return changed


def _selftest_alignment_and_ffmpeg() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root)
        video_path = sources.video_paths["EnvV/v2"]
        _write_video(video_path, n_src_frames=_src_index(109) + 3)
        sources = ReviewSources(**{**sources.__dict__, "video_sha256": {
            **sources.video_sha256, "EnvV/v2": sha256_file(video_path)}})

        def decode(path: Path, src_indices: Sequence[int]) -> list[np.ndarray]:
            return decode_frames(path, src_indices) if path == video_path else _fake_decode(path, src_indices)

        output_dir = root / "out"
        _run(sources, output_dir, root, decode_fn=decode)
        key = _key(output_dir)
        frame_rows = pd.read_csv(output_dir / WORKSHEET_DIR / FRAMES_CSV_FILE)
        v2 = cast(pd.DataFrame, FRAMES[FRAMES["video_id"] == "EnvV/v2"]).reset_index(drop=True)
        tile_checks = []
        for row in records(cast(pd.DataFrame, key[key["video_id"] == "EnvV/v2"])):
            sheet = np.array(Image.open(output_dir / WORKSHEET_DIR / str(row["sheet"])).convert("RGB"))
            mapping = window_mapping(v2, int(row["k_end"]))
            tiles = [_with_caption_strip(_frame(int(src)), tile_lines(tile)) for src, tile in
                     zip(mapping["src_index"], records(mapping))]
            height, width = max(tile.shape[0] for tile in tiles), max(tile.shape[1] for tile in tiles)
            colors = [tuple(int(value) for value in sheet[top + 2, left + 2])
                      for top, left in (tile_origin(index, height, width) for index in range(len(tiles)))]
            tile_checks.append(colors == [_color(int(src)) for src in mapping["src_index"]])
        padded = key[(key["video_id"] == "EnvV/v1") & (key["k_end"] == 10)].iloc[0]
        padded_frames = frame_rows[frame_rows["review_id"] == padded["review_id"]]

        variants = {
            "pureza divergente": (_consistent(STAGE6A_CSV, OFFICIAL_VAL, "EnvV/v1", 230, "purity", "pure"),
                                  "divergem de frames.parquet"),
            "fim do trecho divergente": (_consistent(STAGE6A_CSV, OFFICIAL_VAL, "EnvV/v2", 30,
                                                     "frames_to_segment_end", "68"), "divergem de frames.parquet"),
            "rótulo seguinte divergente": (_consistent(STAGE6A_CSV, SUBJECT_FOLD, "EnvT/b0", 20, "following_label",
                                                       "fall"), "following_label"),
            "categoria fora da regra da Etapa 6A": (_consistent(STAGE6A_CSV, OFFICIAL_VAL, "EnvV/v1", 10, "category",
                                                                "transition"), "categoria"),
        }
        outcomes = {name: _raises(ValueError, lambda csv=csv, index=index: _run(
            _sources(root / f"in{index}", csv=csv), root / f"o{index}", root), message)
            for index, (name, (csv, message)) in enumerate(variants.items())}
        relabeled = FRAMES.copy()
        relabeled.loc[(relabeled["video_id"] == "EnvV/v1") & (relabeled["frame_index"] == 10), "label"] = WALK
        relabeled_fails = _raises(ValueError, lambda: _run(_sources(root / "rl", relabeled), root / "o_rl", root),
                                  "divergem de frames.parquet")
        nothing_written = not any((root / f"o{index}").exists() for index in range(len(variants)))
    expected_src = [_src_index(max(0, k)) for k in range(10 - 23, 11)]
    return all([
        _check("FFmpeg: cada ladrilho das folhas de um AVI real é o quadro src_index esperado",
               len(tile_checks) == 3 and all(tile_checks)),
        _check("alinhamento: replicação de borda e mapeamento frame_index → src_index na planilha de quadros",
               padded_frames["src_index"].tolist() == expected_src
               and padded_frames["replicated"].tolist() == [True] * 13 + [False] * 11
               and int(padded["n_replicated_frames"]) == 13 and int(padded["window_frame_start"]) == 0),
        _check("alinhamento: rótulos por quadro de janela misturada vêm de frames.parquet",
               frame_rows[frame_rows["review_id"] == key[(key["video_id"] == "EnvV/v1")
                                                            & (key["k_end"] == 230)].iloc[0]["review_id"]]["label"]
               .tolist() == ["walk"] * 13 + ["sitting"] * 11),
        *(_check(f"alinhamento: {name} falha", raised) for name, raised in outcomes.items()),
        _check("alinhamento: rótulo do quadro final alterado em frames.parquet falha", relabeled_fails),
        _check("alinhamento: nenhuma saída gravada nas falhas", nothing_written),
    ])


FORBIDDEN_WORKSHEET_TERMS = ("role", "pred", "vote", "share", "category", "match", "evidence", "fold", "order",
                             "error", "correct", "control", "model", "arm", "seed")


def _selftest_blinding() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root)
        plan = plan_review_set(sources)
        output_dir = root / "out"
        _run(sources, output_dir, root)
        worksheet = pd.read_csv(output_dir / WORKSHEET_DIR / WORKSHEET_CSV_FILE, keep_default_na=False)
        frames = pd.read_csv(output_dir / WORKSHEET_DIR / FRAMES_CSV_FILE)
        report = json.loads((output_dir / KEY_DIR / SELECTION_JSON_FILE).read_text(encoding="utf-8"))
        recorded = report["worksheet_sha256"]
        actual = {str(path.relative_to(output_dir)): sha256_file(path)
                  for path in sorted((output_dir / WORKSHEET_DIR).rglob("*")) if path.is_file()}
        key_mode = stat.S_IMODE(os.stat(output_dir / KEY_DIR).st_mode)
        file_modes = {stat.S_IMODE(os.stat(path).st_mode) for path in (output_dir / KEY_DIR).iterdir()}
        worksheet_names = {path.name for path in (output_dir / WORKSHEET_DIR).rglob("*")}
        headers = [line for review_id, item, mapping in plan.reviews for line in sheet_header(review_id, item, mapping)]
    columns = " ".join(worksheet.columns).lower() + " " + " ".join(frames.columns).lower()
    return all([
        _check("cegueira: planilha e quadros sem papel, predição, votos, categoria, pareamento ou evidência",
               not [term for term in FORBIDDEN_WORKSHEET_TERMS if term in columns]),
        _check("cegueira: cabeçalhos das folhas sem papel nem classe confundível",
               not [line for line in headers
                    if any(term in line.lower() for term in ("error", "correct", "control", "walk", "stand_up"))]),
        _check("cegueira: campos de revisão vazios e rótulo verdadeiro registrado",
               all((worksheet[list(REVIEW_FIELDS)] == "").to_numpy().ravel())
               and set(worksheet["true_label"]) == {"standing", "sitting"}),
        _check("chave: diretório separado, com permissão só do dono",
               key_mode == KEY_DIR_MODE and file_modes == {KEY_FILE_MODE}
               and not worksheet_names & {SELECTION_JSON_FILE, KEY_CSV_FILE, SELECTION_SUMMARY_FILE}),
        _check("chave: SHA-256 de cada arquivo da planilha e das folhas registrados", recorded == actual),
    ])


def _report_variant(change: Callable[[dict], None]) -> dict:
    report = copy.deepcopy(STAGE6A_REPORT)
    change(report)
    return report


def _selftest_input_integrity() -> bool:
    csv = STAGE6A_CSV
    val = csv["evidence"] == OFFICIAL_VAL
    first_val = csv.index[val][0]
    invalid_label = csv.copy()
    invalid_label.loc[first_val, "pred_label"] = "running"
    divergent = csv.copy()
    divergent.loc[first_val, "detected_fraction"] = "0.5"
    train_prediction = csv.copy()
    train_prediction.loc[csv.index[csv["evidence"] == OFFICIAL_TRAIN][0], "pred_label"] = "walk"
    fold_arm = csv.copy()
    fold_arm.loc[fold_arm["evidence"] == SUBJECT_FOLD, "arm"] = "B0"
    one_window_missing = csv.drop(index=first_val)

    def extra_fold(report: dict) -> None:
        report["subject_folds"]["folds"].append({**report["subject_folds"]["folds"][0], "fold": "retain_s3-4-6-8"})

    def miscount(report: dict) -> None:
        report["official_val"]["runs"][3]["pairs"]["standing->walk"]["counts"]["n_error"] += 1

    variants: dict[str, tuple[pd.DataFrame, dict, str]] = {
        "coluna ausente": (csv.drop(columns="category"), STAGE6A_REPORT, "colunas"),
        "modelo do val ausente": (cast(pd.DataFrame, csv[~(val & (csv["arm"] == "B0") & (csv["seed"] == "44"))]),
                                  STAGE6A_REPORT, "sem os modelos"),
        "linha repetida": (pd.concat([csv, csv.loc[[first_val]]], ignore_index=True), STAGE6A_REPORT, "repetido"),
        "predição fora dos rótulos": (invalid_label, STAGE6A_REPORT, "predição fora"),
        "atributos divergentes entre modelos": (divergent, STAGE6A_REPORT, "divergentes entre modelos"),
        "janela ausente em um modelo": (one_window_missing, STAGE6A_REPORT, "ausente em algum modelo"),
        "predição em official_train": (train_prediction, STAGE6A_REPORT, "official_train"),
        "dobra com braço inesperado": (fold_arm, STAGE6A_REPORT, "dobra com nome"),
        "contagem divergente do JSON": (csv, _report_variant(miscount), "contagens do CSV"),
        "modelo só no JSON": (csv, _report_variant(extra_fold), "modelos do CSV e do JSON divergem"),
        "JSON de outra etapa": (csv, _report_variant(lambda report: report.update(stage="6B")), "esperado"),
        "JSON canônico": (csv, _report_variant(lambda report: report.update(canonical=True)), "esperado"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        outcomes = {name: _raises(ValueError, lambda csv=csv, report=report, index=index: _run(
            _sources(root / f"in{index}", csv=csv, report=report), root / f"o{index}", root), message)
            for index, (name, (csv, report, message)) in enumerate(variants.items())}
        nothing_written = not any((root / f"o{index}").exists() for index in range(len(variants)))
    return all([
        *(_check(f"integridade: {name} falha", raised) for name, raised in outcomes.items()),
        _check("integridade: nenhuma saída gravada nas falhas", nothing_written),
    ])


def _selftest_missing_inputs() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root)
        absent = root / "absent"
        missing_csv = ReviewSources(**{**sources.__dict__, "windows_csv": absent / REPORT_WINDOWS_FILE})
        missing_json = _sources(root / "j")
        (missing_json.windows_csv.parent / REPORT_JSON_FILE).unlink()
        missing_frames = ReviewSources(**{**sources.__dict__, "frames_path": absent / "frames.parquet"})
        raised_csv = _raises(FileNotFoundError, lambda: _run(missing_csv, root / "o1", root), "Etapa 6A")
        raised_json = _raises(FileNotFoundError, lambda: _run(missing_json, root / "o2", root), "Etapa 6A")
        raised_frames = _raises(FileNotFoundError, lambda: _run(missing_frames, root / "o3", root), "frames.parquet")
        sources.video_paths["EnvV/v2"].unlink()
        output_dir = root / "out"
        _run(sources, output_dir, root)
        report = json.loads((output_dir / KEY_DIR / SELECTION_JSON_FILE).read_text(encoding="utf-8"))
        worksheet = pd.read_csv(output_dir / WORKSHEET_DIR / WORKSHEET_CSV_FILE, keep_default_na=False)
        sheets = sorted(path.name for path in (output_dir / WORKSHEET_DIR / "sheets").iterdir())
        sources.video_paths["EnvV/v1"].write_bytes(b"altered")
        mismatch = _raises(ValueError, lambda: _run(sources, root / "o4", root), "diverge do manifesto")
        nothing_written = not any((root / name).exists() for name in ("o1", "o2", "o3", "o4"))
    v2_items = [item for item in report["items"] if item["video_id"] == "EnvV/v2"]
    return all([
        _check("ausência: CSV ou JSON da Etapa 6A ausentes falham antes de gravar", raised_csv and raised_json),
        _check("ausência: frames.parquet ausente falha antes de gravar", raised_frames),
        _check("ausência: vídeo bruto ausente registrado, sem folha inventada",
               [item["video_id"] for item in report["missing_inputs"]] == ["EnvV/v2"]
               and len(v2_items) == 3 and all(item["sheet"] is None for item in v2_items)
               and set(worksheet.loc[worksheet["video_id"] == "EnvV/v2", "sheet"]) == {""}),
        _check("ausência: folhas dos vídeos presentes continuam gravadas", len(sheets) == 11),
        _check("ausência: SHA-256 do vídeo divergente do manifesto falha", mismatch),
        _check("ausência: nenhuma saída gravada nas falhas", nothing_written),
    ])


def _selftest_protected_outputs() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = _sources(root)
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
        written = _run(sources, output_dir, root / "local")
        first = _tree_hashes(output_dir)
        skipped = not _run(sources, output_dir, root / "local")
        unchanged = _tree_hashes(output_dir) == first
        forced = _run(sources, output_dir, root / "local", force=True)
        rewritten = _tree_hashes(output_dir) == first
        key_mode = stat.S_IMODE(os.stat(output_dir / KEY_DIR).st_mode)
        inputs_unchanged = {name: digest for name, digest in _tree_hashes(root).items()
                            if not name.startswith("local/")} == before
    return all([
        _check("saída: referências, src, data, docs, Etapa 6A, frames e vídeos brutos recusados", refused),
        _check("saída: diretório versionado do repositório fora de runs/local recusado",
               versioned and not (REPOSITORY_ROOT / "review_out").exists()),
        _check("saída: sem --force preserva saída existente e com --force regrava o mesmo conteúdo",
               written and skipped and unchanged and forced and rewritten and key_mode == KEY_DIR_MODE),
        _check("saída: entradas inalteradas", inputs_unchanged),
    ])


SELFTESTS: tuple[Callable[[], bool], ...] = (
    _selftest_selection_and_matching,
    _selftest_determinism,
    _selftest_split_isolation,
    _selftest_alignment_and_ffmpeg,
    _selftest_blinding,
    _selftest_input_integrity,
    _selftest_missing_inputs,
    _selftest_protected_outputs,
)


def run_selftest() -> None:
    results = [selftest() for selftest in SELFTESTS]
    if not all(results):
        sys.exit(1)
    print("selftest do conjunto de revisão standing/sitting: todos os casos passaram")
