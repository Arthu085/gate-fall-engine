import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, TRAIN_STRIDE
from gatefall.data.windowing import build_window_index, window_frame_indices
from gatefall.eval.analysis.other_walk_train_composition import (
    COMPOSITION_JSON_FILE,
    COMPOSITION_SEGMENTS_FILE,
    COMPOSITION_SUMMARY_FILE,
    COMPOSITION_WINDOWS_FILE,
    SUPPORTED_VERDICT,
    UNDEFINED,
    UNSUPPORTED_VERDICT,
    bbox_displacement,
    build_report,
    conservation,
    decide,
    displacement_bin,
    evaluate_rule,
    run_composition,
    train_tables,
)
from gatefall.eval.analysis.selftests.fixtures import _check
from gatefall.eval.analysis.selftests.stratified_error_diagnosis import _raises, _tree_hashes
from gatefall.eval.analysis.window_context_diagnosis import MIXED, PADDED, POSE_DETECTED, POSE_MISSING, PURE
from gatefall.pose.loading import PoseArrays, pose_path
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT

WALK, OTHER, FALL = 0, 9, 1

# (video_id, split, env, subject, rótulo por quadro). v1 tem um trecho
# `other` inicial com janelas replicadas pela borda, um trecho `walk` e um
# trecho `other` curto (10 quadros) no fim; v2 começa com IGNORE_LABEL. v3
# (val) e v4 (test) nunca podem ser lidos.
VIDEO_SPEC: tuple[tuple[str, str, str, int, list[int]], ...] = (
    ("enva/v1", "train", "EnvA", 1, [OTHER] * 30 + [WALK] * 40 + [OTHER] * 10),
    ("enva/v2", "train", "EnvA", 2, [IGNORE_LABEL] * 4 + [WALK] * 30 + [OTHER] * 28 + [FALL] * 6),
    ("envb/v3", "val", "EnvB", 3, [OTHER] * 40),
    ("envb/v4", "test", "EnvB", 4, [WALK] * 40 + [OTHER] * 40),
)
# Sem pose no quadro 0 de v1: as janelas replicadas pela borda ficam sem
# deslocamento definido e com fração de detecção reduzida.
POSE_MISSING_FRAMES: dict[str, set[int]] = {"enva/v1": {0, 47}}
BOX_HEIGHT = 40.0


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": video_id, "split": split, "env": env, "subject": subject,
             "frame_index": frame_index, "label": label}
            for video_id, split, env, subject, labels in VIDEO_SPEC
            for frame_index, label in enumerate(labels)
        ]
    )


def _pose(video_id: str) -> PoseArrays:
    n_frames = next(len(labels) for spec_id, _split, _env, _subject, labels in VIDEO_SPEC if spec_id == video_id)
    frames = np.arange(n_frames, dtype=np.float32)
    # Centro da bbox avança 1 px por quadro na horizontal, altura constante.
    bbox = np.stack([frames, np.zeros(n_frames), frames + 20.0, np.full(n_frames, BOX_HEIGHT)], axis=1)
    keypoints = np.zeros((n_frames, 17, 3), dtype=np.float32)
    keypoints[:, :, 2] = 0.5
    person_found = np.ones(n_frames, dtype=bool)
    person_found[sorted(POSE_MISSING_FRAMES.get(video_id, set()))] = False
    return PoseArrays(keypoints=keypoints, bbox=bbox.astype(np.float32), person_found=person_found,
                      k=n_frames, width=320, height=240)


def _recording_loader(seen: list[str]) -> Callable[[str], PoseArrays]:
    def load(video_id: str) -> PoseArrays:
        seen.append(video_id)
        if not video_id.startswith("enva/"):
            raise AssertionError(f"pose fora de train lida: {video_id}")
        return _pose(video_id)

    return load


def _window(windows: pd.DataFrame, video_id: str, k_end: int) -> pd.Series:
    match = windows[(windows["video_id"] == video_id) & (windows["k_end"] == k_end)]
    return cast(pd.Series, match.iloc[0])


def _selftest_segments() -> bool:
    segments, _windows = train_tables(_frames(), _pose)
    rows = {(row["video_id"], row["segment_start"]): row for row in segments.to_dict("records")}
    first, walk, short = rows[("enva/v1", 0)], rows[("enva/v1", 30)], rows[("enva/v1", 70)]
    after_ignore, before_fall = rows[("enva/v2", 4)], rows[("enva/v2", 34)]
    return _check(
        "trechos: limites, vizinhos, trechos curtos e janelas de treino por trecho",
        len(segments) == 5
        and (first["label"], first["segment_length"], first["preceding_label"], first["following_label"])
        == ("other", 30, "video_start", "walk")
        and (first["n_train_windows"], first["n_pure_train_windows"]) == (8, 8)
        and (walk["n_train_windows"], walk["n_pure_train_windows"]) == (10, 4)
        and (short["short"], short["following_label"], short["n_train_windows"], short["n_pure_train_windows"])
        == (True, "video_end", 2, 0)
        and after_ignore["preceding_label"] == "IGNORE_LABEL"
        and (before_fall["following_label"], before_fall["segment_end"]) == ("fall", 61)
        and set(segments["video_id"]) == {"enva/v1", "enva/v2"},
    )


def _selftest_short_segments() -> bool:
    # Dois trechos `other` de 10 quadros: o inicial gera janelas puras porque o
    # clip de window_frame_indices replica o quadro 0, que também é `other`; o
    # interior tem `walk` antes e nenhuma janela pura.
    labels = [OTHER] * 10 + [WALK] * 30 + [OTHER] * 10 + [WALK] * 30
    frames = pd.DataFrame([
        {"video_id": "envc/v5", "split": "train", "env": "EnvC", "subject": 5,
         "frame_index": frame_index, "label": label}
        for frame_index, label in enumerate(labels)
    ])
    n_frames = len(labels)
    pose = PoseArrays(keypoints=np.full((n_frames, 17, 3), 0.5, dtype=np.float32),
                      bbox=np.tile(np.array([0.0, 0.0, 20.0, BOX_HEIGHT], dtype=np.float32), (n_frames, 1)),
                      person_found=np.ones(n_frames, dtype=bool), k=n_frames, width=320, height=240)
    segments, windows = train_tables(frames, lambda _video_id: pose)
    rows = {row["segment_start"]: row for row in segments.to_dict("records")}
    initial, interior = rows[0], rows[40]
    label_array = np.asarray(labels)
    index = build_window_index(frames, stride=TRAIN_STRIDE)

    def expected(segment: dict) -> tuple[int, int]:
        k_ends = [int(k) for k in index["k_end"] if segment["segment_start"] <= k <= segment["segment_end"]]
        pure = [k for k in k_ends if bool((label_array[window_frame_indices(k, n_frames)] == label_array[k]).all())]
        return len(k_ends), len(pure)

    initial_windows = windows[windows["k_end"] <= initial["segment_end"]]
    return all([
        _check("trechos curtos: o inicial gera janelas puras replicadas pela borda, o interior não",
               (initial["short"], initial["preceding_label"], initial["n_train_windows"],
                initial["n_pure_train_windows"]) == (True, "video_start", 3, 3)
               and (interior["short"], interior["preceding_label"], interior["n_train_windows"],
                    interior["n_pure_train_windows"]) == (True, "walk", 3, 0)
               and set(initial_windows["purity"]) == {PURE} and set(initial_windows["edge_padding"]) == {PADDED}),
        _check("trechos curtos: contagens iguais a build_window_index e window_frame_indices",
               all((segment["n_train_windows"], segment["n_pure_train_windows"]) == expected(segment)
                   for segment in segments.to_dict("records"))),
    ])


def _selftest_alignment() -> bool:
    frames = _frames()
    _segments, windows = train_tables(frames, _pose)
    train = cast(pd.DataFrame, frames[frames["split"] == "train"])
    index = build_window_index(train, stride=TRAIN_STRIDE)
    index = cast(pd.DataFrame, index[index["label"].isin([WALK, OTHER])])
    same_windows = sorted(zip(windows["video_id"], windows["k_end"])) == sorted(zip(index["video_id"], index["k_end"]))
    padded, inner, gap, boundary = (_window(windows, "enva/v1", 8), _window(windows, "enva/v1", 52),
                                    _window(windows, "enva/v1", 48), _window(windows, "enva/v1", 32))
    expected_inner = 23 / BOX_HEIGHT

    def short_pose(video_id: str) -> PoseArrays:
        pose = _pose(video_id)
        return PoseArrays(keypoints=pose.keypoints[:-1], bbox=pose.bbox[:-1], person_found=pose.person_found[:-1],
                          k=pose.k - 1, width=pose.width, height=pose.height)

    gapped = frames.drop(index=frames.index[(frames["video_id"] == "enva/v2") & (frames["frame_index"] == 5)])
    return all([
        _check("alinhamento: janelas iguais a build_window_index em TRAIN_STRIDE, sem fall nem IGNORE_LABEL",
               same_windows and set(windows["label"]) == {"walk", "other"}),
        _check("alinhamento: janela replicada pela borda conta quadros replicados e fica sem deslocamento",
               padded["n_replicated_frames"] == 15 and abs(padded["detected_fraction"] - 8 / 24) < 1e-9
               and padded["displacement_bin"] == UNDEFINED and padded["pose_final"] == POSE_DETECTED
               and padded["purity"] == PURE),
        _check("alinhamento: deslocamento exato entre o primeiro índice e o quadro final",
               abs(inner["displacement"] - expected_inner) < 1e-6 and inner["displacement_bin"] == "0.50-1.00"
               and abs(inner["mean_keypoint_conf"] - 0.5) < 1e-9),
        _check("alinhamento: quadro sem pose dentro da janela reduz a fração de detecção",
               abs(gap["detected_fraction"] - 23 / 24) < 1e-9 and gap["displacement_bin"] == "0.50-1.00"),
        _check("alinhamento: janela que cruza a fronteira walk/other é misturada",
               boundary["purity"] == MIXED and boundary["preceding_label"] == "other"),
        _check("alinhamento: pose com K divergente da grade falha com o vídeo",
               _raises(ValueError, lambda: train_tables(frames, short_pose), "enva/v1")),
        _check("alinhamento: frame_index não contíguo falha",
               _raises(ValueError, lambda: train_tables(gapped, _pose), "contíguo")),
        _check("deslocamento: indefinido sem pose nas extremidades, com intervalo vazio ou altura nula", all([
            bbox_displacement(np.zeros((4, 4)), np.ones(4, dtype=bool), 0, 3) is None,
            bbox_displacement(_pose("enva/v1").bbox, _pose("enva/v1").person_found, 0, 23) is None,
            bbox_displacement(_pose("enva/v2").bbox, _pose("enva/v2").person_found, 5, 5) is None,
            displacement_bin(None) == UNDEFINED, displacement_bin(0.25) == "0.25-0.50",
            displacement_bin(0.2499) == "0.00-0.25", displacement_bin(1.0) == "1.00+",
        ])),
    ])


def _selftest_conservation() -> bool:
    frames = _frames()
    report, segments, windows = build_report(frames, _pose)
    block = report["conservation"]
    tampered_report, _segments, _windows = build_report(frames, _pose)
    tampered_report["summaries"]["other"]["segments"]["n_frames"] -= 1
    return all([
        _check("conservação: quadros, janelas por trecho, por sujeito e por dimensão somam os totais",
               block["ok"] and block["other"]["train_frames"] == 30 + 10 + 28
               and block["walk"]["train_frames"] == 40 + 30
               and block["other"]["train_windows"] == len(windows[windows["label"] == "other"])
               and [entry["env"] for entry in report["environments"]] == ["EnvA"]
               and block["walk"]["environment_windows"] == block["walk"]["train_windows"]
               and all(total == block["walk"]["train_windows"]
                       for total in block["walk"]["dimension_sums"].values())),
        _check("conservação: divergência de quadros falha",
               _raises(ValueError, lambda: conservation(segments, windows, frames, tampered_report["summaries"],
                                                        report["subjects"], report["environments"]), "conservação")),
        _check("conservação: pose ausente no quadro final contada à parte",
               report["summaries"]["other"]["windows"]["counts"]["pose_final"].get(POSE_MISSING, 0) == 1
               and report["summaries"]["walk"]["windows"]["counts"]["pose_final"][POSE_DETECTED]
               == block["walk"]["train_windows"]),
    ])


def _selftest_train_only() -> bool:
    frames = _frames()
    seen: list[str] = []
    report, _segments, _windows = build_report(frames, _recording_loader(seen))
    altered = frames.copy()
    held_out = altered["split"] != "train"
    altered.loc[held_out, "label"] = WALK
    altered.loc[held_out & (altered["frame_index"] % 3 == 0), "label"] = OTHER
    altered_report, _segments, _windows = build_report(altered, _recording_loader([]))
    return all([
        _check("somente train: pose lida apenas dos vídeos de train, uma vez cada",
               seen == ["enva/v1", "enva/v2"]),
        _check("somente train: rótulos de val e test não alteram o relatório",
               json.dumps(report, sort_keys=True) == json.dumps(altered_report, sort_keys=True)),
        _check("somente train: frames sem train falham",
               _raises(ValueError, lambda: build_report(cast(pd.DataFrame, frames[frames["split"] != "train"]),
                                                        _pose), "train")),
    ])


def _rule_windows(spec: list[tuple[int, int, int, int, int]]) -> pd.DataFrame:
    """(sujeito, janelas other, puras other, janelas walk, puras walk)."""
    rows = []
    for subject, n_other, pure_other, n_walk, pure_walk in spec:
        for label, n, n_pure in (("other", n_other, pure_other), ("walk", n_walk, pure_walk)):
            rows += [{"subject": subject, "label": label, "purity": PURE if i < n_pure else MIXED,
                      "pose_final": POSE_DETECTED, "displacement": 0.1, "displacement_bin": "0.00-0.25"}
                     for i in range(n)]
    return pd.DataFrame(rows)


def _r1(spec: list[tuple[int, int, int, int, int]]) -> dict:
    from gatefall.eval.analysis.other_walk_train_composition import RULES

    key, description, share, condition = RULES[0]
    return evaluate_rule(_rule_windows(spec), key, description, share, condition)


def _selftest_decision_rules() -> bool:
    from gatefall.eval.analysis.other_walk_train_composition import RULES

    strong = _r1([(1, 10, 2, 10, 8), (2, 10, 2, 10, 8), (3, 10, 2, 10, 8)])
    boundary = _r1([(1, 10, 7, 10, 8), (2, 10, 7, 10, 8), (3, 10, 7, 10, 8)])
    two_eligible = _r1([(1, 10, 2, 10, 8), (2, 10, 2, 10, 8), (3, 9, 2, 10, 8)])
    two_of_three = _r1([(1, 10, 2, 10, 8), (2, 10, 2, 10, 8), (3, 10, 8, 10, 8)])
    one_of_three = _r1([(1, 10, 0, 10, 10), (2, 10, 8, 10, 8), (3, 10, 8, 10, 8)])
    _key, _description, low_share, low_condition = RULES[1]

    def r2(low_other_every: int | None, low_walk_every: int | None) -> dict:
        windows = _rule_windows([(1, 10, 10, 10, 10), (2, 10, 10, 10, 10), (3, 10, 10, 10, 10)])
        windows["displacement"] = 0.6
        windows["displacement_bin"] = "0.50-1.00"
        for label, every in (("other", low_other_every), ("walk", low_walk_every)):
            if every is not None:
                windows.loc[windows.index[windows["label"] == label][::every], "displacement"] = 0.1
        return evaluate_rule(windows, "R2", "", low_share, low_condition)

    separated, at_threshold, overlapping = r2(1, None), r2(10, None), r2(10, 10)
    return all([
        _check("regras: diferença agregada e consistente em todos os sujeitos é sustentada",
               strong["supported"] and abs(strong["delta"] + 0.6) < 1e-9 and strong["n_agreeing_subjects"] == 3),
        _check("regras: Δ exatamente no limiar −0,10 conta como condição atendida",
               boundary["pooled_condition_met"] and boundary["supported"]),
        _check("regras: menos de 3 sujeitos elegíveis não sustentam a regra",
               two_eligible["n_eligible_subjects"] == 2 and not two_eligible["supported"]),
        _check("regras: concordância de 2 de 3 sustenta; 1 de 3 não",
               two_of_three["supported"] and two_of_three["n_agreeing_subjects"] == 2
               and one_of_three["pooled_condition_met"] and not one_of_three["supported"]),
        _check("regras: R2 rejeita separação clara e Δ exatamente +0,10; sustenta sobreposição",
               not separated["pooled_condition_met"] and abs(separated["delta"] - 1.0) < 1e-9
               and abs(at_threshold["delta"] - 0.1) < 1e-9 and not at_threshold["pooled_condition_met"]
               and overlapping["supported"] and abs(overlapping["delta"]) < 1e-9),
        _check("regras: veredito depende só das regras sustentadas",
               decide([strong, one_of_three])["verdict"] == SUPPORTED_VERDICT
               and decide([two_eligible, one_of_three])["verdict"] == UNSUPPORTED_VERDICT
               and decide([strong])["supported_rules"] == [strong["rule"]]),
    ])


def _write_inputs(root: Path, skip: frozenset[str] = frozenset()) -> tuple[Path, Path]:
    frames_path = root / "processed" / "frames.parquet"
    frames_path.parent.mkdir(parents=True)
    _frames().to_parquet(frames_path)
    pose_root = root / "pose"
    for video_id, split, _env, _subject, _labels in VIDEO_SPEC:
        if split != "train" or video_id in skip:
            continue
        pose = _pose(video_id)
        path = pose_path(video_id, pose_root=pose_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("keypoints", data=pose.keypoints)
            h5_file.create_dataset("bbox", data=pose.bbox)
            h5_file.create_dataset("person_found", data=pose.person_found)
            h5_file.attrs["K"] = pose.k
            h5_file.attrs["width"] = pose.width
            h5_file.attrs["height"] = pose.height
    return frames_path, pose_root


def _selftest_end_to_end_read_only() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames_path, pose_root = _write_inputs(root / "inputs")
        before = _tree_hashes(root / "inputs")
        output_dir = root / "out"
        written = run_composition(frames_path, pose_root, output_dir)
        first = _tree_hashes(output_dir)
        skipped = not run_composition(frames_path, pose_root, output_dir)
        unchanged = _tree_hashes(output_dir) == first
        forced = run_composition(frames_path, pose_root, output_dir, force=True)
        deterministic = _tree_hashes(output_dir) == first
        report = json.loads((output_dir / COMPOSITION_JSON_FILE).read_text(encoding="utf-8"))
        outputs = {path.name for path in output_dir.iterdir()}
        inputs_unchanged = _tree_hashes(root / "inputs") == before
        protected = [frames_path.parent, pose_root, pose_root / "enva", REFERENCE_RUN_ROOT / "x",
                     REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "docs" / "x", REPOSITORY_ROOT / "data"]
        refused = all(_raises(ValueError, lambda path=path: run_composition(frames_path, pose_root, path),
                              "protegido") for path in protected)
        versioned = _raises(ValueError, lambda: run_composition(frames_path, pose_root,
                                                                REPOSITORY_ROOT / "composition_out"), "runs/local")
        missing_root = root / "missing"
        missing_frames, missing_pose = _write_inputs(missing_root, skip=frozenset({"enva/v2"}))
        missing_out = root / "missing_out"
        missing = _raises(FileNotFoundError, lambda: run_composition(missing_frames, missing_pose, missing_out),
                          "enva/v2")
    return all([
        _check("ponta a ponta: grava JSON, resumo e CSVs de trechos e janelas",
               written and outputs == {COMPOSITION_JSON_FILE, COMPOSITION_SUMMARY_FILE, COMPOSITION_SEGMENTS_FILE,
                                       COMPOSITION_WINDOWS_FILE}
               and report["split"] == "train" and not report["canonical"] and report["read_only"]),
        _check("ponta a ponta: SHA-256 só de frames.parquet e da pose de train, sem pose de val/test",
               len(report["input_sha256"]) == 3
               and not any("envb" in path for path in report["input_sha256"])),
        _check("ponta a ponta: sem --force preserva e com --force regrava bytes idênticos",
               skipped and unchanged and forced and deterministic),
        _check("ponta a ponta: entradas inalteradas", inputs_unchanged),
        _check("saída: referências, src, data, docs, frames e pose recusados", refused),
        _check("saída: diretório versionado fora de runs/local recusado",
               versioned and not (REPOSITORY_ROOT / "composition_out").exists()),
        _check("entrada: pose de train ausente falha sem gravar", missing and not missing_out.exists()),
    ])


SELFTESTS: tuple[Callable[[], bool], ...] = (
    _selftest_segments,
    _selftest_short_segments,
    _selftest_alignment,
    _selftest_conservation,
    _selftest_train_only,
    _selftest_decision_rules,
    _selftest_end_to_end_read_only,
)


def run_selftest() -> None:
    results = [selftest() for selftest in SELFTESTS]
    if not all(results):
        sys.exit(1)
    print("selftest da composição de treino other/walk: todos os casos passaram")
