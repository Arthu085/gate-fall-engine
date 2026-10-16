import json
import random
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from gatefall.config import WINDOW_FRAMES
from gatefall.data.windowing import window_frame_indices
from gatefall.eval.analysis import subject_fold_dropout_robustness as robustness
from gatefall.eval.analysis.subject_fold_dropout_robustness import (
    DIAGNOSTICS_DIR,
    MODEL_VARIANTS,
    REPORT_JSON_FILE,
    REPORT_SUMMARY_FILE,
    UNMASKED,
    build_retained_inputs,
    diagnostics_output_dir,
    evaluation_offset,
    run_analysis,
    validate_diagnostics_output,
)
from gatefall.hashing import sha256_file
from gatefall.pose.loading import PoseArrays
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.internal_folds.selftests.fixtures import _check, raises
from gatefall.train.internal_folds.training import FOLD_RUN_FILE
from gatefall.train.subject_folds.cli import PoseInputs, run_prepare, run_train
from gatefall.train.subject_folds.detection_dropout import (
    DROPOUT_PROBABILITY,
    MASKED_FRAMES,
    masked_window_features,
    valid_offsets,
)
from gatefall.train.subject_folds.dropout_summary import write_dropout_summary
from gatefall.train.subject_folds.folds import (
    BASELINE_VARIANT,
    DETECTION_DROPOUT_VARIANT,
    FOLDS_FILE,
    SubjectFoldPartition,
    build_subject_fold_partitions,
    fold_run_dir,
    stats_dir,
    summary_dir,
)
from gatefall.train.subject_folds.selftests.fixtures import (
    OFFICIAL_HELD_OUT_VIDEOS,
    TRAIN_VIDEOS,
    RawRecordingPose,
    clean_pose_features,
    official_train_frames,
    pose_features,
    synthetic_raw_pose,
    synthetic_workspace,
)
from gatefall.train.subject_folds.stats import POSE_STATS_FILE
from gatefall.train.subject_folds.summary import write_summary

SEED = 42
Paths = tuple[Path, Path, Path, Path]


def _partitions() -> list[SubjectFoldPartition]:
    return build_subject_fold_partitions(official_train_frames())


def _rng_states() -> tuple[Any, ...]:
    return np.random.get_state(), torch.get_rng_state().clone(), random.getstate()


def _same_rng_states(before: tuple[Any, ...], after: tuple[Any, ...]) -> bool:
    numpy_before, torch_before, python_before = before
    numpy_after, torch_after, python_after = after
    return (
        numpy_before[0] == numpy_after[0]
        and np.array_equal(numpy_before[1], numpy_after[1])
        and numpy_before[2:] == numpy_after[2:]
        and torch.equal(torch_before, torch_after)
        and python_before == python_after
    )


def _selftest_mask_determinism() -> bool:
    identities = [(video_id, k_end) for video_id in TRAIN_VIDEOS for k_end in range(40)]
    states = _rng_states()
    forward = {identity: evaluation_offset(*identity) for identity in identities}
    backward = {identity: evaluation_offset(*identity) for identity in reversed(identities)}
    rng_isolated = _same_rng_states(states, _rng_states())

    draws = [evaluation_offset(f"Env/v{video}", k_end) for video in range(200) for k_end in range(WINDOW_FRAMES, 120)]
    offsets = [offset for offset in draws if offset is not None]
    rate = len(offsets) / len(draws)
    distribution_ok = (
        abs(rate - DROPOUT_PROBABILITY) < 0.015
        and set(offsets) == set(range(WINDOW_FRAMES - MASKED_FRAMES + 1))
        and all(evaluation_offset(f"Env/v{video}", k_end) is None for video in range(200) for k_end in (0, 1))
        and all(offset in valid_offsets(k_end) for (_, k_end), offset in forward.items() if offset is not None)
    )
    by_video = [[forward[(video_id, k_end)] for k_end in range(40)] for video_id in TRAIN_VIDEOS]
    identity_sensitive = len({tuple(plan) for plan in by_video}) > 1

    partition = _partitions()[0]
    first = build_retained_inputs(partition, RawRecordingPose().inputs())
    second = build_retained_inputs(partition, RawRecordingPose().inputs())
    reproducible = first.digests() == second.digests() and bool(first.masked.any())
    return _check(
        "a máscara de avaliação depende só de (video_id, k_end) e da seed fixa: é reprodutível, independente da "
        "ordem, não consome o RNG global, muda entre vídeos, mascara ~20% das janelas em posições válidas "
        "uniformes e gera as mesmas entradas ao repetir a dobra",
        forward == backward and rng_isolated and distribution_ok and identity_sensitive and reproducible,
    )


def _selftest_raw_pose_consistency() -> bool:
    partition = _partitions()[1]
    inputs = build_retained_inputs(partition, RawRecordingPose().inputs())
    consistent = True
    for index, (video_id, k_end, offset) in enumerate(zip(inputs.video_ids, inputs.k_ends, inputs.offsets)):
        clean = clean_pose_features(video_id)[window_frame_indices(int(k_end), 40)]
        consistent &= np.array_equal(inputs.clean[index], clean)
        if offset == UNMASKED:
            consistent &= np.array_equal(inputs.degraded[index], clean)
        else:
            expected = masked_window_features(synthetic_raw_pose(video_id), int(k_end), int(offset))
            consistent &= np.array_equal(inputs.degraded[index], expected)
            frames = window_frame_indices(int(k_end), 40)
            before_mask = frames < frames[offset]
            consistent &= np.array_equal(inputs.degraded[index][before_mask], clean[before_mask])

    def with_loaders(loader: Callable[[str], np.ndarray], raw: Callable[[str], PoseArrays] | None) -> Callable[[], object]:
        return lambda: build_retained_inputs(partition, PoseInputs(loader=loader, sha256="0" * 64, raw_loader=raw))

    def short_pose(video_id: str) -> PoseArrays:
        pose = synthetic_raw_pose(video_id)
        return replace(pose, keypoints=pose.keypoints[:-1], bbox=pose.bbox[:-1],
                       person_found=pose.person_found[:-1], k=pose.k - 1)

    refused = (
        raises(ValueError, with_loaders(pose_features, synthetic_raw_pose), "divergem das features limpas")
        and raises(ValueError, with_loaders(clean_pose_features, short_pose), "quadros")
        and raises(ValueError, with_loaders(clean_pose_features, None), "pose bruta")
    )
    return _check(
        "as janelas limpas são as do engine; as degradadas são a fatia causal da pose bruta mascarada, iguais às "
        "limpas antes da máscara e fora das janelas sorteadas; pose bruta inconsistente com as features limpas, "
        "com outro número de quadros ou ausente é recusada",
        consistent and bool(inputs.masked.any()) and refused,
    )


def _selftest_subject_isolation() -> bool:
    ok = True
    for partition in _partitions():
        recorder = RawRecordingPose()
        inputs = build_retained_inputs(partition, recorder.inputs())
        retained = set(partition.retained_video_ids)
        ok &= (
            set(recorder.calls) == retained
            and set(recorder.raw_calls) == retained
            and set(inputs.video_ids) == retained
            and set(inputs.subjects.tolist()) <= set(partition.retained_subjects)
        )
    return _check(
        "cada dobra lê features limpas e pose bruta só dos vídeos do próprio grupo retido; nenhum vídeo de "
        "desenvolvimento, de outra dobra ou de val/test oficial entra na avaliação",
        ok,
    )


def _train_all(paths: Paths) -> None:
    root, frames_path, manifest_path, pose_root = paths
    run_prepare(root, frames_path, manifest_path, pose_root, False, RawRecordingPose().inputs())
    folds = [partition.fold for partition in _partitions()]
    for variant in MODEL_VARIANTS:
        run_train(root, frames_path, manifest_path, pose_root, folds, ["A"], [SEED], False,
                  RawRecordingPose().inputs(), variant)
    write_summary(root, force=False)
    write_dropout_summary(root, force=False)


def _analyze(paths: Paths, force: bool = False, recorder: RawRecordingPose | None = None) -> bool:
    root, frames_path, manifest_path, pose_root = paths
    return run_analysis(root, frames_path, manifest_path, pose_root, force, (recorder or RawRecordingPose()).inputs())


def _tree_digest(root: Path, exclude: Path | None = None) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and (exclude is None or not path.is_relative_to(exclude))
    }


def _report(root: Path) -> dict[str, Any]:
    return json.loads((diagnostics_output_dir(root) / REPORT_JSON_FILE).read_text(encoding="utf-8"))


@contextmanager
def _recording_predict() -> Iterator[list[tuple[bytes, np.ndarray]]]:
    calls: list[tuple[bytes, np.ndarray]] = []
    real = robustness.predict

    def recording(model: Any, windows: np.ndarray, *args: Any) -> np.ndarray:
        calls.append((next(model.parameters()).detach().cpu().numpy().tobytes(), windows))
        return real(model, windows, *args)

    with patch.object(robustness, "predict", recording):
        yield calls


def _identical_inputs_check(paths: Paths) -> bool:
    root = paths[0]
    with _recording_predict() as calls:
        _analyze(paths, force=True)
    report = _report(root)
    ok = len(calls) == 4 * len(report["folds"])
    for fold_index, block in enumerate(report["folds"]):
        (b_model, b_clean), (_, b_degraded), (d_model, d_clean), (_, d_degraded) = calls[4 * fold_index:4 * fold_index + 4]
        models = block["models"]
        ok &= (
            b_model != d_model
            and b_clean is d_clean
            and b_degraded is d_degraded
            and not np.array_equal(b_clean, b_degraded)
            and models[BASELINE_VARIANT]["inputs_sha256"] == models[DETECTION_DROPOUT_VARIANT]["inputs_sha256"]
            == block["mask_plan"]["sha256"]
            and models[BASELINE_VARIANT]["clean"]["support"] == models[DETECTION_DROPOUT_VARIANT]["degraded"]["support"]
        )
    return _check(
        "os dois checkpoints de cada dobra recebem os mesmos arrays de rótulos, janelas limpas e janelas degradadas, "
        "materializados uma vez, com os mesmos hashes de máscara e de entrada",
        ok,
    )


def _determinism_check(paths: Paths) -> bool:
    root = paths[0]
    path = diagnostics_output_dir(root) / REPORT_JSON_FILE
    _analyze(paths, force=True)
    first = path.read_bytes()
    _analyze(paths, force=True)
    repeated = path.read_bytes() == first
    with patch.object(robustness, "MODEL_VARIANTS", tuple(reversed(MODEL_VARIANTS))):
        _analyze(paths, force=True)
    reordered = _report(root)
    original = json.loads(first)
    order_independent = all(
        block["mask_plan"] == reference["mask_plan"]
        and all(block["models"][variant]["degraded"] == reference["models"][variant]["degraded"]
                for variant in MODEL_VARIANTS)
        for block, reference in zip(reordered["folds"], original["folds"])
    )
    _analyze(paths, force=True)
    return _check(
        "repetir a análise grava o mesmo JSON byte a byte, e inverter a ordem dos modelos não muda a máscara nem as "
        "métricas degradadas",
        repeated and order_independent and path.read_bytes() == first,
    )


def _report_check(paths: Paths) -> bool:
    root = paths[0]
    report = _report(root)
    ok = (
        report["canonical"] is False
        and report["read_only"] is True
        and report["pooled_score"] is None
        and report["pooled_difference"] is None
        and report["significance_test"] is None
        and report["official_splits_read"] == ["train"]
        and [block["fold"] for block in report["folds"]] == [partition.fold for partition in _partitions()]
    )
    for block in report["folds"]:
        models = block["models"]
        losses = {}
        for variant in MODEL_VARIANTS:
            model = models[variant]
            run_dir = fold_run_dir(root, block["fold"], "A", SEED, variant)
            stored = json.loads((run_dir / "metrics.json").read_text())["final"]["val"]
            change = model["degraded"]["macro_f1_restricted"] - model["clean"]["macro_f1_restricted"]
            losses[variant] = -change
            ok &= (
                model["clean"]["confusion_matrix"] == stored["confusion_matrix"]
                and model["clean"]["support"] == stored["support"]
                and model["clean"]["macro_f1_restricted"] == stored["macro_f1_restricted"]
                and model["change"]["macro_f1_restricted"] == change
                and model["affected"]["windows"] == block["mask_plan"]["masked_windows"] > 0
                and model["unmasked_prediction_changes"] == 0
            )
        difference = block["paired"]["loss_difference"]["macro_f1_restricted"]
        ok &= difference == losses[BASELINE_VARIANT] - losses[DETECTION_DROPOUT_VARIANT]
    return _check(
        "o relatório reproduz final.val de cada run na condição limpa, relata por dobra e modelo a mudança "
        "degradado − limpo e a diferença pareada das perdas, descreve o subconjunto mascarado e não agrega dobras "
        "nem testa significância",
        ok,
    )


def _clean_reproduction_check(paths: Paths) -> bool:
    root = paths[0]
    output = diagnostics_output_dir(root)
    shutil.rmtree(output)
    real = robustness.predict

    def flipped(model: Any, windows: np.ndarray, *args: Any) -> np.ndarray:
        predictions = real(model, windows, *args)
        predictions[0] = (predictions[0] + 1) % 10
        return predictions

    with patch.object(robustness, "predict", flipped):
        refused = raises(RuntimeError, lambda: _analyze(paths), "não reproduz final.val")
    not_written = not output.exists()
    _analyze(paths)
    return _check(
        "se a avaliação limpa diverge de final.val em uma única janela, a análise falha antes de calcular a "
        "condição degradada e não grava nada",
        refused and not_written,
    )


@contextmanager
def _tampered(path: Path, mutate: Callable[[bytes], bytes]) -> Iterator[None]:
    original = path.read_bytes()
    path.write_bytes(mutate(original))
    try:
        yield
    finally:
        path.write_bytes(original)


def _json_edit(edit: Callable[[dict], None]) -> Callable[[bytes], bytes]:
    def mutate(raw: bytes) -> bytes:
        data = json.loads(raw)
        edit(data)
        return json.dumps(data, indent=2).encode()

    return mutate


def _provenance_check(paths: Paths) -> bool:
    root = paths[0]
    output = diagnostics_output_dir(root)
    shutil.rmtree(output)
    fold = _partitions()[0].fold
    baseline_dir = fold_run_dir(root, fold, "A", SEED)
    dropout_dir = fold_run_dir(root, fold, "A", SEED, DETECTION_DROPOUT_VARIANT)
    cases: dict[str, tuple[Path, Callable[[bytes], bytes], type[Exception], str]] = {
        "checkpoint do controle truncado": (baseline_dir / "checkpoint.pt", lambda raw: raw[:-16], RuntimeError,
                                            "run da dobra inválido"),
        "checkpoint 4C trocado": (dropout_dir / "checkpoint.pt",
                                  lambda raw: (baseline_dir / "checkpoint.pt").read_bytes(), RuntimeError,
                                  "run da dobra inválido"),
        "controle da 4C": (dropout_dir / FOLD_RUN_FILE,
                           _json_edit(lambda data: data["control"].update(fold_run_sha256="0" * 64)), RuntimeError,
                           "control"),
        "intervenção da 4C": (dropout_dir / FOLD_RUN_FILE,
                              _json_edit(lambda data: data["intervention"].update(probability=0.5)), RuntimeError,
                              "intervention"),
        "estatística da dobra": (stats_dir(root, fold) / POSE_STATS_FILE, lambda raw: raw + b" ", ValueError,
                                 "proveniência diverge"),
        "folds.json": (root / FOLDS_FILE, _json_edit(lambda data: data.update(membership_sha256="0" * 64)),
                       ValueError, "membership_sha256"),
    }
    failed = []
    for name, (path, mutate, error, fragment) in cases.items():
        with _tampered(path, mutate):
            if not raises(error, lambda: _analyze(paths), fragment) or output.exists():
                failed.append(name)
    root_, frames_path, manifest_path, pose_root = paths
    if not raises(ValueError, lambda: run_analysis(root_, frames_path, manifest_path, pose_root, False,
                                                   RawRecordingPose(sha256="1" * 64).inputs()),
                  "HDF5 de pose mudaram") or output.exists():
        failed.append("hash da pose")
    if failed:
        print(f"  proveniências corrompidas aceitas: {failed}")
    _analyze(paths)
    return _check(
        "checkpoint truncado ou trocado, fold_run.json da Etapa 4C com controle ou intervenção adulterados, "
        "estatística da dobra alterada, folds.json divergente e hash de pose diferente falham sem gravar nada",
        not failed,
    )


def _protected_paths_check(paths: Paths) -> bool:
    root, frames_path, manifest_path, pose_root = paths
    output = diagnostics_output_dir(root)
    before = _tree_digest(root, exclude=root / DIAGNOSTICS_DIR)
    inputs_before = {path: sha256_file(path) for path in (frames_path, manifest_path)}
    recorder = RawRecordingPose()
    _analyze(paths, force=True, recorder=recorder)
    summaries = len([path for path in before if Path(path).is_relative_to(summary_dir(root))]) == 4
    preserved = summaries and _tree_digest(root, exclude=root / DIAGNOSTICS_DIR) == before and inputs_before == {
        path: sha256_file(path) for path in (frames_path, manifest_path)
    }
    published = sorted(path.name for path in (root / DIAGNOSTICS_DIR).rglob("*") if path.is_file())
    only_outputs = published == sorted([REPORT_JSON_FILE, REPORT_SUMMARY_FILE])
    isolated = set(recorder.raw_calls) == set(TRAIN_VIDEOS) and not (
        set(recorder.calls) | set(recorder.raw_calls)
    ) & set(OFFICIAL_HELD_OUT_VIDEOS)
    report_before = (output / REPORT_JSON_FILE).read_bytes()
    skipped = _analyze(paths) is False and (output / REPORT_JSON_FILE).read_bytes() == report_before

    baseline_config = fold_run_dir(root, _partitions()[0].fold, "A", SEED) / "config.yaml"
    real = robustness.predict
    touched: list[bool] = []

    def touching(model: Any, windows: np.ndarray, *args: Any) -> np.ndarray:
        if not touched:
            touched.append(True)
            baseline_config.write_bytes(baseline_config.read_bytes() + b"\n")
        return real(model, windows, *args)

    original_config = baseline_config.read_bytes()
    try:
        with patch.object(robustness, "predict", touching):
            changed_refused = raises(RuntimeError, lambda: _analyze(paths, force=True), "mudaram")
    finally:
        baseline_config.write_bytes(original_config)
    unpublished = (output / REPORT_JSON_FILE).read_bytes() == report_before

    roots_refused = all(
        raises(ValueError, lambda candidate=candidate: validate_diagnostics_output(
            candidate, frames_path, manifest_path, pose_root))
        for candidate in (REFERENCE_RUN_ROOT, REPOSITORY_ROOT / "runs/local/le2i", REPOSITORY_ROOT / "src", pose_root)
    )
    return _check(
        "a análise grava só o JSON e o Markdown em diagnostics/, preserva byte a byte folds.json, estatísticas, "
        "runs 4B/4C, resumos e entradas, lê pose bruta só do train oficial, não sobrescreve sem --force, não "
        "publica se um artefato de entrada muda durante a análise e recusa raízes protegidas ou fora das dobras",
        preserved and only_outputs and isolated and skipped and changed_refused and unpublished and roots_refused,
    )


def _selftest_workspace() -> list[bool]:
    with synthetic_workspace() as paths:
        _train_all(paths)
        root = paths[0]
        missing = raises((FileNotFoundError, RuntimeError), lambda: _analyze((root.parent / "absent", *paths[1:])))
        written = _analyze(paths)
        results = [_check("a análise roda sobre os seis runs sintéticos e falha sem as dobras", written and missing)]
        results.append(_identical_inputs_check(paths))
        results.append(_determinism_check(paths))
        results.append(_report_check(paths))
        results.append(_clean_reproduction_check(paths))
        results.append(_provenance_check(paths))
        results.append(_protected_paths_check(paths))
        return results


def run_selftest() -> bool:
    results = [
        _selftest_mask_determinism(),
        _selftest_raw_pose_consistency(),
        _selftest_subject_isolation(),
        *_selftest_workspace(),
    ]
    return all(results)
