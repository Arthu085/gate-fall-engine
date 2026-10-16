import hashlib
import json
import random
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch
import yaml

from gatefall.config import TARGET_FPS, TRAIN_STRIDE, WINDOW_FRAMES
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import window_frame_indices
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features_from_arrays, feature_blocks
from gatefall.pose.loading import first_observed_index, load_pose
from gatefall.pose.selftests.fixtures import _write_synthetic_pose
from gatefall.train.baseline_a import engine
from gatefall.train.internal_folds.folds import DEVELOPMENT_SPLIT
from gatefall.train.internal_folds.selftests.fixtures import _check, raises
from gatefall.train.internal_folds.training import FOLD_RUN_FILE
from gatefall.train.shared.tcn import TCNClassifier
from gatefall.train.subject_folds import training
from gatefall.train.subject_folds.cli import run_prepare, run_train
from gatefall.train.subject_folds.detection_dropout import (
    DROPOUT_PROBABILITY,
    MASKED_FRAMES,
    DetectionDropoutWindowDataset,
    causal_prefix,
    draw_dropout,
    intervention_record,
    mask_person_detections,
    masked_window_features,
    valid_offsets,
)
from gatefall.train.subject_folds.dropout_summary import (
    FOCUS_CLASSES,
    SUMMARY_JSON_FILE,
    build_dropout_summary,
    evaluate_decision,
    write_dropout_summary,
)
from gatefall.train.subject_folds.folds import (
    BASELINE_VARIANT,
    DETECTION_DROPOUT_VARIANT,
    FOLDS_FILE,
    RESTRICTED_LABELS,
    SubjectFoldPartition,
    build_subject_fold_partitions,
    fold_run_dir,
    stats_dir,
    summary_dir,
)
from gatefall.train.subject_folds.selftests.fixtures import (
    LATE_ACQUISITION_VIDEO,
    OFFICIAL_HELD_OUT_VIDEOS,
    TRAIN_VIDEOS,
    RawRecordingPose,
    clean_pose_features,
    official_train_frames,
    synthetic_raw_pose,
    synthetic_workspace,
)
from gatefall.train.subject_folds.summary import write_summary

SEED = 42
DT = 1.0 / TARGET_FPS
_BLOCKS = {name: (start, end) for name, start, end in feature_blocks()}


def _block(features: np.ndarray, name: str) -> np.ndarray:
    start, end = _BLOCKS[name]
    return features[..., start:end]


def _observed_gap_start(person_found: np.ndarray) -> int:
    """Primeiro quadro i com i e i+4 observados, para mascarar i+1..i+3 entre duas detecções reais."""
    for frame in range(len(person_found) - MASKED_FRAMES - 1):
        if person_found[frame] and person_found[frame + MASKED_FRAMES + 1]:
            return frame
    raise AssertionError("pose sintética sem duas detecções separadas por três quadros")


def _selftest_raw_masking() -> bool:
    pose = synthetic_raw_pose(TRAIN_VIDEOS[1])
    originals = (pose.keypoints.copy(), pose.bbox.copy(), pose.person_found.copy())
    anchor = _observed_gap_start(pose.person_found)
    frames = np.arange(anchor + 1, anchor + 1 + MASKED_FRAMES)
    masked = mask_person_detections(pose, frames)
    others = np.setdiff1d(np.arange(pose.k), frames)
    copy_ok = (
        all(np.array_equal(a, b) for a, b in zip(originals, (pose.keypoints, pose.bbox, pose.person_found)))
        and not np.shares_memory(masked.keypoints, pose.keypoints)
        and not np.shares_memory(masked.person_found, pose.person_found)
        and not masked.person_found[frames].any()
        and not masked.keypoints[frames].any()
        and not masked.bbox[frames].any()
        and np.array_equal(masked.keypoints[others], pose.keypoints[others])
        and np.array_equal(masked.person_found[others], pose.person_found[others])
    )

    with tempfile.TemporaryDirectory() as directory:
        genuine_found = pose.person_found.copy()
        genuine_found[frames] = False
        keypoints = pose.keypoints.copy()
        bbox = pose.bbox.copy()
        keypoints[frames] = 0.0
        bbox[frames] = 0.0
        _write_synthetic_pose(
            Path(directory), "Env/genuine", pose.k, keypoints=keypoints, bbox=bbox, person_found=genuine_found
        )
        genuine = load_pose("Env/genuine", pose_root=Path(directory))
    genuine = replace(genuine, width=pose.width, height=pose.height)
    masked_features = build_pose_features_from_arrays(masked)[0]
    clean = build_pose_features_from_arrays(pose)[0]
    genuine_ok = np.array_equal(masked_features, build_pose_features_from_arrays(genuine)[0])

    reappear = anchor + MASKED_FRAMES + 1
    xy = _block(masked_features, "kp_xy")
    semantics_ok = (
        not _block(masked_features, "kp_conf")[frames].any()
        and all(np.array_equal(xy[frame], _block(clean, "kp_xy")[anchor]) for frame in frames)
        and not _block(masked_features, "kp_velocity")[frames].any()
        and np.allclose(
            _block(masked_features, "kp_velocity")[reappear],
            (xy[reappear] - xy[anchor]) / np.float32((MASKED_FRAMES + 1) * DT),
            atol=1e-4,
        )
        and np.array_equal(masked_features[:anchor + 1], clean[:anchor + 1])
    )
    return _check(
        "a máscara copia PoseArrays sem tocar a pose de origem, grava a ausência no formato da extração "
        "(person_found falso, keypoints e bbox zerados) e gera as mesmas 134 features de uma perda real lida de "
        "HDF5: confiança zero, forward-fill da última observação e dt efetivo acumulado na reaparição",
        copy_ok and genuine_ok and semantics_ok,
    )


def _selftest_causal_consistency() -> bool:
    prefix_ok = True
    window_ok = True
    for video_id in TRAIN_VIDEOS:
        pose = synthetic_raw_pose(video_id)
        clean = build_pose_features_from_arrays(pose)[0]
        for k_end in range(pose.k):
            prefix = build_pose_features_from_arrays(causal_prefix(pose, k_end))[0]
            prefix_ok &= np.array_equal(prefix, clean[: k_end + 1])
            indices = window_frame_indices(k_end, pose.k)
            for offset in valid_offsets(k_end):
                masked_frames = indices[offset : offset + MASKED_FRAMES]
                full = build_pose_features_from_arrays(mask_person_detections(pose, masked_frames))[0]
                before = indices < masked_frames[0]
                window = masked_window_features(pose, k_end, offset)
                window_ok &= (
                    np.array_equal(window, full[indices])
                    and np.array_equal(window[before], clean[indices[before]])
                    and bool(np.all(np.diff(masked_frames) == 1))
                    and int(masked_frames[0]) == k_end - WINDOW_FRAMES + 1 + offset
                    and int(masked_frames[0]) >= 0
                )

    pose = synthetic_raw_pose(TRAIN_VIDEOS[1])
    clean = build_pose_features_from_arrays(pose)[0]
    k_end = 35
    start = k_end - WINDOW_FRAMES + 1
    context_found = first_observed_index(pose.person_found) < start
    boundary = masked_window_features(pose, k_end, 0)
    boundary_ok = np.array_equal(
        _block(boundary, "kp_xy")[:MASKED_FRAMES],
        np.repeat(_block(clean, "kp_xy")[start - 1 : start], MASKED_FRAMES, axis=0),
    )

    late = synthetic_raw_pose(LATE_ACQUISITION_VIDEO)
    first = first_observed_index(late.person_found)
    k_end = 12
    offset = first - (k_end - WINDOW_FRAMES + 1)
    acquisition = masked_window_features(late, k_end, offset)
    indices = window_frame_indices(k_end, late.k)
    shifted_first = first + MASKED_FRAMES
    acquisition_row = int(np.flatnonzero(indices == shifted_first)[0])
    acquisition_ok = (
        bool(late.person_found[shifted_first])
        and not acquisition[indices <= first + MASKED_FRAMES - 1].any()
        and not _block(acquisition, "kp_velocity")[acquisition_row].any()
        and not _block(acquisition, "kp_acceleration")[acquisition_row + 1].any()
    )
    edges_ok = (
        not valid_offsets(0)
        and not valid_offsets(1)
        and list(valid_offsets(2)) == [WINDOW_FRAMES - MASKED_FRAMES]
        and list(valid_offsets(WINDOW_FRAMES - 1)) == list(range(WINDOW_FRAMES - MASKED_FRAMES + 1))
        and raises(ValueError, lambda: masked_window_features(pose, 5, 0), "inválido")
    )
    return _check(
        "as features são causais (o prefixo até k_end reproduz as linhas limpas), a janela mascarada é a fatia da "
        "pose mascarada no vídeo inteiro, as linhas de quadros anteriores à máscara ficam limpas, o forward-fill usa a "
        "observação anterior à janela, a máscara sobre a primeira detecção estende o trecho zerado e só "
        "posições com três quadros reais distintos são sorteáveis",
        prefix_ok and window_ok and context_found and boundary_ok and acquisition_ok and edges_ok,
    )


def _rng_states() -> tuple[tuple[Any, ...], torch.Tensor, object]:
    return cast(tuple[Any, ...], np.random.get_state()), torch.get_rng_state().clone(), random.getstate()


def _same_rng_states(
    before: tuple[tuple[Any, ...], torch.Tensor, object], after: tuple[tuple[Any, ...], torch.Tensor, object]
) -> bool:
    (numpy_before, torch_before, python_before), (numpy_after, torch_after, python_after) = before, after
    return (
        numpy_before[0] == numpy_after[0]
        and np.array_equal(numpy_before[1], numpy_after[1])
        and numpy_before[2:] == numpy_after[2:]
        and torch.equal(torch_before, torch_after)
        and python_before == python_after
    )


def _dropout_dataset(partition: SubjectFoldPartition, seed: int) -> DetectionDropoutWindowDataset:
    return DetectionDropoutWindowDataset(
        partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE, clean_pose_features, synthetic_raw_pose, seed
    )


def _selftest_deterministic_augmentation() -> bool:
    partition = build_subject_fold_partitions(official_train_frames())[0]
    clean = PoseWindowDataset(partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE, clean_pose_features)
    forward = _dropout_dataset(partition, SEED)
    backward = _dropout_dataset(partition, SEED)
    other_seed = _dropout_dataset(partition, SEED + 1)
    indices = list(range(len(clean)))
    states = _rng_states()
    first: dict[tuple[int, int], np.ndarray] = {}
    second: dict[tuple[int, int], np.ndarray] = {}
    changed = 0
    for visit in range(3):
        for index in indices:
            first[(index, visit)] = forward[index][0]
        for index in reversed(indices):
            second[(index, visit)] = backward[index][0]
        changed += sum(not np.array_equal(other_seed[index][0], first[(index, visit)]) for index in indices)
    rng_isolated = _same_rng_states(states, _rng_states())
    order_independent = all(np.array_equal(first[key], second[key]) for key in first)

    windows_ok = True
    applied = 0
    for (index, visit), window in first.items():
        _, label, (video_id, k_end) = clean[index]
        offset = draw_dropout(SEED, index, visit, k_end)
        if offset is None:
            windows_ok &= np.array_equal(window, clean[index][0])
        else:
            applied += 1
            windows_ok &= np.array_equal(window, masked_window_features(synthetic_raw_pose(video_id), k_end, offset))
        windows_ok &= forward.label_at(index) == label
    clean_cache_ok = all(
        np.array_equal(clean._features_for_video(video_id), clean_pose_features(video_id))
        for video_id in partition.development_video_ids
    )

    draws = [draw_dropout(SEED, index, visit, WINDOW_FRAMES + 5) for index in range(4000) for visit in range(5)]
    offsets = [offset for offset in draws if offset is not None]
    rate = len(offsets) / len(draws)
    distribution_ok = (
        abs(rate - DROPOUT_PROBABILITY) < 0.015
        and set(offsets) == set(range(WINDOW_FRAMES - MASKED_FRAMES + 1))
        and all(draw_dropout(SEED, index, 0, 1) is None for index in range(200))
        and applied > 0
    )
    return _check(
        "o sorteio por (seed, janela, visita) é reprodutível e independente da ordem de leitura, não consome o RNG "
        "do NumPy, do PyTorch nem do random, muda com a seed, aplica a máscara em ~20% das janelas com posição "
        "uniforme entre as válidas, devolve a janela limpa quando não sorteia e não altera o cache de features "
        "limpas",
        rng_isolated and order_independent and changed > 0 and windows_ok and clean_cache_ok and distribution_ok,
    )


def _selftest_decision_rule() -> bool:
    cases = {
        (0.03, 0.01, -0.01): True,
        (0.03, 0.01, -0.02): True,
        (0.03, 0.01, -0.021): False,
        (0.03, 0.0, -0.01): False,
        (0.03, -0.01, -0.01): False,
        (0.01, 0.02, 0.03): True,
    }
    wrong = [
        deltas for deltas, expected in cases.items() if evaluate_decision(list(deltas))["criterion_met"] != expected
    ]
    if wrong:
        print(f"  decisões divergentes: {wrong}")
    return _check(
        "o critério exige delta positivo em ao menos duas dobras e nenhum delta abaixo de -0,02, sem agregar dobras",
        not wrong,
    )


class _IndexRecorder:
    def __init__(self, source: PoseWindowDataset, log: list[int]) -> None:
        self._source = source
        self._log = log

    def __len__(self) -> int:
        return len(self._source)

    def label_at(self, index: int) -> int:
        return self._source.label_at(index)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, tuple[str, int]]:
        self._log.append(index)
        return self._source[index]


class _Capture:
    """Registra, por run, a ordem de leitura do treino, as fontes de avaliação e o estado inicial do modelo."""

    def __init__(self) -> None:
        self.runs: list[dict[str, object]] = []

    def __enter__(self) -> "_Capture":
        real_run_training = training.run_training
        runs = self.runs

        def capturing(**kwargs: Any) -> dict | None:
            order: list[int] = []
            train_source = kwargs["train_source"]
            runs.append({"train": train_source, "val": kwargs["val_source"], "test": kwargs["test_source"],
                         "order": order, "init": None})
            return real_run_training(**{**kwargs, "train_source": _IndexRecorder(train_source, order)})

        class RecordingTCN(TCNClassifier):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                digest = hashlib.sha256()
                for tensor in self.state_dict().values():
                    digest.update(tensor.detach().cpu().numpy().tobytes())
                runs[-1]["init"] = digest.hexdigest()

        self._patches = [
            patch.object(training, "run_training", capturing),
            patch.object(engine, "TCNClassifier", RecordingTCN),
        ]
        for active in self._patches:
            active.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for active in self._patches:
            active.stop()


def _train(paths: tuple[Path, Path, Path, Path], folds: list[str], variant: str, force: bool = False,
           recorder: RawRecordingPose | None = None) -> None:
    root, frames_path, manifest_path, pose_root = paths
    pose = (recorder or RawRecordingPose()).inputs()
    run_train(root, frames_path, manifest_path, pose_root, folds, ["A"], [SEED], force, pose, variant)


def _tree_digest(paths: list[Path]) -> dict[str, str]:
    digest: dict[str, str] = {}
    for path in paths:
        files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        digest.update({str(item): sha256_file(item) for item in files})
    return digest


def _protected(root: Path, partitions: list[SubjectFoldPartition]) -> list[Path]:
    return [root / FOLDS_FILE, summary_dir(root) / BASELINE_VARIANT] + [
        path
        for partition in partitions
        for path in (stats_dir(root, partition.fold), fold_run_dir(root, partition.fold, "A", SEED))
    ]


def _windows_equal(left: object, right: object) -> bool:
    first, second = cast(PoseWindowDataset, left), cast(PoseWindowDataset, right)
    return len(first) == len(second) and all(
        np.array_equal(first[index][0], second[index][0]) and first[index][1:] == second[index][1:]
        for index in range(len(first))
    )


def _missing_control_check(paths: tuple[Path, Path, Path, Path], partitions: list[SubjectFoldPartition]) -> bool:
    root = paths[0]
    folds = [partition.fold for partition in partitions]
    refused = raises(FileNotFoundError, lambda: _train(paths, folds, DETECTION_DROPOUT_VARIANT),
                     "controle da Etapa 4B ausente")
    nothing_written = all(
        not fold_run_dir(root, partition.fold, "A", SEED, variant).exists()
        for partition in partitions
        for variant in (BASELINE_VARIANT, DETECTION_DROPOUT_VARIANT)
    )
    no_summary = raises(FileNotFoundError, lambda: build_dropout_summary(root), "runs ausentes")
    return _check(
        "sem os controles da Etapa 4B, o treino aumentado falha antes de treinar qualquer dobra, não treina o "
        "controle, e o resumo pareado recusa runs ausentes",
        refused and nothing_written and no_summary,
    )


def _inconsistent_control_check(paths: tuple[Path, Path, Path, Path], partitions: list[SubjectFoldPartition]) -> bool:
    root = paths[0]
    fold = partitions[0].fold
    folds = [partition.fold for partition in partitions]
    checkpoint = fold_run_dir(root, fold, "A", SEED) / "checkpoint.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original[:-16])
    try:
        refused = raises(RuntimeError, lambda: _train(paths, folds, DETECTION_DROPOUT_VARIANT), "controle da Etapa 4B")
        untouched = checkpoint.read_bytes() == original[:-16] and all(
            not fold_run_dir(root, partition.fold, "A", SEED, DETECTION_DROPOUT_VARIANT).exists()
            for partition in partitions
        )
    finally:
        checkpoint.write_bytes(original)
    return _check(
        "com um controle inconsistente, o treino aumentado falha antes de treinar qualquer dobra e não repara nem "
        "substitui o controle",
        refused and untouched,
    )


def _pairing_check(capture: _Capture, root: Path, partitions: list[SubjectFoldPartition]) -> bool:
    baseline_runs, dropout_runs = capture.runs[: len(partitions)], capture.runs[len(partitions):]
    ok = len(dropout_runs) == len(partitions)
    for partition, control, treatment in zip(partitions, baseline_runs, dropout_runs):
        control_metrics = json.loads((fold_run_dir(root, partition.fold, "A", SEED) / "metrics.json").read_text())
        run_dir = fold_run_dir(root, partition.fold, "A", SEED, DETECTION_DROPOUT_VARIANT)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        ok &= (
            type(control["train"]) is PoseWindowDataset
            and isinstance(treatment["train"], DetectionDropoutWindowDataset)
            and all(type(run[side]) is PoseWindowDataset for run in (control, treatment) for side in ("val", "test"))
            and _windows_equal(control["val"], treatment["val"])
            and treatment["val"] is treatment["test"]
            and control["order"] == treatment["order"]
            and control["init"] is not None
            and control["init"] == treatment["init"]
            and metrics["final"]["val"] == metrics["final"]["test"]
            and metrics["final"]["val"]["support"] == control_metrics["final"]["val"]["support"]
        )
    return _check(
        "o run aumentado de cada dobra lê as janelas de treino na mesma ordem e parte da mesma inicialização do "
        "controle; só a fonte de treino muda, e o retido é avaliado com as mesmas janelas limpas da Etapa 4B",
        ok,
    )


def _isolation_check(
    root: Path, partitions: list[SubjectFoldPartition], recorders: list[RawRecordingPose]
) -> bool:
    ok = True
    for partition, recorder in zip(partitions, recorders):
        ok &= (
            bool(recorder.raw_calls)
            and set(recorder.raw_calls) <= set(partition.development_video_ids)
            and not set(recorder.loaded_videos()) & set(OFFICIAL_HELD_OUT_VIDEOS)
        )
    partition = partitions[0]
    source = training._development_source(
        partition, SEED, clean_pose_features, synthetic_raw_pose, DETECTION_DROPOUT_VARIANT, "teste"
    )
    retained_refused = isinstance(source, DetectionDropoutWindowDataset) and raises(
        RuntimeError, lambda: source._raw_pose(partition.retained_video_ids[0], 40), "fora do lado permitido"
    )
    no_raw = raises(
        ValueError,
        lambda: training._development_source(
            partition, SEED, clean_pose_features, None, DETECTION_DROPOUT_VARIANT, "teste"
        ),
        "pose bruta",
    )
    sidecar = json.loads(
        (fold_run_dir(root, partition.fold, "A", SEED, DETECTION_DROPOUT_VARIANT) / FOLD_RUN_FILE).read_text()
    )
    return _check(
        "a pose bruta só é lida nos vídeos de desenvolvimento da dobra treinada; o retido, val e test oficiais nunca "
        "são lidos pela intervenção, e a variante recusa rodar sem a pose bruta",
        ok and retained_refused and no_raw and sidecar["variant"] == DETECTION_DROPOUT_VARIANT,
    )


def _provenance_check(paths: tuple[Path, Path, Path, Path], partitions: list[SubjectFoldPartition]) -> bool:
    root = paths[0]
    fold = partitions[0].fold
    run_dir = fold_run_dir(root, fold, "A", SEED, DETECTION_DROPOUT_VARIANT)
    control_dir = fold_run_dir(root, fold, "A", SEED)
    sidecar_path = run_dir / FOLD_RUN_FILE
    recorded = json.loads(sidecar_path.read_text())
    control_config = yaml.safe_load((control_dir / "config.yaml").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    content_ok = (
        recorded["stage"] == "4C"
        and recorded["canonical"] is False
        and recorded["intervention"] == intervention_record(SEED)
        and recorded["control"]["fold_run_sha256"] == sha256_file(control_dir / FOLD_RUN_FILE)
        and recorded["control"]["run"] == "baseline_a/seed_42"
        and config["run_name"] == "baseline_a_detection_dropout"
        and {**config, "run_name": control_config["run_name"]} == control_config
    )

    original = sidecar_path.read_text()

    def refused(mutate: Callable[[dict], None], fragment: str) -> bool:
        data = json.loads(original)
        mutate(data)
        sidecar_path.write_text(json.dumps(data))
        try:
            return raises(RuntimeError, lambda: build_dropout_summary(root), fragment) and raises(
                RuntimeError, lambda: _train(paths, [fold], DETECTION_DROPOUT_VARIANT), fragment
            )
        finally:
            sidecar_path.write_text(original)

    cases = {
        "probabilidade": refused(lambda d: d["intervention"].update(probability=0.3), "intervention"),
        "regra de decisão": refused(
            lambda d: d["intervention"]["decision_rule"].update(min_delta=-0.05), "intervention"
        ),
        "fluxo aleatório": refused(lambda d: d["intervention"]["random_stream"].update(seed=43), "intervention"),
        "controle": refused(lambda d: d["control"].update(fold_run_sha256="0" * 64), "control"),
        "variante": refused(lambda d: d.update(variant=BASELINE_VARIANT), "variant"),
        "etapa": refused(lambda d: d.update(stage="4B"), "stage"),
    }

    control_sidecar = control_dir / FOLD_RUN_FILE
    control_original = control_sidecar.read_text()
    control_sidecar.write_text(control_original.replace('"canonical": false', '"canonical": false ', 1))
    try:
        cases["controle regravado"] = raises(RuntimeError, lambda: build_dropout_summary(root), "control")
    finally:
        control_sidecar.write_text(control_original)
    failed = [name for name, ok in cases.items() if not ok]
    if failed:
        print(f"  adulterações de proveniência aceitas: {failed}")
    return _check(
        "fold_run.json do run aumentado registra a intervenção congelada, a regra de decisão, o fluxo aleatório e o "
        "hash do fold_run.json do controle; config.yaml só difere do controle em run_name; adulterações ou um "
        "controle regravado são recusados pelo treino e pelo resumo",
        content_ok and not failed,
    )


def _summary_check(root: Path, partitions: list[SubjectFoldPartition]) -> bool:
    before = _tree_digest([root])
    summary = cast(dict[str, Any], build_dropout_summary(root))
    read_only = _tree_digest([root]) == before
    deltas_ok = len(summary["folds"]) == len(partitions)
    for block, partition in zip(summary["folds"], partitions):
        pair = block["pairs"][0]
        control = json.loads((fold_run_dir(root, partition.fold, "A", SEED) / "metrics.json").read_text())
        treatment = json.loads(
            (fold_run_dir(root, partition.fold, "A", SEED, DETECTION_DROPOUT_VARIANT) / "metrics.json").read_text()
        )
        control_val, treatment_val = control["final"]["val"], treatment["final"]["val"]
        deltas_ok &= (
            pair["delta_macro_f1_restricted"]
            == treatment_val["macro_f1_restricted"] - control_val["macro_f1_restricted"]
            and list(pair["per_class"]) == list(RESTRICTED_LABELS)
            and all(
                values["delta_f1"] == values["treatment_f1"] - values["control_f1"] and values["support"] > 0
                for values in pair["per_class"].values()
            )
        )
    decision = summary["decisions"][0]
    expected = evaluate_decision(list(decision["fold_deltas"].values()))
    decision_ok = (
        {key: decision[key] for key in expected} == expected
        and list(decision["fold_deltas"]) == [partition.fold for partition in partitions]
        and list(decision["focus_classes"]) == list(FOCUS_CLASSES)
        and all(len(by_fold) == len(partitions) for by_fold in decision["focus_classes"].values())
    )
    baseline_before = _tree_digest([summary_dir(root) / BASELINE_VARIANT])
    written = write_dropout_summary(root, force=False)
    path = summary_dir(root) / DETECTION_DROPOUT_VARIANT / SUMMARY_JSON_FILE
    persisted = json.loads(path.read_text())
    not_overwritten = write_dropout_summary(root, force=False) is False
    return _check(
        "o resumo pareado é somente leitura, relata por dobra o delta de macro-F1 restrita e de F1 por classe com "
        "suporte, destaca standing e fall, aplica o critério congelado sem escore agregado e grava só em "
        "summary/detection_dropout sem sobrescrever",
        read_only
        and deltas_ok
        and decision_ok
        and summary["pooled_score"] is None
        and summary["pooled_delta"] is None
        and summary["canonical"] is False
        and written
        and persisted == summary
        and not_overwritten
        and _tree_digest([summary_dir(root) / BASELINE_VARIANT]) == baseline_before,
    )


def _selftest_dropout_workspace() -> list[bool]:
    with synthetic_workspace() as paths:
        root, frames_path, manifest_path, pose_root = paths
        partitions = run_prepare(root, frames_path, manifest_path, pose_root, False, RawRecordingPose().inputs())
        folds = [partition.fold for partition in partitions]
        results = [_missing_control_check(paths, partitions)]
        recorders = [RawRecordingPose() for _ in partitions]
        with _Capture() as capture:
            _train(paths, folds, BASELINE_VARIANT)
            write_summary(root, force=False)
            results.append(_inconsistent_control_check(paths, partitions))
            protected = _protected(root, partitions)
            before = _tree_digest(protected)
            for fold, recorder in zip(folds, recorders):
                _train(paths, [fold], DETECTION_DROPOUT_VARIANT, recorder=recorder)
        results.append(_pairing_check(capture, root, partitions))
        results.append(_isolation_check(root, partitions, recorders))

        run_dir = fold_run_dir(root, folds[0], "A", SEED, DETECTION_DROPOUT_VARIANT)
        sidecar = (run_dir / FOLD_RUN_FILE).read_text()
        _train(paths, [folds[0]], DETECTION_DROPOUT_VARIANT)
        skipped = (run_dir / FOLD_RUN_FILE).read_text() == sidecar
        _train(paths, [folds[0]], DETECTION_DROPOUT_VARIANT, force=True)
        deterministic = (run_dir / FOLD_RUN_FILE).read_text() == sidecar
        results.append(_check(
            "o run aumentado íntegro é pulado, e --force o retreina com checkpoint idêntico",
            skipped and deterministic,
        ))
        results.append(_provenance_check(paths, partitions))
        results.append(_summary_check(root, partitions))
        results.append(_check(
            "folds.json, as estatísticas, os runs de controle e o resumo da Etapa 4B ficam byte a byte iguais após "
            "treinar, retreinar com --force e resumir a variante aumentada",
            _tree_digest(protected) == before,
        ))
        return results


def run_detection_dropout_selftest() -> bool:
    results = [
        _selftest_raw_masking(),
        _selftest_causal_consistency(),
        _selftest_deterministic_augmentation(),
        _selftest_decision_rule(),
        *_selftest_dropout_workspace(),
    ]
    return all(results)
