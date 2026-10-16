import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from gatefall.config import TRAIN_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.features.standardization import load_stats
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG
from gatefall.train.baseline_a.engine import _StandardizedTorchDataset, _predict
from gatefall.train.internal_folds.folds import DEVELOPMENT_SPLIT
from gatefall.train.internal_folds.selftests.fixtures import _check, raises
from gatefall.train.internal_folds.training import FOLD_RUN_FILE
from gatefall.train.shared.tcn import TCNClassifier
from gatefall.train.subject_folds.cli import run_prepare, run_train
from gatefall.train.subject_folds.folds import (
    BASELINE_VARIANT,
    DETECTION_DROPOUT_VARIANT,
    FOLDS_FILE,
    RESTRICTED_LABELS,
    SUBJECT_BALANCED_VARIANT,
    SubjectFoldPartition,
    build_subject_fold_partitions,
    fold_run_dir,
    stats_dir,
    summary_dir,
)
from gatefall.train.subject_folds.dropout_summary import write_dropout_summary
from gatefall.train.subject_folds.sampling_summary import (
    FOCUS_CLASSES,
    SUMMARY_JSON_FILE,
    build_sampling_summary,
    evaluate_decision,
    write_sampling_summary,
)
from gatefall.train.subject_folds.selftests.fixtures import (
    OFFICIAL_HELD_OUT_VIDEOS,
    RawRecordingPose,
    TrainingCapture,
    official_train_frames,
    rng_states,
    same_rng_states,
    synthetic_workspace,
)
from gatefall.train.subject_folds.selftests.detection_dropout import _tree_digest
from gatefall.train.subject_folds.selftests.training import _republish_after
from gatefall.train.subject_folds.subject_balanced_sampling import (
    SubjectBalancedSampler,
    draw_epoch,
    draws_sha256,
    intervention_record,
    window_probabilities,
)
from gatefall.train.subject_folds.summary import write_summary
from gatefall.train.subject_folds.training import (
    development_window_targets,
    subject_balanced_sampling_record,
)

SEED = 42
EPOCHS = BASELINE_A_CONFIG.epochs


def _selftest_probabilities() -> bool:
    subjects = np.array([0] * 6 + [1] * 2 + [3] * 4)
    probabilities = window_probabilities(subjects)
    per_subject = {int(s): float(probabilities[subjects == s].sum()) for s in np.unique(subjects)}
    toy_ok = (
        np.isclose(probabilities.sum(), 1.0)
        and all(np.isclose(total, 1.0 / 3) for total in per_subject.values())
        and all(len(set(probabilities[subjects == s].tolist())) == 1 for s in np.unique(subjects))
        and probabilities[subjects == 1][0] == 1.0 / (3 * 2)
        and raises(ValueError, lambda: window_probabilities([]), "ao menos uma janela")
    )

    fold_ok = True
    for partition in build_subject_fold_partitions(official_train_frames()):
        fold_subjects, _ = development_window_targets(partition)
        fold_probabilities = window_probabilities(fold_subjects)
        totals = [float(fold_probabilities[fold_subjects == s].sum()) for s in partition.development_subjects]
        draws = np.concatenate([draw_epoch(fold_probabilities, SEED, epoch) for epoch in range(400)])
        shares = [float((fold_subjects[draws] == s).mean()) for s in partition.development_subjects]
        target = 1.0 / len(partition.development_subjects)
        fold_ok &= (
            set(fold_subjects.tolist()) == set(partition.development_subjects)
            and all(np.isclose(total, target) for total in totals)
            and all(abs(share - target) < 0.02 for share in shares)
            and all(len(draw_epoch(fold_probabilities, SEED, epoch)) == len(fold_subjects) for epoch in range(3))
        )
    return _check(
        "a probabilidade por janela é 1 / (S · n_s), cada sujeito de desenvolvimento soma 1 / S, a soma é 1, cada "
        "época sorteia exatamente o número original de janelas e a fração sorteada por sujeito converge para 1 / S",
        toy_ok and fold_ok,
    )


def _selftest_reproducibility() -> bool:
    partition = build_subject_fold_partitions(official_train_frames())[0]
    subjects, labels = development_window_targets(partition)
    states = rng_states()
    first = SubjectBalancedSampler(subjects, SEED)
    second = SubjectBalancedSampler(subjects, SEED)
    first_draws = [list(first) for _ in range(4)]
    second_draws = [list(second) for _ in range(4)]
    other = [list(SubjectBalancedSampler(subjects, SEED + 1)) for _ in range(1)]
    rng_isolated = same_rng_states(states, rng_states())
    probabilities = window_probabilities(subjects)
    expected = [draw_epoch(probabilities, SEED, epoch).tolist() for epoch in range(4)]
    record = subject_balanced_sampling_record(partition, SEED)
    reference = SubjectBalancedSampler(subjects, SEED)
    for _ in range(EPOCHS):
        list(reference)
    record_ok = (
        record["draws_sha256"] == draws_sha256(reference.draws)
        and record == subject_balanced_sampling_record(partition, SEED)
        and record["draws_sha256"] != subject_balanced_sampling_record(partition, SEED + 1)["draws_sha256"]
        and record["clean_class_support"] == partition.membership_record["development_support_train_stride"]
        and sum(cast(dict[str, int], record["sampled_class_windows_total"]).values()) == len(labels) * EPOCHS
        and len(first) == len(subjects)
    )
    return _check(
        "o sorteio por (seed, época) é reprodutível, muda entre épocas e seeds, coincide com o registro calculado "
        "antes do treino e não consome o RNG do NumPy, do PyTorch nem do random",
        first_draws == second_draws == expected
        and first_draws[0] != first_draws[1]
        and other[0] != first_draws[0]
        and rng_isolated
        and record_ok,
    )


def _selftest_decision_rule() -> bool:
    cases = {
        (0.03, 0.03, -0.02): True,
        (0.03, 0.10, 0.20): True,
        (0.05, 0.04, -0.021): False,
        (0.05, 0.029, 0.0): False,
        (0.01, 0.02, 0.025): False,
        (0.10, -0.01, 0.05): True,
    }
    wrong = [
        deltas for deltas, expected in cases.items() if evaluate_decision(list(deltas))["criterion_met"] != expected
    ]
    if wrong:
        print(f"  decisões divergentes: {wrong}")
    return _check(
        "o critério exige delta de pelo menos +0,03 em ao menos duas dobras e nenhum delta abaixo de -0,02, sem "
        "agregar dobras",
        not wrong,
    )


def _train(paths: tuple[Path, Path, Path, Path], folds: list[str], variant: str, force: bool = False,
           recorder: RawRecordingPose | None = None) -> None:
    root, frames_path, manifest_path, pose_root = paths
    pose = (recorder or RawRecordingPose()).inputs()
    run_train(root, frames_path, manifest_path, pose_root, folds, ["A"], [SEED], force, pose, variant)


def _protected(root: Path, partitions: list[SubjectFoldPartition]) -> list[Path]:
    return [root / FOLDS_FILE, summary_dir(root) / BASELINE_VARIANT, summary_dir(root) / DETECTION_DROPOUT_VARIANT] + [
        path
        for partition in partitions
        for path in (
            stats_dir(root, partition.fold),
            fold_run_dir(root, partition.fold, "A", SEED),
            fold_run_dir(root, partition.fold, "A", SEED, DETECTION_DROPOUT_VARIANT),
        )
    ]


def _missing_control_check(paths: tuple[Path, Path, Path, Path], partitions: list[SubjectFoldPartition]) -> bool:
    root = paths[0]
    folds = [partition.fold for partition in partitions]
    refused = raises(FileNotFoundError, lambda: _train(paths, folds, SUBJECT_BALANCED_VARIANT),
                     "controle da Etapa 4B ausente")
    nothing_written = all(
        not fold_run_dir(root, partition.fold, "A", SEED, variant).exists()
        for partition in partitions
        for variant in (BASELINE_VARIANT, SUBJECT_BALANCED_VARIANT)
    )
    no_summary = raises(FileNotFoundError, lambda: build_sampling_summary(root), "runs ausentes")
    return _check(
        "sem os controles da Etapa 4B, o treino balanceado falha antes de treinar qualquer dobra, não treina o "
        "controle, e o resumo pareado recusa runs ausentes",
        refused and nothing_written and no_summary,
    )


def _legacy_path_check(control_runs: list[dict[str, Any]]) -> bool:
    ok = len(control_runs) > 0
    for run in control_runs:
        n = len(run["train"])
        order = cast(list[int], run["order"])
        chunks = [order[start : start + n] for start in range(0, len(order), n)]
        train_loader = run["loaders"][0]
        ok &= (
            run["sampler"] is None
            and len(run["loaders"]) == 3
            and train_loader.get("shuffle") is True
            and isinstance(train_loader.get("generator"), torch.Generator)
            and "sampler" not in train_loader
            and len(chunks) == EPOCHS + 1
            and all(sorted(chunk) == list(range(n)) for chunk in chunks)
            and chunks[-1] != list(range(n))
        )
    return _check(
        "sem o hook, o engine mantém o caminho canônico: um único loader de treino embaralhado pelo gerador da "
        "seed, sem sampler, cada época é uma permutação das janelas e final.train reaproveita esse loader",
        ok,
    )


def _pairing_check(
    control_runs: list[dict[str, Any]], treatment_runs: list[dict[str, Any]], root: Path,
    partitions: list[SubjectFoldPartition],
) -> bool:
    ok = len(treatment_runs) == len(partitions)
    for partition, control, treatment in zip(partitions, control_runs, treatment_runs):
        n = len(treatment["train"])
        order = cast(list[int], treatment["order"])
        sampler = cast(SubjectBalancedSampler, treatment["sampler"])
        subjects, _ = development_window_targets(partition)
        probabilities = window_probabilities(subjects)
        expected = [index for epoch in range(EPOCHS) for index in draw_epoch(probabilities, SEED, epoch).tolist()]
        train_loader = treatment["loaders"][0]
        metrics = json.loads(
            (fold_run_dir(root, partition.fold, "A", SEED, SUBJECT_BALANCED_VARIANT) / "metrics.json").read_text()
        )
        control_metrics = json.loads((fold_run_dir(root, partition.fold, "A", SEED) / "metrics.json").read_text())
        ok &= (
            type(treatment["train"]) is PoseWindowDataset
            and isinstance(sampler, SubjectBalancedSampler)
            and train_loader.get("sampler") is sampler
            and isinstance(train_loader.get("generator"), torch.Generator)
            and "shuffle" not in train_loader
            and len(treatment["loaders"]) == 4
            and treatment["loaders"][1].get("shuffle") is False
            and order[: EPOCHS * n] == expected
            and order[EPOCHS * n :] == list(range(n))
            and len(set(expected)) < len(expected)
            and control["init"] is not None
            and control["init"] == treatment["init"]
            and treatment["val"] is treatment["test"]
            and metrics["final"]["val"] == metrics["final"]["test"]
            and metrics["final"]["val"]["support"] == control_metrics["final"]["val"]["support"]
            and metrics["final"]["train"]["support"] == control_metrics["final"]["train"]["support"]
            and set(subjects[np.array(expected)].tolist()) == set(partition.development_subjects)
        )
    return _check(
        "o run balanceado lê as janelas limpas do controle na ordem exata dos sorteios com reposição, parte da mesma "
        "inicialização do controle e avalia o retido com as mesmas janelas limpas; final.train é lido em ordem "
        "sequencial, sem o sampler",
        ok,
    )


def _clean_train_evaluation_check(root: Path, partitions: list[SubjectFoldPartition]) -> bool:
    ok = True
    for partition in partitions:
        run_dir = fold_run_dir(root, partition.fold, "A", SEED, SUBJECT_BALANCED_VARIANT)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        config = BASELINE_A_CONFIG
        model = TCNClassifier(
            input_dim=config.input_dim,
            channels=config.channels,
            kernel_size=config.kernel_size,
            dilations=config.dilations,
            dropout=config.dropout,
            num_classes=config.num_classes,
        )
        device = cast(str, metrics["device"])
        model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device))
        model.to(device)
        source = PoseWindowDataset(partition.frames, DEVELOPMENT_SPLIT, TRAIN_STRIDE, RawRecordingPose().inputs().loader)
        stats = load_stats(stats_dir(root, partition.fold) / "pose.json")
        loader = DataLoader(_StandardizedTorchDataset(source, stats), batch_size=config.batch_size, shuffle=False)
        y_true, y_pred = _predict(model, loader, device)
        matrix = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
        np.add.at(matrix, (y_true, y_pred), 1)
        ok &= matrix.tolist() == metrics["final"]["train"]["confusion_matrix"]
    return _check(
        "final.train do run balanceado reproduz a matriz de confusão do checkpoint sobre todas as janelas limpas do "
        "treino de desenvolvimento, não sobre a amostra sorteada",
        ok,
    )


def _provenance_check(paths: tuple[Path, Path, Path, Path], partitions: list[SubjectFoldPartition]) -> bool:
    root = paths[0]
    partition = partitions[0]
    fold = partition.fold
    run_dir = fold_run_dir(root, fold, "A", SEED, SUBJECT_BALANCED_VARIANT)
    control_dir = fold_run_dir(root, fold, "A", SEED)
    sidecar_path = run_dir / FOLD_RUN_FILE
    recorded = json.loads(sidecar_path.read_text())
    control_config = yaml.safe_load((control_dir / "config.yaml").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    content_ok = (
        recorded["stage"] == "5A"
        and recorded["variant"] == SUBJECT_BALANCED_VARIANT
        and recorded["canonical"] is False
        and recorded["intervention"] == intervention_record(SEED)
        and recorded["sampling"] == subject_balanced_sampling_record(partition, SEED)
        and recorded["control"]["fold_run_sha256"] == sha256_file(control_dir / FOLD_RUN_FILE)
        and recorded["control"]["run"] == "baseline_a/seed_42"
        and config["run_name"] == "baseline_a_subject_balanced"
        and {**config, "run_name": control_config["run_name"]} == control_config
    )

    original = sidecar_path.read_text()

    def refused(mutate: Callable[[dict], None], fragment: str) -> bool:
        data = json.loads(original)
        mutate(data)
        sidecar_path.write_text(json.dumps(data))
        try:
            return raises(RuntimeError, lambda: build_sampling_summary(root), fragment) and raises(
                RuntimeError, lambda: _train(paths, [fold], SUBJECT_BALANCED_VARIANT), fragment
            )
        finally:
            sidecar_path.write_text(original)

    def shift_subject_count(data: dict) -> None:
        epoch = data["sampling"]["sampled_subject_windows_by_epoch"][0]
        first, second = list(epoch)[:2]
        epoch[first] += 1
        epoch[second] -= 1

    cases = {
        "regra de decisão": refused(
            lambda d: d["intervention"]["decision_rule"].update(min_improvement=0.01), "intervention"
        ),
        "balanceamento de classe": refused(lambda d: d["intervention"].update(class_balancing=True), "intervention"),
        "fluxo aleatório": refused(lambda d: d["intervention"]["random_stream"].update(seed=43), "intervention"),
        "contagem por sujeito": refused(shift_subject_count, "sampling"),
        "sorteios": refused(lambda d: d["sampling"].update(draws_sha256="1" * 64), "sampling"),
        "índice de janelas": refused(lambda d: d["sampling"]["window_subject_runs"].reverse(), "sampling"),
        "probabilidade": refused(
            lambda d: d["sampling"]["window_probability"].update({str(partition.development_subjects[0]): 0.5}),
            "sampling",
        ),
        "épocas": refused(lambda d: d["sampling"].update(epochs=EPOCHS - 1), "sampling"),
        "sem sampling": refused(lambda d: d.pop("sampling"), "sampling"),
        "controle": refused(lambda d: d["control"].update(fold_run_sha256="0" * 64), "control"),
        "variante": refused(lambda d: d.update(variant=DETECTION_DROPOUT_VARIANT), "variant"),
        "etapa": refused(lambda d: d.update(stage="4C"), "stage"),
    }
    failed = [name for name, ok in cases.items() if not ok]
    if failed:
        print(f"  adulterações de proveniência aceitas: {failed}")
    return _check(
        "fold_run.json do run balanceado registra a intervenção, a regra de decisão, o fluxo aleatório, as contagens "
        "sorteadas e o hash do controle; config.yaml só difere do controle em run_name; adulterações são recusadas "
        "pelo treino e pelo resumo",
        content_ok and not failed,
    )


def _paired_validation_check(root: Path, partitions: list[SubjectFoldPartition]) -> bool:
    run_dir = fold_run_dir(root, partitions[0].fold, "A", SEED, SUBJECT_BALANCED_VARIANT)

    def shift_retained(metrics: dict) -> None:
        for split in ("val", "test"):
            metrics["final"][split]["support"]["walk"] += 1

    def shift_train(metrics: dict) -> None:
        metrics["final"]["train"]["support"]["walk"] += 1

    def diverge_test(metrics: dict) -> None:
        metrics["final"]["test"]["macro_f1_restricted"] = 0.999

    accepted: list[str] = []
    for name, mutate in (("suporte do retido", shift_retained), ("suporte de treino", shift_train),
                         ("val ≠ test", diverge_test)):
        restore = _republish_after(run_dir, mutate)
        try:
            if not raises(RuntimeError, lambda: build_sampling_summary(root)):
                accepted.append(name)
        finally:
            restore()
    checkpoint = run_dir / "checkpoint.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original[:-16])
    try:
        if not raises(RuntimeError, lambda: build_sampling_summary(root), "checkpoint"):
            accepted.append("checkpoint truncado")
    finally:
        checkpoint.write_bytes(original)
    if accepted:
        print(f"  adulterações aceitas pelo resumo pareado: {accepted}")
    return _check(
        "o resumo pareado recusa run balanceado com suporte do retido ou do treino limpo divergente do controle, val "
        "diferente de test ou checkpoint corrompido",
        not accepted and not (summary_dir(root) / SUBJECT_BALANCED_VARIANT).exists(),
    )


def _summary_check(root: Path, partitions: list[SubjectFoldPartition]) -> bool:
    before = _tree_digest([root])
    summary = cast(dict[str, Any], build_sampling_summary(root))
    read_only = _tree_digest([root]) == before
    blocks_ok = len(summary["folds"]) == len(partitions)
    for block, partition in zip(summary["folds"], partitions):
        pair = block["pairs"][0]
        control = json.loads((fold_run_dir(root, partition.fold, "A", SEED) / "metrics.json").read_text())
        treatment = json.loads(
            (fold_run_dir(root, partition.fold, "A", SEED, SUBJECT_BALANCED_VARIANT) / "metrics.json").read_text()
        )
        sampling = pair["sampling"]
        record = cast(dict[str, Any], subject_balanced_sampling_record(partition, SEED))
        blocks_ok &= (
            pair["delta_macro_f1_restricted"]
            == treatment["final"]["val"]["macro_f1_restricted"] - control["final"]["val"]["macro_f1_restricted"]
            and list(pair["per_class"]) == list(RESTRICTED_LABELS)
            and list(sampling["subjects"]) == [str(subject) for subject in partition.development_subjects]
            and all(
                values["sampled_total"] == record["sampled_subject_windows_total"][subject]
                and values["clean_windows"] == record["subject_windows"][subject]
                for subject, values in sampling["subjects"].items()
            )
            and bool(np.isclose(sum(values["sampled_share"] for values in sampling["subjects"].values()), 1.0))
            and {label: values["clean_support"] for label, values in sampling["classes"].items()}
            == partition.membership_record["development_support_train_stride"]
        )
    decision = summary["decisions"][0]
    expected = evaluate_decision(list(decision["fold_deltas"].values()))
    decision_ok = (
        {key: decision[key] for key in expected} == expected
        and list(decision["focus_classes"]) == list(FOCUS_CLASSES)
        and all(len(by_fold) == len(partitions) for by_fold in decision["focus_classes"].values())
    )
    written = write_sampling_summary(root, force=False)
    persisted = json.loads((summary_dir(root) / SUBJECT_BALANCED_VARIANT / SUMMARY_JSON_FILE).read_text())
    not_overwritten = write_sampling_summary(root, force=False) is False
    return _check(
        "o resumo pareado é somente leitura, relata deltas por dobra, F1 e suporte por classe, standing, other e "
        "fall, contagens sorteadas por sujeito e por classe, aplica o critério congelado sem escore agregado e grava "
        "só em summary/subject_balanced sem sobrescrever",
        read_only
        and blocks_ok
        and decision_ok
        and summary["pooled_score"] is None
        and summary["pooled_delta"] is None
        and summary["canonical"] is False
        and written
        and persisted == summary
        and not_overwritten,
    )


def _selftest_sampling_workspace() -> list[bool]:
    with synthetic_workspace() as paths:
        root, frames_path, manifest_path, pose_root = paths
        partitions = run_prepare(root, frames_path, manifest_path, pose_root, False, RawRecordingPose().inputs())
        folds = [partition.fold for partition in partitions]
        results = [_missing_control_check(paths, partitions)]
        recorders = [RawRecordingPose() for _ in partitions]
        with TrainingCapture() as capture:
            _train(paths, folds, BASELINE_VARIANT)
            write_summary(root, force=False)
            _train(paths, folds, DETECTION_DROPOUT_VARIANT)
            write_dropout_summary(root, force=False)
            protected = _protected(root, partitions)
            before = _tree_digest(protected)
            for fold, recorder in zip(folds, recorders):
                _train(paths, [fold], SUBJECT_BALANCED_VARIANT, recorder=recorder)
        control_runs = capture.runs[: len(partitions)]
        treatment_runs = capture.runs[2 * len(partitions):]
        results.append(_legacy_path_check(control_runs))
        results.append(_pairing_check(control_runs, treatment_runs, root, partitions))
        results.append(_check(
            "o treino balanceado nunca lê val ou test oficiais, e cada dobra só sorteia janelas dos seus sujeitos de "
            "desenvolvimento",
            all(
                not recorder.loaded_videos() & set(OFFICIAL_HELD_OUT_VIDEOS)
                and not recorder.raw_calls
                and set(recorder.loaded_videos()) <= set(partition.development_video_ids) | set(
                    partition.retained_video_ids
                )
                for partition, recorder in zip(partitions, recorders)
            ),
        ))
        results.append(_clean_train_evaluation_check(root, partitions))

        run_dir = fold_run_dir(root, folds[0], "A", SEED, SUBJECT_BALANCED_VARIANT)
        sidecar = (run_dir / FOLD_RUN_FILE).read_text()
        _train(paths, [folds[0]], SUBJECT_BALANCED_VARIANT)
        skipped = (run_dir / FOLD_RUN_FILE).read_text() == sidecar
        _train(paths, [folds[0]], SUBJECT_BALANCED_VARIANT, force=True)
        deterministic = (run_dir / FOLD_RUN_FILE).read_text() == sidecar
        results.append(_check(
            "o run balanceado íntegro é pulado, e --force o retreina com checkpoint e sorteios idênticos",
            skipped and deterministic,
        ))
        results.append(_provenance_check(paths, partitions))
        results.append(_paired_validation_check(root, partitions))
        results.append(_summary_check(root, partitions))
        results.append(_check(
            "folds.json, as estatísticas, os controles da Etapa 4B, os runs detection_dropout e os resumos anteriores "
            "ficam byte a byte iguais após treinar, retreinar com --force e resumir a variante balanceada",
            _tree_digest(protected) == before,
        ))
        return results


def run_subject_balanced_selftest() -> bool:
    results = [
        _selftest_probabilities(),
        _selftest_reproducibility(),
        _selftest_decision_rule(),
        *_selftest_sampling_workspace(),
    ]
    return all(results)
