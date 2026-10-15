import json
from typing import cast

import numpy as np
import pandas as pd
import torch

from gatefall.config import IGNORE_LABEL, NUM_CLASSES
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis.internal_fold_retained_audit import confusion_matrix
from gatefall.eval.analysis.visual_neutralization_counterfactual import (
    REPORT_JSON_FILE,
    REPORT_SUMMARY_FILE,
    REPORT_VIDEOS_FILE,
    NeutralizedVisualFusion,
    NeutralizedVisualGatedFusion,
    load_retained_features,
    paired_changes,
    paired_videos,
    paired_windows,
    predict_retained,
    require_reconciled,
    run_analysis,
)
from gatefall.eval.baseline_b0.cli import FusionModel
from gatefall.eval.baseline_b0.cli import _predict_with_identity as predict_fusion
from gatefall.eval.baseline_b1.cli import GatedFusionModel
from gatefall.eval.baseline_b1.cli import _predict_with_identity as predict_gated_fusion
from gatefall.features.dinov3_standardization import FEATURE_DIM as DINOV3_FEATURE_DIM
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM
from gatefall.runs import REFERENCE_RUN_ROOT
from gatefall.train.internal_folds.cli import _official_train, run_prepare, run_train
from gatefall.train.internal_folds.folds import RETAINED_SPLIT, build_fold_partitions, fold_run_dir, stats_dir
from gatefall.train.internal_folds.selftests.fixtures import (
    OFFICIAL_HELD_OUT_VIDEOS,
    QUALITY,
    TRAIN_VIDEOS,
    RecordingLoaders,
    _check,
    raises,
    synthetic_workspace,
)
from gatefall.train.internal_folds.stats import load_fold_stats
from gatefall.train.internal_folds.training import FeatureLoaders, resolve_fold_config, validate_fold_run
from gatefall.train.shared.metrics import restricted_macro_f1

W, F, OT = 0, 1, 9
WINDOW = 4


def _aligned(predictions: list[int]) -> pd.DataFrame:
    # Vídeo a: 4 janelas rotuladas; vídeo b: 2 rotuladas e 1 ignorada, que
    # não entra na reconciliação nem no pareamento.
    labels = [W, W, F, F, OT, OT, IGNORE_LABEL]
    videos = ["a"] * 4 + ["b"] * 3
    subjects = [1] * 4 + [2] * 3
    k_ends = [0, 1, 2, 3, 0, 1, 2]
    return pd.DataFrame(
        {"video_id": videos, "subject": subjects, "k_end": k_ends, "label": labels, "pred_label": predictions}
    )


def _stored(aligned: pd.DataFrame) -> dict:
    labeled = cast(pd.DataFrame, aligned[aligned["label"] != IGNORE_LABEL])
    labels, predictions = labeled["label"].to_numpy(), labeled["pred_label"].to_numpy()
    return {
        "confusion_matrix": confusion_matrix(labels, predictions),
        "macro_f1_restricted": restricted_macro_f1(labels, predictions)[0],
    }


def _selftest_reconciliation() -> bool:
    original = _aligned([W, W, W, F, OT, W, W])
    stored = _stored(original)
    matching = require_reconciled(original, stored, "selftest")
    swapped = original.assign(pred_label=[F, W, W, W, OT, W, W])
    wrong_f1 = {**stored, "macro_f1_restricted": stored["macro_f1_restricted"] + 1e-6}
    return _check(
        "a reconciliação aceita predições originais com a mesma matriz e a mesma Macro-F1 restrita de "
        "metrics.json, ignora janelas IGNORE_LABEL e recusa o contrafactual quando a matriz ou a Macro-F1 divergem",
        matching["matches_metrics_json"]
        and matching["macro_f1_matches_metrics_json"]
        and matching["labeled_windows"] == 6
        and matching["ignored_windows"] == 1
        and raises(RuntimeError, lambda: require_reconciled(swapped, stored, "selftest"), "não reproduzem")
        and raises(RuntimeError, lambda: require_reconciled(original, wrong_f1, "selftest"), "Macro-F1"),
    )


def _selftest_paired_changes() -> bool:
    original = _aligned([W, W, W, F, OT, W, W])
    counterfactual = _aligned([W, F, F, F, OT, OT, F])
    paired = paired_windows(original, counterfactual, "selftest")
    changes = paired_changes(paired)
    videos = {video["video_id"]: video for video in paired_videos(paired)}
    unpaired = cast(pd.DataFrame, counterfactual[counterfactual["k_end"] != 3])
    return _check(
        "o pareamento conta só janelas rotuladas, separa as que passam a acertar das que passam a errar, "
        "lista as transições e resume a mudança por vídeo; janela sem par é recusada",
        len(paired) == 6
        and changes["changed_windows"] == 3
        and changes["became_correct"] == 2
        and changes["became_incorrect"] == 1
        and changes["transitions"] == [
            {"from": "walk", "to": "fall", "windows": 2},
            {"from": "walk", "to": "other", "windows": 1},
        ]
        and changes["video_changed_fraction"] == {"n_videos": 2, "median": 0.5, "min": 0.5, "max": 0.5}
        and videos["a"]["original_predicted_walk_fraction"] == 0.75
        and videos["a"]["counterfactual_predicted_walk_fraction"] == 0.25
        and videos["b"]["counterfactual_accuracy"] == 1.0
        and raises(ValueError, lambda: paired_windows(original, unpaired, "selftest"), "não pareiam"),
    )


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, ...]] = []

    def record(self, *inputs: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(tensor.clone() for tensor in inputs))
        return inputs[0].sum(dim=(1, 2)).unsqueeze(1).repeat(1, NUM_CLASSES) * torch.arange(NUM_CLASSES)


class RecordingFusion(RecordingModel):
    def __call__(self, pose: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        return self.record(pose, visual)


class RecordingGatedFusion(RecordingModel):
    def __call__(self, pose: torch.Tensor, visual: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        return self.record(pose, visual, quality)


class ListSource:
    def __init__(self, items: list[tuple]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def _selftest_visual_only_intervention() -> bool:
    rng = np.random.default_rng(0)
    pose = torch.from_numpy(rng.normal(size=(3, WINDOW, POSE_FEATURE_DIM)).astype(np.float32))
    visual = torch.from_numpy(rng.normal(size=(3, WINDOW, DINOV3_FEATURE_DIM)).astype(np.float32))
    quality = torch.from_numpy(rng.uniform(size=(3, WINDOW, 2)).astype(np.float32))
    visual_before = visual.clone()

    fusion_model = RecordingFusion()
    NeutralizedVisualFusion(fusion_model)(pose, visual)
    gated_model = RecordingGatedFusion()
    NeutralizedVisualGatedFusion(gated_model)(pose, visual, quality)
    (fusion_pose, fusion_visual), (gated_pose, gated_visual, gated_quality) = fusion_model.calls[0], gated_model.calls[0]
    wrapped = (
        torch.equal(fusion_pose, pose)
        and torch.equal(gated_pose, pose)
        and torch.equal(gated_quality, quality)
        and all(
            tensor.shape == visual.shape and tensor.dtype == visual.dtype and not torch.any(tensor)
            for tensor in (fusion_visual, gated_visual)
        )
        and torch.equal(visual, visual_before)
    )
    return _check(
        "a neutralização entrega a B0 e B1 a pose e a qualidade sem alteração e um tensor visual de zeros com a "
        "mesma forma e dtype, sem mutar o tensor visual original",
        wrapped,
    )


def _selftest_validated_path_preserves_other_inputs() -> bool:
    with synthetic_workspace() as (root, frames_path, manifest_path):
        run_prepare(root, frames_path, manifest_path, force=False, loaders=RecordingLoaders().loaders(),
                    quality=QUALITY)
        frames, _manifest, _snapshot, input_sha256 = _official_train(frames_path, manifest_path)
        partition = build_fold_partitions(frames)[0]
        stats = load_fold_stats(partition, stats_dir(root, partition.fold), frames_path, input_sha256)
        rng = np.random.default_rng(1)
        items = [
            (
                rng.normal(size=(WINDOW, POSE_FEATURE_DIM)).astype(np.float32),
                rng.normal(size=(WINDOW, DINOV3_FEATURE_DIM)).astype(np.float32) + 50.0,
                rng.uniform(size=(WINDOW, 2)).astype(np.float32),
                index % 3,
                ("v", index),
            )
            for index in range(5)
        ]
        fusion_items = [(pose, visual, label, identity) for pose, visual, _quality, label, identity in items]
        records = {}
        for name in ("original", "counterfactual"):
            model = RecordingFusion()
            wrapped: FusionModel = NeutralizedVisualFusion(model) if name == "counterfactual" else model
            fusion = predict_fusion(wrapped, ListSource(fusion_items), stats.pose, stats.dinov3, "cpu", 2)
            gated_model = RecordingGatedFusion()
            gated_wrapped: GatedFusionModel = (
                NeutralizedVisualGatedFusion(gated_model) if name == "counterfactual" else gated_model
            )
            gated = predict_gated_fusion(gated_wrapped, ListSource(items), stats.pose, stats.dinov3, "cpu", 2)
            records[name] = (model.calls, gated_model.calls, fusion, gated)

    def stacked(calls: list[tuple[torch.Tensor, ...]], position: int) -> torch.Tensor:
        return torch.cat([call[position] for call in calls])

    (fusion_original, gated_original, fusion_predictions, gated_predictions) = records["original"]
    (fusion_counterfactual, gated_counterfactual, fusion_cf_predictions, gated_cf_predictions) = records[
        "counterfactual"
    ]
    preserved = (
        torch.equal(stacked(fusion_original, 0), stacked(fusion_counterfactual, 0))
        and torch.equal(stacked(gated_original, 0), stacked(gated_counterfactual, 0))
        and torch.equal(stacked(gated_original, 2), stacked(gated_counterfactual, 2))
        and bool(torch.any(stacked(fusion_original, 1)))
        and not torch.any(stacked(fusion_counterfactual, 1))
        and not torch.any(stacked(gated_counterfactual, 1))
        and fusion_predictions[:3] == fusion_cf_predictions[:3]
        and gated_predictions[:3] == gated_cf_predictions[:3]
    )
    return _check(
        "pelos preditores validados de B0 e B1, a condição contrafactual recebe a mesma pose padronizada, a "
        "mesma qualidade crua, as mesmas identidades e rótulos e só o DINOv3 padronizado zerado",
        preserved,
    )


def _changing_loaders(loaders: FeatureLoaders, calls: dict[str, int]) -> FeatureLoaders:
    def visual(video_id: str) -> np.ndarray:
        calls[video_id] = calls.get(video_id, 0) + 1
        values = loaders.visual(video_id).copy()
        return values + 1.0 if calls[video_id] > 1 else values

    return FeatureLoaders(pose=loaders.pose, visual=visual, quality=loaders.quality)


def _selftest_end_to_end() -> bool:
    with synthetic_workspace() as (root, frames_path, manifest_path):
        run_prepare(root, frames_path, manifest_path, force=False, loaders=RecordingLoaders().loaders(),
                    quality=QUALITY)
        run_train(root, frames_path, manifest_path, "pilot", ["retain_EnvA", "retain_EnvB", "retain_EnvC"],
                  ["A", "B0", "B1"], force=False, loaders=RecordingLoaders().loaders(), quality=QUALITY)
        artifacts = sorted(root.rglob("*.*"))
        before = {path: sha256_file(path) for path in artifacts}
        recorder = RecordingLoaders()
        output_dir = root.parent / "counterfactual"
        written = run_analysis(root, frames_path, manifest_path, output_dir, loaders=recorder.loaders(),
                               quality=QUALITY, device="cpu")
        outputs = [output_dir / name for name in (REPORT_JSON_FILE, REPORT_SUMMARY_FILE, REPORT_VIDEOS_FILE)]
        first = [path.read_bytes() for path in outputs]
        skipped = not run_analysis(root, frames_path, manifest_path, output_dir, loaders=RecordingLoaders().loaders(),
                                   quality=QUALITY, device="cpu")
        run_analysis(root, frames_path, manifest_path, output_dir, force=True, loaders=RecordingLoaders().loaders(),
                     quality=QUALITY, device="cpu")
        deterministic = [path.read_bytes() for path in outputs] == first
        unchanged = {path: sha256_file(path) for path in artifacts} == before
        report = json.loads(first[0].decode("utf-8"))
        csv = pd.read_csv(outputs[2])

        folds = report["folds"]
        runs = [run for fold in folds for run in fold["runs"]]
        references = [fold["reference"] for fold in folds]
        counterfactual_ok = all(
            run["arm"] in ("B0", "B1")
            and run["counterfactual"]["labeled_windows"] == run["original"]["labeled_windows"]
            and sum(run["counterfactual"]["predicted_class_windows"].values()) == run["paired"]["labeled_windows"]
            and run["paired"]["changed_windows"] == sum(item["windows"] for item in run["paired"]["transitions"])
            for run in runs
        )
        isolated = all(
            {video["video_id"] for run in (fold["reference"], *fold["runs"]) for video in run["videos"]}
            == set(fold["retained_videos"])
            and all(video_id.startswith(fold["retained_environment"] + "/") for video_id in fold["retained_videos"])
            for fold in folds
        )
        clean_ok = _check(
            "o contrafactual reconcilia os nove runs sintéticos, neutraliza só B0 e B1, mantém A como referência "
            "sem intervenção, pareia janelas rotuladas, só lê vídeos do train oficial, não altera runs nem "
            "estatísticas, não sobrescreve sem --force e grava saídas byte a byte idênticas ao repetir",
            written
            and skipped
            and deterministic
            and unchanged
            and report["official_splits_read"] == ["train"]
            and report["intervention"]["arms"] == ["B0", "B1"]
            and len(runs) == 6
            and len(references) == 3
            and all(reference["arm"] == "A" and "counterfactual" not in reference for reference in references)
            and all(
                run["reconciliation"]["matches_metrics_json"] and run["reconciliation"]["macro_f1_matches_metrics_json"]
                for run in (*runs, *references)
            )
            and counterfactual_ok
            and isolated
            and set(report["feature_sha256"]) == set(TRAIN_VIDEOS)
            and recorder.loaded_videos() <= set(TRAIN_VIDEOS)
            and not recorder.loaded_videos() & set(OFFICIAL_HELD_OUT_VIDEOS)
            and len(csv) == sum(len(run["videos"]) for fold in folds for run in (fold["reference"], *fold["runs"]))
            and bool(cast(pd.DataFrame, csv[csv["arm"] == "A"])["counterfactual_predicted_walk_fraction"].isna().all())
            and "## O que isto não estabelece" in first[1].decode("utf-8"),
        )

        frames, _manifest, _snapshot, input_sha256 = _official_train(frames_path, manifest_path)
        partition = next(item for item in build_fold_partitions(frames) if item.fold == "retain_EnvB")
        fold_recorder = RecordingLoaders()
        cached, _digests = load_retained_features(partition, fold_recorder.loaders())
        stats = load_fold_stats(partition, stats_dir(root, partition.fold), frames_path, input_sha256)
        run_dir = fold_run_dir(root, partition.fold, "B1", 42)
        config = validate_fold_run(run_dir, resolve_fold_config("B1", 42, stats.files, QUALITY))
        video_ids, _k_ends, _labels, _predictions = predict_retained(partition, config, run_dir, stats, cached,
                                                                     "cpu", True)
        expected_windows = len(build_window_index(partition.side_frames(RETAINED_SPLIT), stride=1,
                                                  drop_ignored=False))
        a_config = validate_fold_run(fold_run_dir(root, partition.fold, "A", 42),
                                     resolve_fold_config("A", 42, stats.files, QUALITY))
        fold_ok = _check(
            "a dobra só carrega e infere os vídeos do próprio ambiente retido, recusa ler vídeo de outra dobra e "
            "recusa neutralizar a arma A",
            fold_recorder.loaded_videos() == set(partition.retained_video_ids)
            and set(video_ids) == set(partition.retained_video_ids)
            and len(video_ids) == expected_windows
            and raises(KeyError, lambda: cached.visual("EnvA/a1"))
            and raises(ValueError, lambda: predict_retained(partition, a_config, run_dir, stats, cached, "cpu", True),
                       "não tem entrada visual"),
        )

        refused = all(
            raises(ValueError, lambda target=target: run_analysis(root, frames_path, manifest_path, target,
                                                                   loaders=RecordingLoaders().loaders(),
                                                                   quality=QUALITY, device="cpu"), "se sobrepõe")
            for target in (root / "counterfactual", REFERENCE_RUN_ROOT / "counterfactual", frames_path.parent)
        )
        changing = raises(
            RuntimeError,
            lambda: run_analysis(root, frames_path, manifest_path, root.parent / "changing",
                                 loaders=_changing_loaders(RecordingLoaders().loaders(), {}), quality=QUALITY,
                                 device="cpu"),
            "features mudaram",
        )
        guard_ok = _check(
            "o contrafactual recusa saída dentro das dobras, das referências ou das entradas e acusa features que "
            "mudam entre a leitura e a verificação final",
            refused and changing,
        )
        return clean_ok and fold_ok and guard_ok


def run_selftest() -> bool:
    results = [
        _selftest_reconciliation(),
        _selftest_paired_changes(),
        _selftest_visual_only_intervention(),
        _selftest_validated_path_preserves_other_inputs(),
        _selftest_end_to_end(),
    ]
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
