import json
from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.eval.analysis.internal_fold_retained_audit import (
    AUDIT_JSON_FILE,
    AUDIT_SUMMARY_FILE,
    VideoFeatures,
    build_report,
    confusion_matrix,
    prediction_summary,
    reconcile,
    run_audit,
    video_integrity,
    video_predictions,
)
from gatefall.features.dinov3_standardization import FEATURE_DIM as DINOV3_FEATURE_DIM
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM
from gatefall.runs import REFERENCE_RUN_ROOT
from gatefall.train.internal_folds.cli import run_prepare, run_train
from gatefall.train.internal_folds.folds import FOLDS_FILE, fold_run_dir
from gatefall.train.internal_folds.selftests.fixtures import (
    OFFICIAL_HELD_OUT_VIDEOS,
    QUALITY,
    TRAIN_VIDEOS,
    RecordingLoaders,
    _check,
    raises,
    synthetic_workspace,
)
from gatefall.train.internal_folds.training import FOLD_RUN_FILE, FeatureLoaders

W, F, OT = 0, 1, 9
N_FRAMES = 6


def _clean_features() -> VideoFeatures:
    rng = np.random.default_rng(0)
    return VideoFeatures(
        pose=rng.normal(size=(N_FRAMES, POSE_FEATURE_DIM)).astype(np.float32),
        visual=rng.normal(size=(N_FRAMES, DINOV3_FEATURE_DIM)).astype(np.float32),
        quality=rng.uniform(size=(N_FRAMES, 2)).astype(np.float32),
    )


def _defects(features: VideoFeatures) -> list[str]:
    return video_integrity("v", "EnvA", 1, N_FRAMES, features)["defects"]


def _selftest_integrity() -> bool:
    clean = _clean_features()
    nan_visual = clean.visual.copy()
    nan_visual[2, 5] = np.nan
    zero_row = clean.visual.copy()
    zero_row[3] = 0.0
    frozen = np.repeat(clean.visual[:1], N_FRAMES, axis=0)
    out_of_range = clean.quality.copy()
    out_of_range[0, 1] = 1.5
    cases = {
        "dinov3 com 1 valor(es) não finito(s)": VideoFeatures(clean.pose, nan_visual, clean.quality),
        "pose com forma": VideoFeatures(clean.pose[:-1], clean.visual, clean.quality),
        "quadro(s) inteiramente zerado(s)": VideoFeatures(clean.pose, zero_row, clean.quality),
        "dinov3 constante no tempo": VideoFeatures(clean.pose, frozen, clean.quality),
        "qualidade com 1 valor(es) fora de [0, 1]": VideoFeatures(clean.pose, clean.visual, out_of_range),
    }
    detected = all(any(fragment in defect for defect in _defects(features)) for fragment, features in cases.items())
    nan_record = video_integrity("v", "EnvA", 1, N_FRAMES, VideoFeatures(clean.pose, nan_visual, clean.quality))
    return _check(
        "a integridade por vídeo não acusa defeito em features limpas e acusa valor não finito, K divergente, "
        "quadro DINOv3 zerado, DINOv3 constante no tempo e qualidade fora de [0, 1], com descritores finitos",
        not _defects(clean) and detected and json.dumps(nan_record, allow_nan=False) is not None,
    )


def _aligned() -> pd.DataFrame:
    # O vídeo a tem 10 janelas rotuladas, todas preditas walk; o vídeo b tem
    # 2 rotuladas, nenhuma predita walk, e 1 ignorada predita walk. Por
    # janela, a fração walk é 10/12; por vídeo, a mediana é 0,5.
    rows = [("a", 1, k, W if k < 5 else F, W) for k in range(10)]
    rows += [("b", 2, 0, OT, OT), ("b", 2, 1, OT, F), ("b", 2, 2, IGNORE_LABEL, W)]
    return pd.DataFrame(rows, columns=["video_id", "subject", "k_end", "label", "pred_label"])


def _selftest_reconciliation_and_per_video_support() -> bool:
    aligned = _aligned()
    labeled = cast(pd.DataFrame, aligned[aligned["label"] != IGNORE_LABEL])
    stored = {"confusion_matrix": confusion_matrix(labeled["label"].to_numpy(), labeled["pred_label"].to_numpy())}
    matching = reconcile(aligned, stored, "selftest")
    shuffled = aligned.assign(pred_label=aligned["pred_label"].to_numpy()[::-1])
    divergent = reconcile(shuffled, stored, "selftest")
    videos = video_predictions(aligned)
    summary = prediction_summary(videos)
    per_video = summary["video_predicted_walk_fraction"]
    return _check(
        "a reinferência reconcilia só as janelas rotuladas com a matriz de metrics.json e acusa predições "
        "trocadas; os resumos guardam o suporte por vídeo e não confundem fração por janela com fração por vídeo",
        matching["matches_metrics_json"]
        and matching["labeled_windows"] == 12
        and matching["ignored_windows"] == 1
        and not divergent["matches_metrics_json"]
        and divergent["differing_cells"] > 0
        and [video["labeled_windows"] for video in videos] == [10, 2]
        and summary["window_predicted_walk_fraction"] == 10 / 12
        and per_video == {"n_videos": 2, "median": 0.5, "min": 0.0, "max": 1.0}
        and summary["videos_mostly_walk"] == 1,
    )


def _with_nan(loaders: FeatureLoaders, video_id: str) -> FeatureLoaders:
    def visual(requested: str) -> np.ndarray:
        values = loaders.visual(requested).copy()
        if requested == video_id:
            values[0, 0] = np.nan
        return values

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
        output_dir = root.parent / "audit"
        written = run_audit(root, frames_path, manifest_path, output_dir, loaders=recorder.loaders(),
                            quality=QUALITY, device="cpu")
        report = json.loads((output_dir / AUDIT_JSON_FILE).read_text(encoding="utf-8"))
        summary = (output_dir / AUDIT_SUMMARY_FILE).read_text(encoding="utf-8")
        skipped = not run_audit(root, frames_path, manifest_path, output_dir, loaders=recorder.loaders(),
                                quality=QUALITY, device="cpu")
        unchanged = {path: sha256_file(path) for path in artifacts} == before

        runs = [run for fold in report["folds"] for run in fold["runs"]]
        folds = {fold["fold"]: fold for fold in report["folds"]}
        environment_c = folds["retain_EnvC"]["normalization"]["sides"]
        shifted = (
            environment_c["retained"]["dinov3"]["mean_z_rms"]["min"]
            > environment_c["development_train"]["dinov3"]["mean_z_rms"]["max"]
        )
        shift_flagged = any(
            text.startswith("`retain_EnvC`, dinov3") and "Todos os vídeos retidos" in text
            for text in report["findings"]["descriptive_shifts"]
        )
        retained_windows = all(
            run["summary"]["labeled_windows"] == fold["alignment"]["recorded_retained_windows"]
            for fold in report["folds"]
            for run in fold["runs"]
        )
        clean_ok = _check(
            "a auditoria reconcilia os nove runs sintéticos com metrics.json, não acusa defeito em entradas "
            "limpas, só lê vídeos do train oficial, marca o deslocamento DINOv3 exclusivo do ambiente retido, "
            "não altera runs nem estatísticas e não sobrescreve saída existente sem --force",
            written
            and skipped
            and unchanged
            and report["official_splits_read"] == ["train"]
            and len(runs) == 9
            and all(run["reconciliation"]["matches_metrics_json"] for run in runs)
            and all(fold["alignment"]["matches_folds_json"] for fold in report["folds"])
            and retained_windows
            and report["findings"]["verified_defects"] == []
            and recorder.loaded_videos() <= set(TRAIN_VIDEOS)
            and not recorder.loaded_videos() & set(OFFICIAL_HELD_OUT_VIDEOS)
            and shifted
            and shift_flagged
            and "## Defeitos técnicos verificados" in summary
            and "## Hipóteses não resolvidas" in summary,
        )

        nan_report, _snapshot = build_report(root, frames_path, manifest_path,
                                             loaders=_with_nan(RecordingLoaders().loaders(), "EnvA/a1"),
                                             quality=QUALITY, device="cpu")
        defect_ok = _check(
            "um valor DINOv3 não finito vira defeito verificado do vídeo, e o relatório continua serializável "
            "sem NaN",
            any(defect.startswith("EnvA/a1: dinov3") for defect in nan_report["findings"]["verified_defects"])
            and json.dumps(nan_report, allow_nan=False) is not None,
        )

        refused = all(
            raises(ValueError, lambda target=target: run_audit(root, frames_path, manifest_path, target,
                                                                loaders=recorder.loaders(), quality=QUALITY,
                                                                device="cpu"), "se sobrepõe")
            for target in (root / "audit", REFERENCE_RUN_ROOT / "audit", frames_path.parent)
        )
        fold_run = fold_run_dir(root, "retain_EnvA", "B0", 42) / FOLD_RUN_FILE
        fold_run.write_text(fold_run.read_text().replace('"seed": 42', '"seed": 43'))
        tampered = raises(RuntimeError, lambda: build_report(root, frames_path, manifest_path,
                                                             loaders=recorder.loaders(), quality=QUALITY,
                                                             device="cpu"), "fold_run.json")
        (root / FOLDS_FILE).write_text("{}")
        corrupted = raises(ValueError, lambda: build_report(root, frames_path, manifest_path,
                                                            loaders=recorder.loaders(), quality=QUALITY,
                                                            device="cpu"), FOLDS_FILE)
        guard_ok = _check(
            "a auditoria recusa saída dentro das dobras, das referências ou das entradas, fold_run.json "
            "adulterado e folds.json inconsistente",
            refused and tampered and corrupted,
        )
        return clean_ok and defect_ok and guard_ok


def run_selftest() -> bool:
    results = [
        _selftest_integrity(),
        _selftest_reconciliation_and_per_video_support(),
        _selftest_end_to_end(),
    ]
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
