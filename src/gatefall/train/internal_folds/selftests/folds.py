import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from gatefall.eval.analysis.internal_fold_feasibility import read_train_partition
from gatefall.runs import LOCAL_RUN_ROOTS, REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG
from gatefall.train.baseline_b0.config import B0_FUSION_CONFIG
from gatefall.train.baseline_b1.config import B1_ADAPTIVE_GATE_CONFIG
from gatefall.train.internal_folds.cli import run_prepare
from gatefall.train.internal_folds.folds import (
    DEVELOPMENT_SPLIT,
    FOLD_RUN_ROOT,
    FOLDS_FILE,
    RETAINED_SPLIT,
    build_fold_partitions,
    fold_run_dir,
    folds_document,
    restricted_loader,
    run_plan,
    validate_output_root,
    validate_partition,
)
from gatefall.train.internal_folds.selftests.fixtures import (
    ENVIRONMENTS,
    OFFICIAL_HELD_OUT_VIDEOS,
    QUALITY,
    TRAIN_VIDEOS,
    RecordingLoaders,
    _check,
    frames_table,
    official_train_frames,
    pose_features,
    raises,
    synthetic_workspace,
)
from gatefall.train.internal_folds.stats import StatsFiles
from gatefall.train.internal_folds.training import resolve_fold_config


def _selftest_fold_membership() -> bool:
    official = official_train_frames()
    partitions = build_fold_partitions(official)
    retained = [partition.retained_environment for partition in partitions]
    sides_ok = all(
        set(partition.development_video_ids).isdisjoint(partition.retained_video_ids)
        and set(partition.development_video_ids) | set(partition.retained_video_ids) == set(TRAIN_VIDEOS)
        and all(video_id.startswith(partition.retained_environment) for video_id in partition.retained_video_ids)
        and not any(video_id.startswith(partition.retained_environment) for video_id in partition.development_video_ids)
        and sorted(partition.development_environments)
        == sorted(set(ENVIRONMENTS) - {partition.retained_environment})
        for partition in partitions
    )
    splits_ok = all(
        set(partition.frames["split"]) == {DEVELOPMENT_SPLIT, RETAINED_SPLIT}
        and set(partition.side_frames(RETAINED_SPLIT)["env"]) == {partition.retained_environment}
        for partition in partitions
    )
    tampered_frames = partitions[0].frames.copy()
    moved_video = partitions[0].retained_video_ids[0]
    tampered_frames.loc[tampered_frames["video_id"] == moved_video, "split"] = DEVELOPMENT_SPLIT
    tampered = replace(partitions[0], frames=tampered_frames)
    with_held_out = frames_table()
    return _check(
        "cada ambiente é retido em exatamente uma dobra, os lados são disjuntos, cobrem o train oficial e a "
        "coluna split da dobra separa desenvolvimento e retido; dobra adulterada e linhas fora de train são "
        "recusadas",
        retained == list(ENVIRONMENTS)
        and [partition.fold for partition in partitions] == [f"retain_{env}" for env in ENVIRONMENTS]
        and sides_ok
        and splits_ok
        and raises(ValueError, lambda: validate_partition(tampered, official), "vídeos divergentes")
        and raises(ValueError, lambda: build_fold_partitions(with_held_out), "só aceitam linhas do train"),
    )


def _selftest_train_only_inputs() -> bool:
    with synthetic_workspace() as (root, frames_path, manifest_path):
        recorder = RecordingLoaders()
        read_back = read_train_partition(frames_path)
        run_prepare(root, frames_path, manifest_path, force=False, loaders=recorder.loaders(), quality=QUALITY)
        document = json.loads((root / FOLDS_FILE).read_text(encoding="utf-8"))
        serialized = json.dumps(document)
        guarded = restricted_loader(pose_features, ["EnvA/a1"], "teste")
        return _check(
            "o prepare lê só linhas de train, nunca carrega feature de val/test oficiais, registra só train em "
            "folds.json e o loader restrito recusa vídeo fora do lado permitido",
            set(read_back["split"]) == {"train"}
            and recorder.loaded_videos() <= set(TRAIN_VIDEOS)
            and document["official_splits_read"] == ["train"]
            and document["canonical"] is False
            and not any(video_id in serialized for video_id in OFFICIAL_HELD_OUT_VIDEOS)
            and raises(RuntimeError, lambda: guarded("EnvB/b1"), "fora do lado permitido"),
        )


def _selftest_output_root_protection() -> bool:
    frames_path = REPOSITORY_ROOT / "data/processed/le2i_cv/frames.parquet"
    manifest_path = REPOSITORY_ROOT / "data/processed/le2i_cv/manifest.parquet"
    refused = [
        REFERENCE_RUN_ROOT,
        REFERENCE_RUN_ROOT / "le2i_cv",
        REPOSITORY_ROOT / LOCAL_RUN_ROOTS["le2i-cv"],
        REPOSITORY_ROOT / LOCAL_RUN_ROOTS["le2i-cv"] / "folds",
        REPOSITORY_ROOT / LOCAL_RUN_ROOTS["le2i"],
        REPOSITORY_ROOT / "runs/local",
        REPOSITORY_ROOT / "runs",
        REPOSITORY_ROOT / "src/gatefall/features/stats",
        REPOSITORY_ROOT / "data",
        REPOSITORY_ROOT,
    ]
    refused_ok = all(
        raises(ValueError, lambda path=path: validate_output_root(path, frames_path, manifest_path), "protegido")
        for path in refused
    )
    accepted_ok = not raises(
        ValueError, lambda: validate_output_root(REPOSITORY_ROOT / FOLD_RUN_ROOT, frames_path, manifest_path)
    )
    return _check(
        "a raiz das dobras recusa referências, runs locais oficiais de le2i e le2i-cv, estatísticas oficiais, "
        "src/, data/ e qualquer ancestral deles, e aceita runs/local/le2i_cv_internal_folds",
        refused_ok
        and accepted_ok
        and raises(ValueError, lambda: fold_run_dir(Path("x"), "retain_EnvA", "C0", 42), "não suportada"),
    )


def _selftest_prepare_output_protection() -> bool:
    with synthetic_workspace() as (root, frames_path, manifest_path):
        loaders = RecordingLoaders().loaders()
        run_prepare(root, frames_path, manifest_path, force=False, loaders=loaders, quality=QUALITY)
        before = (root / FOLDS_FILE).read_bytes()
        rerun = RecordingLoaders()
        run_prepare(root, frames_path, manifest_path, force=False, loaders=rerun.loaders(), quality=QUALITY)
        idempotent = (root / FOLDS_FILE).read_bytes() == before and not rerun.loaded_videos()

        stats_file = root / "retain_EnvA" / "stats" / "pose.json"
        original_stats = stats_file.read_text(encoding="utf-8")
        stats_file.write_text(original_stats.replace('"window_count"', '"window_count" ', 1), encoding="utf-8")
        tampered_refused = raises(
            RuntimeError,
            lambda: run_prepare(root, frames_path, manifest_path, force=False, loaders=loaders, quality=QUALITY),
            "use --force",
        )
        tampered_kept = stats_file.read_text(encoding="utf-8") != original_stats

        other_quality = replace(QUALITY, sha256="1" * 64)
        stats_file.write_text(original_stats, encoding="utf-8")
        document_refused = raises(
            RuntimeError,
            lambda: run_prepare(root, frames_path, manifest_path, force=False, loaders=loaders, quality=other_quality),
            "diverge",
        )
        run_prepare(root, frames_path, manifest_path, force=True, loaders=loaders, quality=other_quality)
        forced = json.loads((root / FOLDS_FILE).read_text(encoding="utf-8"))["quality_features_sha256"] == "1" * 64
        return _check(
            "prepare é idempotente sem --force, não sobrescreve estatísticas adulteradas nem folds.json "
            "divergente, e só --force refaz",
            idempotent and tampered_refused and tampered_kept and document_refused and forced,
        )


ALLOWED_RECIPE_DIFFERENCES = {
    "A": {"seed", "standardization_stats_path", "standardization_stats_sha256"},
    "B0": {
        "seed",
        "pose_standardization_stats_path",
        "pose_standardization_stats_sha256",
        "visual_standardization_stats_path",
        "visual_standardization_stats_sha256",
    },
    "B1": {
        "seed",
        "pose_standardization_stats_path",
        "pose_standardization_stats_sha256",
        "visual_standardization_stats_path",
        "visual_standardization_stats_sha256",
        "quality_features_path",
        "quality_features_sha256",
    },
}
CANONICAL_CONFIGS = {"A": BASELINE_A_CONFIG, "B0": B0_FUSION_CONFIG, "B1": B1_ADAPTIVE_GATE_CONFIG}


def _selftest_deterministic_configuration() -> bool:
    with synthetic_workspace() as (root, _frames_path, _manifest_path):
        pilot = run_plan(root, "pilot")
        assessment = run_plan(root, "assessment")
        plan_ok = (
            pilot == run_plan(root, "pilot")
            and len(pilot) == 9
            and len(assessment) == 45
            and len({cast(str, entry["run_dir"]) for entry in assessment}) == 45
            and [entry["arm"] for entry in pilot[:3]] == ["A", "B0", "B1"]
            and {entry["seed"] for entry in assessment} == {42, 43, 44, 45, 46}
        )
        files = StatsFiles(
            pose_path=root / "retain_EnvA/stats/pose.json",
            pose_sha256="a" * 64,
            dinov3_path=root / "retain_EnvA/stats/dinov3.json",
            dinov3_sha256="b" * 64,
        )
        recipe_ok = True
        for arm, canonical in CANONICAL_CONFIGS.items():
            config = resolve_fold_config(arm, 43, files, QUALITY)
            repeated = resolve_fold_config(arm, 43, files, QUALITY)
            differing = {
                field for field, value in canonical.to_dict().items() if config.to_dict()[field] != value
            }
            recipe_ok &= config == repeated and differing <= ALLOWED_RECIPE_DIFFERENCES[arm] and config.seed == 43
        partitions = build_fold_partitions(official_train_frames())
        repeated_partitions = build_fold_partitions(official_train_frames())
        first = json.dumps(folds_document(partitions, {"frames": "x"}, "q", "0" * 64), sort_keys=True)
        second = json.dumps(folds_document(repeated_partitions, {"frames": "x"}, "q", "0" * 64), sort_keys=True)
        return _check(
            "plano de runs (9 no piloto, 45 na avaliação), configurações por arma e folds.json são "
            "determinísticos, e a receita só difere da canônica em seed e caminhos/hashes de estatística e qualidade",
            plan_ok and recipe_ok and first == second,
        )


def run_folds_selftest() -> bool:
    results = [
        _selftest_fold_membership(),
        _selftest_train_only_inputs(),
        _selftest_output_root_protection(),
        _selftest_prepare_output_protection(),
        _selftest_deterministic_configuration(),
    ]
    return all(results)
