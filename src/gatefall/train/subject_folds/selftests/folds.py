import json
import shutil
from dataclasses import replace
from typing import cast

from gatefall.eval.analysis.internal_fold_feasibility import read_train_partition
from gatefall.eval.analysis.subject_fold_feasibility import validate_subject_identifiers
from gatefall.hashing import sha256_file
from gatefall.runs import ARM_RUN_DIR_NAMES, LOCAL_RUN_ROOTS, REFERENCE_RUN_ROOT, REPOSITORY_ROOT
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG
from gatefall.train.internal_folds.folds import DEVELOPMENT_SPLIT, RETAINED_SPLIT, restricted_loader
from gatefall.train.internal_folds.folds import FOLD_RUN_ROOT as LE2I_CV_FOLD_RUN_ROOT
from gatefall.train.internal_folds.selftests.fixtures import _check, raises
from gatefall.train.subject_folds.cli import run_prepare
from gatefall.train.subject_folds.folds import (
    FOLDS_FILE,
    RETAINED_SUBJECT_GROUPS,
    SUBJECT_FOLD_ROOT,
    build_subject_fold_partitions,
    fold_run_dir,
    folds_document,
    run_plan,
    stats_dir,
    validate_output_root,
    validate_partition,
)
from gatefall.train.subject_folds.selftests.fixtures import (
    OFFICIAL_HELD_OUT_VIDEOS,
    SD,
    SG,
    SUBJECT_BY_VIDEO,
    TRAIN_VIDEOS,
    W,
    RecordingPose,
    frames_table,
    manifest_table,
    official_train_frames,
    pose_features,
    relabel,
    synthetic_workspace,
)
from gatefall.train.subject_folds.stats import OFFICIAL_POSE_STATS, POSE_STATS_FILE, PROVENANCE_FILE
from gatefall.train.subject_folds.training import resolve_fold_config


def _selftest_fold_membership() -> bool:
    official = official_train_frames()
    partitions = build_subject_fold_partitions(official)
    sides_ok = all(
        set(partition.development_video_ids).isdisjoint(partition.retained_video_ids)
        and set(partition.development_video_ids) | set(partition.retained_video_ids) == set(TRAIN_VIDEOS)
        and {SUBJECT_BY_VIDEO[video_id] for video_id in partition.retained_video_ids}
        == set(partition.retained_subjects)
        and {SUBJECT_BY_VIDEO[video_id] for video_id in partition.development_video_ids}
        == set(partition.development_subjects)
        and set(partition.frames["split"]) == {DEVELOPMENT_SPLIT, RETAINED_SPLIT}
        for partition in partitions
    )
    retained_union = [video_id for partition in partitions for video_id in partition.retained_video_ids]
    return _check(
        "os grupos congelados {0}, {1} e {3,4,6,8} formam três dobras com sujeitos e vídeos disjuntos entre os "
        "lados, cada vídeo de train é retido em exatamente uma dobra e a coluna split separa os lados",
        [partition.fold for partition in partitions] == ["retain_s0", "retain_s1", "retain_s3-4-6-8"]
        and [partition.retained_subjects for partition in partitions] == list(RETAINED_SUBJECT_GROUPS)
        and sorted(retained_union) == sorted(TRAIN_VIDEOS)
        and sides_ok,
    )


def _selftest_leakage_guards() -> bool:
    official = official_train_frames()
    partition = build_subject_fold_partitions(official)[0]
    moved_frames = partition.frames.copy()
    moved_frames.loc[moved_frames["video_id"] == partition.retained_video_ids[0], "split"] = DEVELOPMENT_SPLIT
    moved = replace(partition, frames=moved_frames)
    shared_subject = replace(partition, development_subjects=(*partition.development_subjects, 0))
    two_subjects = official.copy()
    two_subjects.loc[(two_subjects["video_id"] == "EnvA/s0a") & (two_subjects["frame_index"] == 0), "subject"] = 1
    guarded = restricted_loader(pose_features, ["EnvA/s1a"], "teste")
    cases = {
        "vídeo movido de lado": raises(ValueError, lambda: validate_partition(moved, official), "vídeos divergentes"),
        "sujeito nos dois lados": raises(
            ValueError, lambda: validate_partition(shared_subject, official), "sujeito retido"
        ),
        "linhas de val/test": raises(
            ValueError, lambda: build_subject_fold_partitions(frames_table()), "só aceitam linhas do train"
        ),
        "grupos sobrepostos": raises(
            ValueError,
            lambda: build_subject_fold_partitions(official, ((0,), (0, 1), (3, 4, 6, 8))),
            "repetem sujeito",
        ),
        "grupos sem cobrir os sujeitos": raises(
            ValueError, lambda: build_subject_fold_partitions(official, ((0,), (1,), (3, 4, 6))), "não cobrem"
        ),
        "sujeito de fora dos grupos": raises(
            ValueError,
            lambda: build_subject_fold_partitions(official.assign(subject=official["subject"].replace(8, 9))),
            "não cobrem",
        ),
        "vídeo com dois sujeitos": raises(
            ValueError,
            lambda: validate_subject_identifiers(two_subjects, manifest_table()),
            "ambíguo",
        ),
        "loader fora do lado": raises(RuntimeError, lambda: guarded("EnvA/s0a"), "fora do lado permitido"),
    }
    failed = [name for name, ok in cases.items() if not ok]
    if failed:
        print(f"  vazamentos não recusados: {failed}")
    return _check(
        "vídeo nos dois lados, sujeito nos dois lados, linhas fora de train, grupos sobrepostos ou incompletos, "
        "sujeito fora dos grupos, vídeo com dois sujeitos e leitura de feature fora do lado são recusados",
        not failed,
    )


def _selftest_missing_classes() -> bool:
    official = official_train_frames()
    retained_missing = relabel(official, {3, 4, 6, 8}, SG, W)
    development_missing = relabel(official, {1, 3, 4, 6, 8}, SD, W)
    return _check(
        "dobra sem uma classe restrita no retido (standing em {3,4,6,8}) ou no treino de desenvolvimento "
        "(sit_down fora do sujeito 0) é recusada antes de gravar qualquer coisa",
        raises(
            ValueError,
            lambda: build_subject_fold_partitions(retained_missing),
            "retain_s3-4-6-8: classes restritas ausentes em retained: ['standing']",
        )
        and raises(
            ValueError,
            lambda: build_subject_fold_partitions(development_missing),
            "retain_s0: classes restritas ausentes em development_train: ['sit_down']",
        ),
    )


def _selftest_train_only_inputs() -> bool:
    with synthetic_workspace() as (root, frames_path, manifest_path, pose_root):
        recorder = RecordingPose()
        read_back = read_train_partition(frames_path)
        run_prepare(root, frames_path, manifest_path, pose_root, force=False, pose=recorder.inputs())
        document = json.loads((root / FOLDS_FILE).read_text(encoding="utf-8"))
        serialized = json.dumps(document)
        return _check(
            "o prepare lê só linhas de train, nunca carrega pose de val/test oficiais e registra só train, os "
            "grupos congelados, input_sha256 e membership_sha256 em folds.json",
            set(read_back["split"]) == {"train"}
            and recorder.loaded_videos() <= set(TRAIN_VIDEOS)
            and document["official_splits_read"] == ["train"]
            and document["canonical"] is False
            and document["retained_subject_groups"] == [[0], [1], [3, 4, 6, 8]]
            and sorted(document["input_sha256"]) == ["frames", "manifest"]
            and document["input_sha256"]["frames"] == sha256_file(frames_path)
            and len(document["membership_sha256"]) == 64
            and not any(video_id in serialized for video_id in OFFICIAL_HELD_OUT_VIDEOS),
        )


def _selftest_protected_paths() -> bool:
    frames_path = REPOSITORY_ROOT / "data/processed/le2i/frames.parquet"
    manifest_path = REPOSITORY_ROOT / "data/processed/le2i/manifest.parquet"
    pose_root = REPOSITORY_ROOT / "data/features/le2i/pose"
    le2i_root = REPOSITORY_ROOT / LOCAL_RUN_ROOTS["le2i"]
    refused = [
        REFERENCE_RUN_ROOT,
        REFERENCE_RUN_ROOT / "le2i",
        le2i_root,
        *(le2i_root / name for name in ARM_RUN_DIR_NAMES.values()),
        le2i_root / "baseline_a_multiseed",
        REPOSITORY_ROOT / LOCAL_RUN_ROOTS["le2i-cv"],
        REPOSITORY_ROOT / LE2I_CV_FOLD_RUN_ROOT,
        REPOSITORY_ROOT / LE2I_CV_FOLD_RUN_ROOT / "retain_Lecture_room",
        REPOSITORY_ROOT / "runs/local/subject_fold_feasibility",
        REPOSITORY_ROOT / "runs/local",
        REPOSITORY_ROOT / "runs",
        REPOSITORY_ROOT / "src/gatefall/features/stats",
        REPOSITORY_ROOT / "data",
        REPOSITORY_ROOT / "docs",
        REPOSITORY_ROOT,
    ]
    accepted = [REPOSITORY_ROOT / SUBJECT_FOLD_ROOT, REPOSITORY_ROOT / SUBJECT_FOLD_ROOT / "attempt"]
    wrongly_accepted = [
        str(path) for path in refused
        if not raises(ValueError, lambda path=path: validate_output_root(path, frames_path, manifest_path, pose_root))
    ]
    wrongly_refused = [
        str(path) for path in accepted
        if raises(ValueError, lambda path=path: validate_output_root(path, frames_path, manifest_path, pose_root))
    ]
    if wrongly_accepted or wrongly_refused:
        print(f"  aceitos indevidamente: {wrongly_accepted}; recusados indevidamente: {wrongly_refused}")
    root = REPOSITORY_ROOT / SUBJECT_FOLD_ROOT
    return _check(
        "a raiz das dobras recusa referências, runs canônicos do Le2i-CS, runs e dobras do Le2i-CV, estatísticas "
        "oficiais, src/, data/, docs/, qualquer ancestral deles e outros diretórios de runs/local; aceita só "
        "runs/local/le2i/subject_folds; run_dir recusa braço, seed ou dobra fora do contrato",
        not wrongly_accepted
        and not wrongly_refused
        and raises(ValueError, lambda: fold_run_dir(root, "retain_s0", "B0", 42), "arma não suportada")
        and raises(ValueError, lambda: fold_run_dir(root, "retain_s0", "A", 43), "seed não suportada")
        and raises(ValueError, lambda: fold_run_dir(root, "retain_s2", "A", 42), "dobra desconhecida"),
    )


def _selftest_prepare_protection() -> bool:
    with synthetic_workspace() as (root, frames_path, manifest_path, pose_root):
        run_prepare(root, frames_path, manifest_path, pose_root, force=False, pose=RecordingPose().inputs())
        before = (root / FOLDS_FILE).read_bytes()
        rerun = RecordingPose()
        run_prepare(root, frames_path, manifest_path, pose_root, force=False, pose=rerun.inputs())
        idempotent = (root / FOLDS_FILE).read_bytes() == before and not rerun.loaded_videos()

        stats_file = stats_dir(root, "retain_s0") / POSE_STATS_FILE
        original_stats = stats_file.read_text(encoding="utf-8")
        stats_file.write_text(original_stats.replace('"window_count"', '"window_count" ', 1), encoding="utf-8")
        tampered_refused = raises(
            RuntimeError,
            lambda: run_prepare(root, frames_path, manifest_path, pose_root, False, RecordingPose().inputs()),
            "use --force",
        )
        tampered_kept = stats_file.read_text(encoding="utf-8") != original_stats
        stats_file.write_text(original_stats, encoding="utf-8")

        changed_pose = RecordingPose(sha256="1" * 64)
        document_refused = raises(
            RuntimeError,
            lambda: run_prepare(root, frames_path, manifest_path, pose_root, False, changed_pose.inputs()),
            "diverge",
        )
        run_prepare(root, frames_path, manifest_path, pose_root, force=True, pose=changed_pose.inputs())
        forced = json.loads((root / FOLDS_FILE).read_text(encoding="utf-8"))["pose_features_sha256"] == "1" * 64

        official_refused = True
        if OFFICIAL_POSE_STATS.is_file():
            shutil.copyfile(OFFICIAL_POSE_STATS, stats_file)
            provenance_path = stats_dir(root, "retain_s0") / PROVENANCE_FILE
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["pose_stats_sha256"] = sha256_file(stats_file)
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            official_refused = raises(
                RuntimeError,
                lambda: run_prepare(root, frames_path, manifest_path, pose_root, False, changed_pose.inputs()),
                "estatística oficial",
            )
        return _check(
            "prepare é idempotente sem --force, não sobrescreve estatística adulterada nem folds.json divergente, "
            "recusa a estatística oficial do Le2i-CS copiada para a dobra e só --force refaz",
            idempotent and tampered_refused and tampered_kept and document_refused and forced and official_refused,
        )


def _selftest_deterministic_planning() -> bool:
    with synthetic_workspace() as (root, _frames_path, _manifest_path, _pose_root):
        plan = run_plan(root)
        plan_ok = (
            plan == run_plan(root)
            and len(plan) == 3
            and [entry["fold"] for entry in plan] == ["retain_s0", "retain_s1", "retain_s3-4-6-8"]
            and {(entry["arm"], entry["seed"]) for entry in plan} == {("A", 42)}
            and len({cast(str, entry["run_dir"]) for entry in plan}) == 3
        )
        stats_path = stats_dir(root, "retain_s0") / POSE_STATS_FILE
        config = resolve_fold_config(42, stats_path, "a" * 64)
        differing = {
            field for field, value in BASELINE_A_CONFIG.to_dict().items() if config.to_dict()[field] != value
        }
        recipe_ok = (
            config == resolve_fold_config(42, stats_path, "a" * 64)
            and differing == {"standardization_stats_path", "standardization_stats_sha256"}
            and config.epochs == 30
            and config.class_weighted is True
            and config.optimizer_name == "adamw"
            and raises(ValueError, lambda: resolve_fold_config(43, stats_path, "a" * 64), "seed não suportada")
            and raises(ValueError, lambda: resolve_fold_config(42, OFFICIAL_POSE_STATS, "a" * 64), "oficial")
        )
        inputs = {"frames": "1" * 64, "manifest": "2" * 64}
        first = folds_document(build_subject_fold_partitions(official_train_frames()), inputs, "3" * 64)
        second = folds_document(build_subject_fold_partitions(official_train_frames()), inputs, "3" * 64)
        reordered = official_train_frames().sample(frac=1.0, random_state=7).reset_index(drop=True)
        third = folds_document(build_subject_fold_partitions(reordered), inputs, "3" * 64)
        return _check(
            "plano (3 runs: A, seed 42), receita e folds.json são determinísticos e independentes da ordem das "
            "linhas; a receita só difere da canônica de A no caminho e hash da estatística, e recusa seed fora do "
            "contrato e a estatística oficial",
            plan_ok
            and recipe_ok
            and json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
            and json.dumps(first, sort_keys=True) == json.dumps(third, sort_keys=True),
        )


def run_folds_selftest() -> bool:
    results = [
        _selftest_fold_membership(),
        _selftest_leakage_guards(),
        _selftest_missing_classes(),
        _selftest_train_only_inputs(),
        _selftest_protected_paths(),
        _selftest_prepare_protection(),
        _selftest_deterministic_planning(),
    ]
    return all(results)
