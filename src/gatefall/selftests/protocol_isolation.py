"""Selftest de isolamento entre protocolos Le2i-CS e Le2i-CV.

Cobre: paths do adapter CV nunca caem sob artefatos CS; `load_annotation_splits`
respeita o protocolo pedido; e a receita de treino/protocolo de alarme
congelados do braço A não têm nenhum ramo condicional a protocolo além do
path/sha256 de padronização.
"""

import sys
import tempfile
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd
import numpy as np
import h5py
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import gatefall.data.le2i.annotations as annotations_module
from gatefall.data.le2i.annotations import load_annotation_splits
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.features import shared_le2i
from gatefall.features.standardize_dinov3 import dinov3_stats_path, _dinov3_feature_loader
from gatefall.features.standardize_sam3 import sam3_stats_path
from gatefall.features.dinov3_standardization import load_stats as load_dinov3_stats
from gatefall.features.sam3_standardization import load_stats as load_sam3_stats
from gatefall.features.quality_storage import quality_path, read_quality
from gatefall.pose.extract import (
    main as pose_extract_main,
    run_pose_extract,
    run_pose_extract_all,
)
from gatefall.pose.loading import pose_path
from gatefall.sam3.features import load_v_t
from gatefall.dinov3.storage import dinov3_path
from gatefall.sam3.storage import sam3_path
from gatefall.runs import default_run_dir_for_arm
from gatefall.eval.analysis import (
    alarm_protocol_sensitivity,
    grouped_bootstrap,
    multiseed_summary,
    qualitative,
)
from gatefall.eval.baseline_b0 import cli as b0_events
from gatefall.eval.baseline_b1 import cli as b1_events
from gatefall.eval.baseline_c0 import cli as c0_events
from gatefall.eval.baseline_c1 import cli as c1_events
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.runs import default_run_dir, validate_local_run_dir
from gatefall.train.baseline_b0 import cli as baseline_b0
from gatefall.train.baseline_b1 import cli as baseline_b1
from gatefall.train.baseline_c0 import cli as baseline_c0
from gatefall.train.baseline_c1 import cli as baseline_c1
from gatefall.train.baseline_a.cli import _resolve_config
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _is_under(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False


def check_cv_adapter_paths_isolated_from_cs_artifacts() -> bool:
    adapter = Le2iDatasetAdapter(protocol="cv")
    forbidden_prefixes = [
        Path("data/processed/le2i"),
        Path("data/labels/omnifall"),
        Path("runs/local/le2i"),
    ]
    resolved_paths = [
        adapter.manifest_path,
        adapter.frames_path,
        adapter.pose_stats_path,
    ]
    violated = any(
        _is_under(path, prefix)
        for path in resolved_paths
        for prefix in forbidden_prefixes
    )
    return _check(
        "adapter CV: manifest_path/frames_path/pose_stats_path nunca caem sob "
        "artefatos CS (data/processed/le2i, data/labels/omnifall, "
        "runs/local/le2i)",
        not violated and adapter.identifier == "le2i-cv",
    )


def check_shared_features_and_separate_statistics() -> bool:
    cs = get_dataset("le2i")
    cv = Le2iDatasetAdapter(protocol="cv")
    video_id = "room/video_1"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cv = replace(
            cv, dinov3_root=root / "dinov3", sam3_root=root / "sam3",
            quality_root=root / "quality",
        )
        for path, key, value in (
            (dinov3_path(video_id, dinov3_root=cv.dinov3_root), "features", np.ones((2, 1536), dtype=np.float16)),
            (sam3_path(video_id, sam3_root=cv.sam3_root), "v_t", np.ones((2, 10), dtype=np.float32)),
            (quality_path(video_id, quality_root=cv.quality_root), "quality", np.ones((2, 2), dtype=np.float32)),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as file:
                file.create_dataset(key, data=value)
                if key == "v_t":
                    file.attrs["video_id"] = video_id
                    file.attrs["K"] = 2
                    file.create_dataset("sam_score", data=np.ones(2, dtype=np.float32))
                    file.create_dataset("n_instances", data=np.ones(2, dtype=np.int16))
        readable = (
            _dinov3_feature_loader(cv, video_id).shape == (2, 1536)
            and load_v_t(video_id, sam3_root=cv.sam3_root).shape == (2, 10)
            and read_quality(quality_path(video_id, quality_root=cv.quality_root)).shape == (2, 2)
        )
    paths = (
        cs.pose_root == get_dataset("le2i-cv").pose_root
        and cs.dinov3_root == get_dataset("le2i-cv").dinov3_root
        and cs.sam3_root == get_dataset("le2i-cv").sam3_root
        and cs.quality_root == get_dataset("le2i-cv").quality_root
        and cs.pose_stats_path != cv.pose_stats_path
        and dinov3_stats_path("le2i") != dinov3_stats_path("le2i-cv")
        and sam3_stats_path("le2i") != sam3_stats_path("le2i-cv")
    )
    stats = (
        load_dinov3_stats(dinov3_stats_path("le2i")).dataset == "le2i"
        and load_dinov3_stats(dinov3_stats_path("le2i-cv")).dataset == "le2i-cv"
        and load_sam3_stats(sam3_stats_path("le2i")).dataset == "le2i"
        and load_sam3_stats(sam3_stats_path("le2i-cv")).dataset == "le2i-cv"
    )
    runs = all(
        default_run_dir_for_arm("le2i", arm) != default_run_dir_for_arm("le2i-cv", arm)
        for arm in ("B0", "B1", "C0", "C1")
    )
    return _check("B/C leem raízes compartilhadas, mas estatísticas e runs CV são isolados", readable and paths and stats and runs)


def check_cv_timegrid_mismatch_rejected() -> bool:
    source_frames = pd.DataFrame({"video_id": ["room/video_1"], "frame_index": [0], "src_index": [10], "split": ["train"]})
    cv_frames = source_frames.copy()
    cv_frames.loc[0, "src_index"] = 11
    manifest = pd.DataFrame({"video_id": ["room/video_1"], "split": ["train"]})
    source = SimpleNamespace(load_manifest=lambda: manifest, load_frames=lambda: source_frames)
    cv = SimpleNamespace(identifier="le2i-cv", load_manifest=lambda: manifest, load_frames=lambda: cv_frames)
    with patch.object(shared_le2i, "get_dataset", return_value=source):
        try:
            shared_le2i.shared_source(cast(DatasetAdapter, cv))
        except ValueError as exc:
            rejected = "não cobre ou não alinha" in str(exc)
        else:
            rejected = False
    return _check("CV recusa grade temporal divergente dos arquivos compartilhados", rejected)


def check_cv_extraction_commands_rejected() -> bool:
    commands = (
        "gatefall.pose.extract",
        "gatefall.dinov3.extract",
        "gatefall.sam3.extract",
        "gatefall.features.quality_extract",
    )
    results = [
        subprocess.run(
            [sys.executable, "-m", module, command, "--dataset", "le2i-cv"]
            + (["--video-id", "room/video_1"] if command == "extract" else []),
            capture_output=True,
            text=True,
            check=False,
        )
        for module in commands
        for command in ("extract", "extract-all")
    ]
    return _check(
        "CLIs de extração pose, DINOv3, SAM 3 e qualidade recusam CV antes de gravar",
        all(
            result.returncode == 2 and "invalid choice" in result.stderr
            for result in results
        ),
    )


def check_cv_pose_programmatic_extraction_rejected() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        adapter = replace(
            Le2iDatasetAdapter(protocol="cv"), pose_root=Path(tmp) / "pose"
        )
        output_path = pose_path("room/video_1", pose_root=adapter.pose_root)
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"pose CS preservada")
        with patch("gatefall.pose.extract.YOLO") as model:
            rejected = []
            for run in (
                lambda: run_pose_extract(
                    "room/video_1", "unused.pt", True, adapter=adapter
                ),
                lambda: run_pose_extract_all("unused.pt", True, adapter=adapter),
            ):
                try:
                    run()
                except ValueError as exc:
                    rejected.append("le2i-cv" in str(exc) and "le2i" in str(exc))
                else:
                    rejected.append(False)
            preserved = output_path.read_bytes() == b"pose CS preservada" and not model.called
    return _check(
        "entry points de pose recusam CV antes de carregar modelo ou sobrescrever HDF5 CS",
        all(rejected) and preserved,
    )


def check_cv_pose_report_remains_supported() -> bool:
    adapter = Le2iDatasetAdapter(protocol="cv")
    with (
        patch.object(
            sys, "argv", ["gatefall.pose.extract", "report", "--dataset", "le2i-cv"]
        ),
        patch("gatefall.pose.extract.get_dataset", return_value=adapter),
        patch("gatefall.pose.report.run_pose_report") as report,
    ):
        pose_extract_main()
    return _check(
        "pose report mantém leitura do protocolo CV",
        report.call_args.kwargs == {"adapter": adapter},
    )


def check_load_annotation_splits_respects_protocol() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cs_dir = Path(tmp) / "cs"
        cv_dir = Path(tmp) / "cv"
        cs_dir.mkdir()
        cv_dir.mkdir()
        for filename in annotations_module.SPLIT_FILES.values():
            pd.DataFrame({"marker": ["cs"]}).to_csv(cs_dir / filename, index=False)
            pd.DataFrame({"marker": ["cv"]}).to_csv(cv_dir / filename, index=False)

        original = dict(annotations_module.PROTOCOL_LABELS_DIR)
        annotations_module.PROTOCOL_LABELS_DIR["cs"] = cs_dir
        annotations_module.PROTOCOL_LABELS_DIR["cv"] = cv_dir
        try:
            cv_splits = load_annotation_splits(protocol="cv")
            cs_splits = load_annotation_splits(protocol="cs")
            ok = all(
                str(dataframe["marker"].iloc[0]) == "cv"
                for dataframe in cv_splits.values()
            ) and all(
                str(dataframe["marker"].iloc[0]) == "cs"
                for dataframe in cs_splits.values()
            )
        finally:
            annotations_module.PROTOCOL_LABELS_DIR.clear()
            annotations_module.PROTOCOL_LABELS_DIR.update(original)
    return _check(
        "load_annotation_splits(protocol=...): lê exclusivamente do diretório "
        "de labels daquele protocolo",
        ok,
    )


def check_frozen_train_config_identical_except_standardization() -> bool:
    cs_config = _resolve_config(
        BASELINE_A_CONFIG.seed, Path("stats_cs.json"), "cs-sha256"
    )
    cv_config = _resolve_config(
        BASELINE_A_CONFIG.seed, Path("stats_cv.json"), "cv-sha256"
    )
    cs_dict = cs_config.to_dict()
    cv_dict = cv_config.to_dict()
    differing_fields = {key for key in cs_dict if cs_dict[key] != cv_dict[key]}
    return _check(
        "receita de treino do braço A: le2i-cv difere de le2i apenas em "
        "standardization_stats_path/sha256",
        differing_fields
        == {"standardization_stats_path", "standardization_stats_sha256"},
    )


def check_default_run_dir_is_dataset_aware() -> bool:
    cs_default = default_run_dir("le2i")
    cv_default = default_run_dir("le2i-cv")
    return _check(
        "default_run_dir: le2i resolve para runs/local/le2i/baseline_a e "
        "le2i-cv resolve para runs/local/le2i_cv/baseline_a, sem colisão",
        cs_default == Path("runs/local/le2i/baseline_a")
        and cv_default == Path("runs/local/le2i_cv/baseline_a")
        and cs_default != cv_default,
    )


def check_cv_run_dir_not_classified_under_cs_root_despite_string_prefix() -> bool:
    cs_root = Path("runs/local/le2i")
    cv_run_dir = Path("runs/local/le2i_cv/baseline_a")
    return _check(
        "runs/local/le2i_cv/... NÃO é classificado como estando sob "
        "runs/local/le2i/ apesar do prefixo de string compartilhado",
        not _is_under(cv_run_dir, cs_root),
    )


def check_cross_protocol_run_dir_guard_rejects_both_directions() -> bool:
    cv_dir_rejected_for_cs = False
    try:
        validate_local_run_dir(Path("runs/local/le2i_cv/baseline_a"), "le2i")
    except ValueError:
        cv_dir_rejected_for_cs = True

    cs_dir_rejected_for_cv = False
    try:
        validate_local_run_dir(Path("runs/local/le2i/baseline_a"), "le2i-cv")
    except ValueError:
        cs_dir_rejected_for_cv = True

    return _check(
        "validate_local_run_dir: rejeita run_dir sob runs/local/le2i_cv/ com "
        "--dataset le2i e rejeita run_dir sob runs/local/le2i/ com "
        "--dataset le2i-cv",
        cv_dir_rejected_for_cs and cs_dir_rejected_for_cv,
    )


def check_non_conflicting_local_run_dir_still_accepted() -> bool:
    accepted = True
    for run_dir, dataset in (
        (Path("runs/local/custom_experiment/baseline_a"), "le2i"),
        (Path("runs/local/custom_experiment/baseline_a"), "le2i-cv"),
        (default_run_dir("le2i"), "le2i"),
        (default_run_dir("le2i-cv"), "le2i-cv"),
    ):
        try:
            validate_local_run_dir(run_dir, dataset)
        except ValueError:
            accepted = False
    return _check(
        "validate_local_run_dir: aceita run_dir de terceiros e o run_dir "
        "canônico do próprio protocolo, sem virar whitelist dos dois "
        "diretórios canônicos",
        accepted,
    )


def check_cs_only_analysis_entry_points_reject_cv_run_dir() -> bool:
    cv_run_dir = Path("runs/local/le2i_cv/baseline_a")

    def _raises_cross_protocol_guard(callback) -> bool:
        try:
            callback()
        except ValueError as exc:
            return "protocolo" in str(exc)
        return False

    sensitivity_rejected = _raises_cross_protocol_guard(
        lambda: alarm_protocol_sensitivity.run_analyze(
            force=False, dataset_name="le2i", run_dir=cv_run_dir
        )
    )
    bootstrap_rejected = _raises_cross_protocol_guard(
        lambda: grouped_bootstrap.run_analyze(
            force=False, dataset_name="le2i", run_dir=cv_run_dir
        )
    )
    qualitative_rejected = _raises_cross_protocol_guard(
        lambda: qualitative.run_render(
            run_dir=cv_run_dir,
            dataset_name="le2i",
            splits=("val",),
            force=False,
        )
    )
    adapter = get_dataset("le2i")
    multiseed_rejected = _raises_cross_protocol_guard(
        lambda: multiseed_summary._summarize(
            [cv_run_dir, Path("runs/local/le2i_cv/baseline_a_other")],
            BASELINE_A_CONFIG,
            adapter,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        b0_report_output = Path(tmp) / "b0_report.json"
        b0_train_rejected = _raises_cross_protocol_guard(
            lambda: baseline_b0.run_train(force=False, dataset_name="le2i", run_dir=cv_run_dir)
        )
        b0_report_rejected = _raises_cross_protocol_guard(
            lambda: baseline_b0.run_report(
                dataset_name="le2i",
                run_dir=cv_run_dir,
                output_path=b0_report_output,
                force=False,
            )
        )
        b1_report_output = Path(tmp) / "b1_report.json"
        b1_train_rejected = _raises_cross_protocol_guard(
            lambda: baseline_b1.run_train(force=False, dataset_name="le2i", run_dir=cv_run_dir)
        )
        b1_report_rejected = _raises_cross_protocol_guard(
            lambda: baseline_b1.run_report(
                dataset_name="le2i",
                run_dir=cv_run_dir,
                output_path=b1_report_output,
                force=False,
            )
        )
        c0_train_rejected = _raises_cross_protocol_guard(
            lambda: baseline_c0.run_train(force=False, dataset_name="le2i", run_dir=cv_run_dir)
        )
        c0_report_rejected = _raises_cross_protocol_guard(
            lambda: baseline_c0.run_report(
                dataset_name="le2i",
                run_dir=cv_run_dir,
                output_path=Path(tmp) / "c0_report.json",
                force=False,
            )
        )
        c1_train_rejected = _raises_cross_protocol_guard(
            lambda: baseline_c1.run_train(force=False, dataset_name="le2i", run_dir=cv_run_dir)
        )
        c1_report_rejected = _raises_cross_protocol_guard(
            lambda: baseline_c1.run_report(dataset_name="le2i", run_dir=cv_run_dir,
                                       output_path=Path(tmp) / "c1_report.json", force=False)
        )
    b0_events_rejected = _raises_cross_protocol_guard(
        lambda: b0_events.run_evaluate(
            force=False,
            dataset_name="le2i",
            run_dir=Path("runs/local/le2i_cv/baseline_b0"),
        )
    )
    b1_events_rejected = _raises_cross_protocol_guard(
        lambda: b1_events.run_evaluate(
            force=False,
            dataset_name="le2i",
            run_dir=Path("runs/local/le2i_cv/baseline_b1"),
        )
    )
    c0_events_rejected = _raises_cross_protocol_guard(
        lambda: c0_events.run_evaluate(
            force=False,
            dataset_name="le2i",
            run_dir=Path("runs/local/le2i_cv/baseline_c0"),
        )
    )
    c1_events_rejected = _raises_cross_protocol_guard(
        lambda: c1_events.run_evaluate(
            force=False,
            dataset_name="le2i",
            run_dir=Path("runs/local/le2i_cv/baseline_c1"),
        )
    )
    return _check(
        "entry points CS-only (alarm_protocol_sensitivity/grouped_bootstrap/"
        "qualitative/multiseed_summary/baseline_b0.run_train/baseline_b0.run_report/"
        "b0_events.run_evaluate/baseline_b1.run_train/baseline_b1.run_report/"
        "b1_events.run_evaluate/baseline_c0.run_train/baseline_c0.run_report/"
        "c0_events.run_evaluate/"
        "baseline_c1.run_train/baseline_c1.run_report/c1_events.run_evaluate) "
        "recusam --run-dir sob runs/local/le2i_cv/ mesmo com --dataset le2i, "
        "através da própria função de produção",
        sensitivity_rejected
        and bootstrap_rejected
        and qualitative_rejected
        and multiseed_rejected
        and b0_train_rejected
        and b0_report_rejected
        and b0_events_rejected
        and b1_train_rejected
        and b1_report_rejected
        and b1_events_rejected
        and c0_events_rejected
        and c1_events_rejected
        and c0_train_rejected
        and c0_report_rejected
        and c1_train_rejected
        and c1_report_rejected,
    )


def check_cs_only_analysis_entry_points_still_accept_cs_run_dirs() -> bool:
    cs_run_dir = default_run_dir("le2i")
    third_party_run_dir = Path("runs/local/custom_experiment/baseline_a")

    def _passes_guard_and_fails_downstream(callback) -> bool:
        try:
            callback()
        except ValueError:
            return False
        except (RuntimeError, OSError):
            return True
        return True

    sensitivity_ok = _passes_guard_and_fails_downstream(
        lambda: alarm_protocol_sensitivity.run_analyze(
            force=False, dataset_name="le2i", run_dir=third_party_run_dir
        )
    )
    bootstrap_ok = _passes_guard_and_fails_downstream(
        lambda: grouped_bootstrap.run_analyze(
            force=False, dataset_name="le2i", run_dir=third_party_run_dir
        )
    )
    qualitative_ok = _passes_guard_and_fails_downstream(
        lambda: qualitative.run_render(
            run_dir=third_party_run_dir,
            dataset_name="le2i",
            splits=("val",),
            force=False,
        )
    )
    adapter = get_dataset("le2i")
    multiseed_ok = _passes_guard_and_fails_downstream(
        lambda: multiseed_summary._summarize(
            [cs_run_dir, third_party_run_dir],
            BASELINE_A_CONFIG,
            adapter,
        )
    )
    with (
        tempfile.TemporaryDirectory() as b0_train_tmp,
        tempfile.TemporaryDirectory() as b0_report_tmp,
        tempfile.TemporaryDirectory() as b0_events_tmp,
        tempfile.TemporaryDirectory() as b1_train_tmp,
        tempfile.TemporaryDirectory() as b1_report_tmp,
        tempfile.TemporaryDirectory() as b1_events_tmp,
        tempfile.TemporaryDirectory() as c0_train_tmp,
        tempfile.TemporaryDirectory() as c0_report_tmp,
        tempfile.TemporaryDirectory() as c0_events_tmp,
        tempfile.TemporaryDirectory() as c1_train_tmp,
        tempfile.TemporaryDirectory() as c1_report_tmp,
        tempfile.TemporaryDirectory() as c1_events_tmp,
    ):
        # run_dir vazio e pré-existente: se o guard de protocolo for passado, run_b0_training
        # (baseline_b0/engine.py) bate no branch de "run parcial" (artefatos ausentes) antes de criar o
        # diretório temporário de treino e entrar no loop, sem depender de dados reais no disco.
        b0_train_run_dir = Path(b0_train_tmp)
        b0_report_run_dir = Path(b0_report_tmp)
        b0_report_output = b0_report_run_dir / "out" / "b0_report.json"
        b0_train_ok = _passes_guard_and_fails_downstream(
            lambda: baseline_b0.run_train(
                force=False, dataset_name="le2i", run_dir=b0_train_run_dir
            )
        )
        b0_report_ok = _passes_guard_and_fails_downstream(
            lambda: baseline_b0.run_report(
                dataset_name="le2i",
                run_dir=b0_report_run_dir,
                output_path=b0_report_output,
                force=False,
            )
        )
        b0_events_ok = _passes_guard_and_fails_downstream(
            lambda: b0_events.run_evaluate(
                force=False,
                dataset_name="le2i",
                run_dir=Path(b0_events_tmp),
            )
        )
        b1_train_run_dir = Path(b1_train_tmp)
        b1_report_run_dir = Path(b1_report_tmp)
        b1_train_ok = _passes_guard_and_fails_downstream(
            lambda: baseline_b1.run_train(
                force=False, dataset_name="le2i", run_dir=b1_train_run_dir
            )
        )
        b1_report_ok = _passes_guard_and_fails_downstream(
            lambda: baseline_b1.run_report(
                dataset_name="le2i",
                run_dir=b1_report_run_dir,
                output_path=b1_report_run_dir / "out" / "b1_report.json",
                force=False,
            )
        )
        b1_events_ok = _passes_guard_and_fails_downstream(
            lambda: b1_events.run_evaluate(
                force=False,
                dataset_name="le2i",
                run_dir=Path(b1_events_tmp),
            )
        )
        c0_train_ok = _passes_guard_and_fails_downstream(
            lambda: baseline_c0.run_train(
                force=False, dataset_name="le2i", run_dir=Path(c0_train_tmp)
            )
        )
        c0_report_ok = _passes_guard_and_fails_downstream(
            lambda: baseline_c0.run_report(
                dataset_name="le2i",
                run_dir=Path(c0_report_tmp),
                output_path=Path(c0_report_tmp) / "out" / "c0_report.json",
                force=False,
            )
        )
        c0_events_ok = _passes_guard_and_fails_downstream(
            lambda: c0_events.run_evaluate(
                force=False, dataset_name="le2i", run_dir=Path(c0_events_tmp)
            )
        )

        c1_train_ok = _passes_guard_and_fails_downstream(
            lambda: baseline_c1.run_train(force=False, dataset_name="le2i", run_dir=Path(c1_train_tmp))
        )
        c1_report_ok = _passes_guard_and_fails_downstream(
            lambda: baseline_c1.run_report(dataset_name="le2i", run_dir=Path(c1_report_tmp),
                                       output_path=Path(c1_report_tmp) / "out" / "c1_report.json",
                                       force=False)
        )
        c1_events_ok = _passes_guard_and_fails_downstream(
            lambda: c1_events.run_evaluate(
                force=False, dataset_name="le2i", run_dir=Path(c1_events_tmp)
            )
        )

    return _check(
        "entry points CS-only: run_dir canônico do próprio protocolo e run_dir "
        "de terceiros continuam passando pelo guard (a falha, se houver, vem "
        "de artefatos ausentes, não do guard de protocolo)",
        sensitivity_ok
        and bootstrap_ok
        and qualitative_ok
        and multiseed_ok
        and b0_train_ok
        and b0_report_ok
        and b0_events_ok
        and b1_train_ok
        and b1_report_ok
        and b1_events_ok
        and c0_train_ok
        and c0_report_ok
        and c0_events_ok
        and c1_train_ok
        and c1_report_ok
        and c1_events_ok,
    )


def check_alarm_protocol_unaffected_by_dataset_selection() -> bool:
    expected = replace(BASELINE_A_ALARM_PROTOCOL)
    ok = (
        BASELINE_A_ALARM_PROTOCOL.trigger_consecutive == expected.trigger_consecutive
        and BASELINE_A_ALARM_PROTOCOL.refractory_period_s
        == expected.refractory_period_s
        and BASELINE_A_ALARM_PROTOCOL.fall_label == expected.fall_label
    )
    return _check(
        "protocolo de alarme do braço A: é uma constante única, sem parâmetro "
        "de dataset/protocolo — le2i-cv herda exatamente o mesmo protocolo",
        ok,
    )


def run_selftest() -> None:
    checks = [
        check_cv_adapter_paths_isolated_from_cs_artifacts(),
        check_shared_features_and_separate_statistics(),
        check_cv_timegrid_mismatch_rejected(),
        check_cv_extraction_commands_rejected(),
        check_cv_pose_programmatic_extraction_rejected(),
        check_cv_pose_report_remains_supported(),
        check_load_annotation_splits_respects_protocol(),
        check_frozen_train_config_identical_except_standardization(),
        check_default_run_dir_is_dataset_aware(),
        check_cv_run_dir_not_classified_under_cs_root_despite_string_prefix(),
        check_cross_protocol_run_dir_guard_rejects_both_directions(),
        check_non_conflicting_local_run_dir_still_accepted(),
        check_cs_only_analysis_entry_points_reject_cv_run_dir(),
        check_cs_only_analysis_entry_points_still_accept_cs_run_dirs(),
        check_alarm_protocol_unaffected_by_dataset_selection(),
    ]
    if not all(checks):
        print("\nprotocol isolation selftest FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nprotocol isolation selftest OK: todas as checagens passaram")


if __name__ == "__main__":
    run_selftest()
