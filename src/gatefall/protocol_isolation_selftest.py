"""Selftest de isolamento entre protocolos Le2i-CS e Le2i-CV.

Cobre: paths do adapter CV nunca caem sob artefatos CS; `load_annotation_splits`
respeita o protocolo pedido; e a receita de treino/protocolo de alarme
congelados do braço A não têm nenhum ramo condicional a protocolo além do
path/sha256 de padronização.
"""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

import gatefall.data.le2i.annotations as annotations_module
from gatefall.data.le2i.annotations import load_annotation_splits
from gatefall.datasets import get_dataset
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.eval import alarm_protocol_sensitivity, grouped_bootstrap, multiseed_summary, qualitative
from gatefall.eval.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.runs import default_run_dir, validate_local_run_dir
from gatefall.train import b0_fusion
from gatefall.train.baseline_a import _resolve_config
from gatefall.train.config import BASELINE_A_CONFIG


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
            lambda: b0_fusion.run_train(force=False, dataset_name="le2i", run_dir=cv_run_dir)
        )
        b0_report_rejected = _raises_cross_protocol_guard(
            lambda: b0_fusion.run_report(
                dataset_name="le2i",
                run_dir=cv_run_dir,
                output_path=b0_report_output,
                force=False,
            )
        )

    return _check(
        "entry points CS-only (alarm_protocol_sensitivity/grouped_bootstrap/"
        "qualitative/multiseed_summary/b0_fusion.run_train/b0_fusion.run_report) "
        "recusam --run-dir sob runs/local/le2i_cv/ mesmo com --dataset le2i, "
        "através da própria função de produção",
        sensitivity_rejected
        and bootstrap_rejected
        and qualitative_rejected
        and multiseed_rejected
        and b0_train_rejected
        and b0_report_rejected,
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
    ):
        # run_dir vazio e pré-existente: se o guard de protocolo for passado, run_b0_training
        # (b0_engine.py) bate no branch de "run parcial" (artefatos ausentes) antes de criar o
        # diretório temporário de treino e entrar no loop, sem depender de dados reais no disco.
        b0_train_run_dir = Path(b0_train_tmp)
        b0_report_run_dir = Path(b0_report_tmp)
        b0_report_output = b0_report_run_dir / "out" / "b0_report.json"
        b0_train_ok = _passes_guard_and_fails_downstream(
            lambda: b0_fusion.run_train(
                force=False, dataset_name="le2i", run_dir=b0_train_run_dir
            )
        )
        b0_report_ok = _passes_guard_and_fails_downstream(
            lambda: b0_fusion.run_report(
                dataset_name="le2i",
                run_dir=b0_report_run_dir,
                output_path=b0_report_output,
                force=False,
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
        and b0_report_ok,
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
