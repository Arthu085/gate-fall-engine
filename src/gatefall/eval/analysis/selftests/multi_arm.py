import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from gatefall.datasets import get_dataset
from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.eval.analysis.multiseed_summary import ARMS, EVENT_ARMS, _resolve_shared_expected, _summarize
from gatefall.eval.analysis.selftests.fixtures import _check, _rewrite_synthetic_config, _synthetic_pair
from gatefall.features.standardize_dinov3 import dinov3_stats_path
from gatefall.features.standardize_sam3 import sam3_stats_path
from gatefall.hashing import sha256_file
from gatefall.train.baseline_b0.config import B0TrainConfig
from gatefall.train.baseline_b1.config import B1TrainConfig
from gatefall.train.baseline_c0.config import C0TrainConfig
from gatefall.train.baseline_c1.config import C1TrainConfig


def _selftest_all_arms() -> bool:
    adapter = get_dataset("le2i")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm in ARMS:
            run_dirs, expected = _synthetic_pair(root / arm, arm)
            report, rows = _summarize(run_dirs, expected, adapter)
            has_events = arm in EVENT_ARMS
            if (
                report["arm"] != arm
                or report["n_seeds"] != 2
                or ("events" in report["aggregate"]) != has_events
                or any(("events" in seed) != has_events for seed in report["seeds"])
                or any(row["metric_group"] == "events" for row in rows) != has_events
            ):
                return _check("todas as armas preservam seu contrato de métricas", False)
            _rewrite_synthetic_config(run_dirs[1], {"lr": 0.5})
            try:
                _summarize(run_dirs, expected, adapter)
            except RuntimeError as exc:
                if "lr" not in str(exc):
                    return _check("configuração científica divergente é rejeitada", False)
            else:
                return _check("configuração científica divergente é rejeitada", False)
            _rewrite_synthetic_config(run_dirs[1], {"lr": expected.lr})
            if has_events:
                event_path = run_dirs[1] / "event_metrics.json"
                with event_path.open(encoding="utf-8") as stream:
                    event_report = json.load(stream)
                event_report["checkpoint_sha256"] = "incompatível"
                with event_path.open("w", encoding="utf-8") as stream:
                    json.dump(event_report, stream)
                try:
                    _summarize(run_dirs, expected, adapter)
                except RuntimeError as exc:
                    if "checkpoint_sha256" not in str(exc):
                        return _check("hash de evento inválido é rejeitado", False)
                else:
                    return _check("hash de evento inválido é rejeitado", False)
                (run_dirs[1] / "event_metrics.json").unlink()
                try:
                    _summarize(run_dirs, expected, adapter)
                except RuntimeError as exc:
                    if "event_metrics.json ausente" not in str(exc):
                        return _check("armas com evento exigem evidência completa", False)
                else:
                    return _check("armas com evento exigem evidência completa", False)
        return _check(
            "A/B0/B1/C0/C1: classificação e evento validados",
            True,
        )


def _selftest_cv_arms_and_protocol_isolation() -> bool:
    adapter = get_dataset("le2i-cv")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm in ("B0", "B1", "C0", "C1"):
            run_dirs, expected = _synthetic_pair(root / arm, arm)
            report, rows = _summarize(run_dirs, expected, adapter)
            if (
                report["arm"] != arm
                or report["n_seeds"] != 2
                or "events" not in report["aggregate"]
                or not any(row["metric_group"] == "events" for row in rows)
            ):
                return _check("braços B/C em le2i-cv agregam classificação e evento", False)
            for dataset_name, wrong_run_dir in (
                ("le2i-cv", Path("runs/local/le2i") / arm / "seed1"),
                ("le2i", Path("runs/local/le2i_cv") / arm / "seed1"),
            ):
                try:
                    _summarize(
                        [wrong_run_dir, run_dirs[1]],
                        expected,
                        get_dataset(dataset_name),
                    )
                except ValueError as exc:
                    if "protocolo" not in str(exc):
                        return _check("sumário rejeita run_dir do outro protocolo", False)
                else:
                    return _check("sumário rejeita run_dir do outro protocolo", False)
    return _check("braços B/C em le2i-cv e isolamento de run_dir", True)


def _selftest_cv_stats_resolution() -> bool:
    cv_dinov3_path = dinov3_stats_path("le2i-cv")
    cv_sam3_path = sam3_stats_path("le2i-cv")
    pose_stats_path = get_dataset("le2i-cv").pose_stats_path
    with (
        patch("gatefall.eval.analysis.multiseed_summary.validate_visual_stats_freshness"),
        patch("gatefall.eval.analysis.multiseed_summary.quality_set_sha256", return_value="synthetic"),
        patch.object(
            Le2iDatasetAdapter,
            "load_frames",
            return_value=pd.DataFrame({"video_id": ["synthetic"]}),
        ),
    ):
        dinov3_configs = [_resolve_shared_expected("le2i-cv", arm) for arm in ("B0", "B1")]
    with (
        patch(
            "gatefall.eval.analysis.multiseed_summary.validated_sam3_inputs",
            return_value=SimpleNamespace(
                frames=pd.DataFrame({"video_id": ["synthetic"]}),
                sam3_features_sha256="synthetic",
                sam3_provenance={},
            ),
        ),
        patch("gatefall.train.baseline_c1.run._source_hashes", return_value=("synthetic", "synthetic")),
    ):
        sam3_configs = [_resolve_shared_expected("le2i-cv", arm) for arm in ("C0", "C1")]
    return _check(
        "B/C em le2i-cv usam estatísticas específicas de CV, não de CS",
        cv_dinov3_path != dinov3_stats_path("le2i")
        and cv_sam3_path != sam3_stats_path("le2i")
        and all(
            isinstance(config, (B0TrainConfig, B1TrainConfig))
            and config.visual_standardization_stats_path == str(cv_dinov3_path)
            and config.visual_standardization_stats_sha256 == sha256_file(cv_dinov3_path)
            and config.pose_standardization_stats_path == str(pose_stats_path)
            for config in dinov3_configs
        )
        and all(
            isinstance(config, (C0TrainConfig, C1TrainConfig))
            and config.visual_standardization_stats_path == str(cv_sam3_path)
            and config.visual_standardization_stats_sha256 == sha256_file(cv_sam3_path)
            and config.pose_standardization_stats_path == str(pose_stats_path)
            for config in sam3_configs
        ),
    )
