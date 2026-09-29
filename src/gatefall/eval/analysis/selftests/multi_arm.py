import json
import tempfile
from pathlib import Path

from gatefall.datasets import get_dataset
from gatefall.eval.analysis.multiseed_summary import ARMS, EVENT_ARMS, _summarize
from gatefall.eval.analysis.selftests.fixtures import _check, _rewrite_synthetic_config, _synthetic_pair


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
            else:
                if (run_dirs[1] / "event_metrics.json").exists():
                    return _check("C0 dispensa artefatos de evento", False)
                (run_dirs[1] / "event_metrics.json").write_text("{}", encoding="utf-8")
                c0_report, c0_rows = _summarize(run_dirs, expected, adapter)
                if "events" in c0_report["aggregate"] or any(
                    row["metric_group"] == "events" for row in c0_rows
                ):
                    return _check("C0 omite evento mesmo se houver arquivo extra", False)
        return _check(
            "A/B0/B1/C0/C1: classificação validada; evento obrigatório exceto C0",
            True,
        )
