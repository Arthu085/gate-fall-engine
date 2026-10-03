import json
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from gatefall.config import EVAL_STRIDE, IGNORE_LABEL
from gatefall.data.windowing import build_window_index
from gatefall.eval.analysis import grouped_bootstrap
from gatefall.eval.analysis.grouped_bootstrap import (
    CSV_COLUMNS,
    GROUPED_BOOTSTRAP_CSV_FILE,
    GROUPED_BOOTSTRAP_JSON_FILE,
    METHOD_NOTE,
    PAIRED_CSV_COLUMNS,
    RunPredictions,
    SplitPredictions,
    _draw_subjects,
    _percentile_ci,
    _validate_paired_support,
    collect_run_predictions,
    paired_output_paths,
    run_analyze,
    run_compare,
    run_grouped_bootstrap_for_split,
    run_paired_bootstrap_for_split,
)
from gatefall.eval.analysis.multiseed_summary import ARMS, RunConfig
from gatefall.eval.analysis.selftests.fixtures import (
    _build_event_split,
    _check,
    _synthetic_classification_arrays,
    _write_synthetic_seed_run,
)
from gatefall.eval.baseline_a import cli as baseline_a_events
from gatefall.eval.baseline_b0 import cli as baseline_b0_events
from gatefall.eval.baseline_b1 import cli as baseline_b1_events
from gatefall.eval.baseline_c0 import cli as baseline_c0_events
from gatefall.eval.baseline_c1 import cli as baseline_c1_events
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, save_alarm_protocol
from gatefall.eval.shared.orchestration import EventEvaluation, Predictions
from gatefall.hashing import sha256_file
from gatefall.runs import REFERENCE_RUN_ROOT, REPOSITORY_ROOT

ARM_EVENT_MODULES = {
    "A": baseline_a_events,
    "B0": baseline_b0_events,
    "B1": baseline_b1_events,
    "C0": baseline_c0_events,
    "C1": baseline_c1_events,
}
N_REPLICATES = 40

# Rótulos por quadro: fall em k=[2,3], fallen em k=[4..6]; o resto é ADL.
FALL_VIDEO_LABELS = [0, 0, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
VIDEO_SPEC: tuple[tuple[str, str, int], ...] = (
    ("v1a", "val", 1),
    ("v1b", "val", 1),
    ("v2", "val", 2),
    ("v3", "test", 3),
    ("v4a", "test", 4),
    ("v4b", "test", 4),
)


class _LoaderCalled(Exception):
    pass


def _frames(
    spec: tuple[tuple[str, str, int], ...] = VIDEO_SPEC,
    labels: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for video_id, split, subject in spec:
        video_labels = (labels or {}).get(video_id, FALL_VIDEO_LABELS)
        for frame_index, label in enumerate(video_labels):
            rows.append(
                {
                    "video_id": video_id,
                    "split": split,
                    "env": "synthetic",
                    "subject": subject,
                    "frame_index": frame_index,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def _evaluation(
    config: RunConfig,
    frames: pd.DataFrame,
    predictions: dict[tuple[str, int], int] | None = None,
    inference_calls: list[str] | None = None,
) -> EventEvaluation:
    def prepare() -> tuple[pd.DataFrame, Callable[[str], tuple[int, Predictions]]]:
        def evaluate_split(split: str) -> tuple[int, Predictions]:
            if inference_calls is not None:
                inference_calls.append(split)
            windows = build_window_index(
                cast(pd.DataFrame, frames[frames["split"] == split]),
                stride=EVAL_STRIDE,
                drop_ignored=False,
            )
            video_ids = [str(video_id) for video_id in windows["video_id"]]
            k_ends = [int(k_end) for k_end in windows["k_end"]]
            true_labels = [int(label) for label in windows["label"]]
            pred_labels = [
                (predictions or {}).get((video_id, k_end), max(label, 0))
                for video_id, k_end, label in zip(video_ids, k_ends, true_labels)
            ]
            return len(windows), (video_ids, k_ends, true_labels, pred_labels)

        return frames, evaluate_split

    return EventEvaluation(config, prepare)


@contextmanager
def _patched_loaders(
    evaluations: dict[str, EventEvaluation], calls: list[tuple]
) -> Iterator[None]:
    def make_loader(arm: str) -> Callable[..., EventEvaluation]:
        def loader(dataset_name: str, run_dir: Path, **kwargs: object) -> EventEvaluation:
            calls.append((arm, dataset_name, Path(run_dir), kwargs))
            if arm not in evaluations:
                raise _LoaderCalled(arm)
            return evaluations[arm]

        return loader

    with ExitStack() as stack:
        for arm, module in ARM_EVENT_MODULES.items():
            stack.enter_context(patch.object(module, "load_event_evaluation", make_loader(arm)))
        yield


def _synthetic_run(run_dir: Path, arm: str, seed: int = 1) -> RunConfig:
    y_true, y_pred = _synthetic_classification_arrays(offset=0)
    event = _build_event_split(1, 1, 1.0, 1.0, 1.0, 0.0, 0, 1.0, 1.0)
    return _write_synthetic_seed_run(run_dir, seed, y_true, y_pred, event, event, arm=arm)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _raises(callback: Callable[[], object], exception: type[BaseException], text: str) -> bool:
    try:
        callback()
    except exception as exc:
        return text in str(exc)
    return False


def _selftest_arm_dispatch() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm in ARMS:
            run_dir = root / arm
            config = _synthetic_run(run_dir, arm)
            calls: list[tuple] = []
            inference_calls: list[str] = []
            evaluation = _evaluation(config, _frames(), inference_calls=inference_calls)
            with _patched_loaders({arm: evaluation}, calls):
                written = run_analyze(
                    force=False, run_dir=run_dir, n_replicates=N_REPLICATES, arm=arm
                )
            report = json.loads((run_dir / GROUPED_BOOTSTRAP_JSON_FILE).read_text("utf-8"))
            csv = pd.read_csv(run_dir / GROUPED_BOOTSTRAP_CSV_FILE)
            expected_kwargs = {"fields_allowed_to_differ": frozenset()} if arm == "A" else {}
            ok = ok and (
                written
                and calls == [(arm, "le2i", run_dir, expected_kwargs)]
                and inference_calls == ["val", "test"]
                and report["arm"] == arm
                and report["dataset"] == "le2i"
                and report["run_name"] == config.run_name
                and report["training_seed"] == config.seed
                and report["checkpoint_sha256"] == sha256_file(run_dir / "checkpoint.pt")
                and list(csv.columns) == CSV_COLUMNS
                and len(csv) == 2 * 16
            )

            default_calls: list[tuple] = []
            with _patched_loaders({}, default_calls):
                try:
                    run_analyze(force=False, dataset_name="le2i-cv", arm=arm)
                except _LoaderCalled:
                    pass
            expected_default = Path("runs/local/le2i_cv") / f"baseline_{arm.lower()}"
            ok = ok and default_calls[0][:3] == (arm, "le2i-cv", expected_default)

        foreign_dir = root / "foreign"
        b0_config = _synthetic_run(foreign_dir, "B0")
        with _patched_loaders({"B1": _evaluation(b0_config, _frames())}, []):
            foreign_rejected = _raises(
                lambda: run_analyze(force=False, run_dir=foreign_dir, arm="B1"),
                ValueError,
                "pertence à arma 'B0'",
            )
        unknown_rejected = _raises(
            lambda: run_analyze(force=False, run_dir=foreign_dir, arm="D"),
            ValueError,
            "arma não suportada",
        )
        ok = (
            ok
            and foreign_rejected
            and unknown_rejected
            and not (foreign_dir / GROUPED_BOOTSTRAP_JSON_FILE).exists()
        )
    return _check(
        "analyze despacha A/B0/B1/C0/C1 para o load_event_evaluation da própria "
        "arma (A com seed congelada), roda inferência uma vez por split, resolve "
        "o run_dir padrão por arma/protocolo e recusa arma estrangeira ou "
        "desconhecida sem gravar saída",
        ok,
    )


def _selftest_arm_a_output_preserved() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "A"
        config = _synthetic_run(run_dir, "A")
        evaluation = _evaluation(config, _frames(), predictions={("v2", 9): 1, ("v4a", 12): 3})
        with _patched_loaders({"A": evaluation}, []):
            run_analyze(force=False, run_dir=run_dir, n_replicates=N_REPLICATES, seed=7)
            report = json.loads((run_dir / GROUPED_BOOTSTRAP_JSON_FILE).read_text("utf-8"))
            run = collect_run_predictions("A", run_dir, evaluation)
            seed_sequences = np.random.SeedSequence(7).spawn(2)
            expected_splits = {
                split: run_grouped_bootstrap_for_split(
                    run.splits[split],
                    run.subject_maps[split],
                    BASELINE_A_ALARM_PROTOCOL,
                    config.num_classes,
                    N_REPLICATES,
                    0.95,
                    np.random.default_rng(seed_sequence),
                )
                for split, seed_sequence in zip(("val", "test"), seed_sequences)
            }
            json_hash = sha256_file(run_dir / GROUPED_BOOTSTRAP_JSON_FILE)
            skipped = not run_analyze(force=False, run_dir=run_dir, seed=7)
            unchanged = sha256_file(run_dir / GROUPED_BOOTSTRAP_JSON_FILE) == json_hash
            forced = run_analyze(force=True, run_dir=run_dir, n_replicates=N_REPLICATES, seed=8)
            rewritten = json.loads((run_dir / GROUPED_BOOTSTRAP_JSON_FILE).read_text("utf-8"))
        legacy_keys = {
            "run_name",
            "checkpoint_path",
            "checkpoint_sha256",
            "training_metrics_path",
            "training_metrics_sha256",
            "alarm_protocol",
            "method",
            "splits",
        }
        ok = (
            legacy_keys <= report.keys()
            and report["splits"] == json.loads(json.dumps(expected_splits))
            and report["method"]
            == {
                "cluster_unit": "subject",
                "n_replicates": N_REPLICATES,
                "confidence_level": 0.95,
                "seed": 7,
                "ci_method": "percentile",
                "note": METHOD_NOTE,
            }
            and report["alarm_protocol"] == BASELINE_A_ALARM_PROTOCOL.to_dict()
            and skipped
            and unchanged
            and forced
            and rewritten["method"]["seed"] == 8
            and not list(run_dir.glob(".grouped_bootstrap*"))
        )
    return _check(
        "arma A preserva o esquema legado, os fluxos SeedSequence(seed).spawn(2) "
        "(val, test), o skip sem --force e a reescrita com --force",
        ok,
    )


def _two_subject_predictions(
    subject_two_preds: list[int], n: int = 10
) -> tuple[SplitPredictions, dict[int, list[str]]]:
    return (
        SplitPredictions(
            video_ids=["s1"] * n + ["s2"] * n,
            k_ends=list(range(n)) * 2,
            true_labels=[0] * (2 * n),
            pred_labels=[0] * n + subject_two_preds,
            usable_windows=2 * n,
            total_windows=2 * n,
            labeled_windows=2 * n,
        ),
        {1: ["s1"], 2: ["s2"]},
    )


def _selftest_paired_draws_are_shared() -> bool:
    n = 13
    labels = [0] * n
    labels[2:4] = [1, 1]
    labels[5:8] = [2, 2, 2]
    video_ids = ["a1"] * n + ["a2"] * n + ["b"] * n + ["c"] * n
    baseline = SplitPredictions(
        video_ids=video_ids,
        k_ends=list(range(n)) * 4,
        true_labels=labels * 4,
        pred_labels=[0] * (4 * n),
        usable_windows=4 * n,
        total_windows=4 * n,
        labeled_windows=4 * n,
    )
    adaptive = replace(baseline, pred_labels=[max(label, 0) for label in labels] * 4)
    subject_to_videos = {1: ["a1", "a2"], 2: ["b"], 3: ["c"]}

    recorded: list[tuple[list[int], list[str], list[int]]] = []
    original = grouped_bootstrap._build_replicate_arrays

    def recording(drawn, *args):
        result = original(drawn, *args)
        recorded.append((drawn.tolist(), result[0].tolist(), result[1].tolist()))
        return result

    with patch.object(grouped_bootstrap, "_build_replicate_arrays", recording):
        run_paired_bootstrap_for_split(
            baseline,
            adaptive,
            subject_to_videos,
            BASELINE_A_ALARM_PROTOCOL,
            10,
            N_REPLICATES,
            0.95,
            np.random.default_rng(np.random.SeedSequence(5)),
        )

    reference_rng = np.random.default_rng(np.random.SeedSequence(5))
    subjects = np.array([1, 2, 3], dtype=np.int64)
    expected_draws = [_draw_subjects(reference_rng, subjects).tolist() for _ in range(N_REPLICATES)]
    pairs = [(recorded[i], recorded[i + 1]) for i in range(0, len(recorded), 2)]
    repeated = [draw for draw, _, _ in recorded if len(set(draw)) < len(draw)]
    ok = (
        len(recorded) == 2 * N_REPLICATES
        and all(baseline_call == adaptive_call for baseline_call, adaptive_call in pairs)
        and [baseline_call[0] for baseline_call, _ in pairs] == expected_draws
        and bool(repeated)
        and all("__draw" in video_id for _, virtual_ids, _ in recorded for video_id in virtual_ids)
    )
    return _check(
        "bootstrap pareado aplica às duas armas o mesmo sorteio ordenado por "
        "réplica (mesmas multiplicidades, posições e IDs virtuais), consumindo "
        "um único sorteio por réplica, inclusive com sujeito multi-vídeo repetido",
        ok,
    )


def _selftest_paired_known_delta() -> bool:
    baseline, subject_to_videos = _two_subject_predictions([1] * 10)
    adaptive, _ = _two_subject_predictions([1] * 5 + [0] * 5)
    n_replicates = 400

    def bootstrap_seeded(function, *predictions) -> dict:
        return function(
            *predictions,
            subject_to_videos,
            BASELINE_A_ALARM_PROTOCOL,
            10,
            n_replicates,
            0.95,
            np.random.default_rng(np.random.SeedSequence(3)),
        )

    paired = bootstrap_seeded(run_paired_bootstrap_for_split, baseline, adaptive)
    repeated = bootstrap_seeded(run_paired_bootstrap_for_split, baseline, adaptive)
    accuracy = paired["classification"]["accuracy"]

    reference_rng = np.random.default_rng(np.random.SeedSequence(3))
    subjects = np.array([1, 2], dtype=np.int64)
    delta_by_subject_one_draws = {0: 0.5, 1: 0.25, 2: 0.0}
    expected_deltas = [
        delta_by_subject_one_draws[int(np.sum(_draw_subjects(reference_rng, subjects) == 1))]
        for _ in range(n_replicates)
    ]
    expected_lower, expected_upper = _percentile_ci(expected_deltas, 0.95)

    baseline_single = bootstrap_seeded(run_grouped_bootstrap_for_split, baseline)
    adaptive_single = bootstrap_seeded(run_grouped_bootstrap_for_split, adaptive)
    naive_lower = (
        adaptive_single["classification"]["accuracy"]["ci_lower"]
        - baseline_single["classification"]["accuracy"]["ci_upper"]
    )
    identical = bootstrap_seeded(run_paired_bootstrap_for_split, baseline, baseline)
    ok = (
        paired == repeated
        and accuracy["baseline_point_estimate"] == 0.5
        and accuracy["adaptive_point_estimate"] == 0.75
        and accuracy["delta_point_estimate"] == 0.25
        and accuracy["delta_ci_lower"] == expected_lower == 0.0
        and accuracy["delta_ci_upper"] == expected_upper == 0.5
        and accuracy["valid_replicates"] == n_replicates
        and naive_lower == -0.5
        and naive_lower != accuracy["delta_ci_lower"]
        and all(
            report["delta_point_estimate"] in (0.0, None)
            and report["delta_ci_lower"] in (0.0, None)
            and report["delta_ci_upper"] in (0.0, None)
            and report["valid_replicates"]
            == baseline_single[group][name]["valid_replicates"]
            for group in ("classification", "events")
            for name, report in identical[group].items()
        )
    )
    return _check(
        "delta pareado conhecido: ponto = adaptativa - baseline (0.75 - 0.5), "
        "IC percentil dos deltas por réplica ([0.0, 0.5]) difere da subtração "
        "de limites independentes (-0.5); armas idênticas dão delta 0; mesma "
        "seed reproduz o relatório",
        ok,
    )


def _selftest_paired_undefined_propagation() -> bool:
    n = 13
    fall = [0] * n
    fall[2:4] = [1, 1]
    fall[5:8] = [2, 2, 2]
    alarm = [0] * n
    alarm[10:13] = [1, 1, 1]
    baseline = SplitPredictions(
        video_ids=["fall"] * n + ["adl"] * n,
        k_ends=list(range(n)) * 2,
        true_labels=fall + [0] * n,
        pred_labels=[0] * (2 * n),
        usable_windows=2 * n,
        total_windows=2 * n,
        labeled_windows=2 * n,
    )
    adaptive = replace(baseline, pred_labels=alarm + [0] * n)
    subject_to_videos = {1: ["fall"], 2: ["adl"]}
    report = run_paired_bootstrap_for_split(
        baseline,
        adaptive,
        subject_to_videos,
        BASELINE_A_ALARM_PROTOCOL,
        10,
        N_REPLICATES,
        0.95,
        np.random.default_rng(np.random.SeedSequence(9)),
    )
    reference_rng = np.random.default_rng(np.random.SeedSequence(9))
    subjects = np.array([1, 2], dtype=np.int64)
    draws_with_fall = sum(
        1 in _draw_subjects(reference_rng, subjects).tolist() for _ in range(N_REPLICATES)
    )
    latency = report["events"]["latency_mean_s"]
    sensitivity = report["events"]["sensitivity"]
    ok = (
        latency["baseline_point_estimate"] is None
        and latency["adaptive_point_estimate"] is not None
        and latency["delta_point_estimate"] is None
        and latency["delta_ci_lower"] is None
        and latency["delta_ci_upper"] is None
        and latency["valid_replicates"] == 0
        and latency["undefined_replicates"] == N_REPLICATES
        and 0 < draws_with_fall < N_REPLICATES
        and sensitivity["valid_replicates"] == draws_with_fall
        and sensitivity["undefined_replicates"] == N_REPLICATES - draws_with_fall
        and sensitivity["delta_point_estimate"] == 1.0
        and sensitivity["delta_ci_lower"] == sensitivity["delta_ci_upper"] == 1.0
    )
    return _check(
        "métrica indefinida em qualquer arma torna a réplica pareada indefinida "
        "(contada, nunca zerada): latência só definida na adaptativa dá delta "
        "nulo com 0 réplicas válidas; sensibilidade só conta réplicas com evento",
        ok,
    )


def _paired_fixture(root: Path, adaptive_arm: str, baseline_arm: str, seeds=(1, 1)):
    baseline_dir = root / baseline_arm
    adaptive_dir = root / adaptive_arm
    baseline_config = _synthetic_run(baseline_dir, baseline_arm, seed=seeds[0])
    adaptive_config = _synthetic_run(adaptive_dir, adaptive_arm, seed=seeds[1])
    return baseline_dir, adaptive_dir, baseline_config, adaptive_config


def _selftest_compare_end_to_end() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for adaptive_arm, baseline_arm in (("B1", "B0"), ("C1", "C0")):
            pair_root = root / adaptive_arm
            baseline_dir, adaptive_dir, baseline_config, adaptive_config = _paired_fixture(
                pair_root, adaptive_arm, baseline_arm
            )
            output_dir = pair_root / "comparison"
            before = {
                path: _tree_hashes(path) for path in (baseline_dir, adaptive_dir)
            }
            inference_calls: list[str] = []
            evaluations = {
                baseline_arm: _evaluation(
                    baseline_config, _frames(), {("v2", 3): 0}, inference_calls
                ),
                adaptive_arm: _evaluation(adaptive_config, _frames(), None, inference_calls),
            }
            with _patched_loaders(evaluations, []):
                written = run_compare(
                    adaptive_arm,
                    baseline_arm,
                    "le2i",
                    adaptive_dir,
                    baseline_dir,
                    output_dir,
                    force=False,
                    n_replicates=N_REPLICATES,
                )
                skipped = not run_compare(
                    adaptive_arm,
                    baseline_arm,
                    "le2i",
                    adaptive_dir,
                    baseline_dir,
                    output_dir,
                    force=False,
                    n_replicates=N_REPLICATES,
                )
            json_path, csv_path = paired_output_paths(output_dir, adaptive_arm, baseline_arm)
            report = json.loads(json_path.read_text("utf-8"))
            csv = pd.read_csv(csv_path)
            shared = report["compatibility"]["shared_config"]
            ok = ok and (
                written
                and skipped
                and inference_calls == ["val", "test", "val", "test"]
                and json_path.name
                == f"paired_bootstrap_{adaptive_arm.lower()}_minus_{baseline_arm.lower()}.json"
                and report["comparison"] == f"{adaptive_arm} - {baseline_arm}"
                and report["training_seed"] == 1
                and report["adaptive"]["arm"] == adaptive_arm
                and report["baseline"]["arm"] == baseline_arm
                and report["adaptive"]["checkpoint_sha256"]
                == sha256_file(adaptive_dir / "checkpoint.pt")
                and report["baseline"]["training_metrics_sha256"]
                == sha256_file(baseline_dir / "metrics.json")
                and report["baseline"]["event_metrics_sha256"]
                == sha256_file(baseline_dir / "event_metrics.json")
                and report["method"]["paired"] is True
                and report["method"]["delta"] == "adaptive - baseline"
                and report["alarm_protocol"] == BASELINE_A_ALARM_PROTOCOL.to_dict()
                and shared["seed"] == 1
                and shared["visual_standardization_stats_sha256"] == "synthetic"
                and "gate_activation" not in shared
                and report["compatibility"]["splits"]["val"]["n_subjects"] == 2
                and report["compatibility"]["splits"]["test"]["subject_to_videos"]
                == {"3": ["v3"], "4": ["v4a", "v4b"]}
                and report["splits"]["val"]["classification"]["accuracy"]["delta_point_estimate"]
                is not None
                and list(csv.columns) == PAIRED_CSV_COLUMNS
                and len(csv) == 2 * 16
                and before == {path: _tree_hashes(path) for path in before}
                and not list(output_dir.glob(".paired_bootstrap*"))
            )
    return _check(
        "compare B1 - B0 e C1 - C0 grava JSON/CSV dedicados com proveniência, "
        "contrato compartilhado e suporte auditáveis, roda inferência uma vez "
        "por split por run, não muta nenhum run de entrada e pula sem --force",
        ok,
    )


def _selftest_compare_rejections() -> bool:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline_dir, adaptive_dir, baseline_config, adaptive_config = _paired_fixture(
            root / "ok", "B1", "B0"
        )
        output_dir = root / "out"

        def attempt(
            exception: type[BaseException],
            text: str,
            adaptive_arm: str = "B1",
            baseline_arm: str = "B0",
            adaptive: RunConfig = adaptive_config,
            baseline: RunConfig = baseline_config,
            adaptive_frames: pd.DataFrame | None = None,
            adaptive_run_dir: Path = adaptive_dir,
            baseline_run_dir: Path = baseline_dir,
            output: Path = output_dir,
        ) -> bool:
            calls: list[tuple] = []
            evaluations = {
                "B0": _evaluation(baseline, _frames()),
                "B1": _evaluation(
                    adaptive, _frames() if adaptive_frames is None else adaptive_frames
                ),
            }
            with _patched_loaders(evaluations, calls):
                rejected = _raises(
                    lambda: run_compare(
                        adaptive_arm,
                        baseline_arm,
                        "le2i",
                        adaptive_run_dir,
                        baseline_run_dir,
                        output,
                        force=True,
                        n_replicates=N_REPLICATES,
                    ),
                    exception,
                    text,
                )
            return rejected and not output_dir.exists()

        for adaptive_arm, baseline_arm in (("B0", "B1"), ("B1", "C0"), ("C1", "B0"), ("A", "B0"), ("B1", "A")):
            checks[f"par {adaptive_arm} - {baseline_arm}"] = attempt(
                ValueError, "não suportada", adaptive_arm=adaptive_arm, baseline_arm=baseline_arm
            )
        seed_baseline_dir, seed_adaptive_dir, seed_baseline, seed_adaptive = _paired_fixture(
            root / "seed", "B1", "B0", seeds=(1, 2)
        )
        checks["seed divergente"] = attempt(
            ValueError,
            "seeds de treino divergentes",
            adaptive=seed_adaptive,
            baseline=seed_baseline,
            adaptive_run_dir=seed_adaptive_dir,
            baseline_run_dir=seed_baseline_dir,
        )
        for field, value in (("lr", 0.5), ("visual_standardization_stats_sha256", "other")):
            checks[f"contrato {field}"] = attempt(
                ValueError, field, adaptive=replace(adaptive_config, **{field: value})
            )
        checks["arma estrangeira"] = attempt(
            ValueError, "pertence à arma 'B0'", adaptive=baseline_config
        )
        checks["protocolo cruzado"] = attempt(
            ValueError, "protocolo", baseline_run_dir=REPOSITORY_ROOT / "runs/local/le2i_cv/baseline_b0"
        )
        checks["referência"] = attempt(
            ValueError, "referência", baseline_run_dir=REFERENCE_RUN_ROOT / "le2i/baseline_b0"
        )
        checks["mesmo run_dir"] = attempt(ValueError, "coincidem", adaptive_run_dir=baseline_dir)
        checks["output dentro do run"] = attempt(
            ValueError, "--output-dir", output=adaptive_dir / "comparison"
        )
        checks["output em referência"] = attempt(
            ValueError, "--output-dir", output=REFERENCE_RUN_ROOT / "le2i/comparison"
        )
        checks["n_replicates"] = _raises(
            lambda: run_compare("B1", "B0", "le2i", adaptive_dir, baseline_dir, output_dir, False, 0),
            ValueError,
            "--n-replicates",
        )

        cases: list[tuple[str, type[BaseException], str, pd.DataFrame]] = [
            (
                "subject divergente",
                ValueError,
                "sujeitos",
                _frames(tuple((v, s, 9 if v == "v2" else subj) for v, s, subj in VIDEO_SPEC)),
            ),
            (
                "vídeo extra",
                ValueError,
                "sujeitos",
                _frames(VIDEO_SPEC + (("v5", "test", 4),)),
            ),
            (
                "rótulo divergente",
                ValueError,
                "true_labels",
                _frames(labels={"v3": FALL_VIDEO_LABELS[:-1] + [IGNORE_LABEL]}),
            ),
            (
                "suporte divergente",
                ValueError,
                "video_ids",
                _frames(labels={"v1b": FALL_VIDEO_LABELS[:-1]}),
            ),
        ]
        for name, exception, text, frames in cases:
            checks[name] = attempt(exception, text, adaptive_frames=frames)

        alarm_path = adaptive_dir / "alarm_protocol.yaml"
        original_alarm = alarm_path.read_bytes()
        save_alarm_protocol(
            replace(BASELINE_A_ALARM_PROTOCOL, trigger_consecutive=4), alarm_path, force=True
        )
        checks["protocolo de alarme"] = attempt(RuntimeError, "incompatível")
        alarm_path.write_bytes(original_alarm)
        event_path = adaptive_dir / "event_metrics.json"
        event_path.unlink()
        checks["evento ausente"] = attempt(RuntimeError, "event_metrics.json ausente")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"  rejeições que falharam: {', '.join(failed)}")
    return _check(
        "compare recusa, antes de gravar qualquer saída, pares fora de B1 - B0/"
        "C1 - C0 ou invertidos, seeds divergentes, contrato compartilhado "
        "divergente, arma estrangeira, protocolo cruzado, referência, run_dir "
        "repetido, output-dir dentro de run, protocolo de alarme divergente, "
        "evento ausente e sujeitos/vídeos/suporte/rótulos divergentes",
        not failed,
    )


def _selftest_paired_support_identity() -> bool:
    def run(video_ids: list[str], k_ends: list[int]) -> RunPredictions:
        split = SplitPredictions(
            video_ids=video_ids,
            k_ends=k_ends,
            true_labels=[0] * len(video_ids),
            pred_labels=[0] * len(video_ids),
            usable_windows=len(video_ids),
            total_windows=len(video_ids),
            labeled_windows=len(video_ids),
        )
        config = cast(RunConfig, None)
        return RunPredictions(
            arm="X",
            run_dir=Path("."),
            config=config,
            splits={"val": split, "test": split},
            subject_maps={"val": {1: ["a", "b"]}, "test": {1: ["a", "b"]}},
        )

    reference = run(["a", "a", "b"], [0, 1, 0])
    reordered = run(["b", "a", "a"], [0, 0, 1])
    shifted = run(["a", "a", "b"], [0, 2, 0])
    return _check(
        "suporte pareado exige a mesma sequência ordenada de (video_id, k_end): "
        "ordem de vídeos ou k_end divergentes são recusados",
        _raises(lambda: _validate_paired_support(reference, reordered), ValueError, "video_ids")
        and _raises(lambda: _validate_paired_support(reference, shifted), ValueError, "k_ends")
        and _validate_paired_support(reference, reference)["val"]["n_subjects"] == 1,
    )


def run_grouped_bootstrap_arms_selftest() -> list[bool]:
    return [
        _selftest_arm_dispatch(),
        _selftest_arm_a_output_preserved(),
        _selftest_paired_draws_are_shared(),
        _selftest_paired_known_delta(),
        _selftest_paired_undefined_propagation(),
        _selftest_compare_end_to_end(),
        _selftest_compare_rejections(),
        _selftest_paired_support_identity(),
    ]
