"""Contratos sintéticos do estudo: plano, causalidade, métricas, estatística, registro e relatório."""

import json
import tempfile
import xml.etree.ElementTree as ElementTree
import zipfile
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from gatefall.study import integrity
from gatefall.study.methods import GATE_ABLATIONS, MOTION_METHODS, PRIMARY_METHODS, Method
from gatefall.study.plan import RunSpec, Step, build_plan, matrix_summary
from gatefall.study.protocol import BATCH_SIZE, EPOCHS, FROZEN_PROTOCOL, SEEDS, protocol_sha256

LABEL_NAMES = tuple(str(i) for i in range(10))
VISUAL_DIMS = {"none": 0, "dinov3": 1536, "sam3": 10}


def _expect(error: type[BaseException], action, message: str) -> None:
    _raised(error, action, message)


def _raised(error: type[BaseException], action, message: str) -> BaseException:
    try:
        action()
    except error as caught:
        return caught
    raise AssertionError(message)


def test_protocol_and_plan() -> None:
    from gatefall.experiments.physics.training import validate_recipe

    digest = protocol_sha256()
    assert digest == protocol_sha256() and len(digest) == 64
    changed = json.loads(json.dumps(FROZEN_PROTOCOL)); changed["training"]["epochs"] = 31
    assert protocol_sha256(changed) != digest
    steps = build_plan()
    counts = matrix_summary(steps)
    expected = {"le2i": (80, 80, 0), "upfall": (40, 40, 0), "transfer": (40, 0, 40),
                "motion": (150, 150, 0), "ablations": (50, 50, 0)}
    for matrix, (train, evaluate, transfer) in expected.items():
        found = counts[matrix]
        assert (found["train"], found["evaluate"], found["transfer"]) == (train, evaluate, transfer), (matrix, found)
    # 80 + 40 + 150 + 30: ablações learned/physics e transferência reaproveitam os runs primários.
    assert counts["unique"]["train"] == 300 and counts["unique"]["transfer"] == 40
    trains = [step for step in steps if step.stage == "train"]
    identities = [step.run.identity_sha256() for step in trains if step.run]
    directories = [step.run.run_dir for step in trains if step.run]
    assert len(set(identities)) == len(identities) == len(set(directories))
    for step in trains:
        assert step.run is not None
        validate_recipe(step.run.method.arm, step.run.method.fusion, step.run.method.visual, EPOCHS,
                        BATCH_SIZE, "best_val", False, 0.0, .05)
        assert step.run.seed in SEEDS and "--seed" in step.command
    first_evaluation = min(index for index, step in enumerate(steps) if step.stage == "evaluate")
    assert all(index < first_evaluation for index, step in enumerate(steps) if step.stage == "train")
    shared = {step.key for step in trains if {"upfall", "ablations"} <= step.matrices}
    assert len(shared) == 20 and all(("b1-learned" in key or "b1-physics" in key or "c1-learned" in key
                                      or "c1-physics" in key) for key in shared)
    motion = [step for step in trains if step.run and step.run.recipe]
    assert all(str(step.run.run_dir).startswith(("runs/local/le2i/motion-", "runs/local/le2i_cv/motion-"))
               for step in motion if step.run)
    unverified = [step.key for step in steps if step.stage in ("bundle", "targets", "train", "evaluate", "transfer")
                  and step.verify is None]
    assert not unverified and next(step for step in steps if step.key == "data:upfall:stage").verify is not None
    transfers = [step for step in steps if step.stage == "transfer"]
    assert all("runs/local/gmdcsa24/study2026/bundles/" in " ".join(step.command) for step in transfers)
    assert {step.run.dataset for step in transfers if step.run} == {"upfall"}
    _expect(ValueError, lambda: build_plan(("quarta-matriz",)), "matriz desconhecida aceita")
    print(f"[PASS] protocolo {digest[:12]}; matrizes 80/40/40/150/50, 300 treinos únicos, teste após treino")


def _fixture_records(perturb_after: int | None, visual: str, k: int = 48):
    from gatefall.experiments.physics.data import RawVideo
    from gatefall.experiments.physics.selftest import fixture_pose

    rng = np.random.default_rng(7)
    records = []
    for split in ("train", "val", "test"):
        pose = fixture_pose(k)
        visual_values = rng.normal(size=(k, VISUAL_DIMS[visual])).astype(np.float32)
        if visual == "sam3":
            visual_values[:, 0] = 1
        quality = np.full(k, .8, dtype=np.float32)
        labels = np.zeros(k, dtype=np.int64); labels[20:30] = 1; labels[30:40] = 2
        if split == "test" and perturb_after is not None:
            later = slice(perturb_after + 1, None)
            pose.keypoints[later, :, :2] += 37.0
            pose.person_found[perturb_after + 3] = False
            visual_values[later] = rng.normal(size=visual_values[later].shape)
            if visual == "sam3":
                visual_values[later, 0] = 1
            quality[later] = .1
        records.append(RawVideo(f"room/{split}", split, labels, pose, visual_values, quality))
    return records


@torch.no_grad()
def _window_logits(model, bundle: Path, stats: dict | None, raw_pose: bool) -> tuple[dict, dict]:
    from gatefall.experiments.physics.data import load_bundle
    from gatefall.experiments.physics.training import Windows, fit_stats

    manifest, videos = load_bundle(bundle)
    stats = fit_stats(manifest, videos) if stats is None else stats
    data = Windows(manifest, videos, stats, "test", 1, labeled_only=False, raw_pose=raw_pose)
    result = {}
    for index in range(len(data)):
        item = data[index]
        logits = model(item["pose"][None], item["visual"][None], item["evidence"][None])
        result[data.index[index][1]] = logits[0]
    return result, stats


def test_causality() -> None:
    """Predição em t não muda quando só quadros posteriores a t mudam (todos os métodos e o classificador motion)."""
    from gatefall.experiments.physics.data import prepare_records
    from gatefall.experiments.physics.training import build_classifier

    cutoff = 30
    cases = [(method, None) for method in PRIMARY_METHODS] + [
        (Method("C1-physics/motion", "C1", "physics", "sam3"), [1, 3])]
    with tempfile.TemporaryDirectory(prefix="gatefall-study-causal-") as temporary:
        root = Path(temporary)
        bundles: dict[str, tuple[Path, Path]] = {}
        for visual in VISUAL_DIMS:
            pair = []
            for name, after in (("original", None), ("future", cutoff)):
                pair.append(prepare_records(_fixture_records(after, visual), root / f"{visual}-{name}",
                                            dataset="synthetic", visual_kind=visual, label_names=LABEL_NAMES,
                                            sources={}))
            bundles[visual] = (pair[0], pair[1])
        for method, horizons in cases:
            torch.manual_seed(3)
            model = build_classifier(VISUAL_DIMS[method.visual], method.fusion, horizons).eval()
            original, future = bundles[method.visual]
            raw = method.fusion == "tcnte"
            before, stats = _window_logits(model, original, None, raw)
            after, future_stats = _window_logits(model, future, None, raw)
            assert future_stats == stats, "estatísticas de normalização dependem de val/test"
            for end, logits in before.items():
                if end <= cutoff:
                    assert torch.equal(logits, after[end]), f"{method.name}: janela {end} viu o futuro"
            assert any(not torch.equal(before[end], after[end]) for end in before if end > cutoff), (
                f"{method.name}: perturbação futura não alterou nenhuma janela posterior")
    print(f"[PASS] causalidade: {len(cases)} classificadores invariantes a quadros após t; "
          "normalização só do treino")


def _prediction_rows(video_id: str, labels: list[int], predictions: list[int]) -> list[dict]:
    return [{"video_id": video_id, "k_end": k, "label": label, "prediction": prediction}
            for k, (label, prediction) in enumerate(zip(labels, predictions))]


def test_metrics() -> None:
    from gatefall.study.metrics import cluster_statistics, confusion, evaluation_metrics, macro_f1

    fall = [8] * 10 + [1] * 10 + [2] * 20 + [6] * 10
    during = [8] * 12 + [1] * 5 + [0] * 33
    after = [8] * 10 + [0] * 12 + [2] * 6 + [0] * 22
    missed = [8] * 50
    adl = [0] * 60
    adl_pred = [0] * 20 + [1] * 4 + [0] * 36
    rows = (_prediction_rows("s1/v1", fall, during) + _prediction_rows("s1/v2", fall, after)
            + _prediction_rows("s2/v3", fall, missed) + _prediction_rows("s2/v4", adl, adl_pred))
    metrics = evaluation_metrics({"predictions": rows}, LABEL_NAMES)
    events = metrics["events"]
    assert (events["n_events"], events["n_detected"], events["n_missed"]) == (3, 2, 1)
    assert events["n_detected_during_fall"] == 1 and events["n_detected_only_after_fall"] == 1
    assert abs(events["latency_detected_s"]["median"] - .9) < 1e-9
    assert abs(events["latency_median_all_events_s"] - 1.4) < 1e-9
    assert events["false_alarms"]["n"] == 1
    assert abs(events["false_alarms"]["hours_total"] - 210 / 10 / 3600) < 1e-12
    assert abs(events["false_alarms"]["per_hour"] - 1 / (210 / 10 / 3600)) < 1e-6
    assert metrics["supported_classes"] == [0, 1, 2, 6, 8]
    matrix = confusion(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert abs(macro_f1(matrix, (0, 1)) - (2 / 3 + .8) / 2) < 1e-12
    assert abs(metrics["macro_f1_supported"] - np.mean([metrics["per_class"][c]["f1"] for c in (0, 1, 2, 6, 8)])) < 1e-12
    two_missed = evaluation_metrics({"predictions": _prediction_rows("a", fall, missed)
                                     + _prediction_rows("b", fall, missed) + _prediction_rows("c", fall, during)},
                                    LABEL_NAMES)
    assert two_missed["events"]["latency_median_all_events_s"] is None
    clusters = cluster_statistics(rows, lambda video_id: video_id.split("/")[0])
    assert set(clusters) == {"s1", "s2"} and clusters["s1"]["events"] == 2 and clusters["s2"]["false_alarms"] == 1
    _expect(ValueError, lambda: macro_f1(matrix, ()), "macro-F1 sem classes aceito")
    print("[PASS] F1 por classe, macro-F1 com suporte, durante/após a queda, FA/h e latência com perdas")


def test_statistics() -> None:
    from gatefall.study.metrics import cluster_statistics
    from gatefall.study.statistics import cluster_bootstrap, metric_functions, paired_seed_deltas

    deltas = paired_seed_deltas({42: .6, 43: .7}, {42: .5, 43: .75})
    assert abs(deltas["mean"] - .025) < 1e-12 and deltas["n_positive"] == 1 and deltas["n_negative"] == 1
    _expect(ValueError, lambda: paired_seed_deltas({42: 1.0}, {43: 1.0}), "sementes não pareadas aceitas")
    fall = [0] * 10 + [1] * 10 + [2] * 10
    rows = []
    for subject in range(6):
        predicted = fall if subject % 2 == 0 else [0] * 30
        rows += _prediction_rows(f"s{subject}/v", fall, predicted)
    stats = {seed: cluster_statistics(rows, lambda video_id: video_id.split("/")[0]) for seed in (42, 43)}
    functions = metric_functions((0, 1, 2))
    same = cluster_bootstrap(stats, stats, functions["event_sensitivity"], resamples=200)
    assert same["estimate"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    single = cluster_bootstrap(stats, None, functions["event_sensitivity"], resamples=200)
    again = cluster_bootstrap(stats, None, functions["event_sensitivity"], resamples=200)
    assert single == again and single["estimate"] == .5 and single["ci_low"] < .5 < single["ci_high"]
    other = {42: {key: value for key, value in stats[42].items() if key != "s0"}, 43: stats[43]}
    _expect(ValueError, lambda: cluster_bootstrap(other, None, functions["event_sensitivity"]),
            "clusters divergentes aceitos")
    print("[PASS] diferenças pareadas por semente e bootstrap determinístico por cluster")


def test_subject_clustering() -> None:
    """Várias gravações por sujeito: o bootstrap por sujeito não trata gravações como independentes."""
    import pandas as pd

    from gatefall.study.metrics import cluster_statistics
    from gatefall.study.report import IncompatibleAggregation, cluster_maps
    from gatefall.study.statistics import cluster_bootstrap, metric_functions

    fall = [0] * 10 + [1] * 10 + [2] * 10
    videos = {f"scene/s{subject}-r{recording}": subject for subject in (1, 2) for recording in range(3)}
    frames = pd.DataFrame([{"video_id": video, "split": "test", "subject": subject}
                           for video, subject in videos.items()]
                          + [{"video_id": "scene/train", "split": "train", "subject": 9}])
    maps = cluster_maps("le2i", set(videos), frames)
    assert len(set(maps["subject"].values())) == 2 and len(set(maps["recording"].values())) == 6
    by_seed: dict[str, dict[int, dict]] = {"subject": {}, "recording": {}}
    for seed in SEEDS:
        rows = []
        for video, subject in videos.items():
            rows += _prediction_rows(video, fall, fall if subject == 1 else [0] * 30)
        for unit in by_seed:
            by_seed[unit][seed] = cluster_statistics(rows, maps[unit].__getitem__)
    sensitivity = metric_functions((0, 1, 2))["event_sensitivity"]
    subject = cluster_bootstrap(by_seed["subject"], None, sensitivity, resamples=400)
    recording = cluster_bootstrap(by_seed["recording"], None, sensitivity, resamples=400)
    assert subject["n_clusters"] == 2 and recording["n_clusters"] == 6
    assert subject["estimate"] == recording["estimate"] == .5
    assert subject["ci_low"] == 0 and subject["ci_high"] == 1
    assert subject["ci_high"] - subject["ci_low"] > recording["ci_high"] - recording["ci_low"]
    paired = cluster_bootstrap(by_seed["subject"], by_seed["subject"], sensitivity, resamples=100)
    assert paired["ci_low"] == paired["ci_high"] == 0
    ambiguous = pd.concat([frames, pd.DataFrame([{"video_id": "scene/s1-r0", "split": "test", "subject": 2}])])
    _expect(IncompatibleAggregation, lambda: cluster_maps("le2i", set(videos), ambiguous), "vídeo com dois sujeitos aceito")
    _expect(IncompatibleAggregation, lambda: cluster_maps("le2i", set(videos) - {"scene/s2-r2"}, frames),
            "teste incompleto aceito")
    views = [f"Subject{s}/Activity1/Trial1/Subject{s}Activity1Trial1Camera{c}" for s in (2, 4) for c in (1, 2)]
    upfall = pd.DataFrame([{"video_id": v, "split": "test", "subject": int(v[7])} for v in views])
    staged = cluster_maps("upfall", set(views), upfall)
    assert len(set(staged["recording"].values())) == 2 and len(set(staged["subject"].values())) == 2
    wrong = upfall.assign(subject=[2, 2, 2, 2])
    _expect(IncompatibleAggregation, lambda: cluster_maps("upfall", set(views), wrong), "sujeito divergente aceito")
    print("[PASS] bootstrap por sujeito (primário) × gravação (secundário) com várias gravações por sujeito")


def test_figures() -> None:
    from gatefall.study.figures import scatter, seed_dot_plot, timeline

    for svg in (seed_dot_plot("t", [("A", [.5, .6]), ("B<&>", [])]),
                scatter("t", "x", "y", [("A", .1, .5), ("B", 2.0, 1.0)]),
                timeline("t", LABEL_NAMES, [0, 0, 1, 1, -1], [("A", [0, 1, 1, 1, 0], [3])], 10.0)):
        ElementTree.fromstring(svg)
    print("[PASS] figuras SVG válidas")


def _synthetic_records(splits: tuple[tuple[str, int], ...]) -> list:
    from gatefall.experiments.physics.data import RawVideo
    from gatefall.experiments.physics.selftest import fixture_pose

    records = []
    for split, count in splits:
        for index in range(count):
            pose = fixture_pose(40)
            labels = np.zeros(pose.k, dtype=np.int64); labels[14:22] = 1; labels[22:32] = 2
            records.append(RawVideo(f"room{index}/{split}", split, labels, pose,
                                    np.zeros((pose.k, 0), dtype=np.float32), np.zeros(pose.k, dtype=np.float32)))
    return records


def _synthetic_runs(root: Path) -> tuple[list[Step], dict]:
    from gatefall.experiments.physics.data import prepare_records
    from gatefall.hashing import sha256_file

    frames = root / "frames.parquet"
    frames.write_bytes(b"grade sintetica")
    bundle = prepare_records(_synthetic_records((("train", 2), ("val", 1), ("test", 2))), root / "bundle",
                             dataset="synthetic", visual_kind="none", label_names=LABEL_NAMES,
                             sources={"frames_sha256": sha256_file(frames)})
    bundle_checks: list[str] = []

    def verify_bundle() -> integrity.Verdict:
        bundle_checks.append("bundle")
        return integrity.bundle(bundle, frames)

    bundle_step = Step("bundle:synthetic:none", "bundle", ("bundle",), (bundle / "manifest.json",), (), "cpu",
                       None, frozenset({"le2i"}), verify_bundle)
    steps = [bundle_step]
    for method in (PRIMARY_METHODS[0], PRIMARY_METHODS[-1]):
        trains = []
        for seed in SEEDS:
            run = RunSpec("synthetic", method, seed, None, root / "runs" / method.slug / f"seed-{seed}", bundle)
            command = ("train", method.arm, method.fusion, str(seed), str(run.run_dir))
            trains.append(Step(f"train:{method.slug}:{seed}", "train", command, (run.run_dir / "metrics.json",),
                               (bundle_step.key,), "cpu", run, frozenset({"le2i"}),
                               partial(integrity.trained_run, run)))
        for step in trains:
            assert step.run is not None
            steps.append(step)
        for step in trains:
            assert step.run is not None
            output = step.run.run_dir / "evaluation-test.json"
            steps.append(Step(step.key.replace("train", "evaluate"), "evaluate",
                              ("evaluate", str(step.run.run_dir)), (output,), (step.key,), "cpu", step.run,
                              frozenset({"le2i"}), partial(integrity.evaluation, step.run, output)))
    return steps, {"bundle": bundle, "frames": frames, "bundle_checks": bundle_checks}


def _synthetic_frames():
    import pandas as pd

    return pd.DataFrame([{"video_id": f"room{index}/{split}", "split": split, "subject": index + 1}
                         for split, count in (("train", 2), ("val", 1), ("test", 2)) for index in range(count)])


def _runner(bundle: Path):
    from gatefall.experiments.physics.training import evaluate, train

    def run(command) -> int:
        if command[0] == "train":
            _, arm, fusion, seed, run_dir = command
            train(bundle, Path(run_dir), arm=arm, fusion=fusion, seed=int(seed), epochs=EPOCHS,
                  batch_size=BATCH_SIZE)
        elif command[0] == "evaluate":
            evaluate(bundle, Path(command[1]), "test")
        else:
            return 3
        return 0
    return run


def _check_stage_filtered(steps: list[Step], paths: dict, runner, logs: Path) -> None:
    from gatefall.study.registry import StudyExecutionError, execute

    calls: list[tuple[str, ...]] = []

    def guarded(command) -> int:
        calls.append(tuple(command))
        return runner(command)

    trains = [step for step in steps if step.stage == "train"]
    evaluations = [step for step in steps if step.stage == "evaluate"]
    bundle, frames = paths["bundle"], paths["frames"]

    def snapshot() -> dict[Path, bytes]:
        runs = [step.run for step in trains if step.run is not None]
        files = [path for run in runs for path in run.run_dir.iterdir()]
        return {path: path.read_bytes() for path in files}

    evaluations[0].outputs[0].unlink()
    paths["bundle_checks"].clear()
    execute(evaluations, plan=steps, log_dir=logs, runner=guarded, allow_dirty=True, weights={})
    assert calls == [evaluations[0].command], "estágio filtrado executou passo não selecionado"
    assert paths["bundle_checks"] == ["bundle"], "dependência compartilhada conferida mais de uma vez"
    calls.clear()
    error = _raised(StudyExecutionError, lambda: execute(evaluations, log_dir=logs, runner=guarded,
                                                         allow_dirty=True, weights={}), "seleção sem plano aceita")
    assert "fora do plano" in str(error) and calls == []

    video = next((bundle / "videos").iterdir())
    content = video.read_bytes()
    hidden = bundle.with_name("bundle-hidden")
    cases = (
        ("ausente", lambda: bundle.rename(hidden), lambda: hidden.rename(bundle)),
        ("desatualizado", lambda: frames.write_bytes(b"grade nova"), lambda: frames.write_bytes(b"grade sintetica")),
        ("inválido", lambda: video.write_bytes(content[:-1]), lambda: video.write_bytes(content)),
    )
    for name, mutate, restore in cases:
        before = snapshot()
        mutate()
        try:
            for selection in (trains, evaluations):
                error = _raised(StudyExecutionError, lambda: execute(selection, plan=steps, log_dir=logs,
                                                                     runner=guarded, allow_dirty=True, weights={}),
                                f"pacote {name} aceito em estágio filtrado")
                assert "bundle:synthetic:none" in str(error) and "nada foi executado" in str(error), error
            assert calls == [] and snapshot() == before, f"pacote {name}: execução ou sobrescrita"
            assert name != "ausente" or not bundle.exists(), "pacote ausente recriado"
        finally:
            restore()
    print("[PASS] estágios filtrados conferem pré-requisitos uma vez e recusam pacote ausente, desatualizado ou inválido")


def _check_external_report(steps: list[Step], paths: dict, root: Path) -> None:
    from gatefall.experiments.physics.data import prepare_records
    from gatefall.experiments.physics.training import evaluate_external
    from gatefall.study.report import IncompatibleAggregation, build_report

    target = prepare_records(_synthetic_records((("test", 2),)), root / "target-bundle", dataset="synthetic-ext",
                             visual_kind="none", label_names=LABEL_NAMES, sources={}, required_splits=("test",))
    transfers = []
    for step in steps:
        if step.stage != "train" or step.run is None:
            continue
        output = step.run.run_dir / "evaluation-external-synthetic-ext.json"
        evaluate_external(paths["bundle"], step.run.run_dir, target)
        transfers.append(Step(step.key.replace("train", "transfer"), "transfer", ("transfer",), (output,),
                              (step.key,), "cpu", step.run, frozenset({"transfer"}),
                              partial(integrity.evaluation, step.run, output, target)))
    frames = {"synthetic-ext": _synthetic_frames()}
    assert (build_report([*steps, *transfers], "transfer", root / "reports", frames) / "summary.csv").is_file()
    manifest = target / "manifest.json"
    original = manifest.read_bytes()
    manifest.write_bytes(original + b"\n")
    error = _raised(IncompatibleAggregation, lambda: build_report([*steps, *transfers], "transfer",
                                                                  root / "reports-stale", frames),
                    "avaliação externa com alvo desatualizado agregada")
    assert "target_manifest_sha256" in str(error) and not (root / "reports-stale").exists()
    manifest.write_bytes(original)
    print("[PASS] relatório recusa avaliação externa cujo pacote alvo mudou desde a avaliação")


def test_registry_and_report() -> None:
    from gatefall.study.registry import (
        RUN_PROVENANCE,
        StudyExecutionError,
        execute,
        render_commands,
        step_status,
        verify_run_provenance,
        write_run_provenance,
    )
    from gatefall.study.report import IncompatibleAggregation, build_report

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="gatefall-study-registry-") as temporary:
        root = Path(temporary)
        steps, paths = _synthetic_runs(root)
        runner = _runner(paths["bundle"])
        logs = root / "logs"
        assert "evaluate" in render_commands(steps)
        _expect(IncompatibleAggregation, lambda: build_report(steps, "le2i", root / "reports", {"synthetic": _synthetic_frames()}),
                "relatório sem avaliações aceito")
        manual = Step("data:x", "data", (), (root / "absent",), (), "manual")
        _expect(StudyExecutionError, lambda: execute([manual], log_dir=logs, runner=runner, allow_dirty=True,
                                                     weights={}), "pré-requisito ausente aceito")
        failing = Step("x:fail", "data", ("fail",), (root / "never",), (), "cpu")
        _expect(StudyExecutionError, lambda: execute([failing, *steps], log_dir=logs, runner=runner,
                                                     allow_dirty=True, weights={}), "falha não interrompeu")
        work = [step for step in steps if step.stage != "bundle"]
        assert not any(step.outputs[0].exists() for step in work), "passo executado após falha"
        log = execute(steps, log_dir=logs, runner=runner, allow_dirty=True, weights={})
        events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
        assert events.count("finished") == len(work) and events[-1] == "complete"
        trained = [step for step in steps if step.stage == "train"]
        assert all(verify_run_provenance(step.run)["origin"] == "executed" for step in trained if step.run)
        relog = execute(steps, log_dir=logs, runner=runner, allow_dirty=True, weights={})
        assert [json.loads(line)["event"] for line in relog.read_text().splitlines()].count("reused") == len(steps)
        resumed_step = next(step for step in steps if step.stage == "evaluate")
        resumed_step.outputs[0].unlink()
        resumed = [json.loads(line) for line in execute(steps, log_dir=logs, runner=runner, allow_dirty=True,
                                                        weights={}).read_text().splitlines()]
        assert [entry["key"] for entry in resumed if entry["event"] == "finished"] == [resumed_step.key]
        assert sum(entry["event"] == "reused" for entry in resumed) == len(steps) - 1
        _check_stage_filtered(steps, paths, runner, logs)
        _check_external_report(steps, paths, root)
        report = build_report(steps, "le2i", root / "reports", {"synthetic": _synthetic_frames()})
        for name in ("summary.csv", "summary.md", "runs.csv", "per_class.csv", "contrasts.csv", "cost.csv",
                     "report-manifest.json", "macro-f1-synthetic.svg", "timeline-synthetic-1.svg"):
            assert (report / name).is_file(), name
        contrast = (report / "contrasts.csv").read_text()
        assert "TCNTE,A,macro_f1_supported" in contrast
        evaluation = trained[0].run.run_dir / "evaluation-test.json" if trained[0].run else None
        assert evaluation is not None
        original = evaluation.read_text()
        tampered = json.loads(original); tampered["predictions"][0]["label"] = 9
        evaluation.write_text(json.dumps(tampered))
        _expect(IncompatibleAggregation, lambda: build_report(steps, "le2i", root / "reports", {"synthetic": _synthetic_frames()}),
                "rótulos de teste divergentes aceitos")
        evaluation.write_text(original)
        spare = trained[1].run
        assert spare is not None
        provenance = json.loads((spare.run_dir / RUN_PROVENANCE).read_text())
        provenance["identity_sha256"] = "0" * 64
        (spare.run_dir / RUN_PROVENANCE).write_text(json.dumps(provenance))
        _expect(StudyExecutionError, lambda: verify_run_provenance(spare), "identidade adulterada aceita")
        last = trained[2].run
        assert last is not None
        (last.run_dir / "checkpoint.pt").write_bytes(b"x")
        _expect(StudyExecutionError, lambda: verify_run_provenance(last), "checkpoint adulterado aceito")
        kept = trained[3].run
        assert kept is not None
        _expect(FileExistsError, lambda: write_run_provenance(kept, "executed", None, {}),
                "proveniência sobrescrevível")
        assert [step_status(step) for step in trained[1:3]] == ["invalid", "invalid"]
        evaluated = next(step for step in steps if step.stage == "evaluate" and step.run == kept)
        assert step_status(evaluated) == "done"
        report_path = evaluated.outputs[0]
        original = report_path.read_text()
        report_path.write_text(json.dumps({**json.loads(original), "selection_performed": True}))
        assert step_status(evaluated) == "invalid", "avaliação com seleção no teste reusada"
        report_path.write_text(original)
        before = sorted(path.name for path in kept.run_dir.iterdir())
        _expect(StudyExecutionError, lambda: execute(steps, log_dir=logs, runner=runner, allow_dirty=True,
                                                     weights={}), "run adulterado reusado")
        assert sorted(path.name for path in kept.run_dir.iterdir()) == before, "recusa alterou o run"
        manifest = paths["bundle"] / "manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
        assert step_status(trained[3]) == "invalid" and step_status(evaluated) == "invalid", "pacote alterado aceito"
    print("[PASS] executor ordenado, reuso verificado, proveniência imutável e relatório com recusas")


def _camera_zip(directory: Path, camera: int) -> Path:
    archive = directory / f"Subject1Activity1Trial1Camera{camera}.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(f"2018-07-04T12_04_17.{camera:06d}.png", b"frame" * camera)
    return archive


def test_step_integrity() -> None:
    from gatefall.data.staged.sequences import stage_upfall_archives
    from gatefall.experiments.physics.data import RawVideo, prepare_records
    from gatefall.experiments.physics.selftest import fixture_pose
    from gatefall.hashing import sha256_file
    from gatefall.study.registry import StudyExecutionError, execute

    with tempfile.TemporaryDirectory(prefix="gatefall-study-integrity-") as temporary:
        root = Path(temporary)
        archives, raw = root / "archives", root / "raw"
        archives.mkdir()
        assert integrity.upfall_staging(archives, raw).status == "pending"
        first = _camera_zip(archives, 1)
        stage_upfall_archives([first], raw)
        assert integrity.upfall_staging(archives, raw).status == "done"
        _camera_zip(archives, 2)
        assert integrity.upfall_staging(archives, raw).status == "pending", "raw existente pulou câmera nova"
        calls: list[tuple[str, ...]] = []

        def runner(command) -> int:
            calls.append(tuple(command))
            stage_upfall_archives(sorted(archives.glob("*.zip")), raw)
            return 0

        stage = Step("data:upfall:stage", "data", ("stage",), (raw,), (), "disco",
                     verify=partial(integrity.upfall_staging, archives, raw))
        logs = root / "logs"
        execute([stage], log_dir=logs, runner=runner, allow_dirty=True, weights={})
        assert len(calls) == 1 and integrity.upfall_staging(archives, raw).status == "done"
        staged = next((raw / "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera2").iterdir())
        staged.write_bytes(b"x")
        assert integrity.upfall_staging(archives, raw).status == "invalid"
        _expect(StudyExecutionError, lambda: execute([stage], log_dir=logs, runner=runner, allow_dirty=True,
                                                     weights={}), "preparação corrompida reusada")
        assert len(calls) == 1 and staged.read_bytes() == b"x", "executor rodou ou sobrescreveu passo inválido"

        frames = root / "frames.parquet"
        frames.write_bytes(b"grade v1")
        records = []
        for split in ("train", "val", "test"):
            pose = fixture_pose(30)
            records.append(RawVideo(f"room/{split}", split, np.zeros(pose.k, dtype=np.int64), pose,
                                    np.zeros((pose.k, 0), dtype=np.float32), np.zeros(pose.k, dtype=np.float32)))
        bundle = prepare_records(records, root / "bundle", dataset="synthetic", visual_kind="none",
                                 label_names=LABEL_NAMES, sources={"frames_sha256": sha256_file(frames)})
        assert integrity.bundle(root / "absent", frames).status == "pending"
        assert integrity.bundle(bundle, frames).status == "done"
        frames.write_bytes(b"grade v2")
        assert integrity.bundle(bundle, frames).status == "invalid", "pacote de grade antiga reusado"
        frames.write_bytes(b"grade v1")
        video = next((bundle / "videos").iterdir())
        content = video.read_bytes()
        video.write_bytes(content[:-1])
        assert "tamanho divergente" in integrity.bundle(bundle, frames).reasons[0]
        video.unlink()
        assert "ausentes" in integrity.bundle(bundle, frames).reasons[0], "pacote incompleto reusado"
        video.write_bytes(content)
        (bundle / "videos" / "extra.h5").write_bytes(b"x")
        assert "não listados" in integrity.bundle(bundle, frames).reasons[0]
        (bundle / "videos" / "extra.h5").unlink()
        assert integrity.bundle(bundle, frames).status == "done"
        (bundle / "manifest.json").unlink()
        assert integrity.bundle(bundle, frames).status == "invalid", "pacote sem manifesto tratado como pendente"
    print("[PASS] integridade antes do reuso: câmera nova, preparação corrompida, pacote incompleto ou desatualizado")


def test_reuse_audit() -> None:
    from gatefall.study.reuse import audit_external_runs

    steps = build_plan(("motion",))
    with tempfile.TemporaryDirectory(prefix="gatefall-study-reuse-") as temporary:
        source = Path(temporary)
        empty = audit_external_runs(steps, source)
        assert empty["planned"] == 150 and empty["counts"]["absent"] == 150
        target = next(step for step in steps if step.stage == "train" and step.run and step.run.recipe)
        assert target.run is not None
        candidate = source / target.run.run_dir.relative_to(target.run.run_dir.parents[3])
        candidate.mkdir(parents=True)
        (candidate / "config.json").write_text("{}")
        partial = audit_external_runs(steps, source)
        entry = next(item for item in partial["entries"] if item["key"] == target.key)
        assert entry["status"] == "incompatible" and "arquivos ausentes" in entry["reasons"][0]
    assert len({method.slug for method in GATE_ABLATIONS}) == len(GATE_ABLATIONS)
    print("[PASS] auditoria de reuso: 150 runs planejados, ausentes e incompletos recusados")


def _historical_runs(root: Path) -> tuple[list[Step], dict]:
    """Runs de movimento treinados sobre o pacote "original", planejados sobre o pacote atual equivalente."""
    from gatefall.experiments.motion.selftest import historical_pair
    from gatefall.experiments.physics.training import train

    original_bundle, original_targets, bundle, targets = historical_pair(root / "artifacts")
    trains, evaluations = [], []
    for seed in SEEDS:
        run = RunSpec("synthetic", MOTION_METHODS[0], seed, "baseline", root / "runs" / f"seed-{seed}", bundle,
                      targets, root / "equivalence" / "motion-sam3.json")
        train(original_bundle, run.run_dir, arm=run.method.arm, fusion=run.method.fusion, seed=seed,
              epochs=EPOCHS, batch_size=BATCH_SIZE, motion_recipe="baseline", targets=original_targets)
        key = f"train:synthetic:motion-a:seed-{seed}:baseline"
        trains.append(Step(key, "train", ("train", str(run.run_dir)), (run.run_dir / "metrics.json",), (), "cpu",
                           run, frozenset({"motion"}), partial(integrity.trained_run, run)))
        output = run.run_dir / "evaluation-test.json"
        evaluations.append(Step(key.replace("train", "evaluate"), "evaluate", ("evaluate", str(run.run_dir)),
                                (output,), (key,), "cpu", run, frozenset({"motion"}),
                                partial(integrity.evaluation, run, output)))
    return [*trains, *evaluations], {"original_bundle": original_bundle, "original_targets": original_targets,
                                     "bundle": bundle, "targets": targets}


def test_attested_reuse() -> None:
    import pandas as pd

    from gatefall.experiments.motion.equivalence import attest
    from gatefall.experiments.physics.training import evaluate
    from gatefall.hashing import sha256_file
    from gatefall.study.registry import (
        RUN_PROVENANCE,
        StudyExecutionError,
        execute,
        run_provenance,
        verify_run_provenance,
    )
    from gatefall.study.report import IncompatibleAggregation, build_report
    from gatefall.study.reuse import audit_candidate

    plan = build_plan(("motion",))
    commands = {step.key: step.command for step in plan if step.stage == "evaluate"}
    attested = commands["evaluate:le2i:motion-a:seed-42:baseline"]
    assert attested[-2:] == ("--equivalence", "runs/local/le2i/study2026/equivalence/motion-sam3.json"), attested
    assert not any("--equivalence" in step.command for step in build_plan(("le2i",)))
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="gatefall-study-attested-") as temporary:
        root = Path(temporary)
        steps, paths = _historical_runs(root)
        trains = [step for step in steps if step.stage == "train"]
        first = trains[0].run
        assert first is not None and first.equivalence is not None
        verdict = integrity.trained_run(first)
        assert verdict.status == "invalid" and "pacote" in verdict.reasons[0], verdict
        assert audit_candidate(trains[0], first.run_dir)[0], "run histórico sem atestado auditado como compatível"
        # Avaliação histórica, feita sobre o pacote original, não vale para o run atestado.
        evaluate(paths["original_bundle"], first.run_dir, "test")
        attest(paths["original_bundle"], paths["original_targets"], paths["bundle"], paths["targets"],
               first.equivalence)
        reasons, used = audit_candidate(trains[0], first.run_dir)
        assert reasons == [] and used is not None and used["sha256"] == sha256_file(first.equivalence)
        assert all(integrity.trained_run(step.run).status == "done" for step in trains if step.run)
        historical = integrity.evaluation(first, first.run_dir / "evaluation-test.json")
        assert historical.status == "invalid" and "equivalence.sha256" in historical.reasons[0], historical
        (first.run_dir / "evaluation-test.json").unlink()
        assert integrity.trained_run(replace(first, equivalence=None)).status == "invalid", "atestado implícito"
        _expect(StudyExecutionError, lambda: run_provenance(first, "executed", None, {}),
                "run atestado registrado como executado")

        def runner(command) -> int:
            if command[0] != "evaluate":
                return 3
            evaluate(paths["bundle"], Path(command[1]), "test", first.equivalence)
            return 0

        log = execute(steps, log_dir=root / "logs", runner=runner, allow_dirty=True, weights={})
        events = [json.loads(line) for line in log.read_text().splitlines()]
        assert [entry["key"] for entry in events if entry["event"] == "reused"] == [step.key for step in trains]
        assert sum(entry["event"] == "finished" for entry in events) == len(SEEDS)
        equivalence_sha256 = verify_run_provenance(first)["equivalence"]["sha256"]
        assert all(verify_run_provenance(step.run)["origin"] == "reused-verified" for step in trains if step.run)
        assert all(step.verify is not None and step.verify().status == "done" for step in steps)
        frames = {"synthetic": pd.DataFrame([{"video_id": "room/test", "split": "test", "subject": 1}])}
        report = build_report(steps, "motion", root / "reports", frames)
        rows = (report / "runs.csv").read_text().splitlines()
        assert len(rows) == len(SEEDS) + 1 and all(f"reused-verified,{equivalence_sha256}" in row for row in rows[1:])

        attestation = first.equivalence.read_bytes()
        first.equivalence.write_bytes(attestation + b"\n")
        assert integrity.trained_run(first).status == "invalid", "atestado trocado após o registro aceito"
        _expect(IncompatibleAggregation, lambda: build_report(steps, "motion", root / "reports-stale", frames),
                "relatório com atestado trocado")
        record = json.loads(attestation)
        record["allowed_differences"]["targets"] = []
        first.equivalence.write_text(json.dumps(record))
        assert all(step.verify is not None and step.verify().status == "invalid" for step in steps)
        first.equivalence.write_bytes(attestation)
        manifest = paths["bundle"] / "manifest.json"
        content = manifest.read_bytes()
        manifest.write_bytes(content + b"\n")
        assert all(step.verify is not None and step.verify().status == "invalid" for step in steps)
        manifest.write_bytes(content)
        assert (first.run_dir / RUN_PROVENANCE).is_file() and not (root / "reports-stale").exists()
    print("[PASS] reuso atestado: auditoria, run e proveniência reused-verified, avaliação refeita no pacote "
          "atual, relatório e recusas de atestado ou pacote alterados")


def run_selftest() -> None:
    test_protocol_and_plan()
    test_metrics()
    test_statistics()
    test_subject_clustering()
    test_figures()
    test_reuse_audit()
    test_attested_reuse()
    test_step_integrity()
    test_causality()
    test_registry_and_report()
    print("study selftest PASS")


if __name__ == "__main__":
    run_selftest()
