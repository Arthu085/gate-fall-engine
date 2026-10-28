"""Contratos sintéticos do estudo: plano, causalidade, métricas, estatística, registro e relatório."""

import json
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import torch

from gatefall.study.methods import GATE_ABLATIONS, PRIMARY_METHODS, Method
from gatefall.study.plan import RunSpec, Step, build_plan, matrix_summary
from gatefall.study.protocol import BATCH_SIZE, EPOCHS, FROZEN_PROTOCOL, SEEDS, protocol_sha256

LABEL_NAMES = tuple(str(i) for i in range(10))
VISUAL_DIMS = {"none": 0, "dinov3": 1536, "sam3": 10}


def _expect(error: type[BaseException], action, message: str) -> None:
    try:
        action()
    except error:
        return
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
            after, _ = _window_logits(model, future, stats, raw)
            for end, logits in before.items():
                if end <= cutoff:
                    assert torch.equal(logits, after[end]), f"{method.name}: janela {end} viu o futuro"
            assert any(not torch.equal(before[end], after[end]) for end in before if end > cutoff), (
                f"{method.name}: perturbação futura não alterou nenhuma janela posterior")
    print(f"[PASS] causalidade: {len(cases)} classificadores invariantes a quadros após t")


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


def test_figures() -> None:
    from gatefall.study.figures import scatter, seed_dot_plot, timeline

    for svg in (seed_dot_plot("t", [("A", [.5, .6]), ("B<&>", [])]),
                scatter("t", "x", "y", [("A", .1, .5), ("B", 2.0, 1.0)]),
                timeline("t", LABEL_NAMES, [0, 0, 1, 1, -1], [("A", [0, 1, 1, 1, 0], [3])], 10.0)):
        ElementTree.fromstring(svg)
    print("[PASS] figuras SVG válidas")


def _synthetic_runs(root: Path) -> tuple[list[Step], dict[str, Path]]:
    from gatefall.experiments.physics.data import RawVideo, prepare_records
    from gatefall.experiments.physics.selftest import fixture_pose

    records = []
    for split, count in (("train", 2), ("val", 1), ("test", 2)):
        for index in range(count):
            pose = fixture_pose(40)
            labels = np.zeros(pose.k, dtype=np.int64); labels[14:22] = 1; labels[22:32] = 2
            records.append(RawVideo(f"room{index}/{split}", split, labels, pose,
                                    np.zeros((pose.k, 0), dtype=np.float32), np.zeros(pose.k, dtype=np.float32)))
    bundle = prepare_records(records, root / "bundle", dataset="synthetic", visual_kind="none",
                             label_names=LABEL_NAMES, sources={})
    steps = []
    for method in (PRIMARY_METHODS[0], PRIMARY_METHODS[-1]):
        trains = []
        for seed in SEEDS:
            run = RunSpec("synthetic", method, seed, None, root / "runs" / method.slug / f"seed-{seed}", bundle)
            command = ("train", method.arm, method.fusion, str(seed), str(run.run_dir))
            trains.append(Step(f"train:{method.slug}:{seed}", "train", command, (run.run_dir / "metrics.json",),
                               (), "cpu", run, frozenset({"le2i"})))
        for step in trains:
            assert step.run is not None
            steps.append(step)
        for step in trains:
            assert step.run is not None
            steps.append(Step(step.key.replace("train", "evaluate"), "evaluate",
                              ("evaluate", str(step.run.run_dir)),
                              (step.run.run_dir / "evaluation-test.json",), (step.key,), "cpu", step.run,
                              frozenset({"le2i"})))
    return steps, {"bundle": bundle}


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


def test_registry_and_report() -> None:
    from gatefall.study.registry import (
        RUN_PROVENANCE,
        StudyExecutionError,
        execute,
        render_commands,
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
        _expect(IncompatibleAggregation, lambda: build_report(steps, "le2i", root / "reports"),
                "relatório sem avaliações aceito")
        manual = Step("data:x", "data", (), (root / "absent",), (), "manual")
        _expect(StudyExecutionError, lambda: execute([manual], log_dir=logs, runner=runner, allow_dirty=True,
                                                     weights={}), "pré-requisito ausente aceito")
        failing = Step("x:fail", "data", ("fail",), (root / "never",), (), "cpu")
        _expect(StudyExecutionError, lambda: execute([failing, *steps], log_dir=logs, runner=runner,
                                                     allow_dirty=True, weights={}), "falha não interrompeu")
        assert not any(step.outputs[0].exists() for step in steps), "passo executado após falha"
        log = execute(steps, log_dir=logs, runner=runner, allow_dirty=True, weights={})
        events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
        assert events.count("finished") == len(steps) and events[-1] == "complete"
        trained = [step for step in steps if step.stage == "train"]
        assert all(verify_run_provenance(step.run)["origin"] == "executed" for step in trained if step.run)
        relog = execute(steps, log_dir=logs, runner=runner, allow_dirty=True, weights={})
        assert [json.loads(line)["event"] for line in relog.read_text().splitlines()].count("reused") == len(steps)
        report = build_report(steps, "le2i", root / "reports")
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
        _expect(IncompatibleAggregation, lambda: build_report(steps, "le2i", root / "reports"),
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
    print("[PASS] executor ordenado, reuso verificado, proveniência imutável e relatório com recusas")


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


def run_selftest() -> None:
    test_protocol_and_plan()
    test_metrics()
    test_statistics()
    test_figures()
    test_reuse_audit()
    test_causality()
    test_registry_and_report()
    print("study selftest PASS")


if __name__ == "__main__":
    run_selftest()
