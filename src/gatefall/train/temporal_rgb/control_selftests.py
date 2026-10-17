"""Controles nativos sintéticos: validadores reais, sem retreino ou dados Le2i."""

import contextlib
import io
import json
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch
import yaml

from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.datasets.le2i import LE2I_LABEL_NAMES, Le2iDatasetAdapter
from gatefall.hashing import sha256_file
from gatefall.train.baseline_a.config import TrainConfig, load_config, save_config
from gatefall.train.imbalance_experiment import cli as imbalance_cli
from gatefall.train.internal_folds import cli as cv_cli, folds as cv_folds, stats as cv_stats, training as cv_training
from gatefall.train.internal_folds.selftests import fixtures as cv_fixtures
from gatefall.train.shared.imbalance import RECIPES
from gatefall.train.shared.metrics import RESTRICTED_CLASSES, classification_summary, restricted_macro_f1, support_by_name
from gatefall.train.shared.tcn import TCNClassifier
from gatefall.train.subject_folds import cli as cs_cli, folds as cs_folds, stats as cs_stats, training as cs_training
from gatefall.train.subject_folds.selftests import fixtures as cs_fixtures
from gatefall.train.temporal_rgb import training
from gatefall.train.temporal_rgb import cli as rgb_cli
from gatefall.train.temporal_rgb.data import Context, Partition, fold_windows, load_context


def _split(labels: np.ndarray) -> dict[str, Any]:
    macro, classes = restricted_macro_f1(labels, labels)
    return {
        "macro_f1_restricted": macro, "f1_by_class": {str(key): value for key, value in classes.items()},
        "support": support_by_name(labels, LE2I_LABEL_NAMES),
        **classification_summary(labels, labels, LE2I_LABEL_NAMES),
    }


def _write_run(root: Path, context: Context, partition: Partition) -> Path:
    directory = root / partition.fold / "runs/baseline_a/seed_42"
    directory.mkdir(parents=True)
    record = partition.membership_record
    if context.protocol == "cs":
        document = cs_folds.read_folds_document(root)
        stats = cs_stats.load_fold_stats(
            record, cs_folds.stats_dir(root, partition.fold), context.input_sha256,
            cast(str, document["pose_features_sha256"]),
        )
        config = cs_training.resolve_fold_config(42, stats.path, stats.sha256)
        sidecar = cs_training.fold_run_document(
            record, "A", 42, stats.sha256, context.input_sha256,
            cast(str, document["pose_features_sha256"]), context.membership_sha256,
        )
    else:
        files = cv_stats.load_recorded_stats_files(cv_folds.stats_dir(root, partition.fold), record, context.input_sha256)
        config = cast(TrainConfig, cv_training.resolve_fold_config("A", 42, files, cv_fixtures.QUALITY))
        sidecar = cv_training.fold_run_document(
            partition.fold, cast(str, record["retained_environment"]),
            cast(list[str], record["development_environments"]), "A", 42, files,
        )
    save_config(config, directory / "config.yaml", force=True)
    model = TCNClassifier(config.input_dim, config.channels, config.kernel_size, config.dilations, config.dropout, config.num_classes)
    torch.save(model.state_dict(), directory / "checkpoint.pt")
    development = fold_windows(partition, cv_folds.DEVELOPMENT_SPLIT)["label"].to_numpy(dtype=np.int64)
    retained = fold_windows(partition, cv_folds.RETAINED_SPLIT)["label"].to_numpy(dtype=np.int64)
    metrics = {
        "run_name": config.run_name, "epochs_trained": config.epochs,
        "history": [{"epoch": epoch, "train_loss": 0.1, "val_macro_f1_restricted": 1.0}
                    for epoch in range(1, config.epochs + 1)],
        "restricted_classes": RESTRICTED_CLASSES, "excluded_classes": [5, 6],
        "config_sha256": sha256_file(directory / "config.yaml"),
        "checkpoint_sha256": sha256_file(directory / "checkpoint.pt"),
        "final": {"train": _split(development), "val": _split(retained), "test": _split(retained)},
    }
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    cv_training.publish_fold_run_document(directory, sidecar)
    return directory


@contextlib.contextmanager
def _workspace(protocol: str) -> Iterator[tuple[Path, Context]]:
    fixture = cs_fixtures if protocol == "cs" else cv_fixtures
    with fixture.synthetic_workspace() as paths, contextlib.ExitStack() as stack:
        root, frames_path, manifest_path = paths[:3]
        adapter = Le2iDatasetAdapter(protocol=protocol)
        object.__setattr__(adapter, "frames_path", frames_path)
        object.__setattr__(adapter, "manifest_path", manifest_path)
        if protocol == "cs":
            pose = cs_fixtures.RecordingPose().inputs()
            stack.enter_context(patch.object(cs_cli, "ADAPTER", adapter))
            stack.enter_context(patch.object(cs_cli, "real_pose_inputs", return_value=pose))
            cs_cli.run_prepare(root, frames_path, manifest_path, adapter.pose_root, force=False, pose=pose)
        else:
            stack.enter_context(patch.object(cv_cli, "ADAPTER", adapter))
            cv_cli.run_prepare(root, frames_path, manifest_path, force=False,
                               loaders=cv_fixtures.RecordingLoaders().loaders(), quality=cv_fixtures.QUALITY)
        context = load_context(protocol, root)
        for partition in context.partitions:
            _write_run(root, context, partition)
        stack.enter_context(patch.object(training, "BASELINE_ROOT", root.parent / "imbalance"))
        for module in (cs_training, cv_training, imbalance_cli):
            stack.enter_context(patch.object(module, "run_training", side_effect=AssertionError("retreino automático")))
        stack.enter_context(patch.object(training, "fit_head", side_effect=AssertionError("treino RGB durante verificação")))
        yield root, context


def _reject(action: Any) -> None:
    try:
        action()
    except (OSError, ValueError, RuntimeError, KeyError):
        return
    raise AssertionError("controle inválido aceito")


def _refresh_hashes(directory: Path) -> None:
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["config_sha256"] = sha256_file(directory / "config.yaml")
    metrics["checkpoint_sha256"] = sha256_file(directory / "checkpoint.pt")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    sidecar_path = directory / "fold_run.json"
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        sidecar.pop("artifacts_sha256")
        cv_training.publish_fold_run_document(directory, sidecar)
    experiment_path = directory / "experiment.json"
    if experiment_path.exists():
        experiment = json.loads(experiment_path.read_text())
        experiment["config_sha256"] = metrics["config_sha256"]
        experiment["checkpoint_sha256"] = metrics["checkpoint_sha256"]
        experiment_path.write_text(json.dumps(experiment), encoding="utf-8")


def _check_native(protocol: str) -> None:
    with _workspace(protocol) as (root, context):
        before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        for partition in context.partitions:
            control = training.baseline_control(context, partition, root)
            assert control["source"] == "native" and not control["equivalent_controls"]
            assert "fold_run.json" in control["artifact_sha256"]
            assert control["metrics"]["final"]["val"]["per_class"] == control["per_class"]
        output = io.StringIO()
        with (contextlib.redirect_stdout(output), patch.object(rgb_cli, "load_context", return_value=context),
              patch("sys.argv", ["temporal_rgb", "verify-controls", "--protocol", protocol, "--fold-root", str(root)])):
            rgb_cli.main()
        assert len(json.loads(output.getvalue())["controls"]) == 3
        assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        partition = context.partitions[0]
        directory = root / partition.fold / "runs/baseline_a/seed_42"
        original = {path.name: path.read_bytes() for path in directory.iterdir()}

        def restore() -> None:
            for name, content in original.items():
                (directory / name).write_bytes(content)

        check = lambda: training.baseline_control(context, partition, root)
        for name in original:
            path = directory / name
            path.unlink()
            _reject(check)
            restore()
            path.write_bytes(b"corrupted")
            _reject(check)
            restore()
        sidecar_path = directory / "fold_run.json"
        for field, value in (("fold", "wrong_fold"), ("seed", 43), ("checkpoint_policy", "best_val"),
                             ("evaluation_split_mapping", {"test": "test"}),
                             ("standardization_stats_sha256", {"pose": "0" * 64})):
            sidecar = json.loads(original["fold_run.json"])
            sidecar[field] = value
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            _reject(check)
            restore()
        config_path = directory / "config.yaml"
        for field, value in (("seed", 43), ("epochs", 29)):
            config = yaml.safe_load(config_path.read_text())
            config[field] = value
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            if field == "epochs":
                metrics = json.loads((directory / "metrics.json").read_text())
                metrics["epochs_trained"] = value
                metrics["history"] = metrics["history"][:value]
                (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            _refresh_hashes(directory)
            _reject(check)
            restore()
        changed = partition.frames.copy()
        mask = changed["video_id"] == partition.development_video_ids[0]
        changed.loc[mask, "label"] = np.roll(changed.loc[mask, "label"].to_numpy(), 1)
        _reject(lambda: training.baseline_control(context, replace(partition, frames=changed), root))
        _reject(lambda: training.baseline_control(replace(context, membership_sha256="f" * 64), partition, root))
        provenance = root / partition.fold / "stats/provenance.json"
        original_stats = provenance.read_bytes()
        data = json.loads(original_stats)
        data["fit_split"] = cv_folds.RETAINED_SPLIT
        provenance.write_text(json.dumps(data), encoding="utf-8")
        _reject(check)
        provenance.write_bytes(original_stats)
        metrics = json.loads(original["metrics.json"])
        labels = fold_windows(partition, cv_folds.DEVELOPMENT_SPLIT)["label"].to_numpy(dtype=np.int64).copy()
        labels[0] = (labels[0] + 1) % 10
        metrics["final"]["train"] = _split(labels)
        (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        _refresh_hashes(directory)
        _reject(check)
        restore()
        with tempfile.TemporaryDirectory() as backup:
            shutil.move(str(directory), str(Path(backup) / "run"))
            _reject(check)
            assert not directory.exists()
            shutil.move(str(Path(backup) / "run"), str(directory))
        assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        print(f"[PASS] controles nativos {protocol}: três dobras, integridade, proveniência, seed, estatísticas e janelas; sem retreino")


def _check_dual(protocol: str) -> None:
    with _workspace(protocol) as (root, context):
        partition = context.partitions[0]
        native = root / partition.fold / "runs/baseline_a/seed_42"
        alternate = training.BASELINE_ROOT / protocol / partition.fold / "baseline/seed_42"
        alternate.mkdir(parents=True)
        for name in ("config.yaml", "metrics.json", "checkpoint.pt"):
            shutil.copyfile(native / name, alternate / name)
        config = load_config(alternate / "config.yaml")
        save_config(replace(config, run_name="imbalance_a_baseline"), alternate / "config.yaml", force=True)
        metrics = json.loads((alternate / "metrics.json").read_text())
        metrics["run_name"] = "imbalance_a_baseline"
        (alternate / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        _, _, _, source = training._sources(protocol, root, partition.fold)
        experiment = {**cast(dict[str, Any], source["provenance"]), "arm": "A", "seed": 42,
                      "recipe": vars(RECIPES["baseline"]), "epochs": config.epochs}
        (alternate / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
        state = torch.load(alternate / "checkpoint.pt", map_location="cpu", weights_only=True)
        torch.save(state, alternate / "checkpoint.pt", _use_new_zipfile_serialization=False)
        _refresh_hashes(alternate)
        control = training.baseline_control(context, partition, root)
        assert control["source"] == "native" and control["equivalent_controls"][0]["source"] == "imbalance"
        assert control["artifact_sha256"]["checkpoint.pt"] != control["equivalent_controls"][0]["artifact_sha256"]["checkpoint.pt"]
        original_metrics = (alternate / "metrics.json").read_bytes()
        metrics = json.loads(original_metrics)
        labels = fold_windows(partition, cv_folds.RETAINED_SPLIT)["label"].to_numpy(dtype=np.int64)
        macro, classes = restricted_macro_f1(labels, np.zeros_like(labels))
        changed = {"macro_f1_restricted": macro, "f1_by_class": {str(key): value for key, value in classes.items()},
                   "support": support_by_name(labels, LE2I_LABEL_NAMES),
                   **classification_summary(labels, np.zeros_like(labels), LE2I_LABEL_NAMES)}
        metrics["final"].update(val=changed, test=changed)
        (alternate / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        _refresh_hashes(alternate)
        _reject(lambda: training.baseline_control(context, partition, root))
        (alternate / "metrics.json").write_bytes(original_metrics)
        _refresh_hashes(alternate)
        state[next(iter(state))] += 0.125
        torch.save(state, alternate / "checkpoint.pt")
        _refresh_hashes(alternate)
        _reject(lambda: training.baseline_control(context, partition, root))
        shutil.copyfile(native / "checkpoint.pt", alternate / "checkpoint.pt")
        _refresh_hashes(alternate)
        with tempfile.TemporaryDirectory() as backup:
            shutil.move(str(native), str(Path(backup) / "run"))
            assert training.baseline_control(context, partition, root)["source"] == "imbalance"
            shutil.move(str(Path(backup) / "run"), str(native))
        (native / "fold_run.json").write_text("{}", encoding="utf-8")
        _reject(lambda: training.baseline_control(context, partition, root))
        print(f"[PASS] controles {protocol}: equivalência científica, serializações distintas, fallback e discordância explícita")


def run_control_selftests() -> None:
    torch.set_num_threads(2)
    for protocol in ("cs", "cv"):
        _check_native(protocol)
        _check_dual(protocol)
