"""Avaliação de eventos da arma A: protocolo de alarme sobre o checkpoint treinado."""

import argparse
import fcntl
import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import IO, cast

import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import build_window_index
from gatefall.datasets import get_dataset
from gatefall.eval.alarm_protocol import (
    BASELINE_A_ALARM_PROTOCOL,
    load_alarm_protocol,
    save_alarm_protocol,
)
from gatefall.eval.events import extract_label_segments, split_event_report
from gatefall.eval.events_selftest import run_events_selftest
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization,
    load_stats,
    validate_stats_layout,
)
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import validate_local_run_dir
from gatefall.hashing import sha256_file
from gatefall.train.artifacts import load_compatible_checkpoint, validate_training_run
from gatefall.train.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.tcn import TCNClassifier

RUN_DIR = Path("runs/local/le2i/baseline_a")
EVENT_LOCK_FILE = ".event-evaluation.lock"


class EventEvaluationLock:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self._stream: IO[str] | None = None

    @property
    def held(self) -> bool:
        return self._stream is not None and not self._stream.closed

    def __enter__(self) -> "EventEvaluationLock":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stream = (self.run_dir / EVENT_LOCK_FILE).open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise RuntimeError(
                f"avaliação de eventos já está em execução para {self.run_dir}"
            ) from exc
        self._stream = stream
        return self

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def _require_event_lock(run_dir: Path, lock: EventEvaluationLock) -> None:
    if not lock.held or lock.run_dir != run_dir.resolve():
        raise RuntimeError(
            f"operação de lifecycle exige lock exclusivo ativo para {run_dir}"
        )


def _load_model(
    config: TrainConfig, checkpoint_path: Path, device: str
) -> TCNClassifier:
    model = load_compatible_checkpoint(checkpoint_path, config).to(device)
    model.eval()
    return model


EVENT_SPLIT_FIELDS = {
    "usable_windows",
    "total_windows",
    "labeled_windows",
    "total_video_time_hours",
    "labeled_time_hours",
    "n_fall_events",
    "n_detected_events",
    "n_missed_events",
    "sensitivity",
    "n_alarms_total",
    "n_false_alarms",
    "n_pre_fall_false_alarms",
    "false_alarms_per_hour",
    "false_alarms_per_hour_labeled_time",
    "window_binary_sensitivity",
    "window_binary_specificity",
    "latency_seconds",
}
EVENT_COUNT_FIELDS = {
    "usable_windows",
    "total_windows",
    "labeled_windows",
    "n_fall_events",
    "n_detected_events",
    "n_missed_events",
    "n_alarms_total",
    "n_false_alarms",
    "n_pre_fall_false_alarms",
}
EVENT_RATE_FIELDS = EVENT_SPLIT_FIELDS - EVENT_COUNT_FIELDS - {"latency_seconds"}


def validate_event_metrics(
    data: object,
    config: TrainConfig,
    checkpoint_path: Path,
    alarm_protocol_path: Path,
    training_metrics_path: Path | None = None,
    protocol_file_path: Path | None = None,
    require_hashes: bool = False,
) -> None:
    if not isinstance(data, Mapping):
        raise ValueError("event_metrics.json deve ser um objeto")
    if data.get("run_name") != config.run_name:
        raise ValueError("event_metrics.json: run_name diverge de config.yaml")
    if data.get("checkpoint_path") != str(checkpoint_path):
        raise ValueError("event_metrics.json: checkpoint_path incompatível")
    if data.get("alarm_protocol_path") != str(alarm_protocol_path):
        raise ValueError("event_metrics.json: alarm_protocol_path incompatível")
    if require_hashes:
        if training_metrics_path is None:
            raise ValueError("training_metrics_path é obrigatório para run local")
        expected_hashes = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_metrics_sha256": sha256_file(training_metrics_path),
            "alarm_protocol_sha256": sha256_file(
                protocol_file_path or alarm_protocol_path
            ),
        }
        for field, expected in expected_hashes.items():
            if data.get(field) != expected:
                raise ValueError(f"event_metrics.json: {field} incompatível")
    splits = data.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"val", "test"}:
        raise ValueError("event_metrics.json: splits deve conter val e test")
    for split in ("val", "test"):
        split_data = splits[split]
        if not isinstance(split_data, Mapping):
            raise ValueError(f"event_metrics.json: split {split} deve ser objeto")
        missing = EVENT_SPLIT_FIELDS - set(split_data)
        if missing:
            raise ValueError(
                f"event_metrics.json: split {split} sem campos: {sorted(missing)}"
            )
        if any(
            not isinstance(split_data[field], int) or split_data[field] < 0
            for field in EVENT_COUNT_FIELDS
        ):
            raise ValueError(
                f"event_metrics.json: contagens de {split} devem ser inteiros não negativos"
            )
        if any(
            not isinstance(split_data[field], (int, float))
            for field in EVENT_RATE_FIELDS
        ):
            raise ValueError(
                f"event_metrics.json: taxas de {split} devem ser numéricas"
            )
        if split_data["usable_windows"] != split_data["total_windows"]:
            raise ValueError(
                f"event_metrics.json: usable_windows != total_windows em {split}"
            )
        if split_data["labeled_windows"] > split_data["total_windows"]:
            raise ValueError(
                f"event_metrics.json: labeled_windows excede total_windows em {split}"
            )
        if (
            split_data["n_detected_events"] + split_data["n_missed_events"]
            != split_data["n_fall_events"]
        ):
            raise ValueError(
                f"event_metrics.json: contagem de eventos inconsistente em {split}"
            )
        latency = split_data["latency_seconds"]
        if not isinstance(latency, Mapping):
            raise ValueError(
                f"event_metrics.json: latency_seconds de {split} deve ser objeto"
            )
        per_event = latency.get("per_event")
        if not isinstance(per_event, list) or len(per_event) != split_data[
            "n_detected_events"
        ]:
            raise ValueError(
                f"event_metrics.json: latências por evento incompatíveis em {split}"
            )
        if any(not isinstance(value, (int, float)) for value in per_event):
            raise ValueError(
                f"event_metrics.json: latências de {split} devem ser numéricas"
            )
        for field in ("mean", "median"):
            value = latency.get(field)
            valid = isinstance(value, (int, float)) if per_event else value is None
            if not valid:
                raise ValueError(
                    f"event_metrics.json: latency_seconds.{field} inválido em {split}"
                )


EVENT_TRANSACTION_FILE = ".event-evaluation-transaction.json"


def _file_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _transaction_path(run_dir: Path, name: object) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise RuntimeError("journal de avaliação contém path inválido")
    return run_dir / name


def _write_event_transaction(run_dir: Path, transaction: dict[str, object]) -> None:
    journal = run_dir / EVENT_TRANSACTION_FILE
    temporary = run_dir / f"{EVENT_TRANSACTION_FILE}.tmp-{uuid.uuid4().hex}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(transaction, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, journal)


def _recover_event_publication(
    run_dir: Path,
    lock: EventEvaluationLock,
    replace_file: Callable[[Path, Path], None] = os.replace,
    after_step: Callable[[str], None] | None = None,
) -> str | None:
    _require_event_lock(run_dir, lock)
    journal = run_dir / EVENT_TRANSACTION_FILE
    if not journal.exists():
        orphaned = list(run_dir.glob(".*.backup-*")) if run_dir.exists() else []
        if orphaned:
            raise RuntimeError(
                "backups de avaliação órfãos sem journal: "
                + ", ".join(path.name for path in orphaned)
            )
        return None
    try:
        with journal.open(encoding="utf-8") as stream:
            transaction = json.load(stream)
        if not isinstance(transaction, dict) or transaction.get("version") != 1:
            raise ValueError("versão/formato inválido")
        protocol_path = _transaction_path(run_dir, transaction["protocol_path"])
        metrics_path = _transaction_path(run_dir, transaction["metrics_path"])
        staged_protocol = _transaction_path(
            run_dir, transaction["staged_protocol"]
        )
        staged_metrics = _transaction_path(run_dir, transaction["staged_metrics"])
        protocol_backup = _transaction_path(
            run_dir, transaction["protocol_backup"]
        )
        metrics_backup = _transaction_path(run_dir, transaction["metrics_backup"])
        new_protocol_hash = str(transaction["new_protocol_sha256"])
        new_metrics_hash = str(transaction["new_metrics_sha256"])
        preserve_previous = transaction.get("preserve_previous") is True
        old_protocol_hash = transaction.get("old_protocol_sha256")
        old_metrics_hash = transaction.get("old_metrics_sha256")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"journal de avaliação inválido: {exc}") from exc

    new_pair_complete = (
        _file_hash(protocol_path) == new_protocol_hash
        and _file_hash(metrics_path) == new_metrics_hash
    )
    if new_pair_complete:
        outcome = "finalized"
    elif preserve_previous:
        for name, target, backup, expected_hash in (
            (
                "restore_protocol",
                protocol_path,
                protocol_backup,
                old_protocol_hash,
            ),
            ("restore_metrics", metrics_path, metrics_backup, old_metrics_hash),
        ):
            if not isinstance(expected_hash, str):
                raise RuntimeError("journal não descreve o par anterior completo")
            if _file_hash(backup) == expected_hash:
                replace_file(backup, target)
            elif _file_hash(target) != expected_hash:
                raise RuntimeError(
                    f"estado ambíguo: não é possível restaurar {target.name}"
                )
            if after_step is not None:
                after_step(name)
        outcome = "restored"
    else:
        protocol_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        outcome = "cleared"

    for name, path in (
        ("cleanup_staged_protocol", staged_protocol),
        ("cleanup_staged_metrics", staged_metrics),
        ("cleanup_protocol_backup", protocol_backup),
        ("cleanup_metrics_backup", metrics_backup),
    ):
        path.unlink(missing_ok=True)
        if after_step is not None:
            after_step(name)
    journal.unlink()
    if after_step is not None:
        after_step("cleanup_journal")
    return outcome


def _promote_event_outputs(
    staged_protocol: Path,
    staged_metrics: Path,
    protocol_path: Path,
    metrics_path: Path,
    lock: EventEvaluationLock,
    replace_file: Callable[[Path, Path], None] = os.replace,
    after_step: Callable[[str], None] | None = None,
    preserve_previous: bool = True,
) -> None:
    token = uuid.uuid4().hex
    run_dir = protocol_path.parent
    _require_event_lock(run_dir, lock)
    if metrics_path.parent != run_dir or any(
        path.parent != run_dir for path in (staged_protocol, staged_metrics)
    ):
        raise ValueError("todos os artefatos da avaliação devem compartilhar run_dir")
    protocol_backup = run_dir / f".{protocol_path.name}.backup-{token}"
    metrics_backup = run_dir / f".{metrics_path.name}.backup-{token}"
    transaction: dict[str, object] = {
        "version": 1,
        "protocol_path": protocol_path.name,
        "metrics_path": metrics_path.name,
        "staged_protocol": staged_protocol.name,
        "staged_metrics": staged_metrics.name,
        "protocol_backup": protocol_backup.name,
        "metrics_backup": metrics_backup.name,
        "new_protocol_sha256": sha256_file(staged_protocol),
        "new_metrics_sha256": sha256_file(staged_metrics),
        "old_protocol_sha256": _file_hash(protocol_path),
        "old_metrics_sha256": _file_hash(metrics_path),
        "preserve_previous": preserve_previous,
    }
    _write_event_transaction(run_dir, transaction)
    if after_step is not None:
        after_step("journal")
    try:
        for name, target, backup in (
            ("backup_protocol", protocol_path, protocol_backup),
            ("backup_metrics", metrics_path, metrics_backup),
        ):
            if target.exists():
                replace_file(target, backup)
            if after_step is not None:
                after_step(name)
        replace_file(staged_protocol, protocol_path)
        if after_step is not None:
            after_step("publish_protocol")
        replace_file(staged_metrics, metrics_path)
        if after_step is not None:
            after_step("publish_metrics")
        if (
            _file_hash(protocol_path) != transaction["new_protocol_sha256"]
            or _file_hash(metrics_path) != transaction["new_metrics_sha256"]
        ):
            raise RuntimeError("par publicado diverge do staging validado")
        _recover_event_publication(
            run_dir,
            lock,
            replace_file=replace_file,
            after_step=after_step,
        )
        if after_step is not None:
            after_step("cleanup")
    except Exception as exc:
        outcome = _recover_event_publication(
            run_dir, lock, replace_file=replace_file
        )
        raise RuntimeError(
            f"falha ao promover avaliação; recovery={outcome}"
        ) from exc


@torch.no_grad()
def _predict_with_identity(
    model: TCNClassifier,
    source: PoseWindowDataset,
    stats: StandardizationStats,
    device: str,
    batch_size: int,
) -> tuple[list[str], list[int], list[int], list[int]]:
    video_ids: list[str] = []
    k_ends: list[int] = []
    true_labels: list[int] = []
    pred_labels: list[int] = []

    batch_windows: list[np.ndarray] = []
    batch_labels: list[int] = []
    batch_identity: list[tuple[str, int]] = []

    def flush() -> None:
        if not batch_windows:
            return
        stacked = np.stack(batch_windows, axis=0)
        standardized = apply_standardization(stacked, stats)
        x = torch.from_numpy(standardized).to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        for (video_id, k_end), label, pred in zip(batch_identity, batch_labels, preds):
            video_ids.append(video_id)
            k_ends.append(k_end)
            true_labels.append(label)
            pred_labels.append(int(pred))

        batch_windows.clear()
        batch_labels.clear()
        batch_identity.clear()

    for i in range(len(source)):
        window, label, (video_id, k_end) = source[i]
        batch_windows.append(window)
        batch_labels.append(label)
        batch_identity.append((video_id, k_end))
        if len(batch_windows) == batch_size:
            flush()
    flush()

    return video_ids, k_ends, true_labels, pred_labels


def _n_fall_segments_in_annotation(frames: pd.DataFrame, split: str) -> int:
    split_frames = cast(
        pd.DataFrame, frames[frames["split"] == split]
    ).sort_values(["video_id", "frame_index"])
    total_segments = 0
    for _video_id, group in split_frames.groupby("video_id", sort=False):
        frame_indices = group["frame_index"].to_numpy()
        labels = group["label"].to_numpy()
        total_segments += len(
            extract_label_segments(frame_indices, labels, BASELINE_A_ALARM_PROTOCOL.fall_label)
        )
    return total_segments


def _run_evaluate_locked(
    force: bool,
    dataset_name: str,
    run_dir: Path,
    lock: EventEvaluationLock,
) -> None:
    _require_event_lock(run_dir, lock)
    adapter = get_dataset(dataset_name)
    checkpoint_path = run_dir / "checkpoint.pt"
    alarm_protocol_path = run_dir / "alarm_protocol.yaml"
    event_metrics_path = run_dir / "event_metrics.json"
    recovery = _recover_event_publication(run_dir, lock)
    if recovery is not None:
        print(f"recovery de avaliação concluído: {recovery}")
    expected_config = replace(
        BASELINE_A_CONFIG,
        standardization_stats_path=str(adapter.pose_stats_path),
        standardization_stats_sha256=sha256_file(adapter.pose_stats_path),
    )
    try:
        config = validate_training_run(run_dir, expected_config=expected_config)
    except RuntimeError as exc:
        raise RuntimeError(
            f"run de treino inválido em {run_dir}: {exc}"
        ) from exc
    if config.eval_stride != EVAL_STRIDE:
        raise ValueError(
            f"config.eval_stride ({config.eval_stride}) diverge de "
            f"EVAL_STRIDE ({EVAL_STRIDE})"
        )
    event_outputs = (alarm_protocol_path, event_metrics_path)
    present_outputs = [path for path in event_outputs if path.is_file()]
    previous_pair_valid = False
    if len(present_outputs) == len(event_outputs):
        try:
            protocol = load_alarm_protocol(alarm_protocol_path)
            if protocol != BASELINE_A_ALARM_PROTOCOL:
                raise ValueError("alarm_protocol.yaml incompatível com o braço A")
            with event_metrics_path.open(encoding="utf-8") as stream:
                existing_report = json.load(stream)
            validate_event_metrics(
                existing_report,
                config,
                checkpoint_path,
                alarm_protocol_path,
                training_metrics_path=run_dir / "metrics.json",
                require_hashes=True,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            if not force:
                raise RuntimeError(
                    f"avaliação de eventos inconsistente ({exc}); use --force para reconstruir"
                ) from exc
        else:
            previous_pair_valid = True
            if not force:
                print(f"skip {event_metrics_path} (avaliação completa e íntegra)")
                return
    elif present_outputs and not force:
        missing_outputs = [str(path) for path in event_outputs if not path.is_file()]
        raise RuntimeError(
            "avaliação de eventos parcial; artefatos ausentes: "
            + ", ".join(missing_outputs)
            + "; use --force para reconstruir"
        )
    stats = load_stats(adapter.pose_stats_path)
    validate_stats_layout(stats)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(config, checkpoint_path, device)

    frames = adapter.load_frames()

    splits: dict[str, dict] = {}
    for split in ("val", "test"):
        source = PoseWindowDataset(
            frames,
            split,
            EVAL_STRIDE,
            lambda video_id: build_pose_features(
                video_id, pose_root=adapter.pose_root
            )[0],
            drop_ignored=False,
        )
        video_ids, k_ends, true_labels, pred_labels = _predict_with_identity(
            model, source, stats, device, batch_size=config.batch_size
        )
        usable_windows = len(source)
        total_windows = len(
            build_window_index(
                cast(pd.DataFrame, frames[frames["split"] == split]),
                stride=EVAL_STRIDE,
                drop_ignored=False,
            )
        )
        if usable_windows != total_windows:
            raise RuntimeError(
                f"split={split!r}: usable_windows ({usable_windows}) != "
                f"total_windows ({total_windows}) apesar de drop_ignored=False"
            )
        labeled_windows = len(
            build_window_index(
                cast(pd.DataFrame, frames[frames["split"] == split]),
                stride=EVAL_STRIDE,
                drop_ignored=True,
            )
        )
        split_report = split_event_report(
            video_ids,
            k_ends,
            true_labels,
            pred_labels,
            BASELINE_A_ALARM_PROTOCOL,
            usable_windows,
            total_windows,
            labeled_windows,
        )

        n_fall_segments_annotation = _n_fall_segments_in_annotation(frames, split)
        if split_report["n_fall_events"] != n_fall_segments_annotation:
            raise ValueError(
                f"split={split!r}: n_fall_events extraído das janelas usáveis "
                f"({split_report['n_fall_events']}) diverge da contagem de "
                "segmentos fall na anotação bruta "
                f"({n_fall_segments_annotation}) — possível janela IGNORE_LABEL "
                "descartada dentro de um run fall, dividindo um evento real em "
                "dois"
            )

        splits[split] = split_report

    report = {
        "run_name": config.run_name,
        "checkpoint_path": str(checkpoint_path),
        "alarm_protocol_path": str(alarm_protocol_path),
        "splits": splits,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    protocol_tmp = run_dir / f".alarm_protocol.pending-{token}.yaml"
    metrics_tmp = run_dir / f".event_metrics.pending-{token}.json"
    save_alarm_protocol(BASELINE_A_ALARM_PROTOCOL, protocol_tmp, force=True)
    report["checkpoint_sha256"] = sha256_file(checkpoint_path)
    report["training_metrics_sha256"] = sha256_file(run_dir / "metrics.json")
    report["alarm_protocol_sha256"] = sha256_file(protocol_tmp)
    with metrics_tmp.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with metrics_tmp.open(encoding="utf-8") as f:
        staged_report = json.load(f)
    staged_protocol = load_alarm_protocol(protocol_tmp)
    if staged_protocol != BASELINE_A_ALARM_PROTOCOL:
        raise RuntimeError("staging de alarm_protocol.yaml divergiu do protocolo")
    validate_event_metrics(
        staged_report,
        config,
        checkpoint_path,
        alarm_protocol_path,
        training_metrics_path=run_dir / "metrics.json",
        protocol_file_path=protocol_tmp,
        require_hashes=True,
    )
    _promote_event_outputs(
        protocol_tmp,
        metrics_tmp,
        alarm_protocol_path,
        event_metrics_path,
        lock,
        preserve_previous=previous_pair_valid,
    )

    print(f"{event_metrics_path}: métricas de eventos gravadas (run_name={config.run_name})")


def run_evaluate(
    force: bool, dataset_name: str = "le2i", run_dir: Path = RUN_DIR
) -> None:
    validate_local_run_dir(run_dir)
    with EventEvaluationLock(run_dir) as lock:
        _run_evaluate_locked(force, dataset_name, run_dir, lock)


def run_selftest() -> None:
    if not run_events_selftest():
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Roda o protocolo de alarme sobre o checkpoint treinado e grava event_metrics.json",
    )
    evaluate_parser.add_argument(
        "--force", action="store_true", help="Sobrescreve o event_metrics.json já existente"
    )
    evaluate_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    evaluate_parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas do protocolo de eventos")

    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(
            force=args.force, dataset_name=args.dataset, run_dir=args.run_dir
        )
    elif args.command == "selftest":
        run_selftest()


if __name__ == "__main__":
    main()
