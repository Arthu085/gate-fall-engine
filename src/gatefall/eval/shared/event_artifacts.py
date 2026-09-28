"""Lifecycle compartilhado dos artefatos de avaliação de eventos."""

import fcntl
import json
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Protocol

from gatefall.hashing import sha256_file


class EventRunConfig(Protocol):
    run_name: str


EVENT_LOCK_FILE = ".event-evaluation.lock"
EVENT_TRANSACTION_FILE = ".event-evaluation-transaction.json"

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
    "n_events_detected_in_fall",
    "n_events_detected_in_fall_or_fallen",
    "fall_sensitivity",
    "fall_or_fallen_sensitivity",
    "detected_events_alarm_within_fall_rate",
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
    "n_events_detected_in_fall",
    "n_events_detected_in_fall_or_fallen",
    "n_alarms_total",
    "n_false_alarms",
    "n_pre_fall_false_alarms",
}
EVENT_RATE_FIELDS = EVENT_SPLIT_FIELDS - EVENT_COUNT_FIELDS - {"latency_seconds"}


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


def validate_event_metrics(
    data: object,
    config: EventRunConfig,
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
        if not (
            0
            <= split_data["n_events_detected_in_fall"]
            <= split_data["n_events_detected_in_fall_or_fallen"]
            <= split_data["n_detected_events"]
        ):
            raise ValueError(
                f"event_metrics.json: n_events_detected_in_fall/"
                f"n_events_detected_in_fall_or_fallen inconsistentes com "
                f"n_detected_events em {split}"
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
        staged_protocol = _transaction_path(run_dir, transaction["staged_protocol"])
        staged_metrics = _transaction_path(run_dir, transaction["staged_metrics"])
        protocol_backup = _transaction_path(run_dir, transaction["protocol_backup"])
        metrics_backup = _transaction_path(run_dir, transaction["metrics_backup"])
        new_protocol_hash = str(transaction["new_protocol_sha256"])
        new_metrics_hash = str(transaction["new_metrics_sha256"])
        preserve_previous = transaction.get("preserve_previous") is True
        old_protocol_hash = transaction.get("old_protocol_sha256")
        old_metrics_hash = transaction.get("old_metrics_sha256")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"journal de avaliação inválido: {exc}") from exc

    if (
        _file_hash(protocol_path) == new_protocol_hash
        and _file_hash(metrics_path) == new_metrics_hash
    ):
        outcome = "finalized"
    elif preserve_previous:
        for name, target, backup, expected_hash in (
            ("restore_protocol", protocol_path, protocol_backup, old_protocol_hash),
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
            run_dir, lock, replace_file=replace_file, after_step=after_step
        )
        if after_step is not None:
            after_step("cleanup")
    except Exception as exc:
        outcome = _recover_event_publication(
            run_dir, lock, replace_file=replace_file
        )
        raise RuntimeError(f"falha ao promover avaliação; recovery={outcome}") from exc
