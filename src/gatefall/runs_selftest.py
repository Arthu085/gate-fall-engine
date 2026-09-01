"""Selftest sintético de segurança de paths e lifecycle de runs."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from gatefall.datasets.le2i import Le2iDatasetAdapter
from gatefall.eval.baseline_a_events import (
    EventEvaluationLock,
    _promote_event_outputs,
    _recover_event_publication,
    validate_event_metrics,
)
from gatefall.eval.alarm_protocol import (
    BASELINE_A_ALARM_PROTOCOL,
    save_alarm_protocol,
)
from gatefall.hashing import sha256_file
from gatefall.runs import REFERENCE_RUN_ROOT, validate_local_run_dir
from gatefall.train.artifacts import validate_training_run
from gatefall.train.config import BASELINE_A_CONFIG, save_config
from gatefall.train.metrics import RESTRICTED_CLASSES
from gatefall.train.tcn import TCNClassifier


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _valid_metrics(config_path: Path, checkpoint_path: Path) -> dict:
    split = {
        "macro_f1_restricted": 0.0,
        "f1_by_class": {str(index): 0.0 for index in RESTRICTED_CLASSES},
        "support": {str(index): 0 for index in range(BASELINE_A_CONFIG.num_classes)},
    }
    return {
        "run_name": BASELINE_A_CONFIG.run_name,
        "epochs_trained": BASELINE_A_CONFIG.epochs,
        "device": "cpu",
        "torch_version": torch.__version__,
        "history": [
            {
                "epoch": epoch,
                "train_loss": 0.0,
                "val_macro_f1_restricted": 0.0,
            }
            for epoch in range(1, BASELINE_A_CONFIG.epochs + 1)
        ],
        "final": {name: dict(split) for name in ("train", "val", "test")},
        "restricted_classes": RESTRICTED_CLASSES,
        "excluded_classes": [5, 6],
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _write_valid_config_and_checkpoint(run_dir: Path) -> None:
    save_config(BASELINE_A_CONFIG, run_dir / "config.yaml", force=True)
    model = TCNClassifier(
        input_dim=BASELINE_A_CONFIG.input_dim,
        channels=BASELINE_A_CONFIG.channels,
        kernel_size=BASELINE_A_CONFIG.kernel_size,
        dilations=BASELINE_A_CONFIG.dilations,
        dropout=BASELINE_A_CONFIG.dropout,
        num_classes=BASELINE_A_CONFIG.num_classes,
    )
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")


def check_reference_protected_outside_repository() -> bool:
    previous = Path.cwd()
    try:
        os.chdir(tempfile.gettempdir())
        absolute_rejected = False
        relative_rejected = False
        try:
            validate_local_run_dir(REFERENCE_RUN_ROOT / "le2i/baseline_a")
        except ValueError:
            absolute_rejected = True
        equivalent = Path(
            os.path.relpath(
                REFERENCE_RUN_ROOT / "le2i/baseline_a", Path.cwd()
            )
        )
        try:
            validate_local_run_dir(equivalent)
        except ValueError:
            relative_rejected = True
    finally:
        os.chdir(previous)
    return _check(
        "runs/reference: path absoluto e relativo equivalente são recusados com CWD=/tmp",
        absolute_rejected and relative_rejected,
    )


def check_relative_path_confinement(root: Path) -> bool:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    inside = raw_dir / "env/video.avi"
    inside.parent.mkdir()
    inside.write_bytes(b"inside")
    outside = root / "outside.avi"
    outside.write_bytes(b"outside")
    escape_link = raw_dir / "escape.avi"
    escape_link.symlink_to(outside)
    adapter = Le2iDatasetAdapter(raw_dir=raw_dir)

    valid = adapter.resolve_video_path("env/video.avi") == inside.resolve()
    rejected = 0
    for value in (str(outside.resolve()), "../outside.avi", "escape.avi"):
        try:
            adapter.resolve_video_path(value)
        except ValueError:
            rejected += 1
    return _check(
        "relative_path: aceita arquivo interno e rejeita absoluto, '..' e symlink escape",
        valid and rejected == 3,
    )


def check_invalid_training_artifacts_do_not_skip(root: Path) -> bool:
    invalid_metrics_dir = root / "invalid-metrics"
    invalid_metrics_dir.mkdir(parents=True)
    _write_valid_config_and_checkpoint(invalid_metrics_dir)
    (invalid_metrics_dir / "metrics.json").write_text("{}", encoding="utf-8")
    metrics_rejected = False
    try:
        validate_training_run(invalid_metrics_dir, BASELINE_A_CONFIG)
    except RuntimeError as exc:
        metrics_rejected = "metrics.json" in str(exc)

    invalid_checkpoint_dir = root / "invalid-checkpoint"
    invalid_checkpoint_dir.mkdir()
    save_config(
        BASELINE_A_CONFIG,
        invalid_checkpoint_dir / "config.yaml",
        force=True,
    )
    (invalid_checkpoint_dir / "checkpoint.pt").write_bytes(b"not-a-checkpoint")
    with (invalid_checkpoint_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(
            _valid_metrics(
                invalid_checkpoint_dir / "config.yaml",
                invalid_checkpoint_dir / "checkpoint.pt",
            ),
            stream,
        )
    checkpoint_rejected = False
    try:
        validate_training_run(invalid_checkpoint_dir, BASELINE_A_CONFIG)
    except RuntimeError as exc:
        checkpoint_rejected = "checkpoint.pt" in str(exc)

    event_schema_rejected = False
    try:
        validate_event_metrics(
            {},
            BASELINE_A_CONFIG,
            invalid_checkpoint_dir / "checkpoint.pt",
            invalid_checkpoint_dir / "alarm_protocol.yaml",
        )
    except ValueError:
        event_schema_rejected = True

    return _check(
        "skip sem force: JSONs com schema inválido e checkpoint inválido são recusados",
        metrics_rejected and checkpoint_rejected and event_schema_rejected,
    )


def check_second_promotion_failure_rolls_back(root: Path) -> bool:
    root.mkdir(parents=True)
    protocol_path = root / "alarm_protocol.yaml"
    metrics_path = root / "event_metrics.json"
    staged_protocol = root / "new-alarm.yaml"
    staged_metrics = root / "new-events.json"
    protocol_path.write_text("old-protocol", encoding="utf-8")
    metrics_path.write_text("old-metrics", encoding="utf-8")
    staged_protocol.write_text("new-protocol", encoding="utf-8")
    staged_metrics.write_text("new-metrics", encoding="utf-8")

    def fail_second_promotion(source: Path, target: Path) -> None:
        if source == staged_metrics and target == metrics_path:
            raise OSError("falha sintética na segunda promoção")
        os.replace(source, target)

    failed = False
    with EventEvaluationLock(root) as lock:
        try:
            _promote_event_outputs(
                staged_protocol,
                staged_metrics,
                protocol_path,
                metrics_path,
                lock,
                replace_file=fail_second_promotion,
            )
        except RuntimeError:
            failed = True
    preserved = (
        protocol_path.read_text(encoding="utf-8") == "old-protocol"
        and metrics_path.read_text(encoding="utf-8") == "old-metrics"
    )
    return _check(
        "publicação de eventos: falha na segunda promoção restaura o par anterior",
        failed and preserved,
    )


class _SimulatedCrash(BaseException):
    pass


def check_event_writer_lock(root: Path) -> bool:
    root.mkdir(parents=True)
    journal = root / ".event-evaluation-transaction.json"
    first = EventEvaluationLock(root)
    first.__enter__()
    simultaneous_rejected = False
    try:
        try:
            with EventEvaluationLock(root):
                pass
        except RuntimeError as exc:
            simultaneous_rejected = "já está em execução" in str(exc)
        untouched = not journal.exists() and not list(root.glob(".*.backup-*"))
    finally:
        first.close()

    normal_release = False
    with EventEvaluationLock(root):
        normal_release = True

    child_code = (
        "import fcntl, pathlib, sys, time; "
        "stream=(pathlib.Path(sys.argv[1])/'.event-evaluation.lock').open('a+'); "
        "fcntl.flock(stream.fileno(), fcntl.LOCK_EX); "
        "print('locked', flush=True); time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(root)],
        stdout=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None or process.stdout.readline().strip() != "locked":
        process.kill()
        process.wait()
        return _check("lock de avaliação: processo auxiliar adquiriu lock", False)
    child_lock_observed = False
    try:
        try:
            with EventEvaluationLock(root):
                pass
        except RuntimeError:
            child_lock_observed = True
    finally:
        process.kill()
        process.wait()
    crash_release = False
    with EventEvaluationLock(root):
        crash_release = True

    unheld = EventEvaluationLock(root)
    recovery_guarded = False
    publication_guarded = False
    try:
        _recover_event_publication(root, unheld)
    except RuntimeError as exc:
        recovery_guarded = "lock exclusivo" in str(exc)
    staged_protocol = root / "guard-new-alarm.yaml"
    staged_metrics = root / "guard-new-events.json"
    staged_protocol.write_text("new", encoding="utf-8")
    staged_metrics.write_text("new", encoding="utf-8")
    try:
        _promote_event_outputs(
            staged_protocol,
            staged_metrics,
            root / "alarm_protocol.yaml",
            root / "event_metrics.json",
            unheld,
            preserve_previous=False,
        )
    except RuntimeError as exc:
        publication_guarded = "lock exclusivo" in str(exc)
    untouched = untouched and not journal.exists() and not list(
        root.glob(".*.backup-*")
    )
    return _check(
        "lock de avaliação: exclusão, liberações e guardas de lifecycle",
        simultaneous_rejected
        and normal_release
        and child_lock_observed
        and crash_release
        and recovery_guarded
        and publication_guarded
        and untouched,
    )


def check_crash_boundaries_recover(root: Path) -> bool:
    boundaries = (
        "journal",
        "backup_protocol",
        "backup_metrics",
        "publish_protocol",
        "publish_metrics",
        "cleanup_staged_protocol",
        "cleanup_staged_metrics",
        "cleanup_protocol_backup",
        "cleanup_metrics_backup",
        "cleanup_journal",
        "cleanup",
    )
    all_ok = True
    for boundary in boundaries:
        case = root / boundary
        case.mkdir(parents=True)
        protocol_path = case / "alarm_protocol.yaml"
        metrics_path = case / "event_metrics.json"
        staged_protocol = case / "new-alarm.yaml"
        staged_metrics = case / "new-events.json"
        protocol_path.write_text("old-protocol", encoding="utf-8")
        metrics_path.write_text("old-metrics", encoding="utf-8")
        staged_protocol.write_text("new-protocol", encoding="utf-8")
        staged_metrics.write_text("new-metrics", encoding="utf-8")

        def crash(step: str) -> None:
            if step == boundary:
                raise _SimulatedCrash

        try:
            with EventEvaluationLock(case) as lock:
                _promote_event_outputs(
                    staged_protocol,
                    staged_metrics,
                    protocol_path,
                    metrics_path,
                    lock,
                    after_step=crash,
                )
        except _SimulatedCrash:
            pass
        with EventEvaluationLock(case) as lock:
            outcome = _recover_event_publication(case, lock)
        expects_new = boundary in {
            "publish_metrics",
            "cleanup_staged_protocol",
            "cleanup_staged_metrics",
            "cleanup_protocol_backup",
            "cleanup_metrics_backup",
            "cleanup_journal",
            "cleanup",
        }
        expected_protocol = "new-protocol" if expects_new else "old-protocol"
        expected_metrics = "new-metrics" if expects_new else "old-metrics"
        all_ok = all_ok and (
            protocol_path.read_text(encoding="utf-8") == expected_protocol
            and metrics_path.read_text(encoding="utf-8") == expected_metrics
            and outcome in ({None, "finalized"} if expects_new else {"restored"})
            and not list(case.glob(".*.backup-*"))
            and not (case / ".event-evaluation-transaction.json").exists()
        )

    recovery_case = root / "recovery-interrupted"
    recovery_case.mkdir(parents=True)
    protocol_path = recovery_case / "alarm_protocol.yaml"
    metrics_path = recovery_case / "event_metrics.json"
    staged_protocol = recovery_case / "new-alarm.yaml"
    staged_metrics = recovery_case / "new-events.json"
    protocol_path.write_text("old-protocol", encoding="utf-8")
    metrics_path.write_text("old-metrics", encoding="utf-8")
    staged_protocol.write_text("new-protocol", encoding="utf-8")
    staged_metrics.write_text("new-metrics", encoding="utf-8")

    def crash_forward(step: str) -> None:
        if step == "publish_protocol":
            raise _SimulatedCrash

    try:
        with EventEvaluationLock(recovery_case) as lock:
            _promote_event_outputs(
                staged_protocol,
                staged_metrics,
                protocol_path,
                metrics_path,
                lock,
                after_step=crash_forward,
            )
    except _SimulatedCrash:
        pass

    def crash_recovery(step: str) -> None:
        if step == "restore_protocol":
            raise _SimulatedCrash

    try:
        with EventEvaluationLock(recovery_case) as lock:
            _recover_event_publication(
                recovery_case, lock, after_step=crash_recovery
            )
    except _SimulatedCrash:
        pass
    with EventEvaluationLock(recovery_case) as lock:
        outcome = _recover_event_publication(recovery_case, lock)
    all_ok = all_ok and (
        outcome == "restored"
        and protocol_path.read_text(encoding="utf-8") == "old-protocol"
        and metrics_path.read_text(encoding="utf-8") == "old-metrics"
        and not list(recovery_case.glob(".*.backup-*"))
    )
    return _check(
        "journal: crashes em todas as fronteiras restauram o antigo ou finalizam o novo",
        all_ok,
    )


def check_hash_links_reject_compatible_swap(root: Path) -> bool:
    root.mkdir(parents=True)
    config_path = root / "config.yaml"
    checkpoint_path = root / "checkpoint.pt"
    metrics_path = root / "metrics.json"
    protocol_path = root / "alarm_protocol.yaml"
    save_config(BASELINE_A_CONFIG, config_path, force=True)
    torch.manual_seed(1)
    first = TCNClassifier(
        input_dim=BASELINE_A_CONFIG.input_dim,
        channels=BASELINE_A_CONFIG.channels,
        kernel_size=BASELINE_A_CONFIG.kernel_size,
        dilations=BASELINE_A_CONFIG.dilations,
        dropout=BASELINE_A_CONFIG.dropout,
        num_classes=BASELINE_A_CONFIG.num_classes,
    )
    torch.save(first.state_dict(), checkpoint_path)
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(_valid_metrics(config_path, checkpoint_path), stream)
    checkpoint_bytes = checkpoint_path.read_bytes()
    metrics_bytes = metrics_path.read_bytes()
    validate_training_run(root, BASELINE_A_CONFIG)
    save_alarm_protocol(BASELINE_A_ALARM_PROTOCOL, protocol_path, force=True)
    event_report = {
        "run_name": BASELINE_A_CONFIG.run_name,
        "checkpoint_path": str(checkpoint_path),
        "alarm_protocol_path": str(protocol_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_metrics_sha256": sha256_file(metrics_path),
        "alarm_protocol_sha256": sha256_file(protocol_path),
    }

    torch.manual_seed(2)
    second = TCNClassifier(
        input_dim=BASELINE_A_CONFIG.input_dim,
        channels=BASELINE_A_CONFIG.channels,
        kernel_size=BASELINE_A_CONFIG.kernel_size,
        dilations=BASELINE_A_CONFIG.dilations,
        dropout=BASELINE_A_CONFIG.dropout,
        num_classes=BASELINE_A_CONFIG.num_classes,
    )
    torch.save(second.state_dict(), checkpoint_path)
    training_rejected = False
    event_rejected = False
    try:
        validate_training_run(root, BASELINE_A_CONFIG)
    except RuntimeError as exc:
        training_rejected = "checkpoint_sha256" in str(exc)
    try:
        validate_event_metrics(
            event_report,
            BASELINE_A_CONFIG,
            checkpoint_path,
            protocol_path,
            training_metrics_path=metrics_path,
            require_hashes=True,
        )
    except ValueError as exc:
        event_rejected = "checkpoint_sha256" in str(exc)

    checkpoint_path.write_bytes(checkpoint_bytes)
    metrics_path.write_bytes(metrics_bytes + b"\n")
    metrics_link_rejected = False
    try:
        validate_event_metrics(
            event_report,
            BASELINE_A_CONFIG,
            checkpoint_path,
            protocol_path,
            training_metrics_path=metrics_path,
            require_hashes=True,
        )
    except ValueError as exc:
        metrics_link_rejected = "training_metrics_sha256" in str(exc)

    metrics_path.write_bytes(metrics_bytes)
    protocol_bytes = protocol_path.read_bytes()
    protocol_path.write_bytes(protocol_bytes + b"\n")
    protocol_link_rejected = False
    try:
        validate_event_metrics(
            event_report,
            BASELINE_A_CONFIG,
            checkpoint_path,
            protocol_path,
            training_metrics_path=metrics_path,
            require_hashes=True,
        )
    except ValueError as exc:
        protocol_link_rejected = "alarm_protocol_sha256" in str(exc)
    return _check(
        "vínculo SHA-256: troca compatível e alterações em metrics/protocolo invalidam vínculos",
        training_rejected
        and event_rejected
        and metrics_link_rejected
        and protocol_link_rejected,
    )


def run_selftest() -> None:
    checks = [check_reference_protected_outside_repository()]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checks.extend(
            [
                check_relative_path_confinement(root / "paths"),
                check_invalid_training_artifacts_do_not_skip(root / "artifacts"),
                check_second_promotion_failure_rolls_back(root / "promotion"),
                check_event_writer_lock(root / "lock"),
                check_crash_boundaries_recover(root / "crashes"),
                check_hash_links_reject_compatible_swap(root / "hash-links"),
            ]
        )
    if not all(checks):
        print("\nruns selftest FALHOU", file=sys.stderr)
        raise SystemExit(1)
    print("\nruns selftest OK: todas as checagens passaram")


if __name__ == "__main__":
    run_selftest()
