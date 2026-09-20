"""Selftest sintético da avaliação de eventos da arma B1."""

import ast
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from gatefall.config import TRAIN_STRIDE
from gatefall.eval.alarm_protocol import (
    AlarmProtocol,
    BASELINE_A_ALARM_PROTOCOL,
    save_alarm_protocol,
)
from gatefall.eval.event_artifacts import (
    EVENT_COUNT_FIELDS,
    EVENT_LOCK_FILE,
    EVENT_RATE_FIELDS,
    EVENT_SPLIT_FIELDS,
    EVENT_TRANSACTION_FILE,
    EventEvaluationLock,
    _promote_event_outputs,
    validate_event_metrics,
)
from gatefall.features.dinov3_standardization import Dinov3StandardizationStats
from gatefall.features.standardization import (
    StandardizationStats,
    excluded_dimension_mask,
)
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import POSE_FEATURE_DIM, feature_names
from gatefall.runs import default_run_dir, default_run_dir_for_arm
from gatefall.train.b0_config import B0_FUSION_CONFIG
from gatefall.train.b1_artifacts import validate_b1_training_run
from gatefall.train.b1_config import B1_ADAPTIVE_GATE_CONFIG
from gatefall.train.b1_model import GATE_INPUT_DIM
from gatefall.train.b1_run import repository_anchored_run_dir
from gatefall.train.config import BASELINE_A_CONFIG

_VISUAL_DIM = 1536
_QUALITY_VALUES = (0.25, 0.75)
_EVENT_REPORT_TOP_LEVEL_FIELDS = {
    "run_name",
    "checkpoint_path",
    "alarm_protocol_path",
    "splits",
    "checkpoint_sha256",
    "training_metrics_sha256",
    "alarm_protocol_sha256",
}


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _pose_stats(mean: float = 0.0, std: float = 1.0) -> StandardizationStats:
    names = feature_names()
    excluded = excluded_dimension_mask(names)
    return StandardizationStats(
        source="pose",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1,
        feature_dim=POSE_FEATURE_DIM,
        feature_names=names,
        excluded_mask=excluded.tolist(),
        mean=[mean] * POSE_FEATURE_DIM,
        std=[std] * POSE_FEATURE_DIM,
        guarded_count=0,
        guarded_mask=[False] * POSE_FEATURE_DIM,
        frames_hash="pose-frames",
    )


def _visual_stats(mean: float = 0.0, std: float = 1.0) -> Dinov3StandardizationStats:
    return Dinov3StandardizationStats(
        source="dinov3",
        dataset="le2i",
        split="train",
        target_fps=10.0,
        window_frames=24,
        stride=TRAIN_STRIDE,
        window_count=1,
        feature_dim=_VISUAL_DIM,
        mean=[mean] * _VISUAL_DIM,
        std=[std] * _VISUAL_DIM,
        guarded_count=0,
        guarded_mask=[False] * _VISUAL_DIM,
        frames_hash="visual-frames",
    )


def _raw_quality_window() -> np.ndarray:
    quality = np.empty((24, GATE_INPUT_DIM), dtype=np.float32)
    quality[:, 0] = _QUALITY_VALUES[0]
    quality[:, 1] = _QUALITY_VALUES[1]
    return quality


class _GatedFusionSource:
    def __init__(self) -> None:
        self.items = [
            (
                np.full((24, POSE_FEATURE_DIM), 5.0, dtype=np.float32),
                np.full((24, _VISUAL_DIM), 10.0, dtype=np.float32),
                _raw_quality_window(),
                index % 3,
                (f"video-{index // 3}", 23 + index),
            )
            for index in range(5)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[str, int]]:
        return self.items[index]


class _RecordingGatedFusionModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.pose_means: list[float] = []
        self.visual_means: list[float] = []
        self.qualities: list[torch.Tensor] = []
        self.cursor = 0

    def __call__(
        self, pose: torch.Tensor, visual: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        batch_size = pose.shape[0]
        self.batch_sizes.append(batch_size)
        self.pose_means.append(float(pose.mean()))
        self.visual_means.append(float(visual.mean()))
        self.qualities.append(quality)
        logits = torch.full((batch_size, 7), -1.0)
        for offset in range(batch_size):
            logits[offset, self.cursor + offset] = 1.0
        self.cursor += batch_size
        return logits


class _FusionSource:
    def __init__(self) -> None:
        self.items = [
            (
                np.full((24, POSE_FEATURE_DIM), 5.0, dtype=np.float32),
                np.full((24, _VISUAL_DIM), 10.0, dtype=np.float32),
                index % 2,
                ("b0-video", 23 + index),
            )
            for index in range(3)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, int, tuple[str, int]]:
        return self.items[index]


class _RecordingFusionModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, pose: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(pose.shape[0])
        logits = torch.zeros((pose.shape[0], 7))
        logits[:, 1] = 1.0
        return logits


def _raises_value_error(callback) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def check_b1_prediction_standardizes_pose_and_visual_and_keeps_quality_raw() -> bool:
    from gatefall.eval.b1_events import _predict_with_identity

    source = _GatedFusionSource()
    model = _RecordingGatedFusionModel()
    video_ids, k_ends, true_labels, pred_labels = _predict_with_identity(
        model,
        source,
        _pose_stats(mean=1.0, std=2.0),
        _visual_stats(mean=4.0, std=3.0),
        device="cpu",
        batch_size=2,
    )
    expected_quality = _raw_quality_window()
    quality_is_raw = all(
        np.array_equal(
            recorded.cpu().numpy(),
            np.repeat(expected_quality[None, ...], recorded.shape[0], axis=0),
        )
        and recorded.dtype == torch.float32
        and recorded.shape[-1] == GATE_INPUT_DIM
        for recorded in model.qualities
    )
    return _check(
        "inferência B1: padroniza pose e DINOv3 separadamente, entrega a "
        "qualidade crua ([q_pose, q_visual]) ao gate sem nenhuma estatística e "
        "preserva identidade/rótulo inclusive no batch final parcial",
        model.batch_sizes == [2, 2, 1]
        and np.allclose(model.pose_means, [2.0, 2.0, 2.0])
        and np.allclose(model.visual_means, [2.0, 2.0, 2.0])
        and quality_is_raw
        and video_ids == ["video-0", "video-0", "video-0", "video-1", "video-1"]
        and k_ends == [23, 24, 25, 26, 27]
        and true_labels == [0, 1, 2, 0, 1]
        and pred_labels == [0, 1, 2, 3, 4],
    )


def check_b1_defaults_to_own_run_and_rejects_arm_a_and_b0() -> bool:
    import gatefall.eval.b1_events as b1_events

    captured: list[Path] = []

    class _Lock:
        def __init__(self, run_dir: Path) -> None:
            captured.append(run_dir)

        def __enter__(self) -> "_Lock":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    original_lock = b1_events.EventEvaluationLock
    original_locked = b1_events._run_evaluate_locked
    original_validate = b1_events.validate_local_run_dir
    setattr(b1_events, "EventEvaluationLock", cast(Any, _Lock))
    setattr(
        b1_events,
        "_run_evaluate_locked",
        cast(Any, lambda *_args, **_kwargs: None),
    )
    setattr(
        b1_events,
        "validate_local_run_dir",
        cast(Any, lambda *_args, **_kwargs: None),
    )
    try:
        b1_events.run_evaluate(force=False, dataset_name="le2i", run_dir=None)
        arm_a_run_dir = default_run_dir("le2i")
        arm_b0_run_dir = default_run_dir_for_arm("le2i", "b0_fusion")
        foreign_rejected = all(
            _raises_value_error(
                lambda candidate=candidate: b1_events.run_evaluate(
                    force=False, dataset_name="le2i", run_dir=candidate
                )
            )
            for run_dir in (arm_a_run_dir, arm_b0_run_dir)
            for candidate in (run_dir, run_dir.parent, run_dir / "stray")
        )
        cv_scope_rejected = _raises_value_error(
            lambda: b1_events.run_evaluate(
                force=False,
                dataset_name="le2i-cv",
                run_dir=Path("runs/local/le2i_cv/b1_adaptive_gate"),
            )
        )
    finally:
        setattr(b1_events, "EventEvaluationLock", original_lock)
        setattr(b1_events, "_run_evaluate_locked", original_locked)
        setattr(b1_events, "validate_local_run_dir", original_validate)

    default_is_optional = inspect.signature(b1_events.run_evaluate).parameters[
        "run_dir"
    ].default is None
    return _check(
        "run_evaluate B1: usa o run canônico b1_adaptive_gate por default, "
        "recusa os run_dirs das armas A e B0 (iguais, ancestrais ou "
        "descendentes) e recusa --dataset le2i-cv fora do escopo atual",
        default_is_optional
        and captured == [default_run_dir_for_arm("le2i", "b1_adaptive_gate")]
        and foreign_rejected
        and cv_scope_rejected,
    )


def check_guards_are_anchored_at_repository_root() -> bool:
    from gatefall.eval.b1_events import (
        guard_not_arm_a_run_dir,
        guard_not_arm_b0_run_dir,
    )

    arm_a_run_dir = repository_anchored_run_dir(default_run_dir("le2i"))
    arm_b0_run_dir = repository_anchored_run_dir(
        default_run_dir_for_arm("le2i", "b0_fusion")
    )
    original_cwd = Path.cwd()
    try:
        os.chdir(tempfile.gettempdir())
        arm_a_rejected = _raises_value_error(
            lambda: guard_not_arm_a_run_dir(arm_a_run_dir, "le2i")
        )
        arm_b0_rejected = _raises_value_error(
            lambda: guard_not_arm_b0_run_dir(arm_b0_run_dir, "le2i")
        )
    finally:
        os.chdir(original_cwd)
    return _check(
        "as guardas da avaliação B1 continuam recusando os run_dirs absolutos "
        "das armas A e B0 quando a CLI roda de outro cwd: o run_dir padrão é "
        "ancorado em REPOSITORY_ROOT, não no diretório corrente",
        arm_a_rejected and arm_b0_rejected,
    )


def check_alarm_protocol_is_the_frozen_baseline_a_protocol() -> bool:
    import gatefall.eval.b1_events as b1_events

    source = inspect.getsource(b1_events)
    no_construction = (
        "AlarmProtocol(" not in source
        and "replace(BASELINE_A_ALARM_PROTOCOL" not in source
    )
    no_divergent_attribute = all(
        value == BASELINE_A_ALARM_PROTOCOL
        for value in vars(b1_events).values()
        if isinstance(value, AlarmProtocol)
    )
    return _check(
        "a avaliação B1 reutiliza o protocolo de alarme congelado do braço A: "
        "não constrói nem deriva um AlarmProtocol próprio e os campos "
        "congelados seguem intactos",
        no_construction
        and no_divergent_attribute
        and BASELINE_A_ALARM_PROTOCOL.fall_label == 1
        and BASELINE_A_ALARM_PROTOCOL.trigger_consecutive == 3
        and BASELINE_A_ALARM_PROTOCOL.refractory_period_s == 5.0
        and BASELINE_A_ALARM_PROTOCOL.eval_stride == 1,
    )


def _empty_event_split() -> dict[str, object]:
    split: dict[str, object] = {}
    for field in EVENT_COUNT_FIELDS:
        split[field] = 0
    for field in EVENT_RATE_FIELDS:
        split[field] = 0.0
    split["latency_seconds"] = {"per_event": [], "mean": None, "median": None}
    return split


def _published_event_report_fields() -> set[str]:
    """Campos de topo que `_run_evaluate_locked` grava em event_metrics.json,
    lidos da própria fonte para que renomear/adicionar/remover um campo quebre
    a checagem."""
    import gatefall.eval.b1_events as b1_events

    tree = ast.parse(inspect.getsource(b1_events._run_evaluate_locked))
    fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "report"
                and isinstance(node.value, ast.Dict)
            ):
                fields.update(
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "report"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                fields.add(target.slice.value)
    return fields


def check_event_artifact_contract_accepts_b1_identity() -> bool:
    checkpoint_path = Path("runs/local/le2i/b1_adaptive_gate/checkpoint.pt")
    protocol_path = Path("runs/local/le2i/b1_adaptive_gate/alarm_protocol.yaml")
    split = _empty_event_split()
    report = {
        "run_name": B1_ADAPTIVE_GATE_CONFIG.run_name,
        "checkpoint_path": str(checkpoint_path),
        "alarm_protocol_path": str(protocol_path),
        "splits": {"val": dict(split), "test": dict(split)},
    }
    accepted = True
    try:
        validate_event_metrics(
            report,
            cast(Any, B1_ADAPTIVE_GATE_CONFIG),
            checkpoint_path,
            protocol_path,
        )
    except ValueError:
        accepted = False
    return _check(
        "contrato de event_metrics.json compartilhado aceita run_name B1 sem "
        "acrescentar nem remover campos do split nem do topo do relatório",
        accepted
        and set(split) == EVENT_SPLIT_FIELDS
        and _published_event_report_fields() == _EVENT_REPORT_TOP_LEVEL_FIELDS,
    )


def _write_foreign_run_dir(run_dir: Path, config_dict: dict) -> None:
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config_dict, stream, sort_keys=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"run_name": "stub"}, stream)
    (run_dir / "checkpoint.pt").write_bytes(b"stub")


def check_b1_run_validation_rejects_b0_and_arm_a_run_dirs() -> bool:
    rejections: list[bool] = []
    for config_dict in (B0_FUSION_CONFIG.to_dict(), BASELINE_A_CONFIG.to_dict()):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_foreign_run_dir(run_dir, config_dict)
            try:
                validate_b1_training_run(run_dir)
            except RuntimeError as exc:
                rejections.append("config.yaml" in str(exc))
            else:
                rejections.append(False)
    return _check(
        "validate_b1_training_run recusa um run_dir cujo config.yaml pertence "
        "às armas B0 ou A, apontando config.yaml na mensagem",
        all(rejections),
    )


def _staged_event_pair(run_dir: Path) -> tuple[Path, Path]:
    staged_protocol = run_dir / ".alarm_protocol.pending-selftest.yaml"
    staged_metrics = run_dir / ".event_metrics.pending-selftest.json"
    save_alarm_protocol(BASELINE_A_ALARM_PROTOCOL, staged_protocol, force=True)
    with staged_metrics.open("w", encoding="utf-8") as stream:
        json.dump({"run_name": B1_ADAPTIVE_GATE_CONFIG.run_name}, stream)
    return staged_protocol, staged_metrics


def check_event_artifact_lifecycle_promotes_and_isolates_b1_run() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        b1_run_dir = root / "b1_adaptive_gate"
        sibling_dirs = [root / "b0_fusion", root / "baseline_a"]
        b1_run_dir.mkdir()
        for sibling in sibling_dirs:
            sibling.mkdir()
        protocol_path = b1_run_dir / "alarm_protocol.yaml"
        metrics_path = b1_run_dir / "event_metrics.json"
        with EventEvaluationLock(b1_run_dir) as lock:
            staged_protocol, staged_metrics = _staged_event_pair(b1_run_dir)
            expected_protocol_hash = sha256_file(staged_protocol)
            expected_metrics_hash = sha256_file(staged_metrics)
            _promote_event_outputs(
                staged_protocol,
                staged_metrics,
                protocol_path,
                metrics_path,
                lock,
                preserve_previous=False,
            )
            published = (
                protocol_path.is_file()
                and metrics_path.is_file()
                and sha256_file(protocol_path) == expected_protocol_hash
                and sha256_file(metrics_path) == expected_metrics_hash
            )
            journal_gone = not (b1_run_dir / EVENT_TRANSACTION_FILE).exists()
            leftovers = list(b1_run_dir.glob(".*.pending-*")) + list(
                b1_run_dir.glob(".*.backup-*")
            )
        siblings_untouched = all(
            not any(sibling.iterdir()) for sibling in sibling_dirs
        )
    return _check(
        "lifecycle de artefatos de evento do B1: publica o par "
        "alarm_protocol.yaml/event_metrics.json no próprio run_dir, remove "
        "journal, staging e backups e não escreve nada nos run_dirs irmãos "
        "das armas B0 e A",
        published and journal_gone and not leftovers and siblings_untouched,
    )


def check_partial_event_outputs_fail_without_force() -> bool:
    import gatefall.eval.b1_events as b1_events

    original_load = b1_events._load_run_assets
    setattr(
        b1_events,
        "_load_run_assets",
        cast(Any, lambda *_args, **_kwargs: (None, None, None, None, None)),
    )
    message = ""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            save_alarm_protocol(
                BASELINE_A_ALARM_PROTOCOL, run_dir / "alarm_protocol.yaml", force=True
            )
            with EventEvaluationLock(run_dir) as lock:
                try:
                    b1_events._run_evaluate_locked(
                        False, "le2i", run_dir, lock
                    )
                except RuntimeError as exc:
                    message = str(exc)
    finally:
        setattr(b1_events, "_load_run_assets", original_load)
    return _check(
        "avaliação B1 com par de artefatos incompleto falha sem --force em vez "
        "de sobrescrever silenciosamente o run",
        "parcial" in message,
    )


def check_b0_evaluator_contract_unchanged() -> bool:
    from gatefall.eval.b0_events import _predict_with_identity

    source = _FusionSource()
    model = _RecordingFusionModel()
    video_ids, k_ends, true_labels, pred_labels = _predict_with_identity(
        model,
        source,
        _pose_stats(),
        _visual_stats(),
        device="cpu",
        batch_size=2,
    )
    return _check(
        "regressão B0: o avaliador da arma B0 continua aceitando uma fonte de "
        "4 elementos e um modelo de dois argumentos, preservando seu contrato "
        "de identidade",
        model.batch_sizes == [2, 1]
        and video_ids == ["b0-video"] * 3
        and k_ends == [23, 24, 25]
        and true_labels == [0, 1, 0]
        and pred_labels == [1, 1, 1],
    )


def check_non_canonical_foreign_arm_run_dir_is_rejected_without_creating_anything() -> bool:
    import gatefall.eval.b1_events as b1_events

    original_validate = b1_events.validate_local_run_dir
    original_guard_a = b1_events.guard_not_arm_a_run_dir
    original_guard_b0 = b1_events.guard_not_arm_b0_run_dir
    original_lock = b1_events.EventEvaluationLock
    original_locked = b1_events._run_evaluate_locked
    noop = cast(Any, lambda *_args, **_kwargs: None)
    setattr(b1_events, "validate_local_run_dir", noop)
    setattr(b1_events, "guard_not_arm_a_run_dir", noop)
    setattr(b1_events, "guard_not_arm_b0_run_dir", noop)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            foreign_dirs = []
            for name, config in (
                ("b0_fusion_seed7", B0_FUSION_CONFIG.to_dict()),
                ("baseline_a_seed13", BASELINE_A_CONFIG.to_dict()),
            ):
                run_dir = root / name
                run_dir.mkdir()
                with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(config, stream, sort_keys=False)
                foreign_dirs.append(run_dir)
            foreign_rejected = all(
                _raises_value_error(
                    lambda candidate=candidate: b1_events.run_evaluate(
                        force=False, dataset_name="le2i", run_dir=candidate
                    )
                )
                for candidate in foreign_dirs
            )
            no_lock_written = not any(
                (run_dir / EVENT_LOCK_FILE).exists() for run_dir in foreign_dirs
            )

            class _Lock:
                def __init__(self, run_dir: Path) -> None:
                    self.run_dir = run_dir

                def __enter__(self) -> "_Lock":
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

            setattr(b1_events, "EventEvaluationLock", cast(Any, _Lock))
            setattr(b1_events, "_run_evaluate_locked", noop)
            fresh_run_dir = root / "b1_adaptive_gate_seed7"
            fresh_accepted = True
            try:
                b1_events.run_evaluate(
                    force=False, dataset_name="le2i", run_dir=fresh_run_dir
                )
            except ValueError:
                fresh_accepted = False
    finally:
        setattr(b1_events, "validate_local_run_dir", original_validate)
        setattr(b1_events, "guard_not_arm_a_run_dir", original_guard_a)
        setattr(b1_events, "guard_not_arm_b0_run_dir", original_guard_b0)
        setattr(b1_events, "EventEvaluationLock", original_lock)
        setattr(b1_events, "_run_evaluate_locked", original_locked)
    return _check(
        "run_evaluate B1 recusa um run_dir não canônico de outra arma "
        "(b0_fusion_seed7, baseline_a_seed13) antes de adquirir o lock, sem "
        "criar diretório nem gravar .event-evaluation.lock, e mantém o "
        "caminho de um run_dir B1 ainda sem config.yaml",
        foreign_rejected and no_lock_written and fresh_accepted,
    )


def run_b1_events_selftest() -> bool:
    checks = [
        check_b1_prediction_standardizes_pose_and_visual_and_keeps_quality_raw(),
        check_b1_defaults_to_own_run_and_rejects_arm_a_and_b0(),
        check_guards_are_anchored_at_repository_root(),
        check_alarm_protocol_is_the_frozen_baseline_a_protocol(),
        check_event_artifact_contract_accepts_b1_identity(),
        check_b1_run_validation_rejects_b0_and_arm_a_run_dirs(),
        check_non_canonical_foreign_arm_run_dir_is_rejected_without_creating_anything(),
        check_event_artifact_lifecycle_promotes_and_isolates_b1_run(),
        check_partial_event_outputs_fail_without_force(),
        check_b0_evaluator_contract_unchanged(),
    ]
    ok = all(checks)
    if not ok:
        print("\nb1 events selftest FALHOU", file=sys.stderr)
    else:
        print("\nb1 events selftest OK: todas as checagens passaram")
    return ok
