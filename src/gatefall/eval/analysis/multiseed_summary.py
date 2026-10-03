"""Sumário multi-seed das armas A, B0, B1, C0 e C1.

Ferramenta somente leitura, conceitualmente separada do bootstrap agrupado por
sujeito (`gatefall.eval.analysis.grouped_bootstrap`): aqui a unidade agregada é o
**treino independente** (uma seed, um checkpoint, um `run_dir` completo), não
a réplica de reamostragem sobre um único checkpoint fixo. Este módulo nunca
mistura as duas noções de variação — jamais combina réplicas de bootstrap com
seeds de treino na mesma estatística.

Cada `--run-dir` deve ser um run local completo da arma selecionada,
diferindo apenas na seed e nos campos de auditoria permitidos pelo validador.
As armas A, B0, B1, C0 e C1 exigem classificação e avaliação de evento
íntegras. O módulo valida um fingerprint sha256 da configuração
normalizada, guarda por seed os blocos de classificação e evento validados
na íntegra (confusion_matrix, per_class, latências por evento) e
agrega estatísticas descritivas (n/mean/std/min/max) sobre `macro_f1_restricted`
e `f1_by_class` (splits train/val/test), `per_class` (splits train/val/test),
a projeção binária queda/caído derivada da confusion_matrix (splits
train/val/test) e todo campo escalar das métricas de evento (splits val/test).
Não seleciona, ranqueia nem promove nenhum run.
"""

import argparse
import hashlib
import json
import os
import statistics
import uuid
from dataclasses import replace
from pathlib import Path

import pandas as pd

from gatefall.datasets import get_dataset
from gatefall.datasets.base import DatasetAdapter
from gatefall.dinov3.dataset_guard import ensure_dinov3_dataset_supported
from gatefall.eval.shared.alarm_protocol import (
    BASELINE_A_ALARM_PROTOCOL,
    load_alarm_protocol,
)
from gatefall.eval.shared.event_artifacts import EVENT_SPLIT_FIELDS, validate_event_metrics
from gatefall.features.dinov3_standardization import load_stats as load_visual_stats
from gatefall.features.dinov3_standardization import (
    validate_stats_freshness as validate_visual_stats_freshness,
)
from gatefall.features.dinov3_standardization import (
    validate_stats_layout as validate_visual_stats_layout,
)
from gatefall.features.quality_storage import quality_set_sha256
from gatefall.features.standardization import load_stats as load_pose_stats
from gatefall.features.standardization import validate_stats_layout as validate_pose_stats_layout
from gatefall.features.standardize_dinov3 import dinov3_stats_path
from gatefall.hashing import sha256_file
from gatefall.runs import validate_local_run_dir
from gatefall.sam3.dataset_guard import ensure_sam3_dataset_supported
from gatefall.train.baseline_a.artifacts import validate_training_run
from gatefall.train.baseline_b0.artifacts import validate_b0_training_run
from gatefall.train.baseline_b0.config import B0_FUSION_CONFIG, B0TrainConfig
from gatefall.train.baseline_b0.run import resolve_b0_config
from gatefall.train.baseline_b1.artifacts import validate_b1_training_run
from gatefall.train.baseline_b1.config import B1_ADAPTIVE_GATE_CONFIG, B1TrainConfig
from gatefall.train.baseline_b1.run import resolve_b1_config
from gatefall.train.baseline_c0.artifacts import validate_c0_training_run
from gatefall.train.baseline_c0.config import C0_FUSION_CONFIG, C0TrainConfig
from gatefall.train.baseline_c0.run import resolve_c0_config_for_inputs as resolve_c0_config
from gatefall.train.shared.sam3_inputs import _validated_inputs as validated_sam3_inputs
from gatefall.train.baseline_c1.artifacts import validate_c1_training_run
from gatefall.train.baseline_c1.config import C1_ADAPTIVE_GATE_CONFIG, C1TrainConfig
from gatefall.train.baseline_c1.run import resolve_c1_config_for_inputs as resolve_c1_config
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.shared.metrics import (
    BINARY_POSITIVE_LABELS,
    RESTRICTED_CLASSES,
    binary_projection_from_confusion_matrix,
)

MULTISEED_SUMMARY_JSON_FILE = "multiseed_summary.json"
MULTISEED_SUMMARY_CSV_FILE = "multiseed_summary.csv"

MIN_SEEDS = 2
ARMS = ("A", "B0", "B1", "C0", "C1")
EVENT_ARMS = frozenset({"A", "B0", "B1", "C0", "C1"})
RunConfig = TrainConfig | B0TrainConfig | B1TrainConfig | C0TrainConfig | C1TrainConfig
AUDIT_FIELDS = frozenset({"seed", "trainable_param_count"})

CLASSIFICATION_SPLITS: tuple[str, ...] = ("train", "val", "test")
EVENT_SPLITS: tuple[str, ...] = ("val", "test")

# Todo campo escalar de `event_metrics.json[splits][split]` é agregado;
# `latency_seconds` não é escalar (é um objeto com `per_event`/`mean`/`median`)
# e é substituído por `latency_seconds_mean`/`latency_seconds_median`. A
# regra é "todo campo escalar", não uma lista escolhida a dedo — por isso a
# derivação a partir de `EVENT_SPLIT_FIELDS` em vez de uma tupla própria.
EVENT_SCALAR_FIELDS: tuple[str, ...] = tuple(
    sorted(EVENT_SPLIT_FIELDS - {"latency_seconds"})
) + ("latency_seconds_mean", "latency_seconds_median")

PER_CLASS_METRIC_FIELDS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "tp",
    "tn",
    "fp",
    "fn",
    "support",
)

# Ordem de campos de `binary_projection_from_confusion_matrix`/
# `binary_projection_summary`.
BINARY_FIELDS: tuple[str, ...] = (
    "tp",
    "tn",
    "fp",
    "fn",
    "precision",
    "recall",
    "specificity",
    "f1",
    "accuracy",
)

CSV_COLUMNS = ["split", "metric_group", "entity", "metric", "n", "mean", "std", "min", "max"]


def _config_fingerprint(config: RunConfig) -> str:
    data = config.to_dict()
    data.pop("seed", None)
    if config.arm != "A":
        data.pop("trainable_param_count", None)
    payload = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_event_scalar(split_data: dict, field: str) -> float | int | None:
    if field == "latency_seconds_mean":
        return split_data["latency_seconds"]["mean"]
    if field == "latency_seconds_median":
        return split_data["latency_seconds"]["median"]
    return split_data[field]


def _aggregate_stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": n,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if n >= 2 else None,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_all(seed_reports: list[dict], adapter: DatasetAdapter) -> dict:
    label_names = adapter.label_names
    restricted_label_names = [label_names[c] for c in RESTRICTED_CLASSES]

    classification = {
        split: {
            "macro_f1_restricted": _aggregate_stats(
                [
                    report["classification"][split]["macro_f1_restricted"]
                    for report in seed_reports
                    if report["classification"][split]["macro_f1_restricted"] is not None
                ]
            ),
            "f1_by_class": {
                name: _aggregate_stats(
                    [
                        report["classification"][split]["f1_by_class"][str(class_id)]
                        for report in seed_reports
                        if report["classification"][split]["f1_by_class"][str(class_id)]
                        is not None
                    ]
                )
                for class_id, name in zip(RESTRICTED_CLASSES, restricted_label_names)
            },
        }
        for split in CLASSIFICATION_SPLITS
    }

    per_class = {
        split: {
            name: {
                metric: _aggregate_stats(
                    [
                        report["classification"][split]["per_class"][name][metric]
                        for report in seed_reports
                        if report["classification"][split]["per_class"][name][metric]
                        is not None
                    ]
                )
                for metric in PER_CLASS_METRIC_FIELDS
            }
            for name in label_names
        }
        for split in CLASSIFICATION_SPLITS
    }

    binary_fall_fallen = {
        split: {
            field: _aggregate_stats(
                [
                    report["binary_fall_fallen"][split][field]
                    for report in seed_reports
                    if report["binary_fall_fallen"][split][field] is not None
                ]
            )
            for field in BINARY_FIELDS
        }
        for split in CLASSIFICATION_SPLITS
    }

    aggregate = {
        "classification": classification,
        "per_class": per_class,
        "binary_fall_fallen": binary_fall_fallen,
    }
    if "events" in seed_reports[0]:
        aggregate["events"] = {
            split: {
                field: _aggregate_stats(
                    [
                        value
                        for report in seed_reports
                        if (value := _extract_event_scalar(report["events"][split], field))
                        is not None
                    ]
                )
                for field in EVENT_SCALAR_FIELDS
            }
            for split in EVENT_SPLITS
        }
    return aggregate


def _csv_rows_from_aggregate(aggregate: dict, adapter: DatasetAdapter) -> list[dict]:
    label_names = adapter.label_names
    restricted_label_names = [label_names[c] for c in RESTRICTED_CLASSES]
    rows: list[dict] = []

    for split in CLASSIFICATION_SPLITS:
        split_aggregate = aggregate["classification"][split]
        rows.append(
            {
                "split": split,
                "metric_group": "classification",
                "entity": "",
                "metric": "macro_f1_restricted",
                **split_aggregate["macro_f1_restricted"],
            }
        )
        for name in restricted_label_names:
            rows.append(
                {
                    "split": split,
                    "metric_group": "classification",
                    "entity": name,
                    "metric": "f1_by_class",
                    **split_aggregate["f1_by_class"][name],
                }
            )

    for split in CLASSIFICATION_SPLITS:
        for name in label_names:
            for metric in PER_CLASS_METRIC_FIELDS:
                rows.append(
                    {
                        "split": split,
                        "metric_group": "per_class",
                        "entity": name,
                        "metric": metric,
                        **aggregate["per_class"][split][name][metric],
                    }
                )

    for split in CLASSIFICATION_SPLITS:
        for field in BINARY_FIELDS:
            rows.append(
                {
                    "split": split,
                    "metric_group": "binary",
                    "entity": "",
                    "metric": field,
                    **aggregate["binary_fall_fallen"][split][field],
                }
            )

    if "events" in aggregate:
        for split in EVENT_SPLITS:
            for field in EVENT_SCALAR_FIELDS:
                rows.append(
                    {
                        "split": split,
                        "metric_group": "events",
                        "entity": "",
                        "metric": field,
                        **aggregate["events"][split][field],
                    }
                )

    return rows


def _resolve_shared_expected(dataset_name: str, arm: str = "A") -> RunConfig:
    if arm not in ARMS:
        raise ValueError(f"arma não suportada: {arm!r}")
    adapter = get_dataset(dataset_name)
    stats_path = adapter.pose_stats_path
    if arm == "A":
        return replace(
            BASELINE_A_CONFIG,
            standardization_stats_path=str(stats_path),
            standardization_stats_sha256=sha256_file(stats_path),
        )
    if arm in {"B0", "B1"}:
        ensure_dinov3_dataset_supported(adapter)
        visual_stats_path = dinov3_stats_path(dataset_name)
        pose_stats = load_pose_stats(stats_path)
        validate_pose_stats_layout(pose_stats)
        visual_stats = load_visual_stats(visual_stats_path)
        validate_visual_stats_layout(visual_stats, dataset_name=dataset_name)
        validate_visual_stats_freshness(visual_stats, adapter.frames_path)
        common = (
            stats_path,
            sha256_file(stats_path),
            visual_stats_path,
            sha256_file(visual_stats_path),
        )
        if arm == "B0":
            return resolve_b0_config(B0_FUSION_CONFIG.seed, *common)
        return resolve_b1_config(
            B1_ADAPTIVE_GATE_CONFIG.seed,
            *common,
            adapter.quality_root,
            quality_set_sha256(
                [str(video_id) for video_id in adapter.load_frames()["video_id"].unique()],
                quality_root=adapter.quality_root,
            ),
        )
    ensure_sam3_dataset_supported(adapter)
    inputs = validated_sam3_inputs(adapter, dataset_name)
    if arm == "C0":
        return resolve_c0_config(C0_FUSION_CONFIG.seed, adapter, inputs)
    return resolve_c1_config(C1_ADAPTIVE_GATE_CONFIG.seed, adapter, inputs)


def _validate_arm_training_run(run_dir: Path, expected: RunConfig) -> RunConfig:
    if expected.arm == "A":
        assert isinstance(expected, TrainConfig)
        return validate_training_run(
            run_dir, expected_config=expected, fields_allowed_to_differ=frozenset({"seed"})
        )
    allowed = AUDIT_FIELDS
    if expected.arm == "B0":
        assert isinstance(expected, B0TrainConfig)
        return validate_b0_training_run(run_dir, expected, allowed)
    if expected.arm == "B1":
        assert isinstance(expected, B1TrainConfig)
        return validate_b1_training_run(run_dir, expected, allowed)
    if expected.arm == "C0":
        assert isinstance(expected, C0TrainConfig)
        return validate_c0_training_run(run_dir, expected, allowed)
    assert isinstance(expected, C1TrainConfig)
    return validate_c1_training_run(run_dir, expected, allowed)


def _reject_duplicate_run_dirs(run_dirs: list[Path]) -> None:
    resolved_seen: dict[Path, Path] = {}
    duplicates: list[str] = []
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved in resolved_seen:
            duplicates.append(str(run_dir))
        else:
            resolved_seen[resolved] = run_dir
    if duplicates:
        raise ValueError(
            f"--run-dir duplicado(s) (mesmo path resolvido): {', '.join(duplicates)}"
        )


def _require_classification_diagnostics(final_split: dict, run_dir: Path, split: str) -> None:
    if "confusion_matrix" not in final_split or "per_class" not in final_split:
        raise RuntimeError(
            f"metrics.json em {run_dir}: final.{split} sem confusion_matrix/"
            "per_class, necessários para o sumário multi-seed de fidelidade "
            "completa"
        )


def validate_run_event_artifacts(run_dir: Path, config: RunConfig) -> dict:
    alarm_protocol_path = run_dir / "alarm_protocol.yaml"
    if not alarm_protocol_path.is_file():
        raise RuntimeError(f"alarm_protocol.yaml ausente em {run_dir}")
    protocol = load_alarm_protocol(alarm_protocol_path)
    if protocol != BASELINE_A_ALARM_PROTOCOL:
        raise RuntimeError(
            f"alarm_protocol.yaml em {run_dir} incompatível com o "
            f"protocolo congelado da arma {config.arm}"
        )

    event_metrics_path = run_dir / "event_metrics.json"
    if not event_metrics_path.is_file():
        raise RuntimeError(f"event_metrics.json ausente em {run_dir}")
    with event_metrics_path.open(encoding="utf-8") as stream:
        event_metrics = json.load(stream)
    try:
        validate_event_metrics(
            event_metrics,
            config,
            run_dir / "checkpoint.pt",
            alarm_protocol_path,
            training_metrics_path=run_dir / "metrics.json",
            require_hashes=True,
        )
    except (ValueError, OSError, TypeError, KeyError) as exc:
        raise RuntimeError(
            f"event_metrics.json inválido em {run_dir}: {exc}"
        ) from exc
    return event_metrics


def _summarize(
    run_dirs: list[Path], shared_expected: RunConfig, adapter: DatasetAdapter
) -> tuple[dict, list[dict]]:
    if len(run_dirs) < MIN_SEEDS:
        raise ValueError(
            f"são necessárias ao menos {MIN_SEEDS} seeds distintas (--run-dir "
            f"repetível); recebido(s) {len(run_dirs)}"
        )
    _reject_duplicate_run_dirs(run_dirs)

    seed_reports: list[dict] = []
    seen_seeds: dict[int, Path] = {}
    fingerprint: str | None = None
    fingerprint_run_dir: Path | None = None

    for run_dir in run_dirs:
        validate_local_run_dir(run_dir, adapter.identifier)
        config = _validate_arm_training_run(run_dir, shared_expected)

        if config.seed in seen_seeds:
            raise RuntimeError(
                f"seed duplicada entre runs: seed={config.seed} em "
                f"{seen_seeds[config.seed]} e {run_dir}"
            )
        seen_seeds[config.seed] = run_dir

        run_fingerprint = _config_fingerprint(config)
        if fingerprint is None:
            fingerprint = run_fingerprint
            fingerprint_run_dir = run_dir
        elif run_fingerprint != fingerprint:
            raise RuntimeError(
                f"config.yaml em {run_dir} diverge (fora dos campos permitidos) da "
                f"configuração de {fingerprint_run_dir}"
            )

        checkpoint_path = run_dir / "checkpoint.pt"
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)

        classification: dict[str, dict] = {}
        binary_fall_fallen: dict[str, dict] = {}
        for split in CLASSIFICATION_SPLITS:
            final_split = metrics["final"][split]
            _require_classification_diagnostics(final_split, run_dir, split)
            classification[split] = final_split
            binary_fall_fallen[split] = binary_projection_from_confusion_matrix(
                final_split["confusion_matrix"], BINARY_POSITIVE_LABELS
            )

        seed_report = {
            "seed": config.seed,
            "run_dir": str(run_dir),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "classification": classification,
            "binary_fall_fallen": binary_fall_fallen,
        }
        if config.arm in EVENT_ARMS:
            event_metrics = validate_run_event_artifacts(run_dir, config)
            seed_report["events"] = {
                split: event_metrics["splits"][split] for split in EVENT_SPLITS
            }
        seed_reports.append(seed_report)

    assert fingerprint is not None
    aggregate = _aggregate_all(seed_reports, adapter)

    report = {
        "arm": shared_expected.arm,
        "config_fingerprint_sha256": fingerprint,
        "n_seeds": len(seed_reports),
        "seeds": seed_reports,
        "aggregate": aggregate,
    }
    csv_rows = _csv_rows_from_aggregate(aggregate, adapter)
    return report, csv_rows


def _write_multiseed_summary_outputs(
    output_dir: Path, report: dict, csv_rows: list[dict], force: bool
) -> bool:
    json_path = output_dir / MULTISEED_SUMMARY_JSON_FILE
    csv_path = output_dir / MULTISEED_SUMMARY_CSV_FILE
    if not force and (json_path.is_file() or csv_path.is_file()):
        present = [str(path) for path in (json_path, csv_path) if path.is_file()]
        print(f"skip {', '.join(present)} (já existe, use --force para sobrescrever)")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    json_tmp = output_dir / f".{MULTISEED_SUMMARY_JSON_FILE}.tmp-{token}"
    csv_tmp = output_dir / f".{MULTISEED_SUMMARY_CSV_FILE}.tmp-{token}"

    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    csv_text = pd.DataFrame(csv_rows, columns=CSV_COLUMNS).to_csv(index=False)

    try:
        json_tmp.write_text(json_text, encoding="utf-8")
        csv_tmp.write_text(csv_text, encoding="utf-8")
        os.replace(json_tmp, json_path)
        os.replace(csv_tmp, csv_path)
    except BaseException:
        json_tmp.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
        raise

    print(f"{json_path}: sumário multi-seed gravado ({report['n_seeds']} seeds)")
    print(f"{csv_path}: {len(csv_rows)} linhas")
    return True


def run_summarize(
    dataset_name: str,
    run_dirs: list[Path],
    output_dir: Path,
    force: bool,
    arm: str = "A",
) -> bool:
    adapter = get_dataset(dataset_name)
    shared_expected = _resolve_shared_expected(dataset_name, arm)
    report, csv_rows = _summarize(run_dirs, shared_expected, adapter)
    return _write_multiseed_summary_outputs(output_dir, report, csv_rows, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help=(
            "Agrega classificação e, quando aplicável, eventos de múltiplos "
            "runs da mesma arma com seeds distintas"
        ),
    )
    summarize_parser.add_argument("--dataset", default="le2i", choices=("le2i", "le2i-cv"))
    summarize_parser.add_argument("--arm", default="A", choices=ARMS)
    summarize_parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        dest="run_dirs",
        required=True,
        help="Repetível: um run_dir por seed (mínimo de %d)" % MIN_SEEDS,
    )
    summarize_parser.add_argument("--output-dir", type=Path, required=True)
    summarize_parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve multiseed_summary.json/.csv já existentes",
    )
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas do sumário multi-seed"
    )

    args = parser.parse_args()
    if args.command == "summarize":
        run_summarize(
            dataset_name=args.dataset,
            run_dirs=args.run_dirs,
            output_dir=args.output_dir,
            force=args.force,
            arm=args.arm,
        )
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.multiseed_summary import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
