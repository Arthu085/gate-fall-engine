"""Análise pós-hoc de sensibilidade do protocolo de alarme congelado da arma A.

Ferramenta independente de estágio, deliberadamente fora do pipeline padrão
(`gatefall.pipeline`) e da suíte de lifecycle de
`gatefall.eval.baseline_a.cli` (não abre o lock, não escreve o journal e
não toca `alarm_protocol.yaml`/`event_metrics.json`). Varre
`trigger_consecutive` (1..5) e uma grade de `refractory_period_s` derivando
cada protocolo de `BASELINE_A_ALARM_PROTOCOL` via `dataclasses.replace`,
recomputando `split_event_report` sobre a mesma passada de inferência —
nunca reexecuta o modelo por célula da grade.

Este módulo nunca seleciona, ranqueia, recomenda ou promove um protocolo
substituto: `BASELINE_A_ALARM_PROTOCOL` permanece a única configuração
congelada usada em `gatefall.eval.baseline_a.cli`. As métricas do split de
teste aqui produzidas são estritamente descritivas — nenhuma decisão de
modelo/protocolo é ou pode ser tomada a partir delas.
"""

import argparse
import json
import math
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from gatefall.config import EVAL_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset
from gatefall.data.windowing import build_window_index
from gatefall.datasets import get_dataset
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL
from gatefall.eval.shared.events import extract_label_segments, split_event_report
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization,
    load_stats,
    validate_stats_layout,
)
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import build_pose_features
from gatefall.runs import validate_local_run_dir
from gatefall.train.baseline_a.artifacts import load_compatible_checkpoint, validate_training_run
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.shared.tcn import TCNClassifier

RUN_DIR = Path("runs/local/le2i/baseline_a")
SENSITIVITY_JSON_FILE = "alarm_protocol_sensitivity.json"
SENSITIVITY_CSV_FILE = "alarm_protocol_sensitivity.csv"

DEFAULT_REFRACTORY_GRID_S: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0)
TRIGGER_CONSECUTIVE_GRID: tuple[int, ...] = (1, 2, 3, 4, 5)

CSV_COLUMNS = [
    "trigger_consecutive",
    "refractory_period_s",
    "is_frozen_protocol",
    "split",
    "n_fall_events",
    "n_detected_events",
    "sensitivity",
    "n_false_alarms",
    "false_alarms_per_hour",
    "latency_mean_s",
    "latency_median_s",
    "fall_sensitivity",
    "fall_or_fallen_sensitivity",
]

NO_SELECTION_NOTE = (
    "nenhuma seleção/ranking/promoção de protocolo foi realizada; métricas "
    "do split de teste são apenas descritivas"
)


def _load_model(
    config: TrainConfig, checkpoint_path: Path, device: str
) -> TCNClassifier:
    model = load_compatible_checkpoint(checkpoint_path, config).to(device)
    model.eval()
    return model


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


@dataclass(frozen=True)
class SplitPredictions:
    video_ids: list[str]
    k_ends: list[int]
    true_labels: list[int]
    pred_labels: list[int]
    usable_windows: int
    total_windows: int
    labeled_windows: int


def _validate_refractory_grid_s(values: Sequence[float]) -> None:
    if len(values) == 0:
        raise ValueError("grade de refractory_period_s não pode ser vazia")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(
                f"refractory_period_s inválido (não finito): {value!r}"
            )
        if value < 0:
            raise ValueError(
                f"refractory_period_s inválido (negativo): {value!r}"
            )


def _is_frozen_protocol(trigger_consecutive: int, refractory_period_s: float) -> bool:
    return (
        trigger_consecutive == BASELINE_A_ALARM_PROTOCOL.trigger_consecutive
        and refractory_period_s == BASELINE_A_ALARM_PROTOCOL.refractory_period_s
    )


def build_sensitivity_rows(
    val: SplitPredictions,
    test: SplitPredictions,
    refractory_grid_s: Sequence[float] = DEFAULT_REFRACTORY_GRID_S,
) -> list[dict]:
    _validate_refractory_grid_s(refractory_grid_s)

    rows: list[dict] = []
    for trigger_consecutive in TRIGGER_CONSECUTIVE_GRID:
        for refractory_period_s in refractory_grid_s:
            protocol = replace(
                BASELINE_A_ALARM_PROTOCOL,
                trigger_consecutive=trigger_consecutive,
                refractory_period_s=refractory_period_s,
                positive_labels=list(BASELINE_A_ALARM_PROTOCOL.positive_labels),
            )
            splits = {}
            for split_name, split in (("val", val), ("test", test)):
                splits[split_name] = split_event_report(
                    split.video_ids,
                    split.k_ends,
                    split.true_labels,
                    split.pred_labels,
                    protocol,
                    split.usable_windows,
                    split.total_windows,
                    split.labeled_windows,
                )
            rows.append(
                {
                    "trigger_consecutive": trigger_consecutive,
                    "refractory_period_s": refractory_period_s,
                    "is_frozen_protocol": _is_frozen_protocol(
                        trigger_consecutive, refractory_period_s
                    ),
                    "splits": splits,
                }
            )
    return rows


def _csv_rows_from_report(rows: list[dict]) -> list[dict]:
    csv_rows: list[dict] = []
    for row in rows:
        for split_name in ("val", "test"):
            split_report = row["splits"][split_name]
            latency = split_report["latency_seconds"]
            csv_rows.append(
                {
                    "trigger_consecutive": row["trigger_consecutive"],
                    "refractory_period_s": row["refractory_period_s"],
                    "is_frozen_protocol": row["is_frozen_protocol"],
                    "split": split_name,
                    "n_fall_events": split_report["n_fall_events"],
                    "n_detected_events": split_report["n_detected_events"],
                    "sensitivity": split_report["sensitivity"],
                    "n_false_alarms": split_report["n_false_alarms"],
                    "false_alarms_per_hour": split_report["false_alarms_per_hour"],
                    "latency_mean_s": latency["mean"],
                    "latency_median_s": latency["median"],
                    "fall_sensitivity": split_report["fall_sensitivity"],
                    "fall_or_fallen_sensitivity": split_report["fall_or_fallen_sensitivity"],
                }
            )
    return csv_rows


def _write_sensitivity_outputs(
    run_dir: Path, report: dict, csv_rows: list[dict], force: bool
) -> bool:
    json_path = run_dir / SENSITIVITY_JSON_FILE
    csv_path = run_dir / SENSITIVITY_CSV_FILE
    if not force and (json_path.is_file() or csv_path.is_file()):
        present = [
            str(path) for path in (json_path, csv_path) if path.is_file()
        ]
        print(f"skip {', '.join(present)} (já existe, use --force para sobrescrever)")
        return False

    run_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    json_tmp = run_dir / f".{SENSITIVITY_JSON_FILE}.tmp-{token}"
    csv_tmp = run_dir / f".{SENSITIVITY_CSV_FILE}.tmp-{token}"

    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    csv_text = pd.DataFrame(csv_rows, columns=CSV_COLUMNS).to_csv(index=False)

    try:
        json_tmp.write_text(json_text, encoding="utf-8")
        csv_tmp.write_text(csv_text, encoding="utf-8")
        # As duas promoções ficam adjacentes de propósito: nada falível corre
        # entre elas, então uma falha não pode promover um artefato sem o
        # outro. É um melhor esforço de pareamento, não uma transação com
        # journal.
        os.replace(json_tmp, json_path)
        os.replace(csv_tmp, csv_path)
    except BaseException:
        json_tmp.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
        raise

    print(f"{json_path}: análise de sensibilidade gravada")
    print(f"{csv_path}: {len(csv_rows)} linhas")
    return True


def run_analyze(
    force: bool,
    dataset_name: str = "le2i",
    run_dir: Path = RUN_DIR,
    refractory_grid_s: Sequence[float] = DEFAULT_REFRACTORY_GRID_S,
) -> None:
    _validate_refractory_grid_s(refractory_grid_s)
    validate_local_run_dir(run_dir, dataset_name)

    adapter = get_dataset(dataset_name)
    checkpoint_path = run_dir / "checkpoint.pt"
    training_metrics_path = run_dir / "metrics.json"

    expected_config = replace(
        BASELINE_A_CONFIG,
        standardization_stats_path=str(adapter.pose_stats_path),
        standardization_stats_sha256=sha256_file(adapter.pose_stats_path),
    )
    config = validate_training_run(run_dir, expected_config=expected_config)
    if config.eval_stride != EVAL_STRIDE:
        raise ValueError(
            f"config.eval_stride ({config.eval_stride}) diverge de "
            f"EVAL_STRIDE ({EVAL_STRIDE})"
        )

    stats = load_stats(adapter.pose_stats_path)
    validate_stats_layout(stats)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(config, checkpoint_path, device)

    frames = adapter.load_frames()

    split_predictions: dict[str, SplitPredictions] = {}
    n_fall_segments_annotation: dict[str, int] = {}
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

        split_predictions[split] = SplitPredictions(
            video_ids=video_ids,
            k_ends=k_ends,
            true_labels=true_labels,
            pred_labels=pred_labels,
            usable_windows=usable_windows,
            total_windows=total_windows,
            labeled_windows=labeled_windows,
        )
        n_fall_segments_annotation[split] = _n_fall_segments_in_annotation(frames, split)

    rows = build_sensitivity_rows(
        split_predictions["val"], split_predictions["test"], refractory_grid_s
    )
    # Cruza a contagem de segmentos fall da anotação bruta uma única vez por
    # split (n_fall_events é invariante ao longo da varredura de protocolo,
    # pois depende só dos rótulos verdadeiros, não de trigger/refratário) —
    # mesma checagem de baseline_a_events._run_evaluate_locked.
    for split in ("val", "test"):
        n_fall_events_split = rows[0]["splits"][split]["n_fall_events"]
        if n_fall_events_split != n_fall_segments_annotation[split]:
            raise ValueError(
                f"split={split!r}: n_fall_events recomputado "
                f"({n_fall_events_split}) diverge da contagem de segmentos "
                f"fall na anotação bruta ({n_fall_segments_annotation[split]})"
            )

    report = {
        "run_name": config.run_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_metrics_path": str(training_metrics_path),
        "training_metrics_sha256": sha256_file(training_metrics_path),
        "frozen_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
        "grid": {
            "trigger_consecutive": list(TRIGGER_CONSECUTIVE_GRID),
            "refractory_period_s": list(refractory_grid_s),
        },
        "metadata": {
            "selection_performed": False,
            "test_split_is_descriptive_only": True,
            "note": NO_SELECTION_NOTE,
        },
        "rows": rows,
    }
    csv_rows = _csv_rows_from_report(rows)

    _write_sensitivity_outputs(run_dir, report, csv_rows, force)


def _parse_refractory_grid(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--refractory-grid inválido: {raw!r} ({exc})"
        ) from exc
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help=(
            "Varredura pós-hoc de sensibilidade do protocolo de alarme sobre "
            "o checkpoint treinado; nunca seleciona/ranqueia/promove protocolo"
        ),
    )
    analyze_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    analyze_parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    analyze_parser.add_argument(
        "--refractory-grid",
        type=_parse_refractory_grid,
        default=DEFAULT_REFRACTORY_GRID_S,
        help=(
            "Lista de refractory_period_s separada por vírgula, na ordem em "
            "que deve aparecer na grade (nunca ordenada); padrão "
            f"{','.join(str(v) for v in DEFAULT_REFRACTORY_GRID_S)}"
        ),
    )
    analyze_parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve alarm_protocol_sensitivity.json/.csv já existentes",
    )
    subparsers.add_parser(
        "selftest",
        help="Roda checagens sintéticas da análise de sensibilidade do protocolo de alarme",
    )

    args = parser.parse_args()
    if args.command == "analyze":
        run_analyze(
            force=args.force,
            dataset_name=args.dataset,
            run_dir=args.run_dir,
            refractory_grid_s=args.refractory_grid,
        )
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.alarm_protocol_sensitivity import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
