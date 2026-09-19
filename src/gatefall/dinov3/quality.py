"""Qualidade visual causal e validação de degradações do DINOv3."""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import avg_pool2d

from gatefall.config import IGNORE_LABEL
from gatefall.data.video_io import decode_frames
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.dinov3.backbone import (
    RESIZE_SIZE,
    configure_deterministic_inference,
    ensure_backbone_paths_exist,
    load_backbone,
    resolve_repo_dir,
    resolve_weights_path,
)
from gatefall.dinov3.dataset_guard import (
    DINOV3_SUPPORTED_DATASET_IDENTIFIERS,
    ensure_dinov3_dataset_supported,
)
from gatefall.dinov3.features import Dinov3Backbone, compute_features
from gatefall.dinov3.preprocessing import normalize_resized_frames, resize_frames

DEFAULT_EVENTS_PER_CLASS = 8
DEFAULT_BATCH_SIZE = 8
DEGRADATION_SWEEPS: dict[str, tuple[float, ...]] = {
    "blur": (0, 2, 3, 6, 12),
    "underexposure": (1.0, 0.75, 0.5, 0.25, 0.125),
    "overexposure": (1.0, 0.75, 0.5, 0.25, 0.125),
    "contrast": (1.0, 0.75, 0.5, 0.25, 0.125),
}


class VisualQualityValidationError(ValueError):
    """Entrada inválida ou falha numérica na validação de qualidade."""


@dataclass(frozen=True)
class VisualQualityComponents:
    q_exposure: np.ndarray
    q_contrast: np.ndarray
    q_sharpness: np.ndarray
    q_visual: np.ndarray


def _validate_resized_batch(batch: torch.Tensor) -> None:
    if batch.ndim != 4:
        raise VisualQualityValidationError("batch deve ter shape [B, 3, 224, 224]")
    if batch.shape[0] == 0:
        raise VisualQualityValidationError("batch não pode ser vazio")
    if tuple(batch.shape[1:]) != (3, RESIZE_SIZE, RESIZE_SIZE):
        raise VisualQualityValidationError("batch deve ter shape [B, 3, 224, 224]")
    if not batch.is_floating_point():
        raise VisualQualityValidationError("batch deve ter dtype de ponto flutuante")
    if not bool(torch.isfinite(batch).all()):
        raise VisualQualityValidationError("batch contém valores não finitos")
    if not bool(torch.all((batch >= 0.0) & (batch <= 1.0))):
        raise VisualQualityValidationError("batch deve estar no intervalo [0, 1]")


def _luminance(batch: torch.Tensor) -> torch.Tensor:
    weights = batch.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return (batch * weights).sum(dim=1)


def compute_visual_quality(batch: torch.Tensor) -> VisualQualityComponents:
    _validate_resized_batch(batch)
    values = batch.to(torch.float32)
    luminance = _luminance(values)
    flat = luminance.flatten(start_dim=1)

    dark_fraction = (flat <= (1.0 / 255.0)).to(torch.float32).mean(dim=1)
    bright_fraction = (flat >= (254.0 / 255.0)).to(torch.float32).mean(dim=1)
    q_clip = (1.0 - dark_fraction - bright_fraction).clamp(0.0, 1.0)
    median = torch.quantile(flat, 0.5, dim=1)
    q_level = (4.0 * median * (1.0 - median)).clamp(0.0, 1.0)
    q_exposure = (q_clip * q_level).clamp(0.0, 1.0)

    percentiles = torch.quantile(
        flat, torch.tensor((0.05, 0.95), device=flat.device), dim=1
    )
    q_contrast = (percentiles[1] - percentiles[0]).clamp(0.0, 1.0)

    center = luminance[:, 1:-1, 1:-1]
    laplacian = (
        4.0 * center
        - luminance[:, :-2, 1:-1]
        - luminance[:, 2:, 1:-1]
        - luminance[:, 1:-1, :-2]
        - luminance[:, 1:-1, 2:]
    )
    high_frequency = laplacian.abs().mean(dim=(1, 2)) / 4.0
    epsilon = torch.finfo(torch.float32).eps
    q_sharpness = (high_frequency / torch.maximum(q_contrast, q_contrast.new_tensor(epsilon))).clamp(0.0, 1.0)
    q_visual = torch.pow(q_exposure * q_contrast * q_sharpness, 1.0 / 3.0)

    arrays = [
        value.detach().cpu().numpy().astype(np.float32, copy=False)
        for value in (q_exposure, q_contrast, q_sharpness, q_visual)
    ]
    if any(not np.isfinite(value).all() for value in arrays):
        raise VisualQualityValidationError("qualidade visual produziu valor não finito")
    return VisualQualityComponents(*arrays)


def apply_degradation(
    batch: torch.Tensor, degradation: str, severity: float
) -> torch.Tensor:
    _validate_resized_batch(batch)
    if not np.isfinite(severity):
        raise VisualQualityValidationError("severidade deve ser finita")
    values = batch.to(torch.float32)
    if degradation == "blur":
        if severity < 0.0 or not float(severity).is_integer():
            raise VisualQualityValidationError(
                "raio do box blur deve ser um inteiro não negativo"
            )
        radius = int(severity)
        if radius == 0:
            return values.clone()
        kernel_size = 2 * radius + 1
        return avg_pool2d(
            values,
            kernel_size,
            stride=1,
            padding=radius,
            count_include_pad=False,
        ).clamp(0.0, 1.0)
    if degradation not in {"underexposure", "overexposure", "contrast"}:
        raise VisualQualityValidationError(f"degradação desconhecida: {degradation!r}")
    if not 0.0 <= severity <= 1.0:
        raise VisualQualityValidationError("ganho deve estar no intervalo [0, 1]")
    if severity == 1.0:
        return values.clone()
    if degradation == "underexposure":
        degraded = severity * values
    elif degradation == "overexposure":
        degraded = 1.0 - severity * (1.0 - values)
    else:
        median = torch.quantile(_luminance(values).flatten(start_dim=1), 0.5, dim=1)
        degraded = median.view(-1, 1, 1, 1) + severity * (
            values - median.view(-1, 1, 1, 1)
        )
    return degraded.clamp(0.0, 1.0).to(torch.float32)


def _event_identity(row: pd.Series) -> str:
    return "|".join(
        str(row[column])
        for column in (
            "split",
            "label",
            "video_id",
            "event_start_frame",
            "event_end_frame",
            "frame_index",
            "src_index",
        )
    )


def _enumerate_events(frames: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "label", "video_id", "frame_index", "src_index"}
    missing = required - set(frames.columns)
    if missing:
        raise VisualQualityValidationError(
            f"frames não contém colunas obrigatórias: {', '.join(sorted(missing))}"
        )
    eligible = cast(
        pd.DataFrame, frames[frames["split"].isin(("train", "val"))]
    ).copy()
    eligible = eligible.sort_values(["split", "video_id", "frame_index", "src_index"])
    rows: list[dict[str, object]] = []
    for group_key, video_frames in eligible.groupby(["split", "video_id"], sort=True):
        split, video_id = cast(tuple[object, object], group_key)
        ordered = video_frames.reset_index(drop=True)
        labels = ordered["label"].to_numpy()
        starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        ends = np.r_[starts[1:], len(ordered)]
        for start, end in zip(starts, ends):
            center_position = int(start + (end - start - 1) // 2)
            center = ordered.iloc[center_position]
            rows.append(
                {
                    **center.to_dict(),
                    "split": str(split),
                    "video_id": str(video_id),
                    "event_start_frame": int(ordered.iloc[int(start)]["frame_index"]),
                    "event_end_frame": int(ordered.iloc[int(end) - 1]["frame_index"]),
                }
            )
    return pd.DataFrame(rows)


def select_validation_samples(
    frames: pd.DataFrame, *, events_per_class: int = DEFAULT_EVENTS_PER_CLASS
) -> pd.DataFrame:
    if events_per_class <= 0:
        raise VisualQualityValidationError("events_per_class deve ser positivo")
    events = _enumerate_events(frames)
    if events.empty:
        return events
    events = cast(pd.DataFrame, events[events["label"] != IGNORE_LABEL]).copy()
    events["selection_key"] = events.apply(
        lambda row: hashlib.sha256(_event_identity(row).encode("utf-8")).hexdigest(),
        axis=1,
    )
    selected = (
        events.sort_values("selection_key")
        .groupby(["split", "label"], sort=True, group_keys=False)
        .head(events_per_class)
    )
    return selected.sort_values(["split", "label", "selection_key"]).reset_index(drop=True)


def cosine_similarity(clean: np.ndarray, corrupted: np.ndarray) -> np.ndarray:
    if clean.ndim != 2 or corrupted.ndim != 2 or clean.shape != corrupted.shape:
        raise VisualQualityValidationError("descritores devem ter o mesmo shape [N, D]")
    clean_values = clean.astype(np.float32, copy=False)
    corrupted_values = corrupted.astype(np.float32, copy=False)
    numerator = np.sum(clean_values * corrupted_values, axis=1)
    denominator = np.linalg.norm(clean_values, axis=1) * np.linalg.norm(
        corrupted_values, axis=1
    )
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0.0,
    ).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise VisualQualityValidationError("similaridade produziu valor não finito")
    return result


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.shape != y_values.shape or x_values.ndim != 1 or x_values.size == 0:
        raise VisualQualityValidationError("correlação requer vetores 1D não vazios de mesmo shape")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise VisualQualityValidationError("correlação recebeu valor não finito")
    x_centered = x_values - x_values.mean()
    y_centered = y_values - y_values.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_ranks = pd.Series(np.asarray(x)).rank(method="average").to_numpy(dtype=np.float64)
    y_ranks = pd.Series(np.asarray(y)).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_correlation(x_ranks, y_ranks)


def aggregate_metrics(
    rows: pd.DataFrame,
    *,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = rows.groupby(group_columns, sort=True, dropna=False)
    for keys, group in grouped:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record: dict[str, object] = dict(zip(group_columns, key_tuple))
        record["n"] = len(group)
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise VisualQualityValidationError(f"{metric} contém valor não finito")
            record[f"{metric}_median"] = float(np.quantile(values, 0.5))
            record[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            record[f"{metric}_q75"] = float(np.quantile(values, 0.75))
        records.append(record)
    return pd.DataFrame(records)


def summarize_within_event_correlations(
    rows: pd.DataFrame,
    *,
    identity_columns: list[str],
) -> dict[str, float | int]:
    required = set(identity_columns) | {"severity_index", "q_visual", "similarity"}
    missing = required - set(rows.columns)
    if missing:
        raise VisualQualityValidationError(
            f"correlações intraevento sem colunas: {', '.join(sorted(missing))}"
        )
    if rows.empty:
        raise VisualQualityValidationError(
            "correlações intraevento requerem ao menos um evento"
        )

    pearson_values: list[float] = []
    spearman_values: list[float] = []
    for _, event_rows in rows.groupby(identity_columns, sort=True):
        severity_indices = set(
            event_rows["severity_index"].to_numpy(dtype=np.int64).tolist()
        )
        if severity_indices != set(range(5)) or len(event_rows) != 5:
            raise VisualQualityValidationError(
                "correlações intraevento requerem exatamente as cinco severidades"
            )
        q_visual = event_rows["q_visual"].to_numpy(dtype=np.float32)
        similarity = event_rows["similarity"].to_numpy(dtype=np.float32)
        pearson_values.append(pearson_correlation(q_visual, similarity))
        spearman_values.append(spearman_correlation(q_visual, similarity))

    summary: dict[str, float | int] = {"within_event_n": len(pearson_values)}
    for name, values in (
        ("within_event_pearson", pearson_values),
        ("within_event_spearman", spearman_values),
    ):
        array = np.asarray(values, dtype=np.float64)
        summary[f"{name}_median"] = float(np.quantile(array, 0.5))
        summary[f"{name}_q25"] = float(np.quantile(array, 0.25))
        summary[f"{name}_q75"] = float(np.quantile(array, 0.75))
        summary[f"{name}_fraction_positive"] = float(np.mean(array > 0.0))
    return summary


def _compute_feature_batch(
    backbone: Dinov3Backbone,
    resized: torch.Tensor,
    *,
    device: str,
) -> np.ndarray:
    features = compute_features(
        backbone, normalize_resized_frames(resized).to(device)
    ).astype(np.float32)
    norms = np.linalg.norm(features, axis=1)
    if not np.isfinite(features).all() or not np.isfinite(norms).all():
        raise VisualQualityValidationError("descritor contém valor não finito")
    if np.any(norms == 0.0):
        raise VisualQualityValidationError("descritor tem norma zero")
    return features


def _selection_hash(selected: pd.DataFrame) -> str:
    identities = sorted(_event_identity(row) for _, row in selected.iterrows())
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()


def _clean_quality_rows(
    frames: pd.DataFrame,
    *,
    adapter: DatasetAdapter,
    batch_size: int,
) -> pd.DataFrame:
    eligible = cast(pd.DataFrame, frames[frames["split"].isin(("train", "val"))])
    paths = adapter.video_paths()
    records: list[dict[str, object]] = []
    for video_id, video_rows in eligible.groupby("video_id", sort=True):
        ordered = video_rows.sort_values("frame_index")
        src_indices = [int(value) for value in ordered["src_index"]]
        decoded = decode_frames(paths[str(video_id)], src_indices)
        if len(decoded) != len(ordered):
            raise VisualQualityValidationError(
                f"decodificação incompleta de {video_id}: {len(decoded)}/{len(ordered)}"
            )
        ordered_records = ordered.to_dict("records")
        for start in range(0, len(decoded), batch_size):
            resized = resize_frames(decoded[start : start + batch_size])
            quality = compute_visual_quality(resized)
            for offset, q_visual in enumerate(quality.q_visual):
                row = ordered_records[start + offset]
                records.append(
                    {
                        "split": str(row["split"]),
                        "label": int(row["label"]),
                        "video_id": str(row["video_id"]),
                        "frame_index": int(row["frame_index"]),
                        "q_visual": float(q_visual),
                    }
                )
    return pd.DataFrame(records)


def _event_quality_rows(clean_rows: pd.DataFrame) -> pd.DataFrame:
    events = _enumerate_events(clean_rows.assign(src_index=clean_rows["frame_index"]))
    records: list[dict[str, object]] = []
    for _, event in events.iterrows():
        mask = (
            (clean_rows["split"] == event["split"])
            & (clean_rows["video_id"] == event["video_id"])
            & (clean_rows["label"] == event["label"])
            & (clean_rows["frame_index"] >= event["event_start_frame"])
            & (clean_rows["frame_index"] <= event["event_end_frame"])
        )
        records.append(
            {
                "split": event["split"],
                "label": int(str(event["label"])),
                "q_visual": float(clean_rows.loc[mask, "q_visual"].median()),
            }
        )
    return pd.DataFrame(records)


def _sweep_rows(
    selected: pd.DataFrame,
    *,
    adapter: DatasetAdapter,
    backbone: Dinov3Backbone,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    paths = adapter.video_paths()
    decoded_by_position: dict[int, np.ndarray] = {}
    for video_id, group in selected.groupby("video_id", sort=True):
        positions = [int(value) for value in group.index]
        group_frames = decode_frames(
            paths[str(video_id)], [int(value) for value in group["src_index"]]
        )
        decoded_by_position.update(zip(positions, group_frames))
    decoded = [decoded_by_position[position] for position in selected.index]
    if not decoded:
        raise VisualQualityValidationError("nenhum evento elegível foi selecionado")
    records: list[dict[str, object]] = []
    for start in range(0, len(decoded), batch_size):
        end = min(start + batch_size, len(decoded))
        clean = resize_frames(decoded[start:end])
        clean_features = _compute_feature_batch(backbone, clean, device=device)
        batch_rows = selected.iloc[start:end].reset_index(drop=True)
        for degradation, severities in DEGRADATION_SWEEPS.items():
            for severity_index, severity in enumerate(severities):
                corrupted = apply_degradation(clean, degradation, severity)
                quality = compute_visual_quality(corrupted)
                corrupted_features = _compute_feature_batch(
                    backbone, corrupted, device=device
                )
                similarities = cosine_similarity(clean_features, corrupted_features)
                for index, row in batch_rows.iterrows():
                    records.append(
                        {
                            "split": str(row["split"]),
                            "label": int(row["label"]),
                            "video_id": str(row["video_id"]),
                            "frame_index": int(row["frame_index"]),
                            "degradation": degradation,
                            "severity_index": severity_index,
                            "severity": severity,
                            "q_visual": float(quality.q_visual[index]),
                            "similarity": float(similarities[index]),
                        }
                    )
    result = pd.DataFrame(records)
    if not np.isfinite(result[["q_visual", "similarity"]].to_numpy()).all():
        raise VisualQualityValidationError("sweep produziu valor não finito")
    return result


def run_quality_validation(
    *,
    adapter: DatasetAdapter,
    repo_dir_value: str | None = None,
    weights_path_value: str | None = None,
    events_per_class: int = DEFAULT_EVENTS_PER_CLASS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    ensure_dinov3_dataset_supported(adapter)
    if batch_size <= 0:
        raise VisualQualityValidationError("batch_size deve ser positivo")
    frames = adapter.load_frames()
    unexpected_splits = set(frames["split"]) - {"train", "val", "test"}
    if unexpected_splits:
        raise VisualQualityValidationError(
            f"splits desconhecidos em frames: {sorted(unexpected_splits)}"
        )
    selected = select_validation_samples(frames, events_per_class=events_per_class)
    if "test" in set(selected["split"]):
        raise VisualQualityValidationError("seleção incluiu o split test")

    repo_dir = resolve_repo_dir(repo_dir_value)
    weights_path = resolve_weights_path(weights_path_value)
    ensure_backbone_paths_exist(repo_dir, weights_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_deterministic_inference()
    backbone = cast(Dinov3Backbone, load_backbone(repo_dir, weights_path, device))

    clean_rows = _clean_quality_rows(frames, adapter=adapter, batch_size=batch_size)
    clean_frames = aggregate_metrics(
        clean_rows,
        group_columns=["split", "label"],
        metric_columns=["q_visual"],
    )
    clean_events = aggregate_metrics(
        _event_quality_rows(clean_rows),
        group_columns=["split", "label"],
        metric_columns=["q_visual"],
    )
    sweep = _sweep_rows(
        selected,
        adapter=adapter,
        backbone=backbone,
        device=device,
        batch_size=batch_size,
    )
    by_severity = aggregate_metrics(
        sweep,
        group_columns=["degradation", "severity_index", "severity"],
        metric_columns=["q_visual", "similarity"],
    )

    degradation_metrics: list[dict[str, object]] = []
    identity_columns = ["split", "label", "video_id", "frame_index"]
    for degradation, group in sweep.groupby("degradation", sort=True):
        ordered = group.sort_values(identity_columns + ["severity_index"])
        adjacent: list[bool] = []
        for _, event_rows in ordered.groupby(identity_columns, sort=True):
            values = event_rows["q_visual"].to_numpy(dtype=np.float64)
            adjacent.extend(bool(right <= left) for left, right in zip(values, values[1:]))
        degradation_metrics.append(
            {
                "degradation": str(degradation),
                "non_increasing_fraction": float(np.mean(adjacent)) if adjacent else 0.0,
                "pooled_pearson_q_similarity": pearson_correlation(
                    group["q_visual"].to_numpy(), group["similarity"].to_numpy()
                ),
                "pooled_spearman_q_similarity": spearman_correlation(
                    group["q_visual"].to_numpy(), group["similarity"].to_numpy()
                ),
                **summarize_within_event_correlations(
                    group, identity_columns=identity_columns
                ),
            }
        )

    selection_counts: list[dict[str, object]] = []
    for group_key, group in selected.groupby(["split", "label"], sort=True):
        split, label = cast(tuple[object, object], group_key)
        selection_counts.append(
            {"split": str(split), "label": int(str(label)), "n": len(group)}
        )

    return {
        "dataset": adapter.identifier,
        "splits": ["train", "val"],
        "events_per_class": events_per_class,
        "selected_events": len(selected),
        "selection_hash": _selection_hash(selected),
        "selection_counts": selection_counts,
        "clean_frame_distributions": clean_frames.to_dict("records"),
        "clean_event_distributions": clean_events.to_dict("records"),
        "sweep_distributions": by_severity.to_dict("records"),
        "degradation_metrics": degradation_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da qualidade visual")
    validate_parser = subparsers.add_parser(
        "validate", help="Valida a proxy visual contra degradações controladas"
    )
    validate_parser.add_argument(
        "--dataset", default="le2i", choices=DINOV3_SUPPORTED_DATASET_IDENTIFIERS
    )
    validate_parser.add_argument("--events-per-class", type=int, default=DEFAULT_EVENTS_PER_CLASS)
    validate_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    validate_parser.add_argument("--repo-dir", default=None)
    validate_parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    if args.command == "selftest":
        from gatefall.dinov3.quality_selftest import run_selftest

        run_selftest()
        return

    try:
        report = run_quality_validation(
            adapter=get_dataset(args.dataset),
            repo_dir_value=args.repo_dir,
            weights_path_value=args.weights,
            events_per_class=args.events_per_class,
            batch_size=args.batch_size,
        )
    except (OSError, RuntimeError, VisualQualityValidationError, ValueError) as exc:
        print(f"\ndinov3 quality validate FALHOU: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
