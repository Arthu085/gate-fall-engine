"""Proxy de qualidade visual por quadro da fonte SAM 3 e sua validação controlada.

`q_sam3 = sam_score * present`: o score do SAM 3 da instância escolhida pela
seleção contínua, zerado quando o quadro não tem máscara utilizável. É uma
proxy operacional de degradação, não uma confiança calibrada. Não altera
`V_t`, a seleção de instância nem o contrato de armazenamento.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gatefall.data.video_io import decode_frames
from gatefall.datasets import DatasetAdapter, get_dataset
from gatefall.dinov3.quality import (
    _selection_hash,
    aggregate_metrics,
    pearson_correlation,
    select_validation_samples,
    spearman_correlation,
)
from gatefall.sam3 import storage
from gatefall.sam3.dataset_guard import (
    SAM3_SUPPORTED_DATASET_IDENTIFIERS,
    ensure_sam3_dataset_supported,
)
from gatefall.sam3.descriptors import compute_descriptor
from gatefall.sam3.features import collect_sam3_provenance
from gatefall.sam3.runtime import (
    TEXT_PROMPT,
    Sam3Instance,
    Sam3RuntimeSegmenter,
    Sam3Segmenter,
    resolve_checkpoint_path,
    resolve_runtime_project_dir,
)
from gatefall.sam3.selection import InstanceSelector
from gatefall.sam3.storage import sam3_path

DEFAULT_EVENTS_PER_CLASS = 8
SAM_SCORE_TOLERANCE = 1e-6
# Raios em pixels da resolução nativa do vídeo (o SAM 3 recebe o quadro nativo,
# não o recorte 224 × 224 do DINOv3); ganhos relativos a [0, 1].
DEGRADATION_SWEEPS: dict[str, tuple[float, ...]] = {
    "blur": (0, 2, 3, 6, 12),
    "underexposure": (1.0, 0.75, 0.5, 0.25, 0.125),
    "overexposure": (1.0, 0.75, 0.5, 0.25, 0.125),
    "contrast": (1.0, 0.75, 0.5, 0.25, 0.125),
}
RUNTIME_PROVENANCE_KEYS: tuple[str, ...] = (
    "sam3_checkpoint_sha256",
    "sam3_source_revision",
    "sam3_inference_autocast_dtype",
)


class Sam3QualityValidationError(ValueError):
    """Entrada inválida ou falha numérica na proxy de qualidade do SAM 3."""


def compute_sam3_quality(sam_score: np.ndarray, present: np.ndarray) -> np.ndarray:
    scores = np.asarray(sam_score)
    presence = np.asarray(present)
    if scores.ndim != 1 or presence.shape != scores.shape:
        raise Sam3QualityValidationError(
            "sam_score e present devem ser vetores 1D de mesmo shape [K]"
        )
    if not np.isfinite(scores).all() or not np.isfinite(presence).all():
        raise Sam3QualityValidationError("sam_score ou present contém valor não finito")
    if not np.isin(presence, (0.0, 1.0)).all():
        raise Sam3QualityValidationError("present deve ser binário (0 ou 1)")
    if np.any(scores < -SAM_SCORE_TOLERANCE) or np.any(scores > 1.0 + SAM_SCORE_TOLERANCE):
        raise Sam3QualityValidationError("sam_score deve estar no intervalo [0, 1]")
    quality = np.clip(scores.astype(np.float32), 0.0, 1.0) * presence.astype(np.float32)
    return quality.astype(np.float32, copy=False)


def _validate_frame(frame: np.ndarray) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise Sam3QualityValidationError("quadro deve ter shape [H, W, 3]")
    if frame.dtype != np.uint8:
        raise Sam3QualityValidationError("quadro deve ter dtype uint8")


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    height, width = values.shape[:2]
    integral = np.zeros((height + 1, width + 1, values.shape[2]), dtype=np.float64)
    integral[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    y0 = np.clip(np.arange(height) - radius, 0, height)
    y1 = np.clip(np.arange(height) + radius + 1, 0, height)
    x0 = np.clip(np.arange(width) - radius, 0, width)
    x1 = np.clip(np.arange(width) + radius + 1, 0, width)
    window_sum = (
        integral[y1][:, x1]
        - integral[y0][:, x1]
        - integral[y1][:, x0]
        + integral[y0][:, x0]
    )
    counts = ((y1 - y0)[:, None] * (x1 - x0)[None, :])[:, :, None]
    return window_sum / counts


def _luminance(values: np.ndarray) -> np.ndarray:
    return values @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)


def apply_frame_degradation(
    frame: np.ndarray, degradation: str, severity: float
) -> np.ndarray:
    _validate_frame(frame)
    if not np.isfinite(severity):
        raise Sam3QualityValidationError("severidade deve ser finita")
    values = frame.astype(np.float64) / 255.0
    if degradation == "blur":
        if severity < 0.0 or not float(severity).is_integer():
            raise Sam3QualityValidationError(
                "raio do box blur deve ser um inteiro não negativo"
            )
        if severity == 0:
            return frame.copy()
        degraded = _box_blur(values, int(severity))
    elif degradation in {"underexposure", "overexposure", "contrast"}:
        if not 0.0 <= severity <= 1.0:
            raise Sam3QualityValidationError("ganho deve estar no intervalo [0, 1]")
        if severity == 1.0:
            return frame.copy()
        if degradation == "underexposure":
            degraded = severity * values
        elif degradation == "overexposure":
            degraded = 1.0 - severity * (1.0 - values)
        else:
            median = float(np.quantile(_luminance(values), 0.5))
            degraded = median + severity * (values - median)
    else:
        raise Sam3QualityValidationError(f"degradação desconhecida: {degradation!r}")
    return np.rint(np.clip(degraded, 0.0, 1.0) * 255.0).astype(np.uint8)


def mask_iou(reference: np.ndarray | None, candidate: np.ndarray | None) -> float:
    """IoU entre máscaras; ausência equivale a máscara vazia e duas vazias concordam (1)."""
    if reference is None and candidate is None:
        return 1.0
    shape = reference.shape if reference is not None else cast(np.ndarray, candidate).shape
    left = reference if reference is not None else np.zeros(shape, dtype=bool)
    right = candidate if candidate is not None else np.zeros(shape, dtype=bool)
    if left.shape != right.shape:
        raise Sam3QualityValidationError("máscaras devem ter o mesmo shape")
    union = int(np.logical_or(left, right).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


@dataclass(frozen=True)
class SelectedObservation:
    q_sam3: float
    present: int
    n_instances: int
    mask: np.ndarray | None


def _observe(
    instances: list[Sam3Instance], selected_index: int | None, *, width: int, height: int
) -> SelectedObservation:
    if selected_index is None:
        return SelectedObservation(0.0, 0, len(instances), None)
    selected = instances[selected_index]
    present = int(
        compute_descriptor(selected.mask, frame_width=width, frame_height=height)[0]
    )
    quality = compute_sam3_quality(
        np.array([selected.score], dtype=np.float32),
        np.array([present], dtype=np.float32),
    )
    return SelectedObservation(
        float(quality[0]), present, len(instances), selected.mask if present else None
    )


def observe_clean_then_degraded(
    clean_instances: list[Sam3Instance],
    degraded_instances: list[Sam3Instance],
    *,
    width: int,
    height: int,
) -> tuple[SelectedObservation, SelectedObservation]:
    """Replica a seleção contínua da extração com o quadro limpo como histórico."""
    selector = InstanceSelector()
    clean_index = selector.select(clean_instances)
    degraded_index = selector.select(degraded_instances)
    return (
        _observe(clean_instances, clean_index, width=width, height=height),
        _observe(degraded_instances, degraded_index, width=width, height=height),
    )


def _instance_bucket(n_instances: int) -> str:
    if n_instances <= 0:
        return "0"
    return "1" if n_instances == 1 else "2+"


def clean_quality_rows(frames: pd.DataFrame, *, sam3_root: Path) -> pd.DataFrame:
    eligible = cast(pd.DataFrame, frames[frames["split"].isin(("train", "val"))])
    records: list[dict[str, object]] = []
    for video_id, video_rows in eligible.groupby("video_id", sort=True):
        path = sam3_path(str(video_id), sam3_root=sam3_root)
        if not path.exists():
            raise Sam3QualityValidationError(f"features SAM 3 não encontradas: {path}")
        ordered = video_rows.sort_values("frame_index")
        v_t = storage.read_v_t(path)
        sam_score = storage.read_sam_score(path)
        n_instances = storage.read_n_instances(path)
        if not (len(v_t) == len(sam_score) == len(n_instances) == len(ordered)):
            raise Sam3QualityValidationError(
                f"{video_id}: K do .h5 diverge das {len(ordered)} linhas da grade"
            )
        quality = compute_sam3_quality(sam_score, v_t[:, 0])
        for position, row in enumerate(ordered.to_dict("records")):
            count = int(n_instances[position])
            records.append(
                {
                    "split": str(row["split"]),
                    "label": int(row["label"]),
                    "video_id": str(row["video_id"]),
                    "frame_index": int(row["frame_index"]),
                    "q_sam3": float(quality[position]),
                    "present": int(v_t[position, 0]),
                    "n_instances": count,
                    "instance_bucket": _instance_bucket(count),
                }
            )
    return pd.DataFrame(records)


def _rate_summary(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    distributions = aggregate_metrics(
        rows, group_columns=group_columns, metric_columns=["q_sam3"]
    )
    rates = (
        rows.groupby(group_columns, sort=True, dropna=False)
        .agg(
            present_rate=("present", "mean"),
            multi_instance_rate=("n_instances", lambda counts: float((counts > 1).mean())),
        )
        .reset_index()
    )
    return distributions.merge(rates, on=group_columns)


def sweep_rows(
    selected: pd.DataFrame,
    *,
    frames_by_position: dict[int, np.ndarray],
    segmenter: Sam3Segmenter,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sample_number, (position, row) in enumerate(
        zip(selected.index, selected.to_dict("records")), start=1
    ):
        print(
            f"sweep {sample_number}/{len(selected)}: {row['video_id']} "
            f"quadro {row['frame_index']}",
            file=sys.stderr,
        )
        clean = frames_by_position[int(cast(int, position))]
        _validate_frame(clean)
        height, width = clean.shape[:2]
        clean_instances = segmenter.segment_frame(clean, TEXT_PROMPT)
        for degradation, severities in DEGRADATION_SWEEPS.items():
            for severity_index, severity in enumerate(severities):
                degraded = apply_frame_degradation(clean, degradation, severity)
                degraded_instances = (
                    clean_instances
                    if np.array_equal(degraded, clean)
                    else segmenter.segment_frame(degraded, TEXT_PROMPT)
                )
                clean_observation, observation = observe_clean_then_degraded(
                    clean_instances, degraded_instances, width=width, height=height
                )
                records.append(
                    {
                        "split": str(row["split"]),
                        "label": int(row["label"]),
                        "video_id": str(row["video_id"]),
                        "frame_index": int(row["frame_index"]),
                        "degradation": degradation,
                        "severity_index": severity_index,
                        "severity": severity,
                        "clean_q_sam3": clean_observation.q_sam3,
                        "q_sam3": observation.q_sam3,
                        "present": observation.present,
                        "n_instances": observation.n_instances,
                        "mask_iou": mask_iou(clean_observation.mask, observation.mask),
                    }
                )
    result = pd.DataFrame(records)
    if result.empty:
        raise Sam3QualityValidationError("nenhum evento elegível foi selecionado")
    if not np.isfinite(result[["q_sam3", "mask_iou"]].to_numpy(dtype=np.float64)).all():
        raise Sam3QualityValidationError("sweep produziu valor não finito")
    return result


def _quantile_summary(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_median": float(np.quantile(array, 0.5)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
        f"{prefix}_fraction_positive": float(np.mean(array > 0.0)),
        f"{prefix}_fraction_negative": float(np.mean(array < 0.0)),
    }


def summarize_degradation(group: pd.DataFrame) -> dict[str, object]:
    """Monotonicidade e correlações de uma degradação, por evento e agregadas.

    Eventos com `q_sam3` ou `mask_iou` constantes ao longo das cinco
    severidades não definem correlação e são contados à parte, em vez de
    entrarem como zero.
    """
    identity_columns = ["split", "label", "video_id", "frame_index"]
    severity_count = int(cast(pd.Series, group["severity_index"]).nunique())
    adjacent: list[bool] = []
    endpoint_drop: list[bool] = []
    severity_spearman: list[float] = []
    iou_spearman: list[float] = []
    constant_q = 0
    for _, event_rows in group.groupby(identity_columns, sort=True):
        ordered = event_rows.sort_values("severity_index")
        if len(ordered) != severity_count:
            raise Sam3QualityValidationError("evento sem todas as severidades do sweep")
        quality = ordered["q_sam3"].to_numpy(dtype=np.float64)
        iou = ordered["mask_iou"].to_numpy(dtype=np.float64)
        severity_index = ordered["severity_index"].to_numpy(dtype=np.float64)
        adjacent.extend(bool(right <= left) for left, right in zip(quality, quality[1:]))
        endpoint_drop.append(bool(quality[-1] < quality[0]))
        if np.ptp(quality) == 0.0:
            constant_q += 1
            continue
        severity_spearman.append(spearman_correlation(severity_index, quality))
        if np.ptp(iou) > 0.0:
            iou_spearman.append(spearman_correlation(quality, iou))
    events = len(endpoint_drop)
    return {
        "events": events,
        "non_increasing_fraction": float(np.mean(adjacent)) if adjacent else 0.0,
        "endpoint_drop_fraction": float(np.mean(endpoint_drop)) if endpoint_drop else 0.0,
        "constant_q_events": constant_q,
        "pooled_spearman_severity_q": spearman_correlation(
            group["severity_index"].to_numpy(), group["q_sam3"].to_numpy()
        ),
        "pooled_pearson_q_mask_iou": pearson_correlation(
            group["q_sam3"].to_numpy(), group["mask_iou"].to_numpy()
        ),
        "pooled_spearman_q_mask_iou": spearman_correlation(
            group["q_sam3"].to_numpy(), group["mask_iou"].to_numpy()
        ),
        "within_event_severity_n": len(severity_spearman),
        **_quantile_summary("within_event_spearman_severity_q", severity_spearman),
        "within_event_mask_iou_n": len(iou_spearman),
        **_quantile_summary("within_event_spearman_q_mask_iou", iou_spearman),
    }


def _clean_rerun_agreement(sweep: pd.DataFrame, clean_rows: pd.DataFrame) -> dict[str, object]:
    clean_reruns = cast(pd.DataFrame, sweep[sweep["severity_index"] == 0])
    stored = cast(pd.DataFrame, clean_rows[["video_id", "frame_index", "q_sam3"]])
    merged = clean_reruns.drop_duplicates(subset=["video_id", "frame_index"]).merge(
        stored.rename(columns={"q_sam3": "stored_q_sam3"}),
        on=["video_id", "frame_index"],
        how="left",
    )
    if bool(cast(pd.Series, merged["stored_q_sam3"]).isna().any()):
        raise Sam3QualityValidationError("quadro reprocessado ausente dos artefatos gravados")
    difference = (merged["clean_q_sam3"] - merged["stored_q_sam3"]).abs().to_numpy(
        dtype=np.float64
    )
    return {
        "n": len(merged),
        "abs_difference_median": float(np.quantile(difference, 0.5)),
        "abs_difference_max": float(difference.max()),
        "within_1e-3_fraction": float(np.mean(difference <= 1e-3)),
    }


def _ensure_runtime_matches_artifacts(
    runtime_manifest: dict[str, object], provenance: dict[str, str]
) -> None:
    divergences = [
        f"{key}: runtime={runtime_manifest.get(key)!r}, artefatos={provenance.get(key)!r}"
        for key in RUNTIME_PROVENANCE_KEYS
        if runtime_manifest.get(key) != provenance.get(key)
    ]
    if divergences:
        raise Sam3QualityValidationError(
            "runtime SAM 3 diverge da proveniência dos artefatos gravados: "
            + "; ".join(divergences)
        )


def _decode_selected(selected: pd.DataFrame, *, adapter: DatasetAdapter) -> dict[int, np.ndarray]:
    paths = adapter.video_paths()
    decoded: dict[int, np.ndarray] = {}
    for video_id, group in selected.groupby("video_id", sort=True):
        frames = decode_frames(
            paths[str(video_id)], [int(value) for value in group["src_index"]]
        )
        if len(frames) != len(group):
            raise Sam3QualityValidationError(
                f"decodificação incompleta de {video_id}: {len(frames)}/{len(group)}"
            )
        decoded.update(zip((int(cast(int, value)) for value in group.index), frames))
    return decoded


def run_quality_validation(
    *,
    adapter: DatasetAdapter,
    segmenter: Sam3Segmenter | None = None,
    runtime_project_dir_value: str | None = None,
    checkpoint_path_value: str | None = None,
    events_per_class: int = DEFAULT_EVENTS_PER_CLASS,
) -> dict[str, object]:
    ensure_sam3_dataset_supported(adapter)
    frames = adapter.load_frames()
    unexpected_splits = set(frames["split"]) - {"train", "val", "test"}
    if unexpected_splits:
        raise Sam3QualityValidationError(
            f"splits desconhecidos em frames: {sorted(unexpected_splits)}"
        )
    eligible = cast(pd.DataFrame, frames[frames["split"].isin(("train", "val"))])
    split_by_video = {
        str(video_id): str(split)
        for video_id, split in eligible.groupby("video_id")["split"].first().items()
    }
    provenance = collect_sam3_provenance(split_by_video, sam3_root=adapter.sam3_root)
    selected = select_validation_samples(frames, events_per_class=events_per_class)
    if "test" in set(selected["split"]):
        raise Sam3QualityValidationError("seleção incluiu o split test")

    clean_rows = clean_quality_rows(frames, sam3_root=adapter.sam3_root)
    decoded = _decode_selected(selected, adapter=adapter)
    if segmenter is not None:
        runtime_manifest = cast(
            dict[str, object], getattr(segmenter, "runtime_manifest", provenance)
        )
        _ensure_runtime_matches_artifacts(runtime_manifest, provenance)
        sweep = sweep_rows(selected, frames_by_position=decoded, segmenter=segmenter)
    else:
        with Sam3RuntimeSegmenter(
            runtime_project_dir=resolve_runtime_project_dir(runtime_project_dir_value),
            checkpoint_path=resolve_checkpoint_path(checkpoint_path_value),
        ) as live_segmenter:
            runtime_manifest = live_segmenter.runtime_manifest
            _ensure_runtime_matches_artifacts(runtime_manifest, provenance)
            sweep = sweep_rows(selected, frames_by_position=decoded, segmenter=live_segmenter)

    by_severity = _rate_summary(
        sweep, ["degradation", "severity_index", "severity"]
    ).merge(
        aggregate_metrics(
            sweep,
            group_columns=["degradation", "severity_index", "severity"],
            metric_columns=["mask_iou"],
        ),
        on=["degradation", "severity_index", "severity", "n"],
    )
    degradation_metrics = [
        {"degradation": str(degradation), **summarize_degradation(group)}
        for degradation, group in sweep.groupby("degradation", sort=True)
    ]
    selection_counts: list[dict[str, object]] = []
    for group_key, group in selected.groupby(["split", "label"], sort=True):
        split, label = cast(tuple[object, object], group_key)
        selection_counts.append(
            {"split": str(split), "label": int(str(label)), "n": len(group)}
        )
    return {
        "dataset": adapter.identifier,
        "splits": ["train", "val"],
        "proxy": "q_sam3 = clip(sam_score, 0, 1) * present",
        "artifact_provenance": provenance,
        "runtime_manifest": runtime_manifest,
        "events_per_class": events_per_class,
        "selected_events": len(selected),
        "selection_hash": _selection_hash(selected),
        "selection_counts": selection_counts,
        "clean_frame_distributions": _rate_summary(
            clean_rows, ["split", "label"]
        ).to_dict("records"),
        "clean_instance_bucket_distributions": _rate_summary(
            clean_rows, ["split", "instance_bucket"]
        ).to_dict("records"),
        "clean_rerun_agreement": _clean_rerun_agreement(sweep, clean_rows),
        "sweep_distributions": by_severity.to_dict("records"),
        "degradation_metrics": degradation_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("selftest", help="Roda checagens sintéticas da proxy q_sam3")
    validate_parser = subparsers.add_parser(
        "validate", help="Valida q_sam3 contra degradações controladas em train/val"
    )
    validate_parser.add_argument(
        "--dataset", default="le2i", choices=SAM3_SUPPORTED_DATASET_IDENTIFIERS
    )
    validate_parser.add_argument(
        "--events-per-class", type=int, default=DEFAULT_EVENTS_PER_CLASS
    )
    validate_parser.add_argument("--runtime-dir", default=None)
    validate_parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.command == "selftest":
        from gatefall.sam3.quality_selftest import run_selftest

        run_selftest()
        return

    try:
        report = run_quality_validation(
            adapter=get_dataset(args.dataset),
            runtime_project_dir_value=args.runtime_dir,
            checkpoint_path_value=args.checkpoint,
            events_per_class=args.events_per_class,
        )
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        print(f"\nsam3 quality validate FALHOU: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
