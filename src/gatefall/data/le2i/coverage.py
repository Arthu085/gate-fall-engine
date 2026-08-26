"""Auditoria de cobertura dos segmentos anotados sobre a duração dos vídeos Le2i."""

import sys
from typing import cast

import pandas as pd

from gatefall.data.le2i.annotations import (
    LABELS_DIR,
    LE2I_LABELS_FILENAME,
    load_annotation_splits,
)
from gatefall.data.le2i.path_matching import normalize_annotation_video_path
from gatefall.data.le2i.verification import load_le2i_manifest


def load_le2i_labels() -> pd.DataFrame:
    path = LABELS_DIR / LE2I_LABELS_FILENAME
    if not path.exists():
        print(
            f"erro: {path} não encontrado. Rode "
            "`uv run python scripts/fetch_labels.py` antes de auditar.",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_csv(path)


def _sweep_gap_and_overlap(
    segments: list[tuple[float, float]], video_duration_s: float
) -> tuple[float, float]:
    if not segments:
        return video_duration_s, 0.0

    # eventos (posição, delta): delta=-1 (fim) ordena antes de delta=+1 (início)
    # na mesma posição, para que segmentos apenas encostados não contem como
    # sobreposição. Entre eventos consecutivos a multiplicidade é constante.
    events = sorted(
        [(start, 1) for start, _ in segments] + [(end, -1) for _, end in segments]
    )

    gap_s = 0.0
    overlap_s = 0.0
    multiplicity = 0
    previous_position = 0.0
    for position, delta in events:
        length = position - previous_position
        if length > 0.0:
            left = max(previous_position, 0.0)
            right = min(position, video_duration_s)
            if right > left:
                clipped_length = right - left
                if multiplicity == 0:
                    gap_s += clipped_length
                elif multiplicity >= 2:
                    overlap_s += (
                        multiplicity * (multiplicity - 1) / 2 * clipped_length
                    )
        multiplicity += delta
        previous_position = position

    if multiplicity == 0 and previous_position < video_duration_s:
        gap_s += video_duration_s - previous_position

    return gap_s, overlap_s


def build_per_video_coverage(
    manifest: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    pooled = cast(pd.DataFrame, pd.concat(splits.values(), ignore_index=True)).copy()
    pooled["video_id"] = pooled["path"].map(normalize_annotation_video_path)

    segments_by_video: dict[str, list[tuple[float, float]]] = {}
    for video_id, group in pooled.groupby("video_id"):
        segments_by_video[str(video_id)] = list(
            zip(group["start"].astype(float), group["end"].astype(float))
        )

    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        video_id = str(row["video_id"])
        fps = float(cast(float, row["fps"]))
        n_frames_counted = int(cast(int, row["n_frames_counted"]))
        video_duration_s = n_frames_counted / fps
        frame_duration_s = 1.0 / fps

        segments = segments_by_video.get(video_id, [])
        overhang_s = sum(
            max(0.0, end - max(start, video_duration_s)) for start, end in segments
        )
        gap_s, overlap_s = _sweep_gap_and_overlap(segments, video_duration_s)

        rows.append(
            {
                "video_id": video_id,
                "env": str(row["env"]),
                "split": str(row["split"]),
                "video_duration_s": video_duration_s,
                "frame_duration_s": frame_duration_s,
                "n_segments": len(segments),
                "segments_total_s": sum(end - start for start, end in segments),
                "gap_s": gap_s,
                "overlap_s": overlap_s,
                "overhang_s": overhang_s,
                "duration_delta_s": abs(
                    float(cast(float, row["duration_s"])) - video_duration_s
                ),
            }
        )
    return pd.DataFrame(rows)


def report_aggregate_totals(
    per_video: pd.DataFrame, total_video_duration_s: float
) -> None:
    print("\n=== totais agregados de cobertura ===")
    segments_total_s = float(cast(pd.Series, per_video["segments_total_s"]).sum())
    gap_s = float(cast(pd.Series, per_video["gap_s"]).sum())
    overlap_s = float(cast(pd.Series, per_video["overlap_s"]).sum())
    overhang_s = float(cast(pd.Series, per_video["overhang_s"]).sum())
    gap_pct = (
        (gap_s / total_video_duration_s * 100) if total_video_duration_s else 0.0
    )
    print(f"duração total dos vídeos: {total_video_duration_s:.2f}s")
    print(f"soma de segments_total_s: {segments_total_s:.2f}s")
    print(f"gap_s total: {gap_s:.2f}s ({gap_pct:.2f}% da duração total)")
    print(f"overlap_s total: {overlap_s:.2f}s")
    print(f"overhang_s total: {overhang_s:.2f}s")


def report_worst_videos(per_video: pd.DataFrame, column: str, n: int = 10) -> None:
    print(f"\n=== top {n} vídeos por {column} ===")
    if float(cast(pd.Series, per_video[column]).sum()) == 0.0:
        print(f"nenhum vídeo com {column} — total é zero")
        return
    worst = cast(
        pd.DataFrame, per_video.sort_values(column, ascending=False)
    ).head(n)
    for _, row in worst.iterrows():
        print(f"  {row['video_id']}: {column}={row[column]:.4f}")


def report_gap_quantiles(per_video: pd.DataFrame) -> None:
    print("\n=== quantis de gap_s ===")
    gap = per_video["gap_s"]
    quantiles = {
        "min": gap.min(),
        "p25": gap.quantile(0.25),
        "median": gap.median(),
        "p75": gap.quantile(0.75),
        "max": gap.max(),
    }
    for name, value in quantiles.items():
        print(f"  {name}: {float(value):.4f}s")


def report_perfectly_tiled_count(per_video: pd.DataFrame) -> None:
    print("\n=== vídeos perfeitamente cobertos (tiled) ===")
    tolerance = per_video["frame_duration_s"]
    tiled = per_video[
        (per_video["gap_s"] < tolerance)
        & (per_video["overlap_s"] < tolerance)
        & (per_video["overhang_s"] < tolerance)
    ]
    print(f"{len(tiled)} de {len(per_video)} vídeos perfeitamente cobertos")


def report_gap_breakdown(per_video: pd.DataFrame, by: str) -> None:
    print(f"\n=== gap_s por {by} ===")
    grouped = per_video.groupby(by).agg(
        gap_s=("gap_s", "sum"), video_duration_s=("video_duration_s", "sum")
    )
    for group_key, row in grouped.iterrows():
        duration = float(cast(float, row["video_duration_s"]))
        gap = float(cast(float, row["gap_s"]))
        pct = (gap / duration * 100) if duration else 0.0
        print(f"  {group_key}: gap_s={gap:.2f}s ({pct:.2f}% de {duration:.2f}s)")


def report_zero_segment_videos(per_video: pd.DataFrame) -> bool:
    print("\n=== vídeos sem nenhum segmento anotado ===")
    zero = per_video[per_video["n_segments"] == 0]
    if zero.empty:
        print("nenhum vídeo sem segmentos")
        return True
    for _, row in zero.iterrows():
        print(f"  {row['video_id']}")
    return False


def _segment_tuples(df: pd.DataFrame) -> set[tuple[str, float, float, str]]:
    return set(
        zip(
            df["path"].astype(str),
            df["start"].astype(float).round(6),
            df["end"].astype(float).round(6),
            df["label"].astype(str),
        )
    )


def report_annotation_source_agreement(
    le2i_csv: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> bool:
    print("\n=== cross-check: le2i.csv vs união dos splits ===")
    splits_union = cast(pd.DataFrame, pd.concat(splits.values(), ignore_index=True))

    print(f"le2i.csv: {len(le2i_csv)} segmentos")
    print(f"união dos splits: {len(splits_union)} segmentos")

    le2i_tuples = _segment_tuples(le2i_csv)
    splits_tuples = _segment_tuples(splits_union)

    only_le2i = sorted(le2i_tuples - splits_tuples)
    only_splits = sorted(splits_tuples - le2i_tuples)

    if only_le2i:
        print(f"apenas em le2i.csv ({len(only_le2i)}), mostrando até 20:")
        for item in only_le2i[:20]:
            print(f"  {item}")
    if only_splits:
        print(f"apenas na união dos splits ({len(only_splits)}), mostrando até 20:")
        for item in only_splits[:20]:
            print(f"  {item}")

    le2i_paths = set(le2i_csv["path"])
    splits_paths = set(splits_union["path"])
    only_le2i_paths = sorted(le2i_paths - splits_paths)
    only_splits_paths = sorted(splits_paths - le2i_paths)
    if only_le2i_paths:
        print(f"paths presentes apenas em le2i.csv: {only_le2i_paths}")
    if only_splits_paths:
        print(f"paths presentes apenas na união dos splits: {only_splits_paths}")

    is_identical = le2i_tuples == splits_tuples
    if is_identical:
        print("OK: conjuntos de segmentos idênticos")
    else:
        print("FALHOU: conjuntos de segmentos divergentes")
    return is_identical


def audit_le2i_coverage() -> None:
    manifest = load_le2i_manifest()
    splits = load_annotation_splits()
    le2i_labels = load_le2i_labels()

    per_video = build_per_video_coverage(manifest, splits)
    total_video_duration_s = float(
        cast(pd.Series, per_video["video_duration_s"]).sum()
    )

    report_aggregate_totals(per_video, total_video_duration_s)
    report_worst_videos(per_video, "gap_s")
    report_worst_videos(per_video, "overlap_s")
    report_worst_videos(per_video, "overhang_s")
    report_worst_videos(per_video, "duration_delta_s")
    report_gap_quantiles(per_video)
    report_perfectly_tiled_count(per_video)
    report_gap_breakdown(per_video, "env")
    report_gap_breakdown(per_video, "split")
    zero_segments_ok = report_zero_segment_videos(per_video)
    agreement_ok = report_annotation_source_agreement(le2i_labels, splits)

    failed: list[str] = []
    if not zero_segments_ok:
        failed.append("vídeos sem segmentos anotados")
    if not agreement_ok:
        failed.append("divergência entre le2i.csv e união dos splits")

    if failed:
        print(f"\naudit FALHOU: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print("\naudit OK: nenhuma falha crítica encontrada")
