"""Grade de reamostragem temporal e rótulo por quadro do Le2i (somente relatório)."""

from typing import cast

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL, TARGET_FPS
from gatefall.data.intervals import sweep_gaps_and_overlap, tag_gap_positions
from gatefall.data.le2i.annotations import load_annotation_splits
from gatefall.data.le2i.path_matching import normalize_annotation_video_path
from gatefall.data.le2i.verification import load_le2i_manifest
from gatefall.data.resampling import build_time_grid, labels_for_grid


def build_grid_frames(
    manifest: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = cast(pd.DataFrame, pd.concat(splits.values(), ignore_index=True)).copy()
    pooled["video_id"] = pooled["path"].map(normalize_annotation_video_path)

    segments_by_video: dict[str, pd.DataFrame] = {
        str(video_id): cast(
            pd.DataFrame, group[["start", "end", "label"]].reset_index(drop=True)
        )
        for video_id, group in pooled.groupby("video_id")
    }

    grid_frames_parts: list[pd.DataFrame] = []
    per_video_rows: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        video_id = str(row["video_id"])
        env = str(row["env"])
        split = str(row["split"])
        subject = int(cast(int, row["subject"]))
        fps = float(cast(float, row["fps"]))
        n_frames_counted = int(cast(int, row["n_frames_counted"]))
        duration_s = n_frames_counted / fps

        if video_id not in segments_by_video:
            raise KeyError(
                f"nenhuma anotação encontrada para video_id={video_id!r}; a bijeção "
                "vídeo<->anotação deveria ter sido garantida por `ingest verify`"
            )
        segments = segments_by_video[video_id]
        segment_tuples = list(
            zip(segments["start"].astype(float), segments["end"].astype(float))
        )

        times, src_indices = build_time_grid(n_frames_counted, fps, TARGET_FPS)
        labels, n_overlap_resolved = labels_for_grid(segments, times, IGNORE_LABEL)
        is_ignore = labels == IGNORE_LABEL
        k = int(times.shape[0])

        gap_position = np.full(k, None, dtype=object)
        gap_length_s = np.full(k, np.nan, dtype=np.float64)
        _, _, gap_intervals = sweep_gaps_and_overlap(segment_tuples, duration_s)
        for gap_start, gap_end, position in tag_gap_positions(
            gap_intervals, duration_s
        ):
            lo = int(np.searchsorted(times, gap_start, side="left"))
            hi = int(np.searchsorted(times, gap_end, side="left"))
            if hi <= lo:
                continue
            gap_position[lo:hi] = position
            gap_length_s[lo:hi] = gap_end - gap_start

        grid_frames_parts.append(
            pd.DataFrame(
                {
                    "video_id": video_id,
                    "env": env,
                    "split": split,
                    "frame_index": np.arange(k, dtype=np.int64),
                    "time_s": times,
                    "src_index": src_indices,
                    "label": labels,
                    "is_ignore": is_ignore,
                    "gap_position": gap_position,
                    "gap_length_s": gap_length_s,
                    "frame_duration_s": 1.0 / fps,
                }
            )
        )

        per_video_rows.append(
            {
                "video_id": video_id,
                "env": env,
                "split": split,
                "subject": subject,
                "fps": fps,
                "n_frames_counted": n_frames_counted,
                "duration_s": duration_s,
                "k": k,
                "naive_k": duration_s * TARGET_FPS,
                "n_overlap_resolved": n_overlap_resolved,
            }
        )

    grid_frames = cast(
        pd.DataFrame,
        pd.concat(grid_frames_parts, ignore_index=True)
        if grid_frames_parts
        else pd.DataFrame(
            columns=[
                "video_id",
                "env",
                "split",
                "frame_index",
                "time_s",
                "src_index",
                "label",
                "is_ignore",
                "gap_position",
                "gap_length_s",
                "frame_duration_s",
            ]
        ),
    )
    per_video = pd.DataFrame(per_video_rows)

    return grid_frames, per_video


def report_total_grid_frames(per_video: pd.DataFrame) -> None:
    print("\n=== total de quadros da grade (K) vs projeção ingênua ===")
    total_k = int(cast(pd.Series, per_video["k"]).sum())
    naive_k_total = float(cast(pd.Series, per_video["naive_k"]).sum())
    shortfall = naive_k_total - total_k
    print(f"K total (soma do floor por vídeo): {total_k}")
    print(f"projeção ingênua (soma de duration_s * {TARGET_FPS}): {naive_k_total:.2f}")
    print(
        f"déficit explicado pelo floor por vídeo: {shortfall:.2f} quadros "
        f"em {len(per_video)} vídeos"
    )


def report_video_extremes_by_duration(per_video: pd.DataFrame, n: int = 5) -> None:
    print(f"\n=== K vs n_frames_counted — {n} vídeos mais curtos e mais longos ===")
    ordered = cast(pd.DataFrame, per_video.sort_values("duration_s"))

    print(f"mais curtos ({n}):")
    for _, row in ordered.head(n).iterrows():
        print(
            f"  {row['video_id']}: duration_s={row['duration_s']:.2f} "
            f"k={row['k']} n_frames_counted={row['n_frames_counted']}"
        )

    print(f"mais longos ({n}):")
    for _, row in ordered.tail(n).iterrows():
        print(
            f"  {row['video_id']}: duration_s={row['duration_s']:.2f} "
            f"k={row['k']} n_frames_counted={row['n_frames_counted']}"
        )


def report_split_frame_counts(grid_frames: pd.DataFrame) -> None:
    print("\n=== contagem de quadros da grade por split ===")
    print(grid_frames.groupby("split").size())


def report_split_label_frame_counts(grid_frames: pd.DataFrame) -> None:
    print("\n=== contagem de quadros por (split, label) ===")
    print(grid_frames.groupby(["split", "label"]).size())

    print("\npercentuais por split:")
    for split, group in grid_frames.groupby("split"):
        group = cast(pd.DataFrame, group)
        total = len(group)
        label_counts = cast(pd.Series, group["label"]).value_counts().sort_index()
        for label, count in label_counts.items():
            pct = (count / total * 100) if total else 0.0
            print(f"  split={split} label={label}: {count} ({pct:.2f}%)")


def report_label_duration_stats(grid_frames: pd.DataFrame) -> None:
    print(f"\n=== quadros e duração por label, a {TARGET_FPS} fps ===")
    counts = cast(pd.Series, grid_frames.groupby("label").size()).sort_index()
    for label, count in counts.items():
        seconds = count / TARGET_FPS
        print(f"  label={label}: {count} quadros ({seconds:.2f}s)")


def report_overlap_resolved_total(per_video: pd.DataFrame) -> None:
    print("\n=== total de sobreposições resolvidas (earlier-start-wins) ===")
    total = int(cast(pd.Series, per_video["n_overlap_resolved"]).sum())
    print(f"n_overlap_resolved total: {total}")


def report_ignore_fraction(grid_frames: pd.DataFrame) -> None:
    print("\n=== fração de quadros IGNORE_LABEL ===")
    total = len(grid_frames)
    ignore_total = int(cast(pd.Series, grid_frames["is_ignore"]).sum())
    overall_pct = (ignore_total / total * 100) if total else 0.0
    print(f"geral: {ignore_total}/{total} ({overall_pct:.2f}%)")

    print("por split:")
    for split, group in grid_frames.groupby("split"):
        group = cast(pd.DataFrame, group)
        group_total = len(group)
        group_ignore = int(cast(pd.Series, group["is_ignore"]).sum())
        pct = (group_ignore / group_total * 100) if group_total else 0.0
        print(f"  {split}: {group_ignore}/{group_total} ({pct:.2f}%)")

    print("por env:")
    for env, group in grid_frames.groupby("env"):
        group = cast(pd.DataFrame, group)
        group_total = len(group)
        group_ignore = int(cast(pd.Series, group["is_ignore"]).sum())
        pct = (group_ignore / group_total * 100) if group_total else 0.0
        print(f"  {env}: {group_ignore}/{group_total} ({pct:.2f}%)")


def report_ignore_gap_position_breakdown(grid_frames: pd.DataFrame) -> None:
    print("\n=== quadros IGNORE_LABEL por gap_position ===")
    ignore = cast(pd.DataFrame, grid_frames[grid_frames["is_ignore"]])
    total = len(ignore)
    counts = cast(pd.Series, ignore["gap_position"]).value_counts(dropna=False)
    for position, count in counts.items():
        pct = (count / total * 100) if total else 0.0
        print(f"  {position}: {count} ({pct:.2f}%)")
    print(
        "  lembrete — auditoria de cobertura em segundos (docs/data/"
        "manifest-verification.md): leading 66,4% / trailing 26,0% / "
        "interior 7,6%"
    )


def report_subframe_seam_ignore_frames(grid_frames: pd.DataFrame) -> None:
    print("\n=== quadros IGNORE_LABEL cujo gap é menor que um quadro-fonte ===")
    seam = cast(
        pd.DataFrame,
        grid_frames[
            grid_frames["is_ignore"]
            & (grid_frames["gap_length_s"] < grid_frames["frame_duration_s"])
        ],
    )
    print(f"contagem: {len(seam)}")
    for _, row in seam.iterrows():
        print(f"  {row['video_id']}: time_s={row['time_s']:.4f}")


def report_longest_ignore_runs(grid_frames: pd.DataFrame) -> None:
    print("\n=== maior sequência de quadros IGNORE_LABEL consecutivos, por env ===")
    sorted_frames = cast(
        pd.DataFrame, grid_frames.sort_values(["video_id", "frame_index"])
    ).reset_index(drop=True)

    video_id = cast(pd.Series, sorted_frames["video_id"])
    is_ignore = cast(pd.Series, sorted_frames["is_ignore"])
    video_changed = video_id != video_id.shift()
    run_starts = (is_ignore != is_ignore.shift()) | video_changed
    sorted_frames = sorted_frames.assign(_run_id=run_starts.cumsum())

    ignore_frames = cast(pd.DataFrame, sorted_frames[sorted_frames["is_ignore"]])
    if ignore_frames.empty:
        print("nenhum quadro IGNORE_LABEL encontrado")
        return

    run_length_series = cast(
        pd.Series, ignore_frames.groupby(["env", "video_id", "_run_id"]).size()
    )
    run_sizes = cast(
        pd.DataFrame, run_length_series.rename("run_length").reset_index()
    )
    for env, group in run_sizes.groupby("env"):
        longest = cast(pd.DataFrame, group).sort_values(
            "run_length", ascending=False
        ).iloc[0]
        print(
            f"  {env}: {int(longest['run_length'])} quadros consecutivos "
            f"(vídeo {longest['video_id']})"
        )


def report_split_env_crosstab(per_video: pd.DataFrame, grid_frames: pd.DataFrame) -> None:
    print("\n=== crosstab split x env (contagem de vídeos) ===")
    print(pd.crosstab(per_video["split"], per_video["env"]))

    print("\n=== crosstab split x env (contagem de quadros da grade) ===")
    print(pd.crosstab(grid_frames["split"], grid_frames["env"]))


def report_split_subject_crosstab(per_video: pd.DataFrame) -> None:
    print("\n=== crosstab split x subject (contagem de vídeos) ===")
    print(pd.crosstab(per_video["split"], per_video["subject"]))


def report_split_env_list(per_video: pd.DataFrame) -> None:
    print("\n=== ambientes distintos por split ===")
    for split, group in per_video.groupby("split"):
        group = cast(pd.DataFrame, group)
        envs = sorted(cast(pd.Series, group["env"]).unique())
        print(f"  {split}: {envs}")


def report_le2i_timegrid() -> None:
    manifest = load_le2i_manifest()
    splits = load_annotation_splits()
    grid_frames, per_video = build_grid_frames(manifest, splits)

    report_total_grid_frames(per_video)
    report_video_extremes_by_duration(per_video)
    report_split_frame_counts(grid_frames)
    report_split_label_frame_counts(grid_frames)
    report_label_duration_stats(grid_frames)
    report_overlap_resolved_total(per_video)
    report_ignore_fraction(grid_frames)
    report_ignore_gap_position_breakdown(grid_frames)
    report_subframe_seam_ignore_frames(grid_frames)
    report_longest_ignore_runs(grid_frames)
    report_split_env_crosstab(per_video, grid_frames)
    report_split_subject_crosstab(per_video)
    report_split_env_list(per_video)

    print("\ntimegrid report OK: relatório concluído (sem checagens críticas)")
