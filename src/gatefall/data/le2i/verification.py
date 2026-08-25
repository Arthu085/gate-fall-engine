"""Relatórios e verificações críticas do manifesto Le2i."""

import sys
from pathlib import Path
from typing import cast

import pandas as pd

from gatefall.data.le2i.annotations import load_annotation_splits
from gatefall.data.le2i.manifest import MANIFEST_PATH, RAW_DIR
from gatefall.data.le2i.path_matching import (
    discover_extracted_videos,
    find_unmatched_video_keys,
    index_annotation_paths,
)
from gatefall.data.manifest import read_manifest
from gatefall.hashing import sha256_file


def load_le2i_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        print(
            f"erro: {MANIFEST_PATH} não encontrado. Rode "
            "`uv run python -m gatefall.data.ingest ingest` antes de verificar.",
            file=sys.stderr,
        )
        sys.exit(1)
    return read_manifest(MANIFEST_PATH)


def report_bijection(
    raw_directory: Path, splits: dict[str, pd.DataFrame]
) -> bool:
    print("\n=== bijeção: vídeos locais <-> labels do OmniFall ===")
    pooled_paths = cast(
        pd.Series,
        pd.concat(
            [dataframe["path"] for dataframe in splits.values()],
            ignore_index=True,
        ),
    ).drop_duplicates()
    local_videos = discover_extracted_videos(raw_directory)
    annotation_paths = index_annotation_paths(pooled_paths)

    only_local, only_annotations = find_unmatched_video_keys(
        local_videos, annotation_paths
    )
    if only_local or only_annotations:
        if only_local:
            print(f"  presentes apenas localmente ({len(only_local)}):")
            for key in only_local:
                print(f"    {key}")
        if only_annotations:
            print(f"  presentes apenas nas labels ({len(only_annotations)}):")
            for key in only_annotations:
                print(f"    {key}")
        return False

    print(f"{len(local_videos)} <-> {len(local_videos)} OK")
    return True


def report_split_disjointness(splits: dict[str, pd.DataFrame]) -> bool:
    print("\n=== disjunção de paths entre splits ===")
    is_disjoint = True
    split_names = list(splits)
    for index, first_name in enumerate(split_names):
        for second_name in split_names[index + 1 :]:
            overlap = set(splits[first_name]["path"]) & set(
                splits[second_name]["path"]
            )
            if overlap:
                is_disjoint = False
                print(
                    f"{first_name} x {second_name}: NÃO disjuntos, "
                    f"overlap={sorted(overlap)}"
                )
    if is_disjoint:
        print("OK: splits disjuntos")
    return is_disjoint


def report_subject_disjointness(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== disjunção de subjects entre splits ===")
    any_overlap = False
    split_names = list(splits)
    for index, first_name in enumerate(split_names):
        for second_name in split_names[index + 1 :]:
            overlap = set(splits[first_name]["subject"]) & set(
                splits[second_name]["subject"]
            )
            if overlap:
                any_overlap = True
                print(
                    f"{first_name} x {second_name}: subjects sobrepostos: "
                    f"{sorted(overlap)}"
                )
    if any_overlap:
        print(
            "conclusão: os conjuntos de subjects se sobrepõem entre splits — "
            "le2i-cs, portanto, não é verdadeiramente cross-subject."
        )
    else:
        print(
            "conclusão: os conjuntos de subjects são disjuntos entre todos os splits."
        )


def report_resolution_distribution(manifest: pd.DataFrame) -> None:
    print("\n=== distribuição de resolução (width, height) ===")
    counts = cast(
        pd.Series, manifest.groupby(["width", "height"]).size()
    ).sort_values(ascending=False)
    print(counts)

    mode_width, mode_height = cast(tuple[int, int], counts.index[0])
    print(f"moda: {mode_width}x{mode_height}")

    outliers = manifest[
        (manifest["width"] != mode_width)
        | (manifest["height"] != mode_height)
    ]
    if outliers.empty:
        print("nenhum vídeo diverge da resolução moda")
    else:
        print("vídeos com resolução divergente da moda:")
        for _, row in outliers.iterrows():
            print(f"  {row['video_id']}: {row['width']}x{row['height']}")


def report_fps_distribution(manifest: pd.DataFrame) -> None:
    print("\n=== distribuição de fps por ambiente ===")
    for environment, group in manifest.groupby("env"):
        print(f"{environment}:")
        print(group["fps"].value_counts())


def report_camera_environment_crosstab(manifest: pd.DataFrame) -> None:
    print("\n=== crosstab cam x env ===")
    crosstab = pd.crosstab(manifest["cam"], manifest["env"])
    print(crosstab)

    camera_to_environments = manifest.groupby("cam")["env"].nunique()
    environment_to_cameras = manifest.groupby("env")["cam"].nunique()
    is_bijective = bool(
        (camera_to_environments == 1).all()
        and (environment_to_cameras == 1).all()
    )
    if is_bijective:
        print("conclusão: cam é uma função 1:1 de env (e vice-versa).")
    else:
        print("conclusão: cam NÃO é uma função 1:1 de env.")


def report_segment_duration_by_class(
    splits: dict[str, pd.DataFrame],
) -> None:
    print("\n=== duração dos segmentos por classe (pooled train+val+test) ===")
    pooled = cast(
        pd.DataFrame, pd.concat(splits.values(), ignore_index=True)
    ).copy()
    pooled["duration"] = pooled["end"] - pooled["start"]
    stats = pooled.groupby("label")["duration"].agg(
        min="min",
        p25=lambda series: series.quantile(0.25),
        median="median",
        p75=lambda series: series.quantile(0.75),
        max="max",
        count="count",
    )
    print(stats)


def report_segment_counts_per_class_per_split(
    splits: dict[str, pd.DataFrame],
) -> None:
    print("\n=== contagem de segmentos por (split, label) ===")
    frames: list[pd.DataFrame] = []
    for split, dataframe in splits.items():
        tagged = cast(pd.DataFrame, dataframe[["label"]].copy())
        tagged["split"] = split
        frames.append(tagged)
    pooled = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
    counts = cast(pd.Series, pooled.groupby(["split", "label"]).size())
    print(counts)

    all_labels = sorted(pooled["label"].unique())
    for split in splits:
        for label in all_labels:
            count = cast(int, counts.get((split, label), 0))
            if count == 0:
                marker = " (test!)" if split == "test" else ""
                print(
                    f"AVISO: (split={split}, label={label}) tem 0 segmentos{marker}"
                )


def report_projected_frame_counts(manifest: pd.DataFrame) -> None:
    print("\n=== projeção de contagem de frames por fps candidato ===")
    total_duration_s = float(cast(pd.Series, manifest["duration_s"]).sum())
    print(f"duração total: {total_duration_s:.2f}s")
    for fps in (10, 12.5, 25):
        projected = total_duration_s * fps
        print(f"  a {fps} fps: {projected:.0f} frames")


def report_sha256_integrity(manifest: pd.DataFrame) -> bool:
    print("\n=== integridade: sha256 dos vídeos no manifesto ===")
    is_valid = True
    for _, row in manifest.iterrows():
        absolute_path = Path(str(row["absolute_path"]))
        if not absolute_path.exists():
            is_valid = False
            print(f"{row['video_id']}: arquivo ausente ({absolute_path})")
            continue
        actual = sha256_file(absolute_path)
        expected = str(row["sha256"])
        if actual != expected:
            is_valid = False
            print(
                f"{row['video_id']}: sha256 divergente "
                f"(esperado {expected}, encontrado {actual})"
            )
    if is_valid:
        print("OK: todos os vídeos íntegros")
    return is_valid


def verify_le2i_manifest() -> None:
    manifest = load_le2i_manifest()
    splits = load_annotation_splits()

    bijection_ok = report_bijection(RAW_DIR, splits)
    disjointness_ok = report_split_disjointness(splits)
    report_subject_disjointness(splits)
    report_resolution_distribution(manifest)
    report_fps_distribution(manifest)
    report_camera_environment_crosstab(manifest)
    report_segment_duration_by_class(splits)
    report_segment_counts_per_class_per_split(splits)
    report_projected_frame_counts(manifest)
    sha256_ok = report_sha256_integrity(manifest)

    failed: list[str] = []
    if not bijection_ok:
        failed.append("bijeção local <-> labels")
    if not disjointness_ok:
        failed.append("disjunção de splits")
    if not sha256_ok:
        failed.append("integridade sha256")

    if failed:
        print(f"\nverify FALHOU: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print("\nverify OK: todas as verificações críticas passaram")
