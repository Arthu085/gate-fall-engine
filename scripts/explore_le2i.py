"""Explora as anotações OmniFall e a árvore extraída do Le2i (script descartável)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

LABELS_DIR = Path("data/labels/omnifall")
RAW_DIR = Path("data/raw/le2i")
SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
VIDEO_EXT = ".avi"
DURATION_THRESHOLD_S = 4.8
N_ITEM1_PATHS = 30
N_ITEM7_PATHS = 30
N_ITEM10_SAMPLES = 5


def _load_splits() -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split, filename in SPLIT_FILES.items():
        path = LABELS_DIR / filename
        if not path.exists():
            print(
                f"erro: {path} não encontrado. Rode scripts/fetch_labels.py antes.",
                file=sys.stderr,
            )
            sys.exit(1)
        splits[split] = pd.read_csv(path)
    return splits


def _check_raw_extracted() -> None:
    if not RAW_DIR.is_dir() or not any(p.is_dir() for p in RAW_DIR.iterdir()):
        print(
            f"erro: nenhum diretório extraído em {RAW_DIR}. "
            "Rode scripts/extract_le2i.py antes.",
            file=sys.stderr,
        )
        sys.exit(1)


def item1_sample_paths(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== item 1: amostra de paths (pool train+val+test) ===")
    pooled = pd.concat([df["path"] for df in splits.values()], ignore_index=True)
    unique_paths = pooled.drop_duplicates().tolist()
    for path in unique_paths[:N_ITEM1_PATHS]:
        print(path)


def item2_path_counts(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== item 2: contagem de paths distintos por split ===")
    pooled = pd.concat([df["path"] for df in splits.values()], ignore_index=True)
    for split, df in splits.items():
        print(f"{split}: {df['path'].nunique()}")
    print(f"pooled total: {pooled.nunique()}")


def item3_disjointness(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== item 3: disjunção entre splits ===")
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            set_a = set(splits[a]["path"])
            set_b = set(splits[b]["path"])
            overlap = set_a & set_b
            if overlap:
                print(f"{a} x {b}: NÃO disjuntos, overlap={sorted(overlap)}")
            else:
                print(f"{a} x {b}: disjuntos")


def item4_subject_cam(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== item 4: subject e cam ===")
    pooled = pd.concat(splits.values(), ignore_index=True)
    print("subject value_counts:")
    print(pooled["subject"].value_counts())
    print("cam value_counts:")
    print(pooled["cam"].value_counts())
    if pooled["subject"].nunique() == 1:
        print("subject é degenerado (apenas um valor único)")
    else:
        print("subject possui IDs reais (mais de um valor único)")


def item5_label_distribution(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== item 5: distribuição de labels por split ===")
    for split, df in splits.items():
        counts = df["label"].value_counts().reindex(range(10), fill_value=0)
        print(f"{split}:")
        print(counts)


def item6_duration_stats(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== item 6: estatísticas de duração ===")
    pooled = pd.concat(splits.values(), ignore_index=True)
    dur = pooled["end"] - pooled["start"]
    print(f"min: {dur.min()}")
    print(f"mediana: {dur.median()}")
    print(f"max: {dur.max()}")
    print(
        f"count com dur < {DURATION_THRESHOLD_S}s: {(dur < DURATION_THRESHOLD_S).sum()}"
    )


def item7_sample_video_paths() -> None:
    print("\n=== item 7: amostra de paths de vídeo na árvore extraída ===")
    video_paths = sorted(
        p.relative_to(RAW_DIR)
        for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() == VIDEO_EXT
    )
    for path in video_paths[:N_ITEM7_PATHS]:
        print(path)


def item8_video_counts_by_dir() -> None:
    print("\n=== item 8: contagem de vídeos por diretório top-level ===")
    video_paths = [
        p.relative_to(RAW_DIR)
        for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() == VIDEO_EXT
    ]
    counts: dict[str, int] = {}
    for path in video_paths:
        top_dir = path.parts[0]
        counts[top_dir] = counts.get(top_dir, 0) + 1
    for top_dir in sorted(counts):
        print(f"{top_dir}: {counts[top_dir]}")
    print(f"total: {len(video_paths)}")


def item9_annotation_dirs() -> None:
    print("\n=== item 9: diretórios de anotação por ambiente ===")
    top_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    for top_dir in top_dirs:
        matches = sorted(
            d for d in top_dir.iterdir() if d.is_dir() and "nnotation" in d.name
        )
        if not matches:
            print(f"{top_dir.name}: nenhuma pasta encontrada")
            continue
        for match in matches:
            n_txt = len(list(match.glob("*.txt")))
            print(f"{top_dir.name} ({match.name}): {n_txt} arquivos .txt")


def _pick_ffprobe_samples() -> list[Path]:
    top_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())[:N_ITEM10_SAMPLES]
    samples: list[Path] = []
    for top_dir in top_dirs:
        videos = sorted(
            p.relative_to(RAW_DIR)
            for p in top_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == VIDEO_EXT
        )
        if videos:
            samples.append(RAW_DIR / videos[0])
    return samples


def _ffprobe_header(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,codec_name:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _ffprobe_count_frames(path: Path) -> str | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()
    return output or None


def item10_ffprobe_samples() -> None:
    print("\n=== item 10: amostras via ffprobe ===")
    if shutil.which("ffprobe") is None:
        print("erro: ffprobe não encontrado no PATH", file=sys.stderr)
        sys.exit(1)

    for path in _pick_ffprobe_samples():
        header = _ffprobe_header(path)
        streams = header.get("streams")
        stream = streams[0] if isinstance(streams, list) and streams else {}
        fmt = header.get("format") if isinstance(header.get("format"), dict) else {}

        r_frame_rate = stream.get("r_frame_rate", "N/A")
        avg_frame_rate = stream.get("avg_frame_rate", "N/A")
        header_nb_frames = stream.get("nb_frames", "N/A")
        width = stream.get("width", "N/A")
        height = stream.get("height", "N/A")
        codec_name = stream.get("codec_name", "N/A")
        duration = fmt.get("duration", "N/A") if isinstance(fmt, dict) else "N/A"

        counted = _ffprobe_count_frames(path)

        flag = "não comparável"
        try:
            header_int = int(str(header_nb_frames))
            counted_int = int(str(counted))
            flag = "OK" if header_int == counted_int else "DIVERGE"
        except (TypeError, ValueError):
            pass

        print(f"{path.relative_to(RAW_DIR)}:")
        print(f"  r_frame_rate={r_frame_rate} avg_frame_rate={avg_frame_rate}")
        print(f"  header nb_frames={header_nb_frames} contado={counted}")
        print(f"  duration={duration} width={width} height={height} codec={codec_name}")
        print(f"  flag: {flag}")


def main() -> None:
    splits = _load_splits()
    item1_sample_paths(splits)
    item2_path_counts(splits)
    item3_disjointness(splits)
    item4_subject_cam(splits)
    item5_label_distribution(splits)
    item6_duration_stats(splits)

    _check_raw_extracted()
    item7_sample_video_paths()
    item8_video_counts_by_dir()
    item9_annotation_dirs()

    item10_ffprobe_samples()


if __name__ == "__main__":
    main()
