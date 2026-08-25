"""Constrói e verifica `data/manifest.parquet` para o Le2i.

Casa os paths de vídeo anotados pelo OmniFall (train/val/test do config
`le2i-cs`) com os arquivos locais extraídos de `data/raw/le2i/`, sonda cada
vídeo via ffprobe e grava um manifesto único (`data/manifest.parquet`) com
metadados de vídeo, split e status de extração de features por branch
(pose/DINOv3/SAM), ainda pendentes de execução.

Uso:
    uv run python -m gatefall.data.ingest ingest [--force]
    uv run python -m gatefall.data.ingest verify
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd

from gatefall.hashing import sha256_file

LABELS_DIR = Path("data/labels/omnifall")
SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
RAW_DIR = Path("data/raw/le2i")
MANIFEST_PATH = Path("data/manifest.parquet")
VIDEO_EXT = ".avi"

_VIDEO_NUM_RE = re.compile(r"video\s*\((\d+)\)", re.IGNORECASE)

_MANIFEST_DTYPES: dict[str, str] = {
    "video_id": "string",
    "dataset": "string",
    "relative_path": "string",
    "absolute_path": "string",
    "env": "string",
    "subject": "int64",
    "cam": "int64",
    "split": "string",
    "fps": "float64",
    "fps_source": "string",
    "n_frames_header": "Int64",
    "n_frames_counted": "int64",
    "duration_s": "float64",
    "width": "int64",
    "height": "int64",
    "codec": "string",
    "sha256": "string",
    "pose_status": "string",
    "dino_status": "string",
    "sam_status": "string",
}


def _normalize_env(name: str) -> str:
    """Espaços e underscores são intercambiáveis no nome do ambiente
    ('Lecture room' no disco vs. 'Lecture_room' no CSV); comparação
    case-insensitive."""
    return name.strip().lower().replace(" ", "_")


def _normalize_omnifall_path(path: str) -> str:
    """'Coffee_room_01/video_1' -> 'coffee_room_01/video_1'"""
    env, _, video = path.partition("/")
    return f"{_normalize_env(env)}/{video.strip().lower()}"


def _normalize_local_path(relative_path: Path) -> str:
    """'Coffee_room_01/Videos/video (1).avi' -> 'coffee_room_01/video_1'
    'Office/video (1).avi'                  -> 'office/video_1'"""
    parts = [part for part in relative_path.parts if part.lower() != "videos"]
    env = parts[0]
    filename = parts[-1]
    match = _VIDEO_NUM_RE.search(filename)
    if match is None:
        raise ValueError(
            f"não foi possível extrair o número do vídeo de {relative_path}"
        )
    n = int(match.group(1))
    return f"{_normalize_env(env)}/video_{n}"


def discover_local_videos(raw_dir: Path) -> dict[str, Path]:
    """{chave_normalizada: caminho relativo a raw_dir} para todo *.avi
    (case-insensitive na extensão) sob raw_dir. sys.exit(1) se dois arquivos
    distintos normalizarem para a mesma chave — nunca descarta silenciosamente
    um vídeo por colisão de normalização (ex.: cópia deixada por uma
    reextração parcial)."""
    result: dict[str, Path] = {}
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() != VIDEO_EXT:
            continue
        relative_path = path.relative_to(raw_dir)
        key = _normalize_local_path(relative_path)
        if key in result and result[key] != relative_path:
            print(
                f"erro: colisão de chave normalizada {key!r} entre "
                f"{result[key]} e {relative_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        result[key] = relative_path
    return result


def normalize_label_paths(paths: pd.Series) -> dict[str, str]:
    """{chave_normalizada: path original do CSV} para uma série de paths
    únicos. sys.exit(1) se dois paths distintos normalizarem para a mesma
    chave — mesma garantia de discover_local_videos."""
    result: dict[str, str] = {}
    for path in paths.unique():
        key = _normalize_omnifall_path(path)
        if key in result and result[key] != path:
            print(
                f"erro: colisão de chave normalizada {key!r} entre "
                f"{result[key]!r} e {path!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        result[key] = path
    return result


def _bijection_diff(
    local: dict[str, Path], labels: dict[str, str]
) -> tuple[list[str], list[str]]:
    """(only_local, only_labels): chaves presentes em apenas um dos lados,
    cada lista ordenada."""
    local_keys = set(local)
    label_keys = set(labels)
    return sorted(local_keys - label_keys), sorted(label_keys - local_keys)


def match_bijection(
    local: dict[str, Path], labels: dict[str, str]
) -> dict[str, Path]:
    """Confere bijeção estrita entre as chaves de `local` e `labels`. Se
    houver qualquer chave de um lado sem par no outro, imprime (stderr) TODAS
    as chaves não casadas de ambos os lados e sai com sys.exit(1) — nunca
    descarta silenciosamente uma linha. Em caso de sucesso, retorna
    {path_original_do_csv: caminho_relativo_local} para as 190 entradas."""
    only_local, only_labels = _bijection_diff(local, labels)

    if only_local or only_labels:
        print(
            "erro: bijeção entre vídeos locais e labels do OmniFall falhou.",
            file=sys.stderr,
        )
        if only_local:
            print(
                f"  presentes apenas localmente ({len(only_local)}):",
                file=sys.stderr,
            )
            for key in only_local:
                print(f"    {key}", file=sys.stderr)
        if only_labels:
            print(
                f"  presentes apenas nas labels ({len(only_labels)}):",
                file=sys.stderr,
            )
            for key in only_labels:
                print(f"    {key}", file=sys.stderr)
        sys.exit(1)

    return {labels[key]: local[key] for key in local}


class ProbedVideo(TypedDict):
    r_frame_rate: str
    avg_frame_rate: str
    n_frames_header: int | None
    n_frames_counted: int
    duration_s: float
    width: int
    height: int
    codec: str


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


def _ffprobe_count_frames(path: Path) -> int:
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
    try:
        return int(output)
    except ValueError as exc:
        raise ValueError(
            f"ffprobe não retornou um inteiro parseável para {path}: {output!r}"
        ) from exc


def _resolve_fps(r_frame_rate: str, avg_frame_rate: str) -> tuple[float, str]:
    """Retorna (fps, fps_source: 'avg_frame_rate' | 'r_frame_rate').
    ffprobe expõe fps como fração 'num/den'; fps NUNCA deve ser assumido como
    25 — ambientes Home_01/Home_02 reportam ~23.9997 fps (500000/20833),
    os demais reportam exatamente 25/1. Prefere avg_frame_rate (média real de
    entrega do stream); cai para r_frame_rate se avg_frame_rate vier como
    '0/0' (ffprobe não conseguiu calcular a média) ou reduzir a uma fração de
    valor 0 (ex.: '0/1') — nenhum vídeo real tem fps 0, então esse valor
    também indica média não calculável, não uma média genuína."""

    def _parse_fraction(value: str) -> float | None:
        num_str, _, den_str = value.partition("/")
        try:
            num = float(num_str)
            den = float(den_str) if den_str else 1.0
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den

    avg = _parse_fraction(avg_frame_rate)
    if avg is not None and avg > 0:
        return avg, "avg_frame_rate"

    r = _parse_fraction(r_frame_rate)
    if r is None:
        raise ValueError(
            f"não foi possível calcular fps de r_frame_rate={r_frame_rate!r} "
            f"nem avg_frame_rate={avg_frame_rate!r}"
        )
    return r, "r_frame_rate"


def probe_video(path: Path) -> ProbedVideo:
    header = _ffprobe_header(path)

    raw_streams = header.get("streams")
    stream: dict[str, object] = {}
    if isinstance(raw_streams, list) and raw_streams and isinstance(raw_streams[0], dict):
        stream = raw_streams[0]

    raw_format = header.get("format")
    fmt: dict[str, object] = raw_format if isinstance(raw_format, dict) else {}

    r_frame_rate = str(stream.get("r_frame_rate", "0/0"))
    avg_frame_rate = str(stream.get("avg_frame_rate", "0/0"))

    n_frames_header: int | None
    try:
        n_frames_header = int(str(stream.get("nb_frames")))
    except (TypeError, ValueError):
        n_frames_header = None

    return ProbedVideo(
        r_frame_rate=r_frame_rate,
        avg_frame_rate=avg_frame_rate,
        n_frames_header=n_frames_header,
        n_frames_counted=_ffprobe_count_frames(path),
        duration_s=float(str(fmt.get("duration", "nan"))),
        width=int(str(stream.get("width", 0))),
        height=int(str(stream.get("height", 0))),
        codec=str(stream.get("codec_name", "")),
    )


def load_label_splits() -> dict[str, pd.DataFrame]:
    """Lê train.csv/val.csv/test.csv de LABELS_DIR; sys.exit(1) com mensagem
    clara se algum arquivo não existir."""
    splits: dict[str, pd.DataFrame] = {}
    for split, filename in SPLIT_FILES.items():
        path = LABELS_DIR / filename
        if not path.exists():
            print(
                f"erro: {path} não encontrado. Rode "
                "`uv run python scripts/fetch_labels.py` antes.",
                file=sys.stderr,
            )
            sys.exit(1)
        splits[split] = pd.read_csv(path)
    return splits


def build_label_index(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Uma linha por path único (não por segmento de queda): agrega
    subject/cam/split por path. Cada vídeo pertence a exatamente um
    subject/cam/split — se os segmentos de um mesmo path divergirem nisso,
    é uma violação de invariante."""
    frames: list[pd.DataFrame] = []
    for split, df in splits.items():
        tagged = cast(pd.DataFrame, df[["path", "subject", "cam"]].copy())
        tagged["split"] = split
        frames.append(tagged)
    pooled = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

    rows: list[dict[str, object]] = []
    for path, group in pooled.groupby("path", sort=False):
        subjects = group["subject"].unique()
        cams = group["cam"].unique()
        splits_seen = group["split"].unique()
        if len(subjects) > 1 or len(cams) > 1 or len(splits_seen) > 1:
            raise ValueError(
                f"invariante violado: {path} possui subject/cam/split "
                f"divergentes entre segmentos (subjects={subjects!r}, "
                f"cams={cams!r}, splits={splits_seen!r})"
            )
        rows.append(
            {
                "path": path,
                "subject": int(subjects[0]),
                "cam": int(cams[0]),
                "split": str(splits_seen[0]),
            }
        )
    return pd.DataFrame(rows, columns=["path", "subject", "cam", "split"])


def build_manifest(raw_dir: Path, label_index: pd.DataFrame) -> pd.DataFrame:
    """
    1. discover_local_videos(raw_dir)
    2. normalize_label_paths(label_index['path'])
    3. matched = match_bijection(local, labels) -- falha alto conforme docstring
    4. para cada (label_path, relative_path) em matched: probe_video, monta a linha
    5. retorna DataFrame ordenado por video_id, com o schema do manifesto
    """
    if shutil.which("ffprobe") is None:
        print("erro: ffprobe não encontrado no PATH", file=sys.stderr)
        sys.exit(1)

    local = discover_local_videos(raw_dir)
    labels = normalize_label_paths(cast(pd.Series, label_index["path"]))
    matched = match_bijection(local, labels)

    label_records = label_index.set_index("path").to_dict("index")

    rows: list[dict[str, object]] = []
    for label_path, relative_path in matched.items():
        absolute_path = (raw_dir / relative_path).resolve()
        probed = probe_video(absolute_path)
        fps, fps_source = _resolve_fps(probed["r_frame_rate"], probed["avg_frame_rate"])
        label_row = label_records[label_path]
        env = label_path.split("/", 1)[0]

        rows.append(
            {
                "video_id": _normalize_omnifall_path(label_path),
                "dataset": "le2i",
                "relative_path": str(relative_path),
                "absolute_path": str(absolute_path),
                "env": env,
                "subject": int(label_row["subject"]),
                "cam": int(label_row["cam"]),
                "split": str(label_row["split"]),
                "fps": fps,
                "fps_source": fps_source,
                "n_frames_header": probed["n_frames_header"],
                "n_frames_counted": probed["n_frames_counted"],
                "duration_s": probed["duration_s"],
                "width": probed["width"],
                "height": probed["height"],
                "codec": probed["codec"],
                "sha256": sha256_file(absolute_path),
                "pose_status": "pending",
                "dino_status": "pending",
                "sam_status": "pending",
            }
        )

    df = pd.DataFrame(rows)
    df = cast(pd.DataFrame, df.astype(_MANIFEST_DTYPES))
    df = cast(pd.DataFrame, df[list(_MANIFEST_DTYPES.keys())])
    return cast(pd.DataFrame, df.sort_values("video_id", ignore_index=True))


def write_manifest(df: pd.DataFrame, path: Path, force: bool) -> None:
    """Recusa sobrescrever sem --force (print + return, não é erro), grava em
    path.with_suffix(path.suffix + '.tmp') via df.to_parquet(tmp_path) e
    depois os.replace(tmp_path, path)."""
    if path.exists() and not force:
        print(f"skip {path} (já existe, use --force para sobrescrever)")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path)
    os.replace(tmp_path, path)
    print(f"{path}: {len(df)} linhas, colunas={list(df.columns)}")


def ingest(force: bool) -> None:
    """Orquestra: load_label_splits -> build_label_index -> build_manifest -> write_manifest.
    Recusa cedo (antes de rodar ffprobe/sha256 em todo o dataset) se o
    manifesto já existe e --force não foi passado — mesma condição que
    write_manifest checa antes de gravar."""
    if MANIFEST_PATH.exists() and not force:
        print(f"skip {MANIFEST_PATH} (já existe, use --force para sobrescrever)")
        return

    splits = load_label_splits()
    label_index = build_label_index(splits)
    manifest = build_manifest(RAW_DIR, label_index)
    write_manifest(manifest, MANIFEST_PATH, force)


def _load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        print(
            f"erro: {MANIFEST_PATH} não encontrado. Rode "
            "`uv run python -m gatefall.data.ingest ingest` antes de verificar.",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_parquet(MANIFEST_PATH)


def report_bijection(raw_dir: Path, splits: dict[str, pd.DataFrame]) -> bool:
    print("\n=== bijeção: vídeos locais <-> labels do OmniFall ===")
    pooled_paths = cast(
        pd.Series,
        pd.concat([df["path"] for df in splits.values()], ignore_index=True),
    ).drop_duplicates()
    local = discover_local_videos(raw_dir)
    labels = normalize_label_paths(pooled_paths)

    only_local, only_labels = _bijection_diff(local, labels)
    if only_local or only_labels:
        if only_local:
            print(f"  presentes apenas localmente ({len(only_local)}):")
            for key in only_local:
                print(f"    {key}")
        if only_labels:
            print(f"  presentes apenas nas labels ({len(only_labels)}):")
            for key in only_labels:
                print(f"    {key}")
        return False

    print(f"{len(local)} <-> {len(local)} OK")
    return True


def report_split_disjointness(splits: dict[str, pd.DataFrame]) -> bool:
    print("\n=== disjunção de paths entre splits ===")
    ok = True
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = set(splits[a]["path"]) & set(splits[b]["path"])
            if overlap:
                ok = False
                print(f"{a} x {b}: NÃO disjuntos, overlap={sorted(overlap)}")
    if ok:
        print("OK: splits disjuntos")
    return ok


def report_subject_disjointness(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== disjunção de subjects entre splits ===")
    any_overlap = False
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = set(splits[a]["subject"]) & set(splits[b]["subject"])
            if overlap:
                any_overlap = True
                print(f"{a} x {b}: subjects sobrepostos: {sorted(overlap)}")
    if any_overlap:
        print(
            "conclusão: os conjuntos de subjects se sobrepõem entre splits — "
            "le2i-cs, portanto, não é verdadeiramente cross-subject."
        )
    else:
        print("conclusão: os conjuntos de subjects são disjuntos entre todos os splits.")


def report_resolution_distribution(manifest: pd.DataFrame) -> None:
    print("\n=== distribuição de resolução (width, height) ===")
    counts = cast(pd.Series, manifest.groupby(["width", "height"]).size()).sort_values(
        ascending=False
    )
    print(counts)

    mode_width, mode_height = cast(tuple[int, int], counts.index[0])
    print(f"moda: {mode_width}x{mode_height}")

    outliers = manifest[
        (manifest["width"] != mode_width) | (manifest["height"] != mode_height)
    ]
    if outliers.empty:
        print("nenhum vídeo diverge da resolução moda")
    else:
        print("vídeos com resolução divergente da moda:")
        for _, row in outliers.iterrows():
            print(f"  {row['video_id']}: {row['width']}x{row['height']}")


def report_fps_distribution(manifest: pd.DataFrame) -> None:
    print("\n=== distribuição de fps por ambiente ===")
    for env, group in manifest.groupby("env"):
        print(f"{env}:")
        print(group["fps"].value_counts())


def report_cam_env_crosstab(manifest: pd.DataFrame) -> None:
    print("\n=== crosstab cam x env ===")
    crosstab = pd.crosstab(manifest["cam"], manifest["env"])
    print(crosstab)

    cam_to_envs = manifest.groupby("cam")["env"].nunique()
    env_to_cams = manifest.groupby("env")["cam"].nunique()
    is_bijective = bool((cam_to_envs == 1).all() and (env_to_cams == 1).all())
    if is_bijective:
        print("conclusão: cam é uma função 1:1 de env (e vice-versa).")
    else:
        print("conclusão: cam NÃO é uma função 1:1 de env.")


def report_segment_duration_by_class(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== duração dos segmentos por classe (pooled train+val+test) ===")
    pooled = cast(pd.DataFrame, pd.concat(splits.values(), ignore_index=True)).copy()
    pooled["duration"] = pooled["end"] - pooled["start"]
    stats = pooled.groupby("label")["duration"].agg(
        min="min",
        p25=lambda s: s.quantile(0.25),
        median="median",
        p75=lambda s: s.quantile(0.75),
        max="max",
        count="count",
    )
    print(stats)


def report_segment_counts_per_class_per_split(splits: dict[str, pd.DataFrame]) -> None:
    print("\n=== contagem de segmentos por (split, label) ===")
    frames: list[pd.DataFrame] = []
    for split, df in splits.items():
        tagged = cast(pd.DataFrame, df[["label"]].copy())
        tagged["split"] = split
        frames.append(tagged)
    pooled = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
    counts = cast(pd.Series, pooled.groupby(["split", "label"]).size())
    print(counts)

    all_labels = sorted(pooled["label"].unique())
    for split in splits:
        for label in all_labels:
            n = cast(int, counts.get((split, label), 0))
            if n == 0:
                marker = " (test!)" if split == "test" else ""
                print(f"AVISO: (split={split}, label={label}) tem 0 segmentos{marker}")


def report_projected_frame_counts(manifest: pd.DataFrame) -> None:
    print("\n=== projeção de contagem de frames por fps candidato ===")
    total_duration_s = float(cast(pd.Series, manifest["duration_s"]).sum())
    print(f"duração total: {total_duration_s:.2f}s")
    for fps in (10, 12.5, 25):
        projected = total_duration_s * fps
        print(f"  a {fps} fps: {projected:.0f} frames")


def report_sha256_integrity(manifest: pd.DataFrame) -> bool:
    """Re-hash cada vídeo referenciado pelo manifesto e compara contra o
    sha256 gravado em `ingest`, no mesmo espírito da verificação de
    PROVENANCE.json em scripts/fetch_labels.py — sem isso, a coluna sha256
    do manifesto nunca é de fato usada para detectar corrupção/reextração
    parcial dos vídeos brutos."""
    print("\n=== integridade: sha256 dos vídeos no manifesto ===")
    ok = True
    for _, row in manifest.iterrows():
        absolute_path = Path(str(row["absolute_path"]))
        if not absolute_path.exists():
            ok = False
            print(f"{row['video_id']}: arquivo ausente ({absolute_path})")
            continue
        actual = sha256_file(absolute_path)
        expected = str(row["sha256"])
        if actual != expected:
            ok = False
            print(
                f"{row['video_id']}: sha256 divergente "
                f"(esperado {expected}, encontrado {actual})"
            )
    if ok:
        print("OK: todos os vídeos íntegros")
    return ok


def verify() -> None:
    manifest = _load_manifest()
    splits = load_label_splits()

    bijection_ok = report_bijection(RAW_DIR, splits)
    disjointness_ok = report_split_disjointness(splits)
    report_subject_disjointness(splits)
    report_resolution_distribution(manifest)
    report_fps_distribution(manifest)
    report_cam_env_crosstab(manifest)
    report_segment_duration_by_class(splits)
    report_segment_counts_per_class_per_split(splits)
    report_projected_frame_counts(manifest)
    sha256_ok = report_sha256_integrity(manifest)

    failed = []
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Constrói data/manifest.parquet")
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve o manifesto já existente em vez de pulá-lo",
    )

    subparsers.add_parser("verify", help="Verifica a integridade do manifesto já construído")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.force)
    elif args.command == "verify":
        verify()


if __name__ == "__main__":
    main()
