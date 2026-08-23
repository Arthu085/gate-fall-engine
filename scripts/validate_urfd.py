#!/usr/bin/env python3
"""Valida a integridade do URF Fall Detection Dataset (câmera 0) já baixado em
data/urfd/: abre cada um dos 70 vídeos, cruza os CSVs de labels contra os
frames de cada vídeo, detecta lacunas de frame_idx e anomalias de rótulo, e
grava um relatório em data/urfd/validation_report.csv.

Este é um validador descritivo: apenas falha de vídeo ausente/não abrível ou
CSV que não parseia é tratada como erro fatal (exit != 0). Anomalias de
conteúdo (lacunas, rótulos fora do esperado) são reportadas, não bloqueiam.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
URFD_DIR = REPO_ROOT / "data" / "urfd"
LABELS_DIR = URFD_DIR / "labels"
VIDEOS_DIR = URFD_DIR / "videos"
INSPECT_DIR = URFD_DIR / "inspect"
REPORT_PATH = URFD_DIR / "validation_report.csv"

LABEL_COLUMNS = [
    "sequence",
    "frame_idx",
    "label",
    "feat_hw_ratio",
    "feat_maj_min_ratio",
    "feat_bbox_occ",
    "feat_max_std_xz",
    "feat_hh_max_ratio",
    "feat_h",
    "feat_d",
    "feat_p40",
]

# Colunas D-K do CSV do URFD: descritores derivados do sensor de profundidade
# Kinect. Carregadas apenas para registrar presença/exclusão — nunca usadas em
# valor, pois o pipeline do GateFall é RGB monocular (ver CLAUDE.md).
DEPTH_FEATURE_COLUMNS = LABEL_COLUMNS[3:]

VALID_LABEL_VALUES = {-1, 0, 1}


@dataclass
class VideoStats:
    exists: bool
    opens: bool
    frame_count: int | None
    fps: float | None
    width: int | None
    height: int | None


@dataclass
class SequenceReport:
    sequence: str
    class_: str
    video_exists: bool
    video_opens: bool
    mp4_frame_count: int | None
    fps: float | None
    width: int | None
    height: int | None
    csv_row_count: int
    csv_first_idx: int
    csv_last_idx: int
    csv_exceeds_mp4: bool
    missing_before_first: str
    missing_internal: str
    label_values: str
    label_invalid_count: int
    fall_missing_0: bool
    fall_missing_1: bool
    adl_label_violation: bool
    label_order_violation: bool
    has_gap: bool


def load_label_tables() -> pd.DataFrame:
    falls = pd.read_csv(LABELS_DIR / "urfall-cam0-falls.csv", header=None, names=LABEL_COLUMNS)
    falls["class_"] = "fall"
    adls = pd.read_csv(LABELS_DIR / "urfall-cam0-adls.csv", header=None, names=LABEL_COLUMNS)
    adls["class_"] = "adl"
    return pd.concat([falls, adls], ignore_index=True)


def video_path_for(sequence: str, class_: str) -> Path:
    return VIDEOS_DIR / class_ / f"{sequence}-cam0.mp4"


def probe_video(path: Path) -> VideoStats:
    if not path.exists():
        return VideoStats(exists=False, opens=False, frame_count=None, fps=None, width=None, height=None)

    cap = cv2.VideoCapture(str(path))
    opens = cap.isOpened()
    if not opens:
        cap.release()
        return VideoStats(exists=True, opens=False, frame_count=None, fps=None, width=None, height=None)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoStats(exists=True, opens=True, frame_count=frame_count, fps=fps, width=width, height=height)


def resolve_index_base(df: pd.DataFrame, video_stats: dict[str, VideoStats]) -> tuple[int, str]:
    global_min = int(df["frame_idx"].min())

    agree_1based = 0
    agree_0based = 0
    total_checked = 0
    for sequence, group in df.groupby("sequence"):
        stats = video_stats.get(sequence)
        if stats is None or not stats.opens or stats.frame_count is None:
            continue
        csv_last_idx = int(group["frame_idx"].max())
        total_checked += 1
        if csv_last_idx == stats.frame_count:
            agree_1based += 1
        if csv_last_idx == stats.frame_count - 1:
            agree_0based += 1

    rate_1based = agree_1based / total_checked if total_checked else 0.0
    rate_0based = agree_0based / total_checked if total_checked else 0.0
    base = 1 if rate_1based >= rate_0based else 0

    evidence = (
        f"Menor frame_idx observado em todo o dataset: {global_min} "
        f"(sinal inicial — {'sugere base 1-based' if global_min >= 1 else 'sugere base 0-based'}).\n"
        f"Cruzamento com contagem de frames do vídeo (csv_last_idx vs. mp4_frame_count), "
        f"em {total_checked} sequências com vídeo abrível:\n"
        f"  hipótese 1-based (csv_last_idx == mp4_frame_count): "
        f"{agree_1based}/{total_checked} ({rate_1based:.1%})\n"
        f"  hipótese 0-based (csv_last_idx == mp4_frame_count - 1): "
        f"{agree_0based}/{total_checked} ({rate_0based:.1%})\n"
        f"Base escolhida: {base}-based."
    )
    return base, evidence


def compact_ranges(indices: list[int]) -> str:
    if not indices:
        return ""

    idxs = sorted(set(indices))
    ranges: list[tuple[int, int]] = []
    start = prev = idxs[0]
    for idx in idxs[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))

    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


def _missing_frames(rows: pd.DataFrame, base: int) -> tuple[list[int], list[int]]:
    present = set(int(x) for x in rows["frame_idx"])
    first_idx = min(present)
    last_idx = max(present)
    missing_before_first = [i for i in range(base, first_idx) if i not in present]
    missing_internal = [i for i in range(first_idx, last_idx + 1) if i not in present]
    return missing_before_first, missing_internal


def analyze_sequence(
    name: str,
    class_: str,
    rows: pd.DataFrame,
    video_stats: VideoStats,
    base: int,
) -> SequenceReport:
    rows_sorted = rows.sort_values("frame_idx")

    csv_row_count = len(rows_sorted)
    csv_first_idx = int(rows_sorted["frame_idx"].min())
    csv_last_idx = int(rows_sorted["frame_idx"].max())

    missing_before_first, missing_internal = _missing_frames(rows_sorted, base)
    has_gap = bool(missing_before_first) or bool(missing_internal)

    csv_exceeds_mp4 = False
    if video_stats.opens and video_stats.frame_count is not None:
        last_idx_0based = csv_last_idx - base
        csv_exceeds_mp4 = last_idx_0based >= video_stats.frame_count

    label_values = sorted(int(v) for v in rows_sorted["label"].unique())
    label_invalid_count = int((~rows_sorted["label"].isin(VALID_LABEL_VALUES)).sum())

    fall_missing_0 = False
    fall_missing_1 = False
    adl_label_violation = False
    label_order_violation = False

    if class_ == "fall":
        fall_missing_0 = 0 not in label_values
        fall_missing_1 = 1 not in label_values
        ordered_labels = rows_sorted["label"].tolist()
        # -1 < 0 < 1 já corresponde à ordem semântica esperada (fora do chão →
        # em queda → caído), então checar não-decrescência basta.
        label_order_violation = any(a > b for a, b in zip(ordered_labels, ordered_labels[1:]))
    elif class_ == "adl":
        adl_label_violation = bool((rows_sorted["label"] != -1).any())

    return SequenceReport(
        sequence=name,
        class_=class_,
        video_exists=video_stats.exists,
        video_opens=video_stats.opens,
        mp4_frame_count=video_stats.frame_count,
        fps=video_stats.fps,
        width=video_stats.width,
        height=video_stats.height,
        csv_row_count=csv_row_count,
        csv_first_idx=csv_first_idx,
        csv_last_idx=csv_last_idx,
        csv_exceeds_mp4=csv_exceeds_mp4,
        missing_before_first=compact_ranges(missing_before_first),
        missing_internal=compact_ranges(missing_internal),
        label_values=",".join(str(v) for v in label_values),
        label_invalid_count=label_invalid_count,
        fall_missing_0=fall_missing_0,
        fall_missing_1=fall_missing_1,
        adl_label_violation=adl_label_violation,
        label_order_violation=label_order_violation,
        has_gap=has_gap,
    )


def extract_inspection_frames(
    video_path: Path,
    sequence: str,
    missing: list[int],
    first_present: int,
    base: int,
) -> None:
    if not missing or not video_path.exists():
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"AVISO: não foi possível abrir {video_path} para extrair frames de inspeção de {sequence}")
        return

    targets = {
        "first_missing": missing[0],
        "last_missing": missing[-1],
        "first_present": first_present,
    }

    out_dir = INSPECT_DIR / sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, frame_idx in targets.items():
        pos = frame_idx - base
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            print(f"AVISO: falha ao ler frame {frame_idx} (pos={pos}) de {sequence} para '{label}'")
            continue
        cv2.imwrite(str(out_dir / f"{label}_{frame_idx}.jpg"), frame)

    cap.release()


def write_csv_report(reports: list[SequenceReport]) -> None:
    fieldnames = [f.name for f in fields(SequenceReport)]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow(asdict(report))


def _sequences_with(reports: list[SequenceReport], predicate) -> list[str]:
    return [r.sequence for r in reports if predicate(r)]


def print_stdout_report(reports: list[SequenceReport], base_evidence: str) -> None:
    total = len(reports)
    videos_opened = sum(1 for r in reports if r.video_opens)
    videos_failed = total - videos_opened

    print("=" * 78)
    print("RELATÓRIO DE VALIDAÇÃO — URF Fall Detection Dataset (câmera 0)")
    print("=" * 78)
    print(
        f"Colunas D-K carregadas apenas para registro de presença (derivadas do "
        f"sensor de profundidade Kinect, excluídas do pipeline RGB monocular): "
        f"{', '.join(DEPTH_FEATURE_COLUMNS)}"
    )
    print(f"Sequências encontradas nos CSVs de labels: {total}/70")
    print(f"Vídeos abertos com sucesso: {videos_opened}/{total}")
    if videos_failed:
        failing = _sequences_with(reports, lambda r: not r.video_opens)
        print(f"Vídeos ausentes ou não abríveis ({videos_failed}): {', '.join(failing)}")

    print()
    print("--- Base de indexação de frame_idx (0-based vs. 1-based) ---")
    print(base_evidence)

    print()
    print("--- Anomalias agregadas ---")

    exceeds_mp4 = _sequences_with(reports, lambda r: r.csv_exceeds_mp4)
    print(f"Sequências com csv_last_idx (0-based) >= mp4_frame_count: {len(exceeds_mp4)}")
    if exceeds_mp4:
        print(f"  {', '.join(exceeds_mp4)}")

    invalid_label = _sequences_with(reports, lambda r: r.label_invalid_count > 0)
    print(f"Sequências com valores de label fora de {{-1, 0, 1}}: {len(invalid_label)}")
    if invalid_label:
        print(f"  {', '.join(invalid_label)}")

    fall_missing_0 = _sequences_with(reports, lambda r: r.fall_missing_0)
    print(f"Sequências fall-* sem nenhum label 0: {len(fall_missing_0)}")
    if fall_missing_0:
        print(f"  {', '.join(fall_missing_0)}")

    fall_missing_1 = _sequences_with(reports, lambda r: r.fall_missing_1)
    print(f"Sequências fall-* sem nenhum label 1: {len(fall_missing_1)}")
    if fall_missing_1:
        print(f"  {', '.join(fall_missing_1)}")

    label_order_violation = _sequences_with(reports, lambda r: r.label_order_violation)
    print(f"Sequências fall-* com ordem de labels não-monotônica: {len(label_order_violation)}")
    if label_order_violation:
        print(f"  {', '.join(label_order_violation)}")

    adl_violation = _sequences_with(reports, lambda r: r.adl_label_violation)
    print(f"Sequências adl-* com algum label != -1: {len(adl_violation)}")
    if adl_violation:
        print(f"  {', '.join(adl_violation)}")

    print()
    print("--- Hipótese de frames ausentes ---")
    leading = [r for r in reports if r.missing_before_first]
    internal = [r for r in reports if r.missing_internal]
    print(f"Sequências com lacuna antes do primeiro frame do CSV: {len(leading)}")
    print(f"Sequências com lacuna interna: {len(internal)}")

    def gap_stats(rs: list[SequenceReport], field_name: str) -> str:
        sizes = []
        for r in rs:
            value = getattr(r, field_name)
            for part in value.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    sizes.append(int(b) - int(a) + 1)
                else:
                    sizes.append(1)
        if not sizes:
            return "sem lacunas"
        return f"min={min(sizes)}, max={max(sizes)}, média={sum(sizes) / len(sizes):.1f}"

    print(f"Tamanho das lacunas iniciais: {gap_stats(leading, 'missing_before_first')}")
    print(f"Tamanho das lacunas internas: {gap_stats(internal, 'missing_internal')}")
    print("=" * 78)


def main() -> int:
    try:
        df = load_label_tables()
    except Exception as exc:  # noqa: BLE001 — falha de parsing é fatal e reportada ao usuário
        print(f"ERRO: falha ao ler os CSVs de labels: {exc}", file=sys.stderr)
        return 1

    seq_class = df.drop_duplicates("sequence").set_index("sequence")["class_"].to_dict()

    video_stats = {
        sequence: probe_video(video_path_for(sequence, class_)) for sequence, class_ in seq_class.items()
    }

    base, evidence = resolve_index_base(df, video_stats)

    reports: list[SequenceReport] = []
    for sequence in sorted(seq_class):
        class_ = seq_class[sequence]
        rows = df[df["sequence"] == sequence]
        stats = video_stats[sequence]

        report = analyze_sequence(sequence, class_, rows, stats, base)
        reports.append(report)

        if report.has_gap:
            missing_before, missing_internal = _missing_frames(rows.sort_values("frame_idx"), base)
            missing_all = sorted(missing_before + missing_internal)
            first_present = int(rows["frame_idx"].min())
            extract_inspection_frames(
                video_path_for(sequence, class_), sequence, missing_all, first_present, base
            )

    write_csv_report(reports)
    print_stdout_report(reports, evidence)

    structural_failure = any(not r.video_exists or not r.video_opens for r in reports)
    return 1 if structural_failure else 0


if __name__ == "__main__":
    sys.exit(main())
