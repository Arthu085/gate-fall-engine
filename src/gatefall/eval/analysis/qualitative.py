"""Diagnóstico qualitativo: renderiza quadros reais nos gatilhos de alarme detectados.

Ferramenta independente de estágio, deliberadamente fora do pipeline padrão
(`gatefall.pipeline`). O subcomando `render` fica fora da suíte de selftests da
CI — depende de vídeo bruto decodificado, que a CI não tem; `selftest` roda na
CI normalmente, pois é totalmente sintético. Lê apenas artefatos já publicados
do run (`config.yaml`, `alarm_protocol.yaml`, `event_metrics.json`) e nunca
escreve ou toca no lock/journal de `gatefall.eval.baseline_a.cli`.
"""

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from gatefall.config import EVAL_STRIDE
from gatefall.data.pose_dataset import PoseWindowDataset

# Este módulo nunca abre vídeo diretamente: toda decodificação passa por
# gatefall.data.video_io.decode_frames (ffmpeg-pipe). Desenho/codificação de
# imagem usam Pillow, não OpenCV — cv2.VideoCapture trava em AVIs brutos do
# Le2i (ver docstring de video_io.py).
from gatefall.data.video_io import decode_frames
from gatefall.datasets import get_dataset
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, AlarmProtocol, load_alarm_protocol
from gatefall.eval.shared.events import (
    Alarm,
    FallEvent,
    associate_events_and_alarms,
    detect_alarms_for_video,
    fall_events_for_video,
)
from gatefall.features.standardization import (
    StandardizationStats,
    apply_standardization,
    load_stats,
    validate_stats_layout,
)
from gatefall.hashing import sha256_file
from gatefall.pose.kinematics import (
    COCO17_SKELETON_EDGES,
    build_pose_features,
)
from gatefall.pose.loading import PoseArrays, load_pose
from gatefall.runs import validate_local_run_dir
from gatefall.train.baseline_a.artifacts import load_compatible_checkpoint, validate_training_run
from gatefall.train.baseline_a.config import BASELINE_A_CONFIG, TrainConfig
from gatefall.train.shared.tcn import TCNClassifier

RUN_DIR = Path("runs/local/le2i/baseline_a")

COLOR_SKELETON = (0, 255, 0)
COLOR_KEYPOINT = (255, 0, 0)
COLOR_BBOX = (255, 255, 0)
COLOR_CAPTION = (255, 255, 255)


@dataclass(frozen=True)
class RenderTarget:
    video_id: str
    trigger_k: int
    src_index: int
    time_s: float
    predicted_label: int
    latency_s: float | None
    is_false_alarm: bool = False


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


def _matched_alarm(event: FallEvent, alarms: list[Alarm]) -> Alarm | None:
    matches = [
        alarm
        for alarm in alarms
        if alarm.video_id == event.video_id
        and event.start_time_s <= alarm.trigger_time_s <= event.association_end_time_s
    ]
    if not matches:
        return None
    return min(matches, key=lambda alarm: alarm.trigger_time_s)


def _build_frame_lookup(frames: pd.DataFrame) -> dict[tuple[str, int], tuple[int, float]]:
    lookup: dict[tuple[str, int], tuple[int, float]] = {}
    for video_id, frame_index, src_index, time_s in zip(
        frames["video_id"], frames["frame_index"], frames["src_index"], frames["time_s"]
    ):
        lookup[(str(video_id), int(frame_index))] = (int(src_index), float(time_s))
    return lookup


def _collect_render_targets(
    video_ids: list[str],
    k_ends: list[int],
    true_labels: list[int],
    pred_labels: list[int],
    protocol: AlarmProtocol,
    frame_lookup: dict[tuple[str, int], tuple[int, float]],
    include_false_alarms: bool = False,
) -> tuple[list[RenderTarget], int]:
    grouped: dict[str, list[int]] = {}
    for index, video_id in enumerate(video_ids):
        grouped.setdefault(video_id, []).append(index)

    all_events: list[FallEvent] = []
    all_alarms: list[Alarm] = []
    pred_by_video_k: dict[tuple[str, int], int] = {}

    for video_id, indices in grouped.items():
        order = sorted(indices, key=lambda i: k_ends[i])
        video_k_ends = np.array([k_ends[i] for i in order], dtype=np.int64)
        video_true = np.array([true_labels[i] for i in order], dtype=np.int64)
        video_pred = np.array([pred_labels[i] for i in order], dtype=np.int64)

        all_events.extend(fall_events_for_video(video_id, video_k_ends, video_true, protocol))
        all_alarms.extend(detect_alarms_for_video(video_id, video_k_ends, video_pred, protocol))

        for k_end, pred_label in zip(video_k_ends.tolist(), video_pred.tolist()):
            pred_by_video_k[(video_id, k_end)] = pred_label

    outcomes, false_alarms = associate_events_and_alarms(all_events, all_alarms, protocol)

    targets: list[RenderTarget] = []
    for outcome in outcomes:
        if not outcome.detected:
            continue
        alarm = _matched_alarm(outcome.event, all_alarms)
        if alarm is None:
            raise RuntimeError(
                f"evento detectado sem alarme correspondente: video_id="
                f"{outcome.event.video_id!r}"
            )
        src_index, time_s = frame_lookup[(alarm.video_id, alarm.trigger_k)]
        if abs(time_s - alarm.trigger_time_s) > 1e-6:
            raise AssertionError(
                f"time_s da tabela de frames ({time_s}) diverge de "
                f"alarm.trigger_time_s ({alarm.trigger_time_s}) para "
                f"video_id={alarm.video_id!r}, trigger_k={alarm.trigger_k}"
            )
        assert outcome.latency_s is not None
        targets.append(
            RenderTarget(
                video_id=alarm.video_id,
                trigger_k=alarm.trigger_k,
                src_index=src_index,
                time_s=time_s,
                predicted_label=pred_by_video_k[(alarm.video_id, alarm.trigger_k)],
                latency_s=outcome.latency_s,
            )
        )

    if include_false_alarms:
        for alarm in false_alarms:
            src_index, time_s = frame_lookup[(alarm.video_id, alarm.trigger_k)]
            targets.append(
                RenderTarget(
                    video_id=alarm.video_id,
                    trigger_k=alarm.trigger_k,
                    src_index=src_index,
                    time_s=time_s,
                    predicted_label=pred_by_video_k[(alarm.video_id, alarm.trigger_k)],
                    latency_s=None,
                    is_false_alarm=True,
                )
            )

    n_detected = sum(1 for outcome in outcomes if outcome.detected)
    return targets, n_detected


def _caption_text(
    video_id: str,
    trigger_k: int,
    time_s: float,
    label_name: str,
    latency_s: float | None,
    imputed: bool,
    is_false_alarm: bool = False,
) -> str:
    if is_false_alarm:
        base = (
            f"{video_id} k={trigger_k} t={time_s:.1f}s pred={label_name} "
            f"(ALARME FALSO)"
        )
    else:
        assert latency_s is not None
        base = (
            f"{video_id} k={trigger_k} t={time_s:.1f}s pred={label_name} "
            f"latencia={latency_s:.1f}s"
        )
    if imputed:
        return f"{base} (pose imputada)"
    return base


def _draw_alarm_frame(
    frame_rgb: np.ndarray,
    pose: PoseArrays,
    k: int,
    caption: str,
    imputed: bool,
) -> np.ndarray:
    image = Image.fromarray(frame_rgb, mode="RGB")
    draw = ImageDraw.Draw(image)

    if not imputed:
        keypoints = pose.keypoints[k]
        bbox = pose.bbox[k]

        for start, end in COCO17_SKELETON_EDGES:
            start_point = (float(keypoints[start, 0]), float(keypoints[start, 1]))
            end_point = (float(keypoints[end, 0]), float(keypoints[end, 1]))
            draw.line([start_point, end_point], fill=COLOR_SKELETON, width=2)

        for index in range(17):
            x = float(keypoints[index, 0])
            y = float(keypoints[index, 1])
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=COLOR_KEYPOINT)

        top_left = (float(bbox[0]), float(bbox[1]))
        bottom_right = (float(bbox[2]), float(bbox[3]))
        draw.rectangle([top_left, bottom_right], outline=COLOR_BBOX, width=2)

    caption_margin_px = 10
    max_caption_width = image.width - 2 * caption_margin_px
    font = _fit_caption_font(caption, max_caption_width)
    _, top, _, bottom = draw.textbbox((0, 0), caption, font=font)
    caption_height = bottom - top
    draw.text(
        (caption_margin_px, image.height - caption_margin_px - caption_height),
        caption,
        fill=COLOR_CAPTION,
        font=font,
    )
    return np.array(image)


CAPTION_FONT_SIZE_DEFAULT = 16
# PIL.ImageFont.load_default(size=N) rasteriza espaços de forma inconsistente
# em tamanhos muito pequenos: alguns espaços (ex.: entre dígito/underscore e
# a letra seguinte) colapsam visualmente a um espaçamento quase nulo, mesmo
# com font.getlength() reportando um avanço não nulo. Verificado empiricamente
# renderizando legendas reais do Le2i (320px): tamanho 11 colapsa
# "k=64 t=6.4s" em "k=64t=6.4s"; tamanho 14 mantém todos os espaços visíveis
# nas legendas reais testadas. Não reduzir sem reverificar visualmente.
CAPTION_FONT_SIZE_MIN = 14


def _fit_caption_font(
    caption: str, max_width_px: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_size = CAPTION_FONT_SIZE_DEFAULT
    font = ImageFont.load_default(size=font_size)
    text_width = font.getlength(caption)
    if text_width <= max_width_px or max_width_px <= 0:
        return font

    font_size = max(int(font_size * (max_width_px / text_width)), CAPTION_FONT_SIZE_MIN)
    font = ImageFont.load_default(size=font_size)
    while font.getlength(caption) > max_width_px and font_size > CAPTION_FONT_SIZE_MIN:
        font_size -= 1
        font = ImageFont.load_default(size=font_size)
    return font


def _figure_filename(video_id: str, trigger_k: int, is_false_alarm: bool = False) -> str:
    prefix = "falsealarm__" if is_false_alarm else ""
    return f"{prefix}{video_id.replace('/', '__')}__k{trigger_k:06d}.png"


def _write_png_atomic(path: Path, frame_rgb: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.stem}.tmp")
    Image.fromarray(frame_rgb, mode="RGB").save(tmp_path, format="PNG")
    os.replace(tmp_path, path)


def _render_video(
    video_id: str,
    video_path: Path,
    pose_root: Path,
    targets: list[RenderTarget],
    figures_dir: Path,
    label_names: tuple[str, ...],
    force: bool,
    decode_frames_fn: Callable[[Path, list[int]], list[np.ndarray]] = decode_frames,
    load_pose_fn: Callable[..., PoseArrays] = load_pose,
    write_png_fn: Callable[[Path, np.ndarray], None] = _write_png_atomic,
) -> tuple[int, int]:
    src_indices = list(dict.fromkeys(target.src_index for target in targets))
    decoded_frames = decode_frames_fn(video_path, src_indices)
    frame_by_src_index = dict(zip(src_indices, decoded_frames))
    pose = load_pose_fn(video_id, pose_root=pose_root)

    written = 0
    skipped = 0
    for target in targets:
        out_path = figures_dir / _figure_filename(
            target.video_id, target.trigger_k, target.is_false_alarm
        )
        if out_path.exists() and not force:
            print(f"skip {out_path} (já existe, use --force para sobrescrever)")
            skipped += 1
            continue

        imputed = not bool(pose.person_found[target.trigger_k])
        label_name = label_names[target.predicted_label]
        caption = _caption_text(
            target.video_id,
            target.trigger_k,
            target.time_s,
            label_name,
            target.latency_s,
            imputed,
            target.is_false_alarm,
        )
        frame_rgb = frame_by_src_index[target.src_index]
        annotated = _draw_alarm_frame(frame_rgb, pose, target.trigger_k, caption, imputed)
        write_png_fn(out_path, annotated)
        written += 1

    return written, skipped


def run_render(
    run_dir: Path,
    dataset_name: str,
    splits: tuple[str, ...],
    force: bool,
    include_false_alarms: bool = False,
) -> None:
    validate_local_run_dir(run_dir, dataset_name)
    adapter = get_dataset(dataset_name)

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

    protocol = load_alarm_protocol(run_dir / "alarm_protocol.yaml")
    if protocol != BASELINE_A_ALARM_PROTOCOL:
        raise ValueError("alarm_protocol.yaml incompatível com o braço A")

    event_metrics_path = run_dir / "event_metrics.json"
    with event_metrics_path.open(encoding="utf-8") as stream:
        event_metrics = json.load(stream)

    stats = load_stats(adapter.pose_stats_path)
    validate_stats_layout(stats)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = run_dir / "checkpoint.pt"
    model = _load_model(config, checkpoint_path, device)

    frames = adapter.load_frames()
    frame_lookup = _build_frame_lookup(frames)
    video_paths = adapter.video_paths()

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, tuple[int, int]] = {}
    for split in splits:
        source = PoseWindowDataset(
            frames,
            split,
            EVAL_STRIDE,
            lambda video_id: build_pose_features(video_id, pose_root=adapter.pose_root)[0],
            drop_ignored=False,
        )
        video_ids, k_ends, true_labels, pred_labels = _predict_with_identity(
            model, source, stats, device, batch_size=config.batch_size
        )
        targets, n_detected = _collect_render_targets(
            video_ids,
            k_ends,
            true_labels,
            pred_labels,
            protocol,
            frame_lookup,
            include_false_alarms,
        )

        expected_n_detected = event_metrics["splits"][split]["n_detected_events"]
        if n_detected != expected_n_detected:
            raise ValueError(
                f"split={split!r}: n_detected_events recomputado ({n_detected}) "
                f"diverge de event_metrics.json ({expected_n_detected})"
            )

        grouped_targets: dict[str, list[RenderTarget]] = {}
        for target in targets:
            grouped_targets.setdefault(target.video_id, []).append(target)

        total_written = 0
        total_skipped = 0
        for video_id, video_targets in grouped_targets.items():
            written, skipped = _render_video(
                video_id,
                video_paths[video_id],
                adapter.pose_root,
                video_targets,
                figures_dir,
                adapter.label_names,
                force,
            )
            total_written += written
            total_skipped += skipped

        summary[split] = (total_written, total_skipped)

    for split, (written, skipped) in summary.items():
        total = written + skipped
        print(f"split={split}: {written} gravados, {skipped} pulados, {total} total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser(
        "render",
        help="Renderiza PNGs dos quadros reais nos gatilhos de alarme detectados",
    )
    render_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    render_parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    render_parser.add_argument(
        "--split", default="both", choices=("val", "test", "both")
    )
    render_parser.add_argument("--force", action="store_true")
    render_parser.add_argument("--include-false-alarms", action="store_true")
    subparsers.add_parser(
        "selftest", help="Roda checagens sintéticas do diagnóstico qualitativo"
    )

    args = parser.parse_args()
    if args.command == "render":
        splits = ("val", "test") if args.split == "both" else (args.split,)
        run_render(
            run_dir=args.run_dir,
            dataset_name=args.dataset,
            splits=splits,
            force=args.force,
            include_false_alarms=args.include_false_alarms,
        )
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.qualitative import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
