"""Diagnóstico qualitativo: renderiza quadros reais nos gatilhos de alarme detectados.

Ferramenta independente de estágio, deliberadamente fora do pipeline padrão
(`gatefall.pipeline`). O subcomando `render` fica fora da suíte de selftests da
CI — depende de vídeo bruto decodificado, que a CI não tem; `selftest` roda na
CI normalmente, pois é totalmente sintético. Atende os braços A, B0, B1, C0 e
C1 reusando o `load_event_evaluation()` de cada avaliador de eventos. Lê apenas
artefatos já publicados do run (`config.yaml`, `alarm_protocol.yaml`,
`event_metrics.json`) e das features (HDF5), e nunca escreve ou toca no
lock/journal de `gatefall.eval.baseline_*`.
"""

import argparse
import json
import os
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

# Este módulo nunca abre vídeo diretamente: toda decodificação passa por
# gatefall.data.video_io.decode_frames (ffmpeg-pipe). Desenho/codificação de
# imagem usam Pillow, não OpenCV — cv2.VideoCapture trava em AVIs brutos do
# Le2i (ver docstring de video_io.py).
from gatefall.data.video_io import decode_frames
from gatefall.datasets import get_dataset
from gatefall.dinov3.backbone import (
    RESIZE_SIZE,
    configure_deterministic_inference,
    load_backbone,
    resolve_repo_dir,
    resolve_weights_path,
)
from gatefall.dinov3.features import Dinov3Backbone
from gatefall.dinov3.preprocessing import preprocess_frames, resize_frames
from gatefall.eval.analysis.gate_degradation import _check_dinov3_backbone
from gatefall.eval.analysis.grouped_bootstrap import load_arm_evaluation
from gatefall.eval.analysis.multiseed_summary import ARMS
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, AlarmProtocol, load_alarm_protocol
from gatefall.eval.shared.events import (
    Alarm,
    FallEvent,
    associate_events_and_alarms,
    detect_alarms_for_video,
    fall_events_for_video,
)
from gatefall.eval.shared.orchestration import EventEvaluation
from gatefall.pose.kinematics import COCO17_SKELETON_EDGES
from gatefall.pose.loading import PoseArrays, load_pose
from gatefall.runs import default_run_dir_for_arm, validate_local_run_dir
from gatefall.sam3.descriptors import bbox_from_mask, compute_descriptor
from gatefall.sam3.runtime import (
    TEXT_PROMPT,
    Sam3RuntimeSegmenter,
    Sam3Segmenter,
    ensure_sam3_runtime_available,
    resolve_checkpoint_path,
    resolve_runtime_project_dir,
)
from gatefall.sam3.selection import InstanceSelector
from gatefall.sam3.storage import read_n_instances, read_sam_score, read_v_t, sam3_path

DINOV3_INPUT_PANEL = "dinov3_input"
SAM3_MASK_PANEL = "sam3_mask"
DINOV3_PCA_PANEL = "dinov3_pca"
SOURCE_PANEL_BY_ARM: dict[str, str] = {
    "B0": DINOV3_INPUT_PANEL,
    "B1": DINOV3_INPUT_PANEL,
    "C0": SAM3_MASK_PANEL,
    "C1": SAM3_MASK_PANEL,
}
FEATURE_PANEL_BY_ARM: dict[str, str] = {
    "B0": DINOV3_PCA_PANEL,
    "B1": DINOV3_PCA_PANEL,
}
PCA_COMPONENTS = 3

COLOR_SKELETON = (0, 255, 0)
COLOR_KEYPOINT = (255, 0, 0)
COLOR_BBOX = (255, 255, 0)
COLOR_CAPTION = (255, 255, 255)
COLOR_MASK = (255, 0, 255)
MASK_ALPHA = 0.45


@dataclass(frozen=True)
class RenderTarget:
    video_id: str
    trigger_k: int
    src_index: int
    time_s: float
    predicted_label: int
    latency_s: float | None
    is_false_alarm: bool = False


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


def _caption_lines(
    video_id: str,
    trigger_k: int,
    time_s: float,
    label_name: str,
    latency_s: float | None,
    imputed: bool,
    is_false_alarm: bool = False,
) -> tuple[str, ...]:
    if is_false_alarm:
        outcome = "(ALARME FALSO)"
    else:
        assert latency_s is not None
        outcome = f"latencia={latency_s:.1f}s"
    lines = [video_id, f"k={trigger_k} t={time_s:.1f}s pred={label_name}", outcome]
    if imputed:
        lines.append("(pose imputada)")
    return tuple(lines)


def _draw_alarm_frame(
    frame_rgb: np.ndarray,
    pose: PoseArrays,
    k: int,
    caption_lines: tuple[str, ...],
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

    return _with_caption_strip(np.array(image), caption_lines)


# PIL.ImageFont.load_default(size=N) rasteriza espaços de forma inconsistente
# em tamanhos muito pequenos: alguns espaços (ex.: entre dígito/underscore e
# a letra seguinte) colapsam visualmente a um espaçamento quase nulo, mesmo
# com font.getlength() reportando um avanço não nulo. Verificado empiricamente
# renderizando legendas reais do Le2i (320px): tamanho 11 colapsa
# "k=64 t=6.4s" em "k=64t=6.4s"; tamanho 14 mantém todos os espaços visíveis
# nas legendas reais testadas. Não reduzir sem reverificar visualmente.
CAPTION_FONT_SIZE = 14
CAPTION_MARGIN_PX = 6
CAPTION_LINE_HEIGHT_PX = CAPTION_FONT_SIZE + 4


def _caption_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=CAPTION_FONT_SIZE)


def _wrap_caption_lines(
    lines: tuple[str, ...],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width_px: int,
) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        current = ""
        for word in line.split(" "):
            candidate = f"{current} {word}" if current else word
            if current and font.getlength(candidate) > max_width_px:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        wrapped.append(current)
    return wrapped


def _figure_filename(
    video_id: str,
    trigger_k: int,
    is_false_alarm: bool = False,
    panel: str | None = None,
) -> str:
    prefix = "falsealarm__" if is_false_alarm else ""
    suffix = f"__{panel}" if panel is not None else ""
    return f"{prefix}{video_id.replace('/', '__')}__k{trigger_k:06d}{suffix}.png"


def _write_png_atomic(path: Path, frame_rgb: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.stem}.tmp")
    Image.fromarray(frame_rgb, mode="RGB").save(tmp_path, format="PNG")
    os.replace(tmp_path, path)


def dinov3_input_frame(frame_rgb: np.ndarray) -> np.ndarray:
    resized = resize_frames([frame_rgb])[0]
    return (resized.permute(1, 2, 0) * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()


def _with_caption_strip(image_rgb: np.ndarray, lines: tuple[str, ...]) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    font = _caption_font()
    widest_word_px = max(
        (font.getlength(word) for line in lines for word in line.split(" ")), default=0.0
    )
    wrap_width_px = max(width - 2 * CAPTION_MARGIN_PX, int(np.ceil(widest_word_px)))
    wrapped = _wrap_caption_lines(lines, font, wrap_width_px)
    canvas = Image.new(
        "RGB",
        (
            max(width, wrap_width_px + 2 * CAPTION_MARGIN_PX),
            height + 2 * CAPTION_MARGIN_PX + CAPTION_LINE_HEIGHT_PX * len(wrapped),
        ),
        (0, 0, 0),
    )
    canvas.paste(Image.fromarray(image_rgb, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(wrapped):
        draw.text(
            (CAPTION_MARGIN_PX, height + CAPTION_MARGIN_PX + index * CAPTION_LINE_HEIGHT_PX),
            line,
            fill=COLOR_CAPTION,
            font=font,
        )
    return np.array(canvas)


def _draw_dinov3_input_panel(frame_rgb: np.ndarray, target: RenderTarget) -> np.ndarray:
    return _with_caption_strip(
        dinov3_input_frame(frame_rgb),
        (
            f"{target.video_id} k={target.trigger_k}",
            "entrada DINOv3 224x224",
            "antes da normalizacao",
        ),
    )


def dinov3_patch_tokens(
    backbone: Dinov3Backbone, frame_rgb: np.ndarray, device: str
) -> np.ndarray:
    batch = preprocess_frames([frame_rgb]).to(device)
    with torch.inference_mode():
        out = backbone.forward_features(batch)
    return out["x_norm_patchtokens"][0].float().cpu().numpy()


def patch_feature_pca_image(patch_tokens: np.ndarray) -> np.ndarray:
    n_patches = patch_tokens.shape[0]
    grid = int(round(np.sqrt(n_patches)))
    if patch_tokens.ndim != 2 or grid * grid != n_patches:
        raise ValueError(
            f"patch tokens com shape {patch_tokens.shape} não formam uma grade quadrada"
        )
    centered = patch_tokens.astype(np.float64) - patch_tokens.astype(np.float64).mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    components = components[:PCA_COMPONENTS]
    # Sinal da PCA é arbitrário: fixa positiva a maior carga absoluta de
    # cada componente, para que o mesmo quadro gere sempre as mesmas cores.
    dominant = components[np.arange(PCA_COMPONENTS), np.argmax(np.abs(components), axis=1)]
    components = components * np.where(dominant < 0, -1.0, 1.0)[:, None]
    projected = centered @ components.T
    low = projected.min(axis=0)
    span = projected.max(axis=0) - low
    normalized = np.where(span > 0, (projected - low) / np.where(span > 0, span, 1.0), 0.0)
    grid_rgb = np.round(normalized * 255.0).astype(np.uint8).reshape(grid, grid, PCA_COMPONENTS)
    upsampled = Image.fromarray(grid_rgb, mode="RGB").resize(
        (RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.NEAREST
    )
    return np.array(upsampled)


def _draw_dinov3_pca_panel(pca_rgb: np.ndarray, target: RenderTarget) -> np.ndarray:
    return _with_caption_strip(
        pca_rgb,
        (
            f"{target.video_id} k={target.trigger_k}",
            "DINOv3 patch-feature PCA",
            "(diagnostico pos-hoc)",
        ),
    )


@dataclass(frozen=True)
class Sam3Observation:
    mask: np.ndarray | None
    v_t: np.ndarray
    sam_score: float
    n_instances: int


@dataclass(frozen=True)
class Sam3VideoContext:
    segmenter: Sam3Segmenter
    src_indices: list[int]
    width: int
    height: int
    v_t: np.ndarray
    sam_score: np.ndarray
    n_instances: np.ndarray


def replay_sam3_selection(
    frames_rgb: list[np.ndarray],
    target_ks: list[int],
    *,
    segmenter: Sam3Segmenter,
    width: int,
    height: int,
) -> dict[int, Sam3Observation]:
    # InstanceSelector é causal: a escolha em k depende das escolhas em
    # 0..k-1, então o replay percorre o prefixo inteiro do vídeo, como em
    # gatefall.sam3.extract.run_frames_through_segmenter.
    targets = set(target_ks)
    observations: dict[int, Sam3Observation] = {}
    selector = InstanceSelector()
    for position, frame in enumerate(frames_rgb):
        instances = segmenter.segment_frame(frame, TEXT_PROMPT)
        selected_index = selector.select(instances)
        if position not in targets:
            continue
        if selected_index is None:
            observations[position] = Sam3Observation(
                mask=None,
                v_t=compute_descriptor(None, frame_width=width, frame_height=height),
                sam_score=0.0,
                n_instances=len(instances),
            )
            continue
        selected = instances[selected_index]
        observations[position] = Sam3Observation(
            mask=selected.mask,
            v_t=compute_descriptor(selected.mask, frame_width=width, frame_height=height),
            sam_score=float(np.float32(selected.score)),
            n_instances=len(instances),
        )
    missing = sorted(targets - observations.keys())
    if missing:
        raise ValueError(f"replay SAM 3 não alcançou os quadros alvo k={missing}")
    return observations


def validate_sam3_observation(
    video_id: str, k: int, observation: Sam3Observation, context: Sam3VideoContext
) -> None:
    mismatches: list[str] = []
    if int(context.n_instances[k]) != observation.n_instances:
        mismatches.append(
            f"n_instances persistido={int(context.n_instances[k])} "
            f"recomputado={observation.n_instances}"
        )
    if np.float32(context.sam_score[k]) != np.float32(observation.sam_score):
        mismatches.append(
            f"sam_score persistido={float(context.sam_score[k])} "
            f"recomputado={observation.sam_score}"
        )
    if not np.array_equal(context.v_t[k].astype(np.float32), observation.v_t):
        mismatches.append("v_t recomputado diverge do persistido")
    if mismatches:
        raise ValueError(
            f"video_id={video_id!r}, k={k}: observação SAM 3 recomputada diverge "
            f"do HDF5 ({'; '.join(mismatches)}); PNG não publicado"
        )


def _sam3_caption_lines(
    target: RenderTarget, observation: Sam3Observation
) -> tuple[str, ...]:
    header = f"{target.video_id} k={target.trigger_k}"
    if observation.mask is None:
        return (header, f"SAM 3 sem instancia (n={observation.n_instances})")
    return (
        header,
        f"mascara SAM 3 score={observation.sam_score:.2f} n={observation.n_instances}",
    )


def _draw_sam3_mask_frame(
    frame_rgb: np.ndarray, observation: Sam3Observation, caption_lines: tuple[str, ...]
) -> np.ndarray:
    blended = frame_rgb.astype(np.float32)
    mask: np.ndarray | None = None
    if observation.mask is not None:
        mask = observation.mask.astype(bool)
        if mask.shape != frame_rgb.shape[:2]:
            raise ValueError(
                f"máscara SAM 3 com shape {mask.shape} diverge do quadro "
                f"{frame_rgb.shape[:2]}"
            )
        color = np.array(COLOR_MASK, dtype=np.float32)
        blended[mask] = (1.0 - MASK_ALPHA) * blended[mask] + MASK_ALPHA * color
    image = Image.fromarray(np.round(blended).astype(np.uint8), mode="RGB")
    if mask is not None:
        bbox = bbox_from_mask(mask)
        if bbox is not None:
            ImageDraw.Draw(image).rectangle(bbox, outline=COLOR_MASK, width=2)
    return _with_caption_strip(np.array(image), caption_lines)


def _sam3_decode_indices(
    video_id: str, targets: list[RenderTarget], context: Sam3VideoContext
) -> list[int]:
    last_k = max(target.trigger_k for target in targets)
    if last_k >= len(context.src_indices):
        raise ValueError(
            f"video_id={video_id!r}: trigger_k={last_k} fora da grade "
            f"({len(context.src_indices)} quadros)"
        )
    for target in targets:
        if context.src_indices[target.trigger_k] != target.src_index:
            raise ValueError(
                f"video_id={video_id!r}, k={target.trigger_k}: src_index do alvo "
                f"({target.src_index}) diverge da grade SAM 3 "
                f"({context.src_indices[target.trigger_k]})"
            )
    return context.src_indices[: last_k + 1]


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
    source_panel: str | None = None,
    sam3_context: Sam3VideoContext | None = None,
    feature_panel: str | None = None,
    patch_tokens_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[int, int]:
    if feature_panel is not None and patch_tokens_fn is None:
        raise ValueError("painel de features DINOv3 exige o backbone carregado")

    def panel_path(target: RenderTarget, panel: str) -> Path:
        return figures_dir / _figure_filename(
            target.video_id, target.trigger_k, target.is_false_alarm, panel
        )

    panels = [panel for panel in (source_panel, feature_panel) if panel is not None]
    pending_panels = {
        panel: [
            target for target in targets if force or not panel_path(target, panel).exists()
        ]
        for panel in panels
    }
    replay_sam3 = bool(pending_panels.get(SAM3_MASK_PANEL))
    if replay_sam3:
        if sam3_context is None:
            raise ValueError("painel SAM 3 exige o contexto de replay do vídeo")
        src_indices = _sam3_decode_indices(video_id, targets, sam3_context)
    else:
        src_indices = list(dict.fromkeys(target.src_index for target in targets))
    decoded_frames = decode_frames_fn(video_path, src_indices)
    frame_by_src_index = dict(zip(src_indices, decoded_frames))
    pose = load_pose_fn(video_id, pose_root=pose_root)

    sam3_observations: dict[int, Sam3Observation] = {}
    if replay_sam3:
        assert sam3_context is not None
        target_ks = sorted({target.trigger_k for target in pending_panels[SAM3_MASK_PANEL]})
        sam3_observations = replay_sam3_selection(
            decoded_frames[: target_ks[-1] + 1],
            target_ks,
            segmenter=sam3_context.segmenter,
            width=sam3_context.width,
            height=sam3_context.height,
        )
        for k in target_ks:
            validate_sam3_observation(video_id, k, sam3_observations[k], sam3_context)

    written = 0
    skipped = 0
    for target in targets:
        out_path = figures_dir / _figure_filename(
            target.video_id, target.trigger_k, target.is_false_alarm
        )
        frame_rgb = frame_by_src_index[target.src_index]
        if out_path.exists() and not force:
            print(f"skip {out_path} (já existe, use --force para sobrescrever)")
            skipped += 1
        else:
            imputed = not bool(pose.person_found[target.trigger_k])
            label_name = label_names[target.predicted_label]
            caption_lines = _caption_lines(
                target.video_id,
                target.trigger_k,
                target.time_s,
                label_name,
                target.latency_s,
                imputed,
                target.is_false_alarm,
            )
            annotated = _draw_alarm_frame(
                frame_rgb, pose, target.trigger_k, caption_lines, imputed
            )
            write_png_fn(out_path, annotated)
            written += 1

        for panel in panels:
            if target not in pending_panels[panel]:
                print(
                    f"skip {panel_path(target, panel)} "
                    "(já existe, use --force para sobrescrever)"
                )
                skipped += 1
                continue
            if panel == DINOV3_INPUT_PANEL:
                image = _draw_dinov3_input_panel(frame_rgb, target)
            elif panel == DINOV3_PCA_PANEL:
                assert patch_tokens_fn is not None
                image = _draw_dinov3_pca_panel(
                    patch_feature_pca_image(patch_tokens_fn(frame_rgb)), target
                )
            else:
                observation = sam3_observations[target.trigger_k]
                image = _draw_sam3_mask_frame(
                    frame_rgb, observation, _sam3_caption_lines(target, observation)
                )
            write_png_fn(panel_path(target, panel), image)
            written += 1

    return written, skipped


def source_panel_for_arm(arm: str, enabled: bool) -> str | None:
    if not enabled:
        return None
    try:
        return SOURCE_PANEL_BY_ARM[arm]
    except KeyError as exc:
        raise ValueError(
            f"--source-panel não se aplica à arma {arm!r}; disponível para "
            f"{', '.join(SOURCE_PANEL_BY_ARM)}"
        ) from exc


def feature_panel_for_arm(arm: str, enabled: bool) -> str | None:
    if not enabled:
        return None
    if arm in ("C0", "C1"):
        raise ValueError(
            f"--feature-panel não se aplica à arma {arm!r}: o runtime oficial "
            "isolado do SAM 3 devolve só máscaras e scores, sem embedding "
            "espacial denso para uma PCA de features; use --source-panel para "
            "a máscara SAM 3 selecionada"
        )
    try:
        return FEATURE_PANEL_BY_ARM[arm]
    except KeyError as exc:
        raise ValueError(
            f"--feature-panel não se aplica à arma {arm!r}; disponível para "
            f"{', '.join(FEATURE_PANEL_BY_ARM)}"
        ) from exc


def resolve_render_run_dir(dataset_name: str, arm: str, run_dir: Path | None) -> Path:
    return default_run_dir_for_arm(dataset_name, arm) if run_dir is None else run_dir


def load_render_evaluation(
    arm: str, dataset_name: str, run_dir: Path
) -> tuple[EventEvaluation, AlarmProtocol, dict]:
    validate_local_run_dir(run_dir, dataset_name)
    evaluation = load_arm_evaluation(arm, dataset_name, run_dir)

    protocol = load_alarm_protocol(run_dir / "alarm_protocol.yaml")
    if protocol != BASELINE_A_ALARM_PROTOCOL:
        raise ValueError(f"alarm_protocol.yaml incompatível com o braço {arm}")

    with (run_dir / "event_metrics.json").open(encoding="utf-8") as stream:
        event_metrics = json.load(stream)
    return evaluation, protocol, event_metrics


def _video_src_indices(frames: pd.DataFrame, video_id: str) -> list[int]:
    video_frames = cast(
        pd.DataFrame, frames[frames["video_id"] == video_id]
    ).sort_values("frame_index")
    frame_indices = [int(index) for index in video_frames["frame_index"]]
    if frame_indices != list(range(len(frame_indices))):
        raise ValueError(f"video_id={video_id!r}: frame_index não é contíguo a partir de 0")
    return [int(index) for index in video_frames["src_index"]]


def _sam3_video_context(
    video_id: str,
    frames: pd.DataFrame,
    manifest: pd.DataFrame,
    sam3_root: Path,
    segmenter: Sam3Segmenter,
) -> Sam3VideoContext:
    src_indices = _video_src_indices(frames, video_id)
    path = sam3_path(video_id, sam3_root=sam3_root)
    v_t = read_v_t(path)
    sam_score = read_sam_score(path)
    n_instances = read_n_instances(path)
    if not v_t.shape[0] == sam_score.shape[0] == n_instances.shape[0] == len(src_indices):
        raise ValueError(
            f"video_id={video_id!r}: K do HDF5 SAM 3 diverge da grade "
            f"({len(src_indices)} quadros)"
        )
    manifest_row = cast(pd.DataFrame, manifest[manifest["video_id"] == video_id])
    if manifest_row.empty:
        raise ValueError(f"video_id={video_id!r} ausente do manifesto")
    return Sam3VideoContext(
        segmenter=segmenter,
        src_indices=src_indices,
        width=int(manifest_row.iloc[0]["width"]),
        height=int(manifest_row.iloc[0]["height"]),
        v_t=v_t,
        sam_score=sam_score,
        n_instances=n_instances,
    )


def run_render(
    run_dir: Path | None,
    dataset_name: str,
    splits: tuple[str, ...],
    force: bool,
    include_false_alarms: bool = False,
    arm: str = "A",
    source_panel: bool = False,
    runtime_dir: str | None = None,
    sam3_checkpoint: str | None = None,
    feature_panel: bool = False,
    repo_dir: str | None = None,
    weights: str | None = None,
) -> None:
    panel = source_panel_for_arm(arm, source_panel)
    features = feature_panel_for_arm(arm, feature_panel)
    run_dir = resolve_render_run_dir(dataset_name, arm, run_dir)
    evaluation, protocol, event_metrics = load_render_evaluation(arm, dataset_name, run_dir)
    adapter = get_dataset(dataset_name)

    frames, evaluate_split = evaluation.prepare()
    frame_lookup = _build_frame_lookup(frames)
    video_paths = adapter.video_paths()
    manifest = adapter.load_manifest() if panel == SAM3_MASK_PANEL else None

    patch_tokens_fn: Callable[[np.ndarray], np.ndarray] | None = None
    if features == DINOV3_PCA_PANEL:
        repo_path = resolve_repo_dir(repo_dir)
        weights_path = resolve_weights_path(weights)
        _check_dinov3_backbone(adapter, frames, repo_path, weights_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        configure_deterministic_inference()
        backbone = cast(Dinov3Backbone, load_backbone(repo_path, weights_path, device))

        def backbone_patch_tokens(frame_rgb: np.ndarray) -> np.ndarray:
            return dinov3_patch_tokens(backbone, frame_rgb, device)

        patch_tokens_fn = backbone_patch_tokens

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        segmenter: Sam3Segmenter | None = None
        if panel == SAM3_MASK_PANEL:
            runtime_project_dir = resolve_runtime_project_dir(runtime_dir)
            checkpoint_path = resolve_checkpoint_path(sam3_checkpoint)
            ensure_sam3_runtime_available(runtime_project_dir, checkpoint_path)
            segmenter = stack.enter_context(
                Sam3RuntimeSegmenter(
                    runtime_project_dir=runtime_project_dir,
                    checkpoint_path=checkpoint_path,
                )
            )

        summary: dict[str, tuple[int, int]] = {}
        for split in splits:
            _, (video_ids, k_ends, true_labels, pred_labels) = evaluate_split(split)
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
                sam3_context = None
                if segmenter is not None:
                    assert manifest is not None
                    sam3_context = _sam3_video_context(
                        video_id, frames, manifest, adapter.sam3_root, segmenter
                    )
                written, skipped = _render_video(
                    video_id,
                    video_paths[video_id],
                    adapter.pose_root,
                    video_targets,
                    figures_dir,
                    adapter.label_names,
                    force,
                    source_panel=panel,
                    sam3_context=sam3_context,
                    feature_panel=features,
                    patch_tokens_fn=patch_tokens_fn,
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
    render_parser.add_argument("--arm", default="A", choices=ARMS)
    render_parser.add_argument("--dataset", default="le2i", choices=("le2i",))
    render_parser.add_argument("--run-dir", type=Path, default=None)
    render_parser.add_argument(
        "--split", default="both", choices=("val", "test", "both")
    )
    render_parser.add_argument("--force", action="store_true")
    render_parser.add_argument("--include-false-alarms", action="store_true")
    render_parser.add_argument(
        "--source-panel",
        action="store_true",
        help="B0/B1: entrada DINOv3 224x224; C0/C1: máscara SAM 3 selecionada",
    )
    render_parser.add_argument("--runtime-dir", default=None)
    render_parser.add_argument("--sam3-checkpoint", default=None)
    render_parser.add_argument(
        "--feature-panel",
        action="store_true",
        help="B0/B1: PCA de 3 componentes dos patch tokens DINOv3 (diagnóstico pós-hoc)",
    )
    render_parser.add_argument("--repo-dir", default=None)
    render_parser.add_argument("--weights", default=None)
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
            arm=args.arm,
            source_panel=args.source_panel,
            runtime_dir=args.runtime_dir,
            sam3_checkpoint=args.sam3_checkpoint,
            feature_panel=args.feature_panel,
            repo_dir=args.repo_dir,
            weights=args.weights,
        )
    elif args.command == "selftest":
        from gatefall.eval.analysis.selftests.qualitative import run_selftest

        run_selftest()


if __name__ == "__main__":
    main()
