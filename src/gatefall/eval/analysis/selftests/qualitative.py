import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFont

from gatefall.dinov3.backbone import NORMALIZE_MEAN, NORMALIZE_STD, RESIZE_SIZE
from gatefall.dinov3.preprocessing import preprocess_frames
from gatefall.eval.analysis.multiseed_summary import ARMS
from gatefall.eval.analysis.qualitative import (
    CAPTION_FONT_SIZE,
    CAPTION_LINE_HEIGHT_PX,
    CAPTION_MARGIN_PX,
    DINOV3_INPUT_PANEL,
    DINOV3_PCA_PANEL,
    SAM3_MASK_PANEL,
    RenderTarget,
    Sam3VideoContext,
    _caption_font,
    _caption_lines,
    _collect_render_targets,
    _draw_alarm_frame,
    _figure_filename,
    _matched_alarm,
    _render_video,
    _with_caption_strip,
    _wrap_caption_lines,
    _write_png_atomic,
    dinov3_input_frame,
    dinov3_patch_tokens,
    feature_panel_for_arm,
    load_render_evaluation,
    patch_feature_pca_image,
    replay_sam3_selection,
    resolve_render_run_dir,
    source_panel_for_arm,
)
from gatefall.eval.analysis.selftests.grouped_bootstrap_arms import (
    _evaluation,
    _frames,
    _patched_loaders,
    _raises,
    _synthetic_run,
)
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL, save_alarm_protocol
from gatefall.eval.shared.events import Alarm, FallEvent
from gatefall.pose.loading import PoseArrays
from gatefall.sam3.extract import run_frames_through_segmenter
from gatefall.sam3.runtime import Sam3Instance


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _synthetic_pose_arrays(k_size: int) -> PoseArrays:
    keypoints = np.zeros((k_size, 17, 3), dtype=np.float32)
    for i in range(17):
        keypoints[:, i, 0] = 10.0 + i
        keypoints[:, i, 1] = 20.0 + i
        keypoints[:, i, 2] = 1.0
    bbox = np.tile(np.array([5.0, 5.0, 50.0, 80.0], dtype=np.float32), (k_size, 1))
    person_found = np.ones(k_size, dtype=bool)
    return PoseArrays(
        keypoints=keypoints, bbox=bbox, person_found=person_found, k=k_size, width=100, height=100
    )


def _selftest_imputed_pose_skips_drawing() -> bool:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    imputed_pose = _synthetic_pose_arrays(1)
    imputed_pose.person_found[0] = False
    imputed_pose.keypoints[0] = 0.0
    imputed_pose.bbox[0] = 0.0
    caption_imputed = _caption_lines("video_x", 0, 0.0, "fall", 0.5, imputed=True)
    result_imputed = _draw_alarm_frame(frame, imputed_pose, 0, caption_imputed, imputed=True)

    non_imputed_pose = _synthetic_pose_arrays(1)
    caption_real = _caption_lines("video_x", 0, 0.0, "fall", 0.5, imputed=False)
    result_real = _draw_alarm_frame(frame, non_imputed_pose, 0, caption_real, imputed=False)

    caption_mentions_imputed = "(pose imputada)" in caption_imputed
    imputed_frame_untouched = np.array_equal(result_imputed[:100], frame)
    drawing_path_changes_pixels = not np.array_equal(result_real[:100], frame)

    return _check(
        "pose imputada não desenha esqueleto/bbox (quadro intacto) mas mantém "
        "legenda; caminho de desenho real altera pixels do quadro",
        caption_mentions_imputed and imputed_frame_untouched and drawing_path_changes_pixels,
    )


def _selftest_decode_frames_called_once_per_video() -> bool:
    call_count = 0
    received_src_indices: list[int] = []

    def fake_decode_frames(video_path: Path, src_indices: list[int]) -> list[np.ndarray]:
        nonlocal call_count
        call_count += 1
        received_src_indices.extend(src_indices)
        return [np.zeros((10, 10, 3), dtype=np.uint8) for _ in src_indices]

    def fake_load_pose(video_id: str, *, pose_root: Path) -> PoseArrays:
        return _synthetic_pose_arrays(20)

    written_paths: list[Path] = []

    def fake_write_png(path: Path, frame_rgb: np.ndarray) -> None:
        written_paths.append(path)

    targets = [
        RenderTarget(
            video_id="env/video_a", trigger_k=3, src_index=3, time_s=0.3,
            predicted_label=1, latency_s=0.1,
        ),
        RenderTarget(
            video_id="env/video_a", trigger_k=7, src_index=3, time_s=0.7,
            predicted_label=1, latency_s=0.5,
        ),
        RenderTarget(
            video_id="env/video_a", trigger_k=9, src_index=9, time_s=0.9,
            predicted_label=2, latency_s=0.7,
        ),
    ]

    written, skipped = _render_video(
        "env/video_a",
        Path("fake_video.avi"),
        Path("fake_pose_root"),
        targets,
        Path("fake_figures_dir"),
        ("walk", "fall", "fallen"),
        force=True,
        decode_frames_fn=fake_decode_frames,
        load_pose_fn=fake_load_pose,
        write_png_fn=fake_write_png,
    )

    distinct_src_indices = {target.src_index for target in targets}
    ok = (
        call_count == 1
        and len(received_src_indices) <= len(targets)
        and set(received_src_indices) == distinct_src_indices
        and written == 3
        and skipped == 0
    )
    return _check(
        "decode_frames é chamado exatamente 1 vez por vídeo, com "
        "src_indices deduplicados cobrindo todos os alvos",
        ok,
    )


def _selftest_draw_alarm_frame_changes_pixels() -> bool:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose = _synthetic_pose_arrays(1)
    caption = _caption_lines("video_z", 0, 0.0, "fall", 0.2, imputed=False)

    raised = False
    result = frame
    try:
        result = _draw_alarm_frame(frame, pose, 0, caption, imputed=False)
    except Exception:
        raised = True

    return _check(
        "desenho de esqueleto/bbox não levanta exceção e altera pixels "
        "visivelmente em relação ao quadro de fundo zerado",
        not raised and not np.array_equal(result[:100], frame),
    )


def _strip_ink_columns(strip: np.ndarray) -> np.ndarray:
    return np.flatnonzero(strip.max(axis=(0, 2)) > 0)


def _selftest_le2i_caption_fits_below_frame() -> bool:
    # Quadro Le2i real tem 320px de largura; a legenda antiga em linha única
    # clipava `latencia=...` à direita. A legenda agora fica numa faixa
    # preta abaixo do quadro, quebrada em linhas que cabem na largura.
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    pose = _synthetic_pose_arrays(1)
    pose.person_found[0] = False
    font = _caption_font()
    max_width_px = frame.shape[1] - 2 * CAPTION_MARGIN_PX

    cases = (
        _caption_lines("coffee_room_01/video_23", 64, 6.4, "fallen", 11.1, imputed=True),
        _caption_lines(
            "coffee_room_01/video_23", 123, 12.3, "fallen", None, imputed=True,
            is_false_alarm=True,
        ),
    )
    ok = isinstance(font, ImageFont.FreeTypeFont) and font.size == CAPTION_FONT_SIZE
    for lines in cases:
        wrapped = _wrap_caption_lines(lines, font, max_width_px)
        result = _draw_alarm_frame(frame, pose, 0, lines, imputed=True)
        strip = result[frame.shape[0] :]
        ink_columns = _strip_ink_columns(strip)
        lines_inked = all(
            strip[
                CAPTION_MARGIN_PX + index * CAPTION_LINE_HEIGHT_PX : CAPTION_MARGIN_PX
                + (index + 1) * CAPTION_LINE_HEIGHT_PX
            ].max()
            > 0
            for index in range(len(wrapped))
        )
        ok = ok and (
            result.shape
            == (
                frame.shape[0] + 2 * CAPTION_MARGIN_PX + CAPTION_LINE_HEIGHT_PX * len(wrapped),
                frame.shape[1],
                3,
            )
            and np.array_equal(result[: frame.shape[0]], frame)
            and " ".join(wrapped).split() == " ".join(lines).split()
            and all(font.getlength(line) <= max_width_px for line in wrapped)
            and ink_columns.size > 0
            and int(ink_columns.max()) < frame.shape[1] - CAPTION_MARGIN_PX // 2
            and lines_inked
        )
    detected_lines = cases[0]
    false_alarm_lines = cases[1]
    ok = (
        ok
        and "latencia=11.1s" in detected_lines
        and any(
            "latencia=11.1s" in line.split(" ")
            for line in _wrap_caption_lines(detected_lines, font, max_width_px)
        )
        and "(ALARME FALSO)" in false_alarm_lines
        and all("latencia" not in line for line in false_alarm_lines)
        and "(pose imputada)" in detected_lines
        and "k=64 t=6.4s pred=fallen" in detected_lines
    )
    return _check(
        "legenda de quadro Le2i de 320px fica numa faixa abaixo do quadro "
        "(pixels do vídeo intactos), com fonte fixa, todas as linhas dentro da "
        "largura e latencia=... inteira; alarme falso mantém o marcador",
        ok,
    )


def _selftest_caption_strip_wraps_narrow_panels() -> bool:
    font = _caption_font()
    image = np.zeros((10, 120, 3), dtype=np.uint8)
    lines = ("k=123 t=12.3s pred=fallen latencia=11.1s",)
    max_width_px = image.shape[1] - 2 * CAPTION_MARGIN_PX
    wrapped = _wrap_caption_lines(lines, font, max_width_px)
    result = _with_caption_strip(image, lines)
    ok = (
        len(wrapped) > 1
        and " ".join(wrapped).split() == lines[0].split()
        and all(font.getlength(line) <= max_width_px for line in wrapped)
        and result.shape[0]
        == image.shape[0] + 2 * CAPTION_MARGIN_PX + CAPTION_LINE_HEIGHT_PX * len(wrapped)
    )

    tiny = np.full((10, 40, 3), 7, dtype=np.uint8)
    word = "latencia=11.1s"
    widened = _with_caption_strip(tiny, (word,))
    ink_columns = _strip_ink_columns(widened[tiny.shape[0] :])
    ok = (
        ok
        and widened.shape[1] > tiny.shape[1]
        and widened.shape[1] >= font.getlength(word) + 2 * CAPTION_MARGIN_PX
        and np.array_equal(widened[: tiny.shape[0], : tiny.shape[1]], tiny)
        and not widened[: tiny.shape[0], tiny.shape[1] :].any()
        and ink_columns.size > 0
        and int(ink_columns.max()) < widened.shape[1] - CAPTION_MARGIN_PX // 2
    )
    return _check(
        "faixa de legenda quebra linhas longas em palavras sem perder texto e, "
        "se uma palavra não cabe na largura, alarga o canvas com preto à "
        "direita em vez de clipar ou cobrir o quadro",
        ok,
    )


def _selftest_matched_alarm_picks_earliest() -> bool:
    event = FallEvent(
        video_id="env/video_k",
        start_time_s=0.0,
        fall_end_time_s=0.5,
        fallen_start_time_s=1.0,
        fallen_end_time_s=1.5,
        association_end_time_s=5.0,
        has_following_fallen=True,
    )
    earlier = Alarm(video_id="env/video_k", trigger_k=10, trigger_time_s=1.0)
    later = Alarm(video_id="env/video_k", trigger_k=20, trigger_time_s=2.0)

    matched = _matched_alarm(event, [later, earlier])
    return _check(
        "_matched_alarm escolhe o alarme com trigger_time_s mais cedo entre "
        "os candidatos dentro da janela do evento",
        matched is earlier,
    )


def _selftest_figure_filename_is_unique_and_stable() -> bool:
    name_a = _figure_filename("env/video_a", 3)
    name_b = _figure_filename("env/video_b", 3)
    name_a_again = _figure_filename("env/video_a", 3)
    return _check(
        "_figure_filename é única para pares (video_id, trigger_k) "
        "distintos e estável para o mesmo par",
        name_a != name_b and name_a == name_a_again,
    )


def _selftest_write_png_atomic_writes_readable_png() -> bool:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    ok = False
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "frame.png"
        _write_png_atomic(out_path, frame)

        exists = out_path.exists()
        no_leftover_tmp = not any(out_path.parent.glob("*.tmp*"))
        with Image.open(out_path) as reread:
            reread.load()
            readable_with_expected_shape = (
                reread.mode == "RGB" and reread.size == (frame.shape[1], frame.shape[0])
            )
        ok = exists and no_leftover_tmp and readable_with_expected_shape

    return _check(
        "_write_png_atomic grava um .png real (não .png.tmp) e o arquivo "
        "final é legível como imagem com o formato esperado",
        ok,
    )


def _selftest_multi_arm_dispatch() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm in ARMS:
            run_dir = root / arm
            config = _synthetic_run(run_dir, arm)
            calls: list[tuple] = []
            evaluation = _evaluation(config, _frames())
            with _patched_loaders({arm: evaluation}, calls):
                loaded, protocol, event_metrics = load_render_evaluation(arm, "le2i", run_dir)
            expected_kwargs = {"fields_allowed_to_differ": frozenset()} if arm == "A" else {}
            expected_default = Path("runs/local/le2i") / f"baseline_{arm.lower()}"
            ok = ok and (
                loaded is evaluation
                and calls == [(arm, "le2i", run_dir, expected_kwargs)]
                and protocol == BASELINE_A_ALARM_PROTOCOL
                and "splits" in event_metrics
                and resolve_render_run_dir("le2i", arm, None) == expected_default
                and resolve_render_run_dir("le2i", arm, run_dir) == run_dir
            )

        foreign_dir = root / "foreign"
        b0_config = _synthetic_run(foreign_dir, "B0")
        with _patched_loaders({"C1": _evaluation(b0_config, _frames())}, []):
            foreign_rejected = _raises(
                lambda: load_render_evaluation("C1", "le2i", foreign_dir),
                ValueError,
                "pertence à arma 'B0'",
            )

        a_dir = root / "A"
        save_alarm_protocol(
            replace(BASELINE_A_ALARM_PROTOCOL, trigger_consecutive=2),
            a_dir / "alarm_protocol.yaml",
            force=True,
        )
        a_config = _synthetic_run(root / "A_clean", "A")
        with _patched_loaders({"A": _evaluation(a_config, _frames())}, []):
            a_protocol_rejected = _raises(
                lambda: load_render_evaluation("A", "le2i", a_dir),
                ValueError,
                "alarm_protocol.yaml incompatível com o braço A",
            )
        ok = ok and foreign_rejected and a_protocol_rejected
    return _check(
        "render despacha A/B0/B1/C0/C1 para o load_event_evaluation da própria "
        "arma (A com seed congelada), resolve o run_dir padrão por arma, recusa "
        "run de arma estrangeira e mantém a recusa de protocolo divergente do A",
        ok,
    )


def _selftest_collect_targets_keeps_earliest_alarm_not_onset() -> bool:
    # fall em k=[10,14], fallen em k=[15,99]; dois alarmes dentro da janela de
    # associação do mesmo evento (k=12 e k=72, separados por mais que o
    # refratário de 5 s). O alvo deve ser o gatilho mais cedo, k=12 — não o
    # início anotado da queda (k=10).
    n_frames = 120
    labels = [0] * n_frames
    for k in range(10, 15):
        labels[k] = 1
    for k in range(15, 100):
        labels[k] = 2
    preds = [0] * n_frames
    for k in (10, 11, 12, 70, 71, 72):
        preds[k] = 2
    video_id = "env/video_e"
    k_ends = list(range(n_frames))
    frame_lookup = {
        (video_id, k): (3 * k, k / BASELINE_A_ALARM_PROTOCOL.target_fps) for k in k_ends
    }
    targets, n_detected = _collect_render_targets(
        [video_id] * n_frames,
        k_ends,
        labels,
        preds,
        BASELINE_A_ALARM_PROTOCOL,
        frame_lookup,
    )
    ok = (
        n_detected == 1
        and len(targets) == 1
        and targets[0].trigger_k == 12
        and targets[0].src_index == 36
        and targets[0].latency_s is not None
        and not targets[0].is_false_alarm
    )
    return _check(
        "_collect_render_targets mantém o gatilho de alarme mais cedo do evento "
        "detectado (k=12), distinto do início anotado da queda (k=10)",
        ok,
    )


def _selftest_figure_filenames_by_panel() -> bool:
    names = {
        _figure_filename("env/video_a", 7),
        _figure_filename("env/video_a", 7, is_false_alarm=True),
        _figure_filename("env/video_a", 7, panel=DINOV3_INPUT_PANEL),
        _figure_filename("env/video_a", 7, panel=DINOV3_PCA_PANEL),
        _figure_filename("env/video_a", 7, is_false_alarm=True, panel=SAM3_MASK_PANEL),
    }
    ok = names == {
        "env__video_a__k000007.png",
        "falsealarm__env__video_a__k000007.png",
        "env__video_a__k000007__dinov3_input.png",
        "env__video_a__k000007__dinov3_pca.png",
        "falsealarm__env__video_a__k000007__sam3_mask.png",
    }
    return _check(
        "nome do PNG principal é o mesmo de antes; painéis de origem ganham "
        "sufixo próprio e preservam o prefixo falsealarm__",
        ok,
    )


def _selftest_source_panel_by_arm() -> bool:
    ok = (
        source_panel_for_arm("A", False) is None
        and source_panel_for_arm("B0", False) is None
        and source_panel_for_arm("B0", True) == DINOV3_INPUT_PANEL
        and source_panel_for_arm("B1", True) == DINOV3_INPUT_PANEL
        and source_panel_for_arm("C0", True) == SAM3_MASK_PANEL
        and source_panel_for_arm("C1", True) == SAM3_MASK_PANEL
        and _raises(lambda: source_panel_for_arm("A", True), ValueError, "não se aplica")
    )
    return _check(
        "--source-panel resolve DINOv3 para B0/B1, SAM 3 para C0/C1 e é "
        "recusado para A",
        ok,
    )


def _selftest_arm_a_render_has_no_panel() -> bool:
    written_paths: list[Path] = []
    targets = [
        RenderTarget(
            video_id="env/video_a", trigger_k=3, src_index=3, time_s=0.3,
            predicted_label=1, latency_s=0.1,
        ),
        RenderTarget(
            video_id="env/video_a", trigger_k=9, src_index=9, time_s=0.9,
            predicted_label=2, latency_s=None, is_false_alarm=True,
        ),
    ]
    written, skipped = _render_video(
        "env/video_a",
        Path("fake_video.avi"),
        Path("fake_pose_root"),
        targets,
        Path("fake_figures_dir"),
        ("walk", "fall", "fallen"),
        force=True,
        decode_frames_fn=lambda _path, indices: [
            np.zeros((10, 10, 3), dtype=np.uint8) for _ in indices
        ],
        load_pose_fn=lambda _video_id, *, pose_root: _synthetic_pose_arrays(20),
        write_png_fn=lambda path, _frame: written_paths.append(path),
    )
    return _check(
        "render sem painel (braço A) grava só os PNGs principais com os nomes "
        "de antes",
        written == 2
        and skipped == 0
        and [path.name for path in written_paths]
        == ["env__video_a__k000003.png", "falsealarm__env__video_a__k000009.png"],
    )


def _selftest_dinov3_panel_matches_backbone_input() -> bool:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    panel_input = dinov3_input_frame(frame)
    normalized = preprocess_frames([frame])[0].numpy()
    mean = np.array(NORMALIZE_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(NORMALIZE_STD, dtype=np.float32).reshape(3, 1, 1)
    denormalized = (normalized * std + mean).transpose(1, 2, 0) * 255.0

    written: dict[str, np.ndarray] = {}
    target = RenderTarget(
        video_id="env/video_b", trigger_k=4, src_index=8, time_s=0.4,
        predicted_label=1, latency_s=0.2,
    )
    _render_video(
        "env/video_b",
        Path("fake_video.avi"),
        Path("fake_pose_root"),
        [target],
        Path("fake_figures_dir"),
        ("walk", "fall", "fallen"),
        force=True,
        decode_frames_fn=lambda _path, indices: [frame for _ in indices],
        load_pose_fn=lambda _video_id, *, pose_root: _synthetic_pose_arrays(20),
        write_png_fn=lambda path, image: written.__setitem__(path.name, image),
        source_panel=DINOV3_INPUT_PANEL,
    )
    panel = written.get("env__video_b__k000004__dinov3_input.png")
    ok = (
        panel_input.shape == (RESIZE_SIZE, RESIZE_SIZE, 3)
        and panel_input.dtype == np.uint8
        and bool(np.abs(denormalized - panel_input.astype(np.float32)).max() < 1e-3)
        and set(written) == {"env__video_b__k000004.png", "env__video_b__k000004__dinov3_input.png"}
        and panel is not None
        and panel.shape[1] == RESIZE_SIZE
        and panel.shape[0] > RESIZE_SIZE
        and np.array_equal(panel[:RESIZE_SIZE], panel_input)
    )
    return _check(
        "painel B é o quadro 224x224 consumido pelo DINOv3 antes da "
        "normalização (inverte exatamente preprocess_frames) e a legenda fica "
        "fora dos pixels de entrada",
        ok,
    )


class _FakeSam3Segmenter:
    """Pessoa à esquerda com score baixo e distrator à direita com score alto.

    No quadro 0 só a pessoa aparece; a partir do quadro 1 o distrator
    surge com score maior. Seleção causal mantém a pessoa (IoU > 0 com a
    última bbox); seleção isolada do quadro alvo escolheria o distrator.
    """

    def __init__(self) -> None:
        self.frames_seen: list[int] = []

    def segment_frame(self, frame_rgb: np.ndarray, text_prompt: str) -> list[Sam3Instance]:
        position = int(frame_rgb[0, 0, 0])
        self.frames_seen.append(position)
        person = np.zeros(frame_rgb.shape[:2], dtype=bool)
        person[10 : 40 + position, 10:30] = True
        instances = [Sam3Instance(mask=person, score=0.4)]
        if position >= 1:
            distractor = np.zeros(frame_rgb.shape[:2], dtype=bool)
            distractor[10:40, 100:130] = True
            instances.append(Sam3Instance(mask=distractor, score=0.95))
        return instances


def _sam3_frames(n_frames: int) -> list[np.ndarray]:
    frames = []
    for position in range(n_frames):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[0, 0, 0] = position
        frames.append(frame)
    return frames


def _selftest_sam3_replay_matches_extraction_and_validates() -> bool:
    n_frames = 6
    frames = _sam3_frames(n_frames)
    v_t, sam_score, n_instances = run_frames_through_segmenter(
        frames, segmenter=_FakeSam3Segmenter(), width=160, height=120
    )

    observations = replay_sam3_selection(
        frames, [3], segmenter=_FakeSam3Segmenter(), width=160, height=120
    )
    isolated = replay_sam3_selection(
        frames[3:4], [0], segmenter=_FakeSam3Segmenter(), width=160, height=120
    )
    causal_choice_kept = (
        observations[3].sam_score == float(np.float32(0.4))
        and isolated[0].sam_score == float(np.float32(0.95))
    )

    segmenter = _FakeSam3Segmenter()
    context = Sam3VideoContext(
        segmenter=segmenter,
        src_indices=list(range(0, 2 * n_frames, 2)),
        width=160,
        height=120,
        v_t=v_t,
        sam_score=sam_score,
        n_instances=n_instances,
    )
    decode_calls: list[list[int]] = []

    def fake_decode(_path: Path, indices: list[int]) -> list[np.ndarray]:
        decode_calls.append(list(indices))
        return [frames[index // 2] for index in indices]

    targets = [
        RenderTarget(
            video_id="env/video_c", trigger_k=3, src_index=6, time_s=0.3,
            predicted_label=1, latency_s=0.1,
        ),
    ]
    written: dict[str, np.ndarray] = {}
    _render_video(
        "env/video_c",
        Path("fake_video.avi"),
        Path("fake_pose_root"),
        targets,
        Path("fake_figures_dir"),
        ("walk", "fall", "fallen"),
        force=True,
        decode_frames_fn=fake_decode,
        load_pose_fn=lambda _video_id, *, pose_root: _synthetic_pose_arrays(n_frames),
        write_png_fn=lambda path, image: written.__setitem__(path.name, image),
        source_panel=SAM3_MASK_PANEL,
        sam3_context=context,
    )
    panel = written.get("env__video_c__k000003__sam3_mask.png")
    person_pixel_tinted = panel is not None and not np.array_equal(panel[20, 20], frames[3][20, 20])
    distractor_pixel_untouched = panel is not None and np.array_equal(
        panel[20, 115], frames[3][20, 115]
    )
    rendered = (
        decode_calls == [[0, 2, 4, 6]]
        and segmenter.frames_seen == [0, 1, 2, 3]
        and set(written)
        == {"env__video_c__k000003.png", "env__video_c__k000003__sam3_mask.png"}
        and person_pixel_tinted
        and distractor_pixel_untouched
    )

    tampered_score = sam_score.copy()
    tampered_score[3] = 0.95
    tampered_context = replace(context, segmenter=_FakeSam3Segmenter(), sam_score=tampered_score)
    tampered_written: list[Path] = []
    tampered_rejected = _raises(
        lambda: _render_video(
            "env/video_c",
            Path("fake_video.avi"),
            Path("fake_pose_root"),
            targets,
            Path("fake_figures_dir"),
            ("walk", "fall", "fallen"),
            force=True,
            decode_frames_fn=fake_decode,
            load_pose_fn=lambda _video_id, *, pose_root: _synthetic_pose_arrays(n_frames),
            write_png_fn=lambda path, _image: tampered_written.append(path),
            source_panel=SAM3_MASK_PANEL,
            sam3_context=tampered_context,
        ),
        ValueError,
        "sam_score persistido",
    )
    return _check(
        "painel C reproduz a seleção causal do SAM 3 do início do vídeo até o "
        "alvo numa única decodificação, sobrepõe a máscara selecionada e recusa "
        "publicar qualquer PNG quando v_t/sam_score/n_instances divergem do HDF5",
        causal_choice_kept and rendered and tampered_rejected and not tampered_written,
    )


def _selftest_feature_panel_by_arm() -> bool:
    ok = (
        feature_panel_for_arm("A", False) is None
        and feature_panel_for_arm("C1", False) is None
        and feature_panel_for_arm("B0", True) == DINOV3_PCA_PANEL
        and feature_panel_for_arm("B1", True) == DINOV3_PCA_PANEL
        and _raises(lambda: feature_panel_for_arm("A", True), ValueError, "não se aplica")
        and _raises(
            lambda: feature_panel_for_arm("C0", True), ValueError, "sem embedding espacial denso"
        )
        and _raises(
            lambda: feature_panel_for_arm("C1", True), ValueError, "sem embedding espacial denso"
        )
    )
    return _check(
        "--feature-panel resolve a PCA DINOv3 só para B0/B1 e recusa A e "
        "C0/C1 (SAM 3 sem embedding espacial denso, sem PCA falsa)",
        ok,
    )


def _synthetic_patch_tokens(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(14 * 14, 768)).astype(np.float32)


def _selftest_patch_feature_pca_image_is_deterministic() -> bool:
    tokens = _synthetic_patch_tokens()
    image = patch_feature_pca_image(tokens)
    again = patch_feature_pca_image(tokens.copy())
    patch_px = RESIZE_SIZE // 14
    blocks_constant = all(
        np.array_equal(
            image[row * patch_px : (row + 1) * patch_px, col * patch_px : (col + 1) * patch_px],
            np.broadcast_to(
                image[row * patch_px, col * patch_px],
                (patch_px, patch_px, 3),
            ),
        )
        for row in range(14)
        for col in range(14)
    )
    full_range = all(
        int(image[:, :, channel].min()) == 0 and int(image[:, :, channel].max()) == 255
        for channel in range(3)
    )
    return _check(
        "PCA de 3 componentes dos patch tokens gera 224x224x3 uint8 "
        "determinístico, um bloco constante por patch da grade 14x14 e cada "
        "componente normalizado para [0,255]; grade não quadrada é recusada",
        image.shape == (RESIZE_SIZE, RESIZE_SIZE, 3)
        and image.dtype == np.uint8
        and np.array_equal(image, again)
        and blocks_constant
        and full_range
        and _raises(
            lambda: patch_feature_pca_image(tokens[:150]), ValueError, "grade quadrada"
        ),
    )


class _FakeDinov3Backbone:
    def __init__(self, tokens: np.ndarray) -> None:
        self.tokens = tokens
        self.inputs: list[torch.Tensor] = []

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.inputs.append(x)
        return {
            "x_norm_clstoken": torch.zeros((1, self.tokens.shape[1])),
            "x_norm_patchtokens": torch.from_numpy(self.tokens).unsqueeze(0),
        }


def _selftest_dinov3_patch_tokens_use_backbone_input() -> bool:
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    tokens = _synthetic_patch_tokens(1)
    backbone = _FakeDinov3Backbone(tokens)
    captured = dinov3_patch_tokens(backbone, frame, "cpu")
    ok = (
        len(backbone.inputs) == 1
        and torch.equal(backbone.inputs[0], preprocess_frames([frame]))
        and captured.shape == tokens.shape
        and captured.dtype == np.float32
        and np.array_equal(captured, tokens)
    )
    return _check(
        "patch tokens vêm de forward_features sobre a entrada normal do "
        "DINOv3 (preprocess_frames) e não do descritor de 1536 dimensões",
        ok,
    )


def _selftest_dinov3_pca_panel_render_path() -> bool:
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    tokens = _synthetic_patch_tokens(2)
    seen_frames: list[np.ndarray] = []

    def fake_patch_tokens(frame_rgb: np.ndarray) -> np.ndarray:
        seen_frames.append(frame_rgb)
        return tokens

    target = RenderTarget(
        video_id="env/video_p", trigger_k=5, src_index=10, time_s=0.5,
        predicted_label=1, latency_s=0.3,
    )

    def render(figures_dir: Path, force: bool, written: dict[str, np.ndarray]) -> tuple[int, int]:
        return _render_video(
            "env/video_p",
            Path("fake_video.avi"),
            Path("fake_pose_root"),
            [target],
            figures_dir,
            ("walk", "fall", "fallen"),
            force=force,
            decode_frames_fn=lambda _path, indices: [frame for _ in indices],
            load_pose_fn=lambda _video_id, *, pose_root: _synthetic_pose_arrays(20),
            write_png_fn=lambda path, image: written.__setitem__(path.name, image),
            source_panel=DINOV3_INPUT_PANEL,
            feature_panel=DINOV3_PCA_PANEL,
            patch_tokens_fn=fake_patch_tokens,
        )

    written: dict[str, np.ndarray] = {}
    counts = render(Path("fake_figures_dir"), True, written)
    pca_panel = written.get("env__video_p__k000005__dinov3_pca.png")
    input_panel = written.get("env__video_p__k000005__dinov3_input.png")
    rendered = (
        counts == (3, 0)
        and list(written)
        == [
            "env__video_p__k000005.png",
            "env__video_p__k000005__dinov3_input.png",
            "env__video_p__k000005__dinov3_pca.png",
        ]
        and len(seen_frames) == 1
        and seen_frames[0] is frame
        and pca_panel is not None
        and pca_panel.shape[1] == RESIZE_SIZE
        and pca_panel.shape[0] > RESIZE_SIZE
        and np.array_equal(pca_panel[:RESIZE_SIZE], patch_feature_pca_image(tokens))
        and input_panel is not None
        and np.array_equal(input_panel[:RESIZE_SIZE], dinov3_input_frame(frame))
    )

    with tempfile.TemporaryDirectory() as tmp:
        figures_dir = Path(tmp)
        (figures_dir / "env__video_p__k000005__dinov3_pca.png").touch()
        skip_written: dict[str, np.ndarray] = {}
        skip_counts = render(figures_dir, False, skip_written)
        skip_ok = (
            skip_counts == (2, 1)
            and "env__video_p__k000005__dinov3_pca.png" not in skip_written
        )

    missing_backbone = _raises(
        lambda: _render_video(
            "env/video_p",
            Path("fake_video.avi"),
            Path("fake_pose_root"),
            [target],
            Path("fake_figures_dir"),
            ("walk", "fall", "fallen"),
            force=True,
            decode_frames_fn=lambda _path, indices: [frame for _ in indices],
            load_pose_fn=lambda _video_id, *, pose_root: _synthetic_pose_arrays(20),
            write_png_fn=lambda _path, _image: None,
            feature_panel=DINOV3_PCA_PANEL,
        ),
        ValueError,
        "exige o backbone",
    )
    return _check(
        "painel __dinov3_pca é gravado ao lado de __dinov3_input (inalterado), "
        "com legenda fora dos pixels da PCA, respeita skip sem --force e exige "
        "o backbone carregado",
        rendered and skip_ok and missing_backbone,
    )


def run_qualitative_selftest() -> bool:
    checks = [
        _selftest_imputed_pose_skips_drawing(),
        _selftest_decode_frames_called_once_per_video(),
        _selftest_draw_alarm_frame_changes_pixels(),
        _selftest_le2i_caption_fits_below_frame(),
        _selftest_caption_strip_wraps_narrow_panels(),
        _selftest_matched_alarm_picks_earliest(),
        _selftest_figure_filename_is_unique_and_stable(),
        _selftest_write_png_atomic_writes_readable_png(),
        _selftest_multi_arm_dispatch(),
        _selftest_collect_targets_keeps_earliest_alarm_not_onset(),
        _selftest_figure_filenames_by_panel(),
        _selftest_source_panel_by_arm(),
        _selftest_arm_a_render_has_no_panel(),
        _selftest_dinov3_panel_matches_backbone_input(),
        _selftest_sam3_replay_matches_extraction_and_validates(),
        _selftest_feature_panel_by_arm(),
        _selftest_patch_feature_pca_image_is_deterministic(),
        _selftest_dinov3_patch_tokens_use_backbone_input(),
        _selftest_dinov3_pca_panel_render_path(),
    ]
    ok = all(checks)
    if not ok:
        print("\nqualitative selftest FALHOU", file=sys.stderr)
    else:
        print("\nqualitative selftest OK: todas as checagens passaram")
    return ok


def run_selftest() -> None:
    if not run_qualitative_selftest():
        sys.exit(1)
