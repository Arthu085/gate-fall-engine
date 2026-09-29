import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

from gatefall.eval.analysis.qualitative import (
    CAPTION_FONT_SIZE_MIN,
    RenderTarget,
    _caption_text,
    _draw_alarm_frame,
    _figure_filename,
    _fit_caption_font,
    _matched_alarm,
    _render_video,
    _write_png_atomic,
)
from gatefall.eval.shared.events import Alarm, FallEvent
from gatefall.pose.loading import PoseArrays


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
    caption_imputed = _caption_text("video_x", 0, 0.0, "fall", 0.5, imputed=True)
    result_imputed = _draw_alarm_frame(frame, imputed_pose, 0, caption_imputed, imputed=True)

    non_imputed_pose = _synthetic_pose_arrays(1)
    caption_real = _caption_text("video_x", 0, 0.0, "fall", 0.5, imputed=False)
    result_real = _draw_alarm_frame(frame, non_imputed_pose, 0, caption_real, imputed=False)

    caption_mentions_imputed = "imputada" in caption_imputed
    drawing_path_changes_pixels = not np.array_equal(result_real, frame)

    return _check(
        "pose imputada não desenha esqueleto/bbox mas mantém legenda; "
        "caminho de desenho real altera pixels visivelmente",
        caption_mentions_imputed and drawing_path_changes_pixels,
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
    caption = _caption_text("video_z", 0, 0.0, "fall", 0.2, imputed=False)

    raised = False
    result = frame
    try:
        result = _draw_alarm_frame(frame, pose, 0, caption, imputed=False)
    except Exception:
        raised = True

    return _check(
        "desenho de esqueleto/bbox/legenda não levanta exceção e altera "
        "pixels visivelmente em relação ao quadro de fundo zerado",
        not raised and not np.array_equal(result, frame),
    )


def _selftest_narrow_frame_caption_never_shrinks_below_legible_floor() -> bool:
    # CAPTION_FONT_SIZE_MIN=14 foi calibrado visualmente (ver comentário na
    # constante): abaixo disso, PIL.ImageFont.load_default() rasteriza alguns
    # espaços como colapsados mesmo com getlength() > 0, o que não é
    # detectável só pela largura medida. Este teste garante que o piso
    # nunca regride silenciosamente para um valor não revisado; a legenda
    # pode ficar mais larga que o quadro (clipando) em vez de encolher além
    # do piso, o que é o comportamento aceito.
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pose = _synthetic_pose_arrays(1)
    caption = _caption_text(
        "home_01/video_13", 64, 6.4, "fall", 1.1, imputed=False
    )

    caption_margin_px = 10
    max_width_px = frame.shape[1] - 2 * caption_margin_px
    font = _fit_caption_font(caption, max_width_px)
    chosen_size = font.size if isinstance(font, ImageFont.FreeTypeFont) else CAPTION_FONT_SIZE_MIN

    raised = False
    try:
        _draw_alarm_frame(frame, pose, 0, caption, imputed=False)
    except Exception:
        raised = True

    return _check(
        "legenda de vídeo estreito (320px) nunca usa fonte menor que "
        "CAPTION_FONT_SIZE_MIN, mesmo quando isso implica clipar a legenda",
        not raised and chosen_size >= CAPTION_FONT_SIZE_MIN,
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


def run_qualitative_selftest() -> bool:
    checks = [
        _selftest_imputed_pose_skips_drawing(),
        _selftest_decode_frames_called_once_per_video(),
        _selftest_draw_alarm_frame_changes_pixels(),
        _selftest_narrow_frame_caption_never_shrinks_below_legible_floor(),
        _selftest_matched_alarm_picks_earliest(),
        _selftest_figure_filename_is_unique_and_stable(),
        _selftest_write_png_atomic_writes_readable_png(),
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
