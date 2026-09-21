"""Relatório de cobertura dos descritores V_t do SAM 3 extraídos para o Le2i."""

import sys
from typing import cast

import h5py
import pandas as pd

from gatefall.datasets import DatasetAdapter
from gatefall.sam3 import descriptors, storage
from gatefall.sam3.dataset_guard import ensure_sam3_dataset_supported
from gatefall.sam3.storage import sam3_path

EXPECTED_VIDEO_COUNT = 190
EXPECTED_TOTAL_FRAMES = 30494
EXPECTED_SPLIT_FRAME_COUNTS = {"train": 22246, "val": 2080, "test": 6168}


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def find_provenance_divergences(
    attrs_by_video: dict[str, dict[str, object]]
) -> list[str]:
    if not attrs_by_video:
        return []

    video_ids = sorted(attrs_by_video)
    reference_id = video_ids[0]
    reference_attrs = attrs_by_video[reference_id]

    divergences: list[str] = []
    missing_by_video: dict[str, set[str]] = {video_id: set() for video_id in video_ids}
    for video_id in video_ids:
        attrs = attrs_by_video[video_id]
        missing_names = [
            name for name in storage.PROVENANCE_ATTR_NAMES if name not in attrs
        ]
        if missing_names:
            missing_by_video[video_id] = set(missing_names)
            divergences.append(
                f"{video_id}: atributo(s) de proveniência ausente(s): "
                f"{', '.join(missing_names)}"
            )

    for video_id in video_ids[1:]:
        attrs = attrs_by_video[video_id]
        reasons: list[str] = []
        for name in storage.PROVENANCE_ATTR_NAMES:
            if name in missing_by_video[video_id] or name in missing_by_video[reference_id]:
                continue
            if name in attrs and name in reference_attrs:
                if not storage.attrs_equal(attrs[name], reference_attrs[name]):
                    reasons.append(f"'{name}' com valor divergente")
        if reasons:
            divergences.append(
                f"{video_id} diverge de {reference_id}: {'; '.join(reasons)}"
            )
    return divergences


def run_sam3_report(adapter: DatasetAdapter) -> None:
    ensure_sam3_dataset_supported(adapter)

    if not adapter.frames_path.exists():
        print(
            f"\nsam3 report FALHOU: {adapter.frames_path} não existe — rode "
            "`uv run python -m gatefall.data.timegrid build` primeiro",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = adapter.load_frames()
    group_sizes = cast(pd.Series, frames.groupby("video_id").size())
    split_by_video = cast(
        pd.Series, frames.groupby("video_id")["split"].first()
    )

    missing: list[str] = []
    mismatched: list[str] = []
    frames_by_split: dict[str, int] = {}
    total_bytes = 0
    total_present = 0
    attrs_by_video: dict[str, dict[str, object]] = {}
    structurally_invalid: dict[str, list[str]] = {}

    for video_id, n_frames in group_sizes.items():
        video_id = str(video_id)
        split = str(split_by_video[video_id])
        path = sam3_path(video_id, sam3_root=adapter.sam3_root)
        if not path.exists():
            missing.append(video_id)
            continue
        with h5py.File(path, "r") as h5_file:
            k = int(cast(int, h5_file.attrs["K"]))
            attrs_by_video[video_id] = {
                name: h5_file.attrs[name]
                for name in storage.PROVENANCE_ATTR_NAMES
                if name in h5_file.attrs
            }
            n_present_video = int(cast(h5py.Dataset, h5_file["v_t"])[:, 0].sum())

        structural_reasons = storage.validate_existing_file(
            path, expected_k=int(n_frames), v_t_dim=descriptors.V_T_DIM, expected_attrs={}
        )
        if structural_reasons:
            structurally_invalid[video_id] = structural_reasons

        if k != int(n_frames):
            mismatched.append(f"{video_id} (K={k}, frames.parquet={int(n_frames)})")
            continue
        frames_by_split[split] = frames_by_split.get(split, 0) + k
        total_present += n_present_video
        total_bytes += path.stat().st_size

    divergences = find_provenance_divergences(attrs_by_video)
    invalid_provenance_by_video: dict[str, list[str]] = {}
    for video_id, attrs in attrs_by_video.items():
        invalid_reasons = storage.find_invalid_required_provenance(
            attrs, storage.REQUIRED_NONEMPTY_PROVENANCE_ATTR_NAMES
        )
        if invalid_reasons:
            invalid_provenance_by_video[video_id] = invalid_reasons

    n_videos = len(group_sizes)
    total_frames = sum(frames_by_split.values())

    print(f"\nvídeos no manifesto de frames: {n_videos} (esperado {EXPECTED_VIDEO_COUNT})")
    if missing:
        print(f"\nvídeos sem .h5 de SAM 3 ({len(missing)}): {missing}")
    if mismatched:
        print(f"\nvídeos com K divergente ({len(mismatched)}): {mismatched}")
    if divergences:
        print(f"\ndivergências de proveniência ({len(divergences)}): {divergences}")
    if structurally_invalid:
        print(
            f"\n.h5 estruturalmente inválidos ({len(structurally_invalid)}): "
            f"{structurally_invalid}"
        )
    if invalid_provenance_by_video:
        print(
            "\nvídeos com atributo(s) de proveniência obrigatório(s) ausente(s), "
            "vazio(s) ou malformado(s) "
            f"({len(invalid_provenance_by_video)}): {invalid_provenance_by_video}"
        )

    print("\nquadros por split:")
    for split, expected in EXPECTED_SPLIT_FRAME_COUNTS.items():
        actual = frames_by_split.get(split, 0)
        print(f"  {split}: {actual} (esperado {expected})")
    print(f"  total: {total_frames} (esperado {EXPECTED_TOTAL_FRAMES})")

    coverage = (total_present / total_frames * 100.0) if total_frames else 0.0
    print(f"\nquadros com instância selecionada (present==1): {total_present} ({coverage:.2f}%)")
    print(f"\ntamanho total em disco: {total_bytes / (1024 ** 2):.2f} MiB")

    checks = [
        _check(f"vídeos == {EXPECTED_VIDEO_COUNT}", n_videos == EXPECTED_VIDEO_COUNT),
        _check("nenhum .h5 ausente", not missing),
        _check("nenhum K divergente", not mismatched),
        _check(f"total de quadros == {EXPECTED_TOTAL_FRAMES}", total_frames == EXPECTED_TOTAL_FRAMES),
        _check("proveniência idêntica em todos os .h5", not divergences),
        _check("nenhum .h5 estruturalmente inválido", not structurally_invalid),
        _check(
            "nenhum atributo de proveniência obrigatório ausente, vazio ou malformado",
            not invalid_provenance_by_video,
        ),
    ]
    for split, expected in EXPECTED_SPLIT_FRAME_COUNTS.items():
        checks.append(
            _check(
                f"quadros do split {split} == {expected}",
                frames_by_split.get(split, 0) == expected,
            )
        )

    if not all(checks):
        print("\nsam3 report FALHOU", file=sys.stderr)
        sys.exit(1)
    print("\nsam3 report OK: todas as checagens passaram")
