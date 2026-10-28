"""Especificações congeladas dos datasets encenados: fontes, partições e caminhos."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SourceKind = Literal["image_sequence", "video"]
SPLIT_NAMES = ("train", "val", "test")
INTEGER_IDENTITY_FIELDS = frozenset({"subject", "activity", "trial", "camera"})


@dataclass(frozen=True)
class SubjectSplit:
    version: str
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = (self.train, self.val, self.test)
        flat = [subject for group in groups for subject in group]
        if len(flat) != len(set(flat)):
            raise ValueError(f"{self.version}: sujeito repetido entre partições")

    def split_of(self, subject: int) -> str:
        for name, group in zip(SPLIT_NAMES, (self.train, self.val, self.test)):
            if subject in group:
                return name
        raise ValueError(f"{self.version}: sujeito {subject} sem partição congelada")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }


@dataclass(frozen=True)
class StagedDatasetSpec:
    identifier: str
    omnifall_name: str
    omnifall_reference_config: str | None
    source_kind: SourceKind
    raw_dir: Path
    processed_dir: Path
    features_dir: Path
    labels_dir: Path
    env: str
    split: SubjectSplit
    role: Literal["primary", "external_test"]
    video_id_pattern: re.Pattern[str]

    def parse_video_id(self, video_id: str) -> dict[str, int | str]:
        match = self.video_id_pattern.fullmatch(video_id)
        if match is None:
            raise ValueError(f"{self.identifier}: video_id fora do padrão: {video_id!r}")
        return {
            key: (int(value) if key in INTEGER_IDENTITY_FIELDS else value)
            for key, value in match.groupdict().items()
        }

    def recording_id(self, video_id: str) -> str:
        """Identidade da gravação física; vistas sincronizadas compartilham esse valor."""
        if self.identifier == "upfall":
            return video_id.rsplit("/", 1)[0]
        return video_id

    def source_relative_path(self, video_id: str) -> Path:
        parts = self.parse_video_id(video_id)
        if self.identifier == "upfall":
            return Path(video_id)
        return Path(f"Subject {parts['subject']}") / str(parts["category"]) / f"{parts['clip']}.mp4"


# Partição por sujeito do UP-Fall, versão 1. O split `up_fall-cs` do OmniFall
# na revisão pinada repete os sujeitos 11, 15 e 16 em train e em val/test (os
# mesmos caminhos aparecem nos dois lados), o que viola o invariante 2. A
# correção mínima mantém os sujeitos de val/test designados pelo OmniFall e os
# remove de train; nenhum sujeito ou gravação aparece em mais de uma partição.
UPFALL_SPLIT = SubjectSplit(
    version="upfall-subject-v1",
    train=(1, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 17),
    val=(11,),
    test=(2, 4, 15, 16),
)

# GMDCSA-24 é teste externo reservado: nenhum vídeo entra em treino, validação,
# seleção de checkpoint, limiar ou normalização.
GMDCSA24_SPLIT = SubjectSplit(
    version="gmdcsa24-external-test-v1",
    train=(),
    val=(),
    test=(1, 2, 3, 4),
)

UPFALL_SPEC = StagedDatasetSpec(
    identifier="upfall",
    omnifall_name="up_fall",
    omnifall_reference_config="up_fall-cs",
    source_kind="image_sequence",
    raw_dir=Path("data/raw/upfall"),
    processed_dir=Path("data/processed/upfall"),
    features_dir=Path("data/features/upfall"),
    labels_dir=Path("data/labels/omnifall_upfall"),
    env="up_fall_lab",
    split=UPFALL_SPLIT,
    role="primary",
    video_id_pattern=re.compile(
        r"Subject(?P<subject>\d+)/Activity(?P<activity>\d+)/Trial(?P<trial>\d+)/"
        r"Subject(?P=subject)Activity(?P=activity)Trial(?P=trial)Camera(?P<camera>\d+)"
    ),
)

GMDCSA24_SPEC = StagedDatasetSpec(
    identifier="gmdcsa24",
    omnifall_name="GMDCSA24",
    omnifall_reference_config="gmdcsa24-cs",
    source_kind="video",
    raw_dir=Path("data/raw/gmdcsa24/GMDCSA24"),
    processed_dir=Path("data/processed/gmdcsa24_external"),
    features_dir=Path("data/features/gmdcsa24_external"),
    labels_dir=Path("data/labels/omnifall_gmdcsa24"),
    env="gmdcsa24_home",
    split=GMDCSA24_SPLIT,
    role="external_test",
    video_id_pattern=re.compile(
        r"Subject_(?P<subject>\d+)/(?P<category>ADL|Fall)/(?P<clip>\d{2})"
    ),
)

STAGED_SPECS: dict[str, StagedDatasetSpec] = {
    spec.identifier: spec for spec in (UPFALL_SPEC, GMDCSA24_SPEC)
}


def get_staged_spec(identifier: str) -> StagedDatasetSpec:
    try:
        return STAGED_SPECS[identifier]
    except KeyError as exc:
        raise ValueError(
            f"dataset encenado não suportado: {identifier!r}; opções: {', '.join(STAGED_SPECS)}"
        ) from exc
