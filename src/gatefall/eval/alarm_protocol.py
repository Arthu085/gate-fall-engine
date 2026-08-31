"""Protocolo de alarme (gatilho, refratário e associação de eventos) das armas do GateFall."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from gatefall.config import EVAL_STRIDE, TARGET_FPS


@dataclass(frozen=True)
class AlarmProtocol:
    fall_label: int
    fallen_label: int
    positive_labels: list[int]
    trigger_consecutive: int
    refractory_period_s: float
    association_end_offset_s: float
    fallback_association_uses_fall_end: bool
    eval_stride: int
    target_fps: float
    latency_decimal_places: int
    pre_fall_diagnostic_window_s: float
    pre_fall_alarms_count_as_false_alarms: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "AlarmProtocol":
        return AlarmProtocol(**data)


BASELINE_A_ALARM_PROTOCOL = AlarmProtocol(
    fall_label=1,
    fallen_label=2,
    positive_labels=[1, 2],
    trigger_consecutive=3,
    refractory_period_s=5.0,
    association_end_offset_s=2.0,
    fallback_association_uses_fall_end=True,
    eval_stride=EVAL_STRIDE,
    target_fps=TARGET_FPS,
    latency_decimal_places=1,
    pre_fall_diagnostic_window_s=1.0,
    pre_fall_alarms_count_as_false_alarms=True,
)


def save_alarm_protocol(protocol: AlarmProtocol, path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"skip {path} (já existe, use --force para sobrescrever)")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    data = protocol.to_dict()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    os.replace(tmp_path, path)

    with path.open("r", encoding="utf-8") as f:
        read_back = yaml.safe_load(f)
    if read_back != data:
        raise RuntimeError(
            f"verificação de leitura pós-gravação falhou para {path}: conteúdo "
            "lido não bate byte a byte com o conteúdo gravado"
        )

    print(f"{path}: protocolo de alarme gravado")
    return True


def load_alarm_protocol(path: Path) -> AlarmProtocol:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AlarmProtocol.from_dict(data)
