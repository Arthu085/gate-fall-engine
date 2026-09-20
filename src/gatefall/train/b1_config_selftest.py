"""Selftest sintético da configuração de treino do B1 (`b1_config.py`).

Trava o invariante experimental #1: fora dos campos exclusivos do gate, a
receita do B1 tem de bater campo a campo com a do B0 e com
`BASELINE_A_CONFIG`.
"""

import sys
import tempfile
from pathlib import Path

from gatefall.train.b0_config import B0_FUSION_CONFIG
from gatefall.train.b1_config import (
    B1_ADAPTIVE_GATE_CONFIG,
    B1TrainConfig,
    load_config,
    save_config,
)
from gatefall.train.config import BASELINE_A_CONFIG

_SHARED_RECIPE_FIELDS = (
    "kernel_size",
    "dilations",
    "channels",
    "dropout",
    "optimizer_name",
    "lr",
    "weight_decay",
    "grad_clip_norm",
    "lr_schedule_name",
    "batch_size",
    "epochs",
    "loss_name",
    "class_weighted",
    "seed",
    "window_frames",
    "train_stride",
    "eval_stride",
    "num_classes",
    "receptive_field",
)

_B1_ONLY_FIELDS = frozenset(
    {
        "gate_input_dim",
        "gate_output_dim",
        "gate_activation",
        "gate_weighted_branch",
        "quality_features_path",
        "quality_features_sha256",
    }
)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_shared_recipe_fields_match_baseline_a() -> bool:
    baseline_dict = BASELINE_A_CONFIG.to_dict()
    b1_dict = B1_ADAPTIVE_GATE_CONFIG.to_dict()
    mismatches = [
        field
        for field in _SHARED_RECIPE_FIELDS
        if b1_dict.get(field) != baseline_dict.get(field)
    ]
    return _check(
        "campos de receita compartilhada de B1_ADAPTIVE_GATE_CONFIG batem "
        f"byte a byte com BASELINE_A_CONFIG (mismatches={mismatches})",
        mismatches == [],
    )


def check_parity_with_b0_outside_b1_specific_fields() -> bool:
    b0_dict = B0_FUSION_CONFIG.to_dict()
    b1_dict = B1_ADAPTIVE_GATE_CONFIG.to_dict()

    extra_in_b1 = set(b1_dict) - set(b0_dict)
    missing_in_b1 = set(b0_dict) - set(b1_dict)
    mismatches = sorted(
        field
        for field in b0_dict
        if field not in {"run_name", "arm"} and b1_dict.get(field) != b0_dict[field]
    )
    ok = (
        extra_in_b1 == _B1_ONLY_FIELDS
        and missing_in_b1 == set()
        and mismatches == []
        and B1_ADAPTIVE_GATE_CONFIG.run_name == "b1_adaptive_gate"
        and B1_ADAPTIVE_GATE_CONFIG.arm == "B1"
    )
    return _check(
        "B1 difere de B0 apenas em run_name/arm e nos campos exclusivos do "
        f"gate {sorted(_B1_ONLY_FIELDS)} (extras={sorted(extra_in_b1)}, "
        f"ausentes={sorted(missing_in_b1)}, divergentes={mismatches})",
        ok,
    )


def check_gate_audit_fields() -> bool:
    ok = (
        B1_ADAPTIVE_GATE_CONFIG.pose_dim == 134
        and B1_ADAPTIVE_GATE_CONFIG.visual_dim == 1536
        and B1_ADAPTIVE_GATE_CONFIG.projection_dim == 128
        and B1_ADAPTIVE_GATE_CONFIG.fused_dim == 256
        and B1_ADAPTIVE_GATE_CONFIG.gate_input_dim == 2
        and B1_ADAPTIVE_GATE_CONFIG.gate_output_dim == 1
        and B1_ADAPTIVE_GATE_CONFIG.gate_activation == "sigmoid"
        and B1_ADAPTIVE_GATE_CONFIG.gate_weighted_branch == "pose"
    )
    return _check(
        "campos exclusivos de auditoria do B1 (gate_input_dim=2, "
        "gate_output_dim=1, gate_activation=sigmoid, gate_weighted_branch="
        "pose) e fused_dim inalterado em 256",
        ok,
    )


def check_save_load_round_trip() -> bool:
    config = B1_ADAPTIVE_GATE_CONFIG
    data = config.to_dict()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "b1_config_roundtrip.yaml"
        save_config(config, path, force=True)
        loaded = load_config(path)

        ok = isinstance(loaded, B1TrainConfig) and loaded.to_dict() == data
        ok = ok and bool(path.read_text(encoding="utf-8"))
    return _check(
        "save/load: round-trip do B1TrainConfig preserva todos os campos", ok
    )


def run_b1_config_selftest() -> bool:
    checks = [
        check_shared_recipe_fields_match_baseline_a(),
        check_parity_with_b0_outside_b1_specific_fields(),
        check_gate_audit_fields(),
        check_save_load_round_trip(),
    ]
    ok = all(checks)
    if not ok:
        print("\nb1 config selftest FALHOU", file=sys.stderr)
    else:
        print("\nb1 config selftest OK: todas as checagens passaram")
    return ok
