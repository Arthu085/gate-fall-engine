"""Selftest sintético da configuração de treino do C0 (`c0_config.py`).

Trava, sobretudo, o invariante experimental #1 do repositório: A, B e C só
podem diferir no vetor de feature por timestep. Qualquer campo da receita de
treino compartilhada (arquitetura da TCN, otimizador, agenda, seed, janela,
número de classes) precisa bater byte a byte com `BASELINE_A_CONFIG`.
"""

import sys
import tempfile
from pathlib import Path

from gatefall.train.c0_config import C0_FUSION_CONFIG, C0TrainConfig, load_config, save_config
from gatefall.train.b0_config import B0_FUSION_CONFIG
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


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def check_shared_recipe_fields_match_baseline_a() -> bool:
    baseline_dict = BASELINE_A_CONFIG.to_dict()
    c0_dict = C0_FUSION_CONFIG.to_dict()
    mismatches = [
        field
        for field in _SHARED_RECIPE_FIELDS
        if c0_dict.get(field) != baseline_dict.get(field)
    ]
    return _check(
        "campos de receita compartilhada de C0_FUSION_CONFIG batem byte a "
        f"byte com BASELINE_A_CONFIG (mismatches={mismatches})",
        mismatches == [],
    )


def check_c0_only_audit_fields() -> bool:
    ok = (
        C0_FUSION_CONFIG.pose_dim == 134
        and C0_FUSION_CONFIG.visual_dim == 10
        and C0_FUSION_CONFIG.projection_dim == 128
        and C0_FUSION_CONFIG.fused_dim == 256
    )
    return _check(
        "campos exclusivos de auditoria do C0 (pose_dim=134, visual_dim=10, "
        "projection_dim=128, fused_dim=256)",
        ok,
    )


def check_differs_from_b0_only_in_visual_source() -> bool:
    b0_dict = B0_FUSION_CONFIG.to_dict()
    c0_dict = C0_FUSION_CONFIG.to_dict()
    shared_fields = set(b0_dict) & set(c0_dict)
    differing = {field for field in shared_fields if b0_dict[field] != c0_dict[field]}
    return _check(
        "C0_FUSION_CONFIG só difere de B0_FUSION_CONFIG em identidade da arma "
        f"e largura da fonte visual (divergentes={sorted(differing)})",
        differing == {"run_name", "arm", "visual_dim"},
    )


def check_save_load_round_trip() -> bool:
    config = C0_FUSION_CONFIG
    data = config.to_dict()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "c0_config_roundtrip.yaml"
        save_config(config, path, force=True)
        loaded = load_config(path)

        ok = isinstance(loaded, C0TrainConfig) and loaded.to_dict() == data
        raw_text = path.read_text(encoding="utf-8")
        ok = ok and bool(raw_text)
    return _check("save/load: round-trip do C0TrainConfig preserva todos os campos", ok)


def run_c0_config_selftest() -> bool:
    checks = [
        check_shared_recipe_fields_match_baseline_a(),
        check_c0_only_audit_fields(),
        check_differs_from_b0_only_in_visual_source(),
        check_save_load_round_trip(),
    ]
    ok = all(checks)
    if not ok:
        print("\nc0 config selftest FALHOU", file=sys.stderr)
    else:
        print("\nc0 config selftest OK: todas as checagens passaram")
    return ok
