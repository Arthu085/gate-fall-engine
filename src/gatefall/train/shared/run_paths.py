"""Caminhos e guardas de run compartilhados entre armas."""

from collections.abc import Mapping
from pathlib import Path

import yaml

from gatefall.runs import REPOSITORY_ROOT, default_run_dir_for_arm


def repository_anchored_run_dir(run_dir: Path) -> Path:
    """Resolve um run_dir padrão (relativo) contra a raiz do repositório.

    `default_run_dir_for_arm` devolve caminho relativo: um `.resolve()` direto
    o ancoraria no cwd e faria a guarda virar no-op quando a CLI roda de outro
    diretório. `validate_local_run_dir` já ancora em `REPOSITORY_ROOT`.
    """
    return (REPOSITORY_ROOT / run_dir).resolve()


def guard_not_foreign_arm_run_dir(run_dir: Path, arm: str) -> None:
    """Recusa sobrescrever um run_dir que já pertence a outra arma.

    As guardas de CLI só cobrem os run_dirs canônicos; um run não canônico de
    outra arma (por exemplo `b0_fusion_seed7`) passaria por elas e seria
    substituído pelo promote com `--force`.
    """
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return
    try:
        with config_path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, ValueError, yaml.YAMLError):
        return
    if not isinstance(data, Mapping):
        return
    declared_arm = data.get("arm")
    if declared_arm is not None and declared_arm != arm:
        raise RuntimeError(
            f"run_dir {run_dir} já contém um run da arma {declared_arm!r}, "
            f"incompatível com a arma {arm!r}; escolha um run_dir próprio — "
            "nem --force sobrescreve o run de outra arma"
        )


COMPARISON_ARM_NAMES: dict[str, str] = {
    "A": "baseline_a",
    "B0": "b0_fusion",
    "B1": "b1_adaptive_gate",
}


def comparison_run_dirs(dataset_name: str) -> dict[str, Path]:
    return {
        arm: repository_anchored_run_dir(default_run_dir_for_arm(dataset_name, arm_name))
        for arm, arm_name in COMPARISON_ARM_NAMES.items()
    }
