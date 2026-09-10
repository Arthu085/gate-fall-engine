"""Carregamento do backbone DINOv3 congelado, sem download nem inicialização aleatória."""

import os
import subprocess
from pathlib import Path
from typing import cast

import torch

DEFAULT_REPO_DIR = Path("data/scratch/dinov3_repo")
DEFAULT_WEIGHTS_PATH = Path(
    "data/scratch/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
MODEL_NAME = "dinov3_vitb16"
FEATURE_DIM = 1536
RESIZE_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

REPO_DIR_ENV_VAR = "GATEFALL_DINOV3_REPO_DIR"
WEIGHTS_PATH_ENV_VAR = "GATEFALL_DINOV3_WEIGHTS_PATH"


def resolve_repo_dir(cli_value: str | None) -> Path:
    if cli_value is not None:
        return Path(cli_value)
    env_value = os.environ.get(REPO_DIR_ENV_VAR)
    if env_value is not None:
        return Path(env_value)
    return DEFAULT_REPO_DIR


def resolve_weights_path(cli_value: str | None) -> Path:
    if cli_value is not None:
        return Path(cli_value)
    env_value = os.environ.get(WEIGHTS_PATH_ENV_VAR)
    if env_value is not None:
        return Path(env_value)
    return DEFAULT_WEIGHTS_PATH


def ensure_backbone_paths_exist(repo_dir: Path, weights_path: Path) -> None:
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"repositório do DINOv3 não encontrado: {repo_dir} — passe "
            f"--repo-dir, defina {REPO_DIR_ENV_VAR} ou clone-o no caminho "
            "padrão antes de rodar a extração"
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"pesos do DINOv3 não encontrados: {weights_path} — passe "
            f"--weights, defina {WEIGHTS_PATH_ENV_VAR} ou baixe-os no caminho "
            "padrão antes de rodar a extração"
        )


def configure_deterministic_inference() -> None:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_backbone(repo_dir: Path, weights_path: Path, device: str) -> torch.nn.Module:
    ensure_backbone_paths_exist(repo_dir, weights_path)
    configure_deterministic_inference()

    model = cast(
        torch.nn.Module,
        torch.hub.load(
            str(repo_dir), MODEL_NAME, source="local", weights=str(weights_path)
        ),
    )
    return model.to(device).eval()


def read_dinov3_repo_commit(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"não foi possível ler o commit do repositório DINOv3 em "
            f"{repo_dir} — verifique se é um clone git válido (com .git)"
        ) from exc
    return result.stdout.strip()
