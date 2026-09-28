"""Configuração de determinismo usada pelos treinos das cinco armas."""

import os

import torch


_CUBLAS_WORKSPACE_CONFIG_DEFAULT = ":4096:8"
_CUBLAS_DETERMINISTIC_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})


def configure_determinism(seed: int) -> str:
    # CUBLAS_WORKSPACE_CONFIG precisa estar no ambiente do processo antes da
    # primeira chamada CUDA (abaixo) para que o cuBLAS use um algoritmo
    # determinístico; nada neste processo toca CUDA antes daqui.
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG_DEFAULT
    elif current not in _CUBLAS_DETERMINISTIC_WORKSPACE_CONFIGS:
        raise ValueError(
            f"CUBLAS_WORKSPACE_CONFIG={current!r} não garante determinismo do "
            "cuBLAS; defina uma das opções suportadas pelo PyTorch "
            f"({sorted(_CUBLAS_DETERMINISTIC_WORKSPACE_CONFIGS)}) ou remova a "
            f"variável para usar o padrão do projeto ({_CUBLAS_WORKSPACE_CONFIG_DEFAULT})."
        )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Retreinos com a mesma seed e as mesmas features devem produzir o
    # mesmo checkpoint nesta máquina/GPU/driver/cuDNN.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return device
