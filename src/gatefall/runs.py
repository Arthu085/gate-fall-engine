"""Guardas compartilhadas para separar runs locais de referências históricas."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_RUN_ROOT = (REPOSITORY_ROOT / "runs/reference").resolve()


def validate_local_run_dir(run_dir: Path) -> None:
    resolved = run_dir.resolve()
    if resolved == REFERENCE_RUN_ROOT or REFERENCE_RUN_ROOT in resolved.parents:
        raise ValueError(
            f"run_dir aponta para referência histórica somente leitura: {run_dir}"
        )
