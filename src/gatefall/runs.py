"""Guardas compartilhadas para separar runs locais de referências históricas."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_RUN_ROOT = (REPOSITORY_ROOT / "runs/reference").resolve()

LOCAL_RUN_ROOTS: dict[str, Path] = {
    "le2i": Path("runs/local/le2i"),
    "le2i-cv": Path("runs/local/le2i_cv"),
}


def default_run_dir(dataset: str) -> Path:
    try:
        root = LOCAL_RUN_ROOTS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"dataset não suportado para run_dir padrão: {dataset!r}; opções "
            f"disponíveis: {', '.join(LOCAL_RUN_ROOTS)}"
        ) from exc
    return root / "baseline_a"


def validate_local_run_dir(run_dir: Path, dataset: str | None = None) -> None:
    resolved = run_dir.resolve()
    if resolved == REFERENCE_RUN_ROOT or REFERENCE_RUN_ROOT in resolved.parents:
        raise ValueError(
            f"run_dir aponta para referência histórica somente leitura: {run_dir}"
        )
    if dataset is None:
        return
    for other_dataset, other_root in LOCAL_RUN_ROOTS.items():
        if other_dataset == dataset:
            continue
        other_resolved = (REPOSITORY_ROOT / other_root).resolve()
        if resolved == other_resolved or other_resolved in resolved.parents:
            raise ValueError(
                f"run_dir {run_dir} pertence ao protocolo {other_dataset!r}, "
                f"incompatível com --dataset {dataset!r}"
            )
