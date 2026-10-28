"""Experimento pareado de supervisão de movimento e coerência cinemática."""

import argparse
from pathlib import Path

from gatefall.experiments.motion import RECIPES
from gatefall.experiments.motion.data import prepare
from gatefall.experiments.motion.report import summarize
from gatefall.experiments.physics.model import FUSIONS
from gatefall.experiments.physics.training import evaluate, train
from gatefall.runs import validate_local_run_dir


def add_training(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B0", "B1", "C0", "C1"), required=True)
    parser.add_argument("--fusion", choices=FUSIONS, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--motion-weight", type=float, default=.1)
    parser.add_argument("--physics-weight", type=float, default=.1)
    parser.add_argument("--checkpoint-policy", choices=("best_val", "last"), default="best_val")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--order-weight", type=float, default=0)
    parser.add_argument("--order-margin", type=float, default=.05)


def compare(output: Path, seeds: list[int], **options) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("sementes não negativas, distintas e não vazias são obrigatórias")
    output = output.resolve()
    validate_local_run_dir(output)
    if output.exists():
        raise FileExistsError(output)
    runs = []
    # Completar todos os treinos antes de consultar qualquer resultado de teste.
    for seed in seeds:
        for recipe in RECIPES:
            path = output / f"seed-{seed}" / recipe
            train(run_dir=path, seed=seed, motion_recipe=recipe, **options)
            runs.append(path)
    for path in runs:
        evaluate(options["bundle"], path, "test")
    return summarize(runs, output / "summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Alvos de treino a partir das poses originais")
    prep.add_argument("--bundle", type=Path, required=True)
    prep.add_argument("--pose-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
    prep.add_argument("--confidence", type=float, default=.5)
    training = sub.add_parser("train", help="Treina uma receita sem consultar teste")
    add_training(training)
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--recipe", choices=RECIPES, required=True, dest="motion_recipe")
    matrix = sub.add_parser("compare", help="Treina e avalia três receitas pareadas")
    add_training(matrix)
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    evaluation = sub.add_parser("evaluate", help="Classificação causal sem alvos futuros")
    evaluation.add_argument("--bundle", type=Path, required=True)
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluation.add_argument("--equivalence", type=Path,
                            help="Atestado que liga um run histórico ao pacote atual; ignorado se o run usa o atual")
    summary = sub.add_parser("summarize", help="Recusa comparações incompatíveis")
    summary.add_argument("--runs", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    sub.add_parser("selftest", help="Contratos analíticos e fluxo sintético CPU")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "selftest":
        from gatefall.experiments.motion.selftest import run_selftest
        run_selftest()
    else:
        result = {"prepare": prepare, "train": train, "evaluate": evaluate,
                  "compare": compare, "summarize": summarize}[command](**args)
        if command == "prepare":
            print(result)
        elif command == "train":
            print(f"Checkpoint: época {result['selected_epoch']}")
        elif command == "evaluate":
            print(f"F1={result['classification']['macro_f1_restricted']:.4f}; "
                  f"FA/h={result['events']['false_alarms_per_hour']:.4f}")
        else:
            print(f"Resumo pronto: {len(result['seeds'])} sementes; seleção pelo teste: não")


if __name__ == "__main__":
    main()
