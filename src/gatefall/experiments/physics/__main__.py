"""CLI de preparação, treino, avaliação e verificação da fusão experimental."""

import argparse
from pathlib import Path

from gatefall.experiments.physics.data import VISUAL_DIMS, prepare_dataset
from gatefall.experiments.physics.model import FUSIONS
from gatefall.experiments.physics.training import evaluate, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepara pacote offline a partir de fontes existentes")
    prep.add_argument("--dataset", choices=("le2i", "le2i-cv"), default="le2i")
    prep.add_argument("--visual", choices=tuple(VISUAL_DIMS), required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--max-gap-seconds", type=float, default=.5)
    prep.add_argument("--outage-probability", type=float, default=.05)
    prep.add_argument("--outage-frames", type=int, default=3)
    training = commands.add_parser("train", help="Treina sem consultar o teste")
    training.add_argument("--bundle", type=Path, required=True)
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--arm", choices=("A", "B0", "B1", "C0", "C1"), required=True)
    training.add_argument("--fusion", choices=FUSIONS, required=True)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--checkpoint-policy", choices=("best_val", "last"), default="best_val")
    training.add_argument("--augment", action="store_true")
    training.add_argument("--order-weight", type=float, default=0)
    training.add_argument("--order-margin", type=float, default=.05)
    evaluation = commands.add_parser("evaluate", help="Avalia explicitamente um split com o checkpoint escolhido")
    evaluation.add_argument("--bundle", type=Path, required=True)
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--split", choices=("train", "val", "test"), default="test")
    commands.add_parser("selftest", help="Verifica contratos sintéticos em CPU")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["visual_kind"] = args.pop("visual")
        print(prepare_dataset(**args))
    elif command == "train":
        result = train(**args)
        print(f"Checkpoint selecionado: época {result['selected_epoch']}")
    elif command == "evaluate":
        result = evaluate(**args)
        print(f"Macro-F1: {result['classification']['macro_f1_restricted']:.4f}; "
              f"FA/h: {result['events']['false_alarms_per_hour']:.4f}")
    else:
        from gatefall.experiments.physics.selftest import run_selftest
        run_selftest()


if __name__ == "__main__":
    main()
