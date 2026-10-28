"""Verificação do TCNTE adaptado; treino e avaliação usam `gatefall.experiments.physics`."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("selftest", help="Contratos sintéticos em CPU")
    parser.parse_args()
    from gatefall.experiments.tcnte.selftest import run_selftest

    run_selftest()


if __name__ == "__main__":
    main()
