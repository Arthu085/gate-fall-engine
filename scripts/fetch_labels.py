"""Baixa anotações do Le2i a partir do dataset OmniFall (HuggingFace)."""

import argparse

from gatefall.data.le2i.annotations import fetch_annotations, verify_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve arquivos já existentes em vez de pulá-los",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Não baixa nada; verifica a integridade dos arquivos já presentes"
        " contra PROVENANCE.json",
    )
    args = parser.parse_args()

    if args.verify:
        verify_annotations()
        return

    fetch_annotations(args.force)


if __name__ == "__main__":
    main()
