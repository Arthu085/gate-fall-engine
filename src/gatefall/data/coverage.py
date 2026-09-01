"""CLI de auditoria de cobertura das anotações sobre o manifesto Le2i."""

import argparse

from gatefall.data.le2i.coverage import audit_le2i_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audita a cobertura dos segmentos anotados sobre a duração dos vídeos",
    )
    audit_parser.add_argument("--dataset", default="le2i", choices=("le2i",))

    args = parser.parse_args()
    if args.command == "audit":
        audit_le2i_coverage()


if __name__ == "__main__":
    main()
