"""Command-line entry point.

This is a thin stub for the project-foundation milestone; the full plan-synthesis
CLI is built in a later milestone. It is wired to the ``camara-pgps`` console
script declared in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse

from camara_pgps.__about__ import __version__
from camara_pgps.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camara-pgps",
        description="Precondition-Guided Plan Synthesis for CAMARA API orchestration.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"camara-pgps {__version__}",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved configuration as JSON and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_config:
        print(get_settings().model_dump_json(indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
