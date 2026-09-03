"""Run the CAMARA-PGPS agent on a request.

    python scripts/run_agent.py "Improve the network quality for my video call"
    python scripts/run_agent.py --request "..." --mode plan --json

Equivalent to ``python -m camara_pgps run ...`` - a bare first argument is taken
as the request text.
"""

from __future__ import annotations

import sys

from camara_pgps.cli import main


def _to_run_argv(raw: list[str]) -> list[str]:
    if raw and not raw[0].startswith("-"):
        return ["run", "--request", raw[0], *raw[1:]]
    return ["run", *raw]


if __name__ == "__main__":
    raise SystemExit(main(_to_run_argv(sys.argv[1:])))
