"""Embed a pre-chunked JSONL knowledge corpus into ChromaDB.

    python scripts/ingest_camara.py [--reset] [--jsonl-dir data/camara_docs] [--jsonl FILE ...]

Equivalent to ``python -m camara_pgps ingest-camara``.
"""

from __future__ import annotations

import argparse

from camara_pgps.retrieval import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    parser.add_argument(
        "--jsonl", action="append", default=[], metavar="PATH",
        help="a JSONL corpus file (repeatable)",
    )
    parser.add_argument(
        "--jsonl-dir",
        default=None,
        help="directory of *.jsonl files (default: retrieval.knowledge_dir)",
    )
    args = parser.parse_args()

    result = run_ingestion(
        jsonl_paths=args.jsonl or None,
        jsonl_dir=args.jsonl_dir,
        reset=args.reset,
    )
    print(
        f"ingested {result.chunks} chunk(s) ({result.written} written) "
        f"into '{result.collection}' at {result.persist_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
