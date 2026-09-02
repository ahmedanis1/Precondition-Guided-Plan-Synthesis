"""Run the CAMARA knowledge ingestion pipeline.

    python scripts/ingest_camara.py [--reset] [--docs-dir path/to/markdown]

Ingests the local CAMARA API registry (and, optionally, a directory of Markdown
docs) into the configured ChromaDB collection.
"""

from __future__ import annotations

import argparse

from camara_pgps.retrieval import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    parser.add_argument(
        "--docs-dir", default=None, help="directory of Markdown docs to also ingest"
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="skip ingesting the local API registry",
    )
    args = parser.parse_args()

    result = run_ingestion(
        include_registry=not args.no_registry,
        docs_dir=args.docs_dir,
        reset=args.reset,
    )
    print(
        f"ingested {result.documents} document(s) -> {result.chunks} chunk(s) "
        f"({result.written} written) into '{result.collection}' at {result.persist_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
