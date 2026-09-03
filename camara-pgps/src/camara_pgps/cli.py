"""Command-line interface.

    python -m camara_pgps run --request "Improve the network quality for my video call"
    python -m camara_pgps list-apis [--api quality-on-demand] [--json]
    python -m camara_pgps ingest-camara [--reset] [--docs-dir data/camara_docs]
    python -m camara_pgps show-config

Wired to the ``camara-pgps`` console script. Heavy dependencies (LangGraph,
ChromaDB, ...) are imported lazily inside each sub-command.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from camara_pgps.__about__ import __version__
from camara_pgps.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camara-pgps",
        description="Precondition-Guided Plan Synthesis for CAMARA API orchestration.",
    )
    parser.add_argument("--version", action="version", version=f"camara-pgps {__version__}")
    parser.add_argument("--show-config", action="store_true", help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the agent on a natural-language request.")
    run_p.add_argument("--request", "-r", required=True, help="The request text.")
    run_p.add_argument(
        "--mode",
        choices=("run", "plan", "validate", "execute"),
        default="run",
        help="How far to take the pipeline (default: run = end to end).",
    )
    run_p.add_argument("--arguments", help='JSON: {"operationId": {"param": value}}.')
    run_p.add_argument("--known-facts", help='JSON: {"fact.key": value}.')
    run_p.add_argument("--json", action="store_true", help="Print the full run as JSON.")

    apis_p = sub.add_parser("list-apis", help="List the registered CAMARA APIs.")
    apis_p.add_argument("--api", help="Show one API (operations, parameters, conditions).")
    apis_p.add_argument("--json", action="store_true")

    ingest_p = sub.add_parser(
        "ingest-camara", help="(Re)build the RAG knowledge base in ChromaDB."
    )
    ingest_p.add_argument("--reset", action="store_true", help="Drop the collection first.")
    ingest_p.add_argument("--docs-dir", help="Directory of Markdown docs to also ingest.")
    ingest_p.add_argument(
        "--no-registry", action="store_true", help="Skip ingesting the API registry."
    )

    sub.add_parser("show-config", help="Print the resolved configuration as JSON.")
    return parser


# --------------------------------------------------------------------------- #
# sub-commands
# --------------------------------------------------------------------------- #

def _cmd_show_config() -> int:
    print(get_settings().model_dump_json(indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from camara_pgps.workflow import get_workflow
    from camara_pgps.workflow.report import explain, state_report

    arguments: dict[str, Any] | None = (
        json.loads(args.arguments) if args.arguments else None
    )
    known_facts: dict[str, Any] | None = (
        json.loads(args.known_facts) if args.known_facts else None
    )

    workflow = get_workflow()
    if args.mode in ("run", "plan"):
        state = getattr(workflow, args.mode)(
            args.request, arguments=arguments, known_facts=known_facts
        )
    else:
        state = getattr(workflow, args.mode)(
            request_text=args.request, arguments=arguments, known_facts=known_facts
        )

    if args.json:
        print(json.dumps(state_report(state), indent=2, default=str))
        return 0 if state.stage.value != "failed" else 1

    print(explain(state))
    provenance = state.metadata.get("provenance")
    if provenance and provenance.get("narrative"):
        print("\n" + provenance["narrative"])
    elif state.plan is not None and state.plan.steps:
        print("\nPlan:")
        for step in state.plan.steps:
            print(f"  {step.index + 1}. {step.operation_id}  {step.method} {step.path}")
    if state.clarification:
        assistant = [m for m in state.messages if m.get("role") == "assistant"]
        if assistant:
            print("\n" + assistant[-1]["content"])
    if state.execution is not None:
        verdict = "OK" if state.execution.succeeded else "FAILED"
        print(f"\nExecution: {verdict}")
        for step_exec in state.execution.steps:
            print(
                f"  step {step_exec.step_index}: {step_exec.operation_id} "
                f"-> {step_exec.status_code}"
            )
    for error in state.errors:
        print(f"error: {error}", file=sys.stderr)
    return 0 if state.stage.value != "failed" else 1


def _cmd_list_apis(args: argparse.Namespace) -> int:
    from camara_pgps.camara.registry import get_registry

    registry = get_registry()

    if args.api:
        api = registry.get_api(args.api)
        if api is None:
            print(
                f"unknown API {args.api!r}; known: {registry.api_names()}",
                file=sys.stderr,
            )
            return 2
        if args.json:
            print(api.model_dump_json(indent=2))
            return 0
        print(f"{api.name}  v{api.version}  {api.title or ''}".rstrip())
        if api.description:
            print(f"  {api.description}")
        for op in api.operations:
            capability = registry.capability_of(op.operation_id)
            print(f"  {op.operation_id}  [{op.method} {op.path}]  capability={capability}")
            required = [p.name for p in op.parameters if p.required]
            if required:
                print(f"    required parameters: {', '.join(required)}")
            for condition in op.preconditions:
                print(f"    precondition: {condition.description}  requires {condition.requires}")
            for condition in op.postconditions:
                print(f"    postcondition: {condition.description}  produces {condition.produces}")
        return 0

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": api.name,
                        "version": api.version,
                        "operations": [op.operation_id for op in api.operations],
                    }
                    for api in registry.list_apis()
                ],
                indent=2,
            )
        )
        return 0

    for api in registry.list_apis():
        ops = ", ".join(op.operation_id for op in api.operations)
        print(f"{api.name:<20} v{api.version:<10} {len(api.operations)} operation(s): {ops}")
    return 0


def _cmd_ingest_camara(args: argparse.Namespace) -> int:
    from camara_pgps.retrieval import run_ingestion

    result = run_ingestion(
        include_registry=not args.no_registry,
        docs_dir=args.docs_dir,
        reset=args.reset,
    )
    print(
        f"ingested {result.documents} document(s) -> {result.chunks} chunk(s) "
        f"({result.written} written) into '{result.collection}' "
        f"at {result.persist_directory}"
    )
    return 0


_COMMANDS = {
    "run": _cmd_run,
    "list-apis": _cmd_list_apis,
    "ingest-camara": _cmd_ingest_camara,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_config or args.command == "show-config":
        return _cmd_show_config()

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
