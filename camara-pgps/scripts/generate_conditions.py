"""Draft a PGPS condition file from a raw CAMARA OpenAPI spec.

    python scripts/generate_conditions.py data/camara_docs/<api-name>.yaml
    python scripts/generate_conditions.py data/camara_docs/<api-name>.yaml --dataset data/camara_docs/<api-name>.dataset.jsonl
    python scripts/generate_conditions.py --check <api-name>

What this script CAN do reliably: every required path/query parameter and
every required request-body field is a mechanical fact about the spec - a
required field is a required field, no judgment involved - so it drafts
``requires`` for you and will never again silently drop one the way two
manual passes earlier in this project's history did (an unresolved $ref
parameter, and a required path param excluded because it looked like a
$ref instead of being resolved first).

What this script CANNOT do, and does not pretend to: decide what an
operation *produces*. An OpenAPI response schema describes what comes back,
not what the CAMARA lifecycle promises about it - "createSession produces a
fact called qos.session.id that getSession/deleteSession will later need"
is a judgment call about the API's semantics, not something derivable from
the spec. So the draft leaves "produces" empty and instead lists the
success response schema's own top-level field names as
"produces_candidates" - a memory-jog, not an answer - for you to turn into
real facts (or discard) by hand.

Similarly, the drafted "requires" entries are the RAW field names from the
spec (e.g. "qosProfile", "duration"), not renamed into this project's
dotted fact-key convention (e.g. "qos.profile.name") - that renaming is
what makes two operations agree they're talking about the same fact, which
is itself a semantic decision only you can make correctly. Rename them
during review; the script's job is to guarantee the *list* of names is
complete and correct, not to guess what to call them.

Output goes to ``data/conditions/<api-name>.conditions.draft.json`` -
deliberately NOT ``*.conditions.json``, so PGPSPlanner (which only loads
``*.conditions.json``) never picks up an unreviewed draft by accident.
Review it, rename/trim the facts, delete the "_raw"/"_candidates" scratch
keys once you've used them, then rename the file to drop ".draft" to make
it live.

``--check <api-name>`` instead validates an already-promoted
``data/conditions/<api-name>.conditions.json``: every key must be a real
operation_id from the matching ``.dataset.jsonl`` (catches typos), and no
operation_id may repeat across any file in ``data/conditions/`` (catches
cross-API collisions) - the two checks done by hand for all 14 operations
earlier in this project's history, now repeatable on demand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camara_pgps.camara.openapi_parser import OpenAPIParser  # noqa: E402
from camara_pgps.models import ParameterLocation  # noqa: E402
from camara_pgps.pgps.conditions_loader import (  # noqa: E402
    DuplicateConditionError,
    load_operation_conditions,
)

CONDITIONS_DIR = PROJECT_ROOT / "data" / "conditions"
CAMARA_DOCS_DIR = PROJECT_ROOT / "data" / "camara_docs"


def _dataset_operation_ids(dataset_path: Path) -> set[str]:
    ids: set[str] = set()
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        op_id = (record.get("structured") or {}).get("operation_id")
        if op_id:
            ids.add(op_id)
    return ids


def generate(spec_path: Path, dataset_path: Path | None, out_path: Path | None) -> Path:
    api = OpenAPIParser().parse_file(spec_path)
    api_name = api.name

    draft: dict[str, dict] = {}
    report_lines: list[str] = [f"Draft for API {api_name!r} ({len(api.operations)} operations)\n"]

    spec_ids = {op.operation_id for op in api.operations}

    dataset_ids: set[str] | None = None
    if dataset_path is not None:
        if not dataset_path.is_file():
            raise SystemExit(f"--dataset file not found: {dataset_path}")
        dataset_ids = _dataset_operation_ids(dataset_path)
        missing_from_dataset = spec_ids - dataset_ids
        missing_from_spec = dataset_ids - spec_ids
        if missing_from_dataset:
            report_lines.append(
                f"  WARNING: in the spec but not in {dataset_path.name}: {sorted(missing_from_dataset)}"
            )
        if missing_from_spec:
            report_lines.append(
                f"  WARNING: in {dataset_path.name} but not parsed from the spec: {sorted(missing_from_spec)}"
            )
        if not missing_from_dataset and not missing_from_spec:
            report_lines.append(f"  operation_ids match {dataset_path.name} exactly ({len(spec_ids)} ids). OK")

    for op in sorted(api.operations, key=lambda o: o.operation_id):
        required_body = sorted(
            p.name for p in op.parameters if p.location is ParameterLocation.BODY and p.required
        )
        required_other = sorted(
            p.name for p in op.parameters if p.location is not ParameterLocation.BODY and p.required
        )
        optional_body = sorted(
            p.name for p in op.parameters if p.location is ParameterLocation.BODY and not p.required
        )
        scopes = op.security_scopes
        produces_candidates = sorted((op.response_schema or {}).get("properties", {}) or {})

        requires_draft = (["auth.token"] if scopes else []) + required_other + required_body

        draft[op.operation_id] = {
            "requires": requires_draft,
            "produces": [],
            "_requires_raw_from_spec": {
                "path_or_query": required_other,
                "body_required": required_body,
                "body_optional": optional_body,
                "security_scopes": scopes,
            },
            "_produces_candidates_from_response_schema": produces_candidates,
        }

        report_lines.append(f"- {op.operation_id} ({op.method} {op.path})")
        report_lines.append(f"    requires (draft, raw names): {requires_draft or '[]'}")
        report_lines.append(
            f"    produces: [] -- candidates from the response schema, for you to judge: {produces_candidates or '(none)'}"
        )

    out_path = out_path or (CONDITIONS_DIR / f"{api_name}.conditions.draft.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines.append(f"\nWritten: {out_path}")
    report_lines.append(
        "Next: review each operation above, rename 'requires' entries into this "
        "project's dotted fact-key convention where they should match another "
        "operation's facts, decide 'produces' from the candidates (or leave "
        "empty for pure reads), delete the '_requires_raw_from_spec' / "
        "'_produces_candidates_from_response_schema' scratch keys, then rename "
        f"the file to {api_name}.conditions.json to make it live."
    )
    print("\n".join(report_lines))
    return out_path


def check(api_name: str) -> None:
    conditions_path = CONDITIONS_DIR / f"{api_name}.conditions.json"
    dataset_path = CAMARA_DOCS_DIR / f"{api_name}.dataset.jsonl"

    if not conditions_path.is_file():
        raise SystemExit(f"Not found: {conditions_path}")

    try:
        all_conditions = load_operation_conditions(CONDITIONS_DIR)
    except DuplicateConditionError as exc:
        raise SystemExit(f"FAILED - duplicate operation_id across data/conditions/*.conditions.json: {exc}")

    file_ids = set(json.loads(conditions_path.read_text(encoding="utf-8")))

    print(f"data/conditions/*.conditions.json: {len(all_conditions)} operation_ids total, no duplicates. OK")

    if dataset_path.is_file():
        dataset_ids = _dataset_operation_ids(dataset_path)
        missing_from_dataset = file_ids - dataset_ids
        missing_from_conditions = dataset_ids - file_ids
        if missing_from_dataset:
            print(f"WARNING: in {conditions_path.name} but not a real operation_id in {dataset_path.name}: {sorted(missing_from_dataset)}")
        if missing_from_conditions:
            print(f"NOTE: in {dataset_path.name} but has no condition entry yet in {conditions_path.name}: {sorted(missing_from_conditions)}")
        if not missing_from_dataset and not missing_from_conditions:
            print(f"{conditions_path.name}: every key matches a real operation_id in {dataset_path.name} exactly. OK")
    else:
        print(f"NOTE: no {dataset_path} found to cross-check operation_ids against.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", nargs="?", help="path to the raw OpenAPI YAML/JSON spec")
    parser.add_argument("--dataset", type=Path, default=None, help="matching *.dataset.jsonl to cross-check operation_ids against")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: data/conditions/<api-name>.conditions.draft.json)")
    parser.add_argument("--check", metavar="API_NAME", default=None, help="validate an already-promoted data/conditions/<API_NAME>.conditions.json instead of generating a draft")
    args = parser.parse_args()

    if args.check:
        check(args.check)
        return 0

    if not args.spec:
        parser.error("spec is required unless --check is given")

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        raise SystemExit(f"Spec file not found: {spec_path}")

    dataset_path = args.dataset
    if dataset_path is None:
        guess = CAMARA_DOCS_DIR / f"{spec_path.stem}.dataset.jsonl"
        if guess.is_file():
            dataset_path = guess

    generate(spec_path, dataset_path, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
