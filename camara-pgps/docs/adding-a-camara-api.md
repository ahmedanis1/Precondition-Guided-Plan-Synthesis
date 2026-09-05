# Adding a new CAMARA API

RAG (selection) and PGPS (planning) are two independent tracks that share
only the `operation_id` string. Neither reads the other's data, and neither
touches `CamaraRegistry` / `data/registry/camara_apis.json` — that registry
is separate, older milestone-3 scaffolding still used elsewhere in the
codebase (parameter extraction, execution, validation, the API/CLI layers),
but it plays no part in how a new API becomes selectable or plannable.

## Track 1 — make it selectable (RAG)

1. Add the raw OpenAPI spec for the new API under `data/camara_docs/<api-name>.yaml`.
2. Run your `openapi_to_dataset.py` extraction script against it to produce
   `data/camara_docs/<api-name>.dataset.jsonl` - facts only (method, path,
   parameters, tags, security scopes), by design no invented semantics at
   this stage. Spot-check a couple of `operation_id` values against the raw
   spec's `operationId` field: they must match byte-for-byte, since this
   string is copied verbatim through every layer below and never
   regenerated.
3. Ingest it:
   ```bash
   python -m camara_pgps ingest-camara --jsonl-dir data/camara_docs --reset
   ```
4. Done - API Selection can now surface the new API's operations. Nothing
   about `CamaraRegistry` or `data/registry/camara_apis.json` is involved;
   it never was for the four real APIs already in the system.

## Track 2 — make it plannable (PGPS)

`scripts/generate_conditions.py` automates everything mechanical in this
track and refuses to guess at anything that isn't:

```bash
python scripts/generate_conditions.py data/camara_docs/<api-name>.yaml
```

It parses the raw spec (`$ref`s resolved, `allOf` merged - the same parser
verified against all four real APIs), cross-checks every `operation_id`
against the matching `.dataset.jsonl` (catches typos immediately - a wrong
key would otherwise fail silently by just never matching, rather than
raising), and writes
`data/conditions/<api-name>.conditions.draft.json` with, per operation:

- `requires`: every required path/query parameter, every required
  request-body field, plus `auth.token` if the operation declares a
  security scope - raw field names, mechanically complete and correct by
  construction (a required field is a required field), but NOT yet renamed
  into this project's dotted fact-key convention (e.g. `sessionId` should
  usually become `qos.session.id`, not stay `sessionId`) - that renaming is
  what makes two operations agree they're describing the same fact, and is
  a judgment call about this project's vocabulary, not something the spec
  states.
- `produces`: left empty, plus a `_produces_candidates_from_response_schema`
  list of the success response's own top-level field names as a memory-jog.
  Deciding what an operation's postconditions actually are - does calling
  it establish a fact (usually an id) that another operation will need
  later? - is the one thing an OpenAPI spec genuinely cannot answer; a
  response schema describes what comes back, not what the API's lifecycle
  promises about it. Pure reads/checks legitimately produce nothing - that's
  fine, `PGPSPlanner._plan_direct` handles operations with no postconditions
  to chain instead of treating them as infeasible.

The output is deliberately named `.conditions.draft.json`, not
`.conditions.json` - `PGPSPlanner` only loads the latter, so a draft can
never accidentally go live. Review it, rename/trim the facts, delete the
`_requires_raw_from_spec` / `_produces_candidates_from_response_schema`
scratch keys, then rename the file to drop `.draft` to promote it.

Afterwards (or any time later, to re-check something already promoted):

```bash
python scripts/generate_conditions.py --check <api-name>
```

validates the live `data/conditions/<api-name>.conditions.json`: every key
still matches a real `operation_id` in the `.dataset.jsonl`, and no
`operation_id` repeats across any file in `data/conditions/`.

`PGPSPlanner()` (no arguments) automatically picks up every
`*.conditions.json` file under `data/conditions/` via
`pgps/conditions_loader.py` - it never touches `CamaraRegistry` or the RAG
corpus to do so.

## Trustworthiness, in one line

`requires` is mechanical and inherits the spec's own trustworthiness;
`produces` is a human judgment call about the API's lifecycle semantics and
should get an explicit review pass (by you, or a second reader) before
being relied on - the `Condition.source` field exists precisely to record
which is which (`"schema"` vs `"manual"`) if you want that provenance
queryable later rather than only living in your head.

## (Optional, separate from the above) intent keyword hints

The offline rule-based intent parser matches capability families using
`registry.capabilities()` plus operation summaries/tags - this is the one
place the old registry is still consulted, and it's unrelated to whether
RAG can select or PGPS can plan the new API. If you want the *offline*
intent path (not the LLM path, which doesn't need this) to route a new
family well, add a keyword set to `agents/intent_agent._FAMILY_SYNONYMS`.

## Verify

```bash
pytest -q                                  # existing tests should pass unchanged
python -m camara_pgps run --request "..."  # exercise the new API end-to-end once wired
```

## Fact-key conventions

Dotted `domain.resource.attribute`. Currently in use:

| Fact key | Meaning |
|---|---|
| `auth.token` | an OAuth2 access token is available (seeded by the workflow) |
| `device.identifier` / `device.phoneNumber` | the target device is identified |
| `qos.profile.name` | a QoS profile name has been chosen |
| `qos.assignment.id` / `qos.assignment` | QoS Provisioning assignment lifecycle |
| `qos.session.id` / `qos.session` | Quality on Demand session lifecycle |
| `qos.session.application_server` / `qos.session.duration` / `qos.session.additional_duration` | session parameters supplied at creation/extension |
