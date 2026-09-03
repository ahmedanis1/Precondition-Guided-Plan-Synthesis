# Adding a new CAMARA API

The registry file `data/registry/camara_apis.json` is the **authoritative source
of executable APIs**. Nothing else in the system hard-codes API names, so adding
an API is a data change plus (optionally) re-ingesting the knowledge base.

## 1. Add the API to the registry

Append an entry to `apis` in `data/registry/camara_apis.json`:

```jsonc
{
  "name": "sim-swap",                       // lowercase slug, unique
  "title": "SIM Swap",
  "version": "2.0.0",
  "description": "Check whether a SIM card has been swapped recently.",
  "base_path": "/sim-swap/v2",
  "tags": ["fraud"],
  "provenance": {
    "source_type": "local_registry",
    "location": "data/registry/camara_apis.json",
    "spec_version": "2.0.0",
    "note": "Hand-authored."
  },
  "operations": [
    {
      "operation_id": "checkSimSwap",       // unique across the whole registry
      "method": "POST",
      "path": "/check",
      "summary": "Check for a recent SIM swap.",
      "capability": "sim-swap.check",        // dotted slug: <family>.<action>
      "api_name": "sim-swap",
      "security_scopes": ["sim-swap:check"],
      "parameters": [
        { "name": "phoneNumber", "location": "body", "required": true,
          "schema": {"type": "string"} },
        { "name": "maxAge", "location": "body", "required": false,
          "schema": {"type": "integer"}, "enum": [240, 1440] }
      ],
      "preconditions": [
        { "kind": "precondition", "description": "An access token is available.",
          "requires": ["auth.token"], "source": "manual" },
        { "kind": "precondition", "description": "The device identifier is known.",
          "requires": ["device.identifier"], "source": "manual" }
      ],
      "postconditions": [
        { "kind": "postcondition", "description": "The swap check result is known.",
          "produces": ["sim-swap.result"], "source": "manual" }
      ]
    }
  ]
}
```

### Field notes

* **`capability`** — a dotted slug validated by `CapabilityMapper`
  (`^[a-z0-9-]+(\.[a-z0-9-]+)+$`). The part before the first `.` is the
  *capability family*; the intent agent selects at family granularity.
* **`requires` / `produces`** — [fact keys](#fact-key-conventions). Reuse an
  existing key when two operations refer to the same thing — that is what creates
  a dependency edge in the PGPS graph.
* **`enum`** on a parameter is enforced by the validator and surfaced by the
  repair agent.

### From an OpenAPI document instead

```python
from camara_pgps.camara import parse_openapi
api = parse_openapi("path/or/https/url/to/spec.yaml")
```

`parse_openapi` fills everything except `preconditions` / `postconditions` /
`capability` (OpenAPI has no standard for them) — add those by hand afterwards,
following the recipe in the [README](../README.md#making-a-condition-from-an-api)
or the table above.

## 2. Re-ingest the knowledge base

```bash
python -m camara_pgps ingest-camara --reset
```

## 3. (Optional) add intent keywords

The offline rule-based intent parser matches capability families using
`registry.capabilities()` plus operation summaries/tags. If you want the
*offline* path to route a new family well, add a keyword set to
`agents/intent_agent._FAMILY_SYNONYMS` (the LLM path does not need this).

## 4. Verify

```bash
python -m camara_pgps list-apis --api sim-swap
pytest -q                       # registry/graph/solver tests validate the new data
python -m camara_pgps run --request "has my SIM been swapped?" \
  --arguments '{"checkSimSwap": {"phoneNumber": "+12025550123"}}'
```

`pytest` will fail loudly if an `operation_id` collides, a `capability` slug is
malformed, or a `requires` fact has no producer and is not an external fact.

## Fact-key conventions

Dotted `domain.resource.attribute`. Currently in use:

| Fact key | Meaning |
|---|---|
| `auth.token` | an OAuth2 access token is available (seeded by the workflow) |
| `device.identifier` / `device.phoneNumber` | the target device is identified |
| `location.area` | an area to check against is known |
| `qos.profile` | a QoS profile name has been chosen |
| `qos.session.id` / `qos.session` / `qos.session.deleted` | QoD session lifecycle |
| `device.location` / `device.location.verification` | location retrieval / verification results |
| `network-insights.subscription.id` / `network-insights.subscription` | NI subscription lifecycle |
