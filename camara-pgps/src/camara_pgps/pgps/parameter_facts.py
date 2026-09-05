"""Canonical parameter-name <-> fact-key vocabulary.

The rest of PGPS talks about state in dotted "fact keys" (``qos.session.id``,
``device.identifier``); real CAMARA operations talk about the same thing as a
literally-cased request parameter (``sessionId``, ``device``). Three stages
need to cross between those two vocabularies, and used to each keep their own
copy of the mapping:

* parameter extraction (``agents/parameter_agent.py``) binds a value found
  under one name to every fact key it also satisfies,
* the executor (``execution/executor.py``) substitutes a fact's value into a
  path/query parameter under its real, exactly-cased name,
* the planner (``pgps/planner.py``) records *why* a later step's parameter
  will already have a value when an earlier step in the same plan produces
  the fact it maps to (so the validator and executor both recognise it as
  resolved instead of asking a person for something PGPS's own plan already
  arranged to fetch).

Two independently hand-maintained tables can silently drift apart - a fact
recognised by one is not necessarily recognised by the other, which is
exactly the kind of inconsistency that made a genuinely chained plan still
look "incomplete" downstream. Both directions now come from the single table
below.

This is deliberately not a per-operation table: the same small, shared
vocabulary covers every CAMARA operation that carries the underlying fact, so
adding a new operation almost never needs a new line here - only a genuinely
new fact category does. That is what keeps this out of the "one entry per
API" hardcoding this project has otherwise been rooting out.
"""

from __future__ import annotations

import re

# normalised parameter name -> fact key(s) it satisfies / supplies. A
# parameter can satisfy more than one fact key when an older, pre-real-CAMARA
# fixture and the real condition data happened to name the same underlying
# concept differently ("qos.profile" is a milestone-3 fixture fact predating
# the real CAMARA QoS Profiles API's own "qos.profile.name" - both are kept
# so a bound value satisfies whichever one a given operation's condition data
# actually asks for). List the real/current convention first - that is the
# one `fact_for_parameter` treats as primary.
PARAMETER_TO_FACTS: dict[str, tuple[str, ...]] = {
    "device": ("device.identifier",),
    "sessionid": ("qos.session.id",),
    "qosprofile": ("qos.profile.name", "qos.profile"),
    "subscriptionid": ("network-insights.subscription.id",),
    "assignmentid": ("qos.assignment.id",),
}

# fact key -> the real, exactly-cased request parameter/path-field name it is
# known by (matters for path-template substitution, e.g. "{sessionId}", and
# for binding a fact a plan step produces into the parameter of a later
# step's request body, e.g. "qosProfile"). Only facts that are ever
# substituted into a raw request need an entry here.
FACT_TO_PARAMETER: dict[str, str] = {
    "device.identifier": "device",
    "qos.session.id": "sessionId",
    "qos.assignment.id": "assignmentId",
    "network-insights.subscription.id": "subscriptionId",
    "qos.profile.name": "qosProfile",
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def fact_for_parameter(name: str) -> str | None:
    """The primary fact key a raw parameter name resolves to, if any."""

    facts = PARAMETER_TO_FACTS.get(_normalize(name))
    return facts[0] if facts else None


def facts_for_parameter(name: str) -> tuple[str, ...]:
    """Every fact key a raw parameter name resolves to (may be more than one)."""

    return PARAMETER_TO_FACTS.get(_normalize(name), ())


def parameter_for_fact(fact: str) -> str | None:
    """The conventional, exactly-cased raw parameter name for a fact key."""

    return FACT_TO_PARAMETER.get(fact)


__all__ = [
    "PARAMETER_TO_FACTS",
    "FACT_TO_PARAMETER",
    "fact_for_parameter",
    "facts_for_parameter",
    "parameter_for_fact",
]
