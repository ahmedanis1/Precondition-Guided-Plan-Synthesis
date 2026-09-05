"""CAMARA API surface models.

These describe a CAMARA network API and its operations after an OpenAPI
document has been parsed (milestone 4). Every later component - retrieval,
API selection, parameter extraction, planning, validation and execution -
consumes :class:`CamaraAPI` / :class:`CamaraOperation` rather than reading raw
OpenAPI dictionaries.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HTTPMethod(StrEnum):
    """HTTP verbs used by CAMARA operations."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ParameterLocation(StrEnum):
    """Where an :class:`APIParameter` is carried in the request."""

    PATH = "path"
    QUERY = "query"
    HEADER = "header"
    COOKIE = "cookie"
    BODY = "body"


class ConditionKind(StrEnum):
    """Whether a :class:`Condition` must hold before or after an operation."""

    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"


class APIParameter(BaseModel):
    """A single input to a CAMARA operation.

    ``schema_`` holds the JSON Schema fragment for the parameter (serialised as
    ``schema``); for body parameters it is the request-body schema.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    location: ParameterLocation
    required: bool = False
    description: str | None = None
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    example: Any | None = None
    default: Any | None = None
    enum: list[Any] = Field(default_factory=list)


class Condition(BaseModel):
    """A pre- or post-condition attached to a CAMARA operation.

    ``description`` is the human/LLM-readable statement. ``requires`` and
    ``produces`` are machine-usable *fact keys* (e.g. ``"device.phoneNumber"``,
    ``"session.id"``): the PGPS dependency model (milestone 10) orders steps so
    that every precondition's ``requires`` set is covered by the ``produces`` of
    an earlier step's postconditions.
    """

    kind: ConditionKind
    description: str
    id: str | None = None
    expression: str | None = None
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    source: str | None = None
    produces_from: dict[str, str] = Field(default_factory=dict)
    """Optional ``{fact_key: response_body_field}`` map for postconditions.

    Declares, generically and per-operation, which field of the operation's
    JSON response body supplies the value for each produced fact (e.g.
    ``{"qos.session.id": "sessionId"}``). This is what lets the execution
    layer read a real value out of any operation's response without a
    hand-maintained Python table per API - see ``execution/executor.py``.
    Only meaningful on postconditions; always empty on preconditions.
    """


class CamaraOperation(BaseModel):
    """A single callable CAMARA API operation."""

    operation_id: str
    method: HTTPMethod
    path: str
    summary: str | None = None
    description: str | None = None
    parameters: list[APIParameter] = Field(default_factory=list)
    request_schema: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None
    preconditions: list[Condition] = Field(default_factory=list)
    postconditions: list[Condition] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    security_scopes: list[str] = Field(default_factory=list)
    api_name: str | None = None

    def required_parameters(self) -> list[APIParameter]:
        """Return only the parameters the caller must supply."""

        return [p for p in self.parameters if p.required]

    def get_parameter(self, name: str) -> APIParameter | None:
        """Look up a parameter by name."""

        return next((p for p in self.parameters if p.name == name), None)


class CamaraAPI(BaseModel):
    """A CAMARA API family (one OpenAPI document) and its operations."""

    name: str
    version: str
    title: str | None = None
    description: str | None = None
    base_path: str | None = None
    spec_url: str | None = None
    source_file: str | None = None
    tags: list[str] = Field(default_factory=list)
    operations: list[CamaraOperation] = Field(default_factory=list)

    def get_operation(self, operation_id: str) -> CamaraOperation | None:
        """Look up an operation by its ``operationId``."""

        return next(
            (op for op in self.operations if op.operation_id == operation_id),
            None,
        )
