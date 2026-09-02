"""On-disk schema for the local CAMARA API registry.

The registry file (``data/registry/camara_apis.json``) is authored as a
``RegistryDocument``. Its API/operation entries are thin extensions of the
shared domain models from :mod:`camara_pgps.models` - they add only what the
registry itself is responsible for: a semantic ``capability`` tag per operation
and :class:`Provenance` per API (answering "where did this come from?").

Because :class:`RegistryAPI` / :class:`RegistryOperation` subclass
:class:`~camara_pgps.models.CamaraAPI` / :class:`~camara_pgps.models.CamaraOperation`,
anything downstream that expects the domain models can consume registry objects
unchanged.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from camara_pgps.models import CamaraAPI, CamaraOperation


class SourceType(StrEnum):
    """How a registry entry was obtained."""

    LOCAL_REGISTRY = "local_registry"
    OPENAPI = "openapi"
    MANUAL = "manual"


class Provenance(BaseModel):
    """Where the information about an API came from."""

    source_type: SourceType
    location: str
    source_url: str | None = None
    spec_version: str | None = None
    retrieved_at: datetime | None = None
    note: str | None = None


class SourceRef(BaseModel):
    """Back-reference stamped on every parsed operation.

    Lets the RAG layer explain where an API, operation or parameter came from
    (which document, which upstream URL, which spec version).
    """

    api_name: str
    version: str
    operation_id: str | None = None
    source: str | None = None
    source_url: str | None = None


class ResponseSpec(BaseModel):
    """One response entry of an operation (success or error)."""

    model_config = ConfigDict(populate_by_name=True)

    status_code: str
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    is_error: bool = False


class RegistryOperation(CamaraOperation):
    """A registry operation entry: a :class:`CamaraOperation` plus registry-owned
    metadata (capability tag, responses/errors, examples and provenance)."""

    capability: str | None = None
    deprecated: bool = False
    responses: list[ResponseSpec] = Field(default_factory=list)
    errors: list[ResponseSpec] = Field(default_factory=list)
    examples: dict[str, Any] = Field(default_factory=dict)
    source: SourceRef | None = None


class RegistryAPI(CamaraAPI):
    """A registry API entry: a :class:`CamaraAPI` plus provenance."""

    provenance: Provenance
    # Narrowing the element type to the RegistryOperation subclass is a supported
    # pydantic pattern; mypy flags it only because ``list`` is invariant.
    operations: list[RegistryOperation] = Field(default_factory=list)  # type: ignore[assignment]


class RegistryDocument(BaseModel):
    """Root object of the registry file."""

    version: str
    description: str | None = None
    generated_at: datetime | None = None
    apis: list[RegistryAPI] = Field(default_factory=list)

    def get_api(self, name: str) -> RegistryAPI | None:
        return next((a for a in self.apis if a.name == name), None)

    def iter_operations(self) -> list[tuple[str, RegistryOperation]]:
        """Return ``(api_name, operation)`` pairs for every operation in the file."""

        return [(api.name, op) for api in self.apis for op in api.operations]
