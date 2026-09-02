"""CAMARA API registry: the authoritative catalogue of executable APIs."""

from camara_pgps.camara.capability_mapper import (
    CapabilityError,
    CapabilityMapper,
    InvalidCapabilityError,
    UnknownCapabilityError,
)
from camara_pgps.camara.openapi_parser import (
    OpenAPIParseError,
    OpenAPIParser,
    build_registry,
    build_registry_document,
    parse_openapi,
)
from camara_pgps.camara.registry import (
    CamaraRegistry,
    DuplicateEntryError,
    RegistryError,
    UnknownAPIError,
    UnknownOperationError,
    get_registry,
)
from camara_pgps.camara.schemas import (
    Provenance,
    RegistryAPI,
    RegistryDocument,
    RegistryOperation,
    ResponseSpec,
    SourceRef,
    SourceType,
)

__all__ = [
    "CamaraRegistry",
    "get_registry",
    "RegistryError",
    "UnknownAPIError",
    "UnknownOperationError",
    "DuplicateEntryError",
    "CapabilityMapper",
    "CapabilityError",
    "InvalidCapabilityError",
    "UnknownCapabilityError",
    "OpenAPIParser",
    "OpenAPIParseError",
    "parse_openapi",
    "build_registry",
    "build_registry_document",
    "RegistryDocument",
    "RegistryAPI",
    "RegistryOperation",
    "ResponseSpec",
    "SourceRef",
    "Provenance",
    "SourceType",
]
