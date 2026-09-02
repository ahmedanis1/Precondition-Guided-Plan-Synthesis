"""CAMARA API registry: the authoritative catalogue of executable APIs."""

from camara_pgps.camara.capability_mapper import (
    CapabilityError,
    CapabilityMapper,
    InvalidCapabilityError,
    UnknownCapabilityError,
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
    "RegistryDocument",
    "RegistryAPI",
    "RegistryOperation",
    "Provenance",
    "SourceType",
]
