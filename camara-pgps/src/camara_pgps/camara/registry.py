"""The CAMARA API registry.

``CamaraRegistry`` is the **authoritative source of executable CAMARA APIs**. Every
component that needs to know which APIs/operations exist, what parameters an
operation takes, what capability it provides, or where the information came from
must go through the registry. Selection and planning resolve LLM output against
it: :meth:`CamaraRegistry.require_operation` raises rather than letting an
unknown (hallucinated) operation through.

Milestone 3 loads a small hand-authored file (``data/registry/camara_apis.json``);
milestone 4 will populate the same structure from real CAMARA OpenAPI documents.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from camara_pgps.camara.capability_mapper import CapabilityMapper
from camara_pgps.camara.schemas import Provenance, RegistryAPI, RegistryDocument
from camara_pgps.config import PROJECT_ROOT, Settings, get_settings
from camara_pgps.models import APIParameter, CamaraAPI, CamaraOperation


class RegistryError(Exception):
    """Base class for registry errors."""


class UnknownAPIError(RegistryError):
    """Raised when an API name is not in the registry."""


class UnknownOperationError(RegistryError):
    """Raised when an operation id is not in the registry."""


class DuplicateEntryError(RegistryError):
    """Raised when the registry file defines the same API or operation id twice."""


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


class CamaraRegistry:
    """Queryable view over a validated :class:`RegistryDocument`."""

    def __init__(
        self,
        document: RegistryDocument,
        *,
        source_path: Path | None = None,
    ) -> None:
        self._document = document
        self._source_path = source_path

        self._apis: dict[str, RegistryAPI] = {}
        for api in document.apis:
            if api.name in self._apis:
                raise DuplicateEntryError(f"Duplicate API name in registry: {api.name!r}")
            self._apis[api.name] = api

        self._operations: dict[str, tuple[str, CamaraOperation]] = {}
        for api in document.apis:
            for operation in api.operations:
                if operation.operation_id in self._operations:
                    raise DuplicateEntryError(
                        f"Duplicate operationId in registry: {operation.operation_id!r}"
                    )
                self._operations[operation.operation_id] = (api.name, operation)

        self._capabilities = CapabilityMapper.from_document(document)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> CamaraRegistry:
        resolved = _resolve(path)
        if not resolved.is_file():
            raise RegistryError(f"Registry file not found: {resolved}")
        document = RegistryDocument.model_validate_json(resolved.read_text("utf-8"))
        return cls(document, source_path=resolved)

    @classmethod
    def from_openapi_dir(cls, directory: str | Path, *, strict: bool = False) -> CamaraRegistry:
        """Build the registry straight from the real CAMARA OpenAPI specs.

        This is the live counterpart to :meth:`from_file`: instead of reading
        a pre-built registry JSON that has to be hand-regenerated whenever an
        API is added or changed, it parses every ``*.yaml`` spec in
        ``directory`` fresh, via :func:`camara_pgps.camara.openapi_parser.build_registry`.
        There is no static file that can go stale.
        """

        from camara_pgps.camara.openapi_parser import build_registry

        resolved = _resolve(directory)
        sources = sorted(resolved.glob("*.yaml"))
        if not sources:
            raise RegistryError(f"No OpenAPI specs (*.yaml) found in {resolved}")
        return build_registry(sources, strict=strict)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> CamaraRegistry:
        settings = settings or get_settings()
        return cls.from_openapi_dir(settings.registry.specs_dir)

    # -- APIs ------------------------------------------------------------------

    def api_names(self) -> list[str]:
        """What CAMARA APIs exist?"""

        return list(self._apis)

    def list_apis(self) -> list[CamaraAPI]:
        return list(self._apis.values())

    def has_api(self, name: str) -> bool:
        return name in self._apis

    def get_api(self, name: str) -> CamaraAPI | None:
        return self._apis.get(name)

    def require_api(self, name: str) -> CamaraAPI:
        return self._require_registry_api(name)

    def _require_registry_api(self, name: str) -> RegistryAPI:
        try:
            return self._apis[name]
        except KeyError:
            raise UnknownAPIError(
                f"Unknown CAMARA API {name!r}. Known APIs: {self.api_names()}"
            ) from None

    # -- operations ----------------------------------------------------------

    def list_operations(self, api_name: str | None = None) -> list[CamaraOperation]:
        """What operations does an API provide? (all of them when ``api_name`` is None)"""

        if api_name is None:
            return [operation for _, operation in self._operations.values()]
        return list(self.require_api(api_name).operations)

    def operation_ids(self, api_name: str | None = None) -> list[str]:
        return [op.operation_id for op in self.list_operations(api_name)]

    def has_operation(self, operation_id: str) -> bool:
        return operation_id in self._operations

    def get_operation(self, operation_id: str) -> CamaraOperation | None:
        entry = self._operations.get(operation_id)
        return entry[1] if entry else None

    def require_operation(self, operation_id: str) -> CamaraOperation:
        """Return an operation or raise - the guard against invented operations."""

        try:
            return self._operations[operation_id][1]
        except KeyError:
            raise UnknownOperationError(
                f"Unknown CAMARA operation {operation_id!r}. It is not in the registry "
                f"({self._source_path or 'in-memory'}); it cannot be executed."
            ) from None

    def api_of(self, operation_id: str) -> str:
        """Which API does this operation belong to?"""

        if operation_id not in self._operations:
            self.require_operation(operation_id)
        return self._operations[operation_id][0]

    # -- parameters --------------------------------------------------------

    def parameters_of(self, operation_id: str) -> list[APIParameter]:
        """What parameters does an operation accept?"""

        return list(self.require_operation(operation_id).parameters)

    def required_parameters_of(self, operation_id: str) -> list[APIParameter]:
        """What parameters does an operation require?"""

        return self.require_operation(operation_id).required_parameters()

    # -- capabilities ----------------------------------------------------------

    @property
    def capability_mapper(self) -> CapabilityMapper:
        return self._capabilities

    def capabilities(self) -> list[str]:
        """The closed vocabulary of capabilities the registry can fulfil."""

        return self._capabilities.capabilities()

    def capability_of(self, operation_id: str) -> str | None:
        """What capability does an operation provide?"""

        self.require_operation(operation_id)
        return self._capabilities.capability_for(operation_id)

    def operations_for_capability(self, capability: str) -> list[CamaraOperation]:
        """Which operations provide a given capability?"""

        return [
            self.require_operation(op_id)
            for op_id in self._capabilities.operations_with(capability)
        ]

    # -- provenance ------------------------------------------------------------

    def provenance_of(self, api_name: str) -> Provenance:
        """Where did the information about this API come from?"""

        return self._require_registry_api(api_name).provenance

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def document(self) -> RegistryDocument:
        return self._document


@lru_cache(maxsize=1)
def get_registry() -> CamaraRegistry:
    """Return the process-wide registry singleton, loaded from configured settings."""

    return CamaraRegistry.from_settings()
