"""CAMARA OpenAPI parser.

Reads a CAMARA OpenAPI 3.x document (file, URL, text or already-parsed mapping)
and produces a :class:`~camara_pgps.camara.schemas.RegistryAPI` that the
:class:`~camara_pgps.camara.registry.CamaraRegistry` can ingest directly::

    OpenAPI specification -> OpenAPIParser -> RegistryAPI (a CamaraAPI) -> CamaraRegistry

Every parsed operation is stamped with a :class:`SourceRef` (source, source_url,
api_name, version, operation_id) so the RAG layer can later explain where an API
or parameter came from.

Every ``$ref`` is resolved and inlined (bounded by ``_MAX_DEREF_DEPTH`` against
cycles) before a schema is stored on an operation - callers such as
:class:`~camara_pgps.validator.parameters.PlanValidator` validate a request body
straight against ``operation.request_schema`` with no resolver of their own, so
by the time a schema reaches an operation it must already be self-contained.

Note: this parser is independent of, and not used by, the real RAG ingestion
pipeline (``openapi_to_dataset.py`` -> ``*.dataset.jsonl`` -> ChromaDB). It
exists for anyone who wants a typed, registry-shaped view of a spec instead.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from camara_pgps.camara.schemas import (
    Provenance,
    RegistryAPI,
    RegistryDocument,
    RegistryOperation,
    ResponseSpec,
    SourceRef,
    SourceType,
)
from camara_pgps.models import APIParameter, HTTPMethod, ParameterLocation

logger = logging.getLogger(__name__)

_HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch"}
_MAX_DEREF_DEPTH = 200


class OpenAPIParseError(ValueError):
    """Raised when a document cannot be parsed as an OpenAPI 3.x spec."""


def _looks_like_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def _load_mapping(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Return ``(document, location)`` for a path, URL, YAML/JSON text, or mapping."""

    if isinstance(source, Mapping):
        return dict(source), None

    text: str
    location: str | None
    if isinstance(source, Path) or (isinstance(source, str) and not _looks_like_url(source)):
        path = Path(source)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            location = str(path)
        else:
            # Not an existing file - treat the string itself as raw spec text.
            text = str(source)
            location = None
    elif isinstance(source, str) and _looks_like_url(source):
        import httpx

        response = httpx.get(source, timeout=30.0)
        response.raise_for_status()
        text = response.text
        location = source
    else:  # pragma: no cover - defensive
        raise OpenAPIParseError(f"Cannot parse OpenAPI source of type {type(source)!r}")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OpenAPIParseError(f"Not valid YAML/JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise OpenAPIParseError("OpenAPI document must be a mapping at the top level")
    return data, location


def _deref(node: Any, root: dict[str, Any], *, depth: int = 0) -> Any:
    """Recursively resolve and inline every internal ``#/...`` ``$ref``."""

    if depth > _MAX_DEREF_DEPTH:
        raise OpenAPIParseError("$ref resolution exceeded the maximum depth (possible cycle)")

    if isinstance(node, list):
        return [_deref(item, root, depth=depth + 1) for item in node]

    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        if not ref.startswith("#/"):
            # An external ref (e.g. "../common/CAMARA_common.yaml#/...") -
            # real CAMARA specs commonly pull shared parameters/responses
            # from a sibling file this project doesn't have on disk. Rather
            # than fail the whole parse over what's usually a non-required
            # header parameter or a generic error response, degrade to an
            # empty schema and note it - callers deriving `requires` never
            # see a phantom required field this way, they just see nothing
            # here, which is the safe direction to be wrong in.
            logger.warning("Unresolved external $ref (left empty): %r", ref)
            return {"description": f"(unresolved external $ref: {ref})"}
        target: Any = root
        for part in ref[2:].split("/"):
            try:
                target = target[part]
            except (KeyError, TypeError) as exc:
                raise OpenAPIParseError(f"Unresolvable $ref: {ref!r}") from exc
        resolved = _deref(target, root, depth=depth + 1)
        # A ref sibling key (rare, but valid in OpenAPI 3.1) overrides the target.
        if len(node) > 1 and isinstance(resolved, dict):
            merged = dict(resolved)
            merged.update({k: v for k, v in node.items() if k != "$ref"})
            return merged
        return resolved

    all_of = node.get("allOf")
    if isinstance(all_of, list):
        merged_required: list[str] = []
        merged_properties: dict[str, Any] = {}
        merged_rest: dict[str, Any] = {k: v for k, v in node.items() if k != "allOf"}
        for member in all_of:
            resolved_member = _deref(member, root, depth=depth + 1)
            if isinstance(resolved_member, dict):
                merged_required.extend(resolved_member.get("required", []))
                merged_properties.update(resolved_member.get("properties", {}))
                for key, value in resolved_member.items():
                    if key not in ("required", "properties"):
                        merged_rest.setdefault(key, value)
        result = dict(merged_rest)
        result["type"] = result.get("type", "object")
        if merged_properties:
            result["properties"] = merged_properties
        if merged_required:
            # de-dupe, preserve order
            seen: set[str] = set()
            result["required"] = [r for r in merged_required if not (r in seen or seen.add(r))]
        return result

    return {key: _deref(value, root, depth=depth + 1) for key, value in node.items()}


class OpenAPIParser:
    """Parses a single OpenAPI 3.x document into a :class:`RegistryAPI`."""

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self.warnings: list[str] = []

    def _warn(self, message: str) -> None:
        if self.strict:
            raise OpenAPIParseError(message)
        logger.warning(message)
        self.warnings.append(message)

    # -- entry points --------------------------------------------------------

    def parse_file(
        self, path: str | Path, *, source_url: str | None = None
    ) -> RegistryAPI:
        return self.parse(path, source_url=source_url)

    def parse_text(self, text: str, *, source_url: str | None = None) -> RegistryAPI:
        return self.parse(text, source_url=source_url)

    def parse_mapping(
        self, data: Mapping[str, Any], *, source_url: str | None = None
    ) -> RegistryAPI:
        return self.parse(data, source_url=source_url)

    def parse(
        self,
        source: str | Path | Mapping[str, Any],
        *,
        source_url: str | None = None,
    ) -> RegistryAPI:
        document, location = _load_mapping(source)

        if "swagger" in document and "openapi" not in document:
            raise OpenAPIParseError(
                f"Swagger {document.get('swagger')!r} documents are not supported; "
                "convert to OpenAPI 3.x first"
            )
        openapi_version = document.get("openapi")
        if not isinstance(openapi_version, str) or not openapi_version.startswith("3."):
            raise OpenAPIParseError(
                f"Unsupported or missing 'openapi' version: {openapi_version!r}"
            )

        info = document.get("info") or {}
        title = info.get("title")
        version = info.get("version") or "0.0.0"
        name = self._slugify(title) if title else "unknown-api"
        description = info.get("description")

        base_path = None
        servers = document.get("servers") or []
        if servers:
            base_path = servers[0].get("url")

        global_security = document.get("security") or []
        global_tags = [t.get("name") for t in document.get("tags") or [] if t.get("name")]

        seen_ids: set[str] = set()
        operations: list[RegistryOperation] = []
        for path_item, path_spec in (document.get("paths") or {}).items():
            if not isinstance(path_spec, dict):
                continue
            path_spec_resolved = _deref(path_spec, document)
            shared_params = path_spec_resolved.get("parameters") or []
            for method_name, op_spec in path_spec_resolved.items():
                if method_name not in _HTTP_METHODS or not isinstance(op_spec, dict):
                    continue
                operation = self._parse_operation(
                    method=method_name,
                    path=path_item,
                    spec=op_spec,
                    shared_params=shared_params,
                    document=document,
                    api_name=name,
                    version=version,
                    location=location,
                    source_url=source_url,
                    global_security=global_security,
                    global_tags=global_tags,
                )
                if operation.operation_id in seen_ids:
                    self._warn(
                        f"Duplicate operationId {operation.operation_id!r} in {location or source_url}"
                    )
                seen_ids.add(operation.operation_id)
                operations.append(operation)

        provenance = Provenance(
            source_type=SourceType.OPENAPI,
            location=location or source_url or "<in-memory>",
            source_url=source_url,
            spec_version=version,
            retrieved_at=datetime.now(UTC),
        )

        return RegistryAPI(
            name=name,
            version=version,
            title=title,
            description=description,
            base_path=base_path,
            spec_url=source_url,
            source_file=location,
            tags=global_tags,
            operations=operations,
            provenance=provenance,
        )

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _slugify(title: str) -> str:
        return "-".join(title.lower().split())

    def _parse_operation(
        self,
        *,
        method: str,
        path: str,
        spec: dict[str, Any],
        shared_params: list[Any],
        document: dict[str, Any],
        api_name: str,
        version: str,
        location: str | None,
        source_url: str | None,
        global_security: list[Any],
        global_tags: list[str],
    ) -> RegistryOperation:
        operation_id = spec.get("operationId")
        if not operation_id:
            operation_id = f"{method}_{path}".replace("/", "_").replace("{", "").replace("}", "")
            self._warn(
                f"Operation {method.upper()} {path} has no operationId; "
                f"synthesised {operation_id!r}"
            )

        all_params_raw = list(shared_params) + list(spec.get("parameters") or [])
        parameters: list[APIParameter] = []
        for raw_param in all_params_raw:
            resolved_param = _deref(raw_param, document)
            if "name" not in resolved_param:
                # Came from an unresolved external $ref (no "name" to give
                # back) - drop it rather than crash. In practice this is
                # always a non-required, non-fact-bearing header like
                # x-correlator, so dropping it changes nothing about
                # `requires`/`produces`.
                logger.warning(
                    "Dropping parameter with no resolvable name in %s %s",
                    method.upper(),
                    path,
                )
                continue
            parameters.append(self._parse_parameter(resolved_param))

        request_schema: dict[str, Any] | None = None
        request_body = spec.get("requestBody")
        if isinstance(request_body, dict):
            resolved_body = _deref(request_body, document)
            content = resolved_body.get("content") or {}
            json_content = content.get("application/json") or next(iter(content.values()), {})
            schema = json_content.get("schema")
            if schema is not None:
                request_schema = _deref(schema, document)
            if request_schema:
                # One BODY-location APIParameter per top-level request-body
                # field - not a single wrapped blob - so the validator's
                # per-parameter checks (required / type / enum) line up with
                # PlanValidator's `body = {name: ... if location is BODY}`
                # reconstruction in validator/parameters.py.
                required_fields = set(request_schema.get("required", []))
                for field_name, field_schema in (request_schema.get("properties") or {}).items():
                    parameters.append(
                        APIParameter(
                            name=field_name,
                            location=ParameterLocation.BODY,
                            required=field_name in required_fields,
                            description=field_schema.get("description")
                            if isinstance(field_schema, dict)
                            else None,
                            schema=field_schema if isinstance(field_schema, dict) else {},
                            enum=(field_schema.get("enum", []) if isinstance(field_schema, dict) else []),
                        )
                    )

        responses: list[ResponseSpec] = []
        errors: list[ResponseSpec] = []
        examples: dict[str, Any] = {}
        for status_code, resp_spec in (spec.get("responses") or {}).items():
            resolved_resp = _deref(resp_spec, document)
            content = (resolved_resp.get("content") or {}).get("application/json") or {}
            resp_schema = content.get("schema")
            is_error = status_code.isdigit() and int(status_code) >= 400
            entry = ResponseSpec(
                status_code=str(status_code),
                description=resolved_resp.get("description"),
                schema=resp_schema,
                is_error=is_error,
            )
            (errors if is_error else responses).append(entry)
            if "example" in content:
                examples[str(status_code)] = content["example"]

        security = spec.get("security", global_security) or []
        scopes: list[str] = []
        for entry in security:
            if isinstance(entry, dict):
                for scope_list in entry.values():
                    scopes.extend(scope_list or [])

        tags = spec.get("tags") or global_tags

        return RegistryOperation(
            operation_id=operation_id,
            method=HTTPMethod(method.upper()),
            path=path,
            summary=spec.get("summary"),
            description=spec.get("description"),
            parameters=parameters,
            request_schema=request_schema,
            response_schema=(responses[0].schema_ if responses else None),
            tags=tags,
            security_scopes=scopes,
            api_name=api_name,
            responses=responses,
            errors=errors,
            examples=examples,
            source=SourceRef(
                api_name=api_name,
                version=version,
                operation_id=operation_id,
                source=location,
                source_url=source_url,
            ),
        )

    @staticmethod
    def _parse_parameter(spec: dict[str, Any]) -> APIParameter:
        location = spec.get("in", "query")
        return APIParameter(
            name=spec["name"],
            location=ParameterLocation(location),
            required=bool(spec.get("required", False)),
            description=spec.get("description"),
            schema=spec.get("schema") or {},
            example=spec.get("example"),
            default=(spec.get("schema") or {}).get("default"),
            enum=(spec.get("schema") or {}).get("enum", []),
        )


def parse_openapi(
    source: str | Path | Mapping[str, Any],
    *,
    source_url: str | None = None,
    strict: bool = False,
) -> RegistryAPI:
    """Convenience one-shot: parse a single OpenAPI document into a ``RegistryAPI``."""

    return OpenAPIParser(strict=strict).parse(source, source_url=source_url)


def build_registry_document(
    sources: list[str | Path | Mapping[str, Any]],
    *,
    version: str = "0.0.0",
    strict: bool = False,
) -> RegistryDocument:
    """Parse several specs into a single :class:`RegistryDocument`."""

    apis = [OpenAPIParser(strict=strict).parse(source) for source in sources]
    return RegistryDocument(version=version, generated_at=datetime.now(UTC), apis=apis)


def build_registry(
    sources: list[str | Path | Mapping[str, Any]],
    *,
    version: str = "0.0.0",
    strict: bool = False,
):
    """Parse several specs and return a ready-to-query ``CamaraRegistry``."""

    from camara_pgps.camara.registry import CamaraRegistry

    document = build_registry_document(sources, version=version, strict=strict)
    return CamaraRegistry(document)


__all__ = [
    "OpenAPIParseError",
    "OpenAPIParser",
    "parse_openapi",
    "build_registry",
    "build_registry_document",
]
