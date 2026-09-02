"""CAMARA OpenAPI parser.

Reads a CAMARA OpenAPI 3.x document (file, URL, text or already-parsed mapping)
and produces a :class:`~camara_pgps.camara.schemas.RegistryAPI` that the
:class:`~camara_pgps.camara.registry.CamaraRegistry` can ingest directly::

    OpenAPI specification -> OpenAPIParser -> RegistryAPI (a CamaraAPI) -> CamaraRegistry

Every parsed operation is stamped with a :class:`SourceRef` (source, source_url,
api_name, version, operation_id) so the RAG layer can later explain where an API
or parameter came from.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
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

_HTTP_METHODS = {"get", "put", "post", "delete", "patch", "head", "options"}
_MAX_DEREF_DEPTH = 200


class OpenAPIParseError(ValueError):
    """Raised when a document cannot be parsed as an OpenAPI 3.x spec."""


class _ExternalRef(Exception):
    """Internal signal: a ``$ref`` points outside the document."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _slugify(text: str) -> str:
    out = []
    prev_dash = False
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def _is_version_segment(segment: str) -> bool:
    seg = segment.lower()
    return seg in {"vwip", "vwip1"} or (seg.startswith("v") and seg[1:2].isdigit())


def _is_success_code(code: str) -> bool:
    code = code.lower()
    return code.startswith("2") or code in {"2xx"}


def _load_document(text: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - passthrough
        raise OpenAPIParseError(f"could not parse document as YAML/JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise OpenAPIParseError("OpenAPI document must be a mapping at the top level")
    return data


def _media_example(media: Mapping[str, Any]) -> Any:
    if "example" in media:
        return media["example"]
    examples = media.get("examples")
    if isinstance(examples, Mapping) and examples:
        first = next(iter(examples.values()))
        if isinstance(first, Mapping) and "value" in first:
            return first["value"]
        return first
    return None


def _first_json_media(content: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(content, Mapping):
        return {}
    if "application/json" in content:
        return dict(content["application/json"])
    for media_type, media in content.items():
        if isinstance(media_type, str) and media_type.endswith(("json", "+json")):
            return dict(media)
    return {}


def _collect_properties(schema: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Flatten ``properties`` / ``required`` through ``allOf`` / ``oneOf`` / ``anyOf``."""

    props: dict[str, Any] = {}
    required: set[str] = set()
    stack: list[Any] = [schema]
    while stack:
        node = stack.pop()
        if not isinstance(node, Mapping):
            continue
        for name, sub in (node.get("properties") or {}).items():
            props.setdefault(name, sub)
        required.update(node.get("required") or [])
        for combinator in ("allOf", "oneOf", "anyOf"):
            stack.extend(node.get(combinator) or [])
    return props, required


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

class OpenAPIParser:
    """Turn an OpenAPI 3.x document into a :class:`RegistryAPI`.

    ``strict=True`` turns every warning (missing operationId, unresolved ``$ref``,
    unknown security scheme, ...) into an :class:`OpenAPIParseError`. Otherwise
    warnings are logged and collected on :attr:`warnings`.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self.warnings: list[str] = []
        self._root: dict[str, Any] = {}

    # -- public API ----------------------------------------------------------

    def parse(
        self,
        document: Mapping[str, Any],
        *,
        source: str | None = None,
        source_url: str | None = None,
        name: str | None = None,
    ) -> RegistryAPI:
        self.warnings = []
        self._root = dict(document)
        self._check_version()

        info = self._root.get("info") or {}
        version = str(info.get("version") or "0.0.0")
        api_name = (
            name
            or self._name_from_servers()
            or _slugify(str(info.get("title") or ""))
            or "camara-api"
        )
        base_path = self._base_path()

        operations = self._operations(api_name, version, source, source_url)

        raw_tags = self._root.get("tags") or []
        tag_names = [t["name"] for t in raw_tags if isinstance(t, Mapping) and t.get("name")]

        return RegistryAPI(
            name=api_name,
            version=version,
            title=info.get("title"),
            description=info.get("description"),
            base_path=base_path,
            spec_url=source_url,
            source_file=source,
            tags=tag_names,
            operations=operations,
            provenance=Provenance(
                source_type=SourceType.OPENAPI,
                location=source or source_url or "<in-memory>",
                source_url=source_url,
                spec_version=version,
                retrieved_at=datetime.now(tz=UTC),
                note=f"Parsed from OpenAPI {self._root.get('openapi')}",
            ),
        )

    def parse_text(
        self,
        text: str,
        *,
        source: str | None = None,
        source_url: str | None = None,
        name: str | None = None,
    ) -> RegistryAPI:
        return self.parse(_load_document(text), source=source, source_url=source_url, name=name)

    def parse_file(
        self,
        path: str | Path,
        *,
        source_url: str | None = None,
        name: str | None = None,
    ) -> RegistryAPI:
        file_path = Path(path)
        if not file_path.is_file():
            raise OpenAPIParseError(f"OpenAPI file not found: {file_path}")
        return self.parse_text(
            file_path.read_text("utf-8"),
            source=str(file_path),
            source_url=source_url,
            name=name,
        )

    def parse_url(
        self, url: str, *, client: Any | None = None, name: str | None = None
    ) -> RegistryAPI:
        import httpx

        owns_client = client is None
        client = client or httpx.Client(timeout=30.0, follow_redirects=True)
        try:
            response = client.get(url)
            response.raise_for_status()
            return self.parse_text(response.text, source=url, source_url=url, name=name)
        finally:
            if owns_client:
                client.close()

    # -- structural extraction --------------------------------------------------

    def _check_version(self) -> None:
        if "openapi" not in self._root:
            if "swagger" in self._root:
                raise OpenAPIParseError(
                    "Swagger / OpenAPI 2.0 documents are not supported; provide OpenAPI 3.x"
                )
            raise OpenAPIParseError("document has no 'openapi' version field")
        if not str(self._root["openapi"]).startswith("3."):
            self._warn(f"unexpected OpenAPI version {self._root['openapi']!r}; parsing as 3.x")

    def _base_path(self) -> str | None:
        servers = self._root.get("servers") or []
        if not servers:
            return None
        url = str(servers[0].get("url", "")).strip()
        if not url:
            return None
        parsed = urlparse(url)
        return parsed.path or url or None

    def _name_from_servers(self) -> str | None:
        servers = self._root.get("servers") or []
        if not servers:
            return None
        parsed = urlparse(str(servers[0].get("url", "")))
        segments = [
            seg
            for seg in parsed.path.split("/")
            if seg and not seg.startswith("{") and not _is_version_segment(seg)
        ]
        return _slugify(segments[-1]) if segments else None

    def _operations(
        self,
        api_name: str,
        version: str,
        source: str | None,
        source_url: str | None,
    ) -> list[RegistryOperation]:
        operations: list[RegistryOperation] = []
        paths = self._root.get("paths") or {}
        for path, path_item in paths.items():
            if not isinstance(path_item, Mapping):
                continue
            shared_params = self._parameters(path_item.get("parameters") or [])
            for method, operation in path_item.items():
                if method.lower() not in _HTTP_METHODS or not isinstance(operation, Mapping):
                    continue
                operations.append(
                    self._operation(
                        path=path,
                        method=method.lower(),
                        operation=operation,
                        shared_params=shared_params,
                        api_name=api_name,
                        version=version,
                        source=source,
                        source_url=source_url,
                    )
                )
        return operations

    def _operation(
        self,
        *,
        path: str,
        method: str,
        operation: Mapping[str, Any],
        shared_params: list[APIParameter],
        api_name: str,
        version: str,
        source: str | None,
        source_url: str | None,
    ) -> RegistryOperation:
        operation_id = operation.get("operationId")
        if not operation_id:
            segments = [
                seg.replace("{", "").replace("}", "")
                for seg in path.split("/")
                if seg
            ]
            operation_id = "_".join([method, *segments]) if segments else method
            self._warn(
                f"operation {method.upper()} {path} has no operationId; "
                f"synthesised {operation_id!r}"
            )

        own_params = self._parameters(operation.get("parameters") or [])
        params = self._merge_params(shared_params, own_params)

        request_schema, body_params, request_example = self._request_body(
            operation.get("requestBody")
        )
        params.extend(body_params)

        responses, success_schema, response_examples = self._responses(
            operation.get("responses") or {}
        )
        errors = [r for r in responses if r.is_error]

        examples: dict[str, Any] = {}
        if request_example is not None:
            examples["request"] = request_example
        param_examples = {p.name: p.example for p in params if p.example is not None}
        if param_examples:
            examples["parameters"] = param_examples
        if response_examples:
            examples["responses"] = response_examples

        return RegistryOperation(
            operation_id=operation_id,
            method=HTTPMethod(method.upper()),
            path=path,
            summary=operation.get("summary"),
            description=operation.get("description"),
            parameters=params,
            request_schema=request_schema,
            response_schema=success_schema,
            tags=list(operation.get("tags") or []),
            security_scopes=self._security_scopes(operation.get("security")),
            api_name=api_name,
            deprecated=bool(operation.get("deprecated", False)),
            responses=responses,
            errors=errors,
            examples=examples,
            source=SourceRef(
                api_name=api_name,
                version=version,
                operation_id=operation_id,
                source=source,
                source_url=source_url,
            ),
        )

    def _parameters(self, raw: list[Any]) -> list[APIParameter]:
        result: list[APIParameter] = []
        for entry in raw:
            resolved = self._deref(entry)
            if not isinstance(resolved, Mapping) or "$ref" in resolved:
                self._warn(f"skipping unresolved parameter: {entry!r}")
                continue
            location = resolved.get("in")
            try:
                param_location = ParameterLocation(str(location))
            except ValueError:
                self._warn(
                    f"parameter {resolved.get('name')!r} has unknown location "
                    f"{location!r}; skipped"
                )
                continue
            schema = resolved.get("schema") or {}
            result.append(
                APIParameter(
                    name=resolved["name"],
                    location=param_location,
                    required=bool(
                        resolved.get("required", param_location is ParameterLocation.PATH)
                    ),
                    description=resolved.get("description"),
                    schema=schema,
                    example=resolved.get("example"),
                    default=schema.get("default"),
                    enum=list(schema.get("enum") or []),
                )
            )
        return result

    @staticmethod
    def _merge_params(
        base: list[APIParameter], override: list[APIParameter]
    ) -> list[APIParameter]:
        merged = {(p.location, p.name): p for p in base}
        for param in override:
            merged[(param.location, param.name)] = param
        return list(merged.values())

    def _request_body(
        self, request_body: Any
    ) -> tuple[dict[str, Any] | None, list[APIParameter], Any]:
        resolved = self._deref(request_body)
        if not isinstance(resolved, Mapping) or "$ref" in resolved:
            return None, [], None

        media = _first_json_media(resolved.get("content") or {})
        schema = self._deref(media.get("schema")) if media.get("schema") is not None else None
        example = _media_example(media)

        params: list[APIParameter] = []
        request_schema: dict[str, Any] | None = None
        if isinstance(schema, Mapping):
            request_schema = dict(schema)
            body_required = bool(resolved.get("required", False))
            properties, required_fields = _collect_properties(schema)
            for prop_name, prop_schema in properties.items():
                prop_schema = prop_schema if isinstance(prop_schema, Mapping) else {}
                params.append(
                    APIParameter(
                        name=prop_name,
                        location=ParameterLocation.BODY,
                        required=body_required and prop_name in required_fields,
                        description=prop_schema.get("description"),
                        schema=dict(prop_schema),
                        example=prop_schema.get("example"),
                        default=prop_schema.get("default"),
                        enum=list(prop_schema.get("enum") or []),
                    )
                )
        return request_schema, params, example

    def _responses(
        self, responses: Mapping[str, Any]
    ) -> tuple[list[ResponseSpec], dict[str, Any] | None, dict[str, Any]]:
        specs: list[ResponseSpec] = []
        success_schema: dict[str, Any] | None = None
        examples: dict[str, Any] = {}
        for code, entry in responses.items():
            resolved = self._deref(entry)
            if not isinstance(resolved, Mapping) or "$ref" in resolved:
                self._warn(f"skipping unresolved response {code!r}")
                continue
            media = _first_json_media(resolved.get("content") or {})
            schema = self._deref(media.get("schema")) if media.get("schema") is not None else None
            schema_dict = dict(schema) if isinstance(schema, Mapping) else None
            is_error = not _is_success_code(str(code))
            specs.append(
                ResponseSpec(
                    status_code=str(code),
                    description=resolved.get("description"),
                    schema=schema_dict,
                    is_error=is_error,
                )
            )
            example = _media_example(media)
            if example is not None:
                examples[str(code)] = example
            if not is_error and success_schema is None and schema_dict is not None:
                success_schema = schema_dict
        return specs, success_schema, examples

    def _security_scopes(self, operation_security: Any) -> list[str]:
        requirements = operation_security
        if requirements is None:
            requirements = self._root.get("security")
        if not requirements:
            return []
        schemes = (self._root.get("components") or {}).get("securitySchemes") or {}
        scopes: list[str] = []
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                continue
            for scheme_name, scope_list in requirement.items():
                if scheme_name not in schemes:
                    self._warn(f"security scheme {scheme_name!r} is not defined in components")
                scopes.extend(scope_list or [])
        return list(dict.fromkeys(scopes))

    # -- $ref resolution ------------------------------------------------------

    def _resolve_ref(self, ref: str) -> Any:
        if not ref.startswith("#/"):
            raise _ExternalRef(ref)
        node: Any = self._root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(ref)
            node = node[part]
        return node

    def _deref(self, node: Any, _seen: frozenset[str] | None = None, _depth: int = 0) -> Any:
        if _seen is None:
            _seen = frozenset()
        if _depth > _MAX_DEREF_DEPTH:
            self._warn("maximum $ref expansion depth reached; leaving remaining refs intact")
            return node
        if isinstance(node, Mapping):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref in _seen:
                    return {"$ref": ref}
                try:
                    target = self._resolve_ref(ref)
                except _ExternalRef:
                    self._warn(f"external $ref left unresolved: {ref}")
                    return dict(node)
                except (KeyError, TypeError):
                    self._warn(f"could not resolve $ref: {ref}")
                    return dict(node)
                resolved = self._deref(target, _seen | {ref}, _depth + 1)
                siblings = {
                    key: self._deref(value, _seen, _depth + 1)
                    for key, value in node.items()
                    if key != "$ref"
                }
                if isinstance(resolved, Mapping) and siblings:
                    return {**resolved, **siblings}
                return resolved
            return {key: self._deref(value, _seen, _depth + 1) for key, value in node.items()}
        if isinstance(node, list):
            return [self._deref(item, _seen, _depth + 1) for item in node]
        return node

    # -- diagnostics --------------------------------------------------------

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        if self.strict:
            raise OpenAPIParseError(message)
        logger.warning("openapi-parser: %s", message)


# --------------------------------------------------------------------------- #
# module-level convenience
# --------------------------------------------------------------------------- #

def parse_openapi(
    source: str | Path | Mapping[str, Any],
    *,
    source_url: str | None = None,
    name: str | None = None,
    strict: bool = False,
) -> RegistryAPI:
    """Parse a spec given as a mapping, file path, URL or raw text."""

    parser = OpenAPIParser(strict=strict)
    if isinstance(source, Mapping):
        return parser.parse(source, source_url=source_url, name=name)
    text = str(source)
    if text.startswith(("http://", "https://")):
        return parser.parse_url(text, name=name)
    if "\n" in text or text.lstrip().startswith("{"):
        return parser.parse_text(text, source=None, source_url=source_url, name=name)
    return parser.parse_file(text, source_url=source_url, name=name)


def build_registry_document(
    sources: Iterable[str | Path | Mapping[str, Any]],
    *,
    version: str = "0.0.0",
    description: str | None = None,
    strict: bool = False,
) -> RegistryDocument:
    """Parse several specs into a single :class:`RegistryDocument`."""

    apis = [parse_openapi(src, strict=strict) for src in sources]
    return RegistryDocument(
        version=version,
        description=description,
        generated_at=datetime.now(tz=UTC),
        apis=apis,
    )


def build_registry(
    sources: Iterable[str | Path | Mapping[str, Any]],
    *,
    version: str = "0.0.0",
    description: str | None = None,
    strict: bool = False,
) -> Any:
    """Parse several specs and return a ready-to-query ``CamaraRegistry``."""

    from camara_pgps.camara.registry import CamaraRegistry

    document = build_registry_document(
        sources, version=version, description=description, strict=strict
    )
    return CamaraRegistry(document)
