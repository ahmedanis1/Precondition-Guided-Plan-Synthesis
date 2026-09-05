"""Documents, chunks and their metadata for the ingestion pipeline.

Covers the first three stages of the pipeline - loading, parsing and chunking -
plus the metadata model that travels with every chunk into ChromaDB:

    api_name, version, operation_id, source, document_type, section
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.camara.schemas import RegistryAPI, RegistryOperation
from camara_pgps.models import (
    APIParameter,
    CamaraOperation,
    Condition,
    ConditionKind,
    HTTPMethod,
    ParameterLocation,
)

_MetaValue = str | int | float | bool


class DocumentType(StrEnum):
    """What kind of CAMARA material a chunk was derived from."""

    API_OVERVIEW = "api_overview"
    OPERATION = "operation"
    PARAMETERS = "parameters"
    RESPONSES = "responses"
    OPENAPI = "openapi"
    MARKDOWN = "markdown"
    TEXT = "text"


class ChunkMetadata(BaseModel):
    """Provenance and routing metadata attached to every chunk."""

    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType = DocumentType.TEXT
    api_name: str | None = None
    version: str | None = None
    operation_id: str | None = None
    source: str | None = None
    section: str | None = None
    extra: dict[str, _MetaValue] = Field(default_factory=dict)

    def to_chroma(self) -> dict[str, _MetaValue]:
        """Flatten to a ChromaDB-safe metadata dict (no ``None`` values)."""

        flat: dict[str, _MetaValue] = {"document_type": self.document_type.value}
        for key in ("api_name", "version", "operation_id", "source", "section"):
            value = getattr(self, key)
            if value is not None:
                flat[key] = value
        flat.update(self.extra)
        return flat

    @classmethod
    def from_chroma(cls, data: Mapping[str, Any] | None) -> ChunkMetadata:
        data = dict(data or {})
        known = {"document_type", "api_name", "version", "operation_id", "source", "section"}
        extra = {k: v for k, v in data.items() if k not in known}
        try:
            document_type = DocumentType(data.get("document_type", DocumentType.TEXT))
        except ValueError:
            document_type = DocumentType.TEXT
            if data.get("document_type"):
                extra["document_type_raw"] = data["document_type"]
        return cls(
            document_type=document_type,
            api_name=data.get("api_name"),
            version=data.get("version"),
            operation_id=data.get("operation_id"),
            source=data.get("source"),
            section=data.get("section"),
            extra=extra,
        )

    def with_section(self, section: str) -> ChunkMetadata:
        return self.model_copy(update={"section": section})


class Document(BaseModel):
    """A loaded, parsed unit of text before chunking.

    ``structured`` mirrors :attr:`Chunk.structured`: an optional machine-readable
    payload (e.g. an operation's ``operation_id`` / ``method`` / ``path`` /
    ``parameters``) that is carried through chunking rather than re-derived.
    """

    text: str
    metadata: ChunkMetadata
    structured: dict[str, Any] | None = None


class Chunk(BaseModel):
    """A chunk ready to be embedded and stored.

    ``structured`` carries an optional machine-readable payload from a JSONL
    corpus record (kept alongside the text; not itself embedded).
    """

    id: str
    text: str
    metadata: ChunkMetadata
    structured: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# loading / parsing
# --------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def load_text_file(path: str | Path, *, source: str | None = None) -> Document:
    """Load a plain-text file as a single document."""

    file_path = Path(path)
    return Document(
        text=file_path.read_text("utf-8"),
        metadata=ChunkMetadata(
            document_type=DocumentType.TEXT,
            source=source or str(file_path),
        ),
    )


def load_markdown(
    text_or_path: str | Path,
    *,
    source: str | None = None,
) -> list[Document]:
    """Parse Markdown into one document per section (split on ATX headings)."""

    if isinstance(text_or_path, Path) or (
        isinstance(text_or_path, str) and "\n" not in text_or_path and Path(text_or_path).is_file()
    ):
        file_path = Path(text_or_path)
        text = file_path.read_text("utf-8")
        source = source or str(file_path)
    else:
        text = str(text_or_path)
        source = source or "<markdown>"

    documents: list[Document] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        section = " > ".join(title for _, title in heading_stack) or None
        documents.append(
            Document(
                text=body,
                metadata=ChunkMetadata(
                    document_type=DocumentType.MARKDOWN,
                    source=source,
                    section=section,
                ),
            )
        )

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return documents


def _condition_payload(condition: Condition) -> dict[str, Any]:
    return {
        "description": condition.description,
        "id": condition.id,
        "expression": condition.expression,
        "requires": list(condition.requires),
        "produces": list(condition.produces),
        "source": condition.source,
    }


def operation_to_structured(
    operation: CamaraOperation,
    *,
    api_name: str,
    version: str | None = None,
) -> dict[str, Any]:
    """Serialise an operation into the JSONL ``structured`` shape.

    The inverse of :func:`structured_to_operation`: it lets the registry-backed
    ingestion path emit the same machine-readable payload a JSONL corpus carries,
    so RAG has a single code path (structured payload -> operation) and never
    needs a runtime registry lookup.
    """

    payload: dict[str, Any] = {
        "operation_id": operation.operation_id,
        "method": operation.method.value,
        "path": operation.path,
        "api_name": api_name,
        "parameters": [
            {
                "name": param.name,
                "location": param.location.value,
                "required": param.required,
                "description": param.description,
                "enum": list(param.enum),
            }
            for param in operation.parameters
        ],
    }
    if version:
        payload["api_version"] = version
    if operation.summary:
        payload["summary"] = operation.summary
    if operation.description:
        payload["description"] = operation.description
    if operation.tags:
        payload["tags"] = list(operation.tags)
    if operation.security_scopes:
        payload["security_scopes"] = list(operation.security_scopes)
    if operation.request_schema is not None:
        payload["request_body"] = operation.request_schema
    if operation.preconditions:
        payload["preconditions"] = [_condition_payload(c) for c in operation.preconditions]
    if operation.postconditions:
        payload["postconditions"] = [
            _condition_payload(c) for c in operation.postconditions
        ]
    capability = getattr(operation, "capability", None)
    if capability:
        payload["capability"] = capability
    return payload


def documents_from_operation(
    operation: CamaraOperation,
    *,
    api_name: str,
    version: str | None,
    source: str | None,
) -> list[Document]:
    """Render one CAMARA operation into an operation / parameters / responses trio.

    Every document in the trio carries the operation's ``structured`` payload so
    the retrieved chunk can be turned back into a :class:`CamaraOperation` without
    a registry lookup.
    """

    structured = operation_to_structured(operation, api_name=api_name, version=version)

    base = ChunkMetadata(
        document_type=DocumentType.OPERATION,
        api_name=api_name,
        version=version,
        operation_id=operation.operation_id,
        source=source,
        section=f"operations/{operation.operation_id}",
    )

    header = [
        f"Operation: {operation.operation_id}",
        f"API: {api_name}",
        f"{operation.method} {operation.path}",
    ]
    if operation.summary:
        header.append(f"Summary: {operation.summary}")
    if operation.description:
        header.append(f"Description: {operation.description}")
    if operation.tags:
        header.append(f"Tags: {', '.join(operation.tags)}")
    if operation.security_scopes:
        header.append(f"Required scopes: {', '.join(operation.security_scopes)}")
    if isinstance(operation, RegistryOperation) and operation.capability:
        header.append(f"Capability: {operation.capability}")
    docs = [Document(text="\n".join(header), metadata=base, structured=structured)]

    if operation.parameters:
        lines = [f"Parameters for {operation.operation_id} ({operation.method} {operation.path}):"]
        for param in operation.parameters:
            bits = [f"- {param.name} (in: {param.location}"]
            bits.append("required" if param.required else "optional")
            if param.enum:
                bits.append(f"enum: {', '.join(str(v) for v in param.enum)}")
            detail = ", ".join(bits) + ")"
            if param.description:
                detail += f" - {param.description}"
            lines.append(detail)
        docs.append(
            Document(
                text="\n".join(lines),
                metadata=base.model_copy(
                    update={
                        "document_type": DocumentType.PARAMETERS,
                        "section": f"operations/{operation.operation_id}/parameters",
                    }
                ),
                structured=structured,
            )
        )

    responses = list(getattr(operation, "responses", []) or [])
    if responses:
        lines = [f"Responses for {operation.operation_id}:"]
        for resp in responses:
            kind = "error" if resp.is_error else "success"
            line = f"- {resp.status_code} ({kind})"
            if resp.description:
                line += f": {resp.description}"
            lines.append(line)
        docs.append(
            Document(
                text="\n".join(lines),
                metadata=base.model_copy(
                    update={
                        "document_type": DocumentType.RESPONSES,
                        "section": f"operations/{operation.operation_id}/responses",
                    }
                ),
                structured=structured,
            )
        )
    return docs

def load_jsonl(path: str | Path) -> list[Chunk]:
    """Load a pre-chunked JSONL knowledge corpus into :class:`Chunk` objects.

    One record per line::

        {"id": "...", "text": "...", "metadata": {...}, "structured": {...}}

    ``id`` and ``structured`` are optional. ``metadata`` is parsed leniently: the
    known keys (``api_name``, ``version``, ``operation_id``, ``source``,
    ``section``, ``document_type``) populate :class:`ChunkMetadata`; anything else
    goes into ``metadata.extra``.
    """

    file_path = Path(path)
    chunks: list[Chunk] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {file_path}"
                ) from exc
            if not isinstance(record, Mapping):
                continue

            text = str(record.get("text", "")).strip()
            if not text:
                continue

            metadata = ChunkMetadata.from_chroma(record.get("metadata") or {})
            if metadata.source is None:
                metadata = metadata.model_copy(update={"source": str(file_path)})

            identifier = str(
                record.get("id")
                or chunk_id(metadata.source, metadata.section, line_number, text)
            )
            chunks.append(
                Chunk(
                    id=identifier,
                    text=text,
                    metadata=metadata,
                    structured=record.get("structured"),
                )
            )
    return chunks


def _structured_conditions(raw: Any, kind: ConditionKind) -> list[Condition]:
    """Parse a ``preconditions`` / ``postconditions`` payload into conditions.

    Read only if the payload carries it - a JSONL corpus without explicit PGPS
    facts yields an operation with no conditions (and hence single-step plans).
    """

    conditions: list[Condition] = []
    for entry in raw or []:
        if isinstance(entry, str):
            conditions.append(Condition(kind=kind, description=entry))
        elif isinstance(entry, Mapping):
            conditions.append(
                Condition(
                    kind=kind,
                    description=str(
                        entry.get("description") or entry.get("expression") or ""
                    ),
                    id=entry.get("id"),
                    expression=entry.get("expression"),
                    requires=[str(v) for v in entry.get("requires") or []],
                    produces=[str(v) for v in entry.get("produces") or []],
                    source=entry.get("source"),
                )
            )
    return conditions


def structured_to_operation(
    data: Mapping[str, Any], *, api_name: str | None = None
) -> CamaraOperation:
    """Build a :class:`CamaraOperation` from a JSONL record's ``structured`` payload.

    This is the bridge in *CAMARA OpenAPI -> JSONL -> Chroma -> RAG -> operation*:
    a retrieved chunk's machine-readable payload becomes the operation object the
    planner and validator consume, so no separate registry lookup is needed.
    """

    for key in ("operation_id", "method", "path"):
        if not data.get(key):
            raise ValueError(f"structured operation payload is missing {key!r}")

    method = str(data["method"]).upper()
    try:
        http_method = HTTPMethod(method)
    except ValueError as exc:
        raise ValueError(f"unsupported HTTP method {method!r}") from exc

    parameters: list[APIParameter] = []
    for entry in data.get("parameters") or []:
        if not isinstance(entry, Mapping) or not entry.get("name"):
            continue
        location = str(entry.get("location") or entry.get("in") or "query").lower()
        try:
            param_location = ParameterLocation(location)
        except ValueError:
            param_location = ParameterLocation.QUERY
        parameters.append(
            APIParameter(
                name=str(entry["name"]),
                location=param_location,
                required=bool(entry.get("required", False)),
                description=entry.get("description"),
                enum=list(entry.get("enum") or []),
            )
        )

    request_body = data.get("request_body")
    request_schema = dict(request_body) if isinstance(request_body, Mapping) else None

    return CamaraOperation(
        operation_id=str(data["operation_id"]),
        method=http_method,
        path=str(data["path"]),
        summary=data.get("summary"),
        description=data.get("description"),
        parameters=parameters,
        request_schema=request_schema,
        preconditions=_structured_conditions(
            data.get("preconditions"), ConditionKind.PRECONDITION
        ),
        postconditions=_structured_conditions(
            data.get("postconditions"), ConditionKind.POSTCONDITION
        ),
        tags=[str(t) for t in data.get("tags") or []],
        security_scopes=[
            str(s) for s in data.get("security_scopes") or data.get("scopes") or []
        ],
        api_name=api_name or data.get("api_name"),
    )


def documents_from_api(api: RegistryAPI) -> list[Document]:
    """Render a whole CAMARA API (overview + every operation) into documents."""

    source = api.provenance.location or api.source_file or api.spec_url

    overview = [
        f"API: {api.title or api.name}",
        f"Name: {api.name}",
        f"Version: {api.version}",
    ]
    if api.description:
        overview.append(f"Description: {api.description}")
    if api.base_path:
        overview.append(f"Base path: {api.base_path}")
    overview.append(
        f"Provenance: {api.provenance.source_type} ({api.provenance.location})"
    )
    if api.provenance.note:
        overview.append(f"Note: {api.provenance.note}")
    overview.append(
        "Operations: " + ", ".join(op.operation_id for op in api.operations)
    )

    documents = [
        Document(
            text="\n".join(overview),
            metadata=ChunkMetadata(
                document_type=DocumentType.API_OVERVIEW,
                api_name=api.name,
                version=api.version,
                source=source,
                section="overview",
            ),
        )
    ]
    for operation in api.operations:
        documents.extend(
            documents_from_operation(
                operation,
                api_name=api.name,
                version=api.version,
                source=source,
            )
        )
    return documents


def documents_from_registry(registry: CamaraRegistry) -> list[Document]:
    """Render every API in a registry into documents."""

    documents: list[Document] = []
    for name in registry.api_names():
        api = registry.document.get_api(name)
        if api is not None:
            documents.extend(documents_from_api(api))
    return documents


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #

def _split_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            for start in range(0, len(para), max_chars - overlap):
                pieces.append(para[start : start + max_chars])
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars:
            pieces.append(current)
            current = para
        else:
            current = candidate
    if current:
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def chunk_id(source: str | None, section: str | None, ordinal: int, text: str) -> str:
    digest = hashlib.sha1(
        f"{source}|{section}|{ordinal}|{text}".encode(),
        usedforsecurity=False,
    ).hexdigest()
    return digest[:16]


def chunk_document(
    document: Document,
    *,
    max_chars: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    parts = _split_text(document.text, max_chars=max_chars, overlap=overlap)
    total = len(parts)
    chunks: list[Chunk] = []
    for ordinal, part in enumerate(parts):
        section = document.metadata.section
        if total > 1:
            section = f"{section or 'body'}#{ordinal}"
        metadata = document.metadata.model_copy(update={"section": section})
        chunks.append(
            Chunk(
                id=chunk_id(document.metadata.source, document.metadata.section, ordinal, part),
                text=part,
                metadata=metadata,
                structured=document.structured,
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    *,
    max_chars: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for document in documents:
        for chunk in chunk_document(document, max_chars=max_chars, overlap=overlap):
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            chunks.append(chunk)
    return chunks


def chunk_texts(chunks: Sequence[Chunk]) -> list[str]:
    return [c.text for c in chunks]
