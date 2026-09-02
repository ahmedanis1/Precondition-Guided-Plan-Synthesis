"""Documents, chunks and their metadata for the ingestion pipeline.

Covers the first three stages of the pipeline - loading, parsing and chunking -
plus the metadata model that travels with every chunk into ChromaDB:

    api_name, version, operation_id, source, document_type, section
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.camara.schemas import RegistryAPI, RegistryOperation
from camara_pgps.models import CamaraOperation

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
        return cls(
            document_type=DocumentType(data.get("document_type", DocumentType.TEXT)),
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
    """A loaded, parsed unit of text before chunking."""

    text: str
    metadata: ChunkMetadata


class Chunk(BaseModel):
    """A chunk ready to be embedded and stored."""

    id: str
    text: str
    metadata: ChunkMetadata


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


def documents_from_operation(
    operation: CamaraOperation,
    *,
    api_name: str,
    version: str | None,
    source: str | None,
) -> list[Document]:
    """Render one CAMARA operation into an operation / parameters / responses trio."""

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
    docs = [Document(text="\n".join(header), metadata=base)]

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
            )
        )
    return docs


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
