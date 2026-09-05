"""RAG knowledge base: ingestion pipeline, embeddings and retriever."""

from camara_pgps.retrieval.documents import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentType,
    chunk_document,
    chunk_documents,
    documents_from_api,
    documents_from_operation,
    documents_from_registry,
    load_jsonl,
    load_markdown,
    load_text_file,
)
from camara_pgps.retrieval.embeddings import (
    Embedder,
    HashingEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from camara_pgps.retrieval.ingestion import (
    IngestionPipeline,
    IngestionResult,
    run_ingestion,
)
from camara_pgps.retrieval.rag import (
    RagContext,
    RagRetriever,
    RetrievedCondition,
    RetrievedOperation,
    RetrievedParameter,
    get_rag_retriever,
)
from camara_pgps.retrieval.retriever import (
    Retriever,
    VectorMatch,
    VectorStore,
    get_retriever,
)

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "Document",
    "DocumentType",
    "chunk_document",
    "chunk_documents",
    "documents_from_api",
    "documents_from_operation",
    "documents_from_registry",
    "load_jsonl",
    "load_markdown",
    "load_text_file",
    "Embedder",
    "HashingEmbedder",
    "OllamaEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
    "IngestionPipeline",
    "IngestionResult",
    "run_ingestion",
    "Retriever",
    "VectorMatch",
    "VectorStore",
    "get_retriever",
    "RagRetriever",
    "RagContext",
    "RetrievedOperation",
    "RetrievedParameter",
    "RetrievedCondition",
    "get_rag_retriever",
]
