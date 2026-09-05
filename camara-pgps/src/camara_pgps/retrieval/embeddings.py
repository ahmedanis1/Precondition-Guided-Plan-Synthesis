"""Embedding backends for the ingestion / retrieval pipeline.

Implementations (selected by ``embedding.provider``):

* :class:`SentenceTransformerEmbedder` (``sentence-transformers``) - a local model,
  imported lazily.
* :class:`OllamaEmbedder` (``ollama``) - calls an Ollama server's embedding
  endpoint over HTTP.
* :class:`HashingEmbedder` (``hash``) - a deterministic, dependency-free hashing
  embedder for offline runs and tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from camara_pgps.config import Settings, get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used across the pipeline."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedder (no model download)."""

    def __init__(self, *, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model_name = f"hash-{dimension}"

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        features = list(tokens)
        for token in tokens:
            padded = f"^{token}$"
            features.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return features

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode(), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return _normalise(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class SentenceTransformerEmbedder:
    """Wrapper around a ``sentence-transformers`` model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector.tolist()] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in vector.tolist()]


class OllamaEmbedder:
    """Embeddings from an Ollama server (``POST {base_url}/api/embed``).

    ``model_name`` is the Ollama model tag, e.g. ``nomic-embed-text`` or
    ``mxbai-embed-large``. Vectors are L2-normalised so cosine and dot-product
    distances agree, matching the other backends.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        *,
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = 0  # set after the first successful call
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def _request(self, texts: list[str]) -> list[list[float]]:
        response = self._http().post(
            f"{self._base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("embeddings")
        if raw is None and "embedding" in payload:  # legacy /api/embeddings shape
            raw = [payload["embedding"]]
        if not raw:
            raise ValueError(f"Ollama returned no embeddings for model {self.model_name!r}")
        vectors = [_normalise([float(value) for value in vector]) for vector in raw]
        if not self.dimension:
            self.dimension = len(vectors[0])
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        out: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            out.extend(self._request(items[start : start + self.batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Build the embedder selected by ``settings.embedding``."""

    settings = settings or get_settings()
    provider = settings.embedding.provider.lower()
    if provider in {"hash", "hashing"}:
        return HashingEmbedder()
    if provider in {"ollama"}:
        return OllamaEmbedder(
            settings.embedding.model_name,
            base_url=settings.embedding.base_url,
            batch_size=settings.embedding.batch_size,
        )
    if provider in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(
            settings.embedding.model_name,
            device=settings.embedding.device,
            batch_size=settings.embedding.batch_size,
        )
    raise ValueError(f"unknown embedding provider: {settings.embedding.provider!r}")
