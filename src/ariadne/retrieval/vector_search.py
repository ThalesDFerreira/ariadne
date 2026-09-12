"""Busca vetorial: o baseline contra o qual o GraphRAG sera comparado."""

from __future__ import annotations

from ariadne.domain.models import RetrievedChunk
from ariadne.llm.embeddings import EmbeddingProvider, build_embedder
from ariadne.storage.database import connection
from ariadne.storage.vector_store import PgVectorStore


class VectorSearch:
    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        store: PgVectorStore | None = None,
    ) -> None:
        self._embedder = embedder or build_embedder()
        self._store = store or PgVectorStore()

    def search(self, query: str, *, limit: int = 5) -> list[RetrievedChunk]:
        """A consulta vira vetor e busca os chunks mais proximos.

        E o caminho que o RAG tradicional faz e onde ele para. Guardar este
        baseline importa: sem ele, nao ha como provar com numero que a
        expansao pelo grafo melhora alguma coisa.
        """
        embedding = self._embedder.embed_query(query)
        with connection() as conn:
            return self._store.search(conn, embedding, limit=limit)
