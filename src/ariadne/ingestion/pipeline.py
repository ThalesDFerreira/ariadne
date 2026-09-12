"""Orquestracao da ingestao: buscar -> gravar -> chunk -> embed -> indexar."""

from __future__ import annotations

from dataclasses import dataclass, field

from ariadne.domain.models import Document
from ariadne.ingestion.chunking import ChunkingConfig, chunk_document
from ariadne.llm.embeddings import EmbeddingProvider, build_embedder
from ariadne.storage.database import connection
from ariadne.storage.schema import apply_schema
from ariadne.storage.vector_store import PgVectorStore


@dataclass
class IngestionReport:
    """O que aconteceu, para a CLI poder contar a historia."""

    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    chunks: int = 0

    def summary(self) -> str:
        return (
            f"{len(self.ingested)} documento(s) indexado(s), "
            f"{len(self.skipped)} inalterado(s), {self.chunks} chunk(s)"
        )


class IngestionPipeline:
    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        store: PgVectorStore | None = None,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self._embedder = embedder or build_embedder()
        self._store = store or PgVectorStore()
        self._chunking = chunking or ChunkingConfig()

    def run(self, documents: list[Document], *, force: bool = False) -> IngestionReport:
        """Ingere os documentos, pulando os que nao mudaram.

        Cada documento vai em sua propria transacao: se o embedding do decimo
        falhar, os nove anteriores continuam validos. Uma transacao unica
        cobrindo tudo transformaria um erro pontual em perda do lote inteiro.
        """
        report = IngestionReport()

        with connection() as conn:
            apply_schema(conn)

        for document in documents:
            with connection() as conn:
                changed = self._store.upsert_document(conn, document)
                if not changed and not force:
                    report.skipped.append(document.title)
                    continue

                document_id = self._store.resolve_document_id(
                    conn, document.source, document.external_id
                )
                if document_id is None:
                    msg = f"documento {document.external_id} sumiu apos o upsert"
                    raise RuntimeError(msg)

                chunks = chunk_document(document, self._chunking)
                for chunk in chunks:
                    chunk.document_id = document_id

                embeddings = self._embedder.embed_documents([c.content for c in chunks])
                self._store.replace_chunks(conn, document_id, chunks, embeddings)

                report.ingested.append(document.title)
                report.chunks += len(chunks)

        return report
