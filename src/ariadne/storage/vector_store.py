"""Persistencia de documentos, chunks e embeddings no Postgres + pgvector."""

from __future__ import annotations

import json
from typing import Any, Protocol
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from ariadne.domain.models import Chunk, Document, RetrievedChunk
from ariadne.llm.embeddings import Vector


def to_pgvector(vector: Vector) -> str:
    """Serializa no formato textual que o pgvector aceita: '[1,2,3]'."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class VectorStore(Protocol):
    """Contrato de armazenamento vetorial, para o backend ser trocavel."""

    def upsert_document(self, conn: psycopg.Connection[Any], document: Document) -> bool: ...

    def replace_chunks(
        self,
        conn: psycopg.Connection[Any],
        document_id: UUID,
        chunks: list[Chunk],
        embeddings: list[Vector],
    ) -> None: ...

    def search(
        self,
        conn: psycopg.Connection[Any],
        embedding: Vector,
        *,
        limit: int = 10,
    ) -> list[RetrievedChunk]: ...


class PgVectorStore:
    """Implementacao sobre pgvector."""

    def upsert_document(self, conn: psycopg.Connection[Any], document: Document) -> bool:
        """Grava o documento. Devolve True se o conteudo mudou (ou e novo).

        O retorno e o que permite pular reprocessamento: se o hash bate, nao ha
        por que gerar embeddings nem, mais adiante, pagar chamadas de LLM.
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content_hash FROM documents
                 WHERE source = %s AND external_id = %s
                """,
                (document.source, document.external_id),
            )
            existing = cur.fetchone()

            if existing is not None and existing[1] == document.content_hash:
                return False

            cur.execute(
                """
                INSERT INTO documents
                    (id, source, external_id, title, url, content, content_hash, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (source, external_id) DO UPDATE SET
                    title        = EXCLUDED.title,
                    url          = EXCLUDED.url,
                    content      = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    metadata     = EXCLUDED.metadata,
                    ingested_at  = now()
                """,
                (
                    document.id,
                    document.source,
                    document.external_id,
                    document.title,
                    document.url,
                    document.content,
                    document.content_hash,
                    json.dumps(document.metadata, ensure_ascii=False),
                ),
            )
        return True

    def resolve_document_id(
        self, conn: psycopg.Connection[Any], source: str, external_id: str
    ) -> UUID | None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE source = %s AND external_id = %s",
                (source, external_id),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def replace_chunks(
        self,
        conn: psycopg.Connection[Any],
        document_id: UUID,
        chunks: list[Chunk],
        embeddings: list[Vector],
    ) -> None:
        """Troca todos os chunks do documento de uma vez.

        Substituir em bloco, e nao tentar casar chunk a chunk, e proposital: o
        chunking pode ter mudado entre ingestoes, e um "merge inteligente"
        deixaria orfaos apontando para texto que nao existe mais.
        """
        if len(chunks) != len(embeddings):
            msg = f"{len(chunks)} chunks para {len(embeddings)} embeddings"
            raise ValueError(msg)

        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            if not chunks:
                return
            cur.executemany(
                """
                INSERT INTO chunks
                    (id, document_id, ordinal, content, section_path, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                [
                    (
                        chunk.id,
                        document_id,
                        chunk.ordinal,
                        chunk.content,
                        chunk.section_path,
                        to_pgvector(embedding),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )

    def search(
        self,
        conn: psycopg.Connection[Any],
        embedding: Vector,
        *,
        limit: int = 10,
    ) -> list[RetrievedChunk]:
        """Vizinhos mais proximos por distancia de cosseno.

        O JOIN com documents nao e conveniencia: e o que garante que todo
        resultado ja sai com a fonte junto, cumprindo a regra de ouro de nunca
        devolver trecho sem citacao.
        """
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT c.id, c.document_id, c.ordinal, c.content, c.section_path,
                       d.title AS document_title, d.url AS document_url,
                       1 - (c.embedding <=> %s::vector) AS score
                  FROM chunks c
                  JOIN documents d ON d.id = c.document_id
                 WHERE c.embedding IS NOT NULL
                 ORDER BY c.embedding <=> %s::vector
                 LIMIT %s
                """,
                (to_pgvector(embedding), to_pgvector(embedding), limit),
            )
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                chunk=Chunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    ordinal=row["ordinal"],
                    content=row["content"],
                    section_path=row["section_path"],
                ),
                document_title=row["document_title"],
                document_url=row["document_url"],
                score=float(row["score"]),
            )
            for row in rows
        ]

    def stats(self, conn: psycopg.Connection[Any]) -> dict[str, int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT (SELECT count(*) FROM documents),
                       (SELECT count(*) FROM chunks),
                       (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL)
                """
            )
            row = cur.fetchone()
        assert row is not None
        return {"documents": row[0], "chunks": row[1], "embedded_chunks": row[2]}
