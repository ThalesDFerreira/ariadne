"""Construcao do grafo a partir dos chunks ja indexados.

O caro aqui e a chamada de LLM por chunk. Com 8 GB de VRAM e um modelo 7B,
450 chunks levam mais de uma hora -- por isso o cache por fingerprint nao e
otimizacao, e requisito: sem ele, qualquer ajuste no pipeline custaria a hora
inteira de novo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import psycopg

from ariadne.domain.graph import (
    ExtractedEntity,
    Extraction,
    GraphEdge,
    RelationType,
    normalize_name,
)
from ariadne.ingestion.entity_resolution import EntityResolver
from ariadne.llm.extraction import OllamaExtractor, chunk_fingerprint
from ariadne.storage.database import connection
from ariadne.storage.graph_store import AgeGraphStore

_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS extraction_cache (
    fingerprint  TEXT PRIMARY KEY,
    chunk_id     UUID NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS extraction_cache_chunk_idx ON extraction_cache (chunk_id);
"""


@dataclass
class GraphReport:
    chunks_processed: int = 0
    chunks_from_cache: int = 0
    nodes: int = 0
    edges: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.chunks_processed} chunk(s) processado(s) "
            f"({self.chunks_from_cache} do cache), "
            f"{self.nodes} entidade(s), {self.edges} relacao(oes)"
        )


class GraphBuilder:
    def __init__(
        self,
        extractor: OllamaExtractor | None = None,
        store: AgeGraphStore | None = None,
        resolver: EntityResolver | None = None,
    ) -> None:
        self._extractor = extractor or OllamaExtractor()
        self._store = store or AgeGraphStore()
        self._resolver = resolver or EntityResolver()

    def ensure_cache(self, conn: psycopg.Connection[Any]) -> None:
        with conn.cursor() as cur:
            cur.execute(_CACHE_DDL)
        conn.commit()

    def build(self, *, limit: int | None = None, refresh: bool = False) -> GraphReport:
        """Percorre os chunks, extrai e grava o grafo."""
        report = GraphReport()

        with connection() as conn:
            self.ensure_cache(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.content, c.section_path, d.title
                      FROM chunks c JOIN documents d ON d.id = c.document_id
                     ORDER BY d.title, c.ordinal
                    """
                    + (f" LIMIT {int(limit)}" if limit else "")
                )
                chunks = cur.fetchall()

        todas_entidades: list[ExtractedEntity] = []
        arestas_brutas: list[tuple[UUID, str, str, RelationType, str]] = []

        for chunk_id, content, section, title in chunks:
            contexto = f"{title} > {section}" if section else title
            fingerprint = chunk_fingerprint(
                content, self._extractor.fingerprint_model, self._extractor.PROMPT_VERSION
            )

            extraction = None if refresh else self._read_cache(fingerprint)
            if extraction is None:
                try:
                    extraction = self._extractor.extract(content, context=contexto)
                except Exception as exc:
                    # Um chunk problematico nao pode derrubar a construcao
                    # inteira depois de uma hora de GPU.
                    report.errors.append(f"{title}: {exc}")
                    continue
                self._write_cache(fingerprint, chunk_id, extraction)
            else:
                report.chunks_from_cache += 1

            report.chunks_processed += 1
            todas_entidades.extend(extraction.entities)
            for rel in extraction.relations:
                arestas_brutas.append((chunk_id, rel.source, rel.target, rel.type, rel.evidence))

        nos = self._resolver.resolve(todas_entidades)
        # A resolucao pode ter fundido nomes, entao as arestas precisam ser
        # reescritas para as chaves canonicas -- senao apontariam para nos que
        # deixaram de existir.
        por_chave = {no.key: no for no in nos}
        alias_para_chave = {
            normalize_name(alias): no.key for no in nos for alias in [*no.aliases, no.name]
        }

        arestas: list[GraphEdge] = []
        for chunk_id, origem, destino, tipo, evidencia in arestas_brutas:
            ok = alias_para_chave.get(normalize_name(origem))
            od = alias_para_chave.get(normalize_name(destino))
            if ok is None or od is None or ok == od:
                continue
            arestas.append(
                GraphEdge(
                    source_key=ok,
                    target_key=od,
                    type=tipo,
                    chunk_id=chunk_id,
                    evidence=evidencia,
                )
            )

        with connection() as conn:
            self._store.upsert_nodes(conn, list(por_chave.values()))
            self._store.upsert_edges(conn, arestas)

        report.nodes = len(por_chave)
        report.edges = len(arestas)
        return report

    def _read_cache(self, fingerprint: str) -> Extraction | None:
        with connection() as conn:
            self.ensure_cache(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM extraction_cache WHERE fingerprint = %s",
                    (fingerprint,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return Extraction.model_validate(row[0])

    def _write_cache(self, fingerprint: str, chunk_id: UUID, extraction: Extraction) -> None:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO extraction_cache (fingerprint, chunk_id, payload)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (fingerprint) DO NOTHING
                    """,
                (fingerprint, chunk_id, extraction.model_dump_json()),
            )
