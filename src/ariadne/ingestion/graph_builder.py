"""Construcao do grafo a partir dos chunks ja indexados.

O caro aqui e a chamada de LLM por chunk. Com 8 GB de VRAM e um modelo 7B,
450 chunks passam de uma hora -- por isso o cache por fingerprint nao e
otimizacao, e requisito: sem ele, qualquer ajuste no pipeline custaria a hora
inteira de novo.
"""

from __future__ import annotations

from collections.abc import Callable
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
    dropped_edges: int = 0
    """Relacoes descartadas por apontarem para entidade que nao sobreviveu."""

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

    def build(
        self,
        *,
        limit: int | None = None,
        refresh: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> GraphReport:
        """Percorre os chunks, extrai e grava o grafo.

        Usa UMA conexao para todo o percurso. A versao anterior abria uma
        conexao por chunk e reexecutava o DDL do cache a cada leitura -- 900
        conexoes e 450 DDLs num corpus deste tamanho.
        """
        report = GraphReport()
        todas_entidades: list[ExtractedEntity] = []
        arestas_brutas: list[tuple[UUID, str, str, RelationType, str]] = []
        # (chunk, nome cru) -- a chave canonica so existe apos a resolucao.
        mencoes_brutas: list[tuple[UUID, str]] = []

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

            # Le o cache inteiro de uma vez: 450 SELECTs individuais custam
            # mais que uma varredura da tabela.
            cache: dict[str, Extraction] = {}
            if not refresh:
                cache = self._load_cache(conn)

            total = len(chunks)
            for i, (chunk_id, content, section, title) in enumerate(chunks, 1):
                contexto = f"{title} > {section}" if section else title
                fingerprint = chunk_fingerprint(
                    content,
                    self._extractor.fingerprint_model,
                    self._extractor.PROMPT_VERSION,
                )

                extraction = cache.get(fingerprint)
                if extraction is None:
                    try:
                        extraction = self._extractor.extract(content, context=contexto)
                    except Exception as exc:
                        # Um chunk problematico nao pode derrubar a construcao
                        # inteira depois de meia hora de GPU.
                        report.errors.append(f"{title}: {type(exc).__name__}: {exc}")
                        continue
                    self._write_cache(conn, fingerprint, chunk_id, extraction)
                else:
                    report.chunks_from_cache += 1

                report.chunks_processed += 1
                todas_entidades.extend(extraction.entities)
                mencoes_brutas.extend((chunk_id, e.name) for e in extraction.entities)
                arestas_brutas.extend(
                    (chunk_id, r.source, r.target, r.type, r.evidence) for r in extraction.relations
                )

                if on_progress and (i % 10 == 0 or i == total):
                    on_progress(i, total)

            nos = self._resolver.resolve(todas_entidades)
            arestas, descartadas = self._rewrite_edges(nos, arestas_brutas)

            self._store.upsert_nodes(conn, nos)
            self._store.upsert_edges(conn, arestas)
            self._write_mentions(conn, nos, mencoes_brutas)
            conn.commit()

        report.nodes = len(nos)
        report.edges = len(arestas)
        report.dropped_edges = descartadas
        return report

    def _rewrite_edges(
        self,
        nos: list[Any],
        brutas: list[tuple[UUID, str, str, RelationType, str]],
    ) -> tuple[list[GraphEdge], int]:
        """Reaponta as arestas para as chaves canonicas apos a resolucao.

        A fusao de entidades faz variantes deixarem de existir como no proprio.
        Sem esta reescrita, a aresta extraida de "CSN" apontaria para uma chave
        que nao esta mais no grafo.
        """
        alias_para_chave = {
            normalize_name(variante): no.key for no in nos for variante in [no.name, *no.aliases]
        }
        for no in nos:
            alias_para_chave[no.key] = no.key

        arestas: list[GraphEdge] = []
        descartadas = 0
        for chunk_id, origem, destino, tipo, evidencia in brutas:
            ok = alias_para_chave.get(normalize_name(origem))
            od = alias_para_chave.get(normalize_name(destino))
            if ok is None or od is None or ok == od:
                descartadas += 1
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
        return arestas, descartadas

    def _write_mentions(
        self,
        conn: psycopg.Connection[Any],
        nos: list[Any],
        brutas: list[tuple[UUID, str]],
    ) -> None:
        """Grava onde cada entidade aparece.

        E o elo que falta para a expansao k-hop: sem ele da para andar pelo
        grafo, mas nao para voltar dele ao texto -- e resposta sem trecho de
        origem nao serve neste projeto.
        """
        alias_para_chave = {
            normalize_name(variante): no.key for no in nos for variante in [no.name, *no.aliases]
        }
        pares = {
            (chave, chunk_id)
            for chunk_id, nome in brutas
            if (chave := alias_para_chave.get(normalize_name(nome))) is not None
        }
        with conn.cursor() as cur:
            cur.execute("DELETE FROM entity_mentions")
            if pares:
                cur.executemany(
                    "INSERT INTO entity_mentions (entity_key, chunk_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    sorted(pares),
                )

    def _load_cache(self, conn: psycopg.Connection[Any]) -> dict[str, Extraction]:
        with conn.cursor() as cur:
            cur.execute("SELECT fingerprint, payload FROM extraction_cache")
            linhas = cur.fetchall()
        saida: dict[str, Extraction] = {}
        for fingerprint, payload in linhas:
            try:
                saida[fingerprint] = Extraction.model_validate(payload)
            except ValueError:
                continue  # cache de um formato antigo: sera regravado
        return saida

    def _write_cache(
        self,
        conn: psycopg.Connection[Any],
        fingerprint: str,
        chunk_id: UUID,
        extraction: Extraction,
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO extraction_cache (fingerprint, chunk_id, payload)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (fingerprint) DO NOTHING
                """,
                (fingerprint, chunk_id, extraction.model_dump_json()),
            )
        # Commit a cada chunk: uma hora de GPU nao pode ser perdida porque o
        # processo caiu no chunk 400.
        conn.commit()
