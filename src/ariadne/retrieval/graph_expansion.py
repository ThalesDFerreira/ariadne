"""Expansao pelo grafo: o passo que o RAG tradicional nao tem.

O fluxo, em quatro tempos:

    1. chunks recuperados pela busca  ->  quais entidades aparecem neles
    2. entidades  ->  vizinhos no grafo, a k saltos
    3. vizinhos  ->  chunks onde ESSES aparecem
    4. esses chunks entram como mais um ranking na fusao

E isso que responde "a Metalurgica Beta depende de bauxita de qual regiao?"
quando um documento diz que a Beta compra da Alfa e OUTRO diz que a Alfa extrai
em Paragominas. Nenhum trecho contem a ligacao; ela existe so no grafo.

O preco e ruido: expandir demais traz chunks que falam de entidades vizinhas
mas nao respondem a pergunta. Por isso a expansao entra como ranking com PESO
MENOR na fusao e passa pelo reranker depois -- ela amplia o campo de busca, nao
decide a resposta.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from ariadne.domain.models import Chunk, RetrievedChunk


class GraphExpansion:
    """Busca chunks vizinhos no grafo a partir de chunks ja recuperados."""

    def __init__(self, graph_name: str | None = None) -> None:
        self._graph = graph_name

    def entities_in_chunks(self, conn: psycopg.Connection[Any], chunk_ids: list[Any]) -> list[str]:
        """Chaves das entidades mencionadas nos chunks dados."""
        if not chunk_ids:
            return []
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT entity_key FROM entity_mentions WHERE chunk_id = ANY(%s)",
                (list(chunk_ids),),
            )
            return [r[0] for r in cur.fetchall()]

    def expand(
        self,
        conn: psycopg.Connection[Any],
        seed_chunks: list[RetrievedChunk],
        *,
        depth: int = 1,
        max_entities: int = 12,
        limit: int = 20,
    ) -> list[RetrievedChunk]:
        """Chunks alcancados a partir dos chunks semente.

        `max_entities` existe porque um chunk denso cita dezenas de entidades,
        e expandir todas transformaria a consulta numa varredura do corpus.
        Como as sementes sao escolhidas esta em _select_discriminative.
        """
        if not seed_chunks:
            return []

        ids_semente = [h.chunk.id for h in seed_chunks]
        chaves = self._select_discriminative(conn, ids_semente, max_entities)
        if not chaves:
            return []

        vizinhas = self._neighbor_keys(conn, chaves, depth)
        alvo = vizinhas - set(chaves)
        if not alvo:
            return []

        return self._chunks_of_entities(conn, sorted(alvo), set(ids_semente), limit)

    # Entidade presente em mais que esta fracao do corpus e hub: liga tudo a
    # tudo e nao discrimina nada.
    HUB_RATIO = 0.05

    def select_seed_entities(
        self, conn: psycopg.Connection[Any], seed_chunks: list[RetrievedChunk], limite: int = 12
    ) -> list[str]:
        """Entidades que guiariam a expansao. Publica para o trace explicar."""
        if not seed_chunks:
            return []
        return self._select_discriminative(conn, [h.chunk.id for h in seed_chunks], limite)

    def _select_discriminative(
        self, conn: psycopg.Connection[Any], chunk_ids: list[Any], limite: int
    ) -> list[str]:
        """Entidades boas para expandir: relevantes a pergunta e nao genericas.

        Duas forcas em tensao, e escolher so uma delas da errado:

        - So as MAIS raras (primeira versao deste metodo) traz entidades que
          aparecem uma unica vez no corpus -- "Percival Farquhar", "Sistema
          Norte". Sao especificas demais e nao levam a lugar nenhum.
        - So as MAIS citadas traz "Brasil" e "Rio de Janeiro", que ligam
          qualquer coisa a qualquer coisa.

        A saida e cortar os hubs por um teto absoluto e, entre as que sobram,
        ordenar por quantas vezes aparecem NOS CHUNKS RECUPERADOS: isso e
        relevancia em relacao a pergunta, nao popularidade no corpus.
        """
        with conn.cursor() as cur:
            cur.execute("SELECT count(DISTINCT chunk_id) FROM entity_mentions")
            linha = cur.fetchone()
            total_chunks = (linha[0] if linha else 0) or 1
            teto = max(3, int(total_chunks * self.HUB_RATIO))

            cur.execute(
                """
                WITH locais AS (
                    SELECT entity_key, count(*) AS freq_local
                      FROM entity_mentions
                     WHERE chunk_id = ANY(%s)
                     GROUP BY entity_key
                ),
                globais AS (
                    SELECT entity_key, count(*) AS freq_global
                      FROM entity_mentions
                     WHERE entity_key IN (SELECT entity_key FROM locais)
                     GROUP BY entity_key
                )
                SELECT l.entity_key
                  FROM locais l JOIN globais g USING (entity_key)
                 WHERE g.freq_global <= %s
                 ORDER BY l.freq_local DESC, g.freq_global DESC, l.entity_key
                 LIMIT %s
                """,
                (list(chunk_ids), teto, limite),
            )
            return [r[0] for r in cur.fetchall()]

    def _neighbor_keys(
        self, conn: psycopg.Connection[Any], chaves: list[str], depth: int
    ) -> set[str]:
        """Vizinhos no grafo, em qualquer direcao.

        Nao dirigido pela mesma razao de find_connection: a informacao util
        chega tanto de quem a entidade aponta quanto de quem aponta para ela.
        """
        from ariadne.storage.database import run_cypher

        depth = max(1, min(depth, 2))
        achadas: set[str] = set()
        for chave in chaves:
            rows = run_cypher(
                conn,
                f"MATCH (a:Entity {{key: $key}})-[:RELATES_TO*1..{depth}]-(b:Entity) "
                f"RETURN DISTINCT b.key LIMIT 40",
                params={"key": chave},
                graph_name=self._graph,
            )
            for row in rows:
                valor = str(row["c0"]).strip('"')
                if valor and valor != "null":
                    achadas.add(valor)
        return achadas

    def _chunks_of_entities(
        self,
        conn: psycopg.Connection[Any],
        chaves: list[str],
        excluir: set[Any],
        limit: int,
    ) -> list[RetrievedChunk]:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT c.id, c.document_id, c.ordinal, c.content, c.section_path,
                       d.title AS document_title, d.url AS document_url,
                       count(DISTINCT m.entity_key) AS hits
                  FROM entity_mentions m
                  JOIN chunks c ON c.id = m.chunk_id
                  JOIN documents d ON d.id = c.document_id
                 WHERE m.entity_key = ANY(%s)
                   AND NOT (c.id = ANY(%s))
                 GROUP BY c.id, c.document_id, c.ordinal, c.content, c.section_path,
                          d.title, d.url
                 ORDER BY hits DESC, c.ordinal
                 LIMIT %s
                """,
                (chaves, list(excluir), limit),
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
                # Quantas entidades vizinhas o chunk cobre. Serve so para
                # ordenar esta lista; a fusao depois olha a posicao, nao o valor.
                score=float(row["hits"]),
            )
            for row in rows
        ]
