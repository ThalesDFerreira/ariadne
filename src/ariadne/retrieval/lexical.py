"""Busca lexical com o full-text search do Postgres.

Existe porque o vetorial erra de um jeito especifico: ele aproxima por
SIGNIFICADO e por isso confunde coisas parecidas. Na Fase 1, a pergunta sobre
minerio de ferro em "Itabira" trouxe em primeiro lugar uma empresa que opera em
"Itabirito" -- palavras diferentes, vetores vizinhos.

A busca lexical erra ao contrario: casa termo exato e ignora sinonimo. O
stemmer portugues reduz "minerio" a `miner` e "mineradora" a `mineradour`, que
nao casam, entao a consulta devolve zero onde o vetorial acertaria.

Nenhuma das duas e suficiente. Juntas, uma cobre o buraco da outra -- e e isso
que a fusao do modulo `fusion` faz.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from ariadne.domain.models import Chunk, RetrievedChunk


class LexicalSearch:
    """Full-text search em portugues, com ranking do proprio Postgres."""

    def search(
        self,
        conn: psycopg.Connection[Any],
        query: str,
        *,
        limit: int = 20,
    ) -> list[RetrievedChunk]:
        """Chunks que casam lexicalmente com a consulta.

        A tsquery e montada com OU entre os termos, e nao com E. Essa foi a
        diferenca entre funcionar e nao funcionar: tanto `plainto_tsquery`
        quanto `websearch_to_tsquery` ligam os termos com AND, entao a pergunta
        "Quem extrai minerio de ferro em Itabira?" so casaria com um trecho que
        contivesse TODOS os termos ao mesmo tempo. Na pratica devolvia zero
        resultado para praticamente qualquer pergunta em linguagem natural.

        O truque para trocar o conectivo sem abrir espaco para injecao: deixar
        o `plainto_tsquery` fazer a analise e a sanitizacao -- ele ja descarta
        stopwords e aplica o stemmer -- e so entao substituir `&` por `|` no
        texto da tsquery resultante. Nada do que o usuario digitou chega cru.

        `ts_rank_cd` leva em conta a PROXIMIDADE entre os termos, nao so a
        frequencia: com OU isso passa a importar muito, porque e o que separa o
        trecho que fala de "minerio de ferro" daquele que so menciona "ferro".
        """
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH q AS (
                    SELECT NULLIF(
                        replace(plainto_tsquery('portuguese', %s)::text, '&', '|'),
                        ''
                    )::tsquery AS tsq
                )
                SELECT c.id, c.document_id, c.ordinal, c.content, c.section_path,
                       d.title AS document_title, d.url AS document_url,
                       ts_rank_cd(c.content_tsv, q.tsq) AS score
                  FROM chunks c
                  JOIN documents d ON d.id = c.document_id
                  CROSS JOIN q
                 WHERE q.tsq IS NOT NULL
                   AND c.content_tsv @@ q.tsq
                 ORDER BY score DESC
                 LIMIT %s
                """,
                (query, limit),
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
