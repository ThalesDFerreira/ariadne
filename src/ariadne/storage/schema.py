"""Schema relacional do corpus.

DDL em SQL puro, aplicado de forma idempotente, em vez de Alembic: nesta fase
o schema ainda muda toda semana e uma ferramenta de migracao so adicionaria
cerimonia. Quando o formato estabilizar, migra-se para migracoes de verdade.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg import sql

from ariadne.config import Settings, get_settings

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY,
    source        TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    title         TEXT NOT NULL,
    url           TEXT,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Chave natural: reingerir a mesma pagina atualiza em vez de duplicar.
    UNIQUE (source, external_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id            UUID PRIMARY KEY,
    document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    content       TEXT NOT NULL,
    section_path  TEXT NOT NULL DEFAULT '',
    embedding     vector(:DIM),
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- Coluna GERADA: o tsvector se mantem sozinho a cada INSERT/UPDATE, sem
-- trigger e sem risco de ficar dessincronizado do texto.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING gin (content_tsv);

-- Liga entidade do grafo aos trechos onde ela foi mencionada. E o que torna a
-- expansao k-hop possivel: recuperar chunks -> descobrir entidades -> andar
-- pelo grafo -> voltar para os chunks das entidades vizinhas.
CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_key  TEXT NOT NULL,
    chunk_id    UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_key, chunk_id)
);

CREATE INDEX IF NOT EXISTS entity_mentions_chunk_idx ON entity_mentions (chunk_id);
"""

# HNSW e um grafo de navegacao: em vez de comparar a consulta com todos os
# vetores, ele salta entre vizinhos cada vez mais proximos. Troca exatidao por
# velocidade -- pode perder um resultado de vez em quando, mas responde em
# milissegundos onde a busca exata levaria segundos. Com poucos milhares de
# chunks a diferenca ainda e pequena; o indice existe para o corpus crescer
# sem precisar reescrever nada.
_INDEX = """
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def apply_schema(conn: psycopg.Connection[Any], settings: Settings | None = None) -> None:
    """Cria tabelas e indices se ainda nao existirem."""
    cfg = settings or get_settings()
    dim = int(cfg.embedding_dim)  # int() explicito: vai concatenado no DDL.
    with conn.cursor() as cur:
        # replace, e nao .format(): o DDL contem '{}'::jsonb, que o format
        # leria como placeholder posicional.
        cur.execute(sql.SQL(_DDL.replace(":DIM", str(dim))))
        cur.execute(sql.SQL(_INDEX))
    conn.commit()
