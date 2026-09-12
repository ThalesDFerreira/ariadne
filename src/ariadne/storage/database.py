"""Pool de conexoes com o Postgres, ja preparado para o Apache AGE.

A pegadinha que este modulo existe para resolver: o AGE so responde em uma
conexao que tenha ag_catalog no search_path. Como um pool reaproveita e cria
conexoes o tempo todo, esse preparo precisa acontecer no hook de abertura --
se ficar espalhado pelo codigo, uma hora sai uma conexao crua do pool e o erro
aparece como "function cypher does not exist", que parece instalacao quebrada
mas nao e.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ariadne.config import Settings, get_settings

_pool: ConnectionPool | None = None


def _prepare_connection(conn: psycopg.Connection[Any]) -> None:
    """Roda a cada conexao nova do pool."""
    with conn.cursor() as cur:
        # Redundante quando shared_preload_libraries=age, mas mantem o projeto
        # funcionando tambem num Postgres sem esse preload configurado.
        cur.execute("LOAD 'age'")
        cur.execute('SET search_path = ag_catalog, "$user", public')
    conn.commit()


def get_pool(settings: Settings | None = None) -> ConnectionPool:
    """Pool unico do processo, criado sob demanda."""
    global _pool
    if _pool is None:
        cfg = settings or get_settings()
        _pool = ConnectionPool(
            conninfo=cfg.pg_dsn,
            min_size=1,
            max_size=10,
            configure=_prepare_connection,
            open=True,
        )
    return _pool


def close_pool() -> None:
    """Fecha o pool. Util em testes e no shutdown do servidor MCP."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection[Any]]:
    """Conexao do pool, com commit no sucesso e rollback no erro.

    E aqui que mora a vantagem de ter vetor e grafo no mesmo banco: o que
    acontecer dentro deste bloco -- embedding e aresta do grafo juntos --
    confirma ou desfaz como uma coisa so.
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def run_cypher(
    conn: psycopg.Connection[Any],
    query: str,
    *,
    graph_name: str | None = None,
    columns: int = 1,
) -> list[dict[str, Any]]:
    """Executa Cypher no AGE e devolve linhas como dicionarios.

    O AGE nao aceita Cypher solto: ele vem embrulhado em um SELECT, e quem
    chama precisa declarar quantas colunas o RETURN produz -- o Postgres exige
    saber o formato do resultado antes de executar. Por isso o parametro
    `columns`, que nao existiria num driver de Neo4j.

    O nome do grafo entra como literal escapado, nunca por f-string, porque
    concatenar identificador em SQL e como se abre buraco de injecao.
    """
    cfg = get_settings()
    target = graph_name or cfg.graph_name
    column_defs = sql.SQL(", ").join(
        sql.SQL("{} agtype").format(sql.Identifier(f"c{i}")) for i in range(columns)
    )
    statement = sql.SQL("SELECT * FROM ag_catalog.cypher({}, {}) AS ({})").format(
        sql.Literal(target),
        sql.Literal(query),
        column_defs,
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(statement)
        return list(cur.fetchall())
