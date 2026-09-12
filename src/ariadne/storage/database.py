"""Pool de conexoes com o Postgres, ja preparado para o Apache AGE.

A pegadinha que este modulo existe para resolver: o AGE so responde em uma
conexao que tenha ag_catalog no search_path. Como um pool reaproveita e cria
conexoes o tempo todo, esse preparo precisa acontecer no hook de abertura --
se ficar espalhado pelo codigo, uma hora sai uma conexao crua do pool e o erro
aparece como "function cypher does not exist", que parece instalacao quebrada
mas nao e.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ariadne.config import Settings, get_settings

# Delimitador do dollar-quoting. Nomeado (e nao $$) para reduzir a chance de
# colidir com o conteudo de uma query.
_DOLLAR_TAG = "$ariadne$"

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
    params: dict[str, Any] | None = None,
    graph_name: str | None = None,
    columns: int = 1,
) -> list[dict[str, Any]]:
    """Executa Cypher no AGE e devolve linhas como dicionarios.

    Tres peculiaridades do AGE aparecem aqui, e todas tem motivo:

    1. Cypher nao roda solto: vem embrulhado num SELECT.
    2. Quem chama declara quantas colunas o RETURN produz, porque o Postgres
       precisa saber o formato do resultado antes de executar. Dai o parametro
       `columns`, que nao existiria num driver de Neo4j.
    3. A query precisa ser um literal dollar-quoted ($$...$$): o parser do AGE
       le o texto direto do source SQL e recusa uma string comum.

    O item 3 impede o escape normal do psycopg, entao VALORES nunca entram por
    interpolacao -- vao em `params`, que o AGE recebe como agtype e referencia
    no Cypher por $nome. E o que evita injecao de Cypher quando a entidade vem
    de um documento, e nao do nosso codigo.
    """
    cfg = get_settings()
    target = graph_name or cfg.graph_name

    if _DOLLAR_TAG in query:
        msg = f"a query Cypher nao pode conter o delimitador {_DOLLAR_TAG}"
        raise ValueError(msg)

    column_defs = sql.SQL(", ").join(
        sql.SQL("{} agtype").format(sql.Identifier(f"c{i}")) for i in range(columns)
    )
    # Seguro como SQL cru porque o delimitador foi validado acima.
    body = sql.SQL("{tag}{q}{tag}").format(tag=sql.SQL(_DOLLAR_TAG), q=sql.SQL(query))

    if params is None:
        statement = sql.SQL("SELECT * FROM ag_catalog.cypher({graph}, {body}) AS ({cols})").format(
            graph=sql.Literal(target), body=body, cols=column_defs
        )
        args: tuple[Any, ...] = ()
    else:
        statement = sql.SQL(
            "SELECT * FROM ag_catalog.cypher({graph}, {body}, {params}::ag_catalog.agtype)"
            " AS ({cols})"
        ).format(
            graph=sql.Literal(target),
            body=body,
            params=sql.Placeholder(),
            cols=column_defs,
        )
        args = (json.dumps(params),)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(statement, args)
        return list(cur.fetchall())
