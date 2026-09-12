"""Fixtures compartilhadas.

Os testes de integracao dependem do Postgres do docker-compose. Em vez de
falharem feio quando o banco esta fora, eles se marcam como skipped com um
motivo legivel -- assim `pytest` continua util mesmo sem o Docker de pe.
"""

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from ariadne.config import Settings, get_settings
from ariadne.storage import database


@pytest.fixture(scope="session")
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture(scope="session")
def _require_database(settings: Settings) -> None:
    try:
        with psycopg.connect(settings.pg_dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error as exc:
        pytest.skip(f"Postgres indisponivel em {settings.pg_host}:{settings.pg_port} ({exc})")


TEST_GRAPH = "ariadne_test"


@pytest.fixture(scope="session")
def test_graph(_require_database: None) -> str:
    """Grafo separado para os testes.

    Sem isolamento, um `clear()` de fixture apagaria o grafo real -- e os
    testes passariam a depender de quais entidades o corpus tem no momento.
    """
    with database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s", (TEST_GRAPH,))
            existe = cur.fetchone()[0]
            if not existe:
                cur.execute("SELECT ag_catalog.create_graph(%s)", (TEST_GRAPH,))
        conn.commit()
    return TEST_GRAPH


@pytest.fixture
def db(_require_database: None) -> Iterator[psycopg.Connection[Any]]:
    """Conexao preparada para o AGE, com rollback no fim.

    Rollback sempre: teste de integracao nao pode deixar sujeira no banco.
    """
    with database.connection() as conn:
        yield conn
        conn.rollback()
