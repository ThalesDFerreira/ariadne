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


@pytest.fixture
def db(_require_database: None) -> Iterator[psycopg.Connection[Any]]:
    """Conexao preparada para o AGE, com rollback no fim.

    Rollback sempre: teste de integracao nao pode deixar sujeira no banco.
    """
    with database.connection() as conn:
        yield conn
        conn.rollback()
