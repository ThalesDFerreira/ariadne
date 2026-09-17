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
from ariadne.storage.schema import apply_schema


@pytest.fixture(scope="session")
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture(scope="session")
def _require_database(settings: Settings) -> None:
    """Exige o banco de pe E o schema aplicado.

    Aplicar o schema aqui nao e comodidade: um banco recem-subido nao tem
    tabela nenhuma, e sem isso os testes quebravam com
    `relation "chunks" does not exist`. Passava despercebido na maquina de
    quem ja tinha ingerido corpus, e quebrava no CI, que sobe o container
    zerado a cada execucao. Foi assim que o primeiro push do projeto descobriu.
    """
    try:
        with psycopg.connect(settings.pg_dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error as exc:
        pytest.skip(f"Postgres indisponivel em {settings.pg_host}:{settings.pg_port} ({exc})")

    with database.connection() as conn:
        apply_schema(conn)


@pytest.fixture(scope="session")
def _require_corpus(_require_database: None) -> None:
    """Pula quando o indice esta vazio.

    Teste que mede recuperacao precisa de algo para recuperar. Sem corpus ele
    nao falha por defeito do codigo -- falha por falta de dado, e um vermelho
    desses ensina a ignorar o vermelho.
    """
    with database.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        linha = cur.fetchone()
    if not linha or not linha[0]:
        pytest.skip("indice vazio -- rode `ariadne ingest-dir data/cvm`")


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
