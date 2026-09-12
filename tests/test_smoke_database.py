"""Fumaca da Fase 0.

Prova a afirmacao que sustenta a escolha de stack do projeto: pgvector e
Apache AGE conversam no mesmo Postgres, na mesma conexao e dentro da mesma
transacao. Se este arquivo passa, a fundacao esta de pe.
"""

import pytest

from ariadne.storage.database import run_cypher

pytestmark = pytest.mark.integration


def test_as_duas_extensoes_estao_instaladas(db):
    rows = db.execute(
        "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'age')"
    ).fetchall()
    assert {r[0] for r in rows} == {"vector", "age"}


def test_search_path_da_conexao_do_pool_enxerga_o_age(db):
    # Se o hook do pool falhar, e aqui que aparece -- e nao num erro confuso
    # de "function cypher does not exist" no meio da ingestao.
    (search_path,) = db.execute("SHOW search_path").fetchone()
    assert "ag_catalog" in search_path


def test_o_grafo_ariadne_existe(db):
    (count,) = db.execute(
        "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = 'ariadne'"
    ).fetchone()
    assert count == 1


def test_busca_vetorial_funciona(db):
    db.execute("CREATE TEMP TABLE smoke_vec (id int, embedding vector(3)) ON COMMIT DROP")
    db.execute("INSERT INTO smoke_vec VALUES (1, '[1,0,0]'), (2, '[0,1,0]')")
    # <=> e distancia de cosseno: 0 = identico, 1 = ortogonal.
    rows = db.execute(
        "SELECT id, embedding <=> '[1,0,0]' AS dist FROM smoke_vec ORDER BY dist"
    ).fetchall()
    assert rows[0][0] == 1
    assert rows[0][1] == pytest.approx(0.0, abs=1e-6)


def test_cypher_cria_e_le_no_a_mesma_transacao(db):
    run_cypher(db, "CREATE (n:Entity {name: 'Teseu'}) RETURN n")
    rows = run_cypher(db, "MATCH (n:Entity {name: 'Teseu'}) RETURN n.name")
    assert len(rows) == 1
    assert "Teseu" in str(rows[0]["c0"])


def test_vetor_e_grafo_commitam_juntos(db):
    """O ponto central: uma transacao so cobrindo os dois mundos.

    Com Neo4j e Qdrant separados nao existe transacao comum -- uma falha no
    meio deixaria embedding orfao sem no, ou no sem embedding.
    """
    db.execute("CREATE TEMP TABLE smoke_chunk (id int, embedding vector(3)) ON COMMIT DROP")
    db.execute("INSERT INTO smoke_chunk VALUES (1, '[0.1,0.2,0.3]')")
    run_cypher(db, "CREATE (n:Entity {name: 'Ariadne', chunk_id: 1}) RETURN n")

    (chunks,) = db.execute("SELECT count(*) FROM smoke_chunk").fetchone()
    nodes = run_cypher(db, "MATCH (n:Entity {name: 'Ariadne'}) RETURN n")
    assert chunks == 1
    assert len(nodes) == 1

    # E o rollback desfaz os dois lados de uma vez.
    db.rollback()
    remaining = run_cypher(db, "MATCH (n:Entity {name: 'Ariadne'}) RETURN n")
    assert remaining == []
