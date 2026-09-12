"""Fumaca da Fase 0.

Prova a afirmacao que sustenta a escolha de stack do projeto: pgvector e
Apache AGE conversam no mesmo Postgres, na mesma conexao e dentro da mesma
transacao. Se este arquivo passa, a fundacao esta de pe.

Os testes de Cypher usam um GRAFO SEPARADO. Sem isso eles escreviam no grafo
real e passaram a quebrar quando o corpus ganhou uma Petrobras de verdade --
teste de infraestrutura nao pode depender do conteudo ingerido.
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


def test_tabelas_do_projeto_ficam_em_public(db):
    """ag_catalog NAO pode vir primeiro, e "$user" nao pode entrar.

    Com ag_catalog na frente, todo CREATE TABLE sem qualificacao cria a tabela
    dentro do schema interno da extensao -- e um DROP EXTENSION age levaria os
    dados do projeto junto.

    Com "$user" no caminho e pior: o AGE cria um schema com o NOME DO GRAFO, e
    aqui o grafo se chama "ariadne" igual ao usuario do banco, entao as tabelas
    iam parar dentro do proprio grafo. Os dois casos ja aconteceram neste repo.
    """
    (search_path,) = db.execute("SHOW search_path").fetchone()
    posicoes = [s.strip().strip('"') for s in search_path.split(",")]
    assert posicoes[0] == "public", f"public deve vir primeiro: {search_path}"
    assert "ag_catalog" in posicoes, f"ag_catalog precisa estar no caminho: {search_path}"
    assert "$user" not in search_path, f'"$user" nao pode estar no caminho: {search_path}'

    rows = db.execute(
        """
        SELECT schemaname FROM pg_tables
         WHERE tablename IN ('documents', 'chunks', 'extraction_cache')
        """
    ).fetchall()
    assert rows, "tabelas do projeto nao encontradas"
    assert all(r[0] == "public" for r in rows), f"tabela fora de public: {rows}"


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


def test_vetor_sobrevive_a_ida_e_volta_ao_banco(db):
    from ariadne.storage.vector_store import to_pgvector

    original = [1e-07, -0.5, 2.0]
    (devolvido,) = db.execute("SELECT %s::vector(3)", (to_pgvector(original),)).fetchone()
    numeros = [float(x) for x in devolvido.strip("[]").split(",")]
    assert numeros == pytest.approx(original, rel=1e-6)


def test_cypher_cria_e_le_no_a_mesma_transacao(db, test_graph):
    run_cypher(db, "CREATE (n:Smoke {name: 'Teseu'}) RETURN n", graph_name=test_graph)
    rows = run_cypher(db, "MATCH (n:Smoke {name: 'Teseu'}) RETURN n.name", graph_name=test_graph)
    assert len(rows) == 1
    assert "Teseu" in str(rows[0]["c0"])
    db.rollback()


def test_vetor_e_grafo_commitam_juntos(db, test_graph):
    """O ponto central: uma transacao so cobrindo os dois mundos.

    Com Neo4j e Qdrant separados nao existe transacao comum -- uma falha no
    meio deixaria embedding orfao sem no, ou no sem embedding.
    """
    db.execute("CREATE TEMP TABLE smoke_chunk (id int, embedding vector(3)) ON COMMIT DROP")
    db.execute("INSERT INTO smoke_chunk VALUES (1, '[0.1,0.2,0.3]')")
    run_cypher(
        db,
        "CREATE (n:Smoke {name: 'Ariadne', chunk_id: 1}) RETURN n",
        graph_name=test_graph,
    )

    (chunks,) = db.execute("SELECT count(*) FROM smoke_chunk").fetchone()
    nodes = run_cypher(db, "MATCH (n:Smoke {name: 'Ariadne'}) RETURN n", graph_name=test_graph)
    assert chunks == 1
    assert len(nodes) == 1

    # E o rollback desfaz os dois lados de uma vez.
    db.rollback()
    remaining = run_cypher(db, "MATCH (n:Smoke {name: 'Ariadne'}) RETURN n", graph_name=test_graph)
    assert remaining == []


def test_cypher_aceita_parametros(db, test_graph):
    """Valores vao por params, nunca concatenados na query."""
    run_cypher(
        db,
        "CREATE (n:Smoke {name: $nome, tipo: $tipo}) RETURN n",
        params={"nome": "Petrobras", "tipo": "Empresa"},
        graph_name=test_graph,
    )
    rows = run_cypher(
        db,
        "MATCH (n:Smoke {name: $nome}) RETURN n.tipo",
        params={"nome": "Petrobras"},
        graph_name=test_graph,
    )
    assert len(rows) == 1
    assert "Empresa" in str(rows[0]["c0"])
    db.rollback()


def test_valor_malicioso_nao_vira_cypher(db, test_graph):
    """Uma entidade extraida de documento e input nao confiavel.

    Se o valor fosse concatenado na query, o trecho abaixo viraria comando.
    Indo por params, ele continua sendo apenas um nome esquisito.
    """
    veneno = "x$ariadne$}) DETACH DELETE (n) //"
    run_cypher(
        db,
        "CREATE (n:Smoke {name: $nome}) RETURN n",
        params={"nome": veneno},
        graph_name=test_graph,
    )
    rows = run_cypher(
        db,
        "MATCH (n:Smoke {name: $nome}) RETURN n.name",
        params={"nome": veneno},
        graph_name=test_graph,
    )
    assert len(rows) == 1
    assert veneno in str(rows[0]["c0"])
    db.rollback()


def test_query_com_o_delimitador_e_rejeitada(db, test_graph):
    """O dollar-quoting so e seguro se o delimitador nao aparecer na query."""
    with pytest.raises(ValueError, match="delimitador"):
        run_cypher(db, "MATCH (n) RETURN $ariadne$ n", graph_name=test_graph)
