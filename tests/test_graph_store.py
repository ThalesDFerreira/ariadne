"""Grafo no Apache AGE, contra o banco real.

Varios destes testes existem porque o AGE se comportou de forma inesperada e
o contorno precisa ficar travado: se um upgrade da extensao mudar qualquer um
desses comportamentos, e melhor descobrir aqui do que num grafo silenciosamente
sem evidencia.
"""

from uuid import uuid4

import pytest

from ariadne.domain.graph import EntityType, GraphEdge, GraphNode, RelationType
from ariadne.storage.graph_store import AgeGraphStore

pytestmark = pytest.mark.integration

CHUNK = uuid4()


@pytest.fixture
def store(db, test_graph):
    s = AgeGraphStore(test_graph)
    s.clear(db)
    s.upsert_nodes(
        db,
        [
            GraphNode(key="alfa", name="Mineradora Alfa", type=EntityType.ORGANIZACAO, mentions=3),
            GraphNode(key="beta", name="Metalurgica Beta", type=EntityType.ORGANIZACAO, mentions=2),
            GraphNode(key="paragominas", name="Paragominas", type=EntityType.LUGAR, mentions=1),
            GraphNode(key="gama", name="Trading Gama", type=EntityType.ORGANIZACAO, mentions=1),
        ],
    )
    s.upsert_edges(
        db,
        [
            GraphEdge(
                source_key="alfa",
                target_key="paragominas",
                type=RelationType.ATUA_EM,
                chunk_id=CHUNK,
                evidence="A Alfa extrai bauxita em Paragominas",
            ),
            GraphEdge(
                source_key="alfa",
                target_key="beta",
                type=RelationType.FORNECE_PARA,
                chunk_id=CHUNK,
                evidence="A Beta compra bauxita da Alfa",
            ),
            GraphEdge(
                source_key="beta",
                target_key="gama",
                type=RelationType.FORNECE_PARA,
                chunk_id=CHUNK,
                evidence="A Gama revende o aluminio da Beta",
            ),
        ],
    )
    return s


def test_contagens(store, db):
    stats = store.stats(db)
    assert stats["nodes"] == 4
    assert stats["edges"] == 3


def test_upsert_e_idempotente(store, db):
    antes = store.stats(db)
    store.upsert_nodes(
        db, [GraphNode(key="alfa", name="Mineradora Alfa", type=EntityType.ORGANIZACAO)]
    )
    store.upsert_edges(
        db,
        [
            GraphEdge(
                source_key="alfa",
                target_key="paragominas",
                type=RelationType.ATUA_EM,
                chunk_id=CHUNK,
                evidence="A Alfa extrai bauxita em Paragominas",
            )
        ],
    )
    assert store.stats(db) == antes


def test_aresta_carrega_a_evidencia(store, db):
    """Regra de ouro no grafo: aresta sem evidencia e afirmacao sem fonte.

    O AGE ignora `SET` em aresta recem-criada por MERGE, o que fazia a
    evidencia sumir silenciosamente. Este teste trava o contorno.
    """
    vizinhos = store.neighbors(db, "alfa")
    assert vizinhos
    for v in vizinhos:
        assert v.evidence.strip(), f"aresta para {v.name} veio sem evidencia"


def test_vizinhanca_cobre_as_duas_direcoes(store, db):
    vizinhos = store.neighbors(db, "beta")
    direcoes = {v.direction for v in vizinhos}
    nomes = {v.name for v in vizinhos}
    assert direcoes == {"saindo", "entrando"}
    assert "Mineradora Alfa" in nomes  # quem fornece para a Beta
    assert "Trading Gama" in nomes  # para quem a Beta fornece


def test_tipo_de_relacao_sobrevive_a_ida_e_volta(store, db):
    vizinhos = store.neighbors(db, "alfa")
    tipos = {v.relation for v in vizinhos}
    assert RelationType.ATUA_EM in tipos
    assert RelationType.FORNECE_PARA in tipos


def test_profundidade_maior_alcanca_o_vizinho_do_vizinho(store, db):
    perto = {v.name for v in store.neighbors(db, "alfa", depth=1)}
    longe = {v.name for v in store.neighbors(db, "alfa", depth=2)}
    assert "Trading Gama" not in perto
    assert "Trading Gama" in longe


def test_caminho_multi_hop_traz_a_justificativa_de_cada_passo(store, db):
    """O caso que motiva o projeto: ligacao que nenhum trecho contem sozinho."""
    caminho = store.shortest_path(db, "alfa", "gama")
    assert len(caminho) == 2
    assert caminho[0].source == "Mineradora Alfa"
    assert caminho[-1].target == "Trading Gama"
    for passo in caminho:
        assert passo.evidence.strip(), "passo do caminho sem evidencia"


def test_caminho_de_um_salto(store, db):
    caminho = store.shortest_path(db, "alfa", "beta")
    assert len(caminho) == 1
    assert caminho[0].relation is RelationType.FORNECE_PARA


def test_entidade_inexistente_nao_quebra(store, db):
    assert store.neighbors(db, "nao-existe") == []
    assert store.shortest_path(db, "nao-existe", "alfa") == []


def test_busca_nao_dirigida_acha_ligacao_pela_ponta_oposta(store, db):
    """Quem pergunta "qual a ligacao entre X e Y" nao quer uma direcao so.

    Nao existe aresta saindo de paragominas, mas ela se liga a gama atraves
    da alfa. Exigir direcao unica devolvia "nenhum caminho" para pares
    claramente conectados.
    """
    assert store.shortest_path(db, "paragominas", "gama", directed=True) == []
    assert store.shortest_path(db, "paragominas", "gama", directed=False)


def test_hub_de_lugar_nao_vira_atalho(store, db):
    """ "As duas ficam no mesmo lugar" e caminho valido e informativamente vazio."""
    from ariadne.domain.graph import EntityType

    caminho = store.shortest_path(db, "beta", "paragominas", avoid_hub_types=())
    assert caminho
    sem_lugar = store.shortest_path(db, "beta", "paragominas", avoid_hub_types=(EntityType.LUGAR,))
    # O destino ainda pode ser um Lugar; o filtro vale para os INTERMEDIARIOS.
    assert all(p.target != "Paragominas" for p in sem_lugar[:-1])


def test_find_key_localiza_pela_forma_normalizada(db, test_graph):
    s = AgeGraphStore(test_graph)
    s.clear(db)
    s.upsert_nodes(
        db,
        [GraphNode(key="petrobras", name="Petrobras", type=EntityType.ORGANIZACAO)],
    )
    assert s.find_key(db, "PETROBRAS S.A.") == "petrobras"
    assert s.find_key(db, "Petrobrás") == "petrobras"
    assert s.find_key(db, "Empresa Inexistente") is None


def test_nome_malicioso_nao_vira_comando(db, test_graph):
    """Nomes vem de documentos: input nao confiavel."""
    s = AgeGraphStore(test_graph)
    s.clear(db)
    veneno = "x'}) DETACH DELETE (n) //"
    s.upsert_nodes(db, [GraphNode(key="veneno", name=veneno, type=EntityType.ORGANIZACAO)])
    assert s.stats(db)["nodes"] == 1
