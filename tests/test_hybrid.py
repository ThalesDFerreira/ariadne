"""Recuperacao hibrida contra o corpus real.

Estes testes assumem o corpus de demonstracao ingerido e o grafo construido
(`ariadne ingest && ariadne graph-build`). Sem isso, pulam com motivo legivel
em vez de falhar.
"""

import httpx
import pytest

from ariadne.config import get_settings
from ariadne.retrieval.hybrid import HybridSearch, SearchMode
from ariadne.retrieval.lexical import LexicalSearch
from ariadne.storage.database import connection

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def motor():
    cfg = get_settings()
    try:
        httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=3.0).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Ollama indisponivel ({exc})")

    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entity_mentions")
        linha = cur.fetchone()
    if not linha or not linha[0]:
        pytest.skip("grafo vazio -- rode `ariadne graph-build`")

    # Sem reranker: o cross-encoder baixa ~2 GB e nao e o alvo destes testes.
    return HybridSearch()


# --- busca lexical ----------------------------------------------------------


def test_lexical_encontra_por_termo_exato(db):
    hits = LexicalSearch().search(db, "privatização BNDES", limit=5)
    assert hits, "busca lexical nao devolveu nada"


def test_lexical_nao_exige_todos_os_termos(db):
    """O bug que quase passou: com AND, pergunta natural devolvia zero.

    "Quem extrai minerio de ferro em Itabira?" so casaria com um trecho que
    contivesse todos os termos ao mesmo tempo -- e nenhum contem.
    """
    hits = LexicalSearch().search(db, "Quem extrai minério de ferro em Itabira?", limit=5)
    assert hits, "pergunta em linguagem natural devolveu zero resultado"


def test_lexical_com_termo_inexistente_nao_quebra(db):
    assert LexicalSearch().search(db, "zzzxxxqqq", limit=5) == []


def test_lexical_com_query_so_de_stopwords_nao_quebra(db):
    """plainto_tsquery devolve vazio aqui; a query precisa lidar com isso."""
    assert LexicalSearch().search(db, "de a o e", limit=5) == []


# --- modos ------------------------------------------------------------------


@pytest.mark.parametrize("modo", [SearchMode.VECTOR, SearchMode.LEXICAL, SearchMode.HYBRID])
def test_todos_os_modos_devolvem_resultado_com_citacao(motor, modo):
    r = motor.search("privatização da Vale", mode=modo, limit=3, rerank=False)
    assert r.hits
    for hit in r.hits:
        assert hit.citation().strip(), "resultado sem citacao de fonte"


def test_trace_conta_cada_etapa(motor):
    r = motor.search("aquisições do Bradesco", mode=SearchMode.HYBRID, limit=3, rerank=False)
    assert r.trace.vector > 0
    assert r.trace.lexical > 0
    assert r.trace.fused > 0
    assert r.trace.reranked == len(r.hits)


def test_hibrido_usa_o_grafo(motor):
    """Se a expansao nao rende nada, o hibrido virou so vetorial+lexical."""
    r = motor.search("minério de ferro", mode=SearchMode.HYBRID, limit=3, rerank=False)
    assert r.trace.graph > 0, "expansao pelo grafo nao trouxe nenhum candidato"
    assert r.trace.entities, "nenhuma entidade semente guiou a expansao"


def test_hibrido_corrige_erro_do_vetorial(motor):
    """O caso registrado na Fase 1.

    A pergunta diz "Itabira", que e da Vale. A busca vetorial trazia em
    primeiro lugar a Gerdau, que opera em "Itabirito" -- palavras parecidas,
    vetores vizinhos, empresa errada.
    """
    pergunta = "Quem extrai minério de ferro em Itabira?"

    vetorial = motor.search(pergunta, mode=SearchMode.VECTOR, limit=3, rerank=False)
    hibrido = motor.search(pergunta, mode=SearchMode.HYBRID, limit=3, rerank=False)

    titulos_hibrido = [h.document_title for h in hibrido.hits]
    assert any("Vale" in t for t in titulos_hibrido), (
        f"o hibrido deveria trazer a Vale; trouxe {titulos_hibrido}"
    )
    # Registro do baseline: se o vetorial sozinho tambem passar a acertar, o
    # comentario acima ficou desatualizado -- o que seria uma boa noticia.
    assert vetorial.hits


def test_pergunta_sem_resposta_no_corpus_nao_inventa(motor):
    r = motor.search("qual a receita de brigadeiro", mode=SearchMode.HYBRID, limit=3, rerank=False)
    # Pode devolver algo (sempre ha um vizinho mais proximo), mas tem que vir
    # com fonte -- o que permite a quem le perceber que nao responde.
    for hit in r.hits:
        assert hit.citation().strip()


def test_limite_e_respeitado(motor):
    r = motor.search("energia elétrica", mode=SearchMode.HYBRID, limit=2, rerank=False)
    assert len(r.hits) <= 2
