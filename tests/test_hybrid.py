"""Recuperacao hibrida contra o corpus real.

Estes testes assumem o corpus de demonstracao ingerido e o grafo construido
(`ariadne ingest-dir data/cvm && ariadne graph-build`). Sem isso, pulam com
motivo legivel em vez de falhar.

AS CONSULTAS SAEM DO GOLDEN SET
-------------------------------
A primeira versao destes testes tinha "privatizacao BNDES" e "Itabira" escritos
no codigo. Eles passaram a falhar no dia em que o corpus virou CVM -- e ninguem
viu, porque teste de integracao pula sozinho quando o banco esta fora, e o banco
passou semanas fora. Consulta escrita a mao envelhece junto com o corpus.

Agora os termos vem de `data/golden/questions.json`, que e versionado junto com
o corpus que ele descreve.
"""

from typing import Any

import httpx
import pytest

from ariadne.config import get_settings
from ariadne.eval.harness import fold, load_questions
from ariadne.retrieval.hybrid import HybridSearch, SearchMode
from ariadne.retrieval.lexical import LexicalSearch
from ariadne.storage.database import connection

pytestmark = pytest.mark.integration

CASOS: list[dict[str, Any]] = load_questions()
PERGUNTA = next(c for c in CASOS if c["kind"] == "factual")
MULTIHOP = [c for c in CASOS if c["kind"] == "multihop"]
# Uma ancora puramente alfabetica: "14,30%" e "24 mes" nao servem para testar
# casamento de termo exato numa tsquery.
TERMO = next(a for c in CASOS for a in c["anchors"] if a.replace(" ", "").isalpha())


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


def test_lexical_encontra_por_termo_exato(db, _require_corpus):
    hits = LexicalSearch().search(db, TERMO, limit=5)
    assert hits, f"busca lexical nao devolveu nada para {TERMO!r}"


def test_lexical_nao_exige_todos_os_termos(db, _require_corpus):
    """O bug que quase passou: com AND, pergunta natural devolvia zero.

    Uma pergunta inteira so casaria com um trecho que contivesse TODOS os seus
    termos ao mesmo tempo -- e nenhum contem. Tanto `plainto_tsquery` quanto
    `websearch_to_tsquery` ligam os termos com E; a tsquery aqui usa OU.
    """
    hits = LexicalSearch().search(db, PERGUNTA["question"], limit=5)
    assert hits, "pergunta em linguagem natural devolveu zero resultado"


def test_lexical_com_termo_inexistente_nao_quebra(db):
    assert LexicalSearch().search(db, "zzzxxxqqq", limit=5) == []


def test_lexical_com_query_so_de_stopwords_nao_quebra(db):
    """plainto_tsquery devolve vazio aqui; a query precisa lidar com isso."""
    assert LexicalSearch().search(db, "de a o e", limit=5) == []


# --- modos ------------------------------------------------------------------


@pytest.mark.parametrize("modo", [SearchMode.VECTOR, SearchMode.LEXICAL, SearchMode.HYBRID])
def test_todos_os_modos_devolvem_resultado_com_citacao(motor, modo):
    r = motor.search(PERGUNTA["question"], mode=modo, limit=3, rerank=False)
    assert r.hits
    for hit in r.hits:
        assert hit.citation().strip(), "resultado sem citacao de fonte"


def test_trace_conta_cada_etapa(motor):
    r = motor.search(PERGUNTA["question"], mode=SearchMode.HYBRID, limit=3, rerank=False)
    assert r.trace.vector > 0
    assert r.trace.lexical > 0
    assert r.trace.fused > 0
    assert r.trace.reranked == len(r.hits)


def test_hibrido_usa_o_grafo(motor):
    """Se a expansao nao rende nada, o hibrido virou so vetorial+lexical."""
    r = motor.search(TERMO, mode=SearchMode.HYBRID, limit=3, rerank=False)
    assert r.trace.graph > 0, "expansao pelo grafo nao trouxe nenhum candidato"
    assert r.trace.entities, "nenhuma entidade semente guiou a expansao"


def _trouxe_as_ancoras(motor, caso: dict[str, Any], **kwargs: Any) -> bool:
    hits = motor.search(caso["question"], **kwargs).hits
    textos = [fold(h.chunk.content) for h in hits]
    return all(any(fold(a) in t for t in textos) for a in caso["anchors"])


def test_multihop_precisa_do_pipeline_completo(motor):
    """A afirmacao que justifica o projeto inteiro, como teste.

    Se o RAG puro resolvesse todas as perguntas multi-hop, nao havia por que
    manter grafo, expansao e cross-encoder. A tabela do README mede isso em
    agregado (75% contra 100%); aqui fica travado como propriedade: existe pelo
    menos uma pergunta que o pipeline completo responde e a busca vetorial nao.

    Se este teste comecar a falhar porque o vetorial passou a acertar tudo, a
    conclusao nao e "conserte o teste" -- e que o corpus ficou facil demais para
    sustentar a comparacao, e o golden set precisa de perguntas melhores.
    """
    assert MULTIHOP, "golden set sem pergunta multi-hop"

    completo = [
        c
        for c in MULTIHOP
        if _trouxe_as_ancoras(motor, c, mode=SearchMode.HYBRID, limit=5, rerank=True)
    ]
    assert completo, "o pipeline completo nao respondeu nenhuma pergunta multi-hop"

    so_vetorial = [
        c
        for c in completo
        if _trouxe_as_ancoras(motor, c, mode=SearchMode.VECTOR, limit=5, rerank=False)
    ]
    assert len(so_vetorial) < len(completo), (
        "a busca vetorial pura resolveu todas as multi-hop -- o corpus nao "
        "sustenta mais a comparacao que motiva o GraphRAG"
    )


def test_pergunta_sem_resposta_no_corpus_nao_inventa(motor):
    r = motor.search("qual a receita de brigadeiro", mode=SearchMode.HYBRID, limit=3, rerank=False)
    # Pode devolver algo (sempre ha um vizinho mais proximo), mas tem que vir
    # com fonte -- o que permite a quem le perceber que nao responde.
    for hit in r.hits:
        assert hit.citation().strip()


def test_limite_e_respeitado(motor):
    r = motor.search("energia elétrica", mode=SearchMode.HYBRID, limit=2, rerank=False)
    assert len(r.hits) <= 2
