"""Camada agentica: roteamento e resposta citada.

Os testes de validacao de citacao sao os mais importantes do arquivo. Todo o
resto do sistema apenas recupera texto que existe; e aqui que um LLM escreve
prosa nova, e prosa nova pode afirmar o que quiser.
"""

from uuid import uuid4

import httpx
import pytest

from ariadne.agents.answering import AnswerGenerator
from ariadne.agents.router import QueryKind, QueryRouter
from ariadne.config import get_settings
from ariadne.domain.models import Chunk, RetrievedChunk

DOC = uuid4()


def hit(texto: str, titulo: str = "Doc") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(document_id=DOC, ordinal=0, content=texto),
        document_title=titulo,
        document_url=f"https://exemplo.org/{titulo}",
        score=1.0,
    )


# --- roteamento heuristico (sem LLM) ----------------------------------------


@pytest.fixture
def router():
    return QueryRouter(use_llm=False)


@pytest.mark.parametrize(
    ("pergunta", "esperado"),
    [
        ("Quando a Vale foi privatizada?", QueryKind.FACTUAL),
        ("Onde fica a sede da Petrobras?", QueryKind.FACTUAL),
        ("Qual a ligação entre o Bradesco e a Previ?", QueryKind.RELACIONAL),
        ("Quais empresas foram privatizadas?", QueryKind.SINTESE),
        ("Liste os bancos do corpus", QueryKind.SINTESE),
    ],
)
def test_classificacao_heuristica(router, pergunta, esperado):
    assert router.route(pergunta).kind is esperado


def test_pergunta_que_casa_com_dois_padroes_e_relacional(router):
    """A pergunta que motivou o projeto inteiro.

    "Quais fornecedores dependem da mesma materia-prima que a X?" tem "quais"
    (sintese) e "dependem/mesma" (relacional). A parte dificil e a ligacao,
    entao relacional tem que vencer -- e por isso a ordem dos testes importa.
    """
    assert (
        router.route("Quais fornecedores dependem da mesma matéria-prima que a Usiminas?").kind
        is QueryKind.RELACIONAL
    )


def test_estrategias_diferem_onde_importa(router):
    factual = router.route("Quando a Vale foi privatizada?")
    relacional = router.route("Qual a ligação entre A e B?")
    sintese = router.route("Quais empresas existem?")

    # Relacional aposta no grafo; factual quase o ignora.
    assert relacional.graph_weight > factual.graph_weight
    # Sintese precisa de recall, entao busca mais candidatos.
    assert sintese.candidate_pool > factual.candidate_pool
    # E nao reordena: o cross-encoder enterra itens validos numa pergunta de lista.
    assert sintese.rerank is False


def test_heuristica_extrai_nomes_proprios(router):
    entidades = router.route("Qual a relação entre Bradesco e Previ?").entities
    assert "Bradesco" in entidades
    assert "Previ" in entidades


def test_router_nao_quebra_sem_ollama():
    """LLM fora do ar degrada para heuristica, nao derruba o sistema."""
    r = QueryRouter(client=httpx.Client(base_url="http://127.0.0.1:9"))
    estrategia = r.route("Quando a Vale foi privatizada?")
    r.close()
    assert estrategia.kind is QueryKind.FACTUAL
    assert "heuristica" in estrategia.reason


# --- validacao de citacoes --------------------------------------------------


@pytest.fixture
def gerador():
    return AnswerGenerator()


def test_sem_trechos_nao_inventa(gerador):
    r = gerador.answer("qualquer coisa", [])
    assert r.grounded is False
    assert "Nao encontrei" in r.text


def test_citacao_valida_e_preservada(gerador):
    chunks = [hit("A Vale foi privatizada em 1997.", "Vale")]
    r = gerador._validate("A Vale foi privatizada em 1997 [1].", chunks)
    assert r.grounded is True
    assert r.sources == ["Vale (https://exemplo.org/Vale)"]
    assert "[1]" in r.text
    assert r.warning is None


def test_citacao_inexistente_e_removida(gerador):
    """Modelo pequeno cita [7] tendo recebido dois trechos.

    Deixar passar e pior do que nao citar: a resposta ganha aparencia de
    verificada justamente onde nao esta.
    """
    chunks = [hit("um"), hit("dois")]
    r = gerador._validate("Afirmacao apoiada [1] e outra inventada [7].", chunks)
    assert "[7]" not in r.text
    assert "[1]" in r.text
    assert r.warning and "inexistente" in r.warning
    assert r.grounded is True


def test_resposta_sem_nenhuma_citacao_e_marcada(gerador):
    chunks = [hit("conteudo")]
    r = gerador._validate("A resposta e essa, sem citar nada.", chunks)
    assert r.grounded is False
    assert r.sources == []
    assert r.warning and "nao citou" in r.warning


def test_so_citacoes_invalidas_derruba_o_grounded(gerador):
    chunks = [hit("conteudo")]
    r = gerador._validate("Tudo inventado [5][9].", chunks)
    assert r.grounded is False
    assert "[5]" not in r.text and "[9]" not in r.text
    assert r.warning and "inexistente" in r.warning and "nao citou" in r.warning


def test_citacao_repetida_nao_duplica_fonte(gerador):
    chunks = [hit("conteudo", "Unico")]
    r = gerador._validate("Isto [1] e aquilo [1] tambem.", chunks)
    assert len(r.sources) == 1


def test_contexto_numera_os_trechos(gerador):
    from ariadne.agents.answering import AnswerContext

    texto = AnswerContext([hit("primeiro", "A"), hit("segundo", "B")]).render()
    assert "[1]" in texto and "[2]" in texto
    assert "primeiro" in texto and "segundo" in texto


def test_gerador_nao_quebra_sem_ollama():
    g = AnswerGenerator(client=httpx.Client(base_url="http://127.0.0.1:9"))
    r = g.answer("pergunta", [hit("conteudo")])
    g.close()
    assert r.grounded is False
    assert r.warning and "falha" in r.warning


# --- ponta a ponta ----------------------------------------------------------


@pytest.mark.integration
def test_agente_responde_com_fonte():
    cfg = get_settings()
    try:
        httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=3.0).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Ollama indisponivel ({exc})")

    from ariadne.agents.orchestrator import KnowledgeAgent

    agente = KnowledgeAgent()
    resultado = agente.ask("Quando a Vale foi privatizada?")
    agente.close()

    assert resultado.hits, "nenhum trecho recuperado"
    assert resultado.strategy.kind in set(QueryKind)
    # A resposta pode nao citar (modelo pequeno erra), mas o sistema tem que
    # DIZER isso em vez de entregar prosa sem procedencia calada.
    assert resultado.answer.grounded or resultado.answer.warning
