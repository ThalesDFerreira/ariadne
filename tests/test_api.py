"""API HTTP local.

A API e fina de proposito -- ela nao decide nada, so embrulha o
`KnowledgeAgent`. Entao o que vale testar aqui e a BORDA: o que chega de fora e
mal formado, e o que sai para a tela.

O agente e substituido por um dublê. Carregar o de verdade traria o
cross-encoder de ~2 GB e transformaria um teste de validacao de entrada num
teste de integracao lento.
"""

from uuid import uuid4

import pytest
from starlette.testclient import TestClient

from ariadne.agents.answering import Answer
from ariadne.agents.orchestrator import AgentResult
from ariadne.agents.router import STRATEGIES, QueryKind
from ariadne.api import server
from ariadne.domain.graph import RelationType
from ariadne.domain.models import Chunk, RetrievedChunk
from ariadne.storage.graph_store import PathStep


class AgenteFalso:
    def __init__(self, resultado: AgentResult | None = None, erro: Exception | None = None) -> None:
        self._resultado = resultado
        self._erro = erro
        self.perguntas: list[str] = []

    def ask(self, pergunta: str) -> AgentResult:
        self.perguntas.append(pergunta)
        if self._erro is not None:
            raise self._erro
        assert self._resultado is not None
        return self._resultado


def resultado(
    *, texto: str = "A PetroReconcavo comprou da 3R Potiguar [1].", grounded: bool = True
) -> AgentResult:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        ordinal=0,
        content="  A PetroReconcavo adquiriu 50% dos ativos da 3R Potiguar.  ",
        section_path="Fato Relevante",
    )
    return AgentResult(
        question="De quem a PetroReconcavo comprou?",
        answer=Answer(
            text=texto,
            sources=["PETRORECONCAVO S.A. - Aquisicao"],
            grounded=grounded,
            warning=None if grounded else "resposta sem citacao valida",
        ),
        strategy=STRATEGIES[QueryKind.FACTUAL],
        hits=[
            RetrievedChunk(
                chunk=chunk,
                score=0.9123456,
                document_title="PETRORECONCAVO S.A.",
                document_url="file:///c/doc.pdf",
            )
        ],
        path=[
            PathStep(
                source="PetroReconcavo",
                relation=RelationType.ADQUIRIU,
                target="3R Potiguar",
                evidence="assinou contrato de compra",
            )
        ],
    )


@pytest.fixture
def cliente(monkeypatch):
    def usar(agente: AgenteFalso) -> TestClient:
        monkeypatch.setattr(server, "_agente", agente)
        return TestClient(server.build_app())

    return usar


# --- validacao de entrada ---------------------------------------------------


def test_pergunta_vazia_e_recusada(cliente):
    r = cliente(AgenteFalso(resultado())).post("/api/ask", json={"question": "   "})
    assert r.status_code == 400
    assert "vazia" in r.json()["error"]


def test_pergunta_sem_campo_e_recusada(cliente):
    assert cliente(AgenteFalso(resultado())).post("/api/ask", json={}).status_code == 400


def test_corpo_que_nao_e_json_e_recusado(cliente):
    r = cliente(AgenteFalso(resultado())).post("/api/ask", content=b"pergunta solta")
    assert r.status_code == 400


def test_pergunta_longa_demais_e_recusada(cliente):
    """Teto de tamanho.

    Pergunta gigante nao melhora a busca: o embedder trunca e o reranker paga o
    custo inteiro. Recusar na borda e mais barato que descobrir depois.
    """
    agente = AgenteFalso(resultado())
    r = cliente(agente).post("/api/ask", json={"question": "a" * (server.MAX_PERGUNTA + 1)})
    assert r.status_code == 400
    assert agente.perguntas == [], "pergunta recusada nao pode chegar ao motor"


def test_entrada_valida_chega_ao_motor_sem_espaco_sobrando(cliente):
    agente = AgenteFalso(resultado())
    cliente(agente).post("/api/ask", json={"question": "  quem comprou?  "})
    assert agente.perguntas == ["quem comprou?"]


# --- o que sai para a tela --------------------------------------------------


def test_resposta_traz_texto_fontes_e_rota(cliente):
    corpo = cliente(AgenteFalso(resultado())).post("/api/ask", json={"question": "q"}).json()

    assert corpo["answer"].startswith("A PetroReconcavo")
    assert corpo["sources"] == ["PETRORECONCAVO S.A. - Aquisicao"]
    assert corpo["grounded"] is True
    assert corpo["strategy"]["kind"] == "factual"
    assert corpo["strategy"]["graph_weight"] == 0.2
    assert isinstance(corpo["seconds"], float)


def test_resposta_sem_fundamento_chega_marcada(cliente):
    """A trava mais importante do sistema tem de sobreviver ate a tela.

    O motor marca `grounded=False` quando a resposta nao citou fonte valida.
    Se a API engolir esse campo, a tela mostra uma resposta com aparencia de
    verificada -- que e pior do que nao responder.
    """
    agente = AgenteFalso(resultado(grounded=False))
    corpo = cliente(agente).post("/api/ask", json={"question": "q"}).json()
    assert corpo["grounded"] is False
    assert corpo["warning"]


def test_trecho_vem_aparado_e_com_citacao(cliente):
    hit = (
        cliente(AgenteFalso(resultado())).post("/api/ask", json={"question": "q"}).json()["hits"][0]
    )
    assert hit["excerpt"].startswith("A PetroReconcavo"), "espaco das bordas deveria sair"
    assert "PETRORECONCAVO" in hit["citation"]
    assert hit["score"] == 0.9123


def test_caminho_do_grafo_vem_com_evidencia(cliente):
    passo = (
        cliente(AgenteFalso(resultado())).post("/api/ask", json={"question": "q"}).json()["path"][0]
    )
    assert passo["relation"] == "ADQUIRIU"
    assert passo["evidence"]


# --- falha ------------------------------------------------------------------


def test_motor_fora_do_ar_vira_erro_legivel(cliente):
    """Ollama ou Postgres fora derrubam a pergunta, nao o servidor.

    A mensagem precisa chegar a tela. Um 500 mudo faz o usuario achar que a
    pagina travou, e o motivo real (`ConnectError`) fica so no terminal.
    """
    agente = AgenteFalso(erro=ConnectionError("[WinError 10061] recusada"))
    r = cliente(agente).post("/api/ask", json={"question": "q"})
    assert r.status_code == 500
    assert "ConnectionError" in r.json()["error"]
    assert "10061" in r.json()["error"]


def test_stats_com_banco_fora_responde_503(cliente, monkeypatch):
    def explode():
        raise RuntimeError("PoolTimeout")

    monkeypatch.setattr(server, "connection", explode)
    r = cliente(AgenteFalso(resultado())).get("/api/stats")
    assert r.status_code == 503
    assert "PoolTimeout" in r.json()["error"]
