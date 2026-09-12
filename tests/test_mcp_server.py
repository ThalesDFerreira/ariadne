"""Servidor MCP: o contrato que qualquer assistente de IA vai consumir."""

import asyncio
import json

import httpx
import pytest

from ariadne.config import get_settings
from ariadne.mcp.server import server

pytestmark = pytest.mark.integration


def run(coro):
    return asyncio.run(coro)


def payload(result):
    """Extrai o conteudo estruturado devolvido pela tool."""
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    return json.loads(result.content[0].text)


@pytest.fixture(scope="module", autouse=True)
def _servicos():
    cfg = get_settings()
    try:
        httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=3.0).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Ollama indisponivel ({exc})")


def test_expoe_as_tools_da_fase():
    nomes = {t.name for t in run(server.list_tools())}
    assert {"search_knowledge", "graph_stats"} <= nomes


def test_as_tools_tem_descricao():
    """Sem descricao, o LLM nao sabe quando chamar a tool."""
    for tool in run(server.list_tools()):
        assert tool.description, f"tool {tool.name} sem descricao"


def test_busca_devolve_trechos():
    r = payload(run(server.call_tool("search_knowledge", {"query": "minério de ferro"})))
    assert r["hits"], "nenhum trecho recuperado -- o corpus foi ingerido?"


def test_regra_de_ouro_todo_trecho_tem_citacao():
    """O invariante do projeto inteiro, verificado no ponto de saida."""
    r = payload(run(server.call_tool("search_knowledge", {"query": "privatização"})))
    for hit in r["hits"]:
        assert hit["citation"].strip(), "trecho devolvido sem citacao de fonte"
        assert hit["content"].strip()


def test_limit_e_respeitado_e_limitado():
    r = payload(run(server.call_tool("search_knowledge", {"query": "banco", "limit": 2})))
    assert len(r["hits"]) <= 2
    # Pedido absurdo nao pode derrubar o servidor nem varrer o indice inteiro.
    grande = payload(run(server.call_tool("search_knowledge", {"query": "banco", "limit": 9999})))
    assert len(grande["hits"]) <= 20


def test_modo_pedido_e_o_modo_usado():
    """Ate a Fase 2 os modos graph e hybrid caiam no vetorial com aviso.

    Agora existem de verdade, e a tool precisa reportar o que realmente usou --
    responder "vector" a um pedido de "hybrid" enganaria o modelo que chamou.
    """
    for modo in ("vector", "lexical", "hybrid"):
        r = payload(run(server.call_tool("search_knowledge", {"query": "vale", "mode": modo})))
        assert r["mode"] == modo, f"pedi {modo}, recebi {r['mode']}"


def test_nota_explica_as_etapas():
    """Quem chamou a tool precisa conseguir dizer de onde veio a resposta."""
    r = payload(
        run(server.call_tool("search_knowledge", {"query": "privatização", "mode": "hybrid"}))
    )
    assert r["note"], "resposta sem explicacao das etapas"
    assert "vetorial=" in r["note"] and "grafo=" in r["note"]


def test_stats_reporta_corpus_e_grafo():
    r = payload(run(server.call_tool("graph_stats", {})))
    assert r["documents"] > 0
    assert r["chunks"] >= r["documents"]
    assert r["nodes"] > 0, "o grafo deveria estar populado -- rodou graph-build?"
    assert r["edges"] > 0


def test_resource_do_esquema_esta_registrado():
    recursos = run(server.list_resources())
    uris = {str(r.uri) for r in recursos}
    assert "ariadne://graph/schema" in uris
