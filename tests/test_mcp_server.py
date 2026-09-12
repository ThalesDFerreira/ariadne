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


def test_modo_ainda_inexistente_avisa_em_vez_de_fingir():
    """Prometer expansao por grafo que nao existe seria mentir para o LLM."""
    r = payload(run(server.call_tool("search_knowledge", {"query": "vale", "mode": "graph"})))
    assert r["mode"] == "vector"
    assert r["note"] and "Fase 3" in r["note"]


def test_modo_vetorial_nao_gera_aviso():
    r = payload(run(server.call_tool("search_knowledge", {"query": "vale", "mode": "vector"})))
    assert r["note"] is None


def test_stats_reporta_corpus_e_grafo_vazio():
    r = payload(run(server.call_tool("graph_stats", {})))
    assert r["documents"] > 0
    assert r["chunks"] >= r["documents"]
    assert r["nodes"] == 0 and r["edges"] == 0
    assert "Fase 2" in (r["note"] or "")


def test_resource_do_esquema_esta_registrado():
    recursos = run(server.list_resources())
    uris = {str(r.uri) for r in recursos}
    assert "ariadne://graph/schema" in uris
