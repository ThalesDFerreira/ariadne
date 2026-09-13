"""API HTTP local: o pedaco que faltava entre a pagina e o motor.

    uv run ariadne serve

POR QUE ISTO EXISTE
-------------------
A pagina de visualizacao (`docs/index.html`) e estatica: le um JSON e desenha.
Ela nao consegue perguntar nada, porque uma pagina aberta no navegador nao tem
caminho ate o Postgres nem ate o Ollama, que rodam na maquina. Faltava o meio
de campo -- alguem que receba a pergunta da pagina, chame o motor e devolva a
resposta. E este modulo.

Consequencia direta: a pagina de perguntar NAO funciona no GitHub Pages, porque
la nao ha banco nem modelo do outro lado. Por isso ela mora em `web/`, fora de
`docs/`, e a visualizacao do grafo continua publicavel sozinha.

SO LOCALHOST, POR PADRAO
------------------------
O bind padrao e 127.0.0.1. Este servidor nao tem autenticacao nenhuma, e quem
alcanca a porta le o corpus inteiro e gasta a GPU da maquina. Expor na rede
exige `--host` explicito, e a mensagem de aviso existe para que isso nunca
aconteca por distracao.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ariadne.agents.orchestrator import AgentResult, KnowledgeAgent
from ariadne.config import get_settings
from ariadne.storage.database import connection
from ariadne.storage.graph_store import AgeGraphStore
from ariadne.storage.schema import apply_schema
from ariadne.storage.vector_store import PgVectorStore

RAIZ_WEB = "web"
MAX_PERGUNTA = 500
"""Teto de caracteres. Pergunta gigante nao melhora a busca e vira um prompt
caro de graca -- o embedder trunca, o reranker nao."""

# Uma instancia so, criada na primeira pergunta. O agente carrega o
# cross-encoder de ~2 GB; construir um por requisicao recarregaria o modelo a
# cada pergunta. O lock serializa as chamadas: e um servidor de um usuario so,
# e disputar a GPU entre duas perguntas nao acelera nenhuma das duas.
_agente: KnowledgeAgent | None = None
_lock = threading.Lock()


def _obter_agente() -> KnowledgeAgent:
    global _agente
    if _agente is None:
        _agente = KnowledgeAgent()
    return _agente


def _responder(pergunta: str) -> dict[str, Any]:
    inicio = time.monotonic()
    with _lock:
        resultado: AgentResult = _obter_agente().ask(pergunta)
    return _serializar(resultado, time.monotonic() - inicio)


def _serializar(r: AgentResult, segundos: float) -> dict[str, Any]:
    return {
        "question": r.question,
        "answer": r.answer.text,
        "sources": r.answer.sources,
        "grounded": r.answer.grounded,
        "warning": r.answer.warning,
        "strategy": {
            "kind": r.strategy.kind.value,
            "reason": r.strategy.reason,
            "entities": list(r.strategy.entities),
            "rerank": r.strategy.rerank,
            "graph_weight": r.strategy.graph_weight,
        },
        "hits": [
            {
                "citation": h.citation(),
                "excerpt": h.chunk.content.strip()[:400],
                "score": round(h.score, 4),
            }
            for h in r.hits
        ],
        "path": [
            {
                "source": p.source,
                "relation": p.relation.value,
                "target": p.target,
                "evidence": p.evidence,
            }
            for p in r.path
        ],
        "seconds": round(segundos, 1),
    }


async def rota_perguntar(request: Request) -> JSONResponse:
    try:
        corpo = await request.json()
    except ValueError:
        return JSONResponse({"error": "corpo precisa ser JSON"}, status_code=400)

    pergunta = str(corpo.get("question", "")).strip()
    if not pergunta:
        return JSONResponse({"error": "pergunta vazia"}, status_code=400)
    if len(pergunta) > MAX_PERGUNTA:
        return JSONResponse(
            {"error": f"pergunta maior que {MAX_PERGUNTA} caracteres"}, status_code=400
        )

    # O motor e sincrono e demora dezenas de segundos com reranking. Fora do
    # threadpool ele travaria o loop de eventos e o servidor pararia de
    # responder qualquer coisa, inclusive o pedido de status.
    try:
        dados = await run_in_threadpool(_responder, pergunta)
    except Exception as exc:  # a mensagem vai para a tela, nao para um log que ninguem le
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
    return JSONResponse(dados)


async def rota_stats(_: Request) -> JSONResponse:
    def ler() -> dict[str, int]:
        with connection() as conn:
            apply_schema(conn)
            stats = PgVectorStore().stats(conn)
            stats.update(AgeGraphStore().stats(conn))
        return stats

    try:
        return JSONResponse(await run_in_threadpool(ler))
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=503)


async def rota_indice(_: Request) -> FileResponse:
    return FileResponse(f"{RAIZ_WEB}/index.html")


def build_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/", rota_indice),
            Route("/api/ask", rota_perguntar, methods=["POST"]),
            Route("/api/stats", rota_stats),
            # A visualizacao do grafo, servida do mesmo lugar por conveniencia.
            Mount("/grafo", StaticFiles(directory="docs", html=True)),
        ]
    )


def serve(host: str = "127.0.0.1", port: int = 18080) -> None:
    import uvicorn

    cfg = get_settings()
    if host not in ("127.0.0.1", "localhost"):
        print(f"AVISO: servindo em {host}. Nao ha autenticacao nenhuma nesta API.")
        print("Quem alcancar esta porta le o corpus inteiro e usa a GPU da maquina.")

    print(f"Ariadne em http://{host}:{port}")
    print(f"  grafo:   http://{host}:{port}/grafo/")
    print(f"  usando:  Postgres {cfg.pg_host}:{cfg.pg_port} e Ollama em {cfg.ollama_base_url}")
    print("A primeira pergunta carrega o cross-encoder e demora mais que as seguintes.")
    uvicorn.run(build_app(), host=host, port=port, log_level="warning")
