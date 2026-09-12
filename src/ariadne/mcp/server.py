"""Servidor MCP do Ariadne.

Por que o MCP entra agora, antes do grafo: ele e a interface real do produto.
Exercitar o sistema pelo mesmo caminho do usuario desde cedo revela problemas
de formato de resposta e de citacao enquanto ainda sao baratos de corrigir --
bem mais barato do que descobri-los na Fase 4, com tres camadas em cima.

REGRA DE OURO: nenhuma tool devolve trecho sem a fonte. Por isso a citacao e
campo obrigatorio do modelo de resposta, e nao algo que o chamador monta.
"""

from __future__ import annotations

from typing import Literal

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from ariadne.retrieval.vector_search import VectorSearch
from ariadne.storage.database import connection
from ariadne.storage.schema import apply_schema
from ariadne.storage.vector_store import PgVectorStore

SearchMode = Literal["vector", "graph", "hybrid"]

server = MCPServer(
    name="ariadne",
    instructions=(
        "Motor de conhecimento sobre um corpus de empresas brasileiras. "
        "Use search_knowledge para responder perguntas sobre esse corpus. "
        "Toda resposta traz a fonte: cite-a ao responder ao usuario."
    ),
)


class SearchHit(BaseModel):
    """Um trecho recuperado, sempre acompanhado da procedencia."""

    content: str
    citation: str = Field(description="Documento e secao de origem deste trecho")
    url: str | None = None
    score: float = Field(description="Similaridade de cosseno, de 0 a 1")


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]
    note: str | None = Field(
        default=None,
        description="Aviso sobre limitacoes do modo usado, quando houver",
    )


class GraphStats(BaseModel):
    documents: int
    chunks: int
    embedded_chunks: int
    nodes: int
    edges: int
    note: str | None = None


@server.tool(
    title="Buscar no conhecimento",
    description=(
        "Busca trechos relevantes no corpus indexado. Cada resultado vem com a "
        "citacao da fonte, que deve ser repassada ao usuario."
    ),
)
def search_knowledge(query: str, mode: SearchMode = "vector", limit: int = 5) -> SearchResponse:
    """Busca no corpus.

    Args:
        query: A pergunta ou termo a buscar.
        mode: Estrategia de busca. Hoje so `vector` esta implementado.
        limit: Quantidade maxima de trechos a devolver.
    """
    limit = max(1, min(limit, 20))

    note: str | None = None
    if mode in ("graph", "hybrid"):
        # Honestidade em vez de silencio: dizer que caiu no vetorial e melhor
        # do que fingir que a expansao pelo grafo aconteceu.
        note = (
            f"O modo '{mode}' ainda nao existe (chega na Fase 3). "
            "Esta resposta usou busca vetorial."
        )

    hits = VectorSearch().search(query, limit=limit)
    return SearchResponse(
        query=query,
        mode="vector",
        note=note,
        hits=[
            SearchHit(
                content=hit.chunk.content,
                citation=hit.citation(),
                url=hit.document_url,
                score=round(hit.score, 4),
            )
            for hit in hits
        ],
    )


@server.tool(
    title="Estatisticas do indice",
    description="Contagens do corpus indexado e do grafo de conhecimento.",
)
def graph_stats() -> GraphStats:
    """Quanto do corpus esta indexado e qual o tamanho do grafo."""
    with connection() as conn:
        apply_schema(conn)
        stats = PgVectorStore().stats(conn)

    return GraphStats(
        documents=stats["documents"],
        chunks=stats["chunks"],
        embedded_chunks=stats["embedded_chunks"],
        nodes=0,
        edges=0,
        note="O grafo ainda esta vazio: a extracao de entidades chega na Fase 2.",
    )


@server.resource(
    "ariadne://graph/schema",
    name="Esquema do grafo",
    description="Tipos de no e de aresta disponiveis no grafo de conhecimento.",
    mime_type="text/markdown",
)
def graph_schema() -> str:
    """Expoe o esquema para o LLM saber o que existe antes de perguntar."""
    return (
        "# Esquema do grafo Ariadne\n\n"
        "O grafo ainda nao foi populado: a extracao de entidades e relacoes\n"
        "chega na Fase 2. Por ora, use `search_knowledge` com `mode=vector`.\n\n"
        "## Previsto\n\n"
        "- No `Entity`: name, type\n"
        "- Aresta `RELATES_TO`: tipo da relacao e o chunk que a justifica\n"
    )


def main() -> None:
    """Entrada do servidor, falando stdio -- o transporte do Claude Desktop."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
