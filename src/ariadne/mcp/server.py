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

from ariadne.domain.graph import EntityType, RelationType, normalize_name
from ariadne.retrieval.vector_search import VectorSearch
from ariadne.storage.database import connection
from ariadne.storage.graph_store import AgeGraphStore
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


class NeighborOut(BaseModel):
    """Uma relacao da vizinhanca, com a frase que a sustenta."""

    name: str
    type: EntityType
    relation: RelationType
    direction: str
    evidence: str = Field(default="", description="Trecho do documento que justifica esta relacao")


class ExploreResponse(BaseModel):
    entity: str
    found: bool
    neighbors: list[NeighborOut]
    note: str | None = None


class PathStepOut(BaseModel):
    source: str
    relation: RelationType
    target: str
    evidence: str = ""


class ConnectionResponse(BaseModel):
    source: str
    target: str
    found: bool
    hops: int
    path: list[PathStepOut]
    note: str | None = None


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
        grafo = AgeGraphStore().stats(conn)

    return GraphStats(
        documents=stats["documents"],
        chunks=stats["chunks"],
        embedded_chunks=stats["embedded_chunks"],
        nodes=grafo["nodes"],
        edges=grafo["edges"],
        note=None
        if grafo["nodes"]
        else "O grafo esta vazio. Rode `ariadne graph-build` para popula-lo.",
    )


@server.tool(
    title="Explorar entidade",
    description=(
        "Mostra com quem uma entidade se relaciona no grafo de conhecimento. "
        "Cada relacao vem com o trecho do documento que a justifica."
    ),
)
def explore_entity(name: str, depth: int = 1) -> ExploreResponse:
    """Vizinhanca de uma entidade.

    Args:
        name: Nome da entidade (ex.: "Petrobras"). Aceita variacoes de grafia.
        depth: Quantos saltos percorrer, de 1 a 3.
    """
    depth = max(1, min(depth, 3))
    store = AgeGraphStore()
    with connection() as conn:
        chave = store.find_key(conn, name) or normalize_name(name)
        vizinhos = store.neighbors(conn, chave, depth=depth)

    nota = None
    if depth > 1:
        nota = (
            "Relacoes a mais de um salto vem sem evidencia por limitacao do "
            "Apache AGE. Use find_connection para ver a justificativa do caminho."
        )
    return ExploreResponse(
        entity=name,
        found=bool(vizinhos),
        note=nota if vizinhos else "Entidade nao encontrada no grafo.",
        neighbors=[
            NeighborOut(
                name=v.name,
                type=v.type,
                relation=v.relation,
                direction=v.direction,
                evidence=v.evidence,
            )
            for v in vizinhos
        ],
    )


@server.tool(
    title="Encontrar conexao",
    description=(
        "Encontra o caminho mais curto entre duas entidades no grafo, com o "
        "trecho de origem que justifica cada passo. Responde perguntas cuja "
        "resposta nao esta escrita em nenhum documento isolado."
    ),
)
def find_connection(entity_a: str, entity_b: str, max_hops: int = 4) -> ConnectionResponse:
    """Menor caminho entre duas entidades.

    Args:
        entity_a: Entidade de origem.
        entity_b: Entidade de destino.
        max_hops: Comprimento maximo do caminho, de 1 a 5.
    """
    max_hops = max(1, min(max_hops, 5))
    store = AgeGraphStore()
    with connection() as conn:
        a = store.find_key(conn, entity_a) or normalize_name(entity_a)
        b = store.find_key(conn, entity_b) or normalize_name(entity_b)
        caminho = store.shortest_path(conn, a, b, max_hops=max_hops)

    return ConnectionResponse(
        source=entity_a,
        target=entity_b,
        found=bool(caminho),
        hops=len(caminho),
        path=[
            PathStepOut(source=p.source, relation=p.relation, target=p.target, evidence=p.evidence)
            for p in caminho
        ],
        note=None
        if caminho
        else "Nenhum caminho encontrado. As entidades podem nao existir no grafo.",
    )


@server.resource(
    "ariadne://graph/schema",
    name="Esquema do grafo",
    description="Tipos de no e de aresta disponiveis no grafo de conhecimento.",
    mime_type="text/markdown",
)
def graph_schema() -> str:
    """Expoe o esquema para o LLM saber o que existe antes de perguntar.

    Gerado a partir dos enums, nao escrito a mao: um esquema documentado que
    diverge do codigo ensina o modelo a pedir tipo que nao existe.
    """
    quebra = chr(10)
    tipos_no = quebra.join(f"- `{t.value}`" for t in EntityType)
    tipos_aresta = quebra.join(f"- `{r.value}`" for r in RelationType)
    with connection() as conn:
        grafo = AgeGraphStore().stats(conn)
    return (
        f"# Esquema do grafo Ariadne{quebra}{quebra}"
        f"Estado: {grafo['nodes']} entidades, {grafo['edges']} relacoes."
        f"{quebra}{quebra}## Tipos de entidade{quebra}{quebra}"
        f"{tipos_no}{quebra}{quebra}"
        f"## Tipos de relacao{quebra}{quebra}"
        f"{tipos_aresta}{quebra}{quebra}"
        f"Toda relacao carrega o trecho do documento que a justifica.{quebra}"
    )


def main() -> None:
    """Entrada do servidor, falando stdio -- o transporte do Claude Desktop."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
