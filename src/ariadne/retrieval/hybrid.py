"""Motor de recuperacao hibrido.

O caminho completo de uma pergunta:

    vetorial  ---\\
    lexical   ----+--> RRF --> expansao k-hop --> RRF --> reranker --> topo
    (grafo)   ---/

Duas fusoes, e nao uma, por um motivo: a expansao pelo grafo parte dos chunks
JA recuperados, entao ela precisa da primeira fusao pronta para saber de onde
partir. Expandir a partir de um so dos rankings herdaria o erro dele.

Os pesos refletem confianca, nao preferencia: vetorial e lexical entram com
peso cheio porque respondem a pergunta diretamente; a expansao entra com peso
menor porque responde "isto e vizinho do que voce perguntou", que e util e
mais ruidoso. O reranker, que le pergunta e trecho juntos, da a palavra final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ariadne.domain.models import RetrievedChunk
from ariadne.llm.embeddings import EmbeddingProvider, build_embedder
from ariadne.retrieval.fusion import RankedList, reciprocal_rank_fusion
from ariadne.retrieval.graph_expansion import GraphExpansion
from ariadne.retrieval.lexical import LexicalSearch
from ariadne.retrieval.reranking import Reranker, build_reranker
from ariadne.storage.database import connection
from ariadne.storage.vector_store import PgVectorStore


class SearchMode(StrEnum):
    VECTOR = "vector"
    LEXICAL = "lexical"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class SearchTrace:
    """Quantos candidatos cada etapa produziu.

    Serve para explicar a resposta e para depurar: quando o resultado vem
    ruim, a primeira pergunta e sempre "qual etapa nao trouxe nada?".
    """

    vector: int = 0
    lexical: int = 0
    graph: int = 0
    fused: int = 0
    reranked: int = 0
    entities: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"vetorial={self.vector} lexical={self.lexical} grafo={self.graph} "
            f"fundidos={self.fused} rerankeados={self.reranked}"
        )


@dataclass
class SearchResult:
    hits: list[RetrievedChunk]
    mode: SearchMode
    trace: SearchTrace


class HybridSearch:
    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        *,
        graph_name: str | None = None,
        candidate_pool: int = 20,
        vector_weight: float = 1.0,
        lexical_weight: float = 1.0,
        graph_weight: float = 0.5,
    ) -> None:
        self._embedder = embedder or build_embedder()
        self._reranker = reranker or build_reranker()
        self._store = PgVectorStore()
        self._lexical = LexicalSearch()
        self._expansion = GraphExpansion(graph_name)
        self._pool = candidate_pool
        self._w_vector = vector_weight
        self._w_lexical = lexical_weight
        self._w_graph = graph_weight

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 5,
        rerank: bool = True,
    ) -> SearchResult:
        trace = SearchTrace()

        with connection() as conn:
            vetorial: list[RetrievedChunk] = []
            lexical: list[RetrievedChunk] = []

            if mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.GRAPH):
                embedding = self._embedder.embed_query(query)
                vetorial = self._store.search(conn, embedding, limit=self._pool)
                trace.vector = len(vetorial)

            if mode in (SearchMode.LEXICAL, SearchMode.HYBRID):
                lexical = self._lexical.search(conn, query, limit=self._pool)
                trace.lexical = len(lexical)

            if mode is SearchMode.VECTOR:
                return self._finish(query, vetorial, mode, trace, limit, rerank)
            if mode is SearchMode.LEXICAL:
                return self._finish(query, lexical, mode, trace, limit, rerank)

            base = reciprocal_rank_fusion(
                [
                    RankedList("vetorial", vetorial, self._w_vector),
                    RankedList("lexical", lexical, self._w_lexical),
                ],
                limit=self._pool,
            )

            # A expansao parte da fusao, nao de um ranking so: partir de um
            # deles herdaria o erro dele na escolha das sementes.
            sementes = self._expansion.select_seed_entities(conn, base[:5])
            expandido = self._expansion.expand(conn, base[:5], limit=self._pool)
            trace.graph = len(expandido)
            # As entidades que REALMENTE guiaram a expansao, nao todas as que
            # aparecem nos chunks: mostrar as outras daria uma explicacao que
            # nao corresponde ao que o sistema fez.
            trace.entities = sementes[:10]

            fundido = reciprocal_rank_fusion(
                [
                    RankedList("base", base, 1.0),
                    RankedList("grafo", expandido, self._w_graph),
                ],
                limit=self._pool,
            )
            trace.fused = len(fundido)

        return self._finish(query, fundido, mode, trace, limit, rerank)

    def _finish(
        self,
        query: str,
        candidatos: list[RetrievedChunk],
        mode: SearchMode,
        trace: SearchTrace,
        limit: int,
        rerank: bool,
    ) -> SearchResult:
        if not trace.fused:
            trace.fused = len(candidatos)
        hits = (
            self._reranker.rerank(query, candidatos, limit=limit) if rerank else candidatos[:limit]
        )
        trace.reranked = len(hits)
        return SearchResult(hits=hits, mode=mode, trace=trace)
