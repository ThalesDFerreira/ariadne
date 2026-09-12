"""Orquestrador: da pergunta a resposta citada.

    pergunta -> roteador -> busca (parametros da estrategia) -> resposta citada

Perguntas relacionais ganham um passo a mais: alem dos trechos, o caminho no
grafo entre as entidades citadas entra no contexto. E o unico jeito de
responder "qual a ligacao entre A e B" quando nenhum documento contem a
ligacao -- ela so existe como sequencia de arestas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ariadne.agents.answering import Answer, AnswerGenerator
from ariadne.agents.router import QueryKind, QueryRouter, Strategy
from ariadne.domain.graph import normalize_name
from ariadne.domain.models import RetrievedChunk
from ariadne.llm.embeddings import build_embedder
from ariadne.retrieval.hybrid import HybridSearch
from ariadne.retrieval.reranking import build_reranker
from ariadne.storage.database import connection
from ariadne.storage.graph_store import AgeGraphStore, PathStep


@dataclass
class AgentResult:
    question: str
    answer: Answer
    strategy: Strategy
    hits: list[RetrievedChunk] = field(default_factory=list)
    path: list[PathStep] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.strategy.kind.value}] {len(self.hits)} trecho(s), "
            f"{len(self.path)} passo(s) de grafo, "
            f"{'com' if self.answer.grounded else 'SEM'} citacao"
        )


class KnowledgeAgent:
    def __init__(
        self,
        router: QueryRouter | None = None,
        generator: AnswerGenerator | None = None,
        *,
        graph_name: str | None = None,
    ) -> None:
        self._router = router or QueryRouter()
        self._generator = generator or AnswerGenerator()
        self._graph = AgeGraphStore(graph_name)
        self._graph_name = graph_name
        # Um motor por peso de grafo, porque o peso e fixado no construtor.
        self._engines: dict[float, HybridSearch] = {}
        # O reranker e COMPARTILHADO entre eles. Sem isso cada estrategia
        # construia o proprio, e o cross-encoder de ~2 GB recarregava toda vez
        # que a pergunta mudava de tipo -- visivel no log como um "Loading
        # weights" por pergunta.
        self._embedder = build_embedder()
        self._reranker = build_reranker()

    def _engine(self, strategy: Strategy) -> HybridSearch:
        chave = strategy.graph_weight
        if chave not in self._engines:
            self._engines[chave] = HybridSearch(
                embedder=self._embedder,
                reranker=self._reranker,
                graph_name=self._graph_name,
                candidate_pool=strategy.candidate_pool,
                graph_weight=strategy.graph_weight,
            )
        return self._engines[chave]

    def ask(self, question: str) -> AgentResult:
        estrategia = self._router.route(question)

        resultado = self._engine(estrategia).search(
            question,
            mode=estrategia.mode,
            limit=estrategia.limit,
            rerank=estrategia.rerank,
        )
        hits = resultado.hits

        caminho: list[PathStep] = []
        if estrategia.kind is QueryKind.RELACIONAL and len(estrategia.entities) >= 2:
            caminho = self._connect(estrategia.entities)

        contexto = self._with_path(hits, caminho)
        resposta = self._generator.answer(question, contexto)

        return AgentResult(
            question=question,
            answer=resposta,
            strategy=estrategia,
            hits=contexto,
            path=caminho,
        )

    def _connect(self, entidades: tuple[str, ...]) -> list[PathStep]:
        """Caminho entre as duas primeiras entidades citadas na pergunta."""
        with connection() as conn:
            a = self._graph.find_key(conn, entidades[0]) or normalize_name(entidades[0])
            b = self._graph.find_key(conn, entidades[1]) or normalize_name(entidades[1])
            return self._graph.shortest_path(conn, a, b)

    def _with_path(
        self, hits: list[RetrievedChunk], caminho: list[PathStep]
    ) -> list[RetrievedChunk]:
        """Transforma o caminho do grafo em trechos citaveis.

        Cada aresta ja carrega a frase que a sustenta, entao o caminho entra no
        contexto como qualquer outro trecho -- e o modelo pode cita-lo pelo
        numero, como as demais fontes. Sem isso, a ligacao encontrada no grafo
        chegaria ao usuario sem procedencia, que e exatamente o que o projeto
        se recusa a fazer.
        """
        if not caminho:
            return hits

        extras: list[RetrievedChunk] = []
        for passo in caminho:
            if not passo.evidence:
                continue
            base = hits[0] if hits else None
            if base is None:
                continue
            extras.append(
                RetrievedChunk(
                    chunk=base.chunk.model_copy(
                        update={
                            "content": (
                                f"{passo.source} —[{passo.relation.value}]— "
                                f"{passo.target}: {passo.evidence}"
                            )
                        }
                    ),
                    document_title=f"grafo: {passo.source} ~ {passo.target}",
                    document_url=None,
                    score=1.0,
                )
            )
        return extras + hits

    def close(self) -> None:
        self._router.close()
        self._generator.close()
