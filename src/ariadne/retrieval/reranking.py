"""Reranking com cross-encoder.

A diferenca que justifica o custo: o embedding e um BI-encoder. Ele transforma
a pergunta num vetor e o trecho em outro vetor, SEPARADAMENTE, e compara os
dois. O trecho foi vetorizado na ingestao, sem nunca ter visto a pergunta --
ele precisa resumir num unico vetor tudo o que poderia ser perguntado sobre
ele.

O cross-encoder le a pergunta E o trecho JUNTOS, de uma vez, e responde uma so
coisa: este trecho responde esta pergunta? Por isso acerta onde o bi-encoder
confunde -- e o caso "Itabira x Itabirito" da Fase 1, em que os vetores sao
vizinhos mas a resposta certa e outra.

O preco: nao da para pre-computar nada. Cada par (pergunta, trecho) e uma
inferencia. Por isso ele nao BUSCA, so reordena algumas dezenas de candidatos
que a busca ja trouxe.

Roda em CPU de proposito: sao poucas dezenas de pares, o que leva cerca de um
segundo, e deixa a GPU livre para o LLM de extracao -- que e quem realmente
precisa dela nesta maquina de 8 GB.
"""

from __future__ import annotations

from typing import Protocol

from ariadne.domain.models import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, limit: int | None = None
    ) -> list[RetrievedChunk]: ...


class NoopReranker:
    """Mantem a ordem. Usado quando o modelo nao esta instalado.

    Devolver a ordem original e melhor do que falhar: o reranking e uma camada
    de precisao, nao um requisito para o sistema responder.
    """

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, limit: int | None = None
    ) -> list[RetrievedChunk]:
        return candidates[:limit] if limit else candidates


class CrossEncoderReranker:
    """Cross-encoder local (BGE reranker), em CPU.

    O modelo e carregado na PRIMEIRA chamada, nao no construtor: sao ~2 GB de
    download na estreia, e a CLI nao pode travar por isso em comandos que nem
    usam reranking.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, limit: int | None = None
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        modelo = self._ensure_model()
        pares = [(query, c.chunk.content) for c in candidates]
        scores = modelo.predict(pares, batch_size=self._batch_size)  # type: ignore[attr-defined]

        reordenado = sorted(
            zip(candidates, scores, strict=True),
            key=lambda par: (-float(par[1]), str(par[0].chunk.id)),
        )
        saida = [
            RetrievedChunk(
                chunk=hit.chunk,
                document_title=hit.document_title,
                document_url=hit.document_url,
                # Score do cross-encoder: escala propria, nao comparavel com a
                # similaridade de cosseno nem com o valor do RRF.
                score=float(score),
            )
            for hit, score in reordenado
        ]
        return saida[:limit] if limit else saida


def build_reranker(enabled: bool = True) -> Reranker:
    """Devolve o cross-encoder, ou um no-op quando ele nao esta disponivel."""
    if not enabled:
        return NoopReranker()
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return NoopReranker()
    return CrossEncoderReranker()
