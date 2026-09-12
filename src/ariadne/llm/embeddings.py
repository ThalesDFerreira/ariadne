"""Geracao de embeddings.

O Protocol abaixo e a fronteira que cumpre a regra "roda local e com API,
trocando so a configuracao": o resto do projeto nunca importa o Ollama, so o
Protocol. Trocar de provedor vira escrever outra classe, nao mexer no
pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from ariadne.config import Settings, get_settings

Vector = list[float]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contrato minimo de quem sabe transformar texto em vetor."""

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class OllamaEmbedder:
    """Embeddings via Ollama local.

    Documento e consulta tem metodos separados porque varios modelos pedem
    prefixos diferentes para cada lado ("query: ..." vs "passage: ..."). O
    BGE-M3 nao exige, mas manter os dois caminhos desde ja evita ter que
    reindexar o corpus inteiro no dia em que trocarmos de modelo.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        batch_size: int = 16,
    ) -> None:
        cfg = settings or get_settings()
        self._model = cfg.embedding_model_ollama
        self._dimensions = cfg.embedding_dim
        self._batch_size = batch_size
        self._client = client or httpx.Client(
            base_url=cfg.ollama_base_url,
            # Generoso de proposito: a primeira chamada carrega o modelo na
            # GPU e pode levar dezenas de segundos.
            timeout=httpx.Timeout(180.0),
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        out: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            out.extend(self._embed(texts[start : start + self._batch_size]))
        return out

    def embed_query(self, text: str) -> Vector:
        return self._embed([text])[0]

    def _embed(self, batch: list[str]) -> list[Vector]:
        response = self._client.post("/api/embed", json={"model": self._model, "input": batch})
        response.raise_for_status()
        vectors: list[Vector] = response.json()["embeddings"]

        if len(vectors) != len(batch):
            msg = f"esperava {len(batch)} vetores, o Ollama devolveu {len(vectors)}"
            raise RuntimeError(msg)
        for vector in vectors:
            if len(vector) != self._dimensions:
                # Falhar aqui e barato; gravar vetor de tamanho errado no
                # pgvector quebra com uma mensagem muito pior, e depois de ja
                # ter processado meio corpus.
                msg = (
                    f"modelo {self._model} devolveu {len(vector)} dimensoes, "
                    f"mas o schema espera {self._dimensions}"
                )
                raise RuntimeError(msg)
        return vectors

    def close(self) -> None:
        self._client.close()


def build_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    """Escolhe o provedor conforme a configuracao."""
    cfg = settings or get_settings()
    # Hoje so ha o caminho local; a assinatura ja acomoda os outros.
    return OllamaEmbedder(cfg)
