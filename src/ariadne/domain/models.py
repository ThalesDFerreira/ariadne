"""Modelos de dominio.

Pydantic puro: nenhuma referencia a banco, HTTP ou LLM. Se um modelo daqui
precisar saber de infraestrutura, a modelagem esta errada.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Um documento ingerido, na integra."""

    id: UUID = Field(default_factory=uuid4)
    source: str
    """De onde veio (ex.: "wikipedia-pt"). Permite reingerir uma fonte so."""

    external_id: str
    """Identificador estavel na origem (ex.: o titulo da pagina).

    Junto com `source`, forma a chave natural: reingerir o mesmo documento
    atualiza em vez de duplicar.
    """

    title: str
    url: str | None = None
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def content_hash(self) -> str:
        """Hash do conteudo, para pular reprocessamento.

        A extracao de entidades custa uma chamada de LLM por chunk -- com 8 GB
        de VRAM, reprocessar o corpus inteiro a toa custa horas. Este hash e o
        que permite reingerir so o que mudou.
        """
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class Chunk(BaseModel):
    """Um pedaco recuperavel de um documento.

    E a unidade da busca: o que o embedding representa e o que a citacao
    aponta. Por isso carrega contexto suficiente para se explicar sozinho.
    """

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    ordinal: int = Field(ge=0)
    """Posicao no documento. Preserva a ordem de leitura na hora de citar."""

    content: str
    section_path: str = ""
    """Trilha de secoes, ex.: "Historia > Fundacao".

    Guardada separada do texto para servir a citacao e, mais tarde, a extracao
    de entidades: saber que um trecho veio de "Controladas" muda o que o LLM
    deve procurar nele.
    """

    @property
    def char_count(self) -> int:
        return len(self.content)


class RetrievedChunk(BaseModel):
    """Um chunk devolvido pela busca, com a procedencia junto.

    Regra de ouro do projeto: nada sai do motor de recuperacao sem a fonte.
    Por isso titulo e url vem no mesmo objeto, e nao numa consulta posterior
    que alguem pode esquecer de fazer.
    """

    chunk: Chunk
    document_title: str
    document_url: str | None
    score: float

    def citation(self) -> str:
        """Citacao pronta para o usuario final."""
        where = f"{self.document_title}"
        if self.chunk.section_path:
            where += f" > {self.chunk.section_path}"
        if self.document_url:
            where += f" ({self.document_url})"
        return where
