"""Modelos do grafo de conhecimento.

O esquema e FECHADO de proposito. Deixar o LLM inventar o rotulo do tipo
produz "Empresa", "Companhia", "Organizacao" e "Organization" como quatro
tipos distintos que nunca se encontram numa consulta -- o grafo fica com
aparencia de rico e utilidade zero. Enum limitado e o que torna o Cypher
previsivel.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class EntityType(StrEnum):
    ORGANIZACAO = "Organizacao"
    PESSOA = "Pessoa"
    LUGAR = "Lugar"
    PRODUTO = "Produto"
    SETOR = "Setor"
    EVENTO = "Evento"
    OUTRO = "Outro"


class RelationType(StrEnum):
    """Relacoes que interessam a perguntas multi-hop.

    Enxuto por escolha: cada tipo novo e uma chance a mais de o modelo escolher
    errado entre dois rotulos parecidos. Melhor poucos tipos bem aplicados do
    que uma taxonomia rica e mal preenchida.
    """

    CONTROLA = "CONTROLA"
    """A controla B (participacao acionaria, subsidiaria)."""

    ADQUIRIU = "ADQUIRIU"
    FORNECE_PARA = "FORNECE_PARA"
    CONCORRE_COM = "CONCORRE_COM"
    ATUA_EM = "ATUA_EM"
    """Setor, mercado ou regiao geografica."""

    PRODUZ = "PRODUZ"
    LOCALIZADA_EM = "LOCALIZADA_EM"
    TRABALHA_EM = "TRABALHA_EM"
    PARTICIPOU_DE = "PARTICIPOU_DE"
    RELACIONADA_A = "RELACIONADA_A"
    """Escape para relacao real que nao cabe nos tipos acima."""


def normalize_name(name: str) -> str:
    """Chave de comparacao para deduplicar entidades.

    Remove acento, caixa, pontuacao e os sufixos societarios que fazem
    "Petrobras", "PETROBRAS S.A." e "Petrobrás S/A" parecerem tres empresas.
    E o primeiro e mais barato estagio da resolucao de entidades.
    """
    texto = unicodedata.normalize("NFKD", name)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower().strip()
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(
        # So sufixo societario. "do Brasil" NAO entra: faz parte do nome real em
        # "Banco do Brasil", e removê-lo reduz a entidade a "banco", colidindo
        # com qualquer outro banco do corpus.
        r"\b(s\s*a|sa|ltda|s\s*/\s*a|inc|corp|corporation|company|cia)\b",
        " ",
        texto,
    )
    return re.sub(r"\s+", " ", texto).strip()


class ExtractedEntity(BaseModel):
    """Entidade como o LLM a devolveu, antes da resolucao."""

    name: str = Field(description="Nome da entidade exatamente como aparece no texto")
    type: EntityType = Field(description="Categoria da entidade")

    @field_validator("name")
    @classmethod
    def _limpa(cls, valor: str) -> str:
        limpo = valor.strip()
        if not limpo:
            msg = "nome de entidade vazio"
            raise ValueError(msg)
        return limpo

    @property
    def key(self) -> str:
        return normalize_name(self.name)


class ExtractedRelation(BaseModel):
    """Relacao dirigida entre duas entidades do mesmo chunk."""

    source: str = Field(description="Nome da entidade de origem")
    target: str = Field(description="Nome da entidade de destino")
    type: RelationType = Field(description="Tipo da relacao")
    evidence: str = Field(
        default="",
        description="Trecho curto do texto que sustenta esta relacao",
    )


class Extraction(BaseModel):
    """O que o LLM devolve para um chunk.

    Os dois campos sao OBRIGATORIOS (sem default) de proposito. Com
    `default_factory=list` eles ficam de fora de `required` no JSON Schema, e
    o structured output so garante o que esta em `required` -- na pratica os
    modelos passavam a devolver `{"relations": [...]}` sem `entities`, o que
    fazia drop_dangling_relations descartar tudo e a extracao sair vazia em
    silencio. Campo opcional em schema de structured output e campo que o
    modelo aprende a omitir.

    A ordem tambem importa: `entities` vem primeiro porque a geracao e
    autoregressiva, e listar as entidades antes ajuda o modelo a escolher
    origem e destino coerentes nas relacoes.
    """

    entities: list[ExtractedEntity] = Field(description="Todas as entidades citadas no texto")
    relations: list[ExtractedRelation] = Field(
        description="Relacoes afirmadas pelo texto; lista vazia se nao houver"
    )

    def drop_dangling_relations(self) -> Extraction:
        """Descarta relacao cujas pontas nao foram extraidas como entidade.

        O modelo as vezes cita nome que nao declarou na lista de entidades.
        Gravar assim criaria no orfao sem tipo, que suja consulta e contagem
        para sempre. Melhor perder a aresta do que corromper o grafo.
        """
        conhecidas = {e.key for e in self.entities}
        validas = [
            r
            for r in self.relations
            if normalize_name(r.source) in conhecidas
            and normalize_name(r.target) in conhecidas
            and normalize_name(r.source) != normalize_name(r.target)
        ]
        return Extraction(entities=self.entities, relations=validas)


class GraphNode(BaseModel):
    """Entidade ja resolvida, pronta para virar vertice."""

    id: UUID = Field(default_factory=uuid4)
    key: str
    name: str
    """Rotulo canonico escolhido entre as variantes vistas."""

    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    mentions: int = 0


class GraphEdge(BaseModel):
    """Aresta com a procedencia junto.

    chunk_id nao e opcional: e o que permite responder "por que voce afirma
    isso?" mostrando o trecho de origem. Aresta sem evidencia e afirmacao sem
    fonte, e o projeto inteiro se recusa a fazer isso.
    """

    source_key: str
    target_key: str
    type: RelationType
    chunk_id: UUID
    evidence: str = ""
