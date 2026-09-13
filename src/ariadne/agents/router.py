"""Roteador de consulta: escolher a estrategia antes de buscar.

Tres tipos de pergunta pedem coisas diferentes do motor:

- FACTUAL   "Quando a Vale foi privatizada?"
            A resposta esta escrita num trecho. Poucos candidatos, reranking
            forte. Expandir pelo grafo so traz ruido.

- RELACIONAL "Qual a ligacao entre o Bradesco e a Previ?"
            A resposta nao esta em trecho nenhum: precisa percorrer o grafo.
            Vale a pena pagar a expansao com peso alto.

- SINTESE   "Quais empresas foram privatizadas com apoio do BNDES?"
            A resposta e uma LISTA montada de varios trechos. Precisa de
            recall: muitos candidatos, corte generoso.

Usar a mesma configuracao para as tres desperdicia esforco nas faceis e falha
nas dificeis. E por isso que existe uma camada decidindo antes.

O roteamento usa o LLM local com schema fechado, e cai numa heuristica quando
o modelo nao responde -- classificar errado degrada a resposta, mas nao poder
classificar nao pode derrubar o sistema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import httpx
from pydantic import BaseModel, Field, ValidationError

from ariadne.config import Settings, get_settings
from ariadne.retrieval.hybrid import SearchMode


class QueryKind(StrEnum):
    FACTUAL = "factual"
    RELACIONAL = "relacional"
    SINTESE = "sintese"


class Routing(BaseModel):
    """O que o LLM devolve. Campos obrigatorios de proposito.

    Mesma licao da extracao: campo com default fica fora de `required` no JSON
    Schema, e o modelo aprende a omiti-lo.
    """

    kind: QueryKind = Field(description="Tipo da pergunta")
    reason: str = Field(description="Uma frase curta justificando a escolha")
    entities: list[str] = Field(
        description="Entidades citadas na pergunta; lista vazia se nao houver"
    )


@dataclass(frozen=True)
class Strategy:
    """Parametros de busca para um tipo de pergunta."""

    kind: QueryKind
    mode: SearchMode
    limit: int
    candidate_pool: int
    graph_weight: float
    rerank: bool
    reason: str = ""
    entities: tuple[str, ...] = ()


STRATEGIES: dict[QueryKind, Strategy] = {
    QueryKind.FACTUAL: Strategy(
        kind=QueryKind.FACTUAL,
        mode=SearchMode.HYBRID,
        limit=4,
        candidate_pool=20,
        # Peso baixo: a resposta esta num trecho, e o vizinho no grafo so
        # empurra para baixo o trecho certo.
        graph_weight=0.2,
        rerank=True,
    ),
    QueryKind.RELACIONAL: Strategy(
        kind=QueryKind.RELACIONAL,
        mode=SearchMode.HYBRID,
        limit=6,
        candidate_pool=30,
        # Aqui o grafo e o ponto: sem ele a pergunta nao tem resposta.
        graph_weight=1.0,
        rerank=True,
    ),
    QueryKind.SINTESE: Strategy(
        kind=QueryKind.SINTESE,
        mode=SearchMode.HYBRID,
        limit=10,
        candidate_pool=40,
        graph_weight=0.6,
        # Sem reranking: o cross-encoder ordena por "melhor resposta unica", e
        # numa pergunta de lista isso enterra itens validos que nao sao o
        # melhor exemplo. Recall importa mais que a ordem.
        rerank=False,
    ),
}

_PROMPT = """Classifique a pergunta do usuario em UM tipo.

- factual: a resposta e um fato que deve estar escrito em algum trecho.
  Ex: "Quando a Vale foi privatizada?", "Onde fica a sede da Petrobras?"
- relacional: a pergunta liga DUAS OU MAIS entidades, ou pede caminho,
  influencia, dependencia ou participacao entre elas.
  Ex: "Qual a ligacao entre o Bradesco e a Previ?",
      "Quais fornecedores dependem da mesma materia-prima que a X?"
- sintese: a resposta e uma LISTA ou um panorama montado de varias fontes.
  Ex: "Quais empresas foram privatizadas?", "Resuma o setor eletrico."

Em `entities`, liste os nomes proprios citados na pergunta.

PERGUNTA: {query}
"""

# Sinais textuais para o caminho de emergencia.
_RELACIONAL = re.compile(
    r"\b(liga[cç][aã]o|rela[cç][aã]o|conex[aã]o|entre|v[ií]nculo|depend[eê]|"
    r"fornece|controla|adquiriu|mesma|mesmo)\b",
    re.IGNORECASE,
)

# Perguntas que DESCREVEM a ponte em vez de nomea-la. Medido: 3 de 4 perguntas
# multi-hop caiam em "factual" -- que usa peso de grafo 0,2 -- porque nenhuma
# delas contem "ligacao" ou "entre". "As concessoes ENVOLVIDAS NA OPERACAO da
# PetroReconcavo" e exatamente esse caso: a ponte esta na oracao subordinada.
#
# A segunda alternativa exige que o substantivo da operacao esteja LIGADO a
# outra coisa ("operacao DA PetroReconcavo", "incorporacao ENVOLVENDO o Banco").
# A primeira versao pedia so uma preposicao antes dele, e com isso mandava
# "Qual o valor da operacao?" -- pergunta factual, sem ponte nenhuma -- para a
# rota relacional, que custa peso de grafo 1,0 e pool 30. Foi um dos 4 falsos
# positivos que a matriz de classificacao acusou.
#
# O elo aceito e possessivo (da/do/das/dos), nunca o "de" solto: em
# "incorporacao DE acoes" o que vem depois e o objeto da operacao, nao uma
# contraparte -- e essa versao intermediaria trocou um falso positivo por
# outro antes de a matriz mostrar qual pergunta tinha mudado de lado.
_PONTE = re.compile(
    r"\b(envolvid[ao]s?|envolvendo|participante|resultante|decorrente|referid[ao]s?)\b"
    r"|\b(opera[cç][aã]o|transa[cç][aã]o|aquisi[cç][aã]o|incorpora[cç][aã]o|"
    r"fus[aã]o|neg[oó]cio|contrato)\s+(d[ao]s?|envolvendo|entre|com)\b"
    r"|\bque\s+(a|o|foi|foram|fez|fizeram|comprou|vendeu|adquiriu)\b",
    re.IGNORECASE,
)
_SINTESE = re.compile(
    r"\b(quais|liste|lista|resuma|resumo|panorama|todos|todas|principais)\b",
    re.IGNORECASE,
)


class QueryRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        use_llm: bool = True,
    ) -> None:
        cfg = settings or get_settings()
        self._model = cfg.extraction_model
        self._use_llm = use_llm
        self._client = client or httpx.Client(
            base_url=cfg.ollama_base_url,
            # Curto de proposito: o roteamento e um passo auxiliar, e nao pode
            # dominar a latencia da resposta. Estourou, usa a heuristica.
            timeout=httpx.Timeout(20.0),
        )

    def route(self, query: str) -> Strategy:
        roteamento = self._classify_llm(query) if self._use_llm else None
        if roteamento is None:
            roteamento = self._classify_heuristic(query)

        base = STRATEGIES[roteamento.kind]
        return Strategy(
            kind=base.kind,
            mode=base.mode,
            limit=base.limit,
            candidate_pool=base.candidate_pool,
            graph_weight=base.graph_weight,
            rerank=base.rerank,
            reason=roteamento.reason,
            entities=tuple(roteamento.entities),
        )

    def _classify_llm(self, query: str) -> Routing | None:
        try:
            resposta = self._client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": _PROMPT.format(query=query)}],
                    "stream": False,
                    "format": Routing.model_json_schema(),
                    "options": {"temperature": 0, "num_predict": 256},
                },
            )
            resposta.raise_for_status()
            return Routing.model_validate_json(resposta.json()["message"]["content"])
        except (httpx.HTTPError, ValidationError, KeyError, OSError):
            return None

    def _classify_heuristic(self, query: str) -> Routing:
        """Caminho de emergencia, sem LLM.

        A ordem dos testes importa: "Quais empresas dependem da mesma
        materia-prima que a X?" casa com os dois padroes, e e relacional --
        a parte dificil da pergunta e a ligacao, nao a listagem.

        `_PONTE` vem antes de sintese pelo mesmo motivo: uma pergunta que
        descreve a ponte ("as concessoes envolvidas na operacao da X") precisa
        do grafo mesmo sem citar "ligacao" ou "entre".
        """
        if _RELACIONAL.search(query) or _PONTE.search(query):
            kind = QueryKind.RELACIONAL
        elif _SINTESE.search(query):
            kind = QueryKind.SINTESE
        else:
            kind = QueryKind.FACTUAL
        return Routing(
            kind=kind,
            reason="classificado por heuristica (LLM indisponivel)",
            entities=_nomes_proprios(query),
        )

    def close(self) -> None:
        self._client.close()


def _nomes_proprios(texto: str) -> list[str]:
    """Palavras capitalizadas que nao iniciam a frase."""
    tokens = texto.split()
    return [
        t.strip(".,?!;:") for i, t in enumerate(tokens) if i > 0 and t[:1].isupper() and len(t) > 2
    ]
