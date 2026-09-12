"""Geracao da resposta final, amarrada as fontes.

Este e o ponto do sistema em que a alucinacao pode entrar. Todo o resto --
chunking, embeddings, grafo, reranking -- so recupera texto que existe. Aqui um
LLM escreve prosa nova, e prosa nova pode afirmar o que quiser.

Tres travas, e nenhuma delas depende de o modelo "se comportar":

1. Os trechos entram NUMERADOS, e o prompt exige citar o numero. Pedir "cite
   as fontes" sem dar um identificador produz citacao inventada.
2. A resposta e validada DEPOIS: citacao a um numero que nao existe e
   removida, e uma resposta sem nenhuma citacao valida e rebaixada a aviso.
3. O objeto devolvido carrega os trechos usados. Quem recebe pode conferir,
   em vez de confiar.

A trava 2 e a que importa: instrucao em prompt e pedido, nao garantia.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel

from ariadne.config import Settings, get_settings
from ariadne.domain.models import RetrievedChunk

_PROMPT = """Responda a pergunta usando SOMENTE os trechos numerados abaixo.

Regras:
- Cite a fonte de cada afirmacao no formato [1], [2], etc.
- Use apenas os numeros que existem na lista.
- Se os trechos nao contiverem a resposta, diga isso claramente em vez de
  deduzir ou completar com conhecimento proprio.
- Responda em portugues, de forma direta.

TRECHOS:
{contexto}

PERGUNTA: {pergunta}
"""

_CITACAO = re.compile(r"\[(\d+)\]")


class Answer(BaseModel):
    """Resposta com a procedencia junto."""

    text: str
    sources: list[str]
    """Citacoes usadas, na ordem em que aparecem na resposta."""

    grounded: bool
    """False quando a resposta nao citou nenhuma fonte valida."""

    warning: str | None = None


@dataclass
class AnswerContext:
    chunks: list[RetrievedChunk] = field(default_factory=list)

    def render(self, max_chars: int = 1200) -> str:
        partes = []
        for i, hit in enumerate(self.chunks, 1):
            texto = hit.chunk.content.strip()[:max_chars]
            partes.append(f"[{i}] ({hit.citation()})\n{texto}")
        return "\n\n".join(partes)


class AnswerGenerator:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._model = cfg.extraction_model
        self._client = client or httpx.Client(
            base_url=cfg.ollama_base_url,
            timeout=httpx.Timeout(180.0),
        )

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        if not chunks:
            return Answer(
                text="Nao encontrei nada no corpus indexado sobre essa pergunta.",
                sources=[],
                grounded=False,
                warning="nenhum trecho recuperado",
            )

        contexto = AnswerContext(chunks).render()
        try:
            resposta = self._client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": _PROMPT.format(contexto=contexto, pergunta=question),
                        }
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 800},
                },
            )
            resposta.raise_for_status()
            texto = resposta.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, OSError) as exc:
            return Answer(
                text="Nao consegui gerar a resposta.",
                sources=[],
                grounded=False,
                warning=f"falha ao chamar o modelo: {exc}",
            )

        return self._validate(texto, chunks)

    def _validate(self, texto: str, chunks: list[RetrievedChunk]) -> Answer:
        """Confere as citacoes contra os trechos que foram realmente enviados.

        Um modelo pequeno cita [7] quando recebeu quatro trechos. Deixar passar
        e pior do que nao citar: a resposta ganha aparencia de verificada
        justamente onde nao esta.
        """
        validos = set(range(1, len(chunks) + 1))
        citados = [int(n) for n in _CITACAO.findall(texto)]
        fora = sorted({n for n in citados if n not in validos})

        limpo = texto
        for n in fora:
            limpo = limpo.replace(f"[{n}]", "")
        limpo = re.sub(r"\s{2,}", " ", limpo).strip()

        usados = sorted({n for n in citados if n in validos})
        fontes = [chunks[n - 1].citation() for n in usados]

        aviso = None
        if fora:
            aviso = (
                f"o modelo citou fonte(s) inexistente(s): "
                f"{', '.join(f'[{n}]' for n in fora)}; removidas da resposta"
            )
        if not fontes:
            aviso = (aviso + "; " if aviso else "") + (
                "a resposta nao citou nenhuma fonte -- trate-a como nao verificada"
            )

        return Answer(text=limpo, sources=fontes, grounded=bool(fontes), warning=aviso)

    def close(self) -> None:
        self._client.close()
