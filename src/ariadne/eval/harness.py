"""Avaliacao comparativa das estrategias de recuperacao.

POR QUE NAO UM JUIZ LLM AQUI
----------------------------
A pergunta que este projeto precisa responder nao e "a resposta esta boa?", e
sim "o GraphRAG recupera melhor que o RAG puro?". Para isso, um juiz de 7B
rodando local adiciona variancia sem adicionar informacao: ele erraria nos dois
lados e o delta ficaria enterrado no ruido.

A metrica aqui e objetiva e deterministica. Cada pergunta do golden set traz
ANCORAS -- termos que precisam aparecer no contexto recuperado para que a
pergunta seja respondivel. Medir se apareceram nao exige julgamento:

    context recall @k  =  ancoras encontradas / ancoras esperadas

Duas propriedades que um juiz LLM nao tem: roda em segundos e da o mesmo numero
toda vez. Qualquer um clona o repo e reproduz a tabela.

A comparacao e feita com acento e caixa normalizados, porque "Parana", "Paraná"
e "PARANA" sao a mesma resposta e penalizar isso mediria grafia, nao
recuperacao.
"""

from __future__ import annotations

import json
import pathlib
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol

from ariadne.retrieval.hybrid import HybridSearch, SearchMode

GOLDEN = pathlib.Path(__file__).resolve().parents[3] / "data/golden/questions.json"


class RouterLike(Protocol):
    """So o que o harness precisa de um roteador."""

    def route(self, query: str) -> Any: ...


def fold(texto: str) -> str:
    """Remove acento e caixa para comparar termos."""
    base = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in base if not unicodedata.combining(c)).lower()


@dataclass
class QuestionResult:
    id: str
    kind: str
    found: int
    expected: int
    rank: int | None
    """Posicao do primeiro trecho que contem alguma ancora (1-based)."""

    @property
    def recall(self) -> float:
        return self.found / self.expected if self.expected else 1.0

    @property
    def hit(self) -> bool:
        return self.found == self.expected


@dataclass
class RunResult:
    label: str
    results: list[QuestionResult] = field(default_factory=list)
    seconds: float = 0.0

    def recall(self, kind: str | None = None) -> float:
        alvo = [r for r in self.results if kind is None or r.kind == kind]
        return sum(r.recall for r in alvo) / len(alvo) if alvo else 0.0

    def hit_rate(self, kind: str | None = None) -> float:
        alvo = [r for r in self.results if kind is None or r.kind == kind]
        return sum(1 for r in alvo if r.hit) / len(alvo) if alvo else 0.0

    def mrr(self, kind: str | None = None) -> float:
        """Mean Reciprocal Rank: quao ALTO o trecho util aparece.

        Recall diz se a informacao foi recuperada; o MRR diz se ela ficou onde
        o modelo vai ler. Trazer a resposta em decimo lugar num contexto de
        cinco e o mesmo que nao trazer.
        """
        alvo = [r for r in self.results if kind is None or r.kind == kind]
        if not alvo:
            return 0.0
        return sum(1 / r.rank if r.rank else 0.0 for r in alvo) / len(alvo)


def load_questions(path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    casos: list[dict[str, Any]] = json.loads((path or GOLDEN).read_text(encoding="utf-8"))
    return casos


def evaluate_run(
    label: str,
    engine: HybridSearch,
    questions: list[dict[str, Any]],
    *,
    mode: SearchMode,
    limit: int = 5,
    rerank: bool = False,
) -> RunResult:
    """Roda o golden set contra uma configuracao e mede o recall de contexto."""
    inicio = time.monotonic()
    resultados: list[QuestionResult] = []

    for caso in questions:
        hits = engine.search(caso["question"], mode=mode, limit=limit, rerank=rerank).hits
        textos = [fold(h.chunk.content) for h in hits]
        ancoras = [fold(a) for a in caso["anchors"]]

        encontradas = sum(1 for a in ancoras if any(a in t for t in textos))
        posicao = next((i for i, t in enumerate(textos, 1) if any(a in t for a in ancoras)), None)
        resultados.append(
            QuestionResult(
                id=caso["id"],
                kind=caso["kind"],
                found=encontradas,
                expected=len(ancoras),
                rank=posicao,
            )
        )

    return RunResult(label=label, results=resultados, seconds=time.monotonic() - inicio)


def evaluate_routed(
    label: str,
    engines: dict[float, HybridSearch],
    router: RouterLike,
    questions: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> RunResult:
    """Avalia o sistema COMO ELE REALMENTE FUNCIONA: com roteamento.

    Medir um peso de grafo fixo para todas as perguntas mede uma configuracao
    que o sistema nunca usa. O roteador escolhe 0,2 para factual e 1,0 para
    relacional justamente porque expandir numa pergunta factual empurra o
    trecho certo para fora do topo -- foi o que a primeira medicao mostrou, e
    ela estava medindo a configuracao errada, nao encontrando um defeito.
    """
    inicio = time.monotonic()
    resultados: list[QuestionResult] = []

    for caso in questions:
        estrategia = router.route(caso["question"])
        engine = engines[estrategia.graph_weight]
        # O limite vem da ESTRATEGIA, nao do parametro. Passar um limite fixo
        # aqui media o roteador com metade dos resultados que ele pede: a
        # estrategia de sintese usa 10, e truncar em 5 derrubava o recall de
        # perguntas de lista em 19 pontos -- um defeito da medicao que parecia
        # defeito do roteamento.
        hits = engine.search(
            caso["question"],
            mode=estrategia.mode,
            limit=estrategia.limit,
            rerank=estrategia.rerank,
        ).hits
        textos = [fold(h.chunk.content) for h in hits]
        ancoras = [fold(a) for a in caso["anchors"]]
        resultados.append(
            QuestionResult(
                id=caso["id"],
                kind=caso["kind"],
                found=sum(1 for a in ancoras if any(a in t for t in textos)),
                expected=len(ancoras),
                rank=next(
                    (i for i, t in enumerate(textos, 1) if any(a in t for a in ancoras)),
                    None,
                ),
            )
        )

    return RunResult(label=label, results=resultados, seconds=time.monotonic() - inicio)


def format_table(runs: list[RunResult]) -> str:
    """Tabela em markdown, pronta para colar no README.

    As colunas por tipo saem dos DADOS, nao de uma lista fixa. A versao
    anterior tinha "factual/relacional/sintese" escritos no codigo, e quando o
    golden set passou a usar "agregacao" e "multihop" a tabela mostrou 0% em
    colunas que nao existiam -- escondendo justamente a comparacao que motivava
    a medicao.
    """
    tipos = sorted({r.kind for run in runs for r in run.results})
    cabecalho = ["estrategia", "recall geral", *tipos, "MRR", "seg"]
    linhas = [
        "| " + " | ".join(cabecalho) + " |",
        "|" + "---|" * len(cabecalho),
    ]
    for r in runs:
        celulas = [
            r.label,
            f"{r.recall():.0%}",
            *[f"{r.recall(t):.0%}" for t in tipos],
            f"{r.mrr():.3f}",
            f"{r.seconds:.0f}",
        ]
        linhas.append("| " + " | ".join(celulas) + " |")
    return "\n".join(linhas)
