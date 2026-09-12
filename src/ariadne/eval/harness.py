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

from ariadne.retrieval.hybrid import HybridSearch, SearchMode

GOLDEN = pathlib.Path(__file__).resolve().parents[3] / "data/golden/questions.json"


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


def load_questions(path: pathlib.Path | None = None) -> list[dict]:
    return json.loads((path or GOLDEN).read_text(encoding="utf-8"))


def evaluate_run(
    label: str,
    engine: HybridSearch,
    questions: list[dict],
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


def format_table(runs: list[RunResult]) -> str:
    """Tabela em markdown, pronta para colar no README."""
    linhas = [
        "| estrategia | recall geral | factual | relacional | sintese | MRR | seg |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        linhas.append(
            f"| {r.label} | {r.recall():.0%} | {r.recall('factual'):.0%} | "
            f"{r.recall('relacional'):.0%} | {r.recall('sintese'):.0%} | "
            f"{r.mrr():.3f} | {r.seconds:.0f} |"
        )
    return "\n".join(linhas)
