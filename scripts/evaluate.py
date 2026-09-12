"""Compara RAG puro com GraphRAG no golden set.

    uv run python scripts/evaluate.py

Produz a tabela que vai para o README. Cada linha e uma configuracao do motor,
e a diferenca entre elas e a unica coisa que interessa: se a expansao pelo
grafo nao mover o numero, ela nao esta se pagando.
"""

from __future__ import annotations

import sys

from ariadne.console import force_utf8_output
from ariadne.eval.harness import RunResult, evaluate_run, format_table, load_questions
from ariadne.llm.embeddings import build_embedder
from ariadne.retrieval.hybrid import HybridSearch, SearchMode
from ariadne.retrieval.reranking import build_reranker


def main() -> int:
    force_utf8_output()
    questions = load_questions()
    print(f"golden set: {len(questions)} perguntas\n")

    # Compartilhados entre as configuracoes: comparar estrategias exige que a
    # unica variavel seja a estrategia, e recarregar o reranker a cada rodada
    # ainda inflaria o tempo de forma enganosa.
    embedder = build_embedder()
    reranker = build_reranker()

    def motor(graph_weight: float, pool: int = 20) -> HybridSearch:
        return HybridSearch(
            embedder=embedder,
            reranker=reranker,
            candidate_pool=pool,
            graph_weight=graph_weight,
        )

    configuracoes = [
        ("RAG puro (vetorial)", motor(0.0), SearchMode.VECTOR, False),
        ("Lexical (BM25)", motor(0.0), SearchMode.LEXICAL, False),
        ("Hibrido sem grafo", motor(0.0), SearchMode.HYBRID, False),
        ("GraphRAG (hibrido + grafo)", motor(1.0), SearchMode.HYBRID, False),
        ("GraphRAG + reranking", motor(1.0), SearchMode.HYBRID, True),
    ]

    runs: list[RunResult] = []
    for label, engine, mode, rerank in configuracoes:
        print(f"avaliando {label}...", flush=True)
        runs.append(evaluate_run(label, engine, questions, mode=mode, limit=5, rerank=rerank))

    print()
    print(format_table(runs))

    base = next(r for r in runs if r.label.startswith("RAG puro"))
    melhor = max(runs, key=lambda r: r.recall())
    delta = melhor.recall() - base.recall()
    print(
        f"\nGanho de {melhor.label} sobre o RAG puro: "
        f"{delta:+.0%} no recall geral, "
        f"{melhor.recall('relacional') - base.recall('relacional'):+.0%} nas relacionais."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
