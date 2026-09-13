"""Mede o efeito liquido de uma heuristica do roteador, ligada contra desligada.

    uv run python scripts/ab_roteador.py

POR QUE ESTE SCRIPT EXISTE
--------------------------
A heuristica `_PONTE` foi escrita para consertar um defeito medido: 3 das 4
perguntas multi-hop caiam em "factual", que usa peso de grafo 0,2, porque
descrevem a ponte em vez de nomea-la ("as concessoes ENVOLVIDAS NA OPERACAO da
PetroReconcavo"). Depois dela, as multi-hop melhoraram e as factuais pioraram
-- 4 falsos positivos onde havia 2.

Isso e uma TROCA, e olhar so a matriz de classificacao nao diz se ela vale. O
que decide e o recall final: se acertar 4 perguntas de um tipo e errar 4 de
outro der o mesmo numero no fim, a heuristica nao melhorou nada, so mudou de
erro -- e entao ela sai.

O A/B roda o mesmo golden set, no mesmo indice, com a unica diferenca sendo a
heuristica ligada ou desligada.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

from ariadne.agents import router as modulo_router
from ariadne.agents.router import STRATEGIES, QueryRouter
from ariadne.console import force_utf8_output
from ariadne.eval.harness import evaluate_routed, format_table, load_questions
from ariadne.llm.embeddings import build_embedder
from ariadne.retrieval.hybrid import HybridSearch
from ariadne.retrieval.reranking import build_reranker

NUNCA_CASA = re.compile(r"(?!x)x")


def distribuicao(router: QueryRouter, questions: list[dict[str, object]]) -> str:
    por_tipo: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for caso in questions:
        escolhido = router.route(str(caso["question"])).kind
        por_tipo[str(caso["kind"])][str(escolhido)] += 1
    return "\n".join(f"  {k:<12} -> {dict(v)}" for k, v in sorted(por_tipo.items()))


def main() -> int:
    force_utf8_output()
    questions = load_questions()
    embedder = build_embedder()
    reranker = build_reranker()

    def motor(graph_weight: float, pool: int) -> HybridSearch:
        return HybridSearch(
            embedder=embedder,
            reranker=reranker,
            candidate_pool=pool,
            graph_weight=graph_weight,
        )

    engines = {e.graph_weight: motor(e.graph_weight, e.candidate_pool) for e in STRATEGIES.values()}
    router = QueryRouter(use_llm=False)

    ponte_original = modulo_router._PONTE

    print("classificacao COM _PONTE:")
    print(distribuicao(router, questions))
    print("\navaliando com _PONTE...", flush=True)
    com = evaluate_routed("roteado (com _PONTE)", engines, router, questions)

    modulo_router._PONTE = NUNCA_CASA
    print("\nclassificacao SEM _PONTE:")
    print(distribuicao(router, questions))
    print("\navaliando sem _PONTE...", flush=True)
    sem = evaluate_routed("roteado (sem _PONTE)", engines, router, questions)
    modulo_router._PONTE = ponte_original

    router.close()

    print()
    print(format_table([sem, com]))

    tipos = sorted({r.kind for r in com.results})
    print("\nefeito liquido de _PONTE (com - sem):")
    print(f"  recall geral  {com.recall() - sem.recall():+.1%}")
    for t in tipos:
        print(f"  {t:<12}  {com.recall(t) - sem.recall(t):+.1%}")
    print(f"  MRR           {com.mrr() - sem.mrr():+.3f}")

    # Empate no recall NAO e evidencia de ausencia de efeito quando o recall
    # esta no teto: e regua sem resolucao. Foi o que aconteceu na primeira
    # medicao (100% nas quatro colunas dos dois lados), e a versao anterior
    # desta regra imprimia "REVERTER" por empate -- decidindo com uma metrica
    # que ja nao distingue nada. O MRR nao satura junto e entra como criterio
    # de desempate; o pipeline e deterministico, entao a diferenca e
    # reprodutivel, nao ruido.
    if com.recall() != sem.recall():
        veredito = "MANTER" if com.recall() > sem.recall() else "REVERTER"
        criterio = "recall"
    elif com.recall() >= 0.999:
        veredito = "MANTER" if com.mrr() > sem.mrr() else "REVERTER"
        criterio = "MRR (recall saturado nos dois lados -- a regua nao separa)"
    else:
        veredito = "REVERTER"
        criterio = "empate fora do teto: heuristica que nao move o numero e complexidade de graca"
    print(f"\nveredito: {veredito}  (criterio: {criterio})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
