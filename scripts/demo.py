"""Roteiro de demonstracao do Ariadne.

Mostra, em sequencia, o que cada camada do motor acrescenta -- pensado para ser
gravado como GIF do README.

    uv run python scripts/demo.py

Cada cena responde a uma pergunta que a anterior nao resolvia, que e a forma
mais curta de justificar por que o sistema tem tantas partes.

AS PERGUNTAS SAEM DO GOLDEN SET, NAO DO CODIGO
----------------------------------------------
A versao anterior tinha "Quem extrai minerio de ferro em Itabira?" e
"Bradesco ~~> Previ" escritos aqui dentro. Funcionava perfeitamente com o corpus
da Wikipedia, e virou uma demo que nao demonstrava nada no dia em que o corpus
passou a ser a CVM -- sem erro nenhum, so cenas vazias. Exemplo escrito a mao e
exemplo que caduca em silencio.

Agora as perguntas vem de `data/golden/questions.json`, e a cena 3 escolhe
sozinha uma pergunta multi-hop em que a busca vetorial pura perde a ancora e a
completa recupera. Se o corpus mudar de novo, a demo muda junto ou fica
honestamente vazia com um aviso.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from ariadne.console import force_utf8_output
from ariadne.eval.harness import fold, load_questions
from ariadne.retrieval.hybrid import HybridSearch, SearchMode
from ariadne.storage.database import connection
from ariadne.storage.graph_store import AgeGraphStore
from ariadne.storage.schema import apply_schema
from ariadne.storage.vector_store import PgVectorStore

PAUSA = 1.2


def titulo(texto: str) -> None:
    print(f"\n\033[1m{texto}\033[0m")
    print("─" * min(len(texto), 72))
    time.sleep(PAUSA)


def achou_ancora(motor: HybridSearch, caso: dict[str, Any], **kwargs: Any) -> bool:
    hits = motor.search(caso["question"], **kwargs).hits
    textos = [fold(h.chunk.content) for h in hits]
    return all(any(fold(a) in t for t in textos) for a in caso["anchors"])


def cena_corpus() -> None:
    titulo("1. O que esta indexado")
    with connection() as conn:
        apply_schema(conn)
        stats = PgVectorStore().stats(conn)
        stats.update(AgeGraphStore().stats(conn))
    print(f"   {stats['documents']} documentos, {stats['chunks']} trechos")
    print(f"   {stats['nodes']} entidades e {stats['edges']} relacoes no grafo")
    time.sleep(PAUSA)


def cena_factual(motor: HybridSearch, casos: list[dict[str, Any]]) -> None:
    titulo("2. Pergunta factual: o RAG comum ja resolve")
    factual = next((c for c in casos if c["kind"] == "factual"), None)
    if factual is None:
        print("   (golden set sem pergunta factual)")
        return
    print(f"   pergunta: {factual['question']}")
    r = motor.search(factual["question"], mode=SearchMode.VECTOR, limit=3, rerank=False)
    if r.hits:
        print(f"   → {r.hits[0].document_title[:76]}")
    print("   Nao e aqui que o grafo se paga -- e honesto dizer isso.")
    time.sleep(PAUSA)


def cena_multihop(motor: HybridSearch, casos: list[dict[str, Any]]) -> dict[str, Any] | None:
    titulo("3. Pergunta que atravessa dois documentos")
    for caso in (c for c in casos if c["kind"] == "multihop"):
        vetorial = achou_ancora(motor, caso, mode=SearchMode.VECTOR, limit=5, rerank=False)
        completo = achou_ancora(motor, caso, mode=SearchMode.HYBRID, limit=5, rerank=True)
        if not vetorial and completo:
            print(f"   pergunta: {caso['question']}")
            print(f"   ancora esperada: {', '.join(caso['anchors'])}")
            print("   busca vetorial pura .......... nao trouxe")
            print("   hibrida + grafo + reranking .. trouxe")
            r = motor.search(caso["question"], mode=SearchMode.HYBRID, limit=5, rerank=True)
            print(f"   etapas: {r.trace.summary()}")
            if r.trace.entities:
                print(f"   entidades que guiaram: {', '.join(r.trace.entities[:4])}")
            time.sleep(PAUSA)
            return caso
    print("   (nenhuma multi-hop separa as estrategias neste corpus)")
    print("   O golden set saturou: e um resultado, nao uma falha da demo.")
    time.sleep(PAUSA)
    return None


def cena_grafo(casos: list[dict[str, Any]]) -> None:
    titulo("4. A ligacao que nenhum documento contem sozinho")
    ponte = next((c.get("bridge_entity") for c in casos if c.get("bridge_entity")), None)
    if ponte is None:
        print("   (golden set sem entidade-ponte declarada)")
        return

    store = AgeGraphStore()
    with connection() as conn:
        chave = store.find_key(conn, ponte)
        if chave is None:
            print(f"   ({ponte} nao esta no grafo -- rode `ariadne graph-build`)")
            return
        vizinhos = store.neighbors(conn, chave, depth=2, limit=60)

    # `neighbors` devolve UMA LINHA POR ARESTA, e a mesma relacao afirmada em
    # trechos diferentes e aresta diferente -- por design, para nao perder a
    # evidencia de cada trecho. Numa demo isso vira "Brava Energia" doze vezes
    # seguidas. Aqui a chave (relacao, nome) vale uma linha so.
    vistos: set[tuple[str, str]] = set()
    print(f"   a partir de {ponte}, dois saltos:")
    mostrados = 0
    for v in vizinhos:
        chave_v = (v.relation.value, v.name)
        if chave_v in vistos:
            continue
        vistos.add(chave_v)
        print(f"     ~[{v.relation.value}]~ {v.name}")
        if v.evidence:
            print(f"        \033[2m{v.evidence[:88]}\033[0m")
        mostrados += 1
        if mostrados >= 5:
            break
    time.sleep(PAUSA)


def cena_citacao(motor: HybridSearch, casos: list[dict[str, Any]]) -> None:
    titulo("5. Nada sai sem a fonte")
    pergunta = casos[0]["question"] if casos else "aquisicao de participacao"
    r = motor.search(pergunta, limit=2, rerank=False)
    for hit in r.hits:
        print(f"   • {hit.citation()[:88]}")
    time.sleep(PAUSA)


def main() -> int:
    force_utf8_output()
    print("\n\033[1mAriadne — motor de conhecimento AI-first\033[0m")
    casos = load_questions()
    motor = HybridSearch()
    cena_corpus()
    cena_factual(motor, casos)
    cena_multihop(motor, casos)
    cena_grafo(casos)
    cena_citacao(motor, casos)
    print("\n\033[1mTudo isso via MCP, em qualquer assistente de IA.\033[0m\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
