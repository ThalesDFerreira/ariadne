"""Roteiro de demonstracao do Ariadne.

Mostra, em sequencia, o que cada camada do motor acrescenta -- pensado para ser
gravado como GIF do README.

    uv run python scripts/demo.py

Cada cena responde a uma pergunta que a anterior nao resolvia, que e a forma
mais curta de justificar por que o sistema tem tantas partes.
"""

from __future__ import annotations

import sys
import time

from ariadne.console import force_utf8_output
from ariadne.domain.graph import normalize_name
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


def cena_corpus() -> None:
    titulo("1. O que esta indexado")
    with connection() as conn:
        apply_schema(conn)
        stats = PgVectorStore().stats(conn)
        stats.update(AgeGraphStore().stats(conn))
    print(f"   {stats['documents']} documentos, {stats['chunks']} trechos")
    print(f"   {stats['nodes']} entidades e {stats['edges']} relacoes no grafo")
    time.sleep(PAUSA)


def cena_busca_vetorial(motor: HybridSearch) -> None:
    titulo("2. Busca vetorial sozinha erra por semelhanca")
    pergunta = "Quem extrai minério de ferro em Itabira?"
    print(f"   pergunta: {pergunta}")
    r = motor.search(pergunta, mode=SearchMode.VECTOR, limit=1, rerank=False)
    if r.hits:
        print(f"   → {r.hits[0].document_title}")
        print("     (Itabira e da Vale; Itabirito, da Gerdau -- vetores vizinhos)")
    time.sleep(PAUSA)


def cena_hibrida(motor: HybridSearch) -> None:
    titulo("3. Hibrida + grafo corrige")
    pergunta = "Quem extrai minério de ferro em Itabira?"
    r = motor.search(pergunta, mode=SearchMode.HYBRID, limit=1, rerank=False)
    print(f"   etapas: {r.trace.summary()}")
    if r.trace.entities:
        print(f"   entidades que guiaram: {', '.join(r.trace.entities[:4])}")
    if r.hits:
        print(f"   → {r.hits[0].document_title}")
    time.sleep(PAUSA)


def cena_grafo() -> None:
    titulo("4. A ligacao que nenhum documento contem")
    store = AgeGraphStore()
    with connection() as conn:
        a = store.find_key(conn, "Bradesco") or normalize_name("Bradesco")
        b = store.find_key(conn, "Previ") or normalize_name("Previ")
        caminho = store.shortest_path(conn, a, b)
    if not caminho:
        print("   (sem caminho -- rode `ariadne graph-build`)")
        return
    print("   Bradesco  ~~~>  Previ")
    for passo in caminho:
        print(f"     {passo.source} ~[{passo.relation.value}]~ {passo.target}")
        if passo.evidence:
            print(f"        \033[2m{passo.evidence[:88]}\033[0m")
    time.sleep(PAUSA)


def cena_citacao(motor: HybridSearch) -> None:
    titulo("5. Nada sai sem a fonte")
    r = motor.search("privatização com apoio do BNDES", limit=2, rerank=False)
    for hit in r.hits:
        print(f"   • {hit.citation()[:88]}")
    time.sleep(PAUSA)


def main() -> int:
    force_utf8_output()
    print("\n\033[1mAriadne — motor de conhecimento AI-first\033[0m")
    motor = HybridSearch()
    cena_corpus()
    cena_busca_vetorial(motor)
    cena_hibrida(motor)
    cena_grafo()
    cena_citacao(motor)
    print("\n\033[1mTudo isso via MCP, em qualquer assistente de IA.\033[0m\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
