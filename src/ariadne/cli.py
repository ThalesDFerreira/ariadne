"""Interface de linha de comando do Ariadne."""

from __future__ import annotations

import argparse
import sys

from ariadne.ingestion.corpus import DEMO_CORPUS
from ariadne.ingestion.pipeline import IngestionPipeline
from ariadne.ingestion.wikipedia import WikipediaSource
from ariadne.retrieval.vector_search import VectorSearch
from ariadne.storage.database import connection
from ariadne.storage.schema import apply_schema
from ariadne.storage.vector_store import PgVectorStore


def cmd_ingest(args: argparse.Namespace) -> int:
    titles = args.titles or DEMO_CORPUS
    source = WikipediaSource()
    print(f"buscando {len(titles)} pagina(s) da Wikipedia...")
    documents = source.fetch_many(titles)
    source.close()

    faltando = len(titles) - len(documents)
    if faltando:
        print(f"  aviso: {faltando} pagina(s) nao encontrada(s)")

    print("gerando embeddings e indexando (a primeira chamada carrega o modelo)...")
    report = IngestionPipeline().run(documents, force=args.force)
    print(report.summary())
    for titulo in report.skipped:
        print(f"  inalterado: {titulo}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    results = VectorSearch().search(args.query, limit=args.limit)
    if not results:
        print("nenhum resultado -- o corpus foi ingerido?")
        return 1
    for i, hit in enumerate(results, 1):
        trecho = hit.chunk.content.replace("\n", " ")[:220]
        print(f"\n[{i}] score={hit.score:.4f}")
        print(f"    fonte: {hit.citation()}")
        print(f"    {trecho}...")
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    with connection() as conn:
        apply_schema(conn)
        stats = PgVectorStore().stats(conn)
    for chave, valor in stats.items():
        print(f"  {chave}: {valor}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ariadne", description="Motor de conhecimento")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingere paginas da Wikipedia")
    p_ingest.add_argument("titles", nargs="*", help="titulos (vazio = corpus de demo)")
    p_ingest.add_argument("--force", action="store_true", help="reindexa mesmo sem mudanca")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="busca vetorial")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_stats = sub.add_parser("stats", help="contagens do indice")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
