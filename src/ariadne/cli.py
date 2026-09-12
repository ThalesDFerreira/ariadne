"""Interface de linha de comando do Ariadne."""

from __future__ import annotations

import argparse
import sys
import time

from ariadne.agents.orchestrator import KnowledgeAgent
from ariadne.console import force_utf8_output
from ariadne.domain.graph import normalize_name
from ariadne.ingestion.corpus import DEMO_CORPUS
from ariadne.ingestion.graph_builder import GraphBuilder
from ariadne.ingestion.pipeline import IngestionPipeline
from ariadne.ingestion.sources import SourceRejectedError, resolve_source
from ariadne.ingestion.wikipedia import WikipediaSource
from ariadne.retrieval.hybrid import HybridSearch, SearchMode
from ariadne.storage.database import connection
from ariadne.storage.graph_store import AgeGraphStore
from ariadne.storage.schema import apply_schema
from ariadne.storage.vector_store import PgVectorStore


def cmd_ingest(args: argparse.Namespace) -> int:
    titles = args.titles or DEMO_CORPUS
    documents = []

    # URL ou caminho passa pela mesma politica de seguranca da tool MCP; o
    # resto e tratado como titulo da Wikipedia.
    referencias = [t for t in titles if _parece_fonte(t)]
    paginas = [t for t in titles if not _parece_fonte(t)]

    for referencia in referencias:
        try:
            documents.append(resolve_source(referencia))
        except SourceRejectedError as exc:
            print(f"  recusado ({referencia}): {exc}")

    if paginas:
        source = WikipediaSource()
        print(f"buscando {len(paginas)} pagina(s) da Wikipedia...")
        documents.extend(source.fetch_many(paginas))
        source.close()

    faltando = len(titles) - len(documents)
    if faltando:
        print(f"  aviso: {faltando} fonte(s) nao ingerida(s)")
    if not documents:
        return 1

    print("gerando embeddings e indexando (a primeira chamada carrega o modelo)...")
    report = IngestionPipeline().run(documents, force=args.force)
    print(report.summary())
    for titulo in report.skipped:
        print(f"  inalterado: {titulo}")
    return 0


def _parece_fonte(referencia: str) -> bool:
    """Distingue "URL/arquivo" de "titulo da Wikipedia"."""
    baixo = referencia.lower()
    return baixo.startswith(("http://", "https://")) or referencia.endswith(
        (".md", ".markdown", ".txt", ".rst")
    )


def cmd_search(args: argparse.Namespace) -> int:
    resultado = HybridSearch().search(
        args.query,
        mode=SearchMode(args.mode),
        limit=args.limit,
        rerank=not args.no_rerank,
    )
    if not resultado.hits:
        print("nenhum resultado -- o corpus foi ingerido?")
        return 1
    if args.explain:
        print(f"  etapas: {resultado.trace.summary()}")
        if resultado.trace.entities:
            print(f"  entidades semente: {', '.join(resultado.trace.entities[:6])}")
    for i, hit in enumerate(resultado.hits, 1):
        trecho = hit.chunk.content.replace("\n", " ")[:220]
        print(f"\n[{i}] score={hit.score:.4f}")
        print(f"    fonte: {hit.citation()}")
        print(f"    {trecho}...")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    agente = KnowledgeAgent()
    try:
        resultado = agente.ask(args.question)
    finally:
        agente.close()

    print(f"  {resultado.summary()}")
    if args.explain:
        print(f"  rota: {resultado.strategy.reason[:100]}")
        if resultado.strategy.entities:
            print(f"  entidades: {', '.join(resultado.strategy.entities)}")
    print()
    print(resultado.answer.text)
    if resultado.answer.sources:
        print("\nFontes:")
        for i, fonte in enumerate(resultado.answer.sources, 1):
            print(f"  [{i}] {fonte}")
    if resultado.answer.warning:
        print(f"\n  aviso: {resultado.answer.warning}")
    return 0 if resultado.answer.grounded else 1


def cmd_graph_build(args: argparse.Namespace) -> int:
    print("extraindo entidades e relacoes (uma chamada de LLM por chunk novo)...")
    inicio = time.monotonic()

    def progresso(feito: int, total: int) -> None:
        decorrido = time.monotonic() - inicio
        ritmo = feito / decorrido if decorrido else 0
        faltam = (total - feito) / ritmo if ritmo else 0
        print(
            f"  {feito}/{total} chunks  ({ritmo:.2f}/s, ~{faltam / 60:.0f} min restantes)",
            flush=True,
        )

    report = GraphBuilder().build(limit=args.limit, refresh=args.refresh, on_progress=progresso)
    print(report.summary())
    if report.dropped_edges:
        print(f"  {report.dropped_edges} relacao(oes) descartada(s) por ponta desconhecida")
    for erro in report.errors[:5]:
        print(f"  erro: {erro}")
    if len(report.errors) > 5:
        print(f"  ... e mais {len(report.errors) - 5} erro(s)")
    return 0


def cmd_explore(args: argparse.Namespace) -> int:
    store = AgeGraphStore()
    with connection() as conn:
        chave = store.find_key(conn, args.name) or normalize_name(args.name)
        vizinhos = store.neighbors(conn, chave, depth=args.depth)
    if not vizinhos:
        print(f"nenhuma relacao encontrada para {args.name!r}")
        return 1
    print(f"{args.name} ({len(vizinhos)} relacao(oes)):")
    for v in vizinhos:
        seta = "->" if v.direction == "saindo" else "<-"
        print(f"  {seta} {v.relation.value:<16} {v.name}")
        if v.evidence:
            print(f"       {v.evidence[:110]}")
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    store = AgeGraphStore()
    with connection() as conn:
        a = store.find_key(conn, args.source) or normalize_name(args.source)
        b = store.find_key(conn, args.target) or normalize_name(args.target)
        caminho = store.shortest_path(conn, a, b, max_hops=args.max_hops)
    if not caminho:
        print(f"nenhum caminho de {args.source!r} ate {args.target!r}")
        return 1
    # "~" e nao "->": a busca e nao dirigida, entao a ordem do caminho nem
    # sempre e a direcao da aresta. Desenhar uma seta aqui afirmaria quem faz o
    # que, e quem responde isso e a evidencia.
    for passo in caminho:
        print(f"  {passo.source} ~[{passo.relation.value}]~ {passo.target}")
        if passo.evidence:
            print(f"     evidencia: {passo.evidence[:110]}")
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    with connection() as conn:
        apply_schema(conn)
        stats = PgVectorStore().stats(conn)
        stats.update(AgeGraphStore().stats(conn))
    for chave, valor in stats.items():
        print(f"  {chave}: {valor}")
    return 0


def main(argv: list[str] | None = None) -> int:
    force_utf8_output()
    parser = argparse.ArgumentParser(prog="ariadne", description="Motor de conhecimento")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingere paginas da Wikipedia")
    p_ingest.add_argument(
        "titles",
        nargs="*",
        help="titulos da Wikipedia, URLs ou arquivos (vazio = corpus de demo)",
    )
    p_ingest.add_argument("--force", action="store_true", help="reindexa mesmo sem mudanca")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="busca no corpus")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.add_argument(
        "--mode",
        choices=[m.value for m in SearchMode],
        default=SearchMode.HYBRID.value,
        help="estrategia de busca (padrao: hybrid)",
    )
    p_search.add_argument("--no-rerank", action="store_true", help="pula o cross-encoder")
    p_search.add_argument("--explain", action="store_true", help="mostra o que cada etapa rendeu")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="pergunta em linguagem natural, com resposta citada")
    p_ask.add_argument("question")
    p_ask.add_argument("--explain", action="store_true", help="mostra a rota escolhida")
    p_ask.set_defaults(func=cmd_ask)

    p_build = sub.add_parser("graph-build", help="extrai o grafo dos chunks indexados")
    p_build.add_argument("--limit", type=int, default=None, help="processa so N chunks")
    p_build.add_argument("--refresh", action="store_true", help="ignora o cache de extracao")
    p_build.set_defaults(func=cmd_graph_build)

    p_explore = sub.add_parser("explore", help="vizinhanca de uma entidade")
    p_explore.add_argument("name")
    p_explore.add_argument("--depth", type=int, default=1)
    p_explore.set_defaults(func=cmd_explore)

    p_connect = sub.add_parser("connect", help="caminho entre duas entidades")
    p_connect.add_argument("source")
    p_connect.add_argument("target")
    p_connect.add_argument("--max-hops", type=int, default=4, dest="max_hops")
    p_connect.set_defaults(func=cmd_connect)

    p_stats = sub.add_parser("stats", help="contagens do indice")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
