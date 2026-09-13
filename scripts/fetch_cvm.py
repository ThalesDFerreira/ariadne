"""Monta o corpus da CVM em disco.

    uv run python scripts/fetch_cvm.py                    # reproduz o manifesto
    uv run python scripts/fetch_cvm.py --refazer-selecao  # escolhe de novo

Baixa os metadados anuais, escolhe documentos em que uma empresa cita outra que
tambem publicou, e salva os PDFs em data/cvm/. Depois disso:

    uv run ariadne ingest-dir data/cvm
    uv run ariadne graph-build

REPRODUTIBILIDADE
-----------------
Por padrao o script SEGUE o manifesto versionado (`data/golden/corpus_cvm.json`),
nao a selecao. Rodar a selecao de novo daria outro corpus -- o CSV do ano
corrente ganha linhas toda semana -- e um corpus diferente invalida o golden
set, cujas ancoras apontam para trechos destes documentos. `--refazer-selecao`
existe para montar um corpus novo de proposito; quem faz isso precisa refazer o
golden set tambem.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import httpx

from ariadne.console import force_utf8_output
from ariadne.ingestion.cvm import (
    Registro,
    baixar_documento,
    baixar_metadados,
    escrever_manifesto,
    ler_manifesto,
    selecionar,
)

RAIZ = pathlib.Path(__file__).resolve().parents[1]
MANIFESTO = "data/golden/corpus_cvm.json"


def main(argv: list[str] | None = None) -> int:
    force_utf8_output()
    parser = argparse.ArgumentParser(description="Baixa o corpus da CVM")
    parser.add_argument("--ano", type=int, default=2025)
    parser.add_argument("--max", type=int, default=60, dest="maximo")
    parser.add_argument("--destino", default="data/cvm")
    parser.add_argument("--manifesto", default=MANIFESTO)
    parser.add_argument(
        "--refazer-selecao",
        action="store_true",
        help="ignora o manifesto e escolhe um corpus novo (invalida o golden set)",
    )
    args = parser.parse_args(argv)

    destino = RAIZ / args.destino
    manifesto = RAIZ / args.manifesto
    cache = RAIZ / "data" / f"ipe_{args.ano}.zip"

    print(f"baixando metadados de {args.ano}...")
    linhas = baixar_metadados(args.ano, cache)
    print(f"  {len(linhas)} registros no ano")

    registros: list[Registro]
    if manifesto.exists() and not args.refazer_selecao:
        registros = ler_manifesto(manifesto, linhas)
        print(f"  seguindo o manifesto: {len(registros)} documento(s)\n")
    else:
        selecao = selecionar(linhas, max_documentos=args.maximo)
        registros = selecao.registros
        print(f"  {selecao.summary()}\n")
        print("pares com publicacao dos dois lados (onde o grafo tem o que ligar):")
        for a, b in selecao.pares[:8]:
            print(f"  {a[:40]:<42} <-> {b[:40]}")
        print()
        escrever_manifesto(selecao, manifesto, args.ano)
        print(f"manifesto escrito em {args.manifesto}\n")

    baixados = 0
    falhas = 0
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for i, registro in enumerate(registros, 1):
            nome = f"{registro.protocolo.replace('/', '-')}.pdf"
            caminho = baixar_documento(registro, destino / nome, client)
            if caminho is None:
                falhas += 1
            else:
                baixados += 1
                # Guarda o titulo legivel ao lado do PDF: o nome do arquivo e o
                # protocolo da CVM, que nao diz nada a quem le a citacao depois.
                caminho.with_suffix(".txt").write_text(
                    f"{registro.empresa}\n{registro.assunto}\n{registro.data}\n",
                    encoding="utf-8",
                )
            if i % 10 == 0:
                print(f"  {i}/{len(registros)} ({baixados} ok, {falhas} falha)", flush=True)

    print(f"\n{baixados} PDF(s) em {destino}")
    if falhas:
        print(f"{falhas} documento(s) nao vieram como PDF e foram ignorados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
