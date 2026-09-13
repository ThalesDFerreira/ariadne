"""Monta o golden set multi-hop, validando a dificuldade por codigo.

Uma pergunta so entra se for GENUINAMENTE multi-hop, e isso nao e opiniao: a
ancora (a resposta) nao pode aparecer em nenhum trecho que tambem mencione a
entidade citada na pergunta. Se aparecesse, um unico trecho responderia e o
grafo seria dispensavel -- a pergunta mediria recuperacao simples, e o teste
estaria inflado a favor do grafo.

Este arquivo existe para que a dificuldade do benchmark seja auditavel. Quem
duvidar do resultado roda o script e ve quais perguntas foram descartadas e por
que.

    uv run python scripts/build_multihop.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import unicodedata

from ariadne.console import force_utf8_output
from ariadne.storage.database import connection

SAIDA = pathlib.Path(__file__).resolve().parents[1] / "data/golden/questions_multihop.json"

# (id, pergunta, entidade citada NA pergunta, ancora esperada)
CANDIDATAS: list[tuple[str, str, str, str]] = [
    # Entidade-ponte CRUZADA: a pergunta cita a empresa A e a resposta so
    # existe no documento da empresa B. E a unica forma de multi-hop possivel
    # quando cada documento descreve a operacao inteira.
    (
        "cruz-petro-producao",
        "Qual a producao media diaria das concessoes envolvidas na operacao da "
        "PetroReconcavo no Rio Grande do Norte?",
        "PetroReconcavo",
        "250 barris",
    ),
    (
        "cruz-brava-financiamento",
        "A compradora dos ativos de midstream da Brava estuda financiar ate que "
        "percentual da transacao?",
        "Brava",
        "100% do valor",
    ),
    (
        "cruz-mrs-votante",
        "Que percentual do capital votante da MRS a compradora passou a deter na "
        "operacao anunciada pela MRS Logistica?",
        "MRS Log",
        "14,30%",
    ),
    (
        "cruz-csn-reducao",
        "Para quanto caiu a participacao da CSN na MRS apos a aquisicao da CSN Mineracao?",
        "CSN Mineracao",
        "7,59%",
    ),
    (
        "cruz-itau-sendas",
        "Qual o valor recebido na venda da participacao na FIC anunciada pela "
        "Companhia Brasileira de Distribuicao?",
        "Companhia Brasileira de Distribui",
        "260 milh",
    ),
    (
        "cruz-bnb-conpel",
        "Ate que data vai o direito de retirada na incorporacao envolvendo o Banco do Nordeste?",
        "Banco do Nordeste",
        "28 de abril",
    ),
    (
        "cruz-marfrig-minerva",
        "Qual empresa a Marfrig adquiriu por meio da MBR?",
        "Minerva",
        "Gelprime",
    ),
    (
        "cruz-rumo-malha",
        "Que operacao a Rumo Malha Norte anunciou junto com a Rumo?",
        "Rumo Malha Norte",
        "incorpora",
    ),
    (
        "cruz-neoenergia-itabapoana",
        "Qual usina a Neoenergia vendeu alem da participacao na LT Itabapoana?",
        "Itabapoana",
        "Baixo Igua",
    ),
    (
        "cruz-santander-leasing",
        "Qual empresa foi incorporada pelo Banco Santander?",
        "Santander",
        "Leasing",
    ),
]


def fold(texto: str) -> str:
    base = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in base if not unicodedata.combining(c)).lower()


def main() -> int:
    force_utf8_output()
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT content FROM chunks")
        trechos = [fold(r[0]) for r in cur.fetchall()]

    aprovadas: list[dict[str, object]] = []
    print(f"{len(CANDIDATAS)} candidatas\n")

    for cid, pergunta, entidade, ancora in CANDIDATAS:
        fa, fe = fold(ancora), fold(entidade)

        existe = [t for t in trechos if fa in t]
        if not existe:
            print(f"  DESCARTADA {cid}: ancora {ancora!r} nao existe no corpus")
            continue

        # O teste de dificuldade: ancora e entidade da pergunta no MESMO trecho
        # significam que um unico chunk responde -- nao e multi-hop.
        juntos = [t for t in trechos if fa in t and fe in t]
        if juntos:
            print(f"  DESCARTADA {cid}: {ancora!r} e {entidade!r} aparecem no mesmo trecho")
            continue

        aprovadas.append(
            {
                "id": cid,
                "question": pergunta,
                "kind": "multihop",
                "anchors": [ancora],
                "bridge_entity": entidade,
            }
        )
        print(f"  ok         {cid}")

    SAIDA.write_text(json.dumps(aprovadas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(aprovadas)} pergunta(s) aprovada(s) -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
