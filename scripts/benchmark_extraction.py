"""Compara modelos de extracao contra o golden set anotado a mao.

Existe para que a escolha do modelo seja uma medicao, e nao um palpite. A
qualidade do grafo inteiro depende dela: um modelo que inventa relacao produz
um grafo que afirma bobagem com confianca, o que e pior que um grafo pobre.

Uso:
    uv run python scripts/benchmark_extraction.py qwen2.5:7b-instruct llama3.1:8b
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

from ariadne.domain.graph import EntityType, ExtractedEntity, normalize_name
from ariadne.ingestion.entity_resolution import EntityResolver
from ariadne.llm.extraction import OllamaExtractor

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "data/golden/extraction_golden.json"


def entity_keys(items: list[dict[str, str]]) -> set[str]:
    return {normalize_name(i["name"]) for i in items}


def relation_keys(items: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {(normalize_name(i["source"]), normalize_name(i["target"]), i["type"]) for i in items}


def prf(esperado: set, obtido: set) -> tuple[float, float, float]:
    """Precisao, recall e F1.

    Precisao pesa mais aqui: alucinar relacao envenena o grafo de forma
    permanente, enquanto deixar de extrair apenas o deixa incompleto.
    """
    if not esperado and not obtido:
        return 1.0, 1.0, 1.0
    acertos = len(esperado & obtido)
    p = acertos / len(obtido) if obtido else 0.0
    r = acertos / len(esperado) if esperado else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def canonical_map(nomes: list[str]) -> dict[str, str]:
    """Mapa nome -> chave canonica, aplicando a MESMA resolucao do pipeline.

    Sem isso o benchmark reprova um acerto: o modelo devolve "Companhia
    Energetica de Minas Gerais" onde o gabarito diz "Cemig", que a resolucao
    de entidades funde de qualquer forma. Medir antes dela seria medir um
    pipeline que nao existe.
    """
    nos = EntityResolver().resolve(
        [ExtractedEntity(name=n, type=EntityType.ORGANIZACAO) for n in nomes]
    )
    mapa: dict[str, str] = {}
    for no in nos:
        for variante in [no.name, *no.aliases]:
            mapa[normalize_name(variante)] = no.key
    return mapa


def avaliar(modelo: str, casos: list[dict]) -> dict[str, float]:
    extractor = OllamaExtractor(model=modelo)
    ents_esp: set = set()
    ents_obt: set = set()
    rels_esp: set = set()
    rels_obt: set = set()
    inicio = time.monotonic()

    for caso in casos:
        saida = extractor.extract(caso["text"], context=caso.get("context", ""))
        pref = caso["id"]

        # Resolve gabarito e saida no MESMO espaco de chaves, como o pipeline faz.
        todos = [e["name"] for e in caso["entities"]] + [e.name for e in saida.entities]
        mapa = canonical_map(todos)

        def chave(nome: str, mapa: dict[str, str] = mapa) -> str:
            return mapa.get(normalize_name(nome), normalize_name(nome))

        ents_esp |= {f"{pref}|{chave(e['name'])}" for e in caso["entities"]}
        ents_obt |= {f"{pref}|{chave(e.name)}" for e in saida.entities}
        rels_esp |= {
            f"{pref}|{(chave(r['source']), chave(r['target']), r['type'])}"
            for r in caso["relations"]
        }
        rels_obt |= {
            f"{pref}|{(chave(r.source), chave(r.target), r.type.value)}" for r in saida.relations
        }

    extractor.close()
    ep, er, ef = prf(ents_esp, ents_obt)
    rp, rr, rf = prf(rels_esp, rels_obt)
    return {
        "entidades_p": ep,
        "entidades_r": er,
        "entidades_f1": ef,
        "relacoes_p": rp,
        "relacoes_r": rr,
        "relacoes_f1": rf,
        "segundos": time.monotonic() - inicio,
    }


def main(modelos: list[str]) -> int:
    casos = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"golden set: {len(casos)} casos\n")

    linhas = []
    for modelo in modelos:
        print(f"avaliando {modelo}...", flush=True)
        try:
            linhas.append((modelo, avaliar(modelo, casos)))
        except Exception as exc:
            print(f"  falhou: {exc}")

    print(
        f"\n{'modelo':<24} {'ent P':>6} {'ent R':>6} {'ent F1':>7} "
        f"{'rel P':>6} {'rel R':>6} {'rel F1':>7} {'seg':>6}"
    )
    print("-" * 82)
    for modelo, m in sorted(linhas, key=lambda kv: -kv[1]["relacoes_f1"]):
        print(
            f"{modelo:<24} {m['entidades_p']:>6.2f} {m['entidades_r']:>6.2f} "
            f"{m['entidades_f1']:>7.2f} {m['relacoes_p']:>6.2f} {m['relacoes_r']:>6.2f} "
            f"{m['relacoes_f1']:>7.2f} {m['segundos']:>6.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["qwen2.5:7b-instruct"]))
