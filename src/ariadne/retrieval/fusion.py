"""Reciprocal Rank Fusion: combinar rankings que nao falam a mesma lingua.

O problema que ela resolve: a busca vetorial devolve similaridade de cosseno
(0 a 1, onde 0,55 ja e um bom resultado) e a lexical devolve `ts_rank_cd`
(escala aberta, depende do tamanho do texto e da frequencia dos termos).
Somar, medir media ou normalizar essas escalas exige calibracao -- e calibracao
feita num corpus quebra no proximo.

A saida do RRF e ignorar o score e olhar so a POSICAO:

    RRF(d) = soma sobre cada ranking de  1 / (k + posicao(d))

Um documento em primeiro lugar contribui 1/(k+1); em decimo, 1/(k+10). Nao
importa se o score era 0,9 ou 900. Isso torna a fusao imune a escala e e a
razao de o RRF funcionar sem nenhum ajuste fino.

O `k` amortece o topo: com k=60 (valor da literatura, usado aqui), a diferenca
entre o 1o e o 2o lugar e pequena, entao um unico ranking confiante nao
atropela o outro. `k` menor deixa os primeiros colocados dominarem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from ariadne.domain.models import RetrievedChunk

DEFAULT_K = 60


@dataclass(frozen=True)
class RankedList:
    """Um ranking de entrada, com o peso que ele tem na fusao."""

    name: str
    results: Sequence[RetrievedChunk]
    weight: float = 1.0


def reciprocal_rank_fusion(
    rankings: Sequence[RankedList],
    *,
    k: int = DEFAULT_K,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Funde varios rankings num so.

    O score devolvido e o valor do RRF, nao mais a similaridade: depois da
    fusao os numeros so fazem sentido comparados entre si, e apresenta-los
    como "similaridade" enganaria quem le.

    Empate e desfeito pelo id do chunk, nao pela ordem de chegada -- sem isso o
    resultado mudaria entre execucoes com os mesmos dados, e teste de busca
    viraria loteria.
    """
    if k < 1:
        msg = "k precisa ser >= 1"
        raise ValueError(msg)

    pontos: dict[UUID, float] = {}
    melhor: dict[UUID, RetrievedChunk] = {}

    for ranking in rankings:
        if ranking.weight <= 0:
            continue
        for posicao, hit in enumerate(ranking.results, start=1):
            chunk_id = hit.chunk.id
            pontos[chunk_id] = pontos.get(chunk_id, 0.0) + ranking.weight / (k + posicao)
            # Guarda a versao com a melhor colocacao original, so para
            # preservar o texto e a citacao; o score sera substituido.
            if chunk_id not in melhor:
                melhor[chunk_id] = hit

    ordenado = sorted(pontos.items(), key=lambda kv: (-kv[1], str(kv[0])))
    if limit is not None:
        ordenado = ordenado[:limit]

    return [
        RetrievedChunk(
            chunk=melhor[chunk_id].chunk,
            document_title=melhor[chunk_id].document_title,
            document_url=melhor[chunk_id].document_url,
            score=score,
        )
        for chunk_id, score in ordenado
    ]
