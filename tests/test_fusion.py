"""Reciprocal Rank Fusion.

Testavel sem banco e sem LLM, e por isso mesmo o lugar certo para travar as
propriedades que fazem a fusao funcionar: imunidade a escala, determinismo e
respeito a posicao.
"""

from uuid import UUID, uuid4

import pytest

from ariadne.domain.models import Chunk, RetrievedChunk
from ariadne.retrieval.fusion import RankedList, reciprocal_rank_fusion

DOC = uuid4()


def hit(nome: str, score: float, chunk_id: UUID | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=chunk_id or uuid4(), document_id=DOC, ordinal=0, content=nome),
        document_title=nome,
        document_url=None,
        score=score,
    )


def nomes(resultados) -> list[str]:
    return [r.document_title for r in resultados]


def test_lista_unica_preserva_a_ordem():
    a, b, c = hit("a", 0.9), hit("b", 0.5), hit("c", 0.1)
    fundido = reciprocal_rank_fusion([RankedList("v", [a, b, c])])
    assert nomes(fundido) == ["a", "b", "c"]


def test_documento_em_ambos_os_rankings_sobe():
    """O ponto do RRF: concordancia entre fontes vale mais que confianca de uma."""
    comum = uuid4()
    vetorial = [hit("so-vetorial", 0.99), hit("comum", 0.40, comum)]
    lexical = [hit("so-lexical", 50.0), hit("comum", 0.10, comum)]

    fundido = reciprocal_rank_fusion(
        [RankedList("vetorial", vetorial), RankedList("lexical", lexical)]
    )
    assert nomes(fundido)[0] == "comum"


def test_e_imune_a_escala_dos_scores():
    """Multiplicar um dos rankings por mil nao pode mudar nada.

    E esta propriedade que dispensa calibrar cosseno contra ts_rank_cd.
    """
    ids = [uuid4() for _ in range(3)]
    normal = [hit("x", 0.9, ids[0]), hit("y", 0.5, ids[1]), hit("z", 0.1, ids[2])]
    inflado = [hit("x", 900.0, ids[0]), hit("y", 500.0, ids[1]), hit("z", 100.0, ids[2])]
    outro = [hit("y", 0.7, ids[1])]

    a = reciprocal_rank_fusion([RankedList("r", normal), RankedList("o", outro)])
    b = reciprocal_rank_fusion([RankedList("r", inflado), RankedList("o", outro)])
    assert nomes(a) == nomes(b)


def test_k_maior_amortece_a_vantagem_do_primeiro_lugar():
    ids = [uuid4(), uuid4()]
    r1 = [hit("p", 1.0, ids[0]), hit("q", 0.9, ids[1])]

    diferenca = {}
    for k in (1, 60):
        fundido = reciprocal_rank_fusion([RankedList("r", r1)], k=k)
        diferenca[k] = fundido[0].score - fundido[1].score
    assert diferenca[60] < diferenca[1]


def test_peso_zero_ignora_o_ranking():
    fundido = reciprocal_rank_fusion(
        [
            RankedList("usado", [hit("a", 1.0)]),
            RankedList("ignorado", [hit("b", 1.0)], weight=0.0),
        ]
    )
    assert nomes(fundido) == ["a"]


def test_peso_maior_puxa_o_ranking_para_cima():
    ids = [uuid4(), uuid4()]
    forte = [hit("forte", 0.1, ids[0])]
    fraco = [hit("fraco", 0.9, ids[1])]
    fundido = reciprocal_rank_fusion(
        [RankedList("a", forte, weight=3.0), RankedList("b", fraco, weight=1.0)]
    )
    assert nomes(fundido)[0] == "forte"


def test_empate_e_deterministico():
    """Sem desempate estavel, o mesmo dado devolveria ordens diferentes."""
    ids = sorted([uuid4(), uuid4()], key=str)
    entrada = [
        RankedList("a", [hit("um", 1.0, ids[0])]),
        RankedList("b", [hit("dois", 1.0, ids[1])]),
    ]
    primeira = nomes(reciprocal_rank_fusion(entrada))
    for _ in range(5):
        assert nomes(reciprocal_rank_fusion(entrada)) == primeira


def test_limit_corta_o_resultado():
    ranking = [hit(str(i), 1.0) for i in range(10)]
    assert len(reciprocal_rank_fusion([RankedList("r", ranking)], limit=3)) == 3


def test_entrada_vazia():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([RankedList("r", [])]) == []


def test_k_invalido_falha():
    with pytest.raises(ValueError, match="k precisa"):
        reciprocal_rank_fusion([RankedList("r", [hit("a", 1.0)])], k=0)


def test_score_devolvido_e_o_do_rrf_e_nao_a_similaridade():
    """Apresentar o valor fundido como "similaridade" enganaria quem le."""
    fundido = reciprocal_rank_fusion([RankedList("r", [hit("a", 0.87)])], k=60)
    assert fundido[0].score == pytest.approx(1 / 61)
