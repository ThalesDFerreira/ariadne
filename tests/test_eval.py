"""Harness de avaliacao.

A metrica precisa ser confiavel antes de servir para decidir qualquer coisa --
uma medicao errada e pior que nenhuma, porque da confianca falsa.
"""

import json

import pytest

from ariadne.eval.harness import (
    QuestionResult,
    RunResult,
    fold,
    format_table,
    load_questions,
)


def qr(kind: str, found: int, expected: int, rank: int | None) -> QuestionResult:
    return QuestionResult(id="x", kind=kind, found=found, expected=expected, rank=rank)


# --- normalizacao -----------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Paraná", "PARANA"),
        ("Ágora", "agora"),
        ("São Francisco", "sao francisco"),
        ("Antárctica", "ANTARCTICA"),
    ],
)
def test_acento_e_caixa_nao_contam(a, b):
    """Penalizar grafia mediria ortografia, nao recuperacao."""
    assert fold(a) == fold(b)


def test_termos_diferentes_continuam_diferentes():
    assert fold("Itabira") != fold("Itabirito")


# --- metricas ---------------------------------------------------------------


def test_recall_parcial():
    assert qr("factual", 1, 2, 1).recall == 0.5
    assert qr("factual", 2, 2, 1).recall == 1.0
    assert qr("factual", 0, 2, None).recall == 0.0


def test_hit_exige_todas_as_ancoras():
    assert qr("factual", 2, 2, 1).hit is True
    assert qr("factual", 1, 2, 1).hit is False


def test_mrr_premia_posicao_alta():
    """Recall diz se recuperou; MRR diz se ficou onde o modelo vai ler."""
    alto = RunResult("a", [qr("factual", 1, 1, 1)])
    baixo = RunResult("b", [qr("factual", 1, 1, 5)])
    assert alto.mrr() == 1.0
    assert baixo.mrr() == 0.2
    # Mesmo recall, MRR diferente: trazer a resposta em quinto num contexto de
    # tres e o mesmo que nao trazer.
    assert alto.recall() == baixo.recall()


def test_mrr_zera_quando_nao_encontrou():
    assert RunResult("x", [qr("factual", 0, 1, None)]).mrr() == 0.0


def test_metricas_por_tipo():
    run = RunResult(
        "x",
        [
            qr("factual", 1, 1, 1),
            qr("factual", 1, 1, 1),
            qr("relacional", 0, 1, None),
            qr("relacional", 0, 1, None),
        ],
    )
    assert run.recall("factual") == 1.0
    assert run.recall("relacional") == 0.0
    assert run.recall() == 0.5
    assert run.hit_rate("factual") == 1.0


def test_run_vazio_nao_divide_por_zero():
    vazio = RunResult("x", [])
    assert vazio.recall() == 0.0
    assert vazio.mrr() == 0.0
    assert vazio.hit_rate() == 0.0


def test_tipo_inexistente_nao_quebra():
    run = RunResult("x", [qr("factual", 1, 1, 1)])
    assert run.recall("nao-existe") == 0.0


# --- golden set -------------------------------------------------------------


def test_golden_set_esta_bem_formado():
    casos = load_questions()
    assert len(casos) >= 30, "golden set pequeno demais para comparar estrategias"
    ids = [c["id"] for c in casos]
    assert len(ids) == len(set(ids)), "ids duplicados no golden set"
    for caso in casos:
        assert caso["question"].strip()
        assert caso["anchors"], f"{caso['id']} sem ancora -- nao mede nada"
        assert caso["kind"] in {"factual", "relacional", "sintese"}


def test_golden_set_tem_relacionais_suficientes():
    """Sem perguntas relacionais, o grafo nao teria como mostrar vantagem."""
    casos = load_questions()
    relacionais = [c for c in casos if c["kind"] == "relacional"]
    assert len(relacionais) >= len(casos) * 0.3


def test_golden_set_e_json_valido_no_disco():
    from ariadne.eval.harness import GOLDEN

    json.loads(GOLDEN.read_text(encoding="utf-8"))


# --- apresentacao -----------------------------------------------------------


def test_tabela_sai_em_markdown():
    tabela = format_table(
        [
            RunResult("A", [qr("factual", 1, 1, 1)], seconds=1.0),
            RunResult("B", [qr("factual", 0, 1, None)], seconds=2.0),
        ]
    )
    assert tabela.startswith("| estrategia")
    assert "| A |" in tabela and "| B |" in tabela
    assert "100%" in tabela and "0%" in tabela
