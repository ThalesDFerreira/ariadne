"""Resolucao de entidades.

Os casos negativos valem mais que os positivos: deixar de fundir gera
duplicata, que e feio mas honesto; fundir errado faz o grafo afirmar bobagem
com confianca total.
"""

import pytest

from ariadne.domain.graph import EntityType, ExtractedEntity, normalize_name
from ariadne.ingestion.entity_resolution import EntityResolver, acronym_of


def ent(nome: str, tipo: EntityType = EntityType.ORGANIZACAO) -> ExtractedEntity:
    return ExtractedEntity(name=nome, type=tipo)


def resolver(*nomes: str, tipo: EntityType = EntityType.ORGANIZACAO):
    return EntityResolver().resolve([ent(n, tipo) for n in nomes])


# --- normalizacao -----------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Petrobras", "PETROBRAS S.A."),
        ("Petrobrás", "Petrobras"),
        ("Vale", "Vale S/A"),
        ("Gerdau", "GERDAU LTDA"),
        ("Ambev", "AmBev  "),
    ],
)
def test_variantes_de_grafia_viram_a_mesma_chave(a, b):
    assert normalize_name(a) == normalize_name(b)


def test_nomes_diferentes_nao_colidem():
    assert normalize_name("Banco do Brasil") != normalize_name("Banco Central do Brasil")
    assert normalize_name("Petrobras") != normalize_name("Petrobras Distribuidora")


# --- siglas -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("extenso", "sigla"),
    [
        ("Companhia Siderúrgica Nacional", "CSN"),
        ("Banco Nacional de Desenvolvimento Econômico e Social", "BNDES"),
        ("Banco do Brasil", "BB"),
        ("Banco Central do Brasil", "BCB"),
    ],
)
def test_sigla_sai_das_iniciais_significativas(extenso, sigla):
    assert acronym_of(extenso) == sigla


def test_sigla_funde_com_a_forma_extensa():
    nos = resolver("CSN", "Companhia Siderúrgica Nacional", "CSN")
    assert len(nos) == 1
    # O rotulo e a variante mais citada, e aqui "CSN" aparece duas vezes.
    # A forma extensa nao se perde: fica como alias e continua pesquisavel.
    assert nos[0].name == "CSN"
    assert "Companhia Siderúrgica Nacional" in nos[0].aliases
    assert nos[0].mentions == 3


def test_sigla_ambigua_nao_e_adivinhada():
    """BB serve a "Banco do Brasil" e a "Banco Bradesco": na duvida, separa."""
    nos = resolver("BB", "Banco do Brasil", "Banco Bradesco")
    assert len(nos) == 3


def test_sigla_nao_funde_entre_tipos_diferentes():
    entidades = [
        ent("CSN", EntityType.ORGANIZACAO),
        ent("Companhia Siderúrgica Nacional", EntityType.LUGAR),
    ]
    assert len(EntityResolver().resolve(entidades)) == 2


# --- os casos perigosos -----------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Medido em 0.928 de similaridade de embedding -- o par mais parecido
        # de todos, e entidades completamente distintas.
        ("Banco do Brasil", "Banco Central do Brasil"),
        ("Petrobras", "Petrobras Distribuidora"),
        ("Usiminas", "Mineração Usiminas"),
        ("Gerdau", "Gerdau Açominas"),
        ("Vale", "Vale do Rio Doce"),
    ],
)
def test_nomes_parecidos_de_entidades_distintas_nao_fundem(a, b):
    nos = resolver(a, b)
    assert len(nos) == 2, f"{a!r} e {b!r} foram fundidos indevidamente"


# --- rotulo canonico --------------------------------------------------------


def test_canonico_e_a_variante_mais_citada():
    nos = resolver("Vale S.A.", "Vale S.A.", "Vale S.A.", "VALE")
    assert nos[0].name == "Vale S.A."


def test_empate_escolhe_a_forma_mais_informativa():
    nos = resolver("CSN", "Companhia Siderúrgica Nacional")
    assert nos[0].name == "Companhia Siderúrgica Nacional"


def test_mencoes_sao_contadas():
    nos = resolver("Petrobras", "PETROBRAS S.A.", "Petrobrás")
    assert len(nos) == 1
    assert nos[0].mentions == 3
    assert len(nos[0].aliases) == 2


def test_lista_vazia():
    assert EntityResolver().resolve([]) == []
