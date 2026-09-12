"""Extracao: schema, cache e higiene da saida do LLM.

Os testes de schema nao sao formalidade. A regra "campo opcional em structured
output e campo que o modelo aprende a omitir" custou uma extracao inteira vazia
antes de ser descoberta.
"""

import httpx
import pytest

from ariadne.config import get_settings
from ariadne.domain.graph import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    RelationType,
)
from ariadne.llm.extraction import OllamaExtractor, chunk_fingerprint


def ent(nome: str) -> ExtractedEntity:
    return ExtractedEntity(name=nome, type=EntityType.ORGANIZACAO)


def rel(origem: str, destino: str) -> ExtractedRelation:
    return ExtractedRelation(
        source=origem, target=destino, type=RelationType.ADQUIRIU, evidence="x"
    )


# --- schema -----------------------------------------------------------------


def test_ambos_os_campos_sao_obrigatorios_no_schema():
    """Se `entities` sair de `required`, o modelo volta a omiti-lo."""
    schema = Extraction.model_json_schema()
    assert set(schema["required"]) == {"entities", "relations"}


def test_entities_vem_antes_de_relations():
    """A geracao e autoregressiva: listar entidades primeiro guia as relacoes."""
    schema = Extraction.model_json_schema()
    assert list(schema["properties"].keys()) == ["entities", "relations"]


def test_tipos_de_entidade_e_relacao_sao_fechados():
    schema = Extraction.model_json_schema()
    definicoes = schema.get("$defs", {})
    assert "enum" in definicoes["EntityType"]
    assert "enum" in definicoes["RelationType"]


# --- higiene da saida -------------------------------------------------------


def test_relacao_com_ponta_desconhecida_e_descartada():
    """Nome citado que o modelo nao declarou como entidade viraria no orfao."""
    bruta = Extraction(
        entities=[ent("Vale")],
        relations=[rel("Vale", "Empresa Fantasma")],
    )
    assert bruta.drop_dangling_relations().relations == []


def test_relacao_valida_sobrevive():
    bruta = Extraction(entities=[ent("WEG"), ent("Katt")], relations=[rel("WEG", "Katt")])
    assert len(bruta.drop_dangling_relations().relations) == 1


def test_autorrelacao_e_descartada():
    bruta = Extraction(entities=[ent("Vale")], relations=[rel("Vale", "VALE S.A.")])
    assert bruta.drop_dangling_relations().relations == []


def test_ponta_casa_pela_forma_normalizada():
    """O modelo escreve "WEG S.A." na relacao e "WEG" na entidade: e a mesma."""
    bruta = Extraction(entities=[ent("WEG"), ent("Katt")], relations=[rel("WEG S.A.", "Katt")])
    assert len(bruta.drop_dangling_relations().relations) == 1


def test_entidade_sem_nome_e_rejeitada():
    with pytest.raises(ValueError, match="vazio"):
        ExtractedEntity(name="   ", type=EntityType.ORGANIZACAO)


# --- cache ------------------------------------------------------------------


def test_fingerprint_muda_com_o_texto():
    a = chunk_fingerprint("texto A", "modelo", "v1")
    b = chunk_fingerprint("texto B", "modelo", "v1")
    assert a != b


def test_fingerprint_muda_com_o_modelo():
    """Trocar de modelo tem que invalidar o cache, senao servimos saida velha."""
    a = chunk_fingerprint("texto", "qwen2.5:7b-instruct", "v1")
    b = chunk_fingerprint("texto", "llama3.1:8b", "v1")
    assert a != b


def test_fingerprint_muda_com_a_versao_do_prompt():
    a = chunk_fingerprint("texto", "modelo", "v1")
    b = chunk_fingerprint("texto", "modelo", "v2")
    assert a != b


def test_fingerprint_e_estavel():
    assert chunk_fingerprint("t", "m", "v1") == chunk_fingerprint("t", "m", "v1")


# --- contra o modelo real ---------------------------------------------------


@pytest.mark.integration
def test_modelo_real_devolve_os_dois_campos():
    cfg = get_settings()
    try:
        httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=3.0).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Ollama indisponivel ({exc})")

    extractor = OllamaExtractor()
    saida = extractor.extract(
        "A WEG comprou a Efacec Energy Service em setembro.",
        context="WEG S.A. > Expansão",
    )
    extractor.close()
    # O ponto do teste: `entities` preenchido. Vazio aqui significa que o
    # schema voltou a deixar o campo opcional.
    assert saida.entities, "modelo devolveu extracao sem entidades"
