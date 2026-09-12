"""Chunking: onde um bug e silencioso e contamina toda a recuperacao."""

import pytest

from ariadne.domain.models import Document
from ariadne.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
    chunk_text,
    parse_blocks,
)

TEXTO = """Vale e uma mineradora multinacional brasileira.
Tambem produz cobre e niquel.

Fundada em 1942 como Companhia Vale do Rio Doce.

== Historia ==

A empresa nasceu durante o governo Vargas.

=== Fundacao ===

Criada para explorar as minas de Itabira.

=== Privatizacao ===

Foi privatizada em 1997.

== Segmentos ==

Atua em logistica e energia.
"""


def test_hierarquia_de_secoes_vira_trilha():
    blocos = parse_blocks(TEXTO)
    trilhas = [b.section_path for b in blocos]
    assert "" in trilhas  # texto antes do primeiro heading
    assert "Historia" in trilhas
    assert "Historia > Fundacao" in trilhas  # subsecao herda a secao pai
    assert "Historia > Privatizacao" in trilhas
    assert "Segmentos" in trilhas


def test_subsecao_nao_vaza_para_a_secao_seguinte():
    """O bug classico: a pilha de secoes nao ser truncada ao subir de nivel."""
    trilhas = {b.section_path for b in parse_blocks(TEXTO)}
    assert "Segmentos" in trilhas
    assert not any(t.startswith("Historia > Privatizacao > ") for t in trilhas)
    assert "Historia > Privatizacao > Segmentos" not in trilhas


def test_nenhum_chunk_ultrapassa_o_limite():
    cfg = ChunkingConfig(max_chars=200, overlap_chars=30)
    for bloco in chunk_text(TEXTO * 5, cfg):
        assert len(bloco.text) <= cfg.max_chars


def test_nenhum_conteudo_e_perdido():
    """Toda palavra do original precisa sobreviver em algum chunk."""
    cfg = ChunkingConfig(max_chars=150, overlap_chars=0)
    juntos = " ".join(b.text for b in chunk_text(TEXTO, cfg))
    for palavra in ("Itabira", "1997", "Vargas", "logistica", "niquel"):
        assert palavra in juntos


def test_chunks_nao_cruzam_fronteira_de_secao():
    for bloco in chunk_text(TEXTO, ChunkingConfig(max_chars=2000)):
        assert " > Segmentos" not in bloco.section_path


def test_paragrafo_gigante_e_quebrado_por_sentenca():
    gigante = " ".join(f"Esta e a sentenca numero {i} do paragrafo." for i in range(80))
    cfg = ChunkingConfig(max_chars=300, overlap_chars=0)
    blocos = chunk_text(gigante, cfg)
    assert len(blocos) > 1
    assert all(len(b.text) <= cfg.max_chars for b in blocos)
    # Quebrou em fronteira de sentenca, nao no meio de uma palavra.
    assert all(b.text.strip().endswith(".") for b in blocos[:-1])


def test_sentenca_unica_maior_que_o_limite_e_cortada_a_forca():
    """Sem escolha: melhor cortar do que estourar o limite do modelo."""
    cfg = ChunkingConfig(max_chars=50, overlap_chars=0)
    blocos = chunk_text("x" * 300, cfg)
    assert len(blocos) >= 6
    assert all(len(b.text) <= cfg.max_chars for b in blocos)


def test_overlap_preserva_o_fim_do_chunk_anterior():
    texto = "\n\n".join(f"Paragrafo numero {i} com algum conteudo." for i in range(12))
    cfg = ChunkingConfig(max_chars=200, overlap_chars=40)
    blocos = chunk_text(texto, cfg)
    assert len(blocos) > 1
    cauda = blocos[0].text[-20:]
    assert cauda in blocos[1].text


def test_chunk_document_numera_em_ordem():
    doc = Document(source="teste", external_id="vale", title="Vale", content=TEXTO)
    chunks = chunk_document(doc)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.document_id == doc.id for c in chunks)


def test_documento_vazio_nao_gera_chunk():
    doc = Document(source="teste", external_id="vazio", title="Vazio", content="   \n\n  ")
    assert chunk_document(doc) == []


@pytest.mark.parametrize("max_chars", [120, 400, 1800])
def test_invariantes_valem_para_varios_tamanhos(max_chars):
    cfg = ChunkingConfig(max_chars=max_chars, overlap_chars=min(50, max_chars // 4))
    blocos = chunk_text(TEXTO * 3, cfg)
    assert blocos
    assert all(b.text.strip() for b in blocos)
    assert all(len(b.text) <= max_chars for b in blocos)
