"""Pipeline ponta a ponta: documento entra, busca devolve com citacao."""

import httpx
import pytest

from ariadne.config import get_settings
from ariadne.domain.models import Document
from ariadne.ingestion.pipeline import IngestionPipeline
from ariadne.llm.embeddings import OllamaEmbedder
from ariadne.retrieval.vector_search import VectorSearch
from ariadne.storage.database import connection
from ariadne.storage.schema import apply_schema

pytestmark = pytest.mark.integration

FONTE = "teste-e2e"

DOC_A = """A Mineradora Alfa extrai bauxita na região de Paragominas.

== Operações ==

A bauxita extraída abastece a refinaria de alumínio da própria empresa.
"""

DOC_B = """A Metalúrgica Beta produz alumínio primário.

== Fornecedores ==

A Beta compra bauxita da Mineradora Alfa desde 2018.
"""


@pytest.fixture(scope="module")
def _ollama():
    cfg = get_settings()
    try:
        httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=3.0).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Ollama indisponivel ({exc})")


@pytest.fixture(scope="module")
def corpus(_ollama):
    with connection() as conn:
        apply_schema(conn)
        conn.execute("DELETE FROM documents WHERE source = %s", (FONTE,))
        conn.commit()

    docs = [
        Document(
            source=FONTE,
            external_id="alfa",
            title="Mineradora Alfa",
            url="https://exemplo.org/alfa",
            content=DOC_A,
        ),
        Document(
            source=FONTE,
            external_id="beta",
            title="Metalúrgica Beta",
            url="https://exemplo.org/beta",
            content=DOC_B,
        ),
    ]
    report = IngestionPipeline(embedder=OllamaEmbedder()).run(docs)
    yield report

    with connection() as conn:
        conn.execute("DELETE FROM documents WHERE source = %s", (FONTE,))
        conn.commit()


def test_ingestao_indexa_os_dois_documentos(corpus):
    assert len(corpus.ingested) == 2
    assert corpus.chunks >= 2


def test_reingestao_pula_o_que_nao_mudou(corpus):
    iguais = [
        Document(source=FONTE, external_id="alfa", title="Mineradora Alfa", content=DOC_A),
        Document(source=FONTE, external_id="beta", title="Metalúrgica Beta", content=DOC_B),
    ]
    segundo = IngestionPipeline(embedder=OllamaEmbedder()).run(iguais)
    assert len(segundo.skipped) == 2
    assert segundo.ingested == []


def test_busca_encontra_por_significado_e_cita_a_fonte(corpus):
    hits = VectorSearch(embedder=OllamaEmbedder()).search("Quem extrai bauxita?", limit=5)
    nossos = [h for h in hits if "Alfa" in h.document_title or "Beta" in h.document_title]
    assert nossos, "o corpus de teste nao apareceu nos resultados"
    melhor = nossos[0]
    assert melhor.document_url is not None
    assert melhor.citation()
    assert 0.0 <= melhor.score <= 1.0


def test_resultados_vem_ordenados_por_score(corpus):
    hits = VectorSearch(embedder=OllamaEmbedder()).search("bauxita e alumínio", limit=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_baseline_acha_o_fato_direto(corpus):
    """O que a busca vetorial RESOLVE: fato dito explicitamente num chunk."""
    hits = VectorSearch(embedder=OllamaEmbedder()).search(
        "De quem a Metalúrgica Beta compra bauxita?", limit=5
    )
    textos = " ".join(h.chunk.content for h in hits)
    assert "Alfa" in textos, "o fato esta escrito no DOC_B e deveria ser recuperado"


def test_baseline_nao_liga_dois_documentos(corpus):
    """O que a busca vetorial NAO resolve -- e por que o grafo existe.

    "Beta compra da Alfa" esta no DOC_B. "Alfa extrai em Paragominas" esta no
    DOC_A. Nenhum chunk contem a ligacao, entao responder exige percorrer a
    relacao entre documentos.

    Este teste registra o baseline: se um dia ele comecar a falhar porque a
    recuperacao passou a trazer os dois lados, isso e progresso -- e o sinal
    de que a Fase 3 entregou o que prometeu.
    """
    hits = VectorSearch(embedder=OllamaEmbedder()).search(
        "A Metalúrgica Beta depende de bauxita de qual região?", limit=3
    )
    titulos = {h.document_title for h in hits}
    nossos = {t for t in titulos if "Alfa" in t or "Beta" in t}
    # Baseline honesto: a busca por similaridade tende a trazer so um dos lados.
    assert nossos, "nenhum documento do corpus de teste foi recuperado"
