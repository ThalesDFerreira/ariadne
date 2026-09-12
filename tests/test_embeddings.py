"""Embeddings contra o Ollama real.

Pula sozinho quando o Ollama nao esta no ar, pela mesma razao dos testes de
banco: a suite precisa continuar util numa maquina sem tudo ligado.
"""

import httpx
import pytest

from ariadne.config import get_settings
from ariadne.llm.embeddings import OllamaEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    cfg = get_settings()
    try:
        httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=3.0).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Ollama indisponivel em {cfg.ollama_base_url} ({exc})")
    return OllamaEmbedder(cfg)


def test_dimensao_bate_com_o_schema(embedder):
    vetor = embedder.embed_query("teste")
    assert len(vetor) == embedder.dimensions == 1024


def test_lote_preserva_a_ordem(embedder):
    textos = ["mineração de ferro", "banco de investimento", "aviação comercial"]
    vetores = embedder.embed_documents(textos)
    assert len(vetores) == 3
    # Cada texto tem que voltar no seu lugar: se o lote embaralhar, todo chunk
    # fica com o embedding do vizinho e a busca inteira vira ruido.
    sozinhos = [embedder.embed_query(t) for t in textos]
    for lote, individual in zip(vetores, sozinhos, strict=True):
        assert _cosseno(lote, individual) > 0.99


def test_lista_vazia_nao_chama_o_modelo(embedder):
    assert embedder.embed_documents([]) == []


def test_proximidade_reflete_significado_e_nao_palavras(embedder):
    """O ponto do embedding: aproximar sentido, nao vocabulario comum."""
    a = embedder.embed_query("A empresa teve queda no faturamento")
    b = embedder.embed_query("A receita da companhia recuou")  # zero palavras em comum
    c = embedder.embed_query("Receita de bolo de chocolate")  # compartilha "receita"

    assert _cosseno(a, b) > _cosseno(a, c)


def _cosseno(u: list[float], v: list[float]) -> float:
    num = sum(x * y for x, y in zip(u, v, strict=True))
    nu = sum(x * x for x in u) ** 0.5
    nv = sum(y * y for y in v) ** 0.5
    return num / (nu * nv)
