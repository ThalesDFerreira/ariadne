"""Armazenamento vetorial contra o Postgres real."""

import pytest

from ariadne.domain.models import Chunk, Document
from ariadne.storage.schema import apply_schema
from ariadne.storage.vector_store import PgVectorStore, to_pgvector

pytestmark = pytest.mark.integration

DIM = 1024


def _vec(seed: float) -> list[float]:
    """Vetor deterministico: o primeiro eixo carrega a identidade."""
    v = [0.0] * DIM
    v[0] = seed
    v[1] = 1.0 - abs(seed)
    return v


@pytest.fixture
def store(db):
    apply_schema(db)
    db.execute("DELETE FROM documents WHERE source = 'teste'")
    return PgVectorStore()


def test_vetor_sobrevive_a_ida_e_volta_ao_banco(db):
    """O que importa e o valor voltar fiel, nao o formato do texto.

    O pgvector aceita notacao cientifica ('1e-07'), entao serializar com repr
    esta correto -- o risco real seria perder precisao no caminho.
    """
    original = [1e-07, -0.5, 2.0]
    (devolvido,) = db.execute("SELECT %s::vector(3)", (to_pgvector(original),)).fetchone()
    numeros = [float(x) for x in devolvido.strip("[]").split(",")]
    assert numeros == pytest.approx(original, rel=1e-6)


def test_documento_novo_e_reportado_como_alterado(store, db):
    doc = Document(source="teste", external_id="a", title="A", content="conteudo")
    assert store.upsert_document(db, doc) is True


def test_reingerir_conteudo_identico_nao_reprocessa(store, db):
    """O hash e o que evita pagar LLM de novo pelo mesmo texto na Fase 2."""
    doc = Document(source="teste", external_id="a", title="A", content="conteudo")
    store.upsert_document(db, doc)
    igual = Document(source="teste", external_id="a", title="A", content="conteudo")
    assert store.upsert_document(db, igual) is False


def test_conteudo_alterado_volta_a_ser_processado(store, db):
    doc = Document(source="teste", external_id="a", title="A", content="antes")
    store.upsert_document(db, doc)
    mudou = Document(source="teste", external_id="a", title="A", content="depois")
    assert store.upsert_document(db, mudou) is True


def test_reingestao_atualiza_em_vez_de_duplicar(store, db):
    for texto in ("v1", "v2", "v3"):
        store.upsert_document(
            db, Document(source="teste", external_id="a", title="A", content=texto)
        )
    (n,) = db.execute(
        "SELECT count(*) FROM documents WHERE source='teste' AND external_id='a'"
    ).fetchone()
    assert n == 1


def test_busca_devolve_o_vizinho_mais_proximo_com_citacao(store, db):
    doc = Document(
        source="teste",
        external_id="b",
        title="Documento B",
        url="https://exemplo.org/b",
        content="irrelevante",
    )
    store.upsert_document(db, doc)
    doc_id = store.resolve_document_id(db, "teste", "b")

    chunks = [
        Chunk(document_id=doc_id, ordinal=0, content="alvo", section_path="Historia"),
        Chunk(document_id=doc_id, ordinal=1, content="distante"),
    ]
    store.replace_chunks(db, doc_id, chunks, [_vec(1.0), _vec(-1.0)])

    hits = store.search(db, _vec(1.0), limit=2)
    assert hits[0].chunk.content == "alvo"
    assert hits[0].score > hits[1].score
    # Regra de ouro: resultado sem procedencia nao serve.
    assert "Documento B" in hits[0].citation()
    assert "Historia" in hits[0].citation()
    assert "https://exemplo.org/b" in hits[0].citation()


def test_replace_chunks_nao_deixa_orfaos(store, db):
    """Rechunkar precisa apagar os chunks antigos, nao somar aos novos."""
    doc = Document(source="teste", external_id="c", title="C", content="x")
    store.upsert_document(db, doc)
    doc_id = store.resolve_document_id(db, "teste", "c")

    antigos = [Chunk(document_id=doc_id, ordinal=i, content=f"velho {i}") for i in range(5)]
    store.replace_chunks(db, doc_id, antigos, [_vec(0.1)] * 5)

    novos = [Chunk(document_id=doc_id, ordinal=i, content=f"novo {i}") for i in range(2)]
    store.replace_chunks(db, doc_id, novos, [_vec(0.2)] * 2)

    (n,) = db.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (doc_id,)).fetchone()
    assert n == 2


def test_contagem_incompativel_falha_antes_de_gravar(store, db):
    doc = Document(source="teste", external_id="d", title="D", content="x")
    store.upsert_document(db, doc)
    doc_id = store.resolve_document_id(db, "teste", "d")
    with pytest.raises(ValueError, match="embeddings"):
        store.replace_chunks(db, doc_id, [Chunk(document_id=doc_id, ordinal=0, content="a")], [])


def test_apagar_documento_leva_os_chunks_junto(store, db):
    doc = Document(source="teste", external_id="e", title="E", content="x")
    store.upsert_document(db, doc)
    doc_id = store.resolve_document_id(db, "teste", "e")
    store.replace_chunks(
        db, doc_id, [Chunk(document_id=doc_id, ordinal=0, content="a")], [_vec(0.5)]
    )

    db.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    (n,) = db.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (doc_id,)).fetchone()
    assert n == 0
