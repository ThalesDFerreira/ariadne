"""Extracao de texto por formato.

Cada formato falha de um jeito proprio, e o pior deles e falhar EM SILENCIO:
um PDF digitalizado devolve zero caractere sem erro nenhum, e o documento
entraria no indice vazio, sumindo da busca sem ninguem entender por que.
"""

import csv

import pytest

from ariadne.ingestion.parsers import SUPPORTED, ParsedDocument, parse_file


def escreve(tmp_path, nome: str, conteudo: str):
    caminho = tmp_path / nome
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


# --- texto ------------------------------------------------------------------


def test_markdown(tmp_path):
    p = escreve(tmp_path, "a.md", "# Titulo\n\nCorpo do texto.")
    r = parse_file(p)
    assert r.kind == "texto"
    assert "Corpo do texto" in r.text


def test_extensao_desconhecida_e_recusada(tmp_path):
    p = escreve(tmp_path, "a.exe", "binario")
    with pytest.raises(Exception, match="nao suportada"):
        parse_file(p)


def test_conjunto_de_extensoes_cobre_o_prometido():
    for esperada in (".pdf", ".docx", ".xlsx", ".csv", ".png", ".jpg", ".md"):
        assert esperada in SUPPORTED


# --- tabelas ----------------------------------------------------------------


def test_csv_vira_frases_com_nome_de_coluna(tmp_path):
    """ "1200" isolado nao diz nada a um embedding; "valor: 1200" diz."""
    p = escreve(tmp_path, "d.csv", "fornecedor,produto,valor\nAlfa,bauxita,1200\n")
    r = parse_file(p)
    assert r.kind == "planilha"
    assert "fornecedor: Alfa" in r.text
    assert "valor: 1200" in r.text


def test_csv_detecta_ponto_e_virgula(tmp_path):
    """Planilha exportada em portugues costuma usar ';'."""
    p = escreve(tmp_path, "d.csv", "a;b\n1;2\n")
    assert "a: 1" in parse_file(p).text


def test_tsv(tmp_path):
    p = escreve(tmp_path, "d.tsv", "a\tb\n1\t2\n")
    assert "a: 1" in parse_file(p).text


def test_csv_so_com_cabecalho_nao_quebra(tmp_path):
    p = escreve(tmp_path, "d.csv", "a,b\n")
    assert parse_file(p).text.strip() == ""


def test_celula_vazia_e_omitida(tmp_path):
    p = escreve(tmp_path, "d.csv", "a,b,c\n1,,3\n")
    texto = parse_file(p).text
    assert "a: 1" in texto and "c: 3" in texto
    assert "b:" not in texto


def test_coluna_sem_cabecalho_ganha_nome(tmp_path):
    caminho = tmp_path / "d.csv"
    with caminho.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", ""])
        w.writerow(["1", "2"])
    assert "coluna 2: 2" in parse_file(caminho).text


# --- docx -------------------------------------------------------------------


def test_docx_converte_titulo_em_secao(tmp_path):
    """Titulo de Word e estilo, nao marcacao.

    Sem converter, o documento inteiro vira um bloco unico e o chunking perde a
    hierarquia que o autor escreveu.
    """
    docx = pytest.importorskip("docx")
    caminho = tmp_path / "a.docx"
    doc = docx.Document()
    doc.add_heading("Contrato", 1)
    doc.add_paragraph("Primeira clausula.")
    doc.add_heading("Prazos", 2)
    doc.add_paragraph("Entrega mensal.")
    doc.save(str(caminho))

    r = parse_file(caminho)
    assert r.kind == "docx"
    assert "== Contrato ==" in r.text
    assert "=== Prazos ===" in r.text
    assert "Primeira clausula" in r.text


def test_docx_inclui_tabelas(tmp_path):
    docx = pytest.importorskip("docx")
    caminho = tmp_path / "t.docx"
    doc = docx.Document()
    tabela = doc.add_table(rows=1, cols=2)
    tabela.rows[0].cells[0].text = "Alfa"
    tabela.rows[0].cells[1].text = "bauxita"
    doc.save(str(caminho))
    assert "Alfa | bauxita" in parse_file(caminho).text


# --- planilha ---------------------------------------------------------------


def test_xlsx_usa_aba_como_secao(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    caminho = tmp_path / "p.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Fornecedores"
    ws.append(["empresa", "insumo"])
    ws.append(["Alfa", "bauxita"])
    wb.save(str(caminho))

    r = parse_file(caminho)
    assert "== Fornecedores ==" in r.text
    assert "empresa: Alfa" in r.text


# --- pdf --------------------------------------------------------------------


def test_pdf_com_texto_nativo(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    caminho = tmp_path / "a.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Texto nativo do PDF.", fontsize=12)
    doc.save(str(caminho))
    doc.close()

    r = parse_file(caminho)
    assert r.kind == "pdf"
    assert "Texto nativo" in r.text
    assert r.pages == 1


# --- contrato do resultado --------------------------------------------------


def test_extracao_por_ocr_vem_marcada():
    """Texto de OCR erra; quem le a resposta depois merece saber a origem."""
    r = ParsedDocument(text="x", kind="ocr", warning="veio de OCR")
    assert r.kind == "ocr"
    assert r.warning
