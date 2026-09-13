"""Selecao do corpus da CVM.

O que faz este corpus valer o trabalho e a COOCORRENCIA: a empresa A publica o
fato relevante dela e a empresa B publica o dela, sobre a mesma operacao. Se a
selecao perde essa propriedade, o corpus vira uma pilha de documentos
auto-contidos -- exatamente o problema da Wikipedia que motivou a troca.

Os testes aqui cobrem os tres pontos onde ela pode se perder: a chave que
identifica a empresa dentro do texto de outra, o criterio de par, e a deteccao
de PDF (o servidor da CVM mente no `content-type`).
"""

import itertools
import zipfile

import httpx
import pytest

from ariadne.ingestion.cvm import (
    Registro,
    baixar_documento,
    baixar_metadados,
    escrever_manifesto,
    ler_manifesto,
    nome_curto,
    selecionar,
)

_seq = itertools.count(1)


def linha(empresa: str, assunto: str, categoria: str = "Fato Relevante") -> dict[str, str]:
    return {
        "Nome_Companhia": empresa,
        "Assunto": assunto,
        "Categoria": categoria,
        "Data_Entrega": "2025-03-01",
        "Protocolo_Entrega": f"P{next(_seq):05d}",
        "Link_Download": "https://cvm.example/doc",
    }


REGISTRO = Registro(
    empresa="PETRORECONCAVO S.A.",
    assunto="Aquisicao de ativos",
    categoria="Fato Relevante",
    data="2025-03-01",
    protocolo="P00001",
    url="https://cvm.example/doc",
)


# --- chave da empresa -------------------------------------------------------


@pytest.mark.parametrize(
    ("razao_social", "esperado"),
    [
        ("PETRORECONCAVO S.A.", "petroreconcavo"),
        ("3R Potiguar S.A.", "3r potiguar"),
        ("CSN MINERACAO S.A.", "csn mineracao"),
        ("Marfrig Global Foods S.A.", "marfrig"),
    ],
)
def test_nome_curto_reduz_ao_nucleo(razao_social, esperado):
    """O nucleo e o que aparece citado no texto do OUTRO lado da operacao.

    Ninguem escreve "Marfrig Global Foods S.A." no corpo de um fato relevante
    alheio; escreve "Marfrig". Casar pela razao social completa nao acharia
    citacao nenhuma.
    """
    assert nome_curto(razao_social) == esperado


def test_razao_social_feita_so_de_ruido_vira_chave_curta():
    """Banco do Brasil S.A. e inteiramente ruido -- sobra string vazia.

    Este e o motivo do filtro `len(k) >= 5` em `selecionar`: `"" in assunto` e
    verdadeiro para QUALQUER texto, entao uma chave vazia casaria com todo
    documento do corpus e criaria uma aresta entre cada par de empresas.
    """
    assert len(nome_curto("Banco do Brasil S.A.")) < 5


def test_chave_vazia_nao_vira_parceira_de_todo_mundo():
    """Regressao do caso acima, medida pela saida de `selecionar`."""
    selecao = selecionar(
        [
            linha("Banco do Brasil S.A.", "Aquisicao de participacao"),
            linha("PETRORECONCAVO S.A.", "Aquisicao de ativos da 3R Potiguar"),
            linha("3R Potiguar S.A.", "Venda de ativos de midstream"),
        ]
    )
    assert all(citada != "Banco do Brasil S.A." for _, citada in selecao.pares)


# --- selecao ----------------------------------------------------------------


def test_assunto_sem_operacao_fica_de_fora():
    """O corpus quer operacoes ENTRE empresas; calendario nao liga ninguem."""
    selecao = selecionar([linha("VALE S.A.", "Calendario anual de eventos corporativos")])
    assert selecao.registros == []


def test_categoria_fora_da_lista_fica_de_fora():
    selecao = selecionar(
        [linha("VALE S.A.", "Aquisicao de participacao", categoria="Formulario de Referencia")]
    )
    assert selecao.registros == []


def test_par_exige_publicacao_dos_dois_lados():
    """Citar nao basta: a citada precisa ter publicado tambem.

    Se so um lado publicou, nao existe o segundo documento para o grafo
    costurar -- e a pergunta multi-hop nao teria resposta.
    """
    so_um_lado = selecionar(
        [linha("PETRORECONCAVO S.A.", "Aquisicao de ativos da Fornecedora Obscura")]
    )
    assert so_um_lado.pares == []

    dois_lados = selecionar(
        [
            linha("PETRORECONCAVO S.A.", "Aquisicao de ativos da 3R Potiguar"),
            linha("3R Potiguar S.A.", "Venda de ativos de midstream"),
        ]
    )
    assert ("PETRORECONCAVO S.A.", "3R Potiguar S.A.") in dois_lados.pares


def test_documentos_dos_dois_lados_entram_no_corpus():
    """Trazer so quem citou deixaria metade da operacao fora do indice."""
    selecao = selecionar(
        [
            linha("PETRORECONCAVO S.A.", "Aquisicao de ativos da 3R Potiguar"),
            linha("3R Potiguar S.A.", "Venda de ativos de midstream"),
        ]
    )
    assert {r.empresa for r in selecao.registros} == {
        "PETRORECONCAVO S.A.",
        "3R Potiguar S.A.",
    }


def test_protocolo_nao_se_repete_entre_pares():
    """Um documento que cita DUAS empresas e candidato em dois pares.

    Sem o controle de vistos ele seria baixado e ingerido em duplicata -- e
    trecho duplicado infla o recall sem recuperar nada novo.
    """
    selecao = selecionar(
        [
            linha("CSN MINERACAO S.A.", "Aquisicao de participacao na MRS Logistica e na Rumo"),
            linha("MRS Logistica S.A.", "Alteracao de participacao acionaria"),
            linha("Rumo S.A.", "Incorporacao de acoes"),
        ]
    )
    protocolos = [r.protocolo for r in selecao.registros]
    assert len(protocolos) == len(set(protocolos))


def test_limite_por_empresa_e_respeitado():
    """Uma empresa falante nao pode dominar o corpus inteiro."""
    selecao = selecionar(
        [
            linha("CSN MINERACAO S.A.", "Aquisicao de participacao na MRS Logistica"),
            linha("CSN MINERACAO S.A.", "Venda de participacao na MRS Logistica"),
            linha("CSN MINERACAO S.A.", "Acordo com a MRS Logistica"),
            linha("MRS Logistica S.A.", "Alteracao de participacao acionaria"),
        ],
        max_por_empresa=1,
    )
    assert sum(1 for r in selecao.registros if r.empresa == "CSN MINERACAO S.A.") == 1


# --- metadados --------------------------------------------------------------


def test_csv_do_zip_e_lido_em_latin1(tmp_path):
    """A CVM publica em latin-1 com separador `;`.

    Ler como utf-8 estouraria em qualquer razao social acentuada -- ou seja, na
    primeira siderurgica. O teste tambem cobre o cache: o zip ja existe em
    disco, entao nenhuma requisicao e feita.
    """
    bruto = (
        "Nome_Companhia;Assunto;Categoria;Data_Entrega;Protocolo_Entrega;Link_Download\r\n"
        "COMPANHIA SIDERÚRGICA NACIONAL;Aquisição;Fato Relevante;"
        "2025-03-01;P1;http://x\r\n"
    ).encode("latin-1")
    zip_path = tmp_path / "ipe_2025.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("ipe_cia_aberta_2025.csv", bruto)

    linhas = baixar_metadados(2025, zip_path)
    assert linhas[0]["Nome_Companhia"] == "COMPANHIA SIDERÚRGICA NACIONAL"
    assert linhas[0]["Assunto"] == "Aquisição"


# --- manifesto --------------------------------------------------------------


def test_manifesto_reproduz_a_mesma_selecao(tmp_path):
    """O manifesto e o que torna a tabela do README reproduzivel.

    Os PDFs sao da CVM e nao entram no repositorio; rodar a selecao de novo da
    outro corpus porque o CSV do ano corrente cresce toda semana. Quem clona
    precisa remontar ESTES documentos, nao documentos parecidos.
    """
    linhas = [
        linha("PETRORECONCAVO S.A.", "Aquisicao de ativos da 3R Potiguar"),
        linha("3R Potiguar S.A.", "Venda de ativos de midstream"),
        linha("VALE S.A.", "Calendario anual de eventos"),
    ]
    selecao = selecionar(linhas)
    caminho = tmp_path / "corpus.json"
    escrever_manifesto(selecao, caminho, 2025)

    remontado = ler_manifesto(caminho, linhas)
    assert [r.protocolo for r in remontado] == [r.protocolo for r in selecao.registros]


def test_manifesto_ignora_documento_retirado_do_ar(tmp_path):
    """Companhia pode retirar um documento; o corpus segue com um a menos.

    Abortar a coleta inteira por causa de um protocolo ausente deixaria o
    projeto irreproduzivel pelo motivo oposto ao que o manifesto resolve.
    """
    linhas = [
        linha("PETRORECONCAVO S.A.", "Aquisicao de ativos da 3R Potiguar"),
        linha("3R Potiguar S.A.", "Venda de ativos de midstream"),
    ]
    selecao = selecionar(linhas)
    caminho = tmp_path / "corpus.json"
    escrever_manifesto(selecao, caminho, 2025)

    sobrou = ler_manifesto(caminho, linhas[:1])
    assert len(sobrou) < len(selecao.registros)
    assert all(r.protocolo == linhas[0]["Protocolo_Entrega"] for r in sobrou)


# --- download ---------------------------------------------------------------


def cliente(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_pdf_com_content_type_mentiroso_e_aceito(tmp_path):
    """O servidor da CVM devolve `text/html` para PDF. Os bytes e que mandam."""

    def handler(request):
        return httpx.Response(
            200,
            content=b"%PDF-1.7\n conteudo do fato relevante",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    with cliente(handler) as http:
        destino = baixar_documento(REGISTRO, tmp_path / "a.pdf", http)

    assert destino is not None
    assert destino.read_bytes().startswith(b"%PDF")


def test_pagina_de_erro_anunciada_como_pdf_e_recusada(tmp_path):
    """O espelho do teste anterior: confiar no header gravaria HTML como .pdf.

    E o parser nao falharia alto -- extrairia o texto da pagina de erro e o
    indexaria como se fosse o documento.
    """

    def handler(request):
        return httpx.Response(
            200,
            content=b"<html><body>Documento nao disponivel</body></html>",
            headers={"content-type": "application/pdf"},
        )

    alvo = tmp_path / "b.pdf"
    with cliente(handler) as http:
        assert baixar_documento(REGISTRO, alvo, http) is None
    assert not alvo.exists()


def test_erro_http_nao_derruba_a_coleta(tmp_path):
    """60 downloads em sequencia: um 500 nao pode abortar os outros 59."""

    def handler(request):
        return httpx.Response(500)

    with cliente(handler) as http:
        assert baixar_documento(REGISTRO, tmp_path / "c.pdf", http) is None


def test_documento_ja_baixado_nao_refaz_requisicao(tmp_path):
    """Retomar coleta interrompida nao deve rebaixar o que ja esta em disco."""
    chamadas: list[str] = []

    def handler(request):
        chamadas.append(str(request.url))
        return httpx.Response(200, content=b"%PDF-1.7\n novo")

    ja_existe = tmp_path / "d.pdf"
    ja_existe.write_bytes(b"%PDF-1.7\n antigo")

    with cliente(handler) as http:
        destino = baixar_documento(REGISTRO, ja_existe, http)

    assert chamadas == []
    assert destino is not None
    assert b"antigo" in destino.read_bytes()
