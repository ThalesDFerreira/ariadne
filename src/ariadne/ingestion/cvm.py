"""Corpus da CVM: fatos relevantes e comunicados ao mercado.

POR QUE ESTE CORPUS SUBSTITUIU A WIKIPEDIA
------------------------------------------
A Fase 1 escolheu a Wikipedia argumentando que os links internos dariam um
grafo de referencia. O argumento estava certo sobre os LINKS e errado sobre o
que importa: artigo de enciclopedia e AUTO-CONTIDO. A pagina da Klabin ja diz
onde fica a fabrica; a da Usiminas ja diz quem e o socio japones. Nao ha
informacao dividida entre documentos, e sem isso o grafo nao tem o que costurar.

A medicao confirmou: de 22 perguntas escritas para exigir dois documentos, 17
foram descartadas porque a resposta cabia num trecho so. E o RAG puro alcancou
94% de recall justamente porque quase nada exigia atravessar documentos.

Os documentos da CVM tem a propriedade que faltava. Numa fusao, a empresa A
publica o fato relevante DELA e a empresa B publica o DELA -- mesma operacao,
documentos separados, cada um com a metade que o outro nao conta. Quem quiser
a operacao inteira precisa ligar os dois.

FONTE
-----
Portal de dados abertos da CVM: um CSV por ano com os metadados de todos os
documentos entregues, e um link de download por registro.

    https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/

Os documentos sao PDF, mas o servidor responde `content-type: text/html` --
por isso o formato e detectado pelos BYTES, nunca pelo header.
"""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DADOS_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{ano}.zip"
CATEGORIAS = ("Fato Relevante", "Comunicado ao Mercado")

# Assuntos que indicam operacao ENTRE empresas -- e o que gera documento
# complementar do outro lado.
_OPERACAO = re.compile(
    r"aquisi|incorpora|fus[ãa]o|combina|participa|acordo|venda|compra|"
    r"joint|cis[ãa]o|desinvestimento|oferta",
    re.IGNORECASE,
)

# Palavras que nao ajudam a identificar a empresa dentro de um texto.
#
# As bordas sao lookarounds, nao `\b`, por um motivo medido. Com `\b(s\.?a\.?)\b`
# a razao social "PETRORECONCAVO S.A." virava a chave "petroreconcavo ." -- o
# `\b` final impede consumir o ultimo ponto, entao ele sobrava colado. E chave
# com ponto nunca aparece no texto de outro documento, entao TODA empresa
# escrita com "S.A." (com ponto final) ficava invisivel como citada. Os 14 pares
# encontrados antes da correcao tinham, sem excecao, "SA" ou "S/A" do lado
# citado. Corrigido: 14 -> 16 pares, 45 -> 60 documentos.
_RUIDO = re.compile(
    r"(?<!\w)(s\.?a\.?|sa|ltda|holding|participacoes|participações|global|foods|"
    r"brasil|do brasil|banco|cia|companhia|energia|s/a)(?!\w)",
    re.IGNORECASE,
)


def fold(texto: str) -> str:
    base = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in base if not unicodedata.combining(c)).lower()


def nome_curto(razao_social: str) -> str:
    """Reduz a razao social ao nucleo que aparece citado em outros documentos."""
    limpo = _RUIDO.sub(" ", fold(razao_social))
    # `.strip(" .,-/")` tira pontuacao orfa das PONTAS. No meio ela fica: a
    # chave precisa casar com o texto, e "energetica de brasilia - ceb" e
    # escrito com o hifen mesmo.
    return re.sub(r"\s+", " ", limpo).strip(" .,-/")


@dataclass
class Registro:
    empresa: str
    assunto: str
    categoria: str
    data: str
    protocolo: str
    url: str

    @property
    def titulo(self) -> str:
        return f"{self.empresa} - {self.assunto}"[:180]


@dataclass
class Selecao:
    registros: list[Registro] = field(default_factory=list)
    pares: list[tuple[str, str]] = field(default_factory=list)
    """Empresas que publicaram sobre a mesma operacao."""

    def summary(self) -> str:
        return (
            f"{len(self.registros)} documento(s) de "
            f"{len({r.empresa for r in self.registros})} empresa(s), "
            f"{len(self.pares)} par(es) com publicacao dos dois lados"
        )


def baixar_metadados(ano: int, destino: Path) -> list[dict[str, str]]:
    """Baixa e le o CSV anual de metadados.

    O arquivo tem dezenas de milhares de linhas e ~2 MB compactado, entao e
    cacheado em disco: reprocessar a selecao nao deve rebaixar tudo.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        resposta = httpx.get(DADOS_URL.format(ano=ano), timeout=180.0, follow_redirects=True)
        resposta.raise_for_status()
        destino.write_bytes(resposta.content)

    with zipfile.ZipFile(destino) as z:
        bruto = z.read(z.namelist()[0]).decode("latin-1")
    return list(csv.DictReader(io.StringIO(bruto), delimiter=";"))


def _registro(linha: dict[str, str]) -> Registro:
    return Registro(
        empresa=linha["Nome_Companhia"].strip(),
        assunto=(linha["Assunto"] or "").strip(),
        categoria=linha["Categoria"],
        data=linha["Data_Entrega"],
        protocolo=linha["Protocolo_Entrega"],
        url=linha["Link_Download"],
    )


def selecionar(
    linhas: list[dict[str, str]],
    *,
    max_documentos: int = 60,
    max_por_empresa: int = 4,
) -> Selecao:
    """Escolhe documentos que se referenciam entre empresas.

    O criterio nao e "documentos interessantes", e sim documentos em que UMA
    empresa cita OUTRA que tambem publicou. E essa coocorrencia que cria a
    ligacao que nenhum documento isolado contem -- exatamente o que faltava no
    corpus anterior.
    """
    fatos = [
        _registro(linha)
        for linha in linhas
        if linha["Categoria"] in CATEGORIAS and _OPERACAO.search(linha["Assunto"] or "")
    ]

    emissoras = {nome_curto(r.empresa): r.empresa for r in fatos}
    emissoras = {k: v for k, v in emissoras.items() if len(k) >= 5}

    citacoes: dict[tuple[str, str], list[Registro]] = defaultdict(list)
    for registro in fatos:
        assunto = fold(registro.assunto)
        eu = nome_curto(registro.empresa)
        for chave, nome in emissoras.items():
            if chave != eu and chave in assunto:
                citacoes[(registro.empresa, nome)].append(registro)

    escolhidos: list[Registro] = []
    pares: list[tuple[str, str]] = []
    por_empresa: dict[str, int] = defaultdict(int)
    vistos: set[str] = set()

    # Pares mais documentados primeiro: quanto mais os dois lados publicaram,
    # mais rica a ligacao que o grafo vai poder montar.
    for (a, b), registros in sorted(citacoes.items(), key=lambda kv: -len(kv[1])):
        if len(escolhidos) >= max_documentos:
            break
        pares.append((a, b))
        # Documentos dos DOIS lados da operacao, nao so de quem citou.
        candidatos = registros + [r for r in fatos if r.empresa == b][:max_por_empresa]
        for registro in candidatos:
            if registro.protocolo in vistos:
                continue
            if por_empresa[registro.empresa] >= max_por_empresa:
                continue
            vistos.add(registro.protocolo)
            por_empresa[registro.empresa] += 1
            escolhidos.append(registro)
            if len(escolhidos) >= max_documentos:
                break

    return Selecao(registros=escolhidos, pares=pares)


def escrever_manifesto(selecao: Selecao, caminho: Path, ano: int) -> None:
    """Fixa em disco QUAIS documentos formam o corpus.

    Os PDFs sao da CVM e nao entram no repositorio, entao a promessa de
    "reproduza a tabela" depende de o clone conseguir remontar o MESMO corpus.
    Nao consegue rodando a selecao de novo: ela le o CSV vivo do ano corrente,
    que ganha linhas toda semana, e o proprio criterio de par ja mudou uma vez
    (a correcao do sufixo da razao social passou de 45 para 60 documentos).

    O manifesto e a lista de protocolos. Pequeno, versionavel, e suficiente
    para reconstruir o corpus byte a byte a partir do servidor da CVM.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conteudo = {
        "ano": ano,
        "documentos": [
            {"protocolo": r.protocolo, "empresa": r.empresa, "data": r.data}
            for r in selecao.registros
        ],
    }
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2), encoding="utf-8")


def ler_manifesto(caminho: Path, linhas: list[dict[str, str]]) -> list[Registro]:
    """Remonta os registros do manifesto, casando por protocolo no CSV do ano.

    Protocolo que sumiu do CSV (documento retirado pela companhia) e ignorado
    com o corpus seguindo em frente: parar tudo por um documento a menos seria
    pior que um corpus com 59.
    """
    conteudo = json.loads(caminho.read_text(encoding="utf-8"))
    querido = [d["protocolo"] for d in conteudo["documentos"]]
    por_protocolo = {linha["Protocolo_Entrega"]: linha for linha in linhas}
    return [_registro(por_protocolo[p]) for p in querido if p in por_protocolo]


def baixar_documento(
    registro: Registro, destino: Path, client: httpx.Client | None = None
) -> Path | None:
    """Salva o documento em disco. Devolve None quando nao for PDF utilizavel.

    O servidor da CVM responde `content-type: text/html` mesmo para PDF, entao
    o formato e decidido pelos primeiros bytes. Confiar no header aqui gravaria
    paginas de erro com extensao .pdf.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        return destino

    http = client or httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        resposta = http.get(registro.url)
        resposta.raise_for_status()
        conteudo = resposta.content
    except httpx.HTTPError:
        return None
    finally:
        if client is None:
            http.close()

    if not conteudo.startswith(b"%PDF"):
        return None

    destino.write_bytes(conteudo)
    return destino
