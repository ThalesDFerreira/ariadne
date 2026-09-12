"""Extracao de entidades e relacoes com LLM local.

Structured output em vez de "peca JSON no prompt e reze": o Ollama aceita um
JSON Schema e restringe a decodificacao a ele, entao o modelo nao consegue
devolver texto solto nem campo faltando. O que ainda pode vir errado e o
CONTEUDO -- tipo trocado, entidade inventada --, nao o formato.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from ariadne.config import Settings, get_settings
from ariadne.domain.graph import Extraction

_PROMPT = """Voce extrai entidades e relacoes de textos em portugues sobre empresas brasileiras.

TIPOS DE RELACAO (preste atencao na DIRECAO -- origem primeiro, destino depois):

- CONTROLA: origem e dona/acionista da destino.
  "A Cosan controla a Comgas"            -> Cosan CONTROLA Comgas
  "X e controlada pelo Governo Y"        -> Governo Y CONTROLA X   (inverta!)
- ADQUIRIU: origem comprou a destino. Use para "comprou", "adquiriu", "aquisicao de".
  "A WEG comprou a Katt"                 -> WEG ADQUIRIU Katt
- FORNECE_PARA: origem vende insumo/produto para a destino.
- CONCORRE_COM: as duas disputam o mesmo mercado.
- ATUA_EM: origem opera num SETOR, MERCADO ou REGIAO. O destino NUNCA e uma empresa.
  "A Copel atua no Parana"               -> Copel ATUA_EM Parana
- PRODUZ: origem fabrica/extrai o produto destino.
- LOCALIZADA_EM: sede ou instalacao da origem fica no lugar destino.
- TRABALHA_EM: pessoa origem trabalha/trabalhou na organizacao destino.
- PARTICIPOU_DE: origem tomou parte no evento destino.
- RELACIONADA_A: use so quando a relacao e real mas nao cabe em nenhuma acima.

REGRAS:
- Extraia SOMENTE o que o texto afirma. Nao use conhecimento externo.
- Use o nome EXATAMENTE como aparece no texto. Se o texto diz "Cemig", escreva
  "Cemig" -- nao expanda para o nome completo, nem abrevie.
- Origem e destino de toda relacao precisam estar na lista de entidades.
- Em `evidence`, copie o trecho curto do texto que sustenta a relacao.
- Ser fundada por alguem NAO e CONTROLA nem FORNECE_PARA.
- Se o texto nao afirma relacao nenhuma, devolva lista vazia. Lista vazia e
  resposta melhor do que relacao inventada.

Contexto do trecho: {context}

TEXTO:
{text}
"""


class ExtractionProvider(Protocol):
    def extract(self, text: str, *, context: str = "") -> Extraction: ...


def chunk_fingerprint(text: str, model: str, prompt_version: str) -> str:
    """Identidade do trabalho de extracao, nao so do texto.

    Inclui modelo e versao do prompt porque trocar qualquer um dos dois muda o
    resultado: cachear so pelo texto devolveria silenciosamente a extracao de
    um modelo antigo depois de uma troca.
    """
    material = f"{prompt_version}|{model}|{text}".encode()
    return hashlib.sha256(material).hexdigest()


class OllamaExtractor:
    """Extrator via Ollama, com schema estrito."""

    PROMPT_VERSION = "v2"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.model = model or cfg.extraction_model
        self._client = client or httpx.Client(
            base_url=cfg.ollama_base_url,
            # 120s, nao 300s: um punhado de chunks longos estoura o tempo de
            # qualquer forma, e esperar 5 minutos por cada um deles antes de
            # desistir custa mais que o pouco que eles renderiam.
            timeout=httpx.Timeout(120.0),
        )

    @property
    def fingerprint_model(self) -> str:
        return self.model

    def extract(self, text: str, *, context: str = "") -> Extraction:
        prompt = _PROMPT.format(context=context or "(sem contexto)", text=text)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": Extraction.model_json_schema(),
            # Temperatura zero: extracao e tarefa determinista. Criatividade
            # aqui significa inventar relacao que o texto nao afirma.
            "options": {
                "temperature": 0,
                # Teto de saida: sem ele, um chunk com dezenas de entidades
                # gera uma resposta interminavel e estoura o timeout.
                "num_predict": 1536,
            },
        }
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        content = response.json()["message"]["content"]

        try:
            extraction = Extraction.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError):
            # Mesmo com schema, um modelo fraco as vezes devolve algo
            # invalido. Um chunk perdido e melhor que a ingestao inteira caida.
            return Extraction(entities=[], relations=[])

        return extraction.drop_dangling_relations()

    def close(self) -> None:
        self._client.close()
