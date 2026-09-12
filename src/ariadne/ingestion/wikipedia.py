"""Fonte de corpus: Wikipedia em portugues.

Escolhida para a Fase 1 por tres motivos praticos: o texto ja vem limpo (sem
PDF no caminho), o tamanho cabe no orcamento de VRAM da maquina, e -- o que
mais importa -- os links internos entre paginas formam um grafo de referencia
gratuito. Na Fase 2 dara para MEDIR a extracao do LLM contra ele, em vez de
julgar no olho.
"""

from __future__ import annotations

from typing import Any

import httpx

from ariadne.domain.models import Document

SOURCE = "wikipedia-pt"
_API = "https://pt.wikipedia.org/w/api.php"
# A Wikipedia exige User-Agent identificavel e bloqueia clientes anonimos.
_USER_AGENT = "Ariadne/0.1 (projeto de estudo; https://github.com/ThalesDFerreira/ariadne)"


class WikipediaSource:
    """Busca paginas da Wikipedia e as devolve como Documentos."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
        )

    def fetch(self, title: str) -> Document | None:
        """Traz uma pagina. Devolve None se ela nao existir."""
        text = self._extract(title)
        if not text:
            return None
        links = self._links(title)
        canonical = self._canonical_title(title)
        return Document(
            source=SOURCE,
            external_id=canonical,
            title=canonical,
            url=f"https://pt.wikipedia.org/wiki/{canonical.replace(' ', '_')}",
            content=text,
            metadata={"links": links},
        )

    def fetch_many(self, titles: list[str]) -> list[Document]:
        docs: list[Document] = []
        for title in titles:
            doc = self.fetch(title)
            if doc is not None:
                docs.append(doc)
        return docs

    def _query(self, params: dict[str, str]) -> dict[str, Any]:
        response = self._client.get(
            _API, params={"action": "query", "format": "json", "redirects": "1", **params}
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        query: dict[str, Any] = data.get("query", {})
        return query

    def _extract(self, title: str) -> str:
        # Um titulo por chamada: com explaintext a API so preenche o extract da
        # primeira pagina, e pedir em lote devolve o resto vazio em silencio.
        pages = self._query({"prop": "extracts", "explaintext": "1", "titles": title})
        for page in pages.get("pages", {}).values():
            extract: str = page.get("extract", "")
            return extract
        return ""

    def _canonical_title(self, title: str) -> str:
        pages = self._query({"prop": "info", "titles": title})
        for page in pages.get("pages", {}).values():
            resolved: str = page.get("title", title)
            return resolved
        return title

    def _links(self, title: str) -> list[str]:
        """Links para outros artigos (namespace 0).

        Vem com ruido -- datas e termos genericos como "1953" entram junto --,
        mas a filtragem fica para a Fase 2, quando soubermos o que conta como
        entidade. Guardar cru agora evita ter que rebaixar tudo depois.
        """
        out: list[str] = []
        params = {"prop": "links", "pllimit": "500", "plnamespace": "0", "titles": title}
        pages = self._query(params)
        for page in pages.get("pages", {}).values():
            out.extend(link["title"] for link in page.get("links", []))
        return out

    def close(self) -> None:
        self._client.close()
