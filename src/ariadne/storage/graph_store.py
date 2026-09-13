"""Persistencia do grafo de conhecimento.

O Protocol e o que mantem a decisao do AGE reversivel: o motor de busca fala
com GraphStore, nunca com Cypher direto. Se o AGE quebrar num upgrade -- risco
real, e extensao menos madura que o Neo4j --, troca-se o adapter por tabelas
com CTE recursiva sem tocar em retrieval/.

Duas limitacoes do AGE encontradas na pratica, ambas contornadas aqui:

1. Nao suporta list comprehension de Cypher (`[n IN nodes(p) | n.name]`):
   devolve erro de sintaxe. Caminhos sao montados salto a salto.
2. Em `-[r:RELATES_TO*1..N]->`, `r` e uma LISTA de arestas, e indexa-la
   (`r[-1].evidence`) nao funciona. Por isso o salto simples usa aresta unica.
"""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

import psycopg
from pydantic import BaseModel

from ariadne.domain.graph import (
    EntityType,
    GraphEdge,
    GraphNode,
    RelationType,
    normalize_name,
)
from ariadne.storage.database import run_cypher


class Neighbor(BaseModel):
    """Um vizinho no grafo, com a aresta que levou ate ele."""

    name: str
    type: EntityType
    relation: RelationType
    direction: str
    """'saindo' quando a entidade consultada e a origem; 'entrando' caso contrario."""

    evidence: str = ""
    chunk_id: UUID | None = None


class PathStep(BaseModel):
    """Um passo do caminho.

    `source` e `target` seguem a ordem de LEITURA do caminho, que nem sempre e
    a direcao da aresta: numa busca nao dirigida, o passo pode ter sido
    percorrido ao contrario. Quem manda e a `evidence`, que traz a frase
    original e nao deixa duvida sobre quem faz o que.
    """

    source: str
    relation: RelationType
    target: str
    evidence: str = ""


class GraphStore(Protocol):
    def upsert_nodes(self, conn: psycopg.Connection[Any], nodes: list[GraphNode]) -> None: ...

    def upsert_edges(self, conn: psycopg.Connection[Any], edges: list[GraphEdge]) -> None: ...

    def neighbors(
        self, conn: psycopg.Connection[Any], key: str, *, depth: int = 1, limit: int = 50
    ) -> list[Neighbor]: ...

    def stats(self, conn: psycopg.Connection[Any]) -> dict[str, int]: ...


class AgeGraphStore:
    """Adapter sobre o Apache AGE.

    Todo valor vai por parametro agtype, nunca concatenado: os nomes vem de
    documentos, que sao input nao confiavel, e concatenar abriria injecao de
    Cypher.

    `graph_name` permite apontar para outro grafo. Os testes usam um grafo
    proprio: sem isso eles operavam sobre o grafo real, e um `clear()` de
    fixture apagaria o corpus inteiro.
    """

    def __init__(self, graph_name: str | None = None) -> None:
        self._graph = graph_name

    def upsert_nodes(self, conn: psycopg.Connection[Any], nodes: list[GraphNode]) -> None:
        for node in nodes:
            run_cypher(
                conn,
                """
                MERGE (n:Entity {key: $key})
                SET n.name = $name, n.type = $type, n.mentions = $mentions,
                    n.aliases = $aliases
                RETURN n
                """,
                graph_name=self._graph,
                params={
                    "key": node.key,
                    "name": node.name,
                    "type": node.type.value,
                    "mentions": node.mentions,
                    "aliases": ", ".join(node.aliases),
                },
            )

    def upsert_edges(self, conn: psycopg.Connection[Any], edges: list[GraphEdge]) -> None:
        """Grava as arestas, uma por (origem, destino, tipo, chunk de origem).

        Duas decisoes forcadas pelo AGE, ambas verificadas na pratica:

        1. O tipo vai como PROPRIEDADE, nao como rotulo da aresta: o AGE nao
           aceita rotulo parametrizado, e interpolar o tipo no Cypher reabriria
           o buraco de injecao que o params fecha.
        2. TODAS as propriedades vao dentro do MERGE. `SET` numa aresta recem
           criada por MERGE nao grava nada no AGE -- testado com propriedade
           unica, multiplas e `+=`, os tres silenciosamente sem efeito. A
           evidencia sumia, e aresta sem evidencia e afirmacao sem fonte.

        Consequencia de (2): como `chunk_id` entra na identidade do MERGE, a
        mesma relacao afirmada em dois trechos vira duas arestas paralelas.
        Isso e aceitavel, e ate util -- relacao corroborada por varias fontes
        fica visivel --, e mantem a operacao idempotente por trecho.
        """
        for edge in edges:
            run_cypher(
                conn,
                """
                MATCH (a:Entity {key: $source}), (b:Entity {key: $target})
                MERGE (a)-[r:RELATES_TO {
                    type: $type, chunk_id: $chunk_id, evidence: $evidence
                }]->(b)
                RETURN r
                """,
                graph_name=self._graph,
                params={
                    "source": edge.source_key,
                    "target": edge.target_key,
                    "type": edge.type.value,
                    "evidence": edge.evidence[:500],
                    "chunk_id": str(edge.chunk_id),
                },
            )

    def neighbors(
        self, conn: psycopg.Connection[Any], key: str, *, depth: int = 1, limit: int = 50
    ) -> list[Neighbor]:
        """Vizinhanca da entidade, nas duas direcoes."""
        depth = max(1, min(depth, 3))
        limit = max(1, min(limit, 200))
        out: list[Neighbor] = []

        padroes = (
            ("saindo", "(a:Entity {key: $key})-[r:RELATES_TO]->(b:Entity)"),
            ("entrando", "(a:Entity {key: $key})<-[r:RELATES_TO]-(b:Entity)"),
        )
        for direcao, padrao in padroes:
            rows = run_cypher(
                conn,
                f"MATCH {padrao} RETURN b.name, b.type, r.type, r.evidence LIMIT {limit}",
                graph_name=self._graph,
                params={"key": key},
                columns=4,
            )
            out.extend(
                Neighbor(
                    name=_agtext(row["c0"]),
                    type=_as_enum(EntityType, _agtext(row["c1"]), EntityType.OUTRO),
                    relation=_as_enum(RelationType, _agtext(row["c2"]), RelationType.RELACIONADA_A),
                    direction=direcao,
                    evidence=_agtext(row["c3"]),
                )
                for row in rows
                if _agtext(row["c0"])
            )

        if depth > 1:
            out.extend(self._far_neighbors(conn, key, depth, limit))
        return out

    def _far_neighbors(
        self, conn: psycopg.Connection[Any], key: str, depth: int, limit: int
    ) -> list[Neighbor]:
        """Vizinhos a mais de um salto.

        Vem sem evidencia: em caminho de comprimento variavel o AGE nao deixa
        acessar a aresta individual. Quem precisa da justificativa usa
        shortest_path, que monta o caminho salto a salto.
        """
        rows = run_cypher(
            conn,
            f"""
            MATCH (a:Entity {{key: $key}})-[:RELATES_TO*2..{depth}]->(b:Entity)
            RETURN DISTINCT b.name, b.type LIMIT {limit}
            """,
            graph_name=self._graph,
            params={"key": key},
            columns=2,
        )
        return [
            Neighbor(
                name=_agtext(row["c0"]),
                type=_as_enum(EntityType, _agtext(row["c1"]), EntityType.OUTRO),
                relation=RelationType.RELACIONADA_A,
                direction="saindo",
                evidence="",
            )
            for row in rows
            if _agtext(row["c0"])
        ]

    def shortest_path(
        self,
        conn: psycopg.Connection[Any],
        source: str,
        target: str,
        *,
        max_hops: int = 4,
        directed: bool = False,
        avoid_hub_types: tuple[EntityType, ...] = (EntityType.LUGAR,),
    ) -> list[PathStep]:
        """Caminho entre duas entidades, com a evidencia de cada aresta.

        Busca por comprimento crescente: o primeiro que casar e o mais curto.

        `directed=False` por padrao. Quem pergunta "qual a ligacao entre X e
        Y?" raramente quer uma direcao so -- a ligacao real entre Vale e BNDES
        passa por `BNDES CONTROLA CSN`, ou seja, chega pela ponta oposta.
        Exigir direcao unica devolvia "nenhum caminho" para pares claramente
        conectados.

        `avoid_hub_types` corta lugares como no INTERMEDIARIO. "As duas
        empresas ficam no Rio de Janeiro" e um caminho tecnicamente valido e
        informativamente vazio, e como quase toda empresa tem sede em algum
        lugar, esses hubs ligam praticamente qualquer par -- afogando as
        conexoes que de fato explicam alguma coisa.
        """
        max_hops = max(1, min(max_hops, 5))
        seta = "->" if directed else "-"
        excluidos = [t.value for t in avoid_hub_types]

        for hops in range(1, max_hops + 1):
            partes: list[str] = []
            retorno: list[str] = []
            filtros: list[str] = []
            for i in range(hops):
                origem = "a" if i == 0 else f"n{i}"
                destino = "b" if i == hops - 1 else f"n{i + 1}"
                sufixo = "" if i == hops - 1 else ":Entity"
                partes.append(f"({origem})-[r{i}:RELATES_TO]{seta}({destino}{sufixo})")
                retorno.extend([f"{origem}.name", f"r{i}.type", f"r{i}.evidence"])
                if i < hops - 1 and excluidos:
                    for tipo in excluidos:
                        filtros.append(f"{destino}.type <> '{tipo}'")
            retorno.append("b.name")

            onde = f" WHERE {' AND '.join(filtros)}" if filtros else ""
            rows = run_cypher(
                conn,
                f"MATCH (a:Entity {{key: $source}}), (b:Entity {{key: $target}}), "
                f"{', '.join(partes)}{onde} RETURN {', '.join(retorno)} LIMIT 1",
                graph_name=self._graph,
                params={"source": source, "target": target},
                columns=len(retorno),
            )
            if rows:
                return _build_path(rows[0], hops)
        return []

    def stats(self, conn: psycopg.Connection[Any]) -> dict[str, int]:
        nos = run_cypher(conn, "MATCH (n:Entity) RETURN count(n)", graph_name=self._graph)
        arestas = run_cypher(
            conn, "MATCH ()-[r:RELATES_TO]->() RETURN count(r)", graph_name=self._graph
        )
        return {
            "nodes": _agint(nos[0]["c0"]) if nos else 0,
            "edges": _agint(arestas[0]["c0"]) if arestas else 0,
        }

    def export(self, conn: psycopg.Connection[Any]) -> dict[str, list[dict[str, Any]]]:
        """Grafo inteiro em estruturas simples, para visualizacao estatica.

        Sai `dict`, nao os modelos do dominio, porque o destino e um JSON lido
        por uma pagina HTML sem servidor -- e porque o formato de arquivo nao
        deve virar refem do formato interno.

        Arestas paralelas (a mesma relacao afirmada em trechos diferentes) sao
        colapsadas numa so, com `count` guardando quantas vezes foi afirmada.
        Desenhar as duas empilhadas nao mostra nada; a contagem mostra.
        """
        nos = run_cypher(
            conn,
            "MATCH (n:Entity) RETURN n.key, n.name, n.type, n.mentions",
            graph_name=self._graph,
            columns=4,
        )
        arestas = run_cypher(
            conn,
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            RETURN a.key, b.key, r.type, r.evidence
            """,
            graph_name=self._graph,
            columns=4,
        )

        colapsadas: dict[tuple[str, str, str], dict[str, Any]] = {}
        for linha in arestas:
            chave = (_agtext(linha["c0"]), _agtext(linha["c1"]), _agtext(linha["c2"]))
            atual = colapsadas.get(chave)
            if atual is None:
                colapsadas[chave] = {
                    "source": chave[0],
                    "target": chave[1],
                    "type": chave[2],
                    "evidence": _agtext(linha["c3"]),
                    "count": 1,
                }
            else:
                atual["count"] += 1

        return {
            "nodes": [
                {
                    "key": _agtext(n["c0"]),
                    "name": _agtext(n["c1"]),
                    "type": _agtext(n["c2"]),
                    "mentions": _agint(n["c3"]),
                }
                for n in nos
            ],
            "edges": list(colapsadas.values()),
        }

    def find_key(self, conn: psycopg.Connection[Any], name: str) -> str | None:
        """Acha a chave canonica a partir de um nome digitado pelo usuario."""
        chave = normalize_name(name)
        rows = run_cypher(
            conn,
            "MATCH (n:Entity {key: $key}) RETURN n.key",
            graph_name=self._graph,
            params={"key": chave},
        )
        return _agtext(rows[0]["c0"]) if rows else None

    def clear(self, conn: psycopg.Connection[Any]) -> None:
        """Esvazia o grafo. Usado em teste e em reconstrucao completa."""
        run_cypher(conn, "MATCH (n:Entity) DETACH DELETE n RETURN 1", graph_name=self._graph)


def _agtext(value: Any) -> str:
    """agtype chega como texto JSON: '"Vale"' vira 'Vale'."""
    if value is None:
        return ""
    texto = str(value)
    if texto.startswith('"') and texto.endswith('"'):
        return texto[1:-1]
    return texto


def _agint(value: Any) -> int:
    try:
        return int(str(value).strip('"'))
    except (TypeError, ValueError):
        return 0


def _as_enum(enum_cls: Any, value: str, default: Any) -> Any:
    try:
        return enum_cls(value)
    except ValueError:
        return default


def _build_path(row: dict[str, Any], hops: int) -> list[PathStep]:
    passos: list[PathStep] = []
    for i in range(hops):
        destino = row[f"c{(i + 1) * 3}"] if i < hops - 1 else row[f"c{hops * 3}"]
        passos.append(
            PathStep(
                source=_agtext(row[f"c{i * 3}"]),
                relation=_as_enum(
                    RelationType, _agtext(row[f"c{i * 3 + 1}"]), RelationType.RELACIONADA_A
                ),
                target=_agtext(destino),
                evidence=_agtext(row[f"c{i * 3 + 2}"]),
            )
        )
    return passos
