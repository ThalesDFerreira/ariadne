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
    """

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
        self, conn: psycopg.Connection[Any], source: str, target: str, *, max_hops: int = 4
    ) -> list[PathStep]:
        """Caminho entre duas entidades, com a evidencia de cada aresta.

        Busca por comprimento crescente: o primeiro que casar e o mais curto.
        """
        max_hops = max(1, min(max_hops, 5))

        for hops in range(1, max_hops + 1):
            partes: list[str] = []
            retorno: list[str] = []
            for i in range(hops):
                origem = "a" if i == 0 else f"n{i}"
                destino = "b" if i == hops - 1 else f"n{i + 1}"
                sufixo = "" if i == hops - 1 else ":Entity"
                partes.append(f"({origem})-[r{i}:RELATES_TO]->({destino}{sufixo})")
                retorno.extend([f"{origem}.name", f"r{i}.type", f"r{i}.evidence"])
            retorno.append("b.name")

            rows = run_cypher(
                conn,
                f"MATCH (a:Entity {{key: $source}}), (b:Entity {{key: $target}}), "
                f"{', '.join(partes)} RETURN {', '.join(retorno)} LIMIT 1",
                params={"source": source, "target": target},
                columns=len(retorno),
            )
            if rows:
                return _build_path(rows[0], hops)
        return []

    def stats(self, conn: psycopg.Connection[Any]) -> dict[str, int]:
        nos = run_cypher(conn, "MATCH (n:Entity) RETURN count(n)")
        arestas = run_cypher(conn, "MATCH ()-[r:RELATES_TO]->() RETURN count(r)")
        return {
            "nodes": _agint(nos[0]["c0"]) if nos else 0,
            "edges": _agint(arestas[0]["c0"]) if arestas else 0,
        }

    def find_key(self, conn: psycopg.Connection[Any], name: str) -> str | None:
        """Acha a chave canonica a partir de um nome digitado pelo usuario."""
        chave = normalize_name(name)
        rows = run_cypher(
            conn,
            "MATCH (n:Entity {key: $key}) RETURN n.key",
            params={"key": chave},
        )
        return _agtext(rows[0]["c0"]) if rows else None

    def clear(self, conn: psycopg.Connection[Any]) -> None:
        """Esvazia o grafo. Usado em teste e em reconstrucao completa."""
        run_cypher(conn, "MATCH (n:Entity) DETACH DELETE n RETURN 1")


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
