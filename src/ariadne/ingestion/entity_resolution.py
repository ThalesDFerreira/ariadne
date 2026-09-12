"""Resolucao de entidades: decidir quando dois nomes sao a mesma coisa.

E o ponto que faz ou quebra um GraphRAG. Errar para menos deixa "Petrobras" e
"PETROBRAS S.A." como dois nos que nunca se encontram; errar para mais funde
"Banco do Brasil" com "Banco Central do Brasil" e produz um grafo que afirma
bobagem com toda a confianca.

POR QUE NAO USAMOS SIMILARIDADE DE EMBEDDING AQUI
-------------------------------------------------
A ideia obvia -- fundir nomes cujo embedding seja proximo -- foi medida com o
BGE-M3 neste corpus e REPROVADA. Os numeros:

    devem fundir      CSN ~ Companhia Siderurgica Nacional      0.384
                      BNDES ~ Banco Nacional de Desenvolv...    0.472
                      Petrobras ~ Petroleo Brasileiro S.A.      0.542
                      Vale ~ Vale S.A.                          0.758

    NAO devem fundir  Petrobras ~ Petrobras Distribuidora       0.783
                      Usiminas ~ Mineracao Usiminas             0.776
                      Banco do Brasil ~ Banco Central do Brasil 0.928

Os conjuntos nao so se sobrepoem: eles se invertem. O par mais parecido de
todos e justamente o que nao pode ser fundido. Em nome curto, o embedding mede
parecenca de palavra, nao identidade -- e "Central" no meio muda a entidade
inteira enquanto mal move o vetor.

Por isso a resolucao aqui e DETERMINISTICA e conservadora: so funde o que da
para justificar por regra. Preferimos um grafo com duplicata a um grafo que
mente.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ariadne.domain.graph import (
    EntityType,
    ExtractedEntity,
    GraphNode,
    normalize_name,
)

# Palavras que nao entram na formacao de sigla.
_IRRELEVANTES = {"de", "da", "do", "das", "dos", "e", "em", "para", "a", "o"}


def acronym_of(name: str) -> str:
    """Sigla formada pelas iniciais das palavras significativas.

    "Companhia Siderurgica Nacional" -> "CSN"
    "Banco Nacional de Desenvolvimento Economico e Social" -> "BNDES"
    """
    palavras = [p for p in re.split(r"\s+", normalize_name(name)) if p and p not in _IRRELEVANTES]
    return "".join(p[0] for p in palavras).upper()


def looks_like_acronym(name: str) -> bool:
    """Nome curto, sem espaco e em caixa alta: candidato a sigla."""
    limpo = name.strip()
    return bool(re.fullmatch(r"[A-Z][A-Z0-9]{1,7}", limpo))


@dataclass
class _Cluster:
    key: str
    types: dict[EntityType, int] = field(default_factory=dict)
    variants: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, type_: EntityType, count: int = 1) -> None:
        self.variants[name] = self.variants.get(name, 0) + count
        self.types[type_] = self.types.get(type_, 0) + count

    @property
    def type(self) -> EntityType:
        """Tipo por VOTO entre as mencoes, nao por primeira ocorrencia.

        O tipo e a parte menos confiavel da saida do LLM -- o mesmo texto
        rendeu "Cemig" como Organizacao num trecho e como Evento em outro. O
        nome, esse, e estavel. Votar corrige o erro pontual em vez de deixar
        que a ultima extracao processada decida.
        """
        return max(self.types.items(), key=lambda kv: (kv[1], kv[0] != EntityType.OUTRO))[0]

    @property
    def mentions(self) -> int:
        return sum(self.variants.values())

    def canonical(self) -> str:
        """Rotulo preferido entre as variantes vistas.

        Criterio: a mais citada; empatou, a mais longa. A mais longa costuma
        ser a forma completa ("Companhia Siderurgica Nacional" em vez de
        "CSN"), que e mais informativa para quem le a resposta.
        """
        return max(self.variants.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


class EntityResolver:
    """Agrupa mencoes na entidade que elas representam.

    Duas regras, ambas justificaveis por escrito:

    1. Mesma chave normalizada (caixa, acento e sufixo societario removidos).
    2. Sigla que bate exatamente com as iniciais de uma forma extensa, quando
       as duas tem o mesmo tipo.

    Qualquer coisa alem disso vira duplicata no grafo -- de proposito.
    """

    def __init__(self, *, resolve_acronyms: bool = True) -> None:
        self._resolve_acronyms = resolve_acronyms

    def resolve(self, entities: list[ExtractedEntity]) -> list[GraphNode]:
        """Agrupa por CHAVE, nao por (chave, tipo).

        Agrupar incluindo o tipo fragmentava a mesma empresa em varios nos
        sempre que o modelo trocava a categoria entre trechos -- e como a
        gravacao no grafo faz MERGE so pela chave, o ultimo no processado
        sobrescrevia os anteriores e as mencoes se perdiam. Com o agrupamento
        por chave, as mencoes somam e o tipo sai por voto.
        """
        clusters: dict[str, _Cluster] = {}

        for entity in entities:
            cluster = clusters.get(entity.key)
            if cluster is None:
                cluster = _Cluster(key=entity.key)
                clusters[entity.key] = cluster
            cluster.add(entity.name, entity.type)

        if self._resolve_acronyms:
            clusters = self._merge_acronyms(clusters)

        return [
            GraphNode(
                key=cluster.key,
                name=cluster.canonical(),
                type=cluster.type,
                aliases=sorted(v for v in cluster.variants if v != cluster.canonical()),
                mentions=cluster.mentions,
            )
            for cluster in clusters.values()
        ]

    def _merge_acronyms(self, clusters: dict[str, _Cluster]) -> dict[str, _Cluster]:
        # Indexa as formas extensas pela sigla que elas geram.
        extensos: dict[str, list[str]] = {}
        for chave, cluster in clusters.items():
            for variante in cluster.variants:
                if " " not in variante.strip():
                    continue
                sigla = acronym_of(variante)
                if len(sigla) >= 2:
                    extensos.setdefault(sigla, []).append(chave)

        resultado = dict(clusters)
        for chave, cluster in list(clusters.items()):
            if chave not in resultado:
                continue
            candidatos = [v for v in cluster.variants if looks_like_acronym(v.strip())]
            if not candidatos:
                continue

            for sigla_bruta in candidatos:
                alvos = extensos.get(sigla_bruta.strip().upper(), [])
                # Ambiguidade nao se resolve no chute: duas formas extensas
                # gerando a mesma sigla ficam separadas.
                alvos = [a for a in alvos if a != chave and a in resultado]
                if len(alvos) != 1:
                    continue

                destino = resultado[alvos[0]]
                for nome, n in cluster.variants.items():
                    destino.variants[nome] = destino.variants.get(nome, 0) + n
                for tipo, n in cluster.types.items():
                    destino.types[tipo] = destino.types.get(tipo, 0) + n
                resultado.pop(chave, None)
                break

        return resultado
