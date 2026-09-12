"""Chunking estrutural.

Chunking e decisao de RECUPERACAO, nao de parsing: o tamanho do pedaco define
o que o embedding consegue representar. Pedaco grande demais dilui o vetor --
um chunk que fala de cinco assuntos nao fica perto de nenhum deles. Pequeno
demais corta a resposta ao meio e chega sem contexto.

A estrategia aqui e respeitar as fronteiras que o autor do texto ja marcou
(secoes e paragrafos) em vez de cortar a cada N caracteres. Cortar as cegas e
o que produz aquele chunk que comeca no meio de uma frase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ariadne.domain.models import Chunk, Document

_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")
# Quebra de sentenca: ponto/!/? seguido de espaco e maiuscula. Simplista de
# proposito -- erra em "S.A." e afins, mas so e usada como ultimo recurso,
# quando um paragrafo sozinho estoura o limite.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])")

_JOIN = "\n\n"


@dataclass(frozen=True)
class ChunkingConfig:
    """Limites em CARACTERES, nao em tokens.

    Tokens seriam mais precisos, mas exigiriam carregar o tokenizer do modelo
    so para cortar texto. Como o BGE-M3 aceita ate 8192 tokens e 1800 chars
    dao ~450 tokens em portugues, ha folga de sobra -- a precisao nao compra
    nada aqui e custaria uma dependencia.
    """

    max_chars: int = 1800
    overlap_chars: int = 200
    min_chars: int = 80
    """Abaixo disso o pedaco vira cauda do anterior: um chunk de duas linhas
    soltas nao se sustenta como resultado de busca."""


@dataclass(frozen=True)
class Block:
    """Um paragrafo com a trilha de secoes a que pertence."""

    section_path: str
    text: str


def parse_blocks(text: str) -> list[Block]:
    """Quebra o texto em paragrafos, rastreando a hierarquia de secoes."""
    stack: list[str] = []
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        joined = "\n".join(buffer).strip()
        buffer.clear()
        if joined:
            blocks.append(Block(section_path=" > ".join(stack), text=joined))

    for raw in text.splitlines():
        line = raw.rstrip()
        heading = _HEADING.match(line.strip())
        if heading:
            flush()
            # len("==") == 2 e o nivel 1; "===" e o nivel 2, e assim por diante.
            level = len(heading.group(1)) - 2
            del stack[level:]
            stack.append(heading.group(2))
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)

    flush()
    return blocks


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Quebra um paragrafo que sozinho nao cabe num chunk."""
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        # Uma unica sentenca maior que o limite: corta na forca, sem escolha.
        while len(sentence) > max_chars:
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        current = sentence
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[Block]:
    """Agrupa paragrafos em pedacos do tamanho alvo, sem cruzar secoes.

    Nao ha overlap entre secoes diferentes de proposito: a troca de secao e
    uma fronteira semantica real, e emendar o fim de "Historia" no comeco de
    "Controladas" so cria um chunk que nao fala de nada.

    Invariante unica e inegociavel: nenhum bloco devolvido passa de max_chars.
    O overlap e a fusao de sobras sao melhorias -- cedem quando conflitam com
    ela.
    """
    cfg = config or ChunkingConfig()
    out: list[Block] = []

    for section, blocks in _group_by_section(parse_blocks(text)):
        pieces: list[str] = []
        for block in blocks:
            # Garante que todo pedaco ja cabe sozinho num chunk.
            pieces.extend(_split_oversized(block.text, cfg.max_chars))

        current = ""
        for piece in pieces:
            candidate = current + _JOIN + piece if current else piece
            if len(candidate) <= cfg.max_chars:
                current = candidate
                continue

            if not current:
                current = piece
                continue

            out.append(Block(section_path=section, text=current))
            tail = current[-cfg.overlap_chars :] if cfg.overlap_chars else ""
            with_tail = tail + _JOIN + piece if tail else piece
            # O overlap so entra se couber; caso contrario recomeca limpo.
            current = with_tail if len(with_tail) <= cfg.max_chars else piece

        if not current:
            continue

        merged = out[-1].text + _JOIN + current if out else ""
        if (
            out
            and out[-1].section_path == section
            and len(current) < cfg.min_chars
            and len(merged) <= cfg.max_chars
        ):
            # Sobra curta demais para sustentar um resultado de busca vira
            # cauda do anterior -- mas so quando ainda cabe.
            out[-1] = Block(section_path=section, text=merged)
        else:
            out.append(Block(section_path=section, text=current))

    return out


def _group_by_section(blocks: list[Block]) -> list[tuple[str, list[Block]]]:
    grouped: list[tuple[str, list[Block]]] = []
    for block in blocks:
        if grouped and grouped[-1][0] == block.section_path:
            grouped[-1][1].append(block)
        else:
            grouped.append((block.section_path, [block]))
    return grouped


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Converte um documento na lista ordenada de chunks que sera indexada."""
    return [
        Chunk(
            document_id=document.id,
            ordinal=i,
            content=block.text,
            section_path=block.section_path,
        )
        for i, block in enumerate(chunk_text(document.content, config))
    ]
