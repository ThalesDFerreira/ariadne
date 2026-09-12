"""Ajustes de terminal.

Existe porque o console do Windows usa cp1252 e estoura em qualquer acento: um
corpus em portugues derrubava a saida com UnicodeEncodeError. A correcao
nasceu privada dentro da CLI e teve que ser repetida no script de demo -- sinal
de que o lugar dela era aqui desde o comeco.
"""

from __future__ import annotations

import sys


def force_utf8_output() -> None:
    """Faz stdout e stderr falarem UTF-8, trocando o que nao couber."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
