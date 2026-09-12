"""Pipeline de entrada: parse -> chunk -> embed -> extract.

Caminho de escrita, tolerante a latencia: roda em lote e pode levar horas.
Otimizar aqui e otimizar throughput, nao tempo de resposta.
"""
