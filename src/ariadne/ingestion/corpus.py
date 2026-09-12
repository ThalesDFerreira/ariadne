"""Corpus de demonstracao.

Lista curada em vez de categoria da Wikipedia: as categorias de pt.wikipedia
sao rasas demais ("Empresas estatais do Brasil" devolve 2 paginas), e o que
esse grafo precisa e densidade de referencia mutua, nao volume.

O criterio da selecao e justamente esse: empresas que aparecem umas nas
paginas das outras -- por participacao acionaria, disputa de mercado,
financiamento pelo BNDES, privatizacao. Um corpus de 25 empresas sem relacao
entre si produziria 25 ilhas, e o GraphRAG nao teria nada a mostrar sobre RAG
puro.
"""

DEMO_CORPUS: list[str] = [
    # Mineracao, siderurgia e petroleo: cadeia de materia-prima em comum.
    "Petrobras",
    "Vale S.A.",
    "Companhia Siderúrgica Nacional",
    "Usiminas",
    "Gerdau",
    "Braskem",
    # Energia: participacoes cruzadas e concessoes.
    "Eletrobras",
    "Cemig",
    "Copel",
    "Engie Brasil",
    # Papel, celulose e agro.
    "Suzano Papel e Celulose",
    "Klabin",
    "JBS",
    "BRF",
    "Cosan",
    "Raízen",
    # Financeiro: aparecem como acionistas e credores dos demais.
    "Banco do Brasil",
    "Itaú Unibanco",
    "Banco Bradesco",
    "Banco Nacional de Desenvolvimento Econômico e Social",
    "B3 (empresa)",
    "Banco Central do Brasil",
    # Industria e consumo.
    "Embraer",
    "WEG (empresa)",
    "Ambev",
    "Localiza",
    "Ultrapar",
    "Natura &Co",
]
