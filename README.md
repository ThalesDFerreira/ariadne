# Ariadne

> AI-first knowledge engine — GraphRAG + MCP server. Turn a pile of documents into a queryable knowledge graph any LLM can use.

Ariadne ingere documentos, extrai automaticamente as entidades e as relacoes entre elas para montar um **grafo de conhecimento**, e responde perguntas combinando busca vetorial com navegacao nesse grafo. Tudo isso e exposto a qualquer assistente de IA atraves de um servidor **MCP** (Model Context Protocol).

O nome vem de Ariadne, que deu a Teseu o fio para atravessar o labirinto — e o que o projeto faz: da o fio que conecta as informacoes soltas.

## O problema

RAG tradicional recupera os trechos mais parecidos com a pergunta e para por ai.

- *"O que o contrato X diz sobre rescisao?"* → RAG comum resolve.
- *"Quais fornecedores dependem da mesma materia-prima que o fornecedor X?"* → RAG comum **falha**, porque a resposta nao esta escrita em lugar nenhum: precisa ser montada percorrendo relacoes entre documentos diferentes.

Ariadne resolve o segundo caso mantendo, alem dos vetores, um grafo `(entidade) -[relacao]-> (entidade)` extraido dos textos, usado para expandir o contexto recuperado.

## Estado atual

**Fase 0 — Fundacao concluida.** Postgres 16 com pgvector 0.8.6 e Apache AGE 1.5.0 no mesmo container, configuracao, lint/tipos/testes e CI. 15 testes passando, dos quais 9 de integracao contra o banco real. Proximo passo: Fase 1.

## Stack

| Camada | Escolha | Motivo |
|---|---|---|
| Linguagem | Python 3.12 + `uv` | padrao do ecossistema de IA |
| Banco | PostgreSQL 16 + **pgvector** + **Apache AGE** | vetor e grafo no mesmo container, na mesma transacao |
| LLM | Ollama (local) com adapter para API | roda 100% offline, trocando so a configuracao |
| Protocolo | **MCP** | conecta em Claude Desktop e Cursor |

### Por que vetor e grafo no mesmo Postgres

Duas razoes, e a segunda e a que decide:

1. **Uma transacao so.** Ingerir um documento grava chunks, embeddings, nos e arestas. Se a extracao falhar no meio, um `ROLLBACK` desfaz tudo junto. Com bancos separados nao existe transacao comum — mais cedo ou mais tarde sobra embedding orfao sem no.
2. **`JOIN` entre Cypher e busca vetorial na mesma query.** O fluxo central do GraphRAG — acha chunks por similaridade, descobre as entidades neles, expande k saltos, volta para os chunks vizinhos — vira um unico `SELECT` resolvido pelo planejador do Postgres, em vez de tres viagens de rede costuradas em Python.

O preco: o Apache AGE e menos maduro que o Neo4j e implementa so um subconjunto do openCypher. Por isso ele fica **atras da interface `GraphStore`** — trocar o backend nao encosta no motor de busca.

## Como rodar

Pre-requisitos: Docker e Python 3.12.

```bash
git clone https://github.com/ThalesDFerreira/ariadne.git
cd ariadne
cp .env.example .env
docker compose up -d --build   # o primeiro build compila o AGE do source (alguns minutos)
uv sync
uv run pytest
```

## Desenvolvimento

```bash
uv run ruff check .      # lint
uv run ruff format .     # formatacao
uv run mypy src          # tipos (modo strict)
uv run pytest            # testes; os de integracao pulam sozinhos se o banco estiver fora
uv run pytest -m integration
```

### Porta do Postgres

O container publica a **15432** no host, e nao a 5432. No Windows, as portas 5432 e 5433 costumam cair dentro dos intervalos que o Hyper-V reserva, e o bind falha com `forbidden by its access permissions`. Para conferir os intervalos reservados na sua maquina:

```bash
netsh interface ipv4 show excludedportrange protocol=tcp
```

## Estrutura

```
src/ariadne/
├── config.py     # fonte unica de configuracao (pydantic-settings)
├── domain/       # modelos Pydantic puros, sem dependencia do resto
├── storage/      # interfaces + adapters: VectorStore, GraphStore, DocumentStore
├── ingestion/    # parse → chunk → embed → extract
├── retrieval/    # vector, bm25, rrf, k-hop, rerank
├── llm/          # adapter de provedor (Ollama | API) + embeddings
├── agents/       # roteador de query
├── mcp/          # servidor MCP
└── eval/         # golden dataset + RAGAS
```

`domain/` nao importa nada do resto — e o nucleo. `storage/` guarda as interfaces e seus adapters, o que permite trocar o backend do grafo sem tocar em `retrieval/`.

## Roadmap

- [x] **Fase 0** — Fundacao: Docker com pgvector + AGE, configuracao, lint/tipos/testes, CI
- [ ] **Fase 1** — Ingestao e RAG baseline: parsing, chunking, embeddings, busca vetorial
- [ ] **Fase 1.5** — MCP minimo ponta a ponta
- [ ] **Fase 2** — O grafo: extracao de entidades/relacoes, entity resolution, Cypher
- [ ] **Fase 3** — Recuperacao hibrida: BM25, RRF, expansao k-hop, reranking
- [ ] **Fase 4** — Servidor MCP completo
- [ ] **Fase 5** — Camada agentica: roteador e perguntas multi-hop
- [ ] **Fase 6** — Avaliacao (RAGAS) e observabilidade
- [ ] **Fase 7** — Vitrine: UI do grafo e dataset publico

## Licenca

MIT
