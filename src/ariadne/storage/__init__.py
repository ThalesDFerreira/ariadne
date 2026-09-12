"""Persistencia: interfaces e seus adapters.

As interfaces (VectorStore, GraphStore, DocumentStore) descrevem o que o
projeto precisa; os adapters resolvem em Postgres, pgvector e Apache AGE.
E esta fronteira que permite trocar o backend do grafo sem tocar em retrieval.
"""
