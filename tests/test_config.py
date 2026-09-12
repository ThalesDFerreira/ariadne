"""Configuracao: defaults, override por ambiente e nao-vazamento de segredo."""

import pytest

from ariadne.config import LLMProvider, Settings


def test_defaults_batem_com_o_docker_compose():
    cfg = Settings()
    assert cfg.pg_port == 5433
    assert cfg.graph_name == "ariadne"
    assert cfg.llm_provider is LLMProvider.OLLAMA


def test_ambiente_sobrescreve_default(monkeypatch):
    monkeypatch.setenv("ARIADNE_PG_HOST", "db.interno")
    monkeypatch.setenv("ARIADNE_EMBEDDING_DIM", "768")
    cfg = Settings()
    assert cfg.pg_host == "db.interno"
    assert cfg.embedding_dim == 768


def test_dsn_monta_e_senha_nao_vaza_no_repr():
    cfg = Settings(pg_password="supersecreta")
    assert "supersecreta" in cfg.pg_dsn
    # O SecretStr existe justamente para o segredo nao cair em log ou traceback.
    assert "supersecreta" not in repr(cfg)


def test_dimensao_de_embedding_precisa_ser_positiva():
    with pytest.raises(ValueError, match="greater than 0"):
        Settings(embedding_dim=0)


def test_provider_remoto_sem_chave_falha_com_mensagem_clara():
    cfg = Settings(llm_provider=LLMProvider.OPENAI, llm_api_key=None)
    with pytest.raises(ValueError, match="ARIADNE_LLM_API_KEY"):
        cfg.require_api_key()


def test_ollama_nao_pede_chave():
    cfg = Settings(llm_provider=LLMProvider.OLLAMA)
    with pytest.raises(ValueError, match="nao usa API key"):
        cfg.require_api_key()
