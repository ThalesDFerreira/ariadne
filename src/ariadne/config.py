"""Configuracao do Ariadne.

Fonte unica de verdade: tudo que varia entre ambientes entra aqui, lido do
ambiente ou do .env. Nenhum outro modulo le os.environ diretamente -- assim
existe um lugar so para descobrir o que o projeto precisa para rodar.
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    """Provedores suportados.

    O projeto roda 100% local com ollama; os demais existem para que trocar de
    provedor seja mudanca de configuracao, nunca de codigo.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Configuracao completa, carregada de ARIADNE_* ou do .env."""

    model_config = SettingsConfigDict(
        env_prefix="ARIADNE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Postgres ---
    pg_host: str = "localhost"
    pg_port: int = 15432
    pg_user: str = "ariadne"
    pg_password: SecretStr = SecretStr("ariadne")
    pg_database: str = "ariadne"

    graph_name: str = "ariadne"

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    llm_api_key: SecretStr | None = None

    # --- Embeddings ---
    # BGE-M3 produz 1024 dimensoes; o schema do pgvector depende desse numero,
    # entao trocar de modelo exige recriar a coluna e reindexar.
    embedding_model: str = "BAAI/bge-m3"
    embedding_model_ollama: str = "bge-m3"
    """Mesmo modelo, nome como o Ollama o registra."""

    embedding_dim: int = Field(default=1024, gt=0)

    @property
    def pg_dsn(self) -> str:
        """DSN libpq, com a senha em claro -- e o formato que o libpq exige.

        Property pura, e NAO computed_field: computed_field entraria no repr()
        e no model_dump() do Pydantic, e como o DSN carrega a senha isso
        anularia o SecretStr justamente onde ele importa (log e traceback).
        Quem chama recebe o segredo porque pediu; quem loga o objeto, nao.
        """
        return (
            f"postgresql://{self.pg_user}:{self.pg_password.get_secret_value()}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    def require_api_key(self) -> SecretStr:
        """Falha cedo e com mensagem clara quando o provedor remoto nao tem chave."""
        if self.llm_provider is LLMProvider.OLLAMA:
            msg = "provider ollama nao usa API key"
            raise ValueError(msg)
        if self.llm_api_key is None:
            msg = f"ARIADNE_LLM_API_KEY e obrigatoria para o provider {self.llm_provider}"
            raise ValueError(msg)
        return self.llm_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instancia unica e cacheada.

    Cache porque ler .env a cada chamada e desperdicio, e porque configuracao
    mudando no meio da execucao so gera bug dificil de achar. Nos testes,
    chame get_settings.cache_clear() para forcar releitura.
    """
    return Settings()
