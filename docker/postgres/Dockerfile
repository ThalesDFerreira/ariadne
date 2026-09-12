# Postgres 16 falando VETOR (pgvector) e GRAFO (Apache AGE) no mesmo processo.
#
# Partimos da imagem oficial do pgvector porque compilar pgvector e o passo
# chato; o AGE, por sua vez, nao tem imagem oficial com pgvector junto, entao
# ele e compilado do source aqui.
# Versao fixa, nao a tag movel 'pg16': mesma razao de fixar o AGE numa tag.
FROM pgvector/pgvector:0.8.6-pg16

# Tag fixa, nao a branch PG16: a branch e desenvolvimento e pode quebrar sem
# aviso, o que furaria a promessa de que clonar o repo e subir sempre funciona.
# Para testar o upstream mais novo: --build-arg AGE_REF=PG16
ARG AGE_REF=PG16/v1.5.0-rc0

# flex/bison sao exigidos pelo parser de Cypher do AGE.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        flex \
        bison \
        postgresql-server-dev-16 \
        libreadline-dev \
        zlib1g-dev; \
    git clone --depth 1 --branch "${AGE_REF}" https://github.com/apache/age.git /tmp/age; \
    make -C /tmp/age -j"$(nproc)" install; \
    rm -rf /tmp/age; \
    apt-get purge -y --auto-remove \
        build-essential git flex bison postgresql-server-dev-16 \
        libreadline-dev zlib1g-dev; \
    rm -rf /var/lib/apt/lists/*
