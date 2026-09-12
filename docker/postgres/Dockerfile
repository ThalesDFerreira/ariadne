# Postgres 16 falando VETOR (pgvector) e GRAFO (Apache AGE) no mesmo processo.
#
# Partimos da imagem oficial do pgvector porque compilar pgvector e o passo
# chato; o AGE, por sua vez, nao tem imagem oficial com pgvector junto, entao
# ele e compilado do source aqui.
FROM pgvector/pgvector:pg16

ARG AGE_BRANCH=PG16

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
    git clone --depth 1 --branch "${AGE_BRANCH}" https://github.com/apache/age.git /tmp/age; \
    make -C /tmp/age -j"$(nproc)" install; \
    rm -rf /tmp/age; \
    apt-get purge -y --auto-remove \
        build-essential git flex bison postgresql-server-dev-16 \
        libreadline-dev zlib1g-dev; \
    rm -rf /var/lib/apt/lists/*
