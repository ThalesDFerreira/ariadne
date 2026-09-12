-- Roda UMA unica vez, quando o volume de dados e criado do zero.
-- Para reexecutar: docker compose down -v && docker compose up -d

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- create_graph estoura se o grafo ja existir, por isso o guard.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'ariadne') THEN
        PERFORM ag_catalog.create_graph('ariadne');
    END IF;
END
$$;
