CREATE SCHEMA IF NOT EXISTS controle;
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS curated;
CREATE SCHEMA IF NOT EXISTS resultados;

CREATE TABLE controle.esquema (version integer PRIMARY KEY, installed_at timestamptz NOT NULL DEFAULT now());
INSERT INTO controle.esquema(version) VALUES (1);
CREATE TABLE controle.execucoes (
    run_id uuid PRIMARY KEY,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status text NOT NULL CHECK (status IN ('collecting','processing','ready','partial','failed')),
    parameters jsonb NOT NULL,
    environment jsonb NOT NULL,
    error text
);
CREATE TABLE curated.ativos (
    ticker text PRIMARY KEY,
    yahoo_symbol text NOT NULL UNIQUE,
    asset_type text NOT NULL CHECK (asset_type IN ('equity','benchmark'))
);
CREATE TABLE controle.posicoes (
    run_id uuid NOT NULL REFERENCES controle.execucoes,
    ticker text NOT NULL REFERENCES curated.ativos,
    weight numeric NOT NULL CHECK (weight >= 0 AND weight <= 1),
    PRIMARY KEY (run_id,ticker)
);
CREATE TABLE raw.respostas (
    raw_id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES controle.execucoes,
    source text NOT NULL CHECK (source IN ('yahoo','bacen','investidor10','investidor10_history')),
    ticker text REFERENCES curated.ativos,
    series integer,
    url text NOT NULL,
    params jsonb NOT NULL,
    observed_at timestamptz NOT NULL,
    status text NOT NULL CHECK (status IN ('ok','error')),
    body bytea,
    error text,
    normalization_status text NOT NULL DEFAULT 'pending'
        CHECK (normalization_status IN ('pending','ok','error','not_applicable')),
    normalization_error text,
    UNIQUE (run_id,raw_id),
    CHECK ((status='ok' AND body IS NOT NULL AND error IS NULL)
        OR (status='error' AND body IS NULL AND error IS NOT NULL))
);
CREATE INDEX respostas_execucao ON raw.respostas(run_id,source,ticker);

CREATE TABLE curated.precos (
    run_id uuid NOT NULL,
    raw_id uuid NOT NULL,
    ticker text NOT NULL REFERENCES curated.ativos,
    date date NOT NULL,
    open numeric CHECK (open>0),
    high numeric CHECK (high>0),
    low numeric CHECK (low>0),
    close numeric CHECK (close>0),
    adjusted_close numeric CHECK (adjusted_close>0),
    volume numeric CHECK (volume>=0),
    currency text NOT NULL CHECK (currency='BRL'),
    timezone text NOT NULL,
    adjustment text NOT NULL,
    quality_flags text NOT NULL,
    PRIMARY KEY(run_id,ticker,date),
    FOREIGN KEY(run_id,raw_id) REFERENCES raw.respostas(run_id,raw_id)
);
CREATE INDEX precos_ativo_data ON curated.precos(ticker,date);

CREATE TABLE curated.eventos (
    run_id uuid NOT NULL,
    raw_id uuid NOT NULL,
    row_number integer NOT NULL CHECK (row_number>=0),
    ticker text NOT NULL REFERENCES curated.ativos,
    event_type text NOT NULL,
    source_event_id text,
    raw_type text,
    ex_date date,
    com_date date,
    payment_date date,
    amount numeric CHECK (amount>=0),
    provider_value numeric,
    split_factor numeric CHECK (split_factor>0),
    currency text,
    tax_basis text NOT NULL,
    PRIMARY KEY(run_id,raw_id,row_number),
    FOREIGN KEY(run_id,raw_id) REFERENCES raw.respostas(run_id,raw_id)
);
CREATE INDEX eventos_ativo ON curated.eventos(ticker,com_date);

CREATE TABLE curated.indicadores (
    run_id uuid NOT NULL,
    raw_id uuid NOT NULL,
    ticker text NOT NULL REFERENCES curated.ativos,
    indicator text NOT NULL,
    indicator_group text,
    value numeric,
    raw_value text,
    unit text NOT NULL,
    period text NOT NULL CHECK (period IN ('current','annual')),
    reference_year integer,
    published_at timestamptz,
    available_at timestamptz NOT NULL,
    point_in_time_verified boolean NOT NULL DEFAULT false,
    FOREIGN KEY(run_id,raw_id) REFERENCES raw.respostas(run_id,raw_id),
    CHECK ((period='current' AND reference_year IS NULL)
        OR (period='annual' AND reference_year BETWEEN 1900 AND 9999))
);
CREATE UNIQUE INDEX indicadores_observacao ON curated.indicadores
    (run_id,ticker,indicator,period,COALESCE(reference_year,0));
CREATE INDEX indicadores_disponibilidade ON curated.indicadores(ticker,available_at);

CREATE TABLE curated.macro (
    run_id uuid NOT NULL,
    raw_id uuid NOT NULL,
    series integer NOT NULL CHECK (series IN (1,11,433)),
    date date NOT NULL,
    name text NOT NULL,
    value numeric,
    unit text NOT NULL,
    frequency text NOT NULL,
    published_at timestamptz,
    available_at timestamptz NOT NULL,
    PRIMARY KEY(run_id,series,date),
    FOREIGN KEY(run_id,raw_id) REFERENCES raw.respostas(run_id,raw_id)
);
CREATE TABLE controle.qualidade (
    run_id uuid PRIMARY KEY REFERENCES controle.execucoes,
    payload jsonb NOT NULL
);
CREATE TABLE controle.cobertura (
    run_id uuid NOT NULL REFERENCES controle.execucoes,
    ticker text NOT NULL REFERENCES curated.ativos,
    payload jsonb NOT NULL,
    PRIMARY KEY(run_id,ticker)
);
CREATE TABLE resultados.relatorios (
    run_id uuid NOT NULL REFERENCES controle.execucoes,
    kind text NOT NULL CHECK(kind IN ('risk','direction')),
    payload jsonb NOT NULL,
    PRIMARY KEY(run_id,kind)
);

-- Não mistura uma ação de uma coleta com outra execução.
CREATE VIEW controle.ultima_execucao_pronta AS
SELECT * FROM controle.execucoes WHERE status='ready'
ORDER BY finished_at DESC,run_id DESC LIMIT 1;
CREATE VIEW curated.precos_atuais AS
SELECT p.* FROM curated.precos p JOIN controle.ultima_execucao_pronta e USING(run_id);
CREATE VIEW curated.indicadores_atuais AS
SELECT p.* FROM curated.indicadores p JOIN controle.ultima_execucao_pronta e USING(run_id);
