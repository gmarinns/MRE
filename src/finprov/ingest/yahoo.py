import json

import requests
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://finprov:finprov@localhost:5432/finprov"
TICKER = "PETR4.SA"

url = f"https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}"
response = requests.get(
    url,
    params={"range": "1y", "interval": "1d"},
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=30,
)
response.raise_for_status()  # falha alto se não for 200

dados_json = response.json()
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS raw.cotacoes (
            id          BIGSERIAL PRIMARY KEY,
            ticker      TEXT        NOT NULL,
            coletado_em TIMESTAMPTZ NOT NULL DEFAULT now(),
            payload     JSONB       NOT NULL
        )
    """)
    )
    conn.execute(
        text("INSERT INTO raw.cotacoes (ticker, payload) VALUES (:t, :p)"),
        {"t": TICKER, "p": json.dumps(dados_json)},
    )

print(f"ok: {TICKER}")
