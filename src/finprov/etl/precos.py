import json

import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://finprov:finprov@localhost:5432/finprov"
engine = create_engine(DATABASE_URL)


def extrair(payload: dict) -> pd.DataFrame:
    resultado = payload["chart"]["result"][0]
    cotacao = resultado["indicators"]["quote"][0]

    df = pd.DataFrame(
        {
            "data": pd.to_datetime(resultado["timestamp"], unit="s", utc=True),
            "abertura": cotacao["open"],
            "maxima": cotacao["high"],
            "minima": cotacao["low"],
            "fechamento": cotacao["close"],
            "volume": cotacao["volume"],
        }
    )

    ajustado = resultado["indicators"].get("adjclose")
    df["fechamento_ajustado"] = (
        ajustado[0]["adjclose"] if ajustado else df["fechamento"]
    )

    df["data"] = df["data"].dt.tz_convert("America/Sao_Paulo").dt.date
    df["ticker"] = resultado["meta"]["symbol"]

    return df.dropna(subset=["fechamento"])


UPSERT = text("""
    INSERT INTO curated.precos
        (ticker, data, abertura, maxima, minima,
         fechamento, fechamento_ajustado, volume)
    VALUES
        (:ticker, :data, :abertura, :maxima, :minima,
         :fechamento, :fechamento_ajustado, :volume)
    ON CONFLICT (ticker, data) DO UPDATE SET
        abertura = EXCLUDED.abertura,
        maxima = EXCLUDED.maxima,
        minima = EXCLUDED.minima,
        fechamento = EXCLUDED.fechamento,
        fechamento_ajustado = EXCLUDED.fechamento_ajustado,
        volume = EXCLUDED.volume
""")


def main() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS curated"))
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS curated.precos (
                ticker              TEXT NOT NULL,
                data                DATE NOT NULL,
                abertura            NUMERIC,
                maxima              NUMERIC,
                minima              NUMERIC,
                fechamento          NUMERIC,
                fechamento_ajustado NUMERIC,
                volume              BIGINT,
                PRIMARY KEY (ticker, data)
            )
        """)
        )

    # pega a coleta mais recente de cada ticker
    brutos = pd.read_sql(
        """
        SELECT DISTINCT ON (ticker) ticker, payload
        FROM raw.cotacoes
        ORDER BY ticker, coletado_em DESC
        """,
        engine,
    )

    for _, linha in brutos.iterrows():
        payload = linha["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)

        df = extrair(payload)

        with engine.begin() as conn:
            conn.execute(UPSERT, df.to_dict(orient="records"))

        print(f"ok: {linha['ticker']} — {len(df)} pregões")


if __name__ == "__main__":
    main()
