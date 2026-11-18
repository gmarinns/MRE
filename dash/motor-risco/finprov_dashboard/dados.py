"""Adaptador PostgreSQL para o motor: sem coleta HTTP ou fallback em disco.

Cada renderização fixa uma execução ready. Todos os ativos usam essa mesma
versão. OHLC é ajustado pelo fator adjusted_close / close, como no painel original.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st

from finprov.storage.database import Database

MINIMO_PREGOES = 60


class DadosIndisponiveis(RuntimeError):
    """Não há dados válidos suficientes na execução do PostgreSQL."""


@dataclass(frozen=True)
class Cotacoes:
    ticker: str
    frame: pd.DataFrame
    origem: str
    coletado_em: datetime
    run_id: str

    @property
    def pregoes(self) -> int:
        return len(self.frame)


def selecionar_execucao() -> str:
    """Resolve latest uma vez por renderização, sem misturar versões de ativos."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise DadosIndisponiveis("Configure DATABASE_URL para acessar o PostgreSQL.")
    try:
        run_id = Database(url).resolve_run()
    except (psycopg2.Error, ValueError) as exc:
        raise DadosIndisponiveis(
            "Não foi possível acessar uma execução ready no PostgreSQL. "
            "Verifique o banco e execute a coleta pelo ETL. "
            "Nenhuma coleta alternativa foi usada."
        ) from exc
    st.session_state["_data_run_id"] = run_id
    return run_id


@st.cache_data(ttl=30, show_spinner=False)
def _precos_execucao(url: str, run_id: str) -> pd.DataFrame:
    with Database(url).connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT p.ticker,p.date,p.open,p.high,p.low,p.close,"
            "p.adjusted_close,p.volume,e.started_at "
            "FROM curated.precos p JOIN controle.execucoes e USING (run_id) "
            "WHERE p.run_id=%s AND e.status='ready' ORDER BY p.date,p.ticker",
            (run_id,),
        )
        return pd.DataFrame(
            cur.fetchall(),
            columns=[
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "started_at",
            ],
        )


def carregar(ticker: str, periodo: str = "2y") -> Cotacoes:
    """Lê preços da execução fixada; período termina na última data dessa execução."""
    if periodo not in {"1y", "2y", "5y"}:
        raise DadosIndisponiveis("Período de histórico inválido.")
    run_id = st.session_state.get("_data_run_id") or selecionar_execucao()
    try:
        bruto = _precos_execucao(os.environ["DATABASE_URL"], run_id)
    except (psycopg2.Error, ValueError, KeyError) as exc:
        raise DadosIndisponiveis(
            "Não foi possível ler os preços do PostgreSQL. "
            "Nenhuma coleta alternativa foi usada."
        ) from exc
    if bruto.empty:
        raise DadosIndisponiveis("A execução selecionada não possui preços.")
    fim = pd.to_datetime(bruto["date"]).max()
    inicio = fim - pd.DateOffset(years=int(periodo[0]))
    frame = bruto.loc[bruto.ticker == ticker.removesuffix(".SA")].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[frame.date > inicio].sort_values("date")
    colunas = ["open", "high", "low", "close", "adjusted_close", "volume"]
    frame[colunas] = frame[colunas].apply(pd.to_numeric, errors="coerce")
    validos = np.isfinite(
        frame[["open", "high", "low", "close", "adjusted_close"]]
    ).all(axis=1)
    validos &= (frame[["open", "high", "low", "close", "adjusted_close"]] > 0).all(
        axis=1
    )
    frame = frame.loc[validos].copy()
    if len(frame) < MINIMO_PREGOES:
        raise DadosIndisponiveis(
            f"Sem histórico suficiente para {ticker} no PostgreSQL: "
            f"{len(frame)} pregões válidos; mínimo {MINIMO_PREGOES}."
        )
    fator = frame["adjusted_close"] / frame["close"]
    frame[["open", "high", "low", "close"]] = frame[
        ["open", "high", "low", "close"]
    ].mul(fator, axis=0)
    coletado_em = pd.Timestamp(frame["started_at"].iloc[0]).to_pydatetime()
    frame = frame[["date", "open", "high", "low", "close", "volume"]].reset_index(
        drop=True
    )
    return Cotacoes(ticker, frame, "postgresql", coletado_em, run_id)


def log_retornos(frame: pd.DataFrame) -> np.ndarray:
    return np.diff(np.log(frame["close"].to_numpy(dtype=float)))


def datas_dos_retornos(frame: pd.DataFrame) -> pd.Series:
    """Datas alinhadas a ``log_retornos`` (o primeiro pregão não tem retorno)."""
    return pd.to_datetime(frame["date"]).iloc[1:].reset_index(drop=True)


def matriz_de_retornos(
    tickers: tuple[str, ...], periodo: str
) -> tuple[pd.DataFrame, list[str]]:
    """Retornos alinhados por data (interseção). Devolve também os que faltaram.

    O alinhamento é por data e não por posição: ativos de bolsas diferentes
    têm feriados diferentes, e alinhar por posição distorce a correlação.
    """
    series, faltaram = {}, []
    for ticker in tickers:
        try:
            frame = carregar(ticker, periodo).frame
        except DadosIndisponiveis:
            faltaram.append(ticker)
            continue
        series[ticker] = pd.Series(
            log_retornos(frame), index=datas_dos_retornos(frame).to_numpy()
        )
    if not series:
        return pd.DataFrame(), faltaram
    matriz = pd.concat(series, axis=1, join="inner").dropna()
    return matriz, faltaram
