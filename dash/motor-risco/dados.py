"""Cotações do Yahoo Finance, com cópia local da última coleta.

Não existe dado sintético aqui. Se a rede falhar, o painel usa a última
coleta real gravada em ``.cache/`` e informa a data; sem rede e sem cópia,
levanta ``DadosIndisponiveis``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

PASTA_CACHE = Path(__file__).parent / ".cache"
COLUNAS = ["date", "open", "high", "low", "close", "volume"]
MINIMO_PREGOES = 60


class DadosIndisponiveis(RuntimeError):
    """Não foi possível obter cotações nem da rede nem da cópia local."""


@dataclass(frozen=True)
class Cotacoes:
    ticker: str
    frame: pd.DataFrame
    origem: str  # "yahoo" | "cache"
    coletado_em: datetime

    @property
    def pregoes(self) -> int:
        return len(self.frame)


def _arquivo(ticker: str, periodo: str) -> Path:
    nome = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    return PASTA_CACHE / f"{nome}_{periodo}.parquet"


def _baixar(ticker: str, periodo: str) -> pd.DataFrame | None:
    try:
        bruto = yf.download(
            ticker,
            period=periodo,
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return None
    if bruto is None or bruto.empty:
        return None
    if isinstance(bruto.columns, pd.MultiIndex):
        bruto.columns = bruto.columns.get_level_values(0)
    frame = bruto.reset_index()
    frame.columns = [str(c).lower() for c in frame.columns]
    if "close" not in frame.columns or "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame = frame[[c for c in COLUNAS if c in frame.columns]]
    frame = frame.dropna(subset=["close"]).reset_index(drop=True)
    return frame if len(frame) >= MINIMO_PREGOES else None


@st.cache_data(ttl=3600, show_spinner=False)
def carregar(ticker: str, periodo: str = "2y") -> Cotacoes:
    """Baixa do Yahoo e grava a cópia local; na falha, lê a cópia."""
    arquivo = _arquivo(ticker, periodo)
    frame = _baixar(ticker, periodo)
    if frame is not None:
        PASTA_CACHE.mkdir(exist_ok=True)
        frame.to_parquet(arquivo, index=False)
        return Cotacoes(ticker, frame, "yahoo", datetime.now())
    if arquivo.exists():
        coletado = datetime.fromtimestamp(arquivo.stat().st_mtime)
        return Cotacoes(ticker, pd.read_parquet(arquivo), "cache", coletado)
    raise DadosIndisponiveis(
        f"Sem cotações para {ticker}: o Yahoo Finance não respondeu e não há "
        "cópia local de uma coleta anterior."
    )


def log_retornos(frame: pd.DataFrame) -> np.ndarray:
    return np.diff(np.log(frame["close"].to_numpy(dtype=float)))


def datas_dos_retornos(frame: pd.DataFrame) -> pd.Series:
    """Datas alinhadas a ``log_retornos`` (o primeiro pregão não tem retorno)."""
    return pd.to_datetime(frame["date"]).iloc[1:].reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
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
