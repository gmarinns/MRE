"""Camada de dados — Yahoo Finance com fallback sintético determinístico."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

try:  # pragma: no cover - dependência opcional
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:  # pragma: no cover
    YFINANCE_AVAILABLE = False


PERIOD_TO_DAYS = {"6mo": 126, "1y": 252, "2y": 504, "5y": 1260}


@dataclass(frozen=True)
class PriceSeries:
    """OHLCV de um ativo + a proveniência mínima de como ele foi obtido."""

    ticker: str
    frame: pd.DataFrame
    source: str  # "yahoo" | "synthetic"
    period: str

    @property
    def is_real(self) -> bool:
        return self.source == "yahoo"

    @property
    def n_obs(self) -> int:
        return len(self.frame)


def deterministic_seed(text: str) -> int:
    """Semente estável entre execuções.

    O beta usava ``hash(ticker)``, que o CPython randomiza por processo
    (PYTHONHASHSEED). Num projeto cujo tema é reprodutibilidade, o dado
    sintético precisa ser bit-a-bit reprodutível entre reinícios do app.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _synthetic_ohlcv(ticker: str, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(deterministic_seed(ticker))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    mu, sigma = 0.0004, 0.021
    # Um pouco de heterocedasticidade para o GARCH ter o que estimar.
    vol_factor = 1.0 + 0.6 * np.abs(np.sin(np.linspace(0, 3 * np.pi, n)))
    log_rets = rng.normal(mu, sigma * vol_factor)
    close = 30.0 * np.exp(np.cumsum(log_rets))
    open_ = close * (1 - rng.uniform(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.006, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.006, n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000_000, 8_000_000, n),
        }
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, period: str = "2y") -> PriceSeries:
    """Baixa OHLCV do Yahoo; cai para GBM sintético se a rede falhar."""
    if YFINANCE_AVAILABLE:
        try:
            raw = yf.download(
                ticker, period=period, interval="1d",
                progress=False, auto_adjust=True, threads=False,
            )
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                frame = raw.reset_index()
                frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
                if "date" not in frame.columns and "index" in frame.columns:
                    frame = frame.rename(columns={"index": "date"})
                if "close" in frame.columns:
                    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
                    keep = [c for c in ("date", "open", "high", "low", "close", "volume") if c in frame]
                    frame = frame[keep].dropna(subset=["close"]).reset_index(drop=True)
                    if len(frame) > 30:
                        return PriceSeries(ticker, frame, "yahoo", period)
        except Exception:  # rede fora, ticker inválido, rate-limit...
            pass

    n = PERIOD_TO_DAYS.get(period, 504)
    return PriceSeries(ticker, _synthetic_ohlcv(ticker, n), "synthetic", period)


def log_returns(frame: pd.DataFrame) -> np.ndarray:
    close = frame["close"].to_numpy(dtype=float)
    return np.diff(np.log(close))


def returns_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """DataFrame indexado por data com o log-retorno — base do alinhamento multi-ativo."""
    out = pd.DataFrame(
        {"date": pd.to_datetime(frame["date"]).to_numpy()[1:], "ret": log_returns(frame)}
    )
    return out.set_index("date")


@st.cache_data(ttl=3600, show_spinner=False)
def load_returns_matrix(tickers: tuple[str, ...], period: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Matriz de retornos **alinhada por data** (não por posição).

    O beta cortava as séries pelo comprimento mínimo (``v[-min_len:]``), o que
    alinha por posição. Ativos com feriados diferentes (ex.: ``^BVSP`` vs.
    ``BRL=X``) ficam defasados e a matriz de correlação sai errada.
    """
    series, sources = {}, {}
    for ticker in tickers:
        ps = load_prices(ticker, period)
        sources[ticker] = ps.source
        series[ticker] = returns_frame(ps.frame)["ret"]
    matrix = pd.concat(series, axis=1, join="inner").dropna()
    matrix.columns = list(tickers)
    return matrix, sources
