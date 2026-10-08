"""Modelos de volatilidade: EWMA (RiskMetrics), GARCH(1,1) e janela rolante."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - dependência opcional
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    ARCH_AVAILABLE = False

TRADING_DAYS = 252


# ── EWMA ──────────────────────────────────────────────────────────────────
def ewma_weights(n: int, lam: float = 0.94) -> np.ndarray:
    """Pesos de decaimento, do mais antigo ao mais recente; somam 1."""
    weights = lam ** np.arange(n - 1, -1, -1)
    return weights / weights.sum()


def ewma_sigma(returns: np.ndarray, lam: float = 0.94) -> float:
    """Volatilidade EWMA: média dos retornos² ponderada pelo decaimento."""
    r2 = np.asarray(returns, dtype=float) ** 2
    return float(np.sqrt(np.sum(ewma_weights(len(r2), lam) * r2)))


def ewma_sigma_series(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """σ_t pela recursão σ²_t = λ σ²_{t-1} + (1-λ) r²_t, para toda a amostra."""
    r2 = pd.Series(np.asarray(returns, dtype=float) ** 2)
    return np.sqrt(r2.ewm(alpha=1.0 - lam, adjust=False).mean().to_numpy())


# ── Janela rolante ────────────────────────────────────────────────────────
def rolling_volatility(
    returns: np.ndarray, window: int = 21, annualize: bool = True
) -> np.ndarray:
    sigma = pd.Series(returns).rolling(window).std(ddof=1).to_numpy()
    return sigma * np.sqrt(TRADING_DAYS) if annualize else sigma


# ── GARCH(1,1) ────────────────────────────────────────────────────────────
def _garch_fallback(returns: np.ndarray, note: str) -> dict:
    nan = float("nan")
    return {
        "available": False,
        "omega": nan,
        "alpha": nan,
        "beta": nan,
        "persistence": nan,
        "loglik": nan,
        "aic": nan,
        "sigma_next": float(np.std(returns[-21:], ddof=1)),
        "sigma_long_run": float(np.std(returns, ddof=1)),
        "conditional_vol": np.full(len(returns), float(np.std(returns, ddof=1))),
        "note": note,
    }


def fit_garch(returns: np.ndarray) -> dict:
    """Ajusta GARCH(1,1) por máxima verossimilhança (biblioteca ``arch``).

    Os retornos são escalados por 100 na estimação (recomendação da própria
    ``arch`` para a estabilidade do otimizador); ω fica nessa escala e os
    desvios-padrão devolvidos voltam para fração.
    """
    returns = np.asarray(returns, dtype=float)
    if not ARCH_AVAILABLE:
        return _garch_fallback(returns, "biblioteca arch não instalada")
    if len(returns) < 60:
        return _garch_fallback(returns, "menos de 60 observações")
    try:
        model = arch_model(
            returns * 100.0, mean="Constant", vol="Garch", p=1, q=1, dist="normal"
        )
        res = model.fit(disp="off", show_warning=False)
        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        persistence = alpha + beta
        forecast = res.forecast(horizon=1, reindex=False)
        long_run = (
            float(np.sqrt(omega / (1.0 - persistence)) / 100.0)
            if persistence < 1.0
            else float("nan")
        )
        return {
            "available": True,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": persistence,
            "loglik": float(res.loglikelihood),
            "aic": float(res.aic),
            "sigma_next": float(np.sqrt(forecast.variance.values[-1, 0]) / 100.0),
            "sigma_long_run": long_run,
            "conditional_vol": np.asarray(res.conditional_volatility) / 100.0,
            "note": "",
        }
    except Exception as exc:  # pragma: no cover - depende do otimizador
        return _garch_fallback(returns, f"estimação falhou ({type(exc).__name__})")


def garch_forecast(garch: dict, n_days: int) -> np.ndarray:
    """σ previsto para t+1 … t+n (em fração ao dia).

    Recursão h_{k+1} = ω + (α+β)·h_k: a previsão parte do σ de amanhã e
    converge para a volatilidade de longo prazo à taxa α+β.
    """
    if not garch.get("available"):
        return np.full(n_days, garch["sigma_next"])
    omega = garch["omega"]
    persistence = garch["persistence"]
    h = (garch["sigma_next"] * 100.0) ** 2
    path = np.empty(n_days)
    for k in range(n_days):
        path[k] = np.sqrt(h) / 100.0
        h = omega + persistence * h
    return path
