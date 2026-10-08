"""Value at Risk por fórmula fechada e por quantil empírico.

Todas as funções devolvem a perda como número positivo, em fração do valor
da posição. O horizonte é levado pela regra da raiz do tempo.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from .types import RiskResult, sha256_of_array, z_score
from .volatility import ewma_sigma, ewma_weights


def historical_var(
    returns: np.ndarray, confidence_level: float = 0.95, horizon_days: int = 1
) -> RiskResult:
    """Quantil empírico dos retornos observados; não assume distribuição."""
    alpha = 1.0 - confidence_level
    quantile = float(np.percentile(returns, alpha * 100.0))
    return RiskResult(
        value=float(-quantile * np.sqrt(horizon_days)),
        method="historical",
        confidence_level=confidence_level,
        params={"n_obs": int(len(returns)), "quantile": quantile},
        input_hash=sha256_of_array(returns),
        horizon_days=horizon_days,
    )


def parametric_var(
    returns: np.ndarray, confidence_level: float = 0.95, horizon_days: int = 1
) -> RiskResult:
    """Delta-Normal: retornos ~ N(μ, σ²)."""
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    z = z_score(confidence_level)
    value = -(mu * horizon_days + z * sigma * np.sqrt(horizon_days))
    return RiskResult(
        value=float(value),
        method="parametric",
        confidence_level=confidence_level,
        params={"mu": mu, "sigma": sigma, "z": z, "n_obs": int(len(returns))},
        input_hash=sha256_of_array(returns),
        horizon_days=horizon_days,
    )


def ewma_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    lam: float = 0.94,
    horizon_days: int = 1,
) -> RiskResult:
    """RiskMetrics: normal com σ ponderado para os dias mais recentes."""
    sigma = ewma_sigma(returns, lam)
    value = -(z_score(confidence_level) * sigma * np.sqrt(horizon_days))
    return RiskResult(
        value=float(value),
        method="ewma",
        confidence_level=confidence_level,
        params={"lambda": lam, "sigma_ewma": sigma, "n_obs": int(len(returns))},
        input_hash=sha256_of_array(returns),
        horizon_days=horizon_days,
    )


def cornish_fisher_quantile(z: float, skew: float, kurtosis: float) -> float:
    """Quantil normal corrigido por assimetria e curtose (não em excesso)."""
    excess = kurtosis - 3.0
    return (
        z
        + (z**2 - 1) * skew / 6.0
        + (z**3 - 3 * z) * excess / 24.0
        - (2 * z**3 - 5 * z) * skew**2 / 36.0
    )


def cornish_fisher_var(
    returns: np.ndarray, confidence_level: float = 0.95, horizon_days: int = 1
) -> RiskResult:
    """VaR paramétrico com o quantil corrigido pela expansão de Cornish-Fisher."""
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    skew = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns, fisher=False))
    z = z_score(confidence_level)
    z_cf = float(cornish_fisher_quantile(z, skew, kurtosis))
    value = -(mu * horizon_days + z_cf * sigma * np.sqrt(horizon_days))
    return RiskResult(
        value=float(value),
        method="cornish_fisher",
        confidence_level=confidence_level,
        params={
            "mu": mu,
            "sigma": sigma,
            "skew": skew,
            "kurtosis": kurtosis,
            "z": z,
            "z_cf": z_cf,
        },
        input_hash=sha256_of_array(returns),
        horizon_days=horizon_days,
    )


def rolling_var_series(
    returns: np.ndarray,
    window: int,
    confidence_level: float,
    method: str = "historical",
    lam: float = 0.94,
) -> np.ndarray:
    """VaR previsto para o dia *i* usando apenas os dados até *i-1*.

    As ``window`` primeiras posições ficam em NaN (ainda não há janela
    completa). Vetorizado com ``sliding_window_view``.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    # windows[j] = returns[j : j + window]; a janela que termina em i-1 é j = i-window
    windows = sliding_window_view(returns, window)[: n - window]
    alpha = 1.0 - confidence_level
    z = z_score(confidence_level)
    if method == "historical":
        out[window:] = -np.percentile(windows, alpha * 100.0, axis=1)
    elif method == "parametric":
        out[window:] = -(windows.mean(axis=1) + z * windows.std(axis=1, ddof=1))
    elif method == "ewma":
        sigma = np.sqrt((windows**2 * ewma_weights(window, lam)).sum(axis=1))
        out[window:] = -(z * sigma)
    else:
        raise ValueError(f"método rolante não suportado: {method}")
    return out
