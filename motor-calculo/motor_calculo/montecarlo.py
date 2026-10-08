"""VaR por simulação de Monte Carlo (GBM com volatilidade do GARCH)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .types import RiskResult, sha256_of_array
from .volatility import fit_garch, garch_forecast


def simulate_paths(
    mu: float,
    garch: dict,
    n_sims: int = 10_000,
    n_days: int = 1,
    dist: str = "normal",
    df_t: float = 5.0,
    seed: int = 42,
    vol_dynamics: bool = True,
    return_paths: bool = False,
) -> np.ndarray:
    """Simula log-retornos acumulados.

    Devolve o retorno acumulado no fim do horizonte, formato ``(n_sims,)``, ou
    a trajetória dia a dia, formato ``(n_sims, n_days)``, se ``return_paths``.

    Com ``vol_dynamics`` a recursão do GARCH roda dentro de cada trajetória
    (volatilidade estocástica). O laço é sobre os dias; cada passo é vetorizado
    sobre todas as simulações.
    """
    rng = np.random.default_rng(seed)
    if dist == "t":
        shocks = rng.standard_t(df_t, size=(n_sims, n_days))
        shocks /= np.sqrt(df_t / (df_t - 2.0))  # padroniza para variância 1
    else:
        shocks = rng.standard_normal((n_sims, n_days))

    if garch.get("available") and vol_dynamics:
        omega, alpha, beta = garch["omega"], garch["alpha"], garch["beta"]
        h = np.full(n_sims, (garch["sigma_next"] * 100.0) ** 2)
        daily = np.empty((n_sims, n_days))
        for day in range(n_days):
            eps = np.sqrt(h) * shocks[:, day]  # choque na escala ×100
            daily[:, day] = mu + eps / 100.0
            h = omega + alpha * eps**2 + beta * h
    else:
        daily = mu + garch_forecast(garch, n_days) * shocks

    paths = np.cumsum(daily, axis=1)
    return paths if return_paths else paths[:, -1]


def montecarlo_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    n_sims: int = 10_000,
    n_days: int = 1,
    dist: str = "normal",
    df_t: float = 5.0,
    seed: int = 42,
    vol_dynamics: bool = True,
    garch: dict | None = None,
) -> tuple[RiskResult, np.ndarray]:
    """VaR como quantil da distribuição simulada; devolve também os cenários."""
    garch = garch if garch is not None else fit_garch(returns)
    mu = float(np.mean(returns))
    sims = simulate_paths(
        mu,
        garch,
        n_sims=n_sims,
        n_days=n_days,
        dist=dist,
        df_t=df_t,
        seed=seed,
        vol_dynamics=vol_dynamics,
    )
    alpha = 1.0 - confidence_level
    value = -float(np.percentile(sims, alpha * 100.0))
    # Erro padrão do quantil empírico: sqrt(α(1-α)/n) / f(q)
    sample = sims[: min(len(sims), 20_000)]
    density = max(float(stats.gaussian_kde(sample).evaluate([-value])[0]), 1e-9)
    std_error = float(np.sqrt(alpha * (1 - alpha) / n_sims) / density)
    result = RiskResult(
        value=value,
        method="montecarlo",
        confidence_level=confidence_level,
        params={
            "n_sims": int(n_sims),
            "n_days": int(n_days),
            "dist": dist,
            "df_t": df_t if dist == "t" else None,
            "seed": int(seed),
            "vol_dynamics": bool(vol_dynamics and garch.get("available")),
            "sigma_garch_next": float(garch["sigma_next"]),
            "mc_std_error": std_error,
        },
        input_hash=sha256_of_array(returns),
        horizon_days=n_days,
    )
    return result, sims


def montecarlo_convergence(
    returns: np.ndarray,
    confidence_level: float,
    sizes: tuple[int, ...],
    dist: str = "normal",
    seed: int = 42,
    garch: dict | None = None,
) -> pd.DataFrame:
    """VaR estimado e seu erro padrão em função do número de simulações."""
    garch = garch if garch is not None else fit_garch(returns)
    rows = []
    for n in sizes:
        result, _ = montecarlo_var(
            returns,
            confidence_level,
            n_sims=int(n),
            dist=dist,
            seed=seed,
            garch=garch,
        )
        rows.append(
            {
                "n_sims": int(n),
                "var": result.value,
                "std_error": result.params["mc_std_error"],
            }
        )
    return pd.DataFrame(rows)
