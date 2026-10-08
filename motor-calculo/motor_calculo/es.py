"""Expected Shortfall (CVaR): perda média nos cenários piores que o VaR."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .types import RiskResult, sha256_of_array, z_score


def _tail_mean(sample: np.ndarray, alpha: float) -> tuple[float, float, int]:
    """Média da cauda esquerda de nível α. Devolve (ES, quantil, n na cauda)."""
    quantile = float(np.percentile(sample, alpha * 100.0))
    # <= o quantil (e não <= -VaR arredondado): a cauda nunca fica vazia
    tail = sample[sample <= quantile]
    return -float(np.mean(tail)), quantile, int(tail.size)


def expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    source: str = "historical",
    sims: np.ndarray | None = None,
    horizon_days: int = 1,
) -> RiskResult:
    """ES por três fontes: ``historical``, ``parametric`` ou ``montecarlo``.

    As fontes histórica e paramétrica partem de retornos diários e são levadas
    ao horizonte pela raiz do tempo, como o VaR. Para ``montecarlo`` os
    cenários em ``sims`` já devem ter sido simulados no horizonte desejado.
    """
    alpha = 1.0 - confidence_level
    root_h = np.sqrt(horizon_days)
    if source == "parametric":
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        z = z_score(confidence_level)
        tail = sigma * float(stats.norm.pdf(z)) / alpha
        value = -mu * horizon_days + tail * root_h
        params = {"source": source, "mu": mu, "sigma": sigma}
    elif source == "montecarlo":
        if sims is None:
            raise ValueError("source='montecarlo' exige os cenários em `sims`")
        value, quantile, n_tail = _tail_mean(np.asarray(sims, dtype=float), alpha)
        params = {"source": source, "n_tail": n_tail, "n_sims": int(len(sims))}
    elif source == "historical":
        value, quantile, n_tail = _tail_mean(np.asarray(returns, dtype=float), alpha)
        value *= root_h
        params = {"source": source, "n_tail": n_tail, "quantile": quantile}
    else:
        raise ValueError(f"fonte de ES não suportada: {source}")
    return RiskResult(
        value=float(value),
        method="expected_shortfall",
        confidence_level=confidence_level,
        params=params,
        input_hash=sha256_of_array(returns),
        horizon_days=horizon_days,
    )
