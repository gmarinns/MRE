"""VaR de carteira multi-ativo por matriz de covariância."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .es import expected_shortfall
from .types import RiskResult, sha256_of_array, z_score
from .var import historical_var


def portfolio_var(
    matrix: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> dict:
    """VaR paramétrico da carteira, sua decomposição e o ganho de diversificação.

    ``matrix`` traz os log-retornos alinhados por data, uma coluna por ativo.
    Os pesos são normalizados para somar 1.

    O VaR componente de cada ativo segue a identidade de Euler
    (Σ wᵢ·∂σ/∂wᵢ = σ), então as componentes somam exatamente o VaR da carteira.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    cov = matrix.cov().to_numpy()
    mu = matrix.mean().to_numpy()
    z = z_score(confidence_level)
    root_h = np.sqrt(horizon_days)

    sigma_p = float(np.sqrt(weights @ cov @ weights))
    mu_p = float(weights @ mu)
    var_parametric = float(-(mu_p * horizon_days + z * sigma_p * root_h))

    individual = -(mu * horizon_days + z * np.sqrt(np.diag(cov)) * root_h)
    undiversified = float(np.sum(weights * individual))

    marginal = (cov @ weights) / sigma_p if sigma_p > 0 else np.zeros_like(weights)
    component = -weights * mu * horizon_days + weights * marginal * (-z) * root_h
    share = component / component.sum() if component.sum() else component

    port_returns = matrix.to_numpy() @ weights
    result = RiskResult(
        value=var_parametric,
        method="portfolio",
        confidence_level=confidence_level,
        params={
            "weights": {
                c: float(w) for c, w in zip(matrix.columns, weights, strict=True)
            },
            "sigma_portfolio": sigma_p,
            "mu_portfolio": mu_p,
            "n_assets": int(matrix.shape[1]),
            "n_obs": int(matrix.shape[0]),
        },
        input_hash=sha256_of_array(matrix.to_numpy()),
        horizon_days=horizon_days,
    )
    return {
        "result": result,
        "var_parametric": var_parametric,
        "var_historical": historical_var(
            port_returns, confidence_level, horizon_days
        ).value,
        "es_historical": expected_shortfall(
            port_returns, confidence_level, horizon_days=horizon_days
        ).value,
        "undiversified_var": undiversified,
        "diversification_benefit": float(undiversified - var_parametric),
        "components": pd.DataFrame(
            {
                "asset": list(matrix.columns),
                "weight": weights,
                "individual_var": individual,
                "component_var": component,
                "share": share,
            }
        ),
        "portfolio_returns": port_returns,
        "correlation": matrix.corr(),
    }
