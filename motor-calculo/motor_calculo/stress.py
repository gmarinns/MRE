"""Stress testing: choques hipotéticos e piores janelas observadas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .es import expected_shortfall
from .types import RiskResult, sha256_of_array

# Cenários hipotéticos: (nome, multiplicador de volatilidade, choque na média diária).
# São choques de sensibilidade, não calibrados em episódios históricos específicos.
SCENARIOS: tuple[tuple[str, float, float], ...] = (
    ("Base", 1.0, 0.0),
    ("Moderado", 1.5, -0.002),
    ("Severo", 2.0, -0.005),
    ("Extremo", 3.0, -0.010),
)


def shock_returns(
    returns: np.ndarray, vol_multiplier: float = 1.0, mean_shift: float = 0.0
) -> np.ndarray:
    """Reescala os desvios em torno da média e desloca a média."""
    returns = np.asarray(returns, dtype=float)
    center = returns.mean()
    return (returns - center) * vol_multiplier + center + mean_shift


def stressed_var(
    returns: np.ndarray,
    confidence_level: float,
    vol_multiplier: float = 1.0,
    mean_shift: float = 0.0,
    horizon_days: int = 1,
) -> RiskResult:
    """VaR histórico recalculado sobre a série chocada."""
    shocked = shock_returns(returns, vol_multiplier, mean_shift)
    alpha = 1.0 - confidence_level
    value = -float(np.percentile(shocked, alpha * 100.0)) * np.sqrt(horizon_days)
    return RiskResult(
        value=float(value),
        method="stressed",
        confidence_level=confidence_level,
        params={"vol_multiplier": vol_multiplier, "mean_shift": mean_shift},
        input_hash=sha256_of_array(returns),
        horizon_days=horizon_days,
    )


def scenario_table(
    returns: np.ndarray, confidence_level: float, horizon_days: int = 1
) -> pd.DataFrame:
    """VaR e ES sob cada cenário de ``SCENARIOS``."""
    rows = []
    for name, vol_multiplier, mean_shift in SCENARIOS:
        shocked = shock_returns(returns, vol_multiplier, mean_shift)
        var = stressed_var(
            returns, confidence_level, vol_multiplier, mean_shift, horizon_days
        )
        es = expected_shortfall(
            shocked, confidence_level, horizon_days=horizon_days
        ).value
        rows.append(
            {
                "scenario": name,
                "vol_multiplier": vol_multiplier,
                "mean_shift": mean_shift,
                "var": var.value,
                "es": es,
            }
        )
    return pd.DataFrame(rows)


def worst_windows(
    returns: np.ndarray, dates, horizons: tuple[int, ...] = (1, 5, 10, 21)
) -> pd.DataFrame:
    """A pior perda acumulada realmente observada em janelas de h dias."""
    returns = np.asarray(returns, dtype=float)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    rows = []
    for h in horizons:
        if len(returns) <= h:
            continue
        accumulated = np.convolve(returns, np.ones(h), mode="valid")
        start = int(np.argmin(accumulated))
        rows.append(
            {
                "horizon_days": h,
                "worst_return": float(np.expm1(accumulated[start])),
                "start": dates[start].date(),
                "end": dates[start + h - 1].date(),
            }
        )
    return pd.DataFrame(rows)
