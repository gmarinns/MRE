"""Drawdown: perda acumulada desde o último topo."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(prices: np.ndarray) -> np.ndarray:
    """Para cada dia, o quanto o preço está abaixo do maior preço já visto (≤ 0)."""
    prices = np.asarray(prices, dtype=float)
    return prices / np.maximum.accumulate(prices) - 1.0


def episodes(prices: np.ndarray, dates, top: int = 5) -> pd.DataFrame:
    """Os ``top`` maiores episódios de drawdown, do mais profundo ao mais raso.

    Um episódio vai do topo (último dia antes da queda) até a recuperação (o
    primeiro dia em que o preço volta ao topo). Se o preço ainda não voltou,
    ``recovery`` fica vazio e ``recovered`` é falso.
    """
    prices = np.asarray(prices, dtype=float)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    dd = drawdown_series(prices)
    under = dd < 0
    rows = []
    i, n = 0, len(dd)
    while i < n:
        if not under[i]:
            i += 1
            continue
        start = i
        while i < n and under[i]:
            i += 1
        end = i - 1  # último dia abaixo do topo
        peak = max(start - 1, 0)
        trough = start + int(np.argmin(dd[start : end + 1]))
        recovered = i < n
        last = i if recovered else end
        rows.append(
            {
                "peak": dates[peak].date(),
                "trough": dates[trough].date(),
                "recovery": dates[i].date() if recovered else None,
                "depth": float(dd[trough]),
                "days_to_trough": int(trough - peak),
                "days_total": int(last - peak),
                "recovered": bool(recovered),
            }
        )
    columns = [
        "peak",
        "trough",
        "recovery",
        "depth",
        "days_to_trough",
        "days_total",
        "recovered",
    ]
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values("depth").head(top).reset_index(drop=True)
