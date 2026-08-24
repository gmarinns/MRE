"""F3.1–F3.2 — Backtesting formal do VaR.

Kupiec (cobertura incondicional), Christoffersen (independência), cobertura
condicional conjunta e o semáforo de Basileia.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def violations_mask(returns: np.ndarray, var_series: np.ndarray) -> np.ndarray:
    """Violação = retorno abaixo do **VaR previsto para aquele dia**.

    O beta comparava os retornos com a *média* da série de VaR nos testes, mas
    com a série rolante no gráfico: o número de violações do KPI não batia com
    os "X" desenhados. Aqui existe uma única definição, usada por todos.
    """
    returns = np.asarray(returns, dtype=float)
    var_series = np.broadcast_to(np.asarray(var_series, dtype=float), returns.shape)
    return returns < -var_series


def kupiec_pof(violations: np.ndarray, confidence_level: float = 0.95) -> dict:
    """Teste POF de Kupiec (1995) — a taxa de violação bate com 1-c?"""
    v = np.asarray(violations, dtype=int)
    n, x = int(v.size), int(v.sum())
    p = 1.0 - confidence_level
    pi_hat = x / n if n else 0.0

    ll_null = (n - x) * np.log(1 - p) + (x * np.log(p) if x else 0.0)
    if x == 0:
        ll_alt = 0.0            # (1-0)^n = 1 -> log-verossimilhança nula, e NÃO LR = 0
    elif x == n:
        ll_alt = 0.0
    else:
        ll_alt = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)

    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return {
        "test": "Kupiec (POF)",
        "lr_stat": float(lr),
        "p_value": float(1 - stats.chi2.cdf(lr, df=1)),
        "df": 1,
        "violations": x,
        "n_obs": n,
        "expected_violations": float(n * p),
        "violation_rate": float(pi_hat),
        "expected_rate": float(p),
        "reject_h0": bool(1 - stats.chi2.cdf(lr, df=1) < 0.05),
    }


def christoffersen_independence(violations: np.ndarray) -> dict:
    """Teste de independência: as violações se agrupam no tempo (clustering)?"""
    v = np.asarray(violations, dtype=int)
    prev, curr = v[:-1], v[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    def _ll(p, hits, total):
        if total == 0 or p <= 0.0 or p >= 1.0:
            return 0.0
        return hits * np.log(p) + (total - hits) * np.log(1 - p)

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    ll_alt = _ll(pi01, n01, n00 + n01) + _ll(pi11, n11, n10 + n11)
    ll_null = _ll(pi, n01 + n11, n00 + n01 + n10 + n11)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    return {
        "test": "Christoffersen (independência)",
        "lr_stat": float(lr),
        "p_value": p_value,
        "df": 1,
        "transitions": {"00": n00, "01": n01, "10": n10, "11": n11},
        "pi01": float(pi01), "pi11": float(pi11),
        "reject_h0": bool(p_value < 0.05),
    }


def conditional_coverage(violations: np.ndarray, confidence_level: float = 0.95) -> dict:
    """LR_cc = LR_uc + LR_ind (χ² com 2 g.l.) — o teste conjunto de Christoffersen."""
    uc = kupiec_pof(violations, confidence_level)
    ind = christoffersen_independence(violations)
    lr = uc["lr_stat"] + ind["lr_stat"]
    p_value = float(1 - stats.chi2.cdf(lr, df=2))
    return {
        "test": "Cobertura condicional (LR_cc)",
        "lr_stat": float(lr), "p_value": p_value, "df": 2,
        "reject_h0": bool(p_value < 0.05),
        "components": {"LR_uc": uc["lr_stat"], "LR_ind": ind["lr_stat"]},
    }


def basel_traffic_light(violations: np.ndarray, confidence_level: float = 0.99) -> dict:
    """Semáforo de Basileia, generalizado para n≠250 pela binomial acumulada."""
    v = np.asarray(violations, dtype=int)
    n, x = int(v.size), int(v.sum())
    p = 1.0 - confidence_level
    cum = float(stats.binom.cdf(x, n, p))
    if cum < 0.95:
        zone, color, note = "Verde", "ok", "Modelo aceitável — sem acréscimo de capital."
    elif cum < 0.9999:
        zone, color, note = "Amarela", "warn", "Violações acima do esperado — acréscimo de capital progressivo."
    else:
        zone, color, note = "Vermelha", "risk", "Modelo rejeitado — revisão obrigatória."
    return {
        "zone": zone, "color": color, "note": note,
        "violations": x, "n_obs": n, "cumulative_prob": cum,
        "reference": "Basel Committee (1996) — escalado para a amostra por binomial.",
    }


def full_backtest(returns: np.ndarray, var_series: np.ndarray, confidence_level: float) -> dict:
    mask = violations_mask(returns, var_series)
    v = mask.astype(int)
    return {
        "mask": mask,
        "kupiec": kupiec_pof(v, confidence_level),
        "independence": christoffersen_independence(v),
        "conditional": conditional_coverage(v, confidence_level),
        "basel": basel_traffic_light(v, confidence_level),
        "avg_var": float(np.nanmean(var_series)),
        "loss_when_violated": float(-np.mean(returns[mask])) if mask.any() else 0.0,
    }
