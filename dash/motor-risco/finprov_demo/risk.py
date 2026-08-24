"""Motor de risco — VaR (5 estimadores), Expected Shortfall, GARCH e carteira.

Todas as funções seguem o contrato ``RiskResult`` da proposta: devolvem o
número **e** os metadados que a camada de proveniência precisa registrar.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

try:  # pragma: no cover - dependência opcional
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    ARCH_AVAILABLE = False

# Namespace fixo -> o mesmo cálculo, com os mesmos insumos, gera o mesmo run_id.
_RUN_NAMESPACE = uuid.UUID("6f1b0c9a-6a1e-5f6b-9d2a-1f7c0b3e4d55")


# ══════════════════════════════════════════════════════════════════════════
#  Contrato
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class RiskResult:
    value: float                      # perda potencial, em fração (0.032 = 3,2%)
    method: str                       # historical | parametric | ewma | cornish_fisher | montecarlo | ...
    confidence_level: float
    params: dict = field(default_factory=dict)
    input_hash: str = ""
    horizon_days: int = 1
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def run_id(self) -> str:
        """Identificador determinístico da execução (insumo + método + parâmetros)."""
        payload = "|".join(
            [
                self.method,
                self.input_hash,
                f"{self.confidence_level:.6f}",
                f"{self.horizon_days}",
                repr(sorted((k, _stable(v)) for k, v in self.params.items())),
            ]
        )
        return str(uuid.uuid5(_RUN_NAMESPACE, payload))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_id"] = self.run_id
        return d


def _stable(value):
    return round(value, 10) if isinstance(value, float) else value


def sha256_of_array(arr: np.ndarray) -> str:
    """Hash dos insumos — a 'certidão de nascimento' do número de risco."""
    arr = np.ascontiguousarray(np.round(np.asarray(arr, dtype=float), 10))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _z(confidence_level: float) -> float:
    return float(stats.norm.ppf(1.0 - confidence_level))


# ══════════════════════════════════════════════════════════════════════════
#  F1.1–F1.2 — VaR de fórmula fechada e empírico
# ══════════════════════════════════════════════════════════════════════════
def historical_var(returns: np.ndarray, confidence_level: float = 0.95,
                   horizon_days: int = 1) -> RiskResult:
    alpha = 1.0 - confidence_level
    q = float(np.percentile(returns, alpha * 100.0))
    value = -q * np.sqrt(horizon_days)
    return RiskResult(
        value=float(value), method="historical", confidence_level=confidence_level,
        params={"n_obs": int(len(returns)), "quantile": q},
        input_hash=sha256_of_array(returns), horizon_days=horizon_days,
    )


def parametric_var(returns: np.ndarray, confidence_level: float = 0.95,
                   horizon_days: int = 1) -> RiskResult:
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    value = -(mu * horizon_days + _z(confidence_level) * sigma * np.sqrt(horizon_days))
    return RiskResult(
        value=float(value), method="parametric", confidence_level=confidence_level,
        params={"mu": mu, "sigma": sigma, "z": _z(confidence_level), "n_obs": int(len(returns))},
        input_hash=sha256_of_array(returns), horizon_days=horizon_days,
    )


def ewma_sigma(returns: np.ndarray, lam: float = 0.94) -> float:
    """Volatilidade EWMA (RiskMetrics). σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}."""
    r2 = np.asarray(returns, dtype=float) ** 2
    n = len(r2)
    weights = lam ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()
    return float(np.sqrt(np.sum(weights * r2)))


def ewma_var(returns: np.ndarray, confidence_level: float = 0.95,
             lam: float = 0.94, horizon_days: int = 1) -> RiskResult:
    sigma = ewma_sigma(returns, lam)
    value = -(_z(confidence_level) * sigma * np.sqrt(horizon_days))
    return RiskResult(
        value=float(value), method="ewma", confidence_level=confidence_level,
        params={"lambda": lam, "sigma_ewma": sigma, "n_obs": int(len(returns))},
        input_hash=sha256_of_array(returns), horizon_days=horizon_days,
    )


def cornish_fisher_var(returns: np.ndarray, confidence_level: float = 0.95,
                       horizon_days: int = 1) -> RiskResult:
    """VaR paramétrico corrigido por assimetria e curtose (expansão de Cornish-Fisher)."""
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    s = float(stats.skew(returns))
    k = float(stats.kurtosis(returns, fisher=False))
    z = _z(confidence_level)
    z_cf = (
        z
        + (z**2 - 1) * s / 6.0
        + (z**3 - 3 * z) * (k - 3.0) / 24.0
        - (2 * z**3 - 5 * z) * s**2 / 36.0
    )
    value = -(mu * horizon_days + z_cf * sigma * np.sqrt(horizon_days))
    return RiskResult(
        value=float(value), method="cornish_fisher", confidence_level=confidence_level,
        params={"mu": mu, "sigma": sigma, "skew": s, "kurtosis": k, "z_cf": float(z_cf)},
        input_hash=sha256_of_array(returns), horizon_days=horizon_days,
    )


# ══════════════════════════════════════════════════════════════════════════
#  F2.1 — GARCH(1,1)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fit_garch(returns: np.ndarray) -> dict:
    """Ajusta GARCH(1,1) por máxima verossimilhança (biblioteca ``arch``).

    Em cache: o beta reajustava o modelo a cada interação de widget, o que
    congelava a interface por segundos a cada clique.
    """
    fallback = {
        "available": False,
        "omega": float("nan"), "alpha": float("nan"), "beta": float("nan"),
        "persistence": float("nan"), "loglik": float("nan"), "aic": float("nan"),
        "sigma_next": float(np.std(returns[-21:], ddof=1)),
        "conditional_vol": np.full(len(returns), float(np.std(returns, ddof=1))),
        "note": "arch indisponível ou estimação falhou — usando desvio-padrão de 21 dias.",
    }
    if not ARCH_AVAILABLE or len(returns) < 60:
        return fallback
    try:
        scaled = np.asarray(returns, dtype=float) * 100.0
        res = arch_model(scaled, mean="Constant", vol="Garch", p=1, q=1, dist="normal").fit(
            disp="off", show_warning=False
        )
        p = res.params
        fc = res.forecast(horizon=1, reindex=False)
        return {
            "available": True,
            "omega": float(p.get("omega", np.nan)),
            "alpha": float(p.get("alpha[1]", np.nan)),
            "beta": float(p.get("beta[1]", np.nan)),
            "mu_scaled": float(p.get("mu", 0.0)),
            "persistence": float(p.get("alpha[1]", 0.0) + p.get("beta[1]", 0.0)),
            "loglik": float(res.loglikelihood),
            "aic": float(res.aic),
            "sigma_next": float(np.sqrt(fc.variance.values[-1, 0]) / 100.0),
            "conditional_vol": np.asarray(res.conditional_volatility, dtype=float) / 100.0,
            "note": "",
        }
    except Exception as exc:  # pragma: no cover - depende do otimizador
        fallback["note"] = f"estimação GARCH falhou ({type(exc).__name__}) — fallback de 21 dias."
        return fallback


def garch_variance_path(garch: dict, last_return: float, n_days: int) -> np.ndarray:
    """σ_t previsto para t = 1..n_days (em fração), pela recursão do GARCH."""
    if not garch.get("available"):
        return np.full(n_days, garch["sigma_next"])
    omega, alpha, beta = garch["omega"], garch["alpha"], garch["beta"]
    h = (garch["sigma_next"] * 100.0) ** 2
    path = []
    for _ in range(n_days):
        path.append(np.sqrt(h) / 100.0)
        h = omega + (alpha + beta) * h  # E[ε²] = h no horizonte à frente
    return np.asarray(path)


# ══════════════════════════════════════════════════════════════════════════
#  F2.2 — Monte Carlo vetorizado (GBM + GARCH)
# ══════════════════════════════════════════════════════════════════════════
def simulate_paths(
    mu: float,
    garch: dict,
    n_sims: int = 10_000,
    n_days: int = 1,
    dist: str = "normal",
    df_t: float = 5.0,
    seed: int = 42,
    vol_dynamics: bool = True,
    last_return: float = 0.0,
) -> np.ndarray:
    """Retorna o array (n_sims,) de log-retornos acumulados no horizonte.

    ``vol_dynamics=True`` propaga a recursão do GARCH **dentro** de cada
    trajetória (σ estocástico), em vez de usar um σ constante como no beta.
    O laço é sobre ``n_days`` (≤ 22), e cada passo é vetorizado sobre as
    ``n_sims`` trajetórias — o custo continua sendo O(n_sims·n_days) em C.
    """
    rng = np.random.default_rng(seed)
    if dist == "t":
        z = rng.standard_t(df_t, size=(n_sims, n_days))
        z /= np.sqrt(df_t / (df_t - 2.0))  # padroniza para variância 1
    else:
        z = rng.standard_normal((n_sims, n_days))

    if garch.get("available") and vol_dynamics:
        omega, alpha, beta = garch["omega"], garch["alpha"], garch["beta"]
        h = np.full(n_sims, (garch["sigma_next"] * 100.0) ** 2)
        total = np.zeros(n_sims)
        for d in range(n_days):
            eps = np.sqrt(h) * z[:, d]           # choque na escala ×100
            total += mu + eps / 100.0
            h = omega + alpha * eps**2 + beta * h
        return total

    sigma_path = garch_variance_path(garch, last_return, n_days)
    return (mu + sigma_path * z).sum(axis=1)


def montecarlo_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    n_sims: int = 10_000,
    n_days: int = 1,
    dist: str = "normal",
    df_t: float = 5.0,
    seed: int = 42,
    vol_dynamics: bool = True,
) -> tuple[RiskResult, np.ndarray]:
    garch = fit_garch(returns)
    mu = float(np.mean(returns))
    sims = simulate_paths(
        mu, garch, n_sims=n_sims, n_days=n_days, dist=dist, df_t=df_t,
        seed=seed, vol_dynamics=vol_dynamics, last_return=float(returns[-1]),
    )
    alpha = 1.0 - confidence_level
    value = -float(np.percentile(sims, alpha * 100.0))
    # Erro padrão do quantil empírico (Bahadur) — quanto o número "treme" com a semente.
    density = max(stats.gaussian_kde(sims[: min(len(sims), 20_000)]).evaluate([-value])[0], 1e-9)
    se = float(np.sqrt(alpha * (1 - alpha) / n_sims) / density)
    result = RiskResult(
        value=value, method="montecarlo", confidence_level=confidence_level,
        params={
            "n_sims": int(n_sims), "n_days": int(n_days), "dist": dist,
            "df_t": df_t if dist == "t" else None, "seed": int(seed),
            "vol_dynamics": bool(vol_dynamics and garch.get("available")),
            "sigma_garch_next": float(garch["sigma_next"]),
            "garch_persistence": float(garch.get("persistence", float("nan"))),
            "mc_std_error": se,
        },
        input_hash=sha256_of_array(returns), horizon_days=n_days,
    )
    return result, sims


def montecarlo_convergence(
    returns: np.ndarray, confidence_level: float, sizes: tuple[int, ...],
    dist: str = "normal", seed: int = 42,
) -> pd.DataFrame:
    """VaR estimado em função de n_sims — mostra a convergência do estimador."""
    garch = fit_garch(returns)
    mu = float(np.mean(returns))
    alpha = 1.0 - confidence_level
    rows = []
    for n in sizes:
        sims = simulate_paths(mu, garch, n_sims=int(n), n_days=1, dist=dist, seed=seed)
        var = -float(np.percentile(sims, alpha * 100.0))
        rows.append({"n_sims": int(n), "var": var,
                     "erro_padrao": float(np.std(sims, ddof=1) / np.sqrt(n))})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  F2.3 — Expected Shortfall (as três fontes)
# ══════════════════════════════════════════════════════════════════════════
def expected_shortfall(
    returns: np.ndarray, confidence_level: float = 0.95,
    source: str = "historical", sims: np.ndarray | None = None,
) -> RiskResult:
    alpha = 1.0 - confidence_level
    if source == "parametric":
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        z = _z(confidence_level)
        value = -mu + sigma * float(stats.norm.pdf(z)) / alpha
        params = {"source": "parametric", "mu": mu, "sigma": sigma}
    elif source == "montecarlo" and sims is not None:
        q = float(np.percentile(sims, alpha * 100.0))
        tail = sims[sims <= q]
        value = -float(np.mean(tail)) if tail.size else -q
        params = {"source": "montecarlo", "n_tail": int(tail.size), "n_sims": int(sims.size)}
    else:
        q = float(np.percentile(returns, alpha * 100.0))
        tail = returns[returns <= q]  # <= o quantil (e não <= -VaR): a cauda nunca fica vazia
        value = -float(np.mean(tail)) if tail.size else -q
        params = {"source": "historical", "n_tail": int(tail.size), "quantile": q}
    return RiskResult(
        value=float(value), method="expected_shortfall", confidence_level=confidence_level,
        params=params, input_hash=sha256_of_array(returns),
    )


# ══════════════════════════════════════════════════════════════════════════
#  F1.3 — séries rolantes (vetorizado)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def rolling_var_series(
    returns: np.ndarray, window: int, confidence_level: float, method: str = "historical",
    lam: float = 0.94,
) -> np.ndarray:
    """VaR previsto para o dia *i* usando **apenas** os dados até *i-1*.

    Vetorizado com ``sliding_window_view``: o beta refazia um ``np.percentile``
    por dia dentro de um laço Python (O(n·w·log w) com overhead de intérprete).
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    windows = sliding_window_view(returns, window)[: n - window]  # janela que termina em i-1
    alpha = 1.0 - confidence_level
    if method == "historical":
        out[window:] = -np.percentile(windows, alpha * 100.0, axis=1)
    elif method == "parametric":
        mu = windows.mean(axis=1)
        sigma = windows.std(axis=1, ddof=1)
        out[window:] = -(mu + _z(confidence_level) * sigma)
    elif method == "ewma":
        w = lam ** np.arange(window - 1, -1, -1)
        w /= w.sum()
        sigma = np.sqrt((windows**2 * w).sum(axis=1))
        out[window:] = -(_z(confidence_level) * sigma)
    else:
        raise ValueError(f"método rolante não suportado: {method}")
    return out


def rolling_volatility(returns: np.ndarray, window: int = 21, annualize: bool = True) -> np.ndarray:
    s = pd.Series(returns).rolling(window).std(ddof=1).to_numpy()
    return s * np.sqrt(252) if annualize else s


def drawdown_series(prices: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(prices)
    return prices / peak - 1.0


# ══════════════════════════════════════════════════════════════════════════
#  F3.3 — VaR de carteira multi-ativo
# ══════════════════════════════════════════════════════════════════════════
def portfolio_var(
    matrix: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> dict:
    """VaR paramétrico por matriz de covariância + VaR histórico da carteira.

    Devolve também o **benefício de diversificação**: a soma dos VaR
    individuais (carteira perfeitamente correlacionada) menos o VaR da carteira.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    cov = matrix.cov().to_numpy()
    mu_vec = matrix.mean().to_numpy()

    sigma_p = float(np.sqrt(weights @ cov @ weights))
    mu_p = float(weights @ mu_vec)
    z = _z(confidence_level)
    var_param = -(mu_p * horizon_days + z * sigma_p * np.sqrt(horizon_days))

    port_returns = matrix.to_numpy() @ weights
    var_hist = historical_var(port_returns, confidence_level, horizon_days)
    es_hist = expected_shortfall(port_returns, confidence_level)

    individual = np.array(
        [-(mu_vec[i] * horizon_days + z * np.sqrt(cov[i, i]) * np.sqrt(horizon_days))
         for i in range(len(weights))]
    )
    undiversified = float(np.sum(np.abs(weights) * individual))

    # Contribuição marginal e componente de risco.
    # Pela identidade de Euler, sum_i w_i * dsigma/dw_i = sigma_p; somando também a
    # parcela de média, as componentes reproduzem exatamente o VaR da carteira.
    marginal = (cov @ weights) / sigma_p if sigma_p > 0 else np.zeros_like(weights)
    component = (
        -weights * mu_vec * horizon_days
        + weights * marginal * (-z) * np.sqrt(horizon_days)
    )

    result = RiskResult(
        value=float(var_param), method="portfolio", confidence_level=confidence_level,
        params={
            "weights": {c: float(w) for c, w in zip(matrix.columns, weights)},
            "sigma_portfolio": sigma_p, "mu_portfolio": mu_p,
            "n_assets": int(matrix.shape[1]), "n_obs": int(matrix.shape[0]),
        },
        input_hash=sha256_of_array(matrix.to_numpy()), horizon_days=horizon_days,
    )
    return {
        "result": result,
        "var_parametric": float(var_param),
        "var_historical": float(var_hist.value),
        "es_historical": float(es_hist.value),
        "undiversified_var": undiversified,
        "diversification_benefit": float(undiversified - var_param),
        "component_var": pd.DataFrame(
            {"ativo": list(matrix.columns), "peso": weights,
             "var_individual": individual, "var_componente": component,
             "contribuicao_%": component / component.sum() if component.sum() else component}
        ),
        "portfolio_returns": port_returns,
        "cov": pd.DataFrame(cov, index=matrix.columns, columns=matrix.columns),
    }


# ══════════════════════════════════════════════════════════════════════════
#  Stress testing
# ══════════════════════════════════════════════════════════════════════════
def worst_windows(returns: np.ndarray, dates: pd.Series, horizons=(1, 5, 10, 21)) -> pd.DataFrame:
    rows = []
    for h in horizons:
        if len(returns) <= h:
            continue
        agg = np.convolve(returns, np.ones(h), mode="valid")
        idx = int(np.argmin(agg))
        rows.append(
            {
                "horizonte (dias)": h,
                "pior retorno acumulado": float(np.expm1(agg[idx])),
                "início": pd.to_datetime(dates.iloc[idx]).date(),
                "fim": pd.to_datetime(dates.iloc[idx + h - 1]).date(),
            }
        )
    return pd.DataFrame(rows)


def stressed_var(
    returns: np.ndarray, confidence_level: float, vol_multiplier: float = 1.0,
    mean_shift: float = 0.0, horizon_days: int = 1,
) -> RiskResult:
    """VaR sob choque: volatilidade multiplicada e/ou deslocamento da média."""
    mu = float(np.mean(returns)) + mean_shift
    sigma = float(np.std(returns, ddof=1)) * vol_multiplier
    shocked = (returns - np.mean(returns)) * vol_multiplier + mu
    alpha = 1.0 - confidence_level
    value = -float(np.percentile(shocked, alpha * 100.0)) * np.sqrt(horizon_days)
    return RiskResult(
        value=value, method="historical", confidence_level=confidence_level,
        params={"stress": True, "vol_multiplier": vol_multiplier,
                "mean_shift": mean_shift, "sigma_stressed": sigma},
        input_hash=sha256_of_array(returns), horizon_days=horizon_days,
    )
