"""Ponte entre a interface e o motor de cálculo.

Todo cálculo exibido no painel passa por este módulo. É aqui que fica o
cache do Streamlit (o motor não conhece Streamlit) e é aqui que, no futuro,
entra o registro de proveniência de cada execução.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

import dados
from motor_calculo import (
    backtesting,
    drawdown,
    es,
    montecarlo,
    portfolio,
    stress,
    var,
    volatility,
)
from motor_calculo.types import RiskResult

METODOS = ["historical", "parametric", "ewma", "cornish_fisher", "montecarlo"]
METODOS_ROLANTES = ["historical", "parametric", "ewma"]


@dataclass
class Contexto:
    """Tudo o que as páginas precisam sobre o ativo em foco."""

    ticker: str
    ativos: list[str]
    periodo: str
    cotacoes: dados.Cotacoes
    retornos: np.ndarray
    datas: pd.Series
    confianca: float
    horizonte: int
    posicao: float
    garch: dict
    var: dict[str, RiskResult]
    es: dict[str, RiskResult]
    simulacoes: np.ndarray

    def em_reais(self, fracao: float) -> float:
        return fracao * self.posicao


@st.cache_data(ttl=3600, show_spinner=False)
def ajustar_garch(retornos: np.ndarray) -> dict:
    return volatility.fit_garch(retornos)


@st.cache_data(ttl=3600, show_spinner=False)
def _calcular(
    retornos: np.ndarray,
    confianca: float,
    horizonte: int,
    n_sims: int,
    dist: str,
    vol_dinamica: bool,
    semente: int,
) -> tuple[dict, dict, np.ndarray]:
    garch = ajustar_garch(retornos)
    resultados = {
        "historical": var.historical_var(retornos, confianca, horizonte),
        "parametric": var.parametric_var(retornos, confianca, horizonte),
        "ewma": var.ewma_var(retornos, confianca, horizon_days=horizonte),
        "cornish_fisher": var.cornish_fisher_var(retornos, confianca, horizonte),
    }
    resultados["montecarlo"], sims = montecarlo.montecarlo_var(
        retornos,
        confianca,
        n_sims=n_sims,
        n_days=horizonte,
        dist=dist,
        seed=semente,
        vol_dynamics=vol_dinamica,
        garch=garch,
    )
    perdas = {
        fonte: es.expected_shortfall(
            retornos, confianca, fonte, sims=sims, horizon_days=horizonte
        )
        for fonte in ("historical", "parametric", "montecarlo")
    }
    return resultados, perdas, sims


def contexto() -> Contexto:
    """Monta o contexto do ativo em foco a partir da barra lateral."""
    s = st.session_state
    periodo = s["periodo"] or "2y"  # o controle segmentado permite desmarcar
    cotacoes = dados.carregar(s["foco"], periodo)
    retornos = dados.log_retornos(cotacoes.frame)
    resultados, perdas, sims = _calcular(
        retornos,
        s["confianca"],
        s["horizonte"],
        s["mc_sims"],
        s["mc_dist"],
        s["mc_vol"],
        s["mc_semente"],
    )
    return Contexto(
        ticker=s["foco"],
        ativos=list(s["ativos"]),
        periodo=periodo,
        cotacoes=cotacoes,
        retornos=retornos,
        datas=dados.datas_dos_retornos(cotacoes.frame),
        confianca=s["confianca"],
        horizonte=s["horizonte"],
        posicao=float(s["posicao"]),
        garch=ajustar_garch(retornos),
        var=resultados,
        es=perdas,
        simulacoes=sims,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def var_rolante(
    retornos: np.ndarray, janela: int, confianca: float, metodo: str
) -> np.ndarray:
    return var.rolling_var_series(retornos, janela, confianca, metodo)


@st.cache_data(ttl=3600, show_spinner=False)
def backtest(
    retornos: np.ndarray, janela: int, confianca: float, metodo: str
) -> dict | None:
    """Backtest do estimador em janela rolante. None se faltar histórico."""
    serie = var.rolling_var_series(retornos, janela, confianca, metodo)
    validos = ~np.isnan(serie)
    if validos.sum() < 30:
        return None
    relatorio = backtesting.full_backtest(retornos[validos], serie[validos], confianca)
    relatorio["validos"] = validos
    relatorio["var_series"] = serie[validos]
    relatorio["returns"] = retornos[validos]
    return relatorio


@st.cache_data(ttl=3600, show_spinner=False)
def leque_montecarlo(
    retornos: np.ndarray,
    confianca: float,
    n_sims: int,
    n_dias: int,
    dist: str,
    vol_dinamica: bool,
    semente: int,
) -> pd.DataFrame:
    """Percentis das trajetórias simuladas, dia a dia (para o gráfico em leque)."""
    trajetorias = montecarlo.simulate_paths(
        float(np.mean(retornos)),
        ajustar_garch(retornos),
        n_sims=n_sims,
        n_days=n_dias,
        dist=dist,
        seed=semente,
        vol_dynamics=vol_dinamica,
        return_paths=True,
    )
    niveis = [(1 - confianca) * 100, 5, 25, 50, 75, 95]
    p = np.percentile(trajetorias, niveis, axis=0)
    return pd.DataFrame(
        {
            "dia": np.arange(1, n_dias + 1),
            "var": p[0],
            "p05": p[1],
            "p25": p[2],
            "p50": p[3],
            "p75": p[4],
            "p95": p[5],
        }
    )


@st.cache_data(ttl=3600, show_spinner=False)
def convergencia(
    retornos: np.ndarray, confianca: float, dist: str, semente: int
) -> pd.DataFrame:
    tamanhos = (500, 1_000, 5_000, 10_000, 50_000, 100_000)
    return montecarlo.montecarlo_convergence(
        retornos,
        confianca,
        tamanhos,
        dist=dist,
        seed=semente,
        garch=ajustar_garch(retornos),
    )


def carteira(matriz: pd.DataFrame, pesos: np.ndarray, confianca: float, h: int) -> dict:
    return portfolio.portfolio_var(matriz, pesos, confianca, h)


def atual() -> Contexto:
    """Contexto já montado pelo app.py nesta execução."""
    return st.session_state["_ctx"]


def var_ewma(retornos: np.ndarray, confianca: float, lam: float, h: int) -> RiskResult:
    return var.ewma_var(retornos, confianca, lam=lam, horizon_days=h)


def pesos_ewma(n: int, lam: float) -> np.ndarray:
    """Pesos do mais recente para o mais antigo."""
    return volatility.ewma_weights(n, lam)[::-1]


def sigma_ewma(retornos: np.ndarray, lam: float) -> np.ndarray:
    return volatility.ewma_sigma_series(retornos, lam)


def previsao_garch(garch: dict, n_dias: int) -> np.ndarray:
    return volatility.garch_forecast(garch, n_dias)


def curva_de_confianca(retornos: np.ndarray, h: int) -> pd.DataFrame:
    """VaR e ES históricos em vários níveis de confiança."""
    niveis = (0.90, 0.925, 0.95, 0.975, 0.99, 0.995)
    return pd.DataFrame(
        {
            "confianca": niveis,
            "var": [var.historical_var(retornos, c, h).value for c in niveis],
            "es": [
                es.expected_shortfall(retornos, c, horizon_days=h).value for c in niveis
            ],
        }
    )


def serie_drawdown(precos: np.ndarray) -> np.ndarray:
    return drawdown.drawdown_series(precos)


def episodios_drawdown(precos: np.ndarray, datas, top: int = 5) -> pd.DataFrame:
    return drawdown.episodes(precos, datas, top)


def cenarios(retornos: np.ndarray, confianca: float, h: int) -> pd.DataFrame:
    return stress.scenario_table(retornos, confianca, h)


def var_estressado(
    retornos: np.ndarray, confianca: float, vol: float, media: float, h: int
) -> RiskResult:
    return stress.stressed_var(retornos, confianca, vol, media, h)


def piores_janelas(retornos: np.ndarray, datas) -> pd.DataFrame:
    return stress.worst_windows(retornos, datas)
