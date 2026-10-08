import numpy as np
import pandas as pd
import pytest
from scipy import stats

from motor_calculo import drawdown, portfolio, stress, var


# ── Carteira ──────────────────────────────────────────────────────────────
def test_carteira_de_um_ativo_e_o_var_parametrico_do_ativo(returns):
    frame = pd.DataFrame({"A": returns})

    result = portfolio.portfolio_var(frame, np.array([1.0]), 0.95)

    assert result["var_parametric"] == pytest.approx(
        var.parametric_var(returns, 0.95).value, rel=1e-10
    )


def test_correlacao_perfeita_nao_diversifica():
    x = np.random.default_rng(11).normal(0, 0.015, 500)
    frame = pd.DataFrame({"A": x, "B": x})

    result = portfolio.portfolio_var(frame, np.array([0.5, 0.5]), 0.95)

    assert result["diversification_benefit"] == pytest.approx(0.0, abs=1e-12)


def test_ativos_independentes_diversificam(matrix):
    result = portfolio.portfolio_var(matrix, np.array([1, 1, 1]), 0.95)

    assert result["diversification_benefit"] > 0
    assert result["var_parametric"] < result["undiversified_var"]


def test_componentes_somam_o_var_da_carteira(matrix):
    result = portfolio.portfolio_var(matrix, np.array([0.5, 0.3, 0.2]), 0.95)
    components = result["components"]

    assert components["component_var"].sum() == pytest.approx(
        result["var_parametric"], rel=1e-10
    )
    assert components["share"].sum() == pytest.approx(1.0)


def test_pesos_sao_normalizados(matrix):
    a = portfolio.portfolio_var(matrix, np.array([50, 30, 20]), 0.95)
    b = portfolio.portfolio_var(matrix, np.array([0.5, 0.3, 0.2]), 0.95)

    assert a["var_parametric"] == pytest.approx(b["var_parametric"])


# ── Drawdown ──────────────────────────────────────────────────────────────
PRICES = np.array([100, 110, 99, 88, 110, 121, 115, 121, 100], dtype=float)
DATES = pd.bdate_range("2024-01-01", periods=len(PRICES))


def test_serie_de_drawdown_em_precos_conhecidos():
    series = drawdown.drawdown_series(PRICES)

    assert series[1] == 0.0  # novo topo
    assert series[3] == pytest.approx(88 / 110 - 1)  # -20%
    assert series[-1] == pytest.approx(100 / 121 - 1)


def test_episodios_identificam_topo_fundo_e_recuperacao():
    table = drawdown.episodes(PRICES, DATES, top=5)

    deepest = table.iloc[0]
    assert deepest["depth"] == pytest.approx(-0.20)
    assert deepest["peak"] == DATES[1].date()
    assert deepest["trough"] == DATES[3].date()
    assert deepest["recovery"] == DATES[4].date()
    assert deepest["days_to_trough"] == 2
    assert deepest["days_total"] == 3
    assert deepest["recovered"]


def test_episodio_em_aberto_nao_tem_data_de_recuperacao():
    table = drawdown.episodes(PRICES, DATES, top=5)

    ongoing = table[~table["recovered"]].iloc[0]
    assert ongoing["peak"] == DATES[7].date()
    assert ongoing["recovery"] is None
    assert len(table) == 3


def test_precos_sempre_subindo_nao_tem_episodio():
    table = drawdown.episodes(np.arange(1.0, 20.0), pd.bdate_range("2024", periods=19))

    assert table.empty


# ── Stress ────────────────────────────────────────────────────────────────
def test_choque_reescala_o_desvio_e_desloca_a_media(returns):
    shocked = stress.shock_returns(returns, vol_multiplier=2.0, mean_shift=-0.01)

    assert shocked.std(ddof=1) == pytest.approx(2.0 * returns.std(ddof=1))
    assert shocked.mean() == pytest.approx(returns.mean() - 0.01)


def test_cenario_base_e_o_var_historico(returns):
    base = stress.stressed_var(returns, 0.95, vol_multiplier=1.0, mean_shift=0.0)

    assert base.value == pytest.approx(var.historical_var(returns, 0.95).value)


def test_cenarios_ficam_mais_severos_em_ordem(returns):
    table = stress.scenario_table(returns, 0.95)

    assert list(table["scenario"]) == ["Base", "Moderado", "Severo", "Extremo"]
    assert table["var"].is_monotonic_increasing
    assert (table["es"] >= table["var"]).all()


def test_dobrar_a_volatilidade_quase_dobra_o_var_de_uma_normal(returns):
    base = stress.stressed_var(returns, 0.95).value
    doubled = stress.stressed_var(returns, 0.95, vol_multiplier=2.0).value
    # Analítico: VaR = -(μ + zσ); com σ dobrado o VaR cresce em -zσ
    expected = base - stats.norm.ppf(0.05) * returns.std(ddof=1)

    assert doubled == pytest.approx(expected, rel=0.05)


def test_pior_janela_encontra_o_choque_plantado():
    series = np.full(100, 0.001)
    series[40:45] = -0.03
    dates = pd.bdate_range("2024-01-01", periods=100)

    table = stress.worst_windows(series, dates, horizons=(1, 5))
    five = table[table["horizon_days"] == 5].iloc[0]

    assert five["start"] == dates[40].date()
    assert five["end"] == dates[44].date()
    assert five["worst_return"] == pytest.approx(np.expm1(-0.15))
