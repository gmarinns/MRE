import numpy as np
import pytest

from motor_calculo import volatility


def test_pesos_ewma_somam_um_e_privilegiam_o_dado_recente():
    weights = volatility.ewma_weights(100, lam=0.94)

    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(weights) > 0)  # do mais antigo ao mais recente
    assert weights[-1] / weights[-2] == pytest.approx(1 / 0.94)


def test_ewma_com_lambda_quase_zero_olha_so_o_ultimo_retorno():
    series = np.array([0.01, -0.02, 0.03, -0.04])

    assert volatility.ewma_sigma(series, lam=1e-12) == pytest.approx(0.04, rel=1e-5)


def test_serie_ewma_segue_a_recursao_do_riskmetrics():
    series = np.array([0.01, -0.02, 0.015, 0.03])
    lam = 0.9
    variance = series[0] ** 2
    expected = [variance]
    for r in series[1:]:
        variance = lam * variance + (1 - lam) * r**2
        expected.append(variance)

    result = volatility.ewma_sigma_series(series, lam)

    assert np.allclose(result, np.sqrt(expected))


def test_volatilidade_rolante_anualiza_pela_raiz_de_252():
    series = np.random.default_rng(1).normal(0, 0.01, 100)

    daily = volatility.rolling_volatility(series, 21, annualize=False)
    annual = volatility.rolling_volatility(series, 21, annualize=True)

    assert np.isnan(daily[:20]).all()
    assert daily[20] == pytest.approx(series[:21].std(ddof=1))
    assert annual[-1] == pytest.approx(daily[-1] * np.sqrt(252))


def test_garch_recupera_parametros_de_uma_serie_simulada():
    # Simula um GARCH(1,1) com parâmetros conhecidos e confere a estimação
    rng = np.random.default_rng(99)
    omega, alpha, beta = 0.05, 0.10, 0.85  # escala ×100, como na estimação
    n = 6000
    h = omega / (1 - alpha - beta)
    series = np.empty(n)
    for t in range(n):
        eps = np.sqrt(h) * rng.standard_normal()
        series[t] = eps / 100.0
        h = omega + alpha * eps**2 + beta * h

    fit = volatility.fit_garch(series)

    assert fit["available"]
    assert fit["alpha"] == pytest.approx(alpha, abs=0.03)
    assert fit["beta"] == pytest.approx(beta, abs=0.05)
    assert fit["persistence"] < 1.0


def test_garch_com_poucos_dados_cai_no_fallback_sem_quebrar():
    series = np.random.default_rng(1).normal(0, 0.01, 30)

    fit = volatility.fit_garch(series)

    assert not fit["available"]
    assert fit["sigma_next"] > 0
    assert "60" in fit["note"]


def test_previsao_garch_converge_para_a_volatilidade_de_longo_prazo():
    garch = {
        "available": True,
        "omega": 0.05,
        "alpha": 0.10,
        "beta": 0.85,
        "persistence": 0.95,
        "sigma_next": 0.03,  # começa bem acima do longo prazo (0,01)
    }
    long_run = np.sqrt(0.05 / (1 - 0.95)) / 100.0

    path = volatility.garch_forecast(garch, 400)

    assert path[0] == pytest.approx(0.03)
    assert np.all(np.diff(path) < 0)  # desce monotonicamente
    assert path[-1] == pytest.approx(long_run, rel=1e-3)
