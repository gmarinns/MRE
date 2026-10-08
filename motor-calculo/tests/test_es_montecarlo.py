import numpy as np
import pytest
from scipy import stats

from motor_calculo import es, montecarlo, var
from motor_calculo.volatility import fit_garch


def test_es_parametrico_bate_com_a_formula_analitica(returns):
    alpha = 0.05
    mu, sigma = returns.mean(), returns.std(ddof=1)
    expected = -mu + sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha

    result = es.expected_shortfall(returns, 0.95, "parametric")

    assert result.value == pytest.approx(expected, rel=1e-12)


def test_es_historico_em_amostra_de_resposta_conhecida():
    # 100 valores; a cauda de 5% são os cinco menores e mais o quantil interpolado
    sample = np.arange(-50, 50) / 100.0
    tail = sample[sample <= np.percentile(sample, 5.0)]

    result = es.expected_shortfall(sample, 0.95, "historical")

    assert result.value == pytest.approx(-tail.mean())
    assert result.params["n_tail"] == 5


def test_es_nunca_e_menor_que_o_var(returns):
    for level in (0.90, 0.95, 0.99):
        var_value = var.historical_var(returns, level).value
        es_value = es.expected_shortfall(returns, level, "historical").value
        assert es_value >= var_value


def test_es_montecarlo_exige_os_cenarios(returns):
    with pytest.raises(ValueError, match="sims"):
        es.expected_shortfall(returns, 0.95, "montecarlo")


def test_montecarlo_converge_para_a_formula_fechada(returns):
    # Sem dinâmica de volatilidade, 1 dia e choques normais, o VaR simulado tem
    # de convergir para a fórmula fechada com o σ que o GARCH previu.
    garch = fit_garch(returns)
    expected = -(returns.mean() + stats.norm.ppf(0.05) * garch["sigma_next"])

    result, _ = montecarlo.montecarlo_var(
        returns, 0.95, n_sims=200_000, seed=7, vol_dynamics=False, garch=garch
    )

    assert result.value == pytest.approx(expected, rel=0.03)


def test_montecarlo_e_reprodutivel_com_a_mesma_semente(returns):
    garch = fit_garch(returns)

    a, sims_a = montecarlo.montecarlo_var(returns, n_sims=5_000, seed=42, garch=garch)
    b, sims_b = montecarlo.montecarlo_var(returns, n_sims=5_000, seed=42, garch=garch)
    c, _ = montecarlo.montecarlo_var(returns, n_sims=5_000, seed=43, garch=garch)

    assert np.array_equal(sims_a, sims_b)
    assert a.run_id == b.run_id
    assert c.value != a.value


def test_trajetorias_terminam_no_retorno_acumulado(returns):
    garch = fit_garch(returns)
    kwargs = dict(mu=0.0, garch=garch, n_sims=500, n_days=10, seed=1)

    paths = montecarlo.simulate_paths(**kwargs, return_paths=True)
    totals = montecarlo.simulate_paths(**kwargs)

    assert paths.shape == (500, 10)
    assert np.array_equal(paths[:, -1], totals)


def test_t_de_student_tem_cauda_mais_pesada_que_a_normal(returns):
    garch = fit_garch(returns)
    kwargs = dict(n_sims=100_000, seed=11, vol_dynamics=False, garch=garch)

    normal, _ = montecarlo.montecarlo_var(returns, 0.999, dist="normal", **kwargs)
    student, _ = montecarlo.montecarlo_var(returns, 0.999, dist="t", df_t=4.0, **kwargs)

    assert student.value > normal.value


def test_erro_padrao_cai_com_mais_simulacoes(returns):
    table = montecarlo.montecarlo_convergence(returns, 0.95, (1_000, 100_000))

    assert table["std_error"].iloc[1] < table["std_error"].iloc[0]


def test_es_segue_a_raiz_do_tempo_como_o_var(returns):
    one_day = es.expected_shortfall(returns, 0.95, "historical").value
    ten_days = es.expected_shortfall(returns, 0.95, "historical", horizon_days=10).value

    assert ten_days == pytest.approx(one_day * np.sqrt(10), rel=1e-12)


def test_es_parametrico_no_horizonte_bate_com_a_formula(returns):
    alpha, h = 0.05, 10
    mu, sigma = returns.mean(), returns.std(ddof=1)
    tail = sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    expected = -mu * h + tail * np.sqrt(h)

    result = es.expected_shortfall(returns, 0.95, "parametric", horizon_days=h)

    assert result.value == pytest.approx(expected, rel=1e-12)
