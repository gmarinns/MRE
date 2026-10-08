import numpy as np
import pytest
from scipy import stats

from motor_calculo import var
from motor_calculo.types import sha256_of_array


def test_historico_em_amostra_de_resposta_conhecida():
    # 100 retornos de -0,50 a 0,49: o percentil 5% (interpolação linear) é -0,4505
    sample = np.arange(-50, 50) / 100.0

    result = var.historical_var(sample, confidence_level=0.95)

    assert result.value == pytest.approx(0.4505, abs=1e-9)
    assert result.method == "historical"
    assert len(result.input_hash) == 16


def test_parametrico_bate_com_a_formula_fechada(returns):
    mu, sigma = returns.mean(), returns.std(ddof=1)
    expected = -(mu + stats.norm.ppf(0.05) * sigma)

    result = var.parametric_var(returns, 0.95)

    assert result.value == pytest.approx(expected, rel=1e-12)


def test_horizonte_segue_a_raiz_do_tempo(returns):
    one_day = var.historical_var(returns, 0.95, horizon_days=1).value
    ten_days = var.historical_var(returns, 0.95, horizon_days=10).value

    assert ten_days == pytest.approx(one_day * np.sqrt(10), rel=1e-12)


def test_var_cresce_com_o_nivel_de_confianca(returns):
    values = [var.historical_var(returns, c).value for c in (0.90, 0.95, 0.99)]

    assert values == sorted(values)


def test_ewma_com_lambda_um_equivale_a_normal_de_media_zero(returns):
    # λ → 1 dá pesos iguais: σ vira a raiz da média dos retornos ao quadrado
    expected = -stats.norm.ppf(0.05) * np.sqrt(np.mean(returns**2))

    result = var.ewma_var(returns, 0.95, lam=1.0)

    assert result.value == pytest.approx(expected, rel=1e-9)


def test_cornish_fisher_sem_assimetria_nem_curtose_e_o_quantil_normal():
    z = stats.norm.ppf(0.05)

    assert var.cornish_fisher_quantile(z, skew=0.0, kurtosis=3.0) == pytest.approx(z)


def test_cornish_fisher_aumenta_o_var_com_cauda_esquerda_pesada():
    rng = np.random.default_rng(3)
    skewed = -rng.gamma(shape=2.0, scale=0.01, size=5000)  # assimetria negativa

    cf = var.cornish_fisher_var(skewed, 0.99).value
    normal = var.parametric_var(skewed, 0.99).value

    assert cf > normal


def test_serie_rolante_bate_com_o_laco_ingenuo(returns):
    window, n = 60, 400

    vectorized = var.rolling_var_series(returns[:n], window, 0.95, "historical")
    naive = [-np.percentile(returns[i - window : i], 5.0) for i in range(window, n)]

    assert np.isnan(vectorized[:window]).all()
    assert np.allclose(vectorized[window:], naive)


def test_serie_rolante_nao_usa_informacao_do_futuro():
    series = np.zeros(200)
    series[150:153] = -0.5  # três dias de choque: o bastante para mover o percentil

    rolling = var.rolling_var_series(series, 50, 0.95, "historical")

    assert rolling[150] == pytest.approx(0.0, abs=1e-12)  # ainda não conhece o choque
    assert rolling[153] > 0.0


@pytest.mark.parametrize("method", ["historical", "parametric", "ewma"])
def test_serie_rolante_aceita_os_tres_metodos(returns, method):
    rolling = var.rolling_var_series(returns[:300], 63, 0.95, method)

    assert np.isfinite(rolling[63:]).all()
    assert (rolling[63:] > 0).all()


def test_serie_rolante_rejeita_metodo_desconhecido(returns):
    with pytest.raises(ValueError, match="não suportado"):
        var.rolling_var_series(returns, 63, 0.95, "inexistente")


def test_run_id_muda_com_o_parametro_e_o_hash_do_insumo_nao(returns):
    a = var.historical_var(returns, 0.95)
    b = var.historical_var(returns, 0.99)

    assert a.run_id != b.run_id
    assert a.input_hash == b.input_hash


def test_hash_dos_insumos_e_estavel_e_sensivel():
    a = np.array([0.01, -0.02, 0.03])
    b = np.array([0.01, -0.02, 0.030000001])

    assert sha256_of_array(a) == sha256_of_array(a.copy())
    assert sha256_of_array(a) != sha256_of_array(b)
