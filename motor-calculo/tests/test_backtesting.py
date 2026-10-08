import numpy as np
import pytest

from motor_calculo import backtesting as bt


def test_violacao_usa_o_var_do_proprio_dia():
    returns = np.array([-0.01, -0.05, -0.01])
    var_series = np.array([0.02, 0.03, 0.10])

    mask = bt.violations_mask(returns, var_series)

    assert mask.tolist() == [False, True, False]
    # comparar com a média da série (0,05) esconderia a violação do dia 2
    assert not bt.violations_mask(returns, np.full(3, var_series.mean())).any()


def test_kupiec_nao_rejeita_a_taxa_esperada():
    violations = np.zeros(250, dtype=int)
    violations[[10, 120]] = 1  # 2 em 250 a 99%: esperado 2,5

    result = bt.kupiec_pof(violations, 0.99)

    assert result["violations"] == 2
    assert not result["reject_h0"]


def test_kupiec_com_zero_violacoes_acusa_modelo_conservador():
    # LR = -2·n·ln(1-p): zero violações não é "modelo perfeito"
    result = bt.kupiec_pof(np.zeros(250, dtype=int), 0.99)

    assert result["lr_stat"] == pytest.approx(-2 * 250 * np.log(0.99), rel=1e-12)
    assert result["reject_h0"]


def test_kupiec_rejeita_excesso_de_violacoes():
    violations = np.zeros(250, dtype=int)
    violations[:20] = 1  # 8% contra 1% esperado

    assert bt.kupiec_pof(violations, 0.99)["reject_h0"]


def test_christoffersen_detecta_agrupamento():
    clustered = np.zeros(300, dtype=int)
    clustered[100:112] = 1
    spread = np.zeros(300, dtype=int)
    spread[::25] = 1

    assert bt.christoffersen_independence(clustered)["reject_h0"]
    assert not bt.christoffersen_independence(spread)["reject_h0"]


def test_matriz_de_transicao_conta_os_pares_consecutivos():
    result = bt.christoffersen_independence(np.array([0, 0, 1, 1, 0, 1]))

    assert result["transitions"] == {"00": 1, "01": 2, "10": 1, "11": 1}


def test_cobertura_condicional_soma_os_dois_testes():
    violations = np.zeros(300, dtype=int)
    violations[[5, 6, 90, 200]] = 1

    joint = bt.conditional_coverage(violations, 0.95)

    assert joint["df"] == 2
    assert joint["lr_stat"] == pytest.approx(
        bt.kupiec_pof(violations, 0.95)["lr_stat"]
        + bt.christoffersen_independence(violations)["lr_stat"]
    )


@pytest.mark.parametrize(
    ("n_violations", "zone"), [(3, "Verde"), (7, "Amarela"), (15, "Vermelha")]
)
def test_semaforo_de_basileia_em_250_dias_a_99(n_violations, zone):
    violations = np.zeros(250, dtype=int)
    violations[:n_violations] = 1

    assert bt.basel_traffic_light(violations, 0.99)["zone"] == zone


def test_backtest_completo_e_coerente_com_a_mascara():
    rng = np.random.default_rng(8)
    returns = rng.normal(0, 0.01, 500)
    var_series = np.full(500, 0.0165)  # ~5% de violações para uma normal

    report = bt.full_backtest(returns, var_series, 0.95)

    assert report["kupiec"]["violations"] == int(report["mask"].sum())
    assert report["loss_when_violated"] > 0.0165


def test_limiares_reproduzem_a_tabela_de_basileia():
    # Basel Committee (1996): 250 dias a 99% -> amarela a partir de 5, vermelha de 10
    limits = bt.basel_thresholds(250, 0.99)

    assert limits["yellow_from"] == 5
    assert limits["red_from"] == 10
    assert limits["expected"] == pytest.approx(2.5)
