import pandas as pd
import pytest

from finprov.risk.report import metrics


def test_historical_var_com_valores_conhecidos():
    retornos = pd.Series([-0.05, -0.03] + [0.01] * 18)
    resultado = metrics(retornos, 0.95, 20)
    # Quantil linear: posição 19 * 0,05 = 0,95 entre -5% e -3%.
    assert resultado["var"] == pytest.approx(0.031)
    assert resultado["es"] == pytest.approx(0.05)
