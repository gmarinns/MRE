from finprov.risk.var import historical_var

def test_historical_var_com_valores_conhecidos():
    retornos = [-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04]
    resultado = historical_var(retornos, 0.95)
    assert resultado == 0.044