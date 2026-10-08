"""Motor de cálculo de risco de mercado do FinProv.

Pacote puro (numpy, pandas, scipy, arch): não conhece Streamlit, banco de dados
nem rede. Recebe retornos, devolve o número e os metadados da execução.

    var.py          VaR histórico, paramétrico, EWMA e Cornish-Fisher
    montecarlo.py   VaR por simulação (GBM + GARCH)
    es.py           Expected Shortfall
    volatility.py   EWMA, GARCH(1,1) e volatilidade rolante
    portfolio.py    VaR de carteira por matriz de covariância
    drawdown.py     perda desde o topo e seus episódios
    stress.py       cenários de estresse e piores janelas
    backtesting.py  Kupiec, Christoffersen e semáforo de Basileia
"""

from .types import RiskResult, sha256_of_array

__version__ = "0.3.0"
__all__ = ["RiskResult", "sha256_of_array", "__version__"]
