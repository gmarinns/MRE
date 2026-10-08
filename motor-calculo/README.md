# motor-calculo

Motor de risco de mercado do FinProv. Pacote Python puro (numpy, pandas, scipy,
arch): não conhece Streamlit, banco de dados nem rede. Recebe retornos e devolve
o número de risco junto com os metadados da execução.

## Módulos

| Módulo | Conteúdo |
|---|---|
| `types.py` | `RiskResult` (valor, método, parâmetros, hash dos insumos, `run_id`) |
| `var.py` | VaR histórico, paramétrico, EWMA, Cornish-Fisher e série rolante |
| `montecarlo.py` | VaR por simulação com volatilidade GARCH e choques normais ou t |
| `es.py` | Expected Shortfall por três fontes |
| `volatility.py` | EWMA, GARCH(1,1), previsão e volatilidade rolante |
| `portfolio.py` | VaR de carteira por covariância e VaR componente |
| `drawdown.py` | Série de drawdown e seus episódios |
| `stress.py` | Cenários de choque e piores janelas observadas |
| `backtesting.py` | Kupiec, Christoffersen, cobertura condicional e Basileia |

## Uso

```python
import numpy as np
from motor_calculo import var

retornos = np.diff(np.log(precos))
resultado = var.historical_var(retornos, confidence_level=0.95)

resultado.value        # 0.026  -> perda de 2,6%
resultado.input_hash   # identifica a série usada
resultado.run_id       # determinístico: mesmo insumo e parâmetros, mesmo id
```

## Instalação e testes

```bash
pip install -e "motor-calculo[dev]"
```

```bash
pytest motor-calculo/tests
```

Os testes nunca usam a função testada como seu próprio oráculo: cada asserção
compara com uma fórmula analítica, uma amostra de resposta conhecida ou uma
implementação ingênua equivalente.
