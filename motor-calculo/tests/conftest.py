"""Insumos compartilhados pelos testes do motor.

Regra dos testes: nunca usar a função sob teste como seu próprio oráculo.
Toda asserção compara com uma fórmula analítica, uma amostra de resposta
conhecida ou uma implementação ingênua equivalente.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def returns() -> np.ndarray:
    """2000 retornos normais: sem assimetria nem cauda pesada."""
    return np.random.default_rng(2024).normal(0.0004, 0.02, 2000)


@pytest.fixture
def matrix() -> pd.DataFrame:
    """Retornos de três ativos independentes."""
    data = np.random.default_rng(5).normal(0, 0.012, (600, 3))
    return pd.DataFrame(data, columns=["A", "B", "C"])
