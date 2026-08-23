import numpy as np

def historical_var(returns, confidence_level):
    alpha = 1 - confidence_level
    percentile = np.percentile(returns, alpha * 100)
    return -1 * percentile