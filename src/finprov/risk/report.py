"""Risco histórico de 1 dia: retornos simples, sem previsão de direção."""

import math

import numpy as np
import pandas as pd


def metrics(returns: pd.Series, confidence: float, window: int) -> dict:
    if not 0.5 < confidence < 1 or window < 20:
        raise ValueError("Confiança deve estar entre 0,5 e 1; janela >= 20.")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Retornos inválidos.")
    if len(values) < window:
        return {
            "status": "insufficient_history",
            "observations": len(values),
            "required": window,
        }
    sample = values[-window:]
    q = float(np.quantile(sample, 1 - confidence, method="linear"))
    tail = sample[sample <= q]
    wealth = np.r_[1.0, np.cumprod(1 + sample)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    result = {
        "status": "ok",
        "observations": len(values),
        "window": window,
        "sample_start": str(returns.index[-window]),
        "sample_end": str(returns.index[-1]),
        "confidence": confidence,
        "horizon_trading_days": 1,
        "return_kind": "simple_adjusted",
        "var": -q,
        "es": float(-tail.mean()),
        "tail_observations": len(tail),
        "annualized_volatility": float(np.std(sample, ddof=1) * np.sqrt(252)),
        "historical_down_frequency": float(np.mean(sample < 0)),
        "historical_up_frequency": float(np.mean(sample > 0)),
        "max_drawdown": float(-drawdown.min()),
        "return_21_sessions": float(np.prod(1 + sample[-21:]) - 1)
        if window >= 21
        else None,
        "warnings": ["small_tail_sample"] if window * (1 - confidence) < 20 else [],
    }
    # Cada previsão de risco usa somente os retornos anteriores ao dia avaliado.
    if len(values) > window:
        actual = values[window:]
        predicted = np.array(
            [
                -np.quantile(values[i - window : i], 1 - confidence, method="linear")
                for i in range(window, len(values))
            ]
        )
        result["backtest"] = {
            "observations": len(actual),
            "breaches": int(np.sum(-actual > predicted)),
            "breach_rate": float(np.mean(-actual > predicted)),
            "expected_breach_rate": 1 - confidence,
            "basis": "retrospective_using_collected_prices",
        }
    else:
        result["backtest"] = None
    return result


def build_report(prices: list, portfolio: dict, confidence=0.95, window=252) -> dict:
    if not 0.5 < confidence < 1 or window < 20:
        raise ValueError("Parâmetros de risco inválidos.")
    tickers = portfolio["tickers"]
    report = {
        "portfolio_definition": portfolio,
        "assets": {},
        "direction_forecast": None,
        "interpretation": "Historical risk, not a forecast of price direction.",
        "portfolio_assumption": "constant_value_weights_daily_rebalancing",
        "costs": "not_included",
        "confidence": confidence,
        "window": window,
    }
    if not prices:
        return {**report, "status": "no_prices"}
    frame = pd.DataFrame(prices)
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Preços duplicados na construção do relatório.")
    frame.loc[frame["close"].isna(), "adjusted_close"] = np.nan
    pivot = frame.pivot(index="date", columns="ticker", values="adjusted_close")
    pivot = pivot.sort_index().reindex(columns=tickers)
    returns = pivot.pct_change(fill_method=None)
    for ticker in tickers:
        sample = returns[ticker].dropna()
        report["assets"][ticker] = metrics(sample, confidence, window)
        report["assets"][ticker]["missing_price_rows"] = int(pivot[ticker].isna().sum())
    missing = [t for t in tickers if pivot[t].dropna().empty]
    last_dates = {
        t: str(pivot[t].last_valid_index()) for t in tickers if t not in missing
    }
    report["last_price_dates"] = last_dates
    if missing:
        return {**report, "status": "incomplete_portfolio", "missing_assets": missing}
    if len(set(last_dates.values())) != 1:
        return {**report, "status": "unaligned_latest_prices"}
    common = returns.dropna(how="any")
    weights = np.array([portfolio["weights"][t] for t in tickers])
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not math.isclose(float(weights.sum()), 1, abs_tol=1e-9)
    ):
        raise ValueError("Pesos inválidos.")
    port = pd.Series(common.to_numpy() @ weights, index=common.index)
    report["portfolio"] = metrics(port, confidence, window)
    report["common_return_observations"] = len(common)
    report["excluded_return_dates"] = int(len(returns) - len(common))
    report["status"] = report["portfolio"]["status"]
    if len(common) >= window:
        # Comparação usa exatamente as mesmas datas para cada ativo e a carteira.
        individuals = {t: metrics(common[t], confidence, window) for t in tickers}
        report["assets_on_common_dates"] = individuals
        weighted = sum(portfolio["weights"][t] * individuals[t]["var"] for t in tickers)
        report["weighted_individual_var_minus_portfolio_var"] = (
            weighted - report["portfolio"]["var"]
        )
        corr = common.tail(window).corr()
        report["correlation"] = {
            t: {
                u: float(corr.loc[t, u]) if pd.notna(corr.loc[t, u]) else None
                for u in tickers
            }
            for t in tickers
        }
    return report


def markdown_report(report: dict) -> str:
    lines = [
        "# Risco histórico da carteira",
        "",
        "Frequências passadas não são probabilidades previstas de alta ou queda.",
        "VaR e ES são frações do capital; horizonte de 1 pregão. "
        "Toda ação tem risco. Não há classificação binária 'sem risco'.",
        "",
        f"Status: `{report['status']}`. Janela: {report['window']} retornos.",
        f"Nível de confiança: {report['confidence']:.0%}.",
        "",
        "| Ativo | VaR | ES | Volatilidade anual | Frequência de quedas |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = dict(report.get("assets_on_common_dates", report["assets"]))
    if report.get("portfolio"):
        rows["CARTEIRA"] = report["portfolio"]
    for ticker, row in rows.items():
        if row["status"] != "ok":
            lines.append(f"| {ticker} | Histórico insuficiente | | | |")
            continue
        lines.append(
            f"| {ticker} | {row['var']:.2%} | {row['es']:.2%} | "
            f"{row['annualized_volatility']:.2%} | "
            f"{row['historical_down_frequency']:.2%} |"
        )
    lines += [
        "",
        "Amostras de cauda pequenas tornam VaR e ES instáveis; consulte os avisos "
        "e compare diferentes janelas. Datas usadas estão em report.json.",
        "",
        "Pesos iguais em valor, rebalanceados diariamente na simulação; "
        "custos não incluídos. Proventos já refletidos no preço ajustado "
        "não são somados novamente.",
        "",
        "Fundamentos atuais e revisados ficam fora dos cálculos históricos. "
        "O backtest usa os preços disponíveis nesta coleta, "
        "não comprova disponibilidade ponto a ponto no passado.",
        "",
        "Consulte report.json para datas, amostras, avisos e violações do VaR.",
    ]
    return "\n".join(lines) + "\n"
