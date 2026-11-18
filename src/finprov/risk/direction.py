"""Baseline experimental de direção para o próximo pregão; não é recomendação."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def supervised_frame(returns: pd.Series) -> pd.DataFrame:
    """Features usam somente dados até t; alvo é retorno de t+1 > 0."""
    features = pd.DataFrame(index=returns.index)
    for lag in range(5):
        features[f"return_lag_{lag}"] = returns.shift(lag)
    for window in (5, 21, 63):
        features[f"momentum_{window}"] = np.expm1(
            np.log1p(returns).rolling(window).sum()
        )
        features[f"volatility_{window}"] = returns.rolling(window).std()
    future = returns.shift(-1)
    features["target"] = (future > 0).astype(float).where(future.notna())
    features["target_date"] = pd.Series(returns.index, index=returns.index).shift(-1)
    return features


def chronological_split(frame: pd.DataFrame, test_size=252):
    columns = [c for c in frame if c not in {"target", "target_date"}]
    labeled = frame.dropna(subset=[*columns, "target", "target_date"])
    if len(labeled) < test_size + 505:
        return labeled.iloc[:0], labeled.iloc[:0], columns
    test = labeled.iloc[-test_size:]
    # Exclusão pelo momento de realização do alvo, não só pela data da feature.
    train = labeled.loc[labeled["target_date"] < test.index[0]]
    return train, test, columns


def model():
    # Sem ajuste de hiperparâmetros usando o teste; scaler só aprende no treino.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        ),
    )


def serialized_model(fitted, columns):
    scaler, classifier = fitted.steps[0][1], fitted.steps[1][1]
    return {
        "features": columns,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "classes": classifier.classes_.tolist(),
    }


def fit_direction(returns: pd.Series, test_size=252) -> dict:
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("Datas devem ser únicas e ordenadas.")
    observed = returns.dropna().to_numpy(dtype=float)
    if not np.isfinite(observed).all() or (observed <= -1).any():
        raise ValueError("Retornos inválidos para direção.")
    if test_size < 30:
        raise ValueError("Teste precisa de pelo menos 30 observações.")
    frame = supervised_frame(returns)
    train, test, columns = chronological_split(frame, test_size)
    if len(train) < 504:
        return {
            "status": "insufficient_history",
            "minimum_training_rows": 504,
            "holdout_rows": test_size,
            "feature_warmup": 63,
        }
    if train["target"].nunique() != 2:
        return {"status": "single_training_class"}
    fitted = model().fit(train[columns], train["target"])
    probability = fitted.predict_proba(test[columns])[:, 1]
    truth = test["target"].to_numpy()
    predicted = (probability >= 0.5).astype(int)
    baseline_probability = float(train["target"].mean())
    baseline = np.full(len(test), int(baseline_probability >= 0.5))
    accuracy = float(accuracy_score(truth, predicted))
    baseline_accuracy = float(accuracy_score(truth, baseline))
    validation = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": str(train.index[0]),
        "train_end": str(train.index[-1]),
        "last_training_target_date": str(train["target_date"].max()),
        "test_start": str(test.index[0]),
        "test_end": str(test.index[-1]),
        "last_test_target_date": str(test["target_date"].iloc[-1]),
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "roc_auc": float(roc_auc_score(truth, probability))
        if len(set(truth)) == 2
        else None,
        "brier_score": float(brier_score_loss(truth, probability)),
        "baseline_accuracy": baseline_accuracy,
        "baseline_brier_score": float(
            brier_score_loss(truth, np.full(len(test), baseline_probability))
        ),
        "baseline": "constant_training_up_frequency_and_majority_class",
        "baseline_up_probability": baseline_probability,
        "accuracy_difference": accuracy - baseline_accuracy,
        "statistical_significance_tested": False,
        "predictions": [
            {
                "as_of": str(day),
                "target_date": str(target_date),
                "probability_up": float(p),
                "actual_up": int(y),
            }
            for day, target_date, p, y in zip(
                test.index, test["target_date"], probability, truth, strict=True
            )
        ],
        "model": serialized_model(fitted, columns),
    }
    # Após avaliar uma única vez o holdout, reajusta para inferência no último dia.
    latest = frame.iloc[[-1]]
    if latest[columns].isna().any(axis=None):
        return {"status": "latest_features_missing", "validation": validation}
    known = frame.dropna(subset=[*columns, "target", "target_date"])
    known = known.loc[known["target_date"] <= latest.index[0]]
    final = model().fit(known[columns], known["target"])
    p_up = float(final.predict_proba(latest[columns])[0, 1])
    return {
        "status": "experimental",
        "horizon_trading_days": 1,
        "as_of": str(latest.index[0]),
        "target": "next_adjusted_return_strictly_positive",
        "probability_up": p_up,
        "probability_not_up": 1 - p_up,
        "predicted_class": "alta" if p_up >= 0.5 else "queda_ou_estabilidade",
        "probability_calibrated": False,
        "evaluation": "observed_accuracy_above_baseline"
        if accuracy > baseline_accuracy
        else "did_not_beat_baseline_accuracy",
        "validation": validation,
        "final_model": serialized_model(final, columns),
        "final_training_rows": len(known),
    }


def build_direction_report(prices: list, portfolio: dict) -> dict:
    result = {
        "horizon_trading_days": 1,
        "status": "experimental",
        "model": "standard_scaler_logistic_regression_C1",
        "fundamentals_used": False,
        "limitation": "Retrospective revised prices; uncalibrated probabilities.",
        "assets": {},
        "portfolio": None,
    }
    if not prices:
        return {**result, "status": "no_prices"}
    frame = pd.DataFrame(prices)
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Preços duplicados.")
    frame.loc[frame["close"].isna(), "adjusted_close"] = np.nan
    tickers = portfolio["tickers"]
    pivot = frame.pivot(index="date", columns="ticker", values="adjusted_close")
    pivot = pivot.sort_index().reindex(columns=tickers)
    # Remove datas sem preço em nenhuma posição (p.ex. feriados do benchmark).
    pivot = pivot.dropna(how="all")
    returns = pivot.pct_change(fill_method=None)
    if returns.empty:
        return {**result, "status": "no_prices"}
    for ticker in tickers:
        result["assets"][ticker] = fit_direction(returns[ticker])
    if returns.iloc[-1].isna().any():
        result["status"] = "incomplete_portfolio"
        return result
    # min_count impede excluir ativo ou inventar retorno zero em dia incompleto.
    # JSONB não preserva a ordem das chaves; fixe a ordem das somas da carteira.
    weights = pd.Series(portfolio["weights"]).reindex(tickers)
    weighted = returns.mul(weights)
    port = weighted.sum(axis=1, min_count=len(tickers))
    result["portfolio"] = fit_direction(port)
    return result


def direction_markdown(report: dict) -> str:
    rows = dict(report["assets"])
    if report.get("portfolio"):
        rows["CARTEIRA"] = report["portfolio"]
    lines = [
        "# Direção experimental: próximo pregão",
        "",
        "Probabilidades do modelo, sem calibração validada. Não são certezas "
        "nem recomendações de compra/venda. Estabilidade pertence à classe não-alta.",
        "",
        "Treino e teste em ordem temporal; último bloco de 252 observações "
        "reservado para teste, sem escolha de hiperparâmetros nesse bloco.",
        "",
        "| Ativo | Data-base | P(alta) | Classe estimada | Acerto teste | Baseline |",
        "|---|---|---:|---|---:|---:|",
    ]
    for ticker, row in rows.items():
        if row["status"] != "experimental":
            lines.append(f"| {ticker} | | | {row['status']} | | |")
            continue
        validation = row["validation"]
        lines.append(
            f"| {ticker} | {row['as_of']} | {row['probability_up']:.1%} | "
            f"{row['predicted_class']} | {validation['accuracy']:.1%} | "
            f"{validation['baseline_accuracy']:.1%} |"
        )
    lines += [
        "",
        "Baseline: prever sempre a classe mais frequente no treino. "
        "Maior acerto observado não comprova superioridade estatística. "
        "O modelo final é reajustado com os alvos já conhecidos após a avaliação.",
        "",
        "Features: retornos defasados, momentum e volatilidade. "
        "Fundamentos atuais não foram usados como dados históricos. "
        "Preços ajustados podem conter revisões posteriores; esta avaliação "
        "não substitui acompanhamento prospectivo e validação em outros períodos.",
    ]
    return "\n".join(lines) + "\n"
