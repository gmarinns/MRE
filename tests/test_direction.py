import numpy as np
import pandas as pd
import pytest

from finprov.risk.direction import chronological_split, fit_direction, supervised_frame


def test_portfolio_is_invariant_to_jsonb_weight_key_order(monkeypatch):
    from finprov.risk.direction import build_direction_report

    tickers = ["VALE3", "AGRO3", "BBAS3"]
    rng = np.random.default_rng(7)
    rows = []
    for ticker in tickers:
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.03, 100))
        rows.extend(
            {
                "ticker": ticker,
                "date": day.strftime("%Y-%m-%d"),
                "close": float(price),
                "adjusted_close": float(price),
            }
            for day, price in zip(
                pd.bdate_range("2020-01-01", periods=100), prices, strict=True
            )
        )
    monkeypatch.setattr("finprov.risk.direction.fit_direction", lambda r: r)
    weights = {"VALE3": 0.2, "AGRO3": 0.3, "BBAS3": 0.5}
    a = build_direction_report(rows, {"tickers": tickers, "weights": weights})
    b = build_direction_report(
        rows, {"tickers": tickers, "weights": dict(sorted(weights.items()))}
    )
    pd.testing.assert_series_equal(a["portfolio"], b["portfolio"], check_exact=True)


def sample(n=1100):
    rng = np.random.default_rng(42)
    return pd.Series(
        rng.normal(0.0003, 0.02, n),
        index=pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
    )


def test_future_changes_do_not_change_past_features():
    returns = sample()
    before = supervised_frame(returns)
    returns.iloc[900:] = -0.2
    after = supervised_frame(returns)
    columns = [c for c in before if c not in {"target", "target_date"}]
    pd.testing.assert_frame_equal(before[columns].iloc[:900], after[columns].iloc[:900])


def test_target_really_is_next_session_not_current():
    values = pd.Series([0.1, -0.05, 0.0, 0.02], index=["a", "b", "c", "d"])
    result = supervised_frame(values)
    assert result["target"].iloc[:3].tolist() == [0, 0, 1]
    assert pd.isna(result["target"].iloc[-1])
    assert result["target_date"].iloc[0] == "b"


def test_last_training_target_is_before_first_test_feature():
    train, test, _ = chronological_split(supervised_frame(sample()))
    assert train["target_date"].max() < test.index.min()
    assert len(test) == 252
    assert set(train.index).isdisjoint(test.index)


def test_scaler_uses_training_only_and_final_model_is_serialized():
    returns = sample()
    result = fit_direction(returns)
    assert result["status"] == "experimental"
    assert 0 <= result["probability_up"] <= 1
    assert result["as_of"] == returns.index[-1]
    assert result["probability_calibrated"] is False
    train, test, columns = chronological_split(supervised_frame(returns))
    means = result["validation"]["model"]["scaler_mean"]
    assert np.allclose(means, train[columns].mean().to_numpy())
    assert result["validation"]["last_training_target_date"] < test.index[0]
    assert len(result["validation"]["predictions"]) == 252
    assert len(result["final_model"]["coefficients"]) == len(columns)


def test_small_dataset_or_single_class_does_not_fake_prediction():
    assert fit_direction(sample(300))["status"] == "insufficient_history"
    assert fit_direction(sample().abs())["status"] == "single_training_class"


def test_missing_last_return_blocks_current_inference():
    returns = sample()
    returns.iloc[-1] = np.nan
    result = fit_direction(returns)
    assert result["status"] == "latest_features_missing"
    assert "probability_up" not in result


def test_invalid_returns_fail():
    returns = sample()
    returns.iloc[-1] = -1
    with pytest.raises(ValueError):
        fit_direction(returns)
