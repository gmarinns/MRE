import copy
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from finprov.config import TICKERS, load_portfolio
from finprov.etl.datasets import indicator_history, investidor10, pt_number, yahoo
from finprov.ingest.bacen import requests_for
from finprov.ingest.investidor10 import history_url
from finprov.pipeline import normalize
from finprov.risk.report import build_report, metrics
from finprov.storage.runs import run_path


def yahoo_payload(ticker="AGRO3", days=300):
    base = datetime(2020, 1, 1, 16, tzinfo=UTC)
    prices = [100 * (1 + 0.025 * np.sin(i / 7)) for i in range(days)]
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": ticker + ".SA",
                        "currency": "BRL",
                        "exchangeTimezoneName": "America/Sao_Paulo",
                    },
                    "timestamp": [
                        int((base + timedelta(days=i)).timestamp()) for i in range(days)
                    ],
                    "indicators": {
                        "quote": [
                            {
                                "open": prices,
                                "high": prices,
                                "low": prices,
                                "close": prices,
                                "volume": [1000] * days,
                            }
                        ],
                        "adjclose": [{"adjclose": prices}],
                    },
                    "events": {
                        "dividends": {
                            "a": {"date": int(base.timestamp()), "amount": 1.23}
                        },
                        "splits": {
                            "b": {
                                "date": int(base.timestamp()),
                                "numerator": 1,
                                "denominator": 10,
                            }
                        },
                    },
                }
            ],
        }
    }


PAGE = """<h1>AGRO3</h1><div id="table-indicators"><section><h3>Rentabilidade</h3>
<article class="indicator-card"><div class="indicator-card-title">ROIC</div>
<div class="indicator-card-value"><span>-0,81%</span></div></article>
<article class="indicator-card"><div class="indicator-card-title">P/L</div>
<div class="indicator-card-value">-1.299,18</div></article>
</section></div><table id="table-dividends-history"><tbody><tr>
<td>JSCP</td><td>07/11/2016</td><td>21/11/2016</td><td>0,40108100</td>
</tr></tbody></table><script>
$.get(`/api/historico-indicadores/320/${period}/?v=2`)
</script>"""


def test_exact_user_universe_equal_weights():
    portfolio = load_portfolio()
    assert portfolio["tickers"] == list(TICKERS)
    assert len(TICKERS) == 12
    assert set(portfolio["weights"].values()) == {1 / 12}


def test_missing_adjustment_never_uses_close():
    data = yahoo_payload(days=2)
    del data["chart"]["result"][0]["indicators"]["adjclose"]
    prices, _ = yahoo(data, "AGRO3")
    assert prices[0]["close"] == 100
    assert prices[0]["adjusted_close"] is None
    assert "missing_adjustment" in prices[0]["quality_flags"]


def test_reverse_split_and_cash_type_are_preserved():
    _, events = yahoo(yahoo_payload(days=2), "AGRO3")
    assert events[0]["event_type"] == "cash_distribution_unclassified"
    assert events[0]["payment_date"] is None
    assert events[1]["event_type"] == "reverse_split"
    assert events[1]["split_factor"] == 0.1


def test_duplicate_dates_and_wrong_symbol_rejected():
    data = yahoo_payload(days=2)
    data["chart"]["result"][0]["timestamp"] *= 2
    with pytest.raises(ValueError):
        yahoo(data, "AGRO3")
    with pytest.raises(ValueError, match="Símbolo"):
        yahoo(yahoo_payload(), "PETR4")


def test_bad_price_preserved_as_quality_issue():
    data = yahoo_payload(days=2)
    data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][0] = -1
    rows, _ = yahoo(data, "AGRO3")
    assert rows[0]["adjusted_close"] is None
    assert "invalid_adjusted_close" in rows[0]["quality_flags"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-1.299,18", -1299.18),
        ("0,40108100", 0.401081),
        ("194,84%", 194.84),
        ("-", None),
        ("0,00", 0),
    ],
)
def test_portuguese_numbers(text, expected):
    assert pt_number(text) == expected


def test_i10_dates_are_not_fabricated_and_negative_ratios_kept():
    indicators, events = investidor10(PAGE, "AGRO3")
    assert indicators[0]["value"] == -0.81
    assert indicators[0]["unit"] == "percent"
    assert indicators[1]["value"] == -1299.18
    assert events[0]["event_type"] == "jcp"
    assert events[0]["com_date"] == "2016-11-07"
    assert events[0]["payment_date"] == "2016-11-21"
    assert events[0]["ex_date"] is None
    assert history_url(PAGE).endswith("/320/5/?v=2")


def test_i10_layout_changes_fail_explicitly():
    with pytest.raises(ValueError):
        investidor10("<h1>AGRO3</h1><p>32,83</p>", "AGRO3")
    with pytest.raises(ValueError):
        investidor10(PAGE, "VALE3")


def test_stock_bonus_is_not_cash_or_guessed_split():
    _, events = investidor10(PAGE.replace("JSCP", "Bonificação"), "AGRO3")
    assert events[0]["event_type"] == "stock_bonus"
    assert events[0]["amount"] is None
    assert events[0]["split_factor"] is None
    assert events[0]["provider_value"] == 0.401081


def test_annual_revisions_not_treated_as_past_publications():
    data = {
        "P/L": [
            {"year": "Atual", "value": 30},
            {"year": "2025", "value": -100, "updated_at": "2023-01-01"},
        ]
    }
    result = indicator_history(data, "AGRO3")
    assert len(result) == 1
    assert result[0]["reference_year"] == 2025
    assert result[0]["published_at"] is None
    assert result[0]["point_in_time_verified"] is False


def test_bacen_chunks_have_no_overlap_and_exclusive_end():
    requests = list(requests_for(date(2015, 1, 1), date(2026, 11, 17)))
    chunks = [params for code, _, params in requests if code == 11]
    assert len(chunks) == 3
    assert chunks[0]["dataFinal"] == "31/12/2019"
    assert chunks[1]["dataInicial"] == "01/01/2020"
    assert chunks[-1]["dataFinal"] == "16/11/2026"


def test_run_paths_reject_traversal(tmp_path):
    with pytest.raises(ValueError):
        run_path(tmp_path, "../manifest.json")


def test_normalize_records_availability_and_separates_sources():
    inputs = [
        {
            "status": "ok",
            "source": "investidor10",
            "ticker": "AGRO3",
            "_raw": PAGE.encode(),
            "url": "https://investidor10.com.br/acoes/agro3/",
            "observed_at": "2026-11-17T18:00:00+00:00",
        }
    ]
    tables, errors = normalize(inputs, "2015-01-01", "2026-11-17")
    assert not errors
    assert tables["indicators"][0]["available_at"].startswith("2026-11-17")
    assert tables["events"][0]["source_url"] == inputs[0]["url"]
    assert "raw_sha256" not in tables["events"][0]


def test_simple_return_portfolio_is_exact_not_weighted_logs():
    dates = pd.date_range("2020-01-01", periods=31).strftime("%Y-%m-%d")
    rows = []
    # A dobra, B cai pela metade. 50/50 => +25%, não 0% por média de logs.
    for ticker, multiplier in [("AGRO3", 2), ("VALE3", 0.5)]:
        for i, day in enumerate(dates):
            price = 100 * multiplier**i
            rows.append(
                {"ticker": ticker, "date": day, "close": price, "adjusted_close": price}
            )
    portfolio = {"tickers": ["AGRO3", "VALE3"], "weights": {"AGRO3": 0.5, "VALE3": 0.5}}
    report = build_report(rows, portfolio, window=20)
    assert report["portfolio"]["var"] == pytest.approx(-0.25)
    assert report["portfolio"]["historical_down_frequency"] == 0


def test_missing_asset_never_silently_reweights():
    prices, _ = yahoo(yahoo_payload(), "AGRO3")
    report = build_report(prices, load_portfolio())
    assert report["status"] == "incomplete_portfolio"
    assert "portfolio" not in report


def test_missing_price_does_not_bridge_return_interval():
    prices, _ = yahoo(yahoo_payload(days=30), "AGRO3")
    prices[10]["adjusted_close"] = None
    report = build_report(
        prices, {"tickers": ["AGRO3"], "weights": {"AGRO3": 1}}, window=20
    )
    assert report["common_return_observations"] == 27


def test_rolling_backtest_excludes_current_day():
    series = pd.Series(
        [0.01] * 20 + [-0.8], index=pd.date_range("2020-01-01", periods=21)
    )
    result = metrics(series, 0.95, 20)
    assert result["backtest"]["observations"] == 1
    assert result["backtest"]["breaches"] == 1


def test_drawdown_includes_initial_capital():
    result = metrics(pd.Series([-0.1] + [0.0] * 19), 0.95, 20)
    assert result["max_drawdown"] == pytest.approx(0.1)


def test_unaligned_latest_date_blocks_portfolio():
    prices, _ = yahoo(yahoo_payload(), "AGRO3")
    second = copy.deepcopy(prices[:-1])
    for row in second:
        row["ticker"] = "VALE3"
    report = build_report(
        prices + second,
        {"tickers": ["AGRO3", "VALE3"], "weights": {"AGRO3": 0.5, "VALE3": 0.5}},
    )
    assert report["status"] == "unaligned_latest_prices"
