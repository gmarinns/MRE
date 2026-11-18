"""Interface original e motor real, alimentados apenas pelo PostgreSQL."""

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dados  # noqa: E402
from motor_calculo import var  # noqa: E402

RUN_ID = "00000000-0000-0000-0000-000000000001"
TICKERS = (
    "AGRO3 VALE3 BBSE3 FLRY3 SAPR4 GOAU3 KLBN3 PETR4 ABEV3 CPFE3 BBAS3 VLID3 ^BVSP"
).split()


class FakeDatabase:
    run_id = RUN_ID
    rows = []

    def __init__(self, url):
        pass

    def resolve_run(self):
        return self.run_id

    @contextmanager
    def connection(self):
        yield self

    @contextmanager
    def cursor(self):
        yield self

    def execute(self, query, params):
        assert "curated.precos" in query
        assert params == (RUN_ID,)

    def fetchall(self):
        return self.rows


@pytest.fixture(autouse=True)
def database_only(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("O dashboard tentou acessar HTTP.")

    monkeypatch.setattr(requests.Session, "request", fail)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setattr(dados, "Database", FakeDatabase)
    monkeypatch.setattr(FakeDatabase, "run_id", RUN_ID)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end="2026-11-17", periods=800)
    rows = []
    for ticker in TICKERS:
        closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(dates))))
        rows.extend(
            (
                ticker,
                date,
                close,
                close * 1.02,
                close * 0.98,
                close,
                close * 0.5,
                1000,
                pd.Timestamp("2026-11-18", tz="UTC"),
            )
            for date, close in zip(dates, closes, strict=True)
        )
    monkeypatch.setattr(FakeDatabase, "rows", rows)
    st.cache_data.clear()
    st.session_state.clear()
    yield
    st.cache_data.clear()
    st.session_state.clear()


def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


@pytest.mark.parametrize(
    "page",
    [
        "painel",
        "var",
        "es",
        "volatilidade",
        "drawdown",
        "carteira",
        "stress",
        "backtesting",
        "proveniencia",
    ],
)
def test_original_pages_with_real_motor(page):
    view = app()
    assert not view.exception
    if page != "painel":
        view.switch_page(f"paginas/{page}.py").run()
    assert not view.exception
    assert not view.error
    ctx = view.session_state["_ctx"]
    assert ctx.cotacoes.origem == "postgresql"
    assert ctx.cotacoes.run_id == RUN_ID
    assert len(ctx.ativos) == 12
    expected = var.historical_var(ctx.retornos, ctx.confianca, ctx.horizonte)
    assert ctx.var["historical"].value == expected.value
    if page != "proveniencia":
        assert view.get("plotly_chart")


def test_parameters_recalculate_motor():
    view = app()
    view.select_slider(key="confianca").set_value(0.99).run()
    view.selectbox(key="foco").select("VALE3.SA").run()
    assert not view.exception
    ctx = view.session_state["_ctx"]
    assert ctx.ticker == "VALE3.SA"
    assert (
        ctx.var["historical"].value == var.historical_var(ctx.retornos, 0.99, 1).value
    )


def test_missing_database_configuration(monkeypatch):
    monkeypatch.delenv("DATABASE_URL")
    view = app()
    assert not view.exception
    assert "DATABASE_URL" in view.error[0].value
    assert not view.metric


def test_database_unavailable_has_no_fallback(monkeypatch):
    def unavailable(self):
        raise psycopg2.OperationalError("offline")

    monkeypatch.setattr(FakeDatabase, "resolve_run", unavailable)
    view = app()
    assert not view.exception
    assert "Nenhuma coleta alternativa" in view.error[0].value
    assert not view.metric


def test_empty_database(monkeypatch):
    monkeypatch.setattr(FakeDatabase, "rows", [])
    view = app()
    assert not view.exception
    assert "não possui preços" in view.error[0].value


def test_adjustment_period_and_single_run(monkeypatch):
    dados.selecionar_execucao()
    monkeypatch.setattr(FakeDatabase, "run_id", "newer-run")
    cot = dados.carregar("VALE3.SA", "1y")
    raw = [r for r in FakeDatabase.rows if r[0] == "VALE3"][-1]
    assert cot.frame.iloc[-1]["close"] == pytest.approx(raw[6])
    assert cot.frame.iloc[-1]["open"] == pytest.approx(raw[2] * 0.5)
    assert cot.frame.iloc[-1]["volume"] == 1000
    assert cot.frame.date.min() > pd.Timestamp("2025-11-17")
    assert cot.run_id == RUN_ID
    matrix, missing = dados.matriz_de_retornos(("VALE3.SA", "PETR4.SA"), "1y")
    assert not missing
    assert matrix.shape[1] == 2


def test_missing_asset_does_not_fetch_market():
    dados.selecionar_execucao()
    with pytest.raises(dados.DadosIndisponiveis, match="histórico suficiente"):
        dados.carregar("UNKNOWN.SA")
