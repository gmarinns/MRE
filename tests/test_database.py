"""Integração real; exige banco descartável finprov_test_* explicitamente indicado."""

import json
import os
from datetime import date

import psycopg2
import pytest

from finprov.config import TICKERS, load_portfolio
from finprov.pipeline import collect, export_files
from finprov.storage.database import Database
from finprov.storage.runs import canonical_json
from finprov.validation import validate_run
from tests.test_data_pipeline import PAGE, yahoo_payload


@pytest.fixture
def database():
    url = os.environ.get("FINPROV_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Defina FINPROV_TEST_DATABASE_URL para testes PostgreSQL.")
    db = Database(url)
    if not db.database.startswith("finprov_test_"):
        pytest.fail("Os testes só podem apagar bancos com prefixo finprov_test_.")
    db.initialize(reset=True, confirm_database=db.database)
    return db, url


def fake_fetch(client, url, params):
    if "finance/chart/" in url:
        ticker = url.rsplit("/", 1)[-1].removesuffix(".SA")
        data = yahoo_payload(ticker, days=1000)
        if ticker == "^BVSP":
            data["chart"]["result"][0]["meta"]["symbol"] = "^BVSP"
        return canonical_json(data)
    if "bcb.gov.br" in url:
        return canonical_json([{"data": "02/01/2020", "valor": "1.25"}])
    if "/acoes/" in url:
        ticker = url.rstrip("/").rsplit("/", 1)[-1].upper()
        index = TICKERS.index(ticker)
        page = PAGE.replace("AGRO3", ticker).replace("/320/", f"/{1000 + index}/")
        return page.replace("-1.299,18", f"{index + 1},25").encode()
    index = int(url.split("historico-indicadores/")[1].split("/")[0]) - 1000
    return canonical_json({"P/L": [{"year": "2020", "value": index + 10}]})


def run_collection(url, sources=None):
    return collect(
        url,
        load_portfolio(),
        date(2020, 1, 1),
        date(2024, 1, 1),
        sources or ["yahoo", "bacen", "investidor10"],
        delay=0,
        progress=lambda _: None,
    )


def test_full_postgres_pipeline_all_assets_and_recalculation(
    database, monkeypatch, tmp_path
):
    db, url = database
    visited = []

    def capture(client, endpoint, params):
        visited.append(endpoint)
        return fake_fetch(client, endpoint, params)

    monkeypatch.setattr("finprov.pipeline.fetch", capture)
    result = run_collection(url)
    assert result["status"] == "ready", result
    assert len(visited) == len(set(visited)) == 40
    run_id = result["run_id"]
    tables = db.tables(run_id)
    for index, ticker in enumerate(TICKERS):
        current = next(
            r
            for r in tables["indicators"]
            if r["ticker"] == ticker
            and r["period"] == "current"
            and r["indicator"] == "P/L"
        )
        assert current["value"] == index + 1.25
        assert current["raw_id"]
        assert current["source_url"].endswith(f"/{ticker.lower()}/")
    # Nenhuma consulta à rede na validação ou na exportação.
    monkeypatch.setattr("finprov.pipeline.fetch", lambda *a: pytest.fail("network"))
    check = validate_run(url, run_id, recalculate=True)
    assert check["status"] == "ok", check
    assert all(check["recalculated"].values())
    assert len(db.inputs(run_id)) == 40
    assert all(
        r["_raw"] and r["normalization_status"] == "ok" for r in db.inputs(run_id)
    )
    assert export_files(url, tmp_path)["status"] == "exported"
    exported = tmp_path / "runs" / run_id
    assert json.loads((exported / "manifest.json").read_bytes())["run_id"] == run_id
    assert len(json.loads((exported / "prices.json").read_bytes())) == 13000
    assert not (tmp_path / "objects").exists()


def test_partial_does_not_replace_ready_run(database, monkeypatch):
    db, url = database
    monkeypatch.setattr("finprov.pipeline.fetch", fake_fetch)
    ready = run_collection(url)

    def broken(client, endpoint, params):
        if "/acoes/vale3/" in endpoint:
            return b"<h1>VALE3</h1><p>Layout changed</p>"
        return fake_fetch(client, endpoint, params)

    monkeypatch.setattr("finprov.pipeline.fetch", broken)
    partial = run_collection(url, ["investidor10"])
    assert partial["status"] == "partial"
    assert db.resolve_run() == ready["run_id"]
    inputs = db.inputs(partial["run_id"])
    bad = next(
        r for r in inputs if r["ticker"] == "VALE3" and r["source"] == "investidor10"
    )
    assert b"Layout changed" in bad["_raw"]
    assert bad["normalization_status"] == "error"
    assert validate_run(url, partial["run_id"])["status"] == "failed"
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT run_id::text FROM curated.precos_atuais")
        assert cur.fetchall() == [(ready["run_id"],)]


def test_network_failure_is_preserved(database, monkeypatch):
    db, url = database

    def broken(*args):
        raise ConnectionError("provider unavailable")

    monkeypatch.setattr("finprov.pipeline.fetch", broken)
    result = run_collection(url, ["investidor10"])
    assert result["status"] == "partial"
    assert len(db.inputs(result["run_id"])) == 12
    assert all(
        r["status"] == "error" and r["_raw"] is None
        for r in db.inputs(result["run_id"])
    )


def test_constraints_and_curated_transaction_rollback(database, monkeypatch):
    db, url = database
    monkeypatch.setattr("finprov.pipeline.fetch", fake_fetch)
    result = run_collection(url)
    before = db.tables(result["run_id"])
    # Repetir a carga sem identidade nova não duplica linhas.
    with pytest.raises(psycopg2.IntegrityError):
        db.save_curated(result["run_id"], before, [])
    assert db.tables(result["run_id"]) == before
    with db.connection() as conn, conn.cursor() as cur:
        with pytest.raises(psycopg2.IntegrityError):
            cur.execute("UPDATE curated.precos SET adjusted_close=-1")
        conn.rollback()
    assert db.tables(result["run_id"]) == before


def test_reset_guard_and_public_preserved(database):
    db, _ = database
    with pytest.raises(ValueError, match="nome exato"):
        db.initialize(reset=True, confirm_database="wrong")
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE TABLE public.unrelated_test(value integer)")
        cur.execute("INSERT INTO public.unrelated_test VALUES(17)")
    try:
        db.initialize(reset=True, confirm_database=db.database)
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT value FROM public.unrelated_test")
            assert cur.fetchone()[0] == 17
    finally:
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute("DROP TABLE public.unrelated_test")


def test_reset_refuses_unknown_objects_atomically(database):
    db, _ = database
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE TABLE curated.unrelated_test(value integer)")
    try:
        with pytest.raises(ValueError, match="não reconhecidas"):
            db.initialize(reset=True, confirm_database=db.database)
        assert ("controle", "execucoes") in db.inventory()
    finally:
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute("DROP TABLE curated.unrelated_test")
