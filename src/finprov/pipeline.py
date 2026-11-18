"""ETL único: fontes -> raw PostgreSQL -> curated -> resultados."""

import csv
import io
import json
import platform
import subprocess
import time
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

from finprov.etl import datasets
from finprov.ingest import bacen, investidor10, yahoo
from finprov.ingest.http import fetch, session
from finprov.risk.direction import build_direction_report, direction_markdown
from finprov.risk.report import build_report, markdown_report
from finprov.storage.database import Database
from finprov.storage.runs import canonical_json, digest, run_path

PARSER_VERSION = "3"


def public_input(entry):
    return {key: value for key, value in entry.items() if key != "_raw"}


def coverage_rows(tables, inputs, errors, parameters):
    """Uma linha por posição: a cobertura não depende de um único exemplo."""

    def source_status(source, ticker):
        if any(e["source"] == source and e.get("ticker") == ticker for e in errors):
            return "normalization_error"
        entries = [
            e for e in inputs if e["source"] == source and e.get("ticker") == ticker
        ]
        return entries[0]["status"] if entries else "not_requested"

    rows = []
    for ticker in parameters["portfolio"]["tickers"]:
        prices = [r for r in tables["prices"] if r["ticker"] == ticker]
        current = [
            r
            for r in tables["indicators"]
            if r["ticker"] == ticker and r["period"] == "current"
        ]
        annual = [
            r
            for r in tables["indicators"]
            if r["ticker"] == ticker and r["period"] == "annual"
        ]
        rows.append(
            {
                "ticker": ticker,
                "weight": parameters["portfolio"]["weights"][ticker],
                "investidor10_url": investidor10.page_url(ticker),
                "yahoo_status": source_status("yahoo", ticker),
                "investidor10_status": source_status("investidor10", ticker),
                "investidor10_history_status": source_status(
                    "investidor10_history", ticker
                ),
                "price_rows": len(prices),
                "last_price_date": max((r["date"] for r in prices), default=None),
                "current_indicators": len(current),
                "current_indicators_with_value": sum(
                    r["value"] is not None for r in current
                ),
                "annual_indicators": len(annual),
                "annual_indicators_with_value": sum(
                    r["value"] is not None for r in annual
                ),
                "investidor10_events": sum(
                    r["ticker"] == ticker and r["source"] == "investidor10"
                    for r in tables["events"]
                ),
            }
        )
    return rows


def environment() -> dict:
    source = Path(__file__).parent
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=source,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    return {
        "python": platform.python_version(),
        "parser_version": PARSER_VERSION,
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "requests",
                "beautifulsoup4",
                "scikit-learn",
                "scipy",
            )
        },
        "git_commit": git_sha,
    }


def normalize(inputs: list, start: str, end: str) -> tuple[dict, list]:
    tables = {name: [] for name in ("prices", "events", "indicators", "macro")}
    errors = []
    for entry in inputs:
        if entry["status"] != "ok":
            continue
        raw = entry["_raw"]
        source = entry["source"]
        try:
            if source == "yahoo":
                prices, events = datasets.yahoo(json.loads(raw), entry["ticker"])
                batch = {
                    "prices": [r for r in prices if start <= r["date"] < end],
                    "events": events,
                }
                if not batch["prices"]:
                    raise ValueError("Nenhum preço dentro do intervalo solicitado.")
            elif source == "bacen":
                rows = datasets.bacen(json.loads(raw), entry["series"])
                batch = {"macro": [r for r in rows if start <= r["date"] < end]}
            elif source == "investidor10":
                indicators, events = datasets.investidor10(
                    raw.decode("utf-8"), entry["ticker"]
                )
                batch = {"indicators": indicators, "events": events}
            elif source == "investidor10_history":
                batch = {
                    "indicators": datasets.indicator_history(
                        json.loads(raw), entry["ticker"]
                    )
                }
            else:
                raise ValueError(f"Fonte desconhecida: {source}")
            for name, rows in batch.items():
                for row in rows:
                    row.update(
                        source=source,
                        raw_id=entry.get("raw_id"),
                        source_url=entry["url"],
                        observed_at=entry["observed_at"],
                    )
                    if name in {"indicators", "macro"}:
                        row["available_at"] = entry["observed_at"]
                    tables[name].append(row)
        except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
            errors.append(
                {
                    "source": source,
                    "ticker": entry.get("ticker"),
                    "series": entry.get("series"),
                    "error": str(exc),
                    "raw_id": entry.get("raw_id"),
                }
            )
    for rows in tables.values():
        rows.sort(key=lambda r: canonical_json(r))
    return tables, errors


def csv_bytes(rows: list) -> bytes:
    stream = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(
            stream, fieldnames=sorted({k for r in rows for k in r}), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def readiness(parameters, tables, inputs, errors, risk, direction):
    problems = []
    if set(parameters["sources"]) != {"yahoo", "bacen", "investidor10"}:
        problems.append("A execução completa exige Yahoo, BACEN e Investidor10.")
    if errors or any(e["status"] != "ok" for e in inputs):
        problems.append("Há falhas de coleta ou normalização.")
    coverage = coverage_rows(tables, inputs, errors, parameters)
    for row in coverage:
        ticker = row["ticker"]
        if (
            row["investidor10_status"] != "ok"
            or row["investidor10_history_status"] != "ok"
            or not row["current_indicators_with_value"]
            or not row["annual_indicators_with_value"]
        ):
            problems.append(f"{ticker}: fundamentos incompletos.")
        if risk.get("assets", {}).get(ticker, {}).get("status") != "ok":
            problems.append(f"{ticker}: risco indisponível.")
        if direction.get("assets", {}).get(ticker, {}).get("status") != "experimental":
            problems.append(f"{ticker}: direção indisponível.")
    if risk.get("status") != "ok":
        problems.append("Risco da carteira indisponível.")
    if (direction.get("portfolio") or {}).get("status") != "experimental":
        problems.append("Direção da carteira indisponível.")
    for code in (1, 11, 433):
        if not any(
            r["series"] == code and r["value"] is not None for r in tables["macro"]
        ):
            problems.append(f"BACEN {code}: sem valores.")
    return problems, coverage


def collect(
    database_url,
    portfolio: dict,
    start: date,
    end: date,
    sources: list[str],
    confidence=0.95,
    window=252,
    delay=0.5,
    progress=print,
):
    if (
        start >= end
        or not sources
        or len(sources) != len(set(sources))
        or set(sources) - {"yahoo", "bacen", "investidor10"}
    ):
        raise ValueError("Intervalo ou fontes inválidos.")
    if not 0.5 < confidence < 1 or window < 20:
        raise ValueError("Confiança ou janela inválida.")
    db = Database(database_url)
    parameters = dict(
        portfolio=portfolio,
        start=start.isoformat(),
        end=end.isoformat(),
        sources=sources,
        confidence=confidence,
        window=window,
        direction=True,
    )
    run_id = db.start_run(parameters, environment())
    progress(f"Execução: {run_id}")
    try:
        with session() as client:

            def capture(source, url, params=None, **identity):
                entry = {
                    "source": source,
                    "url": url,
                    "params": params or {},
                    **identity,
                }
                progress(f"Coletando {source}: {identity}")
                try:
                    raw = fetch(client, url, params)
                    entry.update(status="ok", _raw=raw)
                except Exception as exc:
                    raw = None
                    entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
                entry["observed_at"] = datetime.now(UTC).isoformat()
                # Persiste antes de interpretar; erro do banco não é erro da fonte.
                db.add_response(run_id, entry)
                if delay:
                    time.sleep(delay)
                return raw

            if "yahoo" in sources:
                for ticker in [*portfolio["tickers"], "^BVSP"]:
                    url, params = yahoo.request_spec(ticker, start, end)
                    capture("yahoo", url, params, ticker=ticker)
            if "bacen" in sources:
                for code, url, params in bacen.requests_for(start, end):
                    capture("bacen", url, params, series=code)
            if "investidor10" in sources:
                for ticker in portfolio["tickers"]:
                    page = investidor10.page_url(ticker)
                    raw = capture("investidor10", page, ticker=ticker)
                    if raw is not None:
                        url = investidor10.history_url(
                            raw.decode("utf-8", errors="replace")
                        )
                        if url:
                            capture("investidor10_history", url, ticker=ticker)
                        else:
                            db.add_response(
                                run_id,
                                dict(
                                    source="investidor10_history",
                                    ticker=ticker,
                                    url=page,
                                    status="error",
                                    error="Endpoint não anunciado",
                                    observed_at=datetime.now(UTC).isoformat(),
                                ),
                            )
        db.mark(run_id, "processing")
        inputs = db.inputs(run_id)  # O ETL lê os bytes que realmente foram persistidos.
        tables, errors = normalize(inputs, parameters["start"], parameters["end"])
        db.save_curated(run_id, tables, errors)
        tables = db.tables(run_id)  # Os modelos leem curated, não as respostas HTTP.
        risk = build_report(tables["prices"], portfolio, confidence, window)
        direction = build_direction_report(tables["prices"], portfolio)
        problems, coverage = readiness(
            parameters, tables, inputs, errors, risk, direction
        )
        quality = {
            "normalization_errors": errors,
            "counts": {k: len(v) for k, v in tables.items()},
            "price_flags": [r for r in tables["prices"] if r["quality_flags"]],
            "missing_source_inputs": [
                public_input(r) for r in inputs if r["status"] != "ok"
            ],
            "problems": problems,
            "fundamentals_policy": "Observed now; not verified point-in-time history.",
            "event_policy": "Sources kept separate; never add twice.",
        }
        status = "partial" if problems else "ready"
        db.finish(run_id, risk, direction, quality, coverage, status)
        return {
            "run_id": run_id,
            "status": status,
            "counts": quality["counts"],
            "problems": problems,
        }
    except BaseException as exc:
        db.mark(run_id, "failed", f"{type(exc).__name__}: {exc}")
        raise


def export_files(database_url, root="data", run_id="latest"):
    """Exportação de conveniência; não é um segundo ETL nem alimenta os modelos."""
    db = Database(database_url)
    run = db.run(run_id)
    if run["status"] not in {"ready", "partial"}:
        raise ValueError("A execução ainda não possui resultados finalizados.")
    run_id = run["run_id"]
    files = {}
    for name, rows in db.tables(run_id).items():
        files[f"{name}.json"] = canonical_json(rows)
        files[f"{name}.csv"] = csv_bytes(rows)
    files["coverage.csv"] = csv_bytes(run["coverage"])
    files["coverage.json"] = canonical_json(run["coverage"])
    files["quality.json"] = canonical_json(run["quality"])
    files["report.json"] = canonical_json(run["reports"]["risk"])
    files["report.md"] = markdown_report(run["reports"]["risk"]).encode("utf-8")
    files["direction.json"] = canonical_json(run["reports"]["direction"])
    files["direction.md"] = direction_markdown(run["reports"]["direction"]).encode(
        "utf-8"
    )
    folder = run_path(root, run_id)
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (folder / name).write_bytes(content)
    manifest = {
        k: v for k, v in run.items() if k not in {"reports", "quality", "coverage"}
    }
    manifest["artifacts"] = {name: digest(content) for name, content in files.items()}
    content = canonical_json(manifest)
    (folder / "manifest.json").write_bytes(content)
    (folder / "manifest.sha256").write_text(digest(content), encoding="utf-8")
    (Path(root) / "latest.txt").write_text(run_id, encoding="utf-8")
    return {"run_id": run_id, "output": str(folder), "status": "exported"}
