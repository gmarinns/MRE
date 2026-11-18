"""Validação da execução lida do PostgreSQL, incluindo recálculo opcional."""

from finprov.pipeline import readiness
from finprov.risk.direction import build_direction_report
from finprov.risk.report import build_report
from finprov.storage.database import Database


def validate_run(database_url, run_id="latest", recalculate=False):
    db = Database(database_url)
    run = db.run(run_id)
    run_id = run["run_id"]
    tables = db.tables(run_id)
    inputs = db.inputs(run_id)
    parameters = run["parameters"]
    errors = [
        {
            "source": r["source"],
            "ticker": r["ticker"],
            "raw_id": r["raw_id"],
            "error": r["normalization_error"],
        }
        for r in inputs
        if r["normalization_status"] == "error"
    ]
    risk = run["reports"].get("risk", {})
    direction = run["reports"].get("direction", {})
    problems, coverage = readiness(parameters, tables, inputs, errors, risk, direction)
    if run["status"] != "ready":
        problems.append(f"Execução não pronta: {run['status']}.")
    if any(r["normalization_status"] == "pending" for r in inputs):
        problems.append("Há respostas sem normalização.")
    counts = {name: len(rows) for name, rows in tables.items()}
    if counts != (run["quality"] or {}).get("counts"):
        problems.append("Contagens atuais diferem do controle de qualidade salvo.")
    recalculated = None
    if recalculate:
        risk_now = build_report(
            tables["prices"],
            parameters["portfolio"],
            parameters["confidence"],
            parameters["window"],
        )
        direction_now = build_direction_report(
            tables["prices"], parameters["portfolio"]
        )
        recalculated = {
            "risk_identical": risk_now == risk,
            "direction_identical": direction_now == direction,
        }
        if not all(recalculated.values()):
            problems.append("Recálculo difere dos relatórios persistidos.")
    return {
        "run_id": run_id,
        "status": "ok" if not problems else "failed",
        "counts": counts,
        "raw_responses": len(inputs),
        "coverage": coverage,
        "problems": problems,
        "recalculated": recalculated,
        "price_flag_count": sum(bool(r["quality_flags"]) for r in tables["prices"]),
        "limitation": "Conferência técnica não prova capacidade preditiva.",
    }
