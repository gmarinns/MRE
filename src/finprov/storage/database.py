"""PostgreSQL como fonte única: raw -> curated -> resultados."""

import json
import math
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values
from sqlalchemy.engine import make_url

TABLES = {
    "prices": (
        "precos",
        "ticker date open high low close adjusted_close volume currency timezone "
        "adjustment quality_flags",
    ),
    "events": (
        "eventos",
        "ticker event_type source_event_id raw_type ex_date com_date payment_date "
        "amount provider_value split_factor currency tax_basis",
    ),
    "indicators": (
        "indicadores",
        "ticker indicator indicator_group value raw_value unit period reference_year "
        "published_at available_at point_in_time_verified",
    ),
    "macro": (
        "macro",
        "series date name value unit frequency published_at available_at",
    ),
}
OWNED = {
    "raw": {"cotacoes", "respostas"},
    "curated": {"precos", "ativos", "eventos", "indicadores", "macro"},
    "controle": {"esquema", "execucoes", "posicoes", "qualidade", "cobertura"},
    "resultados": {"relatorios"},
}


def native(value):
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [native(v) for v in value]
    return value


class Database:
    def __init__(self, url):
        parsed = make_url(url)
        if parsed.get_backend_name() != "postgresql":
            raise ValueError("O ETL unificado requer PostgreSQL.")
        self.database = parsed.database
        self.options = dict(
            host=parsed.host or "localhost",
            port=parsed.port or 5432,
            dbname=parsed.database,
            user=parsed.username,
            password=parsed.password,
            connect_timeout=5,
        )

    @contextmanager
    def connection(self):
        connection = psycopg2.connect(**self.options)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def inventory(self):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT schemaname,tablename FROM pg_tables WHERE schemaname "
                "NOT IN ('pg_catalog','information_schema') ORDER BY 1,2"
            )
            return cur.fetchall()

    def initialize(self, reset=False, confirm_database=None):
        """Reset explícito e restrito; dependências externas impedem a remoção."""
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(71301801)")
            if reset:
                cur.execute("SELECT current_database()")
                if confirm_database != cur.fetchone()[0]:
                    raise ValueError("Confirme o nome exato do banco para reconstruir.")
                cur.execute(
                    "SELECT schemaname,tablename FROM pg_tables WHERE "
                    "schemaname IN ('raw','curated','controle','resultados')"
                )
                unknown = [(s, t) for s, t in cur.fetchall() if t not in OWNED[s]]
                if unknown:
                    raise ValueError(
                        f"Tabelas não reconhecidas; reset recusado: {unknown}"
                    )
                cur.execute("DROP VIEW IF EXISTS curated.precos_atuais")
                cur.execute("DROP VIEW IF EXISTS curated.indicadores_atuais")
                cur.execute("DROP VIEW IF EXISTS controle.ultima_execucao_pronta")
                for schema, names in [
                    ("public", ["finprov_records", "finprov_runs"]),
                    ("resultados", ["relatorios"]),
                    ("controle", ["cobertura", "qualidade", "posicoes"]),
                    ("curated", ["precos", "eventos", "indicadores", "macro"]),
                    ("raw", ["respostas", "cotacoes"]),
                    ("curated", ["ativos"]),
                    ("controle", ["execucoes", "esquema"]),
                ]:
                    for name in names:
                        cur.execute(
                            sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                                sql.Identifier(schema), sql.Identifier(name)
                            )
                        )
                for schema in OWNED:
                    cur.execute(
                        sql.SQL("DROP SCHEMA IF EXISTS {}").format(
                            sql.Identifier(schema)
                        )
                    )
            cur.execute("SELECT to_regclass('controle.esquema')")
            if cur.fetchone()[0]:
                cur.execute("SELECT version FROM controle.esquema")
                if cur.fetchall() != [(1,)]:
                    raise ValueError("Versão do banco incompatível.")
                return {"status": "already_initialized", "version": 1}
            cur.execute(
                Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            )
        return {"status": "initialized", "version": 1}

    def start_run(self, parameters, environment):
        portfolio = parameters["portfolio"]
        weights = portfolio["weights"]
        if (
            set(weights) != set(portfolio["tickers"])
            or not weights
            or any(not math.isfinite(w) or w < 0 for w in weights.values())
            or not math.isclose(sum(weights.values()), 1, abs_tol=1e-9)
        ):
            raise ValueError("Composição da carteira inválida.")
        run_id = str(uuid4())
        with self.connection() as conn, conn.cursor() as cur:
            for ticker in [*portfolio["tickers"], "^BVSP"]:
                benchmark = ticker.startswith("^")
                cur.execute(
                    "INSERT INTO curated.ativos VALUES (%s,%s,%s) "
                    "ON CONFLICT (ticker) DO NOTHING",
                    (
                        ticker,
                        ticker if benchmark else ticker + ".SA",
                        "benchmark" if benchmark else "equity",
                    ),
                )
            cur.execute(
                "INSERT INTO controle.execucoes(run_id,status,parameters,environment) "
                "VALUES (%s,'collecting',%s,%s)",
                (run_id, Json(parameters), Json(environment)),
            )
            execute_values(
                cur,
                "INSERT INTO controle.posicoes VALUES %s",
                [(run_id, t, w) for t, w in weights.items()],
            )
        return run_id

    def add_response(self, run_id, entry):
        raw_id = str(uuid4())
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO raw.respostas(raw_id,run_id,source,ticker,series,url,"
                "params,observed_at,status,body,error,normalization_status) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    raw_id,
                    run_id,
                    entry["source"],
                    entry.get("ticker"),
                    entry.get("series"),
                    entry["url"],
                    Json(entry.get("params", {})),
                    entry["observed_at"],
                    entry["status"],
                    entry.get("_raw"),
                    entry.get("error"),
                    "pending" if entry["status"] == "ok" else "not_applicable",
                ),
            )
        return raw_id

    def inputs(self, run_id):
        with (
            self.connection() as conn,
            conn.cursor(cursor_factory=RealDictCursor) as cur,
        ):
            cur.execute(
                "SELECT * FROM raw.respostas WHERE run_id=%s "
                "ORDER BY observed_at,raw_id",
                (run_id,),
            )
            rows = []
            for row in cur.fetchall():
                entry = native(dict(row))
                body = entry.pop("body")
                entry["_raw"] = bytes(body) if body is not None else None
                rows.append(entry)
            return rows

    def mark(self, run_id, status, error=None):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE controle.execucoes SET status=%s,error=%s,"
                "finished_at=CASE WHEN %s IN ('failed','partial','ready') "
                "THEN now() ELSE NULL END WHERE run_id=%s",
                (status, error, status, run_id),
            )

    def save_curated(self, run_id, tables, errors):
        with self.connection() as conn, conn.cursor() as cur:
            for dataset, (table, fields) in TABLES.items():
                columns = ["run_id", "raw_id", *fields.split()]
                if dataset == "events":
                    columns.append("row_number")
                values = []
                for index, original in enumerate(tables[dataset]):
                    row = {**original, "run_id": run_id, "row_number": index}
                    row["indicator_group"] = row.get("group")
                    values.append(tuple(row.get(c) for c in columns))
                if values:
                    query = sql.SQL("INSERT INTO curated.{} ({}) VALUES %s").format(
                        sql.Identifier(table),
                        sql.SQL(",").join(map(sql.Identifier, columns)),
                    )
                    execute_values(cur, query.as_string(conn), values, page_size=1000)
            cur.execute(
                "UPDATE raw.respostas SET normalization_status='ok' "
                "WHERE run_id=%s AND status='ok'",
                (run_id,),
            )
            for error in errors:
                cur.execute(
                    "UPDATE raw.respostas SET normalization_status='error',"
                    "normalization_error=%s WHERE run_id=%s AND raw_id=%s",
                    (error["error"], run_id, error["raw_id"]),
                )

    def tables(self, run_id):
        result = {}
        with (
            self.connection() as conn,
            conn.cursor(cursor_factory=RealDictCursor) as cur,
        ):
            for dataset, (table, _) in TABLES.items():
                cur.execute(
                    sql.SQL(
                        "SELECT c.*,r.source,r.url AS source_url,r.observed_at "
                        "FROM curated.{} c JOIN raw.respostas r "
                        "ON c.raw_id=r.raw_id AND c.run_id=r.run_id "
                        "WHERE c.run_id=%s"
                    ).format(sql.Identifier(table)),
                    (run_id,),
                )
                rows = []
                for record in cur.fetchall():
                    row = native(dict(record))
                    row.pop("run_id")
                    row.pop("row_number", None)
                    if "indicator_group" in row:
                        row["group"] = row.pop("indicator_group")
                    rows.append(row)
                result[dataset] = sorted(
                    rows, key=lambda r: json.dumps(r, sort_keys=True)
                )
        return result

    def finish(self, run_id, risk, direction, quality, coverage, status):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO controle.qualidade VALUES (%s,%s)", (run_id, Json(quality))
            )
            execute_values(
                cur,
                "INSERT INTO controle.cobertura VALUES %s",
                [(run_id, r["ticker"], Json(r)) for r in coverage],
            )
            execute_values(
                cur,
                "INSERT INTO resultados.relatorios VALUES %s",
                [(run_id, "risk", Json(risk)), (run_id, "direction", Json(direction))],
            )
            cur.execute(
                "UPDATE controle.execucoes SET status=%s,finished_at=now() "
                "WHERE run_id=%s",
                (status, run_id),
            )

    def resolve_run(self, run_id="latest"):
        with self.connection() as conn, conn.cursor() as cur:
            if run_id == "latest":
                cur.execute("SELECT run_id FROM controle.ultima_execucao_pronta")
            else:
                cur.execute(
                    "SELECT run_id FROM controle.execucoes WHERE run_id=%s", (run_id,)
                )
            row = cur.fetchone()
            if row is None:
                raise ValueError("Execução não encontrada; latest exige status ready.")
            return str(row[0])

    def run(self, run_id="latest"):
        run_id = self.resolve_run(run_id)
        with (
            self.connection() as conn,
            conn.cursor(cursor_factory=RealDictCursor) as cur,
        ):
            cur.execute("SELECT * FROM controle.execucoes WHERE run_id=%s", (run_id,))
            result = native(dict(cur.fetchone()))
            cur.execute(
                "SELECT kind,payload FROM resultados.relatorios WHERE run_id=%s",
                (run_id,),
            )
            result["reports"] = {r["kind"]: r["payload"] for r in cur.fetchall()}
            cur.execute(
                "SELECT payload FROM controle.qualidade WHERE run_id=%s", (run_id,)
            )
            row = cur.fetchone()
            result["quality"] = row["payload"] if row else None
            cur.execute(
                "SELECT payload FROM controle.cobertura WHERE run_id=%s "
                "ORDER BY ticker",
                (run_id,),
            )
            result["coverage"] = [r["payload"] for r in cur.fetchall()]
            return result
