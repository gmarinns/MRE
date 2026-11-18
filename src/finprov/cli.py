"""CLI do ETL unificado PostgreSQL."""

import argparse
import json
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

import psycopg2

from finprov.config import load_portfolio
from finprov.pipeline import collect, export_files
from finprov.storage.database import Database
from finprov.validation import validate_run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="finprov")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("db-init", help="Criar o modelo PostgreSQL")
    setup.add_argument(
        "--reset", action="store_true", help="Excluir estruturas antigas do FinProv"
    )
    setup.add_argument("--confirm-database", help="Nome exato do banco a reconstruir")
    commands.add_parser("db-status", help="Listar tabelas e execuções")
    ingest = commands.add_parser(
        "collect", help="Fontes -> raw -> curated -> resultados"
    )
    ingest.add_argument("--config")
    ingest.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    ingest.add_argument(
        "--end",
        type=date.fromisoformat,
        default=datetime.now(ZoneInfo("America/Sao_Paulo")).date(),
        help="Fim exclusivo",
    )
    ingest.add_argument(
        "--sources",
        nargs="+",
        choices=["yahoo", "bacen", "investidor10"],
        default=["yahoo", "bacen", "investidor10"],
    )
    ingest.add_argument("--confidence", type=float, default=0.95)
    ingest.add_argument("--window", type=int, default=252)
    check = commands.add_parser("validate", help="Validar execução do banco")
    check.add_argument("run_id", nargs="?", default="latest")
    check.add_argument("--recalculate", action="store_true")
    export = commands.add_parser("export", help="Exportar CSV/JSON/Markdown do banco")
    export.add_argument("run_id", nargs="?", default="latest")
    export.add_argument("--data-dir", default="data")
    args = parser.parse_args(argv)
    url = os.environ.get("DATABASE_URL")
    if not url:
        parser.error("Defina DATABASE_URL para o PostgreSQL do FinProv.")
    try:
        db = Database(url)
        if args.command == "db-init":
            result = db.initialize(args.reset, args.confirm_database)
        elif args.command == "db-status":
            result = {"database": db.database, "tables": db.inventory()}
        elif args.command == "collect":
            result = collect(
                url,
                load_portfolio(args.config),
                args.start,
                args.end,
                args.sources,
                args.confidence,
                args.window,
            )
        elif args.command == "validate":
            result = validate_run(url, args.run_id, args.recalculate)
        else:
            result = export_files(url, args.data_dir, args.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("status") in {"partial", "failed"}:
            raise SystemExit(2)
    except (ValueError, OSError, psycopg2.Error) as exc:
        parser.exit(2, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
