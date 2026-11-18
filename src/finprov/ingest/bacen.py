"""SGS: consultas em blocos de até cinco anos, sem confundir taxa com preço."""

from datetime import date, timedelta

SERIES = {
    11: {"name": "Selic diária", "unit": "percent_per_day", "frequency": "daily"},
    1: {"name": "USD/BRL venda", "unit": "BRL_per_USD", "frequency": "daily"},
    433: {"name": "IPCA", "unit": "percent_per_month", "frequency": "monthly"},
}


def requests_for(start: date, end: date):
    if start >= end:
        raise ValueError("Intervalo vazio.")
    for code in SERIES:
        cursor = start
        while cursor < end:
            boundary = min(date(cursor.year + 5, 1, 1), end)
            yield (
                code,
                f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados",
                {
                    "formato": "json",
                    "dataInicial": cursor.strftime("%d/%m/%Y"),
                    "dataFinal": (boundary - timedelta(days=1)).strftime("%d/%m/%Y"),
                },
            )
            cursor = boundary
