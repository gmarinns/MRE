"""Yahoo Chart: preços e eventos, sem efeitos colaterais na importação."""

from datetime import UTC, date, datetime


def request_spec(ticker: str, start: date, end: date) -> tuple[str, dict]:
    if start >= end:
        raise ValueError("O fim exclusivo deve ser posterior ao início.")
    symbol = (
        ticker if ticker.startswith("^") or ticker.endswith(".SA") else ticker + ".SA"
    )
    params = {
        "period1": int(datetime.combine(start, datetime.min.time(), UTC).timestamp()),
        "period2": int(datetime.combine(end, datetime.min.time(), UTC).timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", params
