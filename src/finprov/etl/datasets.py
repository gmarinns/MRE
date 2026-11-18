"""Normalização explícita; valores ausentes nunca viram zero ou preço ajustado."""

import math
import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from finprov.ingest.bacen import SERIES


def number(value):
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Valor numérico não finito.")
    return result


def pt_number(value: str):
    text = value.strip().replace("\xa0", "").replace(" ", "")
    if text in {"", "-", "--", "—", "N/A", "N/D"}:
        return None
    return number(
        text.replace("R$", "").replace("%", "").replace(".", "").replace(",", ".")
    )


def stamp_date(timestamp, timezone):
    return (
        datetime.fromtimestamp(timestamp, UTC)
        .astimezone(ZoneInfo(timezone))
        .date()
        .isoformat()
    )


def yahoo(payload: dict, ticker: str) -> tuple[list, list]:
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo sem série válida para {ticker}: {chart.get('error')}")
    result = chart["result"][0]
    expected = ticker if ticker.startswith("^") else ticker + ".SA"
    meta = result["meta"]
    if meta["symbol"] != expected or meta.get("currency") != "BRL":
        raise ValueError(f"Símbolo/moeda inesperado para {ticker}.")
    tz = meta.get("exchangeTimezoneName", "America/Sao_Paulo")
    timestamps = result.get("timestamp") or []
    if not timestamps:
        raise ValueError(f"Yahoo retornou série vazia: {ticker}")
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    columns = {
        key: quote.get(key, [None] * len(timestamps))
        for key in ("open", "high", "low", "close", "volume")
    }
    columns["adjusted_close"] = adj if adj is not None else [None] * len(timestamps)
    if any(len(values) != len(timestamps) for values in columns.values()):
        raise ValueError("Comprimento de colunas Yahoo inconsistente.")
    prices, seen = [], set()
    for index, ts in enumerate(timestamps):
        day = stamp_date(ts, tz)
        if day in seen:
            raise ValueError(f"Pregão duplicado: {ticker} {day}")
        seen.add(day)
        row = {key: number(values[index]) for key, values in columns.items()}
        flags = []
        for field in ("open", "high", "low", "close", "adjusted_close"):
            if row[field] is not None and row[field] <= 0:
                flags.append(f"invalid_{field}")
                row[field] = None
        if row["close"] is None:
            flags.append("missing_close")
        if row["adjusted_close"] is None:
            flags.append("missing_adjustment")
        if row["volume"] is not None and row["volume"] < 0:
            flags.append("invalid_volume")
            row["volume"] = None
        if all(row[k] is not None for k in ("open", "high", "low", "close")):
            if (
                not row["low"]
                <= min(row["open"], row["close"])
                <= max(row["open"], row["close"])
                <= row["high"]
            ):
                flags.append("inconsistent_ohlc")
        prices.append(
            {
                "ticker": ticker,
                "date": day,
                "currency": "BRL",
                "timezone": tz,
                **row,
                "quality_flags": ";".join(flags),
                "adjustment": "provider_adjusted" if adj else "unavailable",
            }
        )
    events = []
    for kind, entries in result.get("events", {}).items():
        for key, event in entries.items():
            if kind not in {"dividends", "splits"}:
                continue
            row = {
                "ticker": ticker,
                "source_event_id": str(key),
                "ex_date": stamp_date(event["date"], tz),
                "com_date": None,
                "payment_date": None,
                "currency": "BRL",
                "amount": None,
                "split_factor": None,
                "tax_basis": "unspecified",
            }
            if kind == "dividends":
                row.update(
                    event_type="cash_distribution_unclassified",
                    amount=number(event["amount"]),
                )
                if row["amount"] is None or row["amount"] < 0:
                    raise ValueError("Provento Yahoo inválido.")
            else:
                factor = float(event["numerator"]) / float(event["denominator"])
                if not math.isfinite(factor) or factor <= 0:
                    raise ValueError("Fator de desdobramento inválido.")
                row.update(
                    event_type="reverse_split" if factor < 1 else "split",
                    split_factor=factor,
                )
            events.append(row)
    return sorted(prices, key=lambda r: r["date"]), events


def bacen(payload, code: int) -> list:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"SGS {code}: resposta vazia ou inválida.")
    rows = []
    for item in payload:
        day = datetime.strptime(item["data"], "%d/%m/%Y").date().isoformat()
        rows.append(
            {
                "series": code,
                "date": day,
                "value": number(item["valor"]),
                **SERIES[code],
                "published_at": None,
            }
        )
    if len({r["date"] for r in rows}) != len(rows):
        raise ValueError(f"SGS {code}: datas duplicadas.")
    return rows


def _br_date(value: str):
    return (
        datetime.strptime(value.strip(), "%d/%m/%Y").date().isoformat()
        if value.strip() not in {"", "-", "--", "A definir"}
        else None
    )


def investidor10(html: str, ticker: str) -> tuple[list, list]:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1")
    if heading is None or heading.get_text(strip=True).upper() != ticker:
        raise ValueError(f"Investidor10: página inesperada para {ticker}.")
    cards = soup.select("#table-indicators .indicator-card")
    if not cards:
        raise ValueError("Investidor10 mudou de layout ou não forneceu indicadores.")
    indicators = []
    for card in cards:
        title = card.select_one(".indicator-card-title")
        value = card.select_one(".indicator-card-value")
        if title is None or value is None:
            raise ValueError("Cartão de indicador incompleto.")
        label = title.get_text("", strip=True)
        raw = value.get_text("", strip=True)
        group = card.find_parent("section")
        group_title = group.find("h3") if group else None
        indicators.append(
            {
                "ticker": ticker,
                "indicator": label,
                "group": group_title.get_text(strip=True) if group_title else None,
                "value": pt_number(raw),
                "raw_value": raw,
                "unit": "percent" if "%" in raw else "provider_decimal",
                "reference_year": None,
                "period": "current",
                "published_at": None,
                "point_in_time_verified": False,
            }
        )
    table = soup.select_one("#table-dividends-history")
    if table is None:
        raise ValueError("Tabela de proventos ausente; não significa zero eventos.")
    events = []
    for tr in table.select("tbody tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        if len(cells) != 4:
            raise ValueError("Layout da tabela de proventos mudou.")
        raw_type, com_date, payment_date, amount = cells
        type_map = {
            "dividendos": "dividend",
            "jscp": "jcp",
            "jcp": "jcp",
            "juros sobre capital próprio": "jcp",
            "rend. trib.": "taxable_income",
            "bonificação": "stock_bonus",
        }
        value = pt_number(amount)
        if value is None or value < 0:
            raise ValueError("Valor de provento inválido.")
        event_type = type_map.get(raw_type.lower(), "unclassified_event")
        is_cash = event_type in {"dividend", "jcp", "taxable_income"}
        events.append(
            {
                "ticker": ticker,
                "event_type": event_type,
                "raw_type": raw_type,
                "com_date": _br_date(com_date),
                "payment_date": _br_date(payment_date),
                "ex_date": None,
                "amount": value if is_cash else None,
                "provider_value": value,
                "currency": "BRL" if is_cash else None,
                "split_factor": None,
                "tax_basis": "unspecified",
            }
        )
    return indicators, events


def indicator_history(payload, ticker: str) -> list:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Histórico de indicadores indisponível.")
    rows = []
    for label, entries in payload.items():
        if not isinstance(entries, list):
            raise ValueError("Formato do histórico de indicadores mudou.")
        for item in entries:
            year = str(item["year"])
            if year.lower() == "atual":
                continue  # Indicadores atuais vêm exclusivamente da página.
            if not re.fullmatch(r"\d{4}", year):
                raise ValueError(f"Ano inválido: {year}")
            rows.append(
                {
                    "ticker": ticker,
                    "indicator": label,
                    "value": number(item.get("value")),
                    "unit": "percent"
                    if item.get("type") == "percent"
                    else "provider_decimal",
                    "reference_year": int(year),
                    "period": "annual",
                    "published_at": None,
                    "point_in_time_verified": False,
                }
            )
    if not rows:
        raise ValueError("Histórico sem observações anuais.")
    return rows
