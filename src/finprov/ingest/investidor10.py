"""Somente página pública e o histórico gratuito de cinco anos nela anunciado."""

import re

from finprov.config import TICKERS


def page_url(ticker: str) -> str:
    if ticker not in TICKERS:
        raise ValueError("Ticker não configurado.")
    return f"https://investidor10.com.br/acoes/{ticker.lower()}/"


def history_url(html: str) -> str | None:
    # O ID é descoberto na própria página, nunca inferido do ticker/empresa.
    match = re.search(r"/api/historico-indicadores/(\d+)/\$\{period\}/\?v=(\d+)", html)
    if not match:
        return None
    return (
        "https://investidor10.com.br/api/historico-indicadores/"
        f"{match[1]}/5/?v={match[2]}"
    )
