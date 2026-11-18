"""Universo e hipóteses explícitas da carteira experimental."""

import math
import tomllib
from pathlib import Path

TICKERS = (
    "AGRO3",
    "VALE3",
    "BBSE3",
    "FLRY3",
    "SAPR4",
    "GOAU3",
    "KLBN3",
    "PETR4",
    "ABEV3",
    "CPFE3",
    "BBAS3",
    "VLID3",
)


def load_portfolio(path: str | Path | None = None) -> dict:
    config = {"name": "Carteira TCC", "tickers": list(TICKERS), "weighting": "equal"}
    if path:
        with Path(path).open("rb") as stream:
            config = tomllib.load(stream)["portfolio"]
    tickers = config["tickers"]
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("A carteira deve conter ativos únicos.")
    if any(t not in TICKERS for t in tickers):
        raise ValueError("Ativo fora do universo definido para este TCC.")
    if config.get("weighting") == "equal":
        weights = [1 / len(tickers)] * len(tickers)
    else:
        weights = config.get("weights", [])
    if (
        len(weights) != len(tickers)
        or any(not math.isfinite(w) or w < 0 for w in weights)
        or not math.isclose(sum(weights), 1, abs_tol=1e-9)
    ):
        raise ValueError("Pesos devem ser finitos, não negativos e somar 1.")
    return {
        "name": config.get("name", "Carteira TCC"),
        "tickers": tickers,
        "weights": dict(zip(tickers, weights, strict=True)),
    }
