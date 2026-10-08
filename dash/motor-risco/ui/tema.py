"""Paleta, layout dos gráficos e formatação numérica em pt-BR.

Cada cor tem uma função e só uma:

- ``serie``    identidade de uma série (ordem fixa, validada para daltonismo);
- ``rampa``    intensidade em uma mesma grandeza (um matiz, claro → escuro);
- ``bom`` / ``alerta`` / ``critico``   estado. Nunca identificam uma série;
- ``tinta*``   todo texto. Rótulo nenhum usa a cor da série.
"""

from __future__ import annotations

import streamlit as st

_CLARO = {
    "superficie": "#fcfcfb",
    "tinta": "#0b0b0b",
    "tinta2": "#52514e",
    "tinta3": "#898781",
    "grade": "#e1e0d9",
    "eixo": "#c3c2b7",
    "neutro": "#c3c2b7",
    "serie": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "rampa": ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
    "bom": "#0ca30c",
    "alerta": "#fab219",
    "critico": "#d03b3b",
    "divergente": ["#1c5cab", "#f0efec", "#d03b3b"],
}

_ESCURO = {
    "superficie": "#1a1a19",
    "tinta": "#ffffff",
    "tinta2": "#c3c2b7",
    "tinta3": "#898781",
    "grade": "#2c2c2a",
    "eixo": "#383835",
    "neutro": "#52514e",
    "serie": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    "rampa": ["#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
    "bom": "#0ca30c",
    "alerta": "#fab219",
    "critico": "#e66767",
    "divergente": ["#3987e5", "#383835", "#e66767"],
}


def modo() -> str:
    """'light' ou 'dark', conforme o tema ativo no navegador."""
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def cores() -> dict:
    return _ESCURO if modo() == "dark" else _CLARO


def com_alfa(hex_color: str, alfa: float) -> str:
    """Converte '#rrggbb' em 'rgba(r,g,b,alfa)'."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alfa})"


def layout(altura: int = 380, legenda: bool = True, **extra) -> dict:
    """Layout comum: fundo transparente, grade em fio contínuo, texto em tinta."""
    c = cores()
    eixo = {
        "showgrid": True,
        "gridcolor": c["grade"],
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": c["eixo"],
        "tickfont": {"color": c["tinta2"], "size": 12},
        "title": {"font": {"color": c["tinta2"], "size": 12}},
        "automargin": True,
    }
    base = {
        "height": altura,
        "margin": {"l": 8, "r": 16, "t": 28 if legenda else 12, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": c["tinta2"], "size": 13},
        "separators": ",.",
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": c["superficie"],
            "bordercolor": c["eixo"],
            "font": {"color": c["tinta"], "size": 13},
        },
        "showlegend": legenda,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": c["tinta2"], "size": 12},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "xaxis": dict(eixo),
        "yaxis": dict(eixo),
        "bargap": 0.06,
    }
    base.update(extra)
    return base


# ── Formatação pt-BR ──────────────────────────────────────────────────────
def _troca(texto: str) -> str:
    """1,234.56 → 1.234,56"""
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(valor: float, casas: int = 2, sinal: bool = False) -> str:
    formato = f"{{:{'+' if sinal else ''}.{casas}f}}"
    return _troca(formato.format(valor * 100)) + "%"


def brl(valor: float, casas: int = 0) -> str:
    return "R$ " + _troca(f"{valor:,.{casas}f}")


def num(valor: float, casas: int = 2) -> str:
    return _troca(f"{valor:,.{casas}f}")


def nivel(confianca: float) -> str:
    """0.975 → '97,5%'; 0.95 → '95%'."""
    return f"{round(confianca * 100, 4):g}%".replace(".", ",")


def pp(diferenca: float, casas: int = 1) -> str:
    """Diferença entre dois percentuais, em pontos percentuais."""
    return _troca(f"{diferenca * 100:+.{casas}f}") + " p.p."


def prazo(dias: int) -> str:
    return "1 dia" if dias == 1 else f"{dias} dias úteis"


def inteiro(valor: float) -> str:
    return _troca(f"{int(valor):,}")


ROTULO_METODO = {
    "historical": "Histórico",
    "parametric": "Paramétrico",
    "ewma": "EWMA",
    "cornish_fisher": "Cornish-Fisher",
    "montecarlo": "Monte Carlo",
}
