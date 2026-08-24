"""Paleta, tema e utilidades visuais compartilhadas pela tela de proposta."""

from __future__ import annotations

import streamlit as st

# ── Paleta ────────────────────────────────────────────────────────────────
PALETTE = {
    "bg": "#0D1117",
    "surface": "#161B22",
    "grid": "#243040",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "primary": "#8B5CF6",
    "risk": "#EF4444",
    "prov": "#10B981",
    "accent": "#38BDF8",
    "warn": "#F59E0B",
    "pink": "#EC4899",
}

# Cor fixa por método -> mesma cor em todo gráfico, tabela e legenda do app.
METHOD_COLORS = {
    "historical": PALETTE["warn"],
    "parametric": PALETTE["primary"],
    "ewma": PALETTE["accent"],
    "cornish_fisher": PALETTE["pink"],
    "montecarlo": PALETTE["risk"],
    "expected_shortfall": PALETTE["prov"],
    "portfolio": PALETTE["primary"],
}

METHOD_LABELS = {
    "historical": "VaR Histórico",
    "parametric": "VaR Paramétrico",
    "ewma": "VaR EWMA (RiskMetrics)",
    "cornish_fisher": "VaR Cornish-Fisher",
    "montecarlo": "VaR Monte Carlo",
    "expected_shortfall": "Expected Shortfall",
    "portfolio": "VaR de Carteira",
}


def theme_type() -> str:
    """'dark' | 'light' — lido do tema ativo do Streamlit, com fallback seguro."""
    try:
        return st.context.theme.type or "dark"
    except Exception:
        return "dark"


def plotly_template() -> str:
    return "plotly_dark" if theme_type() == "dark" else "plotly_white"


def axis_text_color() -> str:
    return PALETTE["text"] if theme_type() == "dark" else "#1F2937"


def annotation_bg() -> str:
    """Fundo das anotações — acompanha o tema (evita chip escuro em página clara)."""
    return "rgba(13,17,23,.75)" if theme_type() == "dark" else "rgba(255,255,255,.85)"


def base_layout(title: str | None = None, height: int = 380) -> dict:
    """Layout padrão dos gráficos.

    Fundo **transparente** de propósito: o beta fixava ``paper_bgcolor="#0D1117"``,
    o que desenhava um retângulo escuro no meio da página quando o usuário
    estava no tema claro. Transparente herda o fundo real do Streamlit.
    """
    return dict(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)) if title else None,
        template=plotly_template(),
        height=height,
        margin=dict(l=8, r=8, t=46 if title else 16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=axis_text_color(), size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )


CUSTOM_CSS = """
<style>
/* Tabs mais legíveis e com respiro */
div[data-testid="stTabs"] button[role="tab"] { padding: 0.4rem 0.9rem; }
div[data-testid="stTabs"] button[role="tab"] p { font-size: 0.92rem; font-weight: 500; }

/* Cartões de métrica com borda sutil — evita o "texto solto" do layout padrão */
div[data-testid="stMetric"] {
    background: rgba(139, 92, 246, 0.06);
    border: 1px solid rgba(139, 92, 246, 0.22);
    border-radius: 10px;
    padding: 12px 14px 10px 14px;
}
div[data-testid="stMetricLabel"] p { font-size: 0.78rem; letter-spacing: .02em; }
div[data-testid="stMetricValue"] { font-size: 1.55rem; }

/* Dataframes e gráficos coladinhos no cartão */
div[data-testid="stDataFrame"] { border-radius: 8px; }

/* Sidebar um pouco mais estreita e densa */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.55rem; }

/* Badge de status usado no cabeçalho */
.fp-badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.74rem; font-weight: 600; letter-spacing: .02em; margin-right: 6px;
    border: 1px solid transparent;
}
.fp-badge-ok   { background: rgba(16,185,129,.14); color:#10B981; border-color: rgba(16,185,129,.35); }
.fp-badge-warn { background: rgba(245,158,11,.14); color:#F59E0B; border-color: rgba(245,158,11,.35); }
.fp-badge-info { background: rgba(56,189,248,.14); color:#38BDF8; border-color: rgba(56,189,248,.35); }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="fp-badge fp-badge-{kind}">{text}</span>'
