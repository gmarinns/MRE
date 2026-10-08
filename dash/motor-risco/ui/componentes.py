"""Peças de página reutilizadas: cabeçalho, indicadores e cartão de gráfico."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.tema import brl, pct

CONFIG_PLOTLY = {"displayModeBar": False, "locale": "pt-BR"}


def cabecalho(titulo: str, pergunta: str) -> None:
    """Título da página e a pergunta que a medida responde."""
    st.title(titulo)
    st.markdown(f":gray[{pergunta}]")


def indicadores(itens: list[dict]) -> None:
    """Linha de indicadores. Cada item aceita as chaves de ``st.metric``."""
    for coluna, item in zip(st.columns(len(itens)), itens, strict=True):
        coluna.metric(border=True, height="stretch", **item)


def perda(fracao: float, posicao: float) -> dict:
    """Indicador de perda: percentual em destaque, valor em reais abaixo."""
    return {
        "value": pct(fracao),
        "delta": brl(fracao * posicao),
        "delta_color": "off",
        "delta_arrow": "off",
    }


def grafico(
    fig: go.Figure,
    titulo: str,
    legenda: str | None = None,
    dados: pd.DataFrame | None = None,
    chave: str | None = None,
) -> None:
    """Cartão com título, gráfico e a tabela equivalente em "Ver dados"."""
    with st.container(border=True):
        st.markdown(f"**{titulo}**")
        if legenda:
            st.caption(legenda)
        st.plotly_chart(
            fig, width="stretch", theme=None, config=CONFIG_PLOTLY, key=chave
        )
        if dados is not None:
            with st.expander("Ver dados"):
                st.dataframe(dados, width="stretch", hide_index=True)


def como_ler(texto: str) -> None:
    with st.expander("Como ler esta página", icon=":material/menu_book:"):
        st.markdown(texto)


def estado(ok: bool, texto_ok: str, texto_ruim: str) -> None:
    """Estado com ícone e rótulo (a cor nunca vai sozinha)."""
    if ok:
        st.success(texto_ok, icon=":material/check_circle:")
    else:
        st.error(texto_ruim, icon=":material/error:")
