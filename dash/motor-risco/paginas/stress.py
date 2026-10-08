"""Stress: o VaR sob choques hipotéticos e as piores janelas já observadas."""

import pandas as pd
import streamlit as st

import servicos
from ui import componentes as ui
from ui import graficos
from ui.tema import brl, nivel, num, pct, prazo

ctx = servicos.atual()
conf, dias = nivel(ctx.confianca), prazo(ctx.horizonte)
ret = ctx.retornos

ui.cabecalho(
    "Stress",
    "E se o mercado ficar muito mais turbulento do que o histórico recente?",
)

cenarios = servicos.cenarios(ret, ctx.confianca, ctx.horizonte)
base = float(cenarios["var"].iloc[0])
extremo = float(cenarios["var"].iloc[-1])
ui.indicadores(
    [
        {"label": f"VaR {conf} · base", **ui.perda(base, ctx.posicao)},
        {"label": "VaR · cenário extremo", **ui.perda(extremo, ctx.posicao)},
        {
            "label": "Amplificação",
            "value": num(extremo / base, 1) + "×",
            "help": "VaR do cenário extremo dividido pelo VaR base.",
        },
    ]
)

ui.grafico(
    graficos.colunas_ordenadas(list(cenarios["scenario"]), list(cenarios["var"])),
    f"VaR {conf} em {dias} sob cada cenário",
    "Cenários hipotéticos de sensibilidade; não reproduzem episódios históricos.",
    dados=pd.DataFrame(
        {
            "Cenário": cenarios["scenario"],
            "Volatilidade": [num(v, 1) + "×" for v in cenarios["vol_multiplier"]],
            "Choque na média diária": [
                pct(v, 1, sinal=True) for v in cenarios["mean_shift"]
            ],
            "VaR": [pct(v) for v in cenarios["var"]],
            "ES": [pct(v) for v in cenarios["es"]],
            "VaR em reais": [brl(v * ctx.posicao) for v in cenarios["var"]],
        }
    ),
    chave="stress_cenarios",
)

esquerda, direita = st.columns(2, gap="medium")
with esquerda, st.container(border=True):
    st.markdown("**Monte o seu cenário**")
    vol = st.slider(
        "Multiplicador de volatilidade", 1.0, 5.0, 2.0, 0.25, key="stress_vol"
    )
    media = st.slider(
        "Choque na média diária (pontos percentuais)",
        -2.0,
        1.0,
        -0.5,
        0.1,
        key="stress_media",
    )
    proprio = servicos.var_estressado(
        ret, ctx.confianca, vol, media / 100, ctx.horizonte
    )
    ui.indicadores(
        [
            {
                "label": f"VaR {conf} no cenário",
                **ui.perda(proprio.value, ctx.posicao),
            },
            {
                "label": "Frente ao base",
                "value": num(proprio.value / base, 1) + "×",
            },
        ]
    )
with direita, st.container(border=True):
    st.markdown("**As piores janelas que já aconteceram**")
    janelas = servicos.piores_janelas(ret, ctx.datas)
    st.dataframe(
        pd.DataFrame(
            {
                "Janela": [prazo(int(h)) for h in janelas["horizon_days"]],
                "Pior retorno": [pct(v, 1) for v in janelas["worst_return"]],
                "Em reais": [brl(v * ctx.posicao) for v in janelas["worst_return"]],
                "De": [f"{d:%d/%m/%Y}" for d in janelas["start"]],
                "Até": [f"{d:%d/%m/%Y}" for d in janelas["end"]],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Perdas acumuladas realmente observadas no período selecionado. Não "
        "dependem de nenhum modelo."
    )

ui.como_ler(
    """
- O VaR parte do histórico recente. O stress pergunta o que acontece se o
  futuro for **pior** que esse histórico.
- Cada cenário multiplica os desvios dos retornos em torno da média
  (volatilidade maior) e desloca a média para baixo, e então recalcula o VaR
  histórico sobre a série chocada.
- Os multiplicadores são escolhas de sensibilidade. Calibrá-los em crises
  reais exigiria um histórico que contenha essas crises.
"""
)
